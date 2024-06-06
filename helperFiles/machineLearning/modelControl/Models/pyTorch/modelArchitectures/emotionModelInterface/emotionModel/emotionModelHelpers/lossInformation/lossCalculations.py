# General
import random
import torch
import math

# Helper classes
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
from ..generalMethods.generalMethods import generalMethods
from ..emotionDataInterface import emotionDataInterface
from ..modelParameters import modelParameters

# Loss methods
from ......Helpers.lossFunctions import pytorchLossMethods
from ......Helpers.lossFunctions import weightLoss


class lossCalculations:

    def __init__(self, accelerator, model, allEmotionClasses, activityLabelInd, useFinalParams=False):
        # General parameters
        self.allEmotionClasses = allEmotionClasses  # The number of classes (intensity levels) within each emotion to predict. Dim: numEmotions
        self.emotionLength = model.compressedLength  # The number of indices in every final emotion distribution.
        self.activityLabelInd = activityLabelInd
        self.numEmotions = model.numEmotions  # The number of emotions to predict.
        self.useFinalParams = useFinalParams  # Whether to use the HPC parameters.
        self.accelerator = accelerator  # Hugging face model optimizations.
        self.model = model

        # Positional encoder information.
        self.maxNumEncodedSignals = model.signalEncoderModel.encodeSignals.positionalEncodingInterface.maxNumEncodedSignals  # The number of classes in the positional encoder.

        # Initialize helper classes.
        self.dataInterface = emotionDataInterface()
        self.modelParameters = modelParameters
        self.generalMethods = generalMethods()
        self.modelHelpers = modelHelpers()

        # Specify the model's loss functions (READ BEFORE USING!!). 
        #       Classification Options: "NLLLoss", "KLDivLoss", "CrossEntropyLoss", "BCEWithLogitsLoss"
        #       Custom Classification Options: "weightedKLDiv", "diceLoss", "FocalLoss"
        #       Regression Options: "MeanSquaredError", "MeanAbsoluteError", "Huber", "SmoothL1Loss", "PoissonNLLLoss", "GammaNLLLoss"
        #       Custom Regression Options: "R2", "pearson", "LogCoshLoss", "weightedMSE"
        self.emotionDist_lossType = "MeanSquaredError"  # The loss enforcing correct distribution shape.
        self.activityClass_lossType = "CrossEntropyLoss"  # The loss enforcing correct activity recognition.
        # Initialize the loss function WITHOUT the class weights.
        self.activityClassificationLoss = pytorchLossMethods(lossType=self.activityClass_lossType, class_weights=None).loss_fn
        self.emotionClassificationLoss = pytorchLossMethods(lossType=self.emotionDist_lossType, class_weights=None).loss_fn
        self.positionalEncoderLoss = pytorchLossMethods(lossType="MeanSquaredError", class_weights=None).loss_fn
        self.reconstructionLoss = pytorchLossMethods(lossType="MeanSquaredError", class_weights=None).loss_fn

    # ---------------------------------------------------------------------- #
    # -------------------------- Loss Calculations ------------------------- #

    @staticmethod
    def getData(data, mask):
        if data is None or mask is None: return data
        return data[mask]

    def getReconstructionDataMask(self, allLabelsMask, reconstructionIndex):
        # Find the boolean flags for the data involved in the loss calculation.
        if allLabelsMask is not None and reconstructionIndex is not None:
            return self.dataInterface.getEmotionColumn(allLabelsMask, reconstructionIndex)  # Dim: numExperiments

    def getOptimalLoss(self, method, allInputData, allLabelsMask, reconstructionIndex):
        # Isolate the signals for this loss (For example, training vs. testing).
        reconstructionDataMask = self.getReconstructionDataMask(allLabelsMask, reconstructionIndex)
        signalData = self.getData(allInputData, reconstructionDataMask)  # Dim: numExperiments, numSignals, signalLength

        # Calculate the final losses.
        finalLoss = method(signalData).mean()
        return finalLoss

    def calculatePositionalEncodingLoss(self, predictedPositionIndices):
        # Extract the positional encoding information.
        batchSize, numSignals = predictedPositionIndices.size()

        # Reshape the data for the positional encoding loss.
        targetPositionIndices = torch.arange(numSignals, device=self.accelerator.device, dtype=torch.float32).repeat(batchSize, 1)
        # targetPositionIndices dim: batchSize, numSignals

        # Calculate the positional encoding loss.
        positionalEncodingLoss = self.positionalEncoderLoss(predictedPositionIndices, targetPositionIndices).mean()
        if not self.useFinalParams and random.random() < 0.01: self.errorPerClass(predictedPositionIndices, targetPositionIndices)

        # Weight the final loss based on the number of signals.
        classWeights = self.generalMethods.minMaxScale_noInverse(targetPositionIndices, scale=0.5, buffer=0) + 1.5
        positionalEncodingLoss = (positionalEncodingLoss * classWeights).sum() / classWeights.sum()

        return positionalEncodingLoss

    def calculateSignalEncodingLoss(self, allSignalData, allEncodedData, allReconstructedData, allPredictedIndexProbabilities, allDecodedPredictedIndexProbabilities, allSignalEncodingLayerLoss, allLabelsMask=None, reconstructionIndex=None):
        # Find the boolean flags for the data involved in the loss calculation.
        reconstructionDataMask = self.getReconstructionDataMask(allLabelsMask, reconstructionIndex)
        # Isolate the signals for this loss (For example, training vs. testing).
        signalData = self.getData(allSignalData, reconstructionDataMask)  # Dim: numExperiments, numSignals, compressedLength
        encodedData = self.getData(allEncodedData, reconstructionDataMask)  # Dim: numExperiments, numCondensedSignals, compressedLength
        reconstructedData = self.getData(allReconstructedData, reconstructionDataMask)  # Dim: numExperiments, numSignals, compressedLength
        signalEncodingLayerLoss = self.getData(allSignalEncodingLayerLoss, reconstructionDataMask)  # Dim: numExperiments
        predictedIndexProbabilities = self.getData(allPredictedIndexProbabilities, reconstructionDataMask)  # Dim: numExperiments, numSignals, maxNumEncodedSignals
        decodedPredictedIndexProbabilities = self.getData(allDecodedPredictedIndexProbabilities, reconstructionDataMask)  # Dim: numExperiments, numSignals, maxNumEncodedSignals
        assert signalData.shape[0] != 0, "There are no signals for this loss calculation."

        # Calculate the error in signal reconstruction (encoding loss).
        signalReconstructedLoss = self.reconstructionLoss(reconstructedData, signalData)
        signalReconstructedLoss = signalReconstructedLoss.mean(dim=2).mean(dim=1).mean()

        # Enforce that the compressed data has a mean of 0 and a standard deviation of 1.
        encodedSignalMeanLoss, encodedSignalMinMaxLoss = self.calculateMinMaxLoss(encodedData, expectedMean=0, expectedMinMax=self.modelParameters.getSignalMinMaxScale(), dim=-1, minMaxBuffer=0.1)
        # Reduce the loss to a singular value.
        encodedSignalMinMaxLoss = encodedSignalMinMaxLoss.mean(dim=2).mean(dim=1).mean()
        encodedSignalMeanLoss = encodedSignalMeanLoss.mean(dim=1).mean()

        # If there is a layer loss, average the loss.
        if signalEncodingLayerLoss is not None: signalEncodingLayerLoss = signalEncodingLayerLoss.mean()

        # Positional encoding loss.
        decodedPositionalEncodingLoss = self.calculatePositionalEncodingLoss(decodedPredictedIndexProbabilities)
        positionalEncodingLoss = self.calculatePositionalEncodingLoss(predictedIndexProbabilities)

        # Assert that nothing is wrong with the loss calculations.
        self.modelHelpers.assertVariableIntegrity(encodedSignalMeanLoss, variableName="encoded signal mean loss", assertGradient=False)
        self.modelHelpers.assertVariableIntegrity(positionalEncodingLoss, variableName="positional encoding loss", assertGradient=False)
        self.modelHelpers.assertVariableIntegrity(signalReconstructedLoss, variableName="encoded signal reconstructed loss", assertGradient=False)
        self.modelHelpers.assertVariableIntegrity(encodedSignalMinMaxLoss, variableName="encoded signal standard deviation loss", assertGradient=False)
        self.modelHelpers.assertVariableIntegrity(decodedPositionalEncodingLoss, variableName="Decoded positional encoding loss", assertGradient=False)
        if signalEncodingLayerLoss is not None: self.modelHelpers.assertVariableIntegrity(signalEncodingLayerLoss, "encoded signal layer loss", assertGradient=False)

        return signalReconstructedLoss, encodedSignalMeanLoss, encodedSignalMinMaxLoss, positionalEncodingLoss, decodedPositionalEncodingLoss, signalEncodingLayerLoss

    def calculateAutoencoderLoss(self, allEncodedData, allCompressedData, allReconstructedEncodedData, allAutoencoderLayerLoss, allLabelsMask, reconstructionIndex):
        # Find the boolean flags for the data involved in the loss calculation.
        reconstructionDataMask = self.getReconstructionDataMask(allLabelsMask, reconstructionIndex)
        # Isolate the signals for this loss (For example, training vs. testing).
        reconstructedEncodedData = self.getData(allReconstructedEncodedData, reconstructionDataMask)  # Dim: numExperiments, numSignals, sequenceLength
        compressedData = self.getData(allCompressedData, reconstructionDataMask)  # Dim: numExperiments, numSignals, compressedLength
        encodedData = self.getData(allEncodedData, reconstructionDataMask)  # Dim: numExperiments, numSignals, sequenceLength
        autoencoderLayerLoss = self.getData(allAutoencoderLayerLoss, reconstructionDataMask)  # Dim: numExperiments
        assert encodedData.shape[0] != 0, f"There are no signals for this loss calculation. reconstructionDataMask: {reconstructionDataMask}"

        # Calculate the error in signal reconstruction (autoencoder loss).
        reconstructedLoss = self.reconstructionLoss(reconstructedEncodedData, encodedData)
        reconstructedLoss = reconstructedLoss.mean(dim=2).mean(dim=1).mean()

        # Enforce that the compressed data has a mean of 0 and a standard deviation of 1.
        compressedMeanLoss, compressedMinMaxLoss = self.calculateMinMaxLoss(compressedData, expectedMean=0, expectedMinMax=self.modelParameters.getSignalMinMaxScale(), dim=-1, minMaxBuffer=0.1)
        # Reduce the loss to a singular value.
        compressedMinMaxLoss = compressedMinMaxLoss.mean(dim=1).mean()
        compressedMeanLoss = compressedMeanLoss.mean(dim=1).mean()

        # If there is a layer loss, average the loss.
        if autoencoderLayerLoss is not None: autoencoderLayerLoss = autoencoderLayerLoss.mean()

        # Assert that nothing is wrong with the loss calculations. 
        self.modelHelpers.assertVariableIntegrity(compressedMeanLoss, variableName="autoencoder mean loss", assertGradient=False)
        self.modelHelpers.assertVariableIntegrity(reconstructedLoss, variableName="autoencoder reconstructed loss", assertGradient=False)
        self.modelHelpers.assertVariableIntegrity(compressedMinMaxLoss, variableName="autoencoder standard deviation loss", assertGradient=False)
        if autoencoderLayerLoss is not None: self.modelHelpers.assertVariableIntegrity(autoencoderLayerLoss, "autoencoder layer loss", assertGradient=False)

        return reconstructedLoss, compressedMeanLoss, compressedMinMaxLoss, autoencoderLayerLoss

    def calculateSignalMappingLoss(self, allEncodedData, allManifoldData, allTransformedManifoldData, allReconstructedEncodedData, allLabelsMask, reconstructionIndex):
        # Find the boolean flags for the data involved in the loss calculation.
        reconstructionDataMask = self.getReconstructionDataMask(allLabelsMask, reconstructionIndex)
        # Isolate the signals for this loss (For example, training vs testing).
        encodedData = allEncodedData[reconstructionDataMask]  # Dim: numExperiments, numCondensedSignals, compressedLength
        manifoldData = allManifoldData[reconstructionDataMask]  # Dim: numExperiments, numCondensedSignals, manifoldLength
        transformedManifoldData = allTransformedManifoldData[reconstructionDataMask]  # Dim: numExperiments, finalNumSignals, compressedLength
        reconstructedEncodedData = allReconstructedEncodedData[reconstructionDataMask]  # Dim: numExperiments, numCondensedSignals, compressedLength
        assert manifoldData.shape[0] != 0, "There are no signals for this loss calculation."

        # Calculate the error in signal reconstruction (autoencoder loss) and assert the integrity of the loss.
        manifoldReconstructedLoss = self.reconstructionLoss(reconstructedEncodedData, encodedData)
        manifoldReconstructedLoss = manifoldReconstructedLoss.mean(axis=2).mean()

        # Enforce that the compressed data has a mean of 0 and a standard deviation of 1.
        manifoldMeanLoss, manifoldMinMaxLoss = self.calculateMinMaxLoss(manifoldData, expectedMean=0, expectedMinMax=self.modelParameters.getSignalMinMaxScale(), dim=-1, minMaxBuffer=0.1)
        # Reduce the loss to a singular value.
        manifoldMeanLoss = manifoldMeanLoss.mean()
        manifoldMinMaxLoss = manifoldMinMaxLoss.mean()

        # Enforce the same properties across all the time series data.

        # Assert that nothing is wrong with the loss calculations. 
        self.modelHelpers.assertVariableIntegrity(manifoldMeanLoss, "manifold mean loss", assertGradient=False)
        self.modelHelpers.assertVariableIntegrity(manifoldReconstructedLoss, "manifold reconstructed loss", assertGradient=False)
        self.modelHelpers.assertVariableIntegrity(manifoldMinMaxLoss, "manifold standard deviation loss", assertGradient=False)

        return manifoldReconstructedLoss, manifoldMeanLoss, manifoldMinMaxLoss

    def calculateActivityLoss(self, predictedActivityLabels, allLabels, allLabelsMask, activityClassWeights):
        # Find the boolean flags for the data involved in the loss calculation.
        activityDataMask = self.dataInterface.getActivityColumn(allLabelsMask, self.activityLabelInd)  # Dim: numExperiments
        trueActivityLabels = self.dataInterface.getActivityLabels(allLabels, allLabelsMask, self.activityLabelInd)

        # Calculate the activity classification accuracy/loss and assert the integrity of the loss.
        activityLosses = self.activityClassificationLoss(predictedActivityLabels[activityDataMask], trueActivityLabels.long())
        activityLoss = weightLoss(activityLosses, activityClassWeights, trueActivityLabels)
        assert not activityLoss.isnan().any().item() and not activityLoss.isinf().any().item(), f"Check your inputs to (or the method) self.activityClassificationLoss. Found {activityLoss} value"

        return activityLoss

    def calculateEmotionsLoss(self, emotionInd, predictedEmotionlabels, allLabels, allLabelsMask, allEmotionClassWeights):
        # Calculate the loss from predicting similar basic emotions.
        emotionOrthogonalityLoss = self.lossCalculations.scoreEmotionOrthonormality(allBasicEmotionDistributions)
        assert not emotionOrthogonalityLoss.isnan().any().item() and not emotionOrthogonalityLoss.isinf().any().item()
        # Calculate the loss from model-specific weights.
        modelSpecificWeights = self.lossCalculations.scoreModelWeights(self.model.predictUserEmotions.allSubjectWeights)
        assert not modelSpecificWeights.isnan().any().item() and not modelSpecificWeights.isinf().any().item()
        # Add all the losses together into one value.
        finalLoss = reconstructedLoss + activityLoss * 2 + emotionOrthogonalityLoss / 2  # + modelSpecificWeights*0.01

        # Get the valid emotion indices (ones with training points).
        batchEmotionTrainingMask = self.dataInterface.getEmotionMasks(batchTrainingMask)
        validEmotionInds = self.dataInterface.getLabelInds_withPoints(batchEmotionTrainingMask)

        emotionLoss = 0
        # For each emotion we are predicting that has training data.
        for validEmotionInd in validEmotionInds:
            # Calculate and add the loss due to misclassifying the emotion.
            emotionLoss = self.lossCalculations.calculateEmotionLoss(validEmotionInd, predictedBatchEmotions, trueBatchLabels,
                                                                     batchTrainingMask, allEmotionClassWeights)  # Calculate the error in the emotion predictions
            emotionLoss += emotionLoss / len(validEmotionInds)  # Add all the losses together into one value.
        # Average all the losses that were added together.
        finalLoss += emotionLoss * 2

    def calculateEmotionLoss(self, emotionInd, predictedEmotionlabels, allLabels, allLabelsMask, allEmotionClassWeights):
        # Organize the emotion's training information.
        emotionLabels = self.dataInterface.getEmotionLabels(emotionInd, allLabels, allLabelsMask)
        emotionClassWeights = allEmotionClassWeights[emotionInd]

        # Get the predicted and true emotion distributions.
        predictedTrainingEmotions, trueTrainingEmotions = self.dataInterface.getEmotionDistributions(emotionInd, predictedEmotionlabels, allLabels, allLabelsMask)
        # predictedTrainingEmotions = F.normalize(predictedTrainingEmotions, dim=1, p=1)
        # assert (predictedTrainingEmotions >= 0).all()

        # Calculate an array of possible emotion ratings.
        numEmotionClasses = self.allEmotionClasses[emotionInd]
        possibleEmotionRatings = torch.arange(0, numEmotionClasses, numEmotionClasses / self.emotionLength, device=allLabels.device) - 0.5
        # Calculate the weighted prediction losses
        mseLossDistributions = (emotionLabels[:, None] - possibleEmotionRatings) ** 2
        emotionDistributionLosses = (mseLossDistributions * predictedTrainingEmotions).sum(dim=1)

        # Calculate the error in the emotion predictions
        # emotionDistributionLosses = self.emotionClassificationLoss(predictedTrainingEmotions, trueTrainingEmotions.float()).sum(dim=-1)
        emotionDistributionLoss = weightLoss(emotionDistributionLosses, emotionClassWeights, emotionLabels).mean()
        assert not emotionDistributionLoss.isnan().any().item() and not emotionDistributionLoss.isinf().any().item(), print(predictedTrainingEmotions, trueTrainingEmotions.float(), emotionDistributionLoss)

        return emotionDistributionLoss

    # ---------------------------------------------------------------------- #
    # ------------------------- Loss Helper Methods ------------------------ # 

    @staticmethod
    def scoreEmotionOrthonormality(allBasicEmotionDistributions):
        assert not allBasicEmotionDistributions.isnan().any().item() and not allBasicEmotionDistributions.isinf().any().item()
        batchSize, numInterpreterHeads, numBasicEmotion, emotionLength = allBasicEmotionDistributions.shape
        allBasicEmotionDistributionsAbs = allBasicEmotionDistributions.abs()

        # Calculate the overlap in probability between each basic emotion.
        allBasicEmotionDistributionsAbs_T = allBasicEmotionDistributionsAbs.permute(0, 1, 3, 2)  # batchSize, self.numInterpreterHeads, emotionLength, numBasicEmotions
        probabilityOverlap_basicEmotions = allBasicEmotionDistributionsAbs.sqrt() @ allBasicEmotionDistributionsAbs_T.sqrt()
        # Zero out self-overlap as each signal SHOULD be overlapping with itself.
        probabilityOverlap_basicEmotions -= torch.eye(numBasicEmotion, numBasicEmotion, device=allBasicEmotionDistributions.device).view(1, 1, numBasicEmotion, numBasicEmotion)
        # For each interpretation of emotions, the basis states should be orthonormal.
        basicEmotion_orthoganalityLoss = probabilityOverlap_basicEmotions.mean()

        # Calculate the overlap in probability for each basic emotion across each interpretation.
        allInterpretationEmotions = allBasicEmotionDistributionsAbs.permute(0, 2, 1, 3)  # batchSize, numBasicEmotions, numInterpreterHeads, emotionLength
        allInterpretationEmotions_T = allBasicEmotionDistributionsAbs.permute(0, 2, 3, 1)  # batchSize, numBasicEmotions, emotionLength, numInterpreterHeads
        probabilityOverlap_interpretations = allInterpretationEmotions.sqrt() @ allInterpretationEmotions_T.sqrt()
        # Zero out self-overlap as each signal SHOULD be overlapping with itself.
        probabilityOverlap_interpretations -= torch.eye(numInterpreterHeads, numInterpreterHeads, device=allBasicEmotionDistributions.device).view(1, 1, numInterpreterHeads, numInterpreterHeads)
        # Between all interpretations, each basis state should be different.
        emotionInterpretation_orthoganalityLoss = probabilityOverlap_basicEmotions.mean()

        return basicEmotion_orthoganalityLoss + emotionInterpretation_orthoganalityLoss

    @staticmethod
    def scoreModelWeights(allSubjectWeights):
        """
        allSubjectWeights : numSubjects, self.numInterpreterHeads, numBasicEmotions, 1
        """
        allSubjectWeights = allSubjectWeights.squeeze(3)
        # Calculate the different in how each subject interprets their emotions.
        allSubjectWeights_subjectDeviation = allSubjectWeights[None, :, :, :] - allSubjectWeights[:, None, :, :]
        # For every basic emotion, every subject should have the same interpretation (weight for each interpretation).
        subjectDeviationNorm = torch.norm(allSubjectWeights_subjectDeviation, dim=3)[0, 1:]
        subjectDeviationNormLoss = subjectDeviationNorm.mean()

        # # For every predicted emotion, the model should recombine the emotions similarly (correlation among emotions).
        # allBasicEmotionWeights = self.model.predictComplexEmotions.allBasicEmotionWeights.squeeze(3).squeeze(1)
        # allBasicEmotionWeights_Norms = torch.cdist(allBasicEmotionWeights, allBasicEmotionWeights, p=2.0)
        # allBasicEmotionWeights_Loss = allBasicEmotionWeights_Norms.mean(dim=1)
        # weightRegularizationLoss dimension: self.numEmotions, 1

        return subjectDeviationNormLoss

    # ---------------------------------------------------------------------- #
    # ----------------------- Standardization Losses ----------------------- #

    @staticmethod
    def errorPerClass(output, target):
        # Calculate the error per class
        num_classes = output.size(1)
        class_errors = torch.zeros(num_classes)
        output = output.detach().argmax(dim=-1)

        for i in range(num_classes):
            # Mask for the current class
            class_mask = (target == i)
            if class_mask.sum() > 0:
                error = output[class_mask] - target[class_mask]
                class_errors[i] = error.float().abs().mean()

        print("Loss per class:", class_errors)
        print("Final classes:", output[0:num_classes])

    @staticmethod
    def gradient_penalty(inputs, outputs, dims):
        # Calculate the gradient wrt the inputs.
        gradients = torch.autograd.grad(
            grad_outputs=torch.ones_like(outputs, device=outputs.device),
            allow_unused=False,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
            outputs=outputs,
            inputs=inputs
        )

        # Get the size of the gradients for normalization.
        batchSize, elemA, elemB = gradients[0].size()

        # Calculate the norm of the gradients.
        gradients_norm = torch.norm(gradients[0], p='fro', dim=dims)
        gradients_norm = gradients_norm / math.sqrt(elemA*elemB)

        return gradients_norm

    def calculateMinMaxLoss(self, inputData, expectedMean=0, expectedMinMax=1, dim=-1, minMaxBuffer=0.0):
        # Calculate the min-max loss.
        minMaxData = self.generalMethods.minMaxScale_noInverse(inputData, scale=expectedMinMax, buffer=minMaxBuffer)
        minMaxLoss = (minMaxData - expectedMean).pow(2)

        # Calculate the mean error.
        meanData = inputData.mean(dim=dim)
        meanError = (meanData - expectedMean).pow(2)

        return meanError, minMaxLoss

    @staticmethod
    def calculateStandardizationLoss(inputData, expectedMean=0, expectedStandardDeviation=1, dim=-1):
        # Calculate the data statistics on the last dimension.
        standardDeviationData = inputData.std(dim=dim)
        meanData = inputData.mean(dim=dim)

        # Calculate the squared deviation from mean = 0; std = 1.
        standardDeviationError = (standardDeviationData - expectedStandardDeviation).pow(2)
        meanError = (meanData - expectedMean).pow(2)

        return meanError, standardDeviationError

    def calculateDataDistributionLoss(self, originalData, predictedData, dim=-1):
        # Calculate the data statistics on the last dimension.
        standardDeviationData = originalData.std(dim=dim)
        meanData = originalData.mean(dim=dim)

        meanError, standardDeviationError = self.calculateStandardizationLoss(inputData=predictedData, expectedMean=meanData, expectedStandardDeviation=standardDeviationData, dim=dim)
        return meanError, standardDeviationError

    @staticmethod
    def standardize(data, dataMean=None, dataSTD=None):
        if dataMean is None and dataSTD is None:
            dataMean = data.mean(dim=-1, keepdim=True)
            dataSTD = data.std(dim=-1, keepdim=True)

        return (data - dataMean) / (1e-10 + dataSTD), dataMean, dataSTD
