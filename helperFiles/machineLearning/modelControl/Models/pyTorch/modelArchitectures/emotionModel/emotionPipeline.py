# General
import torch
import torch.optim as optim
import transformers
import random
import time

# Import files for machine learning
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionModelHelpers.generalMethods.modelHelpers import modelHelpers
from .emotionModelHelpers.lossInformation.organizeTrainingLosses import organizeTrainingLosses
from .emotionModelHelpers.modelVisualizations.modelVisualizations import modelVisualizations

# Import files for the emotion model
from .emotionModelHelpers.emotionDataInterface import emotionDataInterface
from ...Helpers.modelMigration import modelMigration
from .emotionModelHead import emotionModelHead


class emotionPipeline:

    def __init__(self, accelerator, modelID, datasetName, modelName, allEmotionClasses, sequenceLength, maxNumSignals,
                 numSubjectIdentifiers, demographicLength, numSubjects, userInputParams, emotionNames,
                 activityNames, featureNames, submodel, fullTest=True, debuggingResults=False):
        # General parameters.
        self.numSubjectIdentifiers = numSubjectIdentifiers  # The number of subject identifiers to consider. Dim: [numSubjectIdentifiers]
        self.demographicLength = demographicLength  # The amount of demographic information provided to the model (age, weight, etc.). Dim: [numDemographics]
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.sequenceLength = sequenceLength  # The length of each incoming signal. Type: int
        self.device = accelerator.device  # Specify whether to use the CPU or GPU capabilities.
        self.accelerator = accelerator  # Hugging face interface to speed up the training process.
        self.modelName = modelName  # The unique name of the model to initialize.
        self.fullTest = fullTest  # Whether to run a full test or not.
        self.modelID = modelID  # A unique integer identifier for this model.

        # Pre-initialize later parameters.
        self.optimizer = None
        self.scheduler = None
        self.model = None

        # Dataset-specific parameters.
        self.allEmotionClasses = allEmotionClasses  # The number of classes (intensity levels) within each emotion to predict. Dim: [numEmotions]
        self.activityLabelInd = len(emotionNames)  # The index of the activity label in the label array.
        self.numActivities = len(activityNames)  # The number of activities we are predicting. Type: int
        self.numEmotions = len(emotionNames)  # The number of emotions we are predicting. Type: int
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.maxNumSignals = maxNumSignals  # The maximum number of signals to consider. Type: int
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.featureNames = featureNames  # The names of each signal in the model. Dim: numSignals
        self.datasetName = datasetName  # The name of the specific dataset being used in this model (case, wesad, etc.)

        # Initialize the emotion model.
        if modelName == "emotionModel":
            self.model = emotionModelHead(submodel, self.accelerator, sequenceLength, maxNumSignals, numSubjectIdentifiers, demographicLength, userInputParams,
                                          emotionNames, activityNames, featureNames, numSubjects, datasetName, debuggingResults)
        # Assert that the model has been initialized.
        assert hasattr(self, 'model'), f"Unknown Model Type Requested: {modelName}"

        # Extract relevant properties from the model.
        self.generalTimeWindow = self.model.timeWindows[-1]  # The default time window to use for training and testing.

        # Initialize helper classes.
        self.organizeLossInfo = organizeTrainingLosses(self.accelerator, self.model, allEmotionClasses, self.activityLabelInd, self.generalTimeWindow)
        self.modelVisualization = modelVisualizations(accelerator, self.generalTimeWindow, modelSubfolder="trainingFigures/")
        self.modelMigration = modelMigration(accelerator)
        self.dataInterface = emotionDataInterface()
        self.generalMethods = generalMethods()
        self.modelHelpers = modelHelpers()

        if submodel == "emotionPrediction":
            # Finalize model setup.
            self.model.sharedEmotionModel.lastActivityLayer = self.modelHelpers.getLastActivationLayer(self.organizeLossInfo.activityClass_lossType,
                                                                                                       predictingProb=True)  # Apply activation on the last layer: 'softmax', 'logsoftmax', or None.
            self.model.sharedEmotionModel.lastEmotionLayer = self.modelHelpers.getLastActivationLayer(self.organizeLossInfo.emotionDist_lossType, predictingProb=True)  # Apply activation on the last layer: 'softmax', 'logsoftmax', or None.
        else:
            self.generalTimeWindowInd = self.model.timeWindows.index(self.generalTimeWindow)

        # Finish setting up the mode.
        self.addOptimizer(submodel)  # Initialize the optimizer (for back propagation)

        # Assert data integrity of the inputs.
        assert len(self.emotionNames) == len(self.allEmotionClasses), f"Found {len(self.emotionNames)} emotions with {len(self.allEmotionClasses)} classes specified."

    def acceleratorInterface(self, dataLoader=None):
        if dataLoader is None:
            self.optimizer, self.scheduler, self.model = self.accelerator.prepare(self.optimizer, self.scheduler, self.model)
        else:
            # Hugging face integration.
            self.optimizer, self.scheduler, self.model, dataLoader = self.accelerator.prepare(self.optimizer, self.scheduler, self.model, dataLoader)

        return dataLoader

    def addOptimizer(self, submodel):
        # Get the models, while considering whether they are distributed or not.
        trainingInformation, signalEncoderModel, autoencoderModel, signalMappingModel, sharedEmotionModel, specificEmotionModel = self.getDistributedModels(model=None, submodel=None)

        modelParams = [
            # Specify the model parameters for the signal encoding.
            {'params': signalEncoderModel.parameters(), 'weight_decay': 1E-10, 'lr': 2E-4 if self.fullTest else 2E-4}]
        if submodel in ["autoencoder", "emotionPrediction"]:
            modelParams.append(
                # Specify the model parameters for the autoencoder.
                {'params': autoencoderModel.parameters(), 'weight_decay': 1E-10, 'lr': 2E-4 if self.fullTest else 2E-4})
        if submodel == "emotionPrediction":
            modelParams.extend([
                # Specify the model parameters for the signal mapping.
                {'params': signalMappingModel.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4 if self.fullTest else 1E-4},

                # Specify the model parameters for the feature extraction.
                {'params': sharedEmotionModel.extractCommonFeatures.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4 if self.fullTest else 1E-4},

                # Specify the model parameters for the human activity recognition.
                {'params': sharedEmotionModel.extractActivityFeatures.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4 if self.fullTest else 1E-4},
                {'params': specificEmotionModel.classifyHumanActivity.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4 if self.fullTest else 1E-4},

                # Specify the model parameters for the emotion prediction.
                {'params': sharedEmotionModel.predictBasicEmotions.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4 if self.fullTest else 1E-4},
                {'params': specificEmotionModel.predictUserEmotions.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4 if self.fullTest else 1E-4},
                {'params': specificEmotionModel.predictComplexEmotions.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4 if self.fullTest else 1E-4}])

        # Set the optimizer.
        self.optimizer = self.setOptimizer(modelParams, lr=1E-4, weight_decay=0)

        # Set the learning rate scheduler.
        self.scheduler = self.getLearningRateScheduler(submodel)

    def prepareInformation(self, dataLoader):
        # Load in all the data and labels for final predictions.
        allData, allLabels, allTrainingMasks, allTestingMasks = dataLoader.dataset.getAll()
        allSignalData, allDemographicData, allSubjectIdentifiers = self.dataInterface.separateData(allData, self.sequenceLength, self.numSubjectIdentifiers, self.demographicLength)
        reconstructionIndex = self.dataInterface.getReconstructionIndex(allTrainingMasks)
        assert reconstructionIndex is not None

        # Assert the integrity of the dataloader.
        assert allLabels.shape[1] == self.numEmotions + 1, f"Found {allLabels.shape[1]} labels, but expected {self.numEmotions} emotions + 1 activity label."
        assert allLabels.shape == allTrainingMasks.shape, "We should specify the training indices for each label"
        assert allLabels.shape == allTestingMasks.shape, "We should specify the testing indices for each label"

        return allData, allLabels, allTrainingMasks, allTestingMasks, allSignalData, allDemographicData, allSubjectIdentifiers, reconstructionIndex

    def trainModel(self, dataLoader, submodel, numEpochs=500, trainingFlag=True):
        """
        Stored items in the dataLoader.dataset:
            allData: The standardized testing and training data. numExperiments, numSignals, signalInfoLength
            allLabels: Integer labels representing class indices. Dim: numExperiments, numLabels (where numLabels = numEmotions + 1)
            allTestingMasks: Boolean flags representing if the label is a testing label. Dim: numExperiments, numLabels (where numLabels = numEmotions + 1)
            allTrainingMasks: Boolean flags representing if the label is a training label. Dim: numExperiments, numLabels (where numLabels = numEmotions + 1)
                Note: signalInfoLength = sequenceLength + 1 + demographicLength (The extra +1 is for the subject index)
                Note: the last dimension in allLabels is for human activity recognition.
        """
        # Hugging face integration.
        self.accelerator.print(f"\nTraining {self.datasetName} model", flush=True)
        model = self.getDistributedModel()

        # Load in all the data and labels for final predictions and calculate the activity and emotion class weights.
        allData, allLabels, allTrainingMasks, allTestingMasks, allSignalData, allDemographicData, allSubjectIdentifiers, reconstructionIndex = self.prepareInformation(dataLoader)
        allEmotionClassWeights, activityClassWeights = self.organizeLossInfo.getClassWeights(allLabels, allTrainingMasks, allTestingMasks, self.numActivities)

        # Assert valid input parameters.
        assert allLabels.shape[1] == self.numEmotions + 1, f"Found {allLabels.shape[1]} labels, but expected {self.numEmotions} emotions + 1 activity label."
        assert allLabels.shape == allTrainingMasks.shape, "We should specify the training indices for each label"
        assert allLabels.shape == allTestingMasks.shape, "We should specify the testing indices for each label"
        if trainingFlag: assert numEpochs == 1, f"numEpochs: {numEpochs}"

        # For each training epoch.
        for epoch in range(numEpochs):
            if 5 < numEpochs: self.accelerator.print(f"\tRound: {epoch}", flush=True)
            numPointsAnalyzed = 0

            # For each minibatch.
            for dataInd, data in enumerate(dataLoader):
                # Accumulate gradients.
                with self.accelerator.accumulate(model):
                    # Extract the data, labels, and testing/training indices.
                    batchData, trueBatchLabels, batchTrainingMask, batchTestingMask = data
                    # Add the data, labels, and training/testing indices to the device (GPU/CPU)
                    batchTrainingMask, batchTestingMask = batchTrainingMask.to(self.device), batchTestingMask.to(self.device)
                    batchData, trueBatchLabels = batchData.to(self.device), trueBatchLabels.to(self.device)

                    # Set the model intro the training mode.
                    numPointsAnalyzed += batchData.size(0)
                    self.setupTraining(submodel)

                    # Only analyze data that can produce meaningful training results.
                    if submodel in ["signalEncoder", "autoencoder"]:
                        # Get the current training data mask.
                        trainingColumn = self.dataInterface.getEmotionColumn(batchTrainingMask, reconstructionIndex)

                        # Apply the training data mask
                        batchTrainingMask = batchTrainingMask[trainingColumn]
                        trueBatchLabels = trueBatchLabels[trainingColumn]
                        batchData = batchData[trainingColumn]
                        if batchData.size(0) == 0: continue  # We are not training on any points

                    # Separate the data into signal, demographic, and subject identifier information.
                    signalData, demographicData, subjectIdentifiers = self.dataInterface.separateData(batchData, self.sequenceLength, self.numSubjectIdentifiers, self.demographicLength)
                    # demographicData dimension: batchSize, numSignals, demographicLength
                    # signalData dimension: batchSize, numSignals, sequenceLength
                    # subjectInds dimension: batchSize, numSubjectIdentifiers

                    # Randomly choose to add noise to the model.
                    augmentedSignalData = signalData.clone()
                    addingNoiseFlag = random.random() < 0
                    addingNoiseRange = [0, 1]
                    addingNoiseSTD = 0

                    if addingNoiseFlag:
                        # Augment the data to add some noise to the model.
                        addingNoiseSTD, addingNoiseRange = self.getAugmentationDeviation(submodel)
                        augmentedSignalData = self.dataInterface.addNoise(augmentedSignalData, trainingFlag=True, noiseSTD=addingNoiseSTD)
                        # augmentedSignalData dimension: batchSize, numSignals, sequenceLength

                    # ------------ Forward pass through the model  ------------- #

                    # Train the signal encoder
                    if submodel == "signalEncoder":
                        # Randomly choose to use an inflated number of signals.
                        maxNumSignals = 96 if self.datasetName in ["case"] else max(model.maxNumSignals, 256)
                        maxNumSignals = random.choices(population=[model.maxNumSignals, maxNumSignals], weights=[0.8, 0.2], k=1)[0]

                        # Augment the signals to train an arbitrary sequence length and order.
                        initialSignalData, augmentedSignalData = self.dataInterface.changeNumSignals(signalDatas=(signalData, augmentedSignalData), minNumSignals=model.numEncodedSignals, maxNumSignals=maxNumSignals, alteredDim=1)
                        initialSignalData, augmentedSignalData = self.dataInterface.changeSignalLength(model.timeWindows[0], signalDatas=(initialSignalData, augmentedSignalData))
                        print("Input size:", augmentedSignalData.size())

                        # with self.accelerator.autocast():
                        # Perform the forward pass through the model.
                        encodedData, reconstructedData, signalEncodingLayerLoss = model.signalEncoding(augmentedSignalData, initialSignalData, decodeSignals=True, calculateLoss=True, trainingFlag=True)
                        # encodedData dimension: batchSize, numEncodedSignals, sequenceLength
                        # reconstructedData dimension: batchSize, numSignals, sequenceLength
                        # signalEncodingLayerLoss dimension: batchSize

                        # Assert that nothing is wrong with the predictions.
                        self.modelHelpers.assertVariableIntegrity(signalEncodingLayerLoss, "signal encoder layer loss", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(reconstructedData, "reconstructed signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(augmentedSignalData, "augmented signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(signalData, "initial signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(encodedData, "encoded data", assertGradient=False)

                        # Calculate the error in signal compression (signal encoding loss).
                        signalReconstructedLoss, encodedSignalMeanLoss, encodedSignalStandardDeviationLoss, signalEncodingTrainingLayerLoss \
                            = self.organizeLossInfo.calculateSignalEncodingLoss(initialSignalData, encodedData, reconstructedData, signalEncodingLayerLoss, batchTrainingMask, reconstructionIndex)
                        if signalReconstructedLoss.item() == 0: self.accelerator.print("Not useful\n\n\n\n\n\n"); continue

                        # Initialize basic core loss value.
                        futureCompressionsFactor = augmentedSignalData.size(1) / self.model.numEncodedSignals  # Increase the learning rate if compressing more signals.
                        sequenceLengthFactor = augmentedSignalData.size(2) / self.sequenceLength  # Increase the learning rate for longer sequences.
                        compressionFactor = augmentedSignalData.size(1) / encodedData.size(1)  # Increase the learning rate for larger compressions.
                        noiseFactor = 1 - (addingNoiseSTD / addingNoiseRange[1]) + 0.5  # Lower the learning rate for high noise levels.
                        finalLoss = signalReconstructedLoss

                        # Compile the loss into one value
                        if 0.25 < encodedSignalStandardDeviationLoss:
                            finalLoss = finalLoss + 0.1 * encodedSignalStandardDeviationLoss
                        if 0.001 < signalEncodingTrainingLayerLoss:
                            finalLoss = finalLoss + 0.5 * signalEncodingTrainingLayerLoss
                        if 0.25 < encodedSignalMeanLoss:
                            finalLoss = finalLoss + 0.1 * encodedSignalMeanLoss
                        # Account for the current training state when calculating the loss.
                        finalLoss = compressionFactor * noiseFactor * sequenceLengthFactor * futureCompressionsFactor * finalLoss

                        # Update the user.
                        self.accelerator.print(finalLoss.item(), signalReconstructedLoss.item(), encodedSignalMeanLoss.item(), encodedSignalStandardDeviationLoss.item(), signalEncodingTrainingLayerLoss.item(), "\n")

                    # Train the autoencoder
                    elif submodel == "autoencoder":
                        # Augment the time series length to train an arbitrary sequence length.
                        initialSignalData, augmentedSignalData = self.dataInterface.changeSignalLength(model.timeWindows[0], (signalData, augmentedSignalData))
                        print("Input size:", augmentedSignalData.size())

                        # Perform the forward pass through the model.
                        encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss = \
                            model.compressData(augmentedSignalData, initialSignalData, reconstructSignals=True, calculateLoss=True, compileVariables=False, compileLosses=False, fullReconstruction=True, trainingFlag=True)
                        # denoisedDoubleReconstructedData dimension: batchSize, numSignals, sequenceLength
                        # reconstructedEncodedData dimension: batchSize, numEncodedSignals, sequenceLength
                        # compressedData dimension: batchSize, numEncodedSignals, compressedLength
                        # autoencoderLayerLoss dimension: batchSize

                        # Assert that nothing is wrong with the predictions.
                        self.modelHelpers.assertVariableIntegrity(denoisedDoubleReconstructedData, "denoised double reconstructed data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(reconstructedEncodedData, "reconstructed encoded data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(autoencoderLayerLoss, "autoencoder layer loss", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(compressedData, "compressed data", assertGradient=False)

                        # Calculate the error in signal reconstruction (autoencoder loss).
                        encodedReconstructedLoss, compressedMeanLoss, compressedStandardDeviationLoss, autoencoderTrainingLayerLoss = \
                            self.organizeLossInfo.calculateAutoencoderLoss(encodedData, compressedData, reconstructedEncodedData, autoencoderLayerLoss, batchTrainingMask, reconstructionIndex)
                        # Calculate the error in signal reconstruction (encoding loss).
                        signalReconstructedLoss = self.organizeLossInfo.signalEncodingLoss(initialSignalData, denoisedDoubleReconstructedData).mean(dim=2).mean(dim=1).mean()

                        # Initialize basic core loss value.
                        compressionFactorSE = augmentedSignalData.size(1) / self.model.numEncodedSignals
                        compressionFactor = augmentedSignalData.size(2) / self.model.compressedLength
                        finalLoss = encodedReconstructedLoss

                        # Compile the loss into one value
                        if 0.25 < compressedStandardDeviationLoss:
                            finalLoss = finalLoss + 0.1 * compressedStandardDeviationLoss
                        if 0.001 < autoencoderTrainingLayerLoss:
                            finalLoss = finalLoss + autoencoderTrainingLayerLoss
                        if 0.25 < compressedMeanLoss:
                            finalLoss = finalLoss + 0.1 * compressedMeanLoss
                        finalLoss = compressionFactor * (finalLoss + compressionFactorSE * signalReconstructedLoss)

                        # Update the user.
                        self.accelerator.print(finalLoss.item(), encodedReconstructedLoss.item(), compressedMeanLoss.item(), compressedStandardDeviationLoss.item(), autoencoderTrainingLayerLoss.item(), signalReconstructedLoss.item(), "\n")

                    elif submodel == "emotionPrediction":
                        # Perform the forward pass through the model.
                        _, _, _, compressedData, _, _, _, mappedSignalData, reconstructedCompressedData, featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions \
                            = model.emotionPrediction(augmentedSignalData, signalData, subjectIdentifiers, remapSignals=True, compileVariables=False, trainingFlag=True)
                        # eachBasicEmotionDistribution dimension: batchSize, self.numInterpreterHeads, self.numBasicEmotions, self.emotionLength
                        # finalEmotionDistributions dimension: self.numEmotions, batchSize, self.emotionLength
                        # activityDistribution dimension: batchSize, self.numActivities
                        # featureData dimension: batchSize, self.numCommonFeatures

                        # Assert that nothing is wrong with the predictions.
                        self.modelHelpers.assertVariableIntegrity(featureData, "feature data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(mappedSignalData, "mapped signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(activityDistribution, "activity distribution", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(finalEmotionDistributions, "final emotion distributions", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(eachBasicEmotionDistribution, "basic emotion distributions", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(reconstructedCompressedData, "reconstructed compressed data", assertGradient=False)

                        # Calculate the error in emotion and activity prediction models.
                        manifoldReconstructedLoss, manifoldMeanLoss, manifoldStandardDeviationLoss = self.organizeLossInfo.calculateSignalMappingLoss(
                            encodedData, manifoldData, transformedManifoldData, reconstructedEncodedData, batchTrainingMask, reconstructionIndex)
                        emotionLoss, emotionOrthogonalityLoss, modelSpecificWeights = self.organizeLossInfo.calculateEmotionsLoss(activityDistribution, trueBatchLabels, batchTrainingMask, activityClassWeights)
                        activityLoss = self.organizeLossInfo.calculateActivityLoss(activityDistribution, trueBatchLabels, batchTrainingMask, activityClassWeights)

                        # Compile the loss into one value
                        manifoldLoss = 0.8 * manifoldReconstructedLoss + 0.1 * manifoldMeanLoss + 0.1 * manifoldStandardDeviationLoss
                        finalLoss = emotionLoss * 0.45 + emotionOrthogonalityLoss * 0.05 + modelSpecificWeights * 0.05 + activityLoss * 0.4 + manifoldLoss * 0.05
                    else:
                        raise Exception()

                    # ------------------- Update the Model  -------------------- #

                    # Prevent too high losses from randomizing weights.
                    while 100 < finalLoss: finalLoss = finalLoss / 10

                    t1 = time.time()
                    # Calculate the gradients.
                    self.accelerator.backward(finalLoss)  # Calculate the gradients.
                    t2 = time.time();
                    self.accelerator.print(f"Backprop {self.datasetName} {numPointsAnalyzed}:", t2 - t1)
                    # Backpropagation the gradient.
                    self.optimizer.step()  # Adjust the weights.
                    self.optimizer.zero_grad()  # Zero your gradients to restart the gradient tracking.
                    self.accelerator.print("LR:", self.scheduler.get_last_lr())
            # Finalize all the parameters.
            self.scheduler.step()  # Update the learning rate.

            # ----------------- Evaluate Model Performance  ---------------- # 

            # Prepare the model/data for evaluation.
            self.setupTrainingFlags(self.model, trainingFlag=False)  # Set all models into evaluation mode.

        # Prepare the model/data for evaluation.
        self.setupTrainingFlags(self.model, trainingFlag=False)  # Set all models into evaluation mode.
        self.accelerator.wait_for_everyone()  # Wait before continuing.

        # ------------------------------------------------------------------ #

    @staticmethod
    def setOptimizer(params, lr, weight_decay, submodel=None):
        # General options to choose from:
        #     nAdam = optim.NAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, momentum_decay=0.004, decoupled_weight_decay=True)
        #     adamW = optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False, maximize=False)
        #     rAdam = optim.RAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, decoupled_weight_decay=True)

        # General guidelines:
        #     AdamR (Adam with a warm restart) is an adaptation of the Adam optimizer that incorporates warm restarts to the LR to avoid poor local minima and encourages exploration of the loss landscape by periodically resetting the LR. Use when you converge to suboptimal solutions.
        #     NAdam combines Adam with Nesterov momentum: a variant of the classical momentum technique, taking into account the momentum in a slightly different way that tends to result in better convergence in practice. Use in deep architectures that is sensitive to the optimizer.
        #     AdamW modifies the way Adam implements weight decay, decoupling it from the gradient updates, leading to a more effective use of L2 regularization. Use when regularization is a problem.
        #     If decoupled_weight_decay is True, then it adds adamW to the optimizer
        #     Common WD values: 1E-2 to 1E-6
        #     Common LR values: 1E-6 to 1

        if submodel == "signalEncoder":
            # AdamW hurting the complexity too much; RAdam makes really simple encoded states;
            return optim.NAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, momentum_decay=0.004, decoupled_weight_decay=True)
        elif submodel == "autoencoder":
            return optim.NAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, momentum_decay=0.004, decoupled_weight_decay=True)
        elif submodel == "emotionPrediction":
            # adam and RAdam are okay, AdamW is a bit better (best?); NAdam is also a bit better
            return optim.NAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, momentum_decay=0.004, decoupled_weight_decay=True)
        else:
            assert False, "No optimizer initialized"

    def getLearningRateScheduler(self, submodel):
        # Options:
        # Slow ramp up: transformers.get_constant_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=30)
        # Cosine waveform: optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-8, last_epoch=-1, verbose=False)
        # Reduce on plateau (need further editing of loop): optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        # Defined lambda function: optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_function); lambda_function = lambda epoch: (epoch/50) if epoch < -1 else 1
        # torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, last_epoch=-1)

        # Train the autoencoder
        if submodel == "signalEncoder":
            return transformers.get_constant_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=5 if self.fullTest else 5)
        elif submodel == "autoencoder":
            return transformers.get_constant_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=5 if self.fullTest else 5)
        elif submodel == "emotionPrediction":
            return transformers.get_constant_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=5 if self.fullTest else 5)
        else:
            assert False, "No model initialized"

    def getTrainingEpoch(self, submodel):
        # Get the models, while considering whether they are distributed or not.
        trainingInformation, signalEncoderModel, autoencoderModel, signalMappingModel, sharedEmotionModel, specificEmotionModel = self.getDistributedModels()

        if submodel == "signalEncoder":
            return len(signalEncoderModel.trainingLosses_timeReconstructionAnalysis[self.generalTimeWindowInd])
        elif submodel == "autoencoder":
            return len(autoencoderModel.trainingLosses_timeReconstructionAnalysis[self.generalTimeWindowInd])
        elif submodel == "emotionPrediction":
            return len(specificEmotionModel.trainingLosses_signalReconstruction)
        else:
            raise Exception()

    @staticmethod
    def getSubmodelsSaving(submodel):
        # Get the submodels to save
        if submodel == "signalEncoder":
            submodelsSaving = ["trainingInformation", "signalEncoderModel"]
        elif submodel == "autoencoder":
            submodelsSaving = ["trainingInformation", "signalEncoderModel", "autoencoderModel"]
        elif submodel == "emotionPrediction":
            submodelsSaving = ["trainingInformation", "signalEncoderModel", "autoencoderModel", "signalMappingModel", "specificEmotionModel", "sharedEmotionModel"]
        else:
            assert False, "No model initialized"

        return submodelsSaving

    def getAugmentationDeviation(self, submodel):
        # Get the submodels to save
        if submodel == "signalEncoder":
            addingNoiseRange = (0, 0.01)
        elif submodel == "autoencoder":
            addingNoiseRange = (0, 0.01)
        elif submodel == "emotionPrediction":
            addingNoiseRange = (0, 0.01)
        else:
            assert False, "No model initialized"

        return self.generalMethods.biased_high_sample(*addingNoiseRange, randomValue=random.uniform(a=0, b=1)), addingNoiseRange

    def setupTraining(self, submodel):
        # Do not train the model at all.
        self.setupTrainingFlags(self.model, trainingFlag=False)

        # Get the models, while considering whether they are distributed or not.
        trainingInformation, signalEncoderModel, autoencoderModel, signalMappingModel, sharedEmotionModel, specificEmotionModel = self.getDistributedModels()

        # Label the model we are training.
        if submodel == "signalEncoder":
            self.setupTrainingFlags(signalEncoderModel, trainingFlag=True)
        elif submodel == "autoencoder":
            self.setupTrainingFlags(autoencoderModel, trainingFlag=True)
        elif submodel == "emotionPrediction":
            self.setupTrainingFlags(signalMappingModel, trainingFlag=True)
            self.setupTrainingFlags(specificEmotionModel, trainingFlag=True)
            self.setupTrainingFlags(sharedEmotionModel, trainingFlag=True)
        else:
            assert False, "No model initialized"

    @staticmethod
    def setupTrainingFlags(model, trainingFlag):
        # Set the model to training mode.
        if trainingFlag:
            model.train()
            # Or evaluation mode.
        else:
            model.eval()

        # Recursively set the mode for all submodules
        for submodule in model.modules():  # This ensures all submodules are included
            if trainingFlag:
                submodule.train()
            else:
                submodule.eval()

        # For each model parameter.
        for param in model.parameters():
            # Change the gradient tracking status
            param.requires_grad = trainingFlag

    def getDistributedModel(self):
        # Check if the model is wrapped with DDP
        if hasattr(self.model, 'module'):
            return self.model.module
        else:
            return self.model

    def getDistributedModels(self, model=None, submodel=None):
        if model is None:
            model = self.getDistributedModel()
        # Get the specific models.
        trainingInformation = model.trainingInformation
        signalEncoderModel = model.signalEncoderModel
        autoencoderModel = model.autoencoderModel
        signalMappingModel = model.signalMappingModel
        sharedEmotionModel = model.sharedEmotionModel
        specificEmotionModel = model.specificEmotionModel

        if submodel == "trainingInformation":
            return trainingInformation
        elif submodel == "signalEncoder":
            return signalEncoderModel
        elif submodel == "autoencoder":
            return autoencoderModel
        elif submodel == "emotionPrediction":
            return signalMappingModel, sharedEmotionModel, specificEmotionModel
        elif submodel is None:
            return trainingInformation, signalEncoderModel, autoencoderModel, signalMappingModel, sharedEmotionModel, specificEmotionModel
        else:
            assert False, "No model initialized"
