# General
import random
import time

# Import files for machine learning
from .emotionPipelineHelpers import emotionPipelineHelpers


class emotionPipeline(emotionPipelineHelpers):

    def __init__(self, accelerator, modelID, datasetName, modelName, allEmotionClasses, sequenceLength, maxNumSignals, numSubjectIdentifiers, demographicLength,
                 numSubjects, userInputParams, emotionNames, activityNames, featureNames, submodel, useParamsHPC, debuggingResults=False):
        # General parameters.
        super().__init__(accelerator, modelID, datasetName, modelName, allEmotionClasses, sequenceLength, maxNumSignals, numSubjectIdentifiers, demographicLength,
                         numSubjects, userInputParams, emotionNames, activityNames, featureNames, submodel, useParamsHPC, debuggingResults)
        # General parameters.
        self.maxBatchSignals = maxNumSignals
        self.calculateFullLoss = False
        self.addingNoiseFlag = False

        # Finish setting up the model.
        self.modelHelpers.l2Normalization(self.model, maxNorm=20, checkOnly=True)
        self.compileOptimizer(submodel)  # Initialize the optimizer (for back propagation)
        self.modelHelpers.switchActivationLayers(self.model, switchState=True)
        self.resetModel()

    def trainModel(self, dataLoader, submodel, numEpochs=500, constrainedTraining=False):
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
        assert numEpochs == 1, f"numEpochs: {numEpochs}"

        # Prepare the model for training.
        self.setupTraining(submodel)

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
                    batchTrainingMask, batchTestingMask = batchTrainingMask.to(self.accelerator.device), batchTestingMask.to(self.accelerator.device)
                    batchData, trueBatchLabels = batchData.to(self.accelerator.device), trueBatchLabels.to(self.accelerator.device)

                    # Set the model intro the training mode.
                    numPointsAnalyzed += batchData.size(0)

                    # Only analyze data that can produce meaningful training results.
                    if submodel in ["signalEncoder", "autoencoder"]:
                        # Get the current training data mask.
                        trainingColumn = self.dataInterface.getEmotionColumn(batchTrainingMask, reconstructionIndex)

                        # Apply the training data mask
                        batchTrainingMask = batchTrainingMask[trainingColumn]
                        trueBatchLabels = trueBatchLabels[trainingColumn]
                        batchData = batchData[trainingColumn]
                        if batchData.size(0) == 0:
                            if self.accelerator.sync_gradients:
                                self.backpropogateModel(constrainedTraining)
                            continue  # We are not training on any points (or need to refresh training)

                    # Separate the data into signal, demographic, and subject identifier information.
                    signalData, demographicData, subjectIdentifiers = self.dataInterface.separateData(batchData, self.sequenceLength, self.numSubjectIdentifiers, self.demographicLength)
                    # demographicData dimension: batchSize, numSignals, demographicLength
                    # signalData dimension: batchSize, numSignals, sequenceLength
                    # subjectInds dimension: batchSize, numSubjectIdentifiers

                    # Randomly choose to add noise to the model.
                    if self.accelerator.sync_gradients:
                        self.calculateFullLoss = random.random() < 0.8 and not constrainedTraining
                        self.addingNoiseFlag = random.random() < 0.5 and not constrainedTraining

                    # Randomly choose to add noise to the model.
                    augmentedSignalData = signalData.clone()
                    addingNoiseRange = [0, 1]
                    addingNoiseSTD = 0

                    if self.addingNoiseFlag:
                        # Augment the data to add some noise to the model.
                        addingNoiseSTD, addingNoiseRange = self.modelParameters.getAugmentationDeviation(submodel)
                        augmentedSignalData = self.dataInterface.addNoise(augmentedSignalData, trainingFlag=True, noiseSTD=addingNoiseSTD)
                        # augmentedSignalData dimension: batchSize, numSignals, sequenceLength

                    # ------------ Forward pass through the model  ------------- #

                    # Train the signal encoder
                    if submodel == "signalEncoder":
                        if self.accelerator.sync_gradients:
                            # Randomly choose to use an inflated number of signals.
                            self.maxBatchSignals = 96 if self.datasetName in ["case"] else max(model.maxNumSignals, 256)
                            self.maxBatchSignals = random.choices(population=[model.maxNumSignals, self.maxBatchSignals], weights=[0.6, 0.4], k=1)[0]

                        # Augment the signals to train an arbitrary sequence length and order.
                        initialSignalData, augmentedSignalData = self.dataInterface.changeNumSignals(signalDatas=(signalData, augmentedSignalData), minNumSignals=model.numEncodedSignals, maxNumSignals=self.maxBatchSignals, alteredDim=1)
                        initialSignalData, augmentedSignalData = self.dataInterface.changeSignalLength(model.timeWindows[0], signalDatas=(initialSignalData, augmentedSignalData))
                        print("Input size:", augmentedSignalData.size())

                        # Perform the forward pass through the model.
                        encodedData, reconstructedData, signalEncodingLayerLoss = model.signalEncoding(augmentedSignalData, initialSignalData, decodeSignals=True, calculateLoss=self.calculateFullLoss, trainingFlag=True)
                        # encodedData dimension: batchSize, numEncodedSignals, sequenceLength
                        # reconstructedData dimension: batchSize, numSignals, sequenceLength
                        # signalEncodingLayerLoss dimension: batchSize

                        # Assert that nothing is wrong with the predictions.
                        self.modelHelpers.assertVariableIntegrity(signalEncodingLayerLoss, variableName="signal encoder layer loss", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(reconstructedData, variableName="reconstructed signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(augmentedSignalData, variableName="augmented signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(signalData, variableName="initial signal data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(encodedData, variableName="encoded data", assertGradient=False)

                        # Calculate the error in signal compression (signal encoding loss).
                        signalReconstructedLoss, encodedSignalMeanLoss, encodedSignalStandardDeviationLoss, signalEncodingTrainingLayerLoss \
                            = self.organizeLossInfo.calculateSignalEncodingLoss(initialSignalData, encodedData, reconstructedData, signalEncodingLayerLoss, batchTrainingMask, reconstructionIndex)
                        if signalReconstructedLoss.item() == 0: self.accelerator.print("Not useful\n\n\n\n\n\n"); continue

                        # Initialize basic core loss value.
                        compressionFactor = augmentedSignalData.size(1) / encodedData.size(1)  # Increase the learning rate for larger compressions.
                        noiseFactor = 1 - (addingNoiseSTD / addingNoiseRange[1]) + 0.5  # Lower the learning rate for high noise levels.
                        finalLoss = compressionFactor * signalReconstructedLoss

                        # Compile the loss into one value
                        if 0.1 < encodedSignalStandardDeviationLoss:
                            finalLoss = finalLoss + encodedSignalStandardDeviationLoss
                        if 0.01 < signalEncodingTrainingLayerLoss:
                            finalLoss = finalLoss + 0.5 * signalEncodingTrainingLayerLoss
                        if 0.1 < encodedSignalMeanLoss:
                            finalLoss = finalLoss + encodedSignalMeanLoss
                        # Account for the current training state when calculating the loss.
                        finalLoss = noiseFactor * finalLoss

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
                        self.modelHelpers.assertVariableIntegrity(denoisedDoubleReconstructedData, variableName="denoised double reconstructed data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(reconstructedEncodedData, variableName="reconstructed encoded data", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(autoencoderLayerLoss, variableName="autoencoder layer loss", assertGradient=False)
                        self.modelHelpers.assertVariableIntegrity(compressedData, variableName="compressed data", assertGradient=False)

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
                        if 0.1 < compressedStandardDeviationLoss:
                            finalLoss = finalLoss + 0.1 * compressedStandardDeviationLoss
                        if 0.01 < autoencoderTrainingLayerLoss:
                            finalLoss = finalLoss + 0.5 * autoencoderTrainingLayerLoss
                        if 0.1 < compressedMeanLoss:
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
                    while 10 < finalLoss: finalLoss = finalLoss / 10

                    t1 = time.time()
                    # Calculate the gradients.
                    self.accelerator.backward(finalLoss)  # Calculate the gradients.
                    self.backpropogateModel(constrainedTraining)
                    t2 = time.time(); self.accelerator.print(f"Backprop {self.datasetName} {numPointsAnalyzed}:", t2 - t1)
            # Finalize all the parameters.
            if not constrainedTraining:
                self.scheduler.step()  # Update the learning rate.
            else:
                self.constrainedScheduler.step()  # Update the learning rate.

        # Prepare the model/data for evaluation.
        self.setupTrainingFlags(self.model, trainingFlag=False)  # Set all models into evaluation mode.
        self.accelerator.wait_for_everyone()  # Wait before continuing.

    def backpropogateModel(self, constrainedTraining):
        # Clip the gradients if they are too large.
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), 20)  # Apply gradient clipping: Small: <1; Medium: 5-10; Large: >20

        if constrainedTraining:
            self.constrainedOptimizer.step()
            self.constrainedOptimizer.zero_grad()  # Zero your gradients to restart the gradient tracking.
            self.accelerator.print("LR:", self.constrainedScheduler.get_last_lr())
        else:
            # Backpropagation the gradient.
            self.optimizer.step()  # Adjust the weights.
            self.optimizer.zero_grad()  # Zero your gradients to restart the gradient tracking.
            self.accelerator.print("LR:", self.scheduler.get_last_lr())
