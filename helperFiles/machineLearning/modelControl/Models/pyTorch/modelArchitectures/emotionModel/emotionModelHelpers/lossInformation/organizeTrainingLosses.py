# General
import time

# PyTorch
import torch

# Helper classes
from .lossCalculations import lossCalculations


class organizeTrainingLosses(lossCalculations):

    def __init__(self, accelerator, model, allEmotionClasses, activityLabelInd, generalTimeWindow):
        super(organizeTrainingLosses, self).__init__(accelerator, model, allEmotionClasses, activityLabelInd)
        self.generalTimeWindow = generalTimeWindow

    # ---------------------------------------------------------------------- #
    # -------------------------- Loss Calculations ------------------------- #     

    def storeTrainingLosses(self, submodel, modelPipeline, lossDataLoader, fastPass):
        self.accelerator.print(f"\nCalculating loss for {modelPipeline.model.datasetName} model", flush=True)

        # Prepare the model/data for evaluation.
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model

        # Load in all the data and labels for final predictions.
        allData, allLabels, allTrainingMasks, allTestingMasks = lossDataLoader.dataset.getAll()
        allSignalData, allDemographicData, allSubjectIdentifiers = self.dataInterface.separateData(allData, model.sequenceLength, model.numSubjectIdentifiers, model.demographicLength)
        reconstructionIndex = self.dataInterface.getReconstructionIndex(allTrainingMasks)
        assert reconstructionIndex is not None

        # Stop gradient tracking.
        with torch.no_grad():

            if submodel == "emotionPrediction":
                # Segment the data into its time window.
                segmentedSignalData = self.dataInterface.getRecentSignalPoints(allSignalData, self.generalTimeWindow)

                t1 = time.time()
                # Pass all the data through the model and store the emotions, activity, and intermediate variables.
                allEncodedData, allReconstructedData, allSignalEncodingLayerLoss, allCompressedData, allReconstructedEncodedData, allDenoisedDoubleReconstructedData, allAutoencoderLayerLoss, allMappedSignalData, \
                    allReconstructedCompressedData, allFeatureData, allActivityDistributions, allBasicEmotionDistributions, allFinalEmotionDistributions \
                    = model.fullDataPass(submodel, lossDataLoader, timeWindow=self.generalTimeWindow, reconstructSignals=False, compileVariables=False, trainingFlag=False)
                t2 = time.time()
                self.accelerator.print("Full Pass", t2 - t1),

                # Calculate the signal encoding loss.
                # manifoldReconstructedTestingLoss, manifoldMeanTestingLoss, manifoldStandardDeviationTestingLoss = \
                #     self.calculateManifoldReductionLoss(allEncodedData, allManifoldData, allTransformedManifoldData, allReconstructedEncodedData, allTestingMasks, reconstructionIndex)
                # manifoldReconstructedTrainingLoss, manifoldMeanTrainingLoss, manifoldStandardDeviationTrainingLoss = \
                #     self.calculateManifoldReductionLoss(allEncodedData, allManifoldData, allTransformedManifoldData, allReconstructedEncodedData, allTrainingMasks, reconstructionIndex)
                # # Store the latent reconstruction loss.
                # self.storeLossInformation(manifoldReconstructedTrainingLoss, manifoldReconstructedTestingLoss, model.signalMappingModel.trainingLosses_encodingReconstruction,
                #                           model.signalMappingModel.testingLosses_encodingReconstruction)
                # self.storeLossInformation(manifoldStandardDeviationTrainingLoss, manifoldStandardDeviationTestingLoss, model.signalMappingModel.trainingLosses_manifoldSTD, model.signalMappingModel.testingLosses_manifoldSTD)
                # self.storeLossInformation(manifoldMeanTrainingLoss, manifoldMeanTestingLoss, model.signalMappingModel.trainingLosses_manifoldMean, model.signalMappingModel.testingLosses_manifoldMean)

                # Calculate the activity classification accuracy/loss and assert the integrity of the loss.
                # activityTestingLoss = self.calculateActivityLoss(allActivityDistributions, allLabels, allTestingMasks, activityClassWeights)
                # activityTrainingLoss = self.calculateActivityLoss(allActivityDistributions, allLabels, allTrainingMasks, activityClassWeights)
                # # Store the activity loss information.
                # self.testingLosses_activities.append(activityTestingLoss)
                # self.trainingLosses_activities.append(activityTrainingLoss)

                # # For each emotion we are predicting.
                # for emotionInd in range(self.numEmotions):
                #     if allEmotionClassWeights[emotionInd] is torch.nan: continue
                #     # Calculate and add the loss due to misclassifying the emotion.
                #     emotionTestingLoss = self.calculateEmotionLoss(emotionInd, allFinalEmotionDistributions, allLabels, allTestingMasks, allEmotionClassWeights) # Calculate the error in the emotion predictions
                #     emotionTrainingLoss = self.calculateEmotionLoss(emotionInd, allFinalEmotionDistributions, allLabels, allTrainingMasks, allEmotionClassWeights) # Calculate the error in the emotion predictions
                #     # Store the loss information.
                #     self.testingLosses_emotions[emotionInd].append(emotionTestingLoss)
                #     self.trainingLosses_emotions[emotionInd].append(emotionTrainingLoss)
                return None

            # For each time window analysis.
            for timeWindowInd in range(len(model.timeWindows)):
                timeWindow = model.timeWindows[timeWindowInd]
                # If we are debugging and do not need to store all the loss values.
                if fastPass and timeWindow != self.generalTimeWindow: continue

                # Segment the data into its time window.
                segmentedSignalData = self.dataInterface.getRecentSignalPoints(allSignalData, timeWindow)

                t1 = time.time()
                # Pass all the data through the model and store the emotions, activity, and intermediate variables.
                segmentedEncodedData, segmentedReconstructedData, segmentedSignalEncodingLayerLoss, segmentedCompressedData, segmentedReconstructedEncodedData, segmentedDenoisedDoubleReconstructedData, segmentedAutoencoderLayerLoss, \
                    segmentedMappedSignalData, segmentedReconstructedCompressedData, segmentedFeatureData, segmentedActivityDistributions, segmentedBasicEmotionDistributions, segmentedFinalEmotionDistributions \
                    = model.fullDataPass(submodel, lossDataLoader, timeWindow=timeWindow, reconstructSignals=True, compileVariables=False, trainingFlag=False)
                t2 = time.time()
                self.accelerator.print(f"{timeWindow} Second Pass", t2 - t1)

                if submodel == "signalEncoder":
                    # Calculate the signal encoding loss.
                    signalReconstructedTestingLoss, encodedMeanTestingLoss, encodedStandardDeviationTestingLoss, signalEncodingTestingLayerLoss = \
                        self.calculateSignalEncodingLoss(segmentedSignalData, segmentedEncodedData, segmentedReconstructedData, segmentedSignalEncodingLayerLoss, allTestingMasks, reconstructionIndex)
                    signalReconstructedTrainingLoss, encodedMeanTrainingLoss, encodedStandardDeviationTrainingLoss, signalEncodingTrainingLayerLoss = \
                        self.calculateSignalEncodingLoss(segmentedSignalData, segmentedEncodedData, segmentedReconstructedData, segmentedSignalEncodingLayerLoss, allTrainingMasks, reconstructionIndex)

                    # Store the signal encoder loss information.
                    self.storeLossInformation(signalReconstructedTrainingLoss, signalReconstructedTestingLoss, model.signalEncoderModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd], model.signalEncoderModel.testingLosses_timeReconstructionAnalysis[timeWindowInd])
                    self.storeLossInformation(encodedStandardDeviationTrainingLoss, encodedStandardDeviationTestingLoss, model.signalEncoderModel.trainingLosses_timeSTDAnalysis[timeWindowInd], model.signalEncoderModel.testingLosses_timeSTDAnalysis[timeWindowInd])
                    self.storeLossInformation(signalEncodingTrainingLayerLoss, signalEncodingTestingLayerLoss, model.signalEncoderModel.trainingLosses_timeLayerAnalysis[timeWindowInd], model.signalEncoderModel.testingLosses_timeLayerAnalysis[timeWindowInd])
                    self.storeLossInformation(encodedMeanTrainingLoss, encodedMeanTestingLoss, model.signalEncoderModel.trainingLosses_timeMeanAnalysis[timeWindowInd], model.signalEncoderModel.testingLosses_timeMeanAnalysis[timeWindowInd])
                    # Store information about the training process.
                    model.signalEncoderModel.numEncodingsBufferPath_timeAnalysis[timeWindowInd].append(model.signalEncoderModel.trainingMethods.keepNumEncodingBuffer)
                    model.signalEncoderModel.numEncodingsPath_timeAnalysis[timeWindowInd].append(model.signalEncoderModel.trainingMethods.numEncodings)
                    # Calculate and Store the optimal loss only once.
                    if len(model.signalEncoderModel.trainingLosses_timeReconstructionSVDAnalysis[timeWindowInd]) == 0:
                        optimalTrainingLoss = self.getOptimalLoss(model.signalEncoderModel.calculateOptimalLoss, segmentedSignalData, allTrainingMasks, reconstructionIndex)
                        optimalTestingLoss = self.getOptimalLoss(model.signalEncoderModel.calculateOptimalLoss, segmentedSignalData, allTestingMasks, reconstructionIndex)

                        # Store the signal encoder loss information.
                        self.storeLossInformation(optimalTrainingLoss, optimalTestingLoss, model.signalEncoderModel.trainingLosses_timeReconstructionSVDAnalysis[timeWindowInd], model.signalEncoderModel.testingLosses_timeReconstructionSVDAnalysis[timeWindowInd])
                    # Inform the user about the final loss.
                    print(f"\tSignal encoder {timeWindow} second losses:", signalReconstructedTrainingLoss.item(), signalReconstructedTestingLoss.item())

                elif submodel == "autoencoder":
                    # Calculate the error in signal reconstruction (autoencoder loss).
                    reconstructedEncodedTestingLoss, compressedMeanTestingLoss, compressedStandardDeviationTestingLoss, autoencoderTestingLayerLoss = \
                        self.calculateAutoencoderLoss(segmentedEncodedData, segmentedCompressedData, segmentedReconstructedEncodedData, segmentedAutoencoderLayerLoss, allTestingMasks, reconstructionIndex)
                    reconstructedEncodedTrainingLoss, compressedMeanTrainingLoss, compressedStandardDeviationTrainingLoss, autoencoderTrainingLayerLoss = \
                        self.calculateAutoencoderLoss(segmentedEncodedData, segmentedCompressedData, segmentedReconstructedEncodedData, segmentedAutoencoderLayerLoss, allTrainingMasks, reconstructionIndex)

                    # Store the latent reconstruction loss.
                    self.storeLossInformation(reconstructedEncodedTrainingLoss, reconstructedEncodedTestingLoss, model.autoencoderModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd], model.autoencoderModel.testingLosses_timeReconstructionAnalysis[timeWindowInd])
                    self.storeLossInformation(compressedStandardDeviationTrainingLoss, compressedStandardDeviationTestingLoss, model.autoencoderModel.trainingLosses_timeSTDAnalysis[timeWindowInd], model.autoencoderModel.testingLosses_timeSTDAnalysis[timeWindowInd])
                    self.storeLossInformation(autoencoderTrainingLayerLoss, autoencoderTestingLayerLoss, model.autoencoderModel.trainingLosses_timeLayerAnalysis[timeWindowInd], model.autoencoderModel.testingLosses_timeLayerAnalysis[timeWindowInd])
                    self.storeLossInformation(compressedMeanTrainingLoss, compressedMeanTestingLoss, model.autoencoderModel.trainingLosses_timeMeanAnalysis[timeWindowInd], model.autoencoderModel.testingLosses_timeMeanAnalysis[timeWindowInd])
                    # Store information about the training process.
                    model.autoencoderModel.numEncodingsBufferPath_timeAnalysis[timeWindowInd].append(model.autoencoderModel.trainingMethods.keepNumEncodingBuffer)
                    model.autoencoderModel.numEncodingsPath_timeAnalysis[timeWindowInd].append(model.autoencoderModel.trainingMethods.numEncodings)

                    # Inform the user about the final loss.
                    print(f"\tAutoencoder {timeWindow} second losses:", reconstructedEncodedTrainingLoss.item(), reconstructedEncodedTestingLoss.item())

    # --------------------------- Helper Methods --------------------------- #

    def storeLossInformation(self, trainingLoss, testingLoss, trainingHolder, testingHolder):
        # Gather the losses from each device (for distributed training). NOTE: don't use gather as it doesn't remove duplicates.
        gatheredTrainingLoss, gatheredTestingLoss = self.accelerator.gather_for_metrics((trainingLoss, testingLoss))

        # Store the loss information.
        trainingHolder.append(gatheredTrainingLoss.detach().mean().item())
        testingHolder.append(gatheredTestingLoss.detach().mean().item())

    # ----------------------- Class Weighting Methods ---------------------- #

    def getClassWeights(self, allLabels, allTrainingMasks, allTestingMasks, numActivities):
        # Initialize placeholder class weights for each emotion as nan.
        allEmotionClassWeights = [torch.nan for _ in range(self.numEmotions)]

        # Get the valid emotion indices (ones with training points).
        emotionTrainingMasks = self.dataInterface.getEmotionMasks(allTrainingMasks, self.numEmotions)
        validEmotionInds = self.dataInterface.getLabelInds_withPoints(emotionTrainingMasks)
        # Create a boolean flag to find any good point.
        validLabelsMask = allTrainingMasks | allTestingMasks

        # For each emotion that has training data.
        for validEmotionInd in validEmotionInds:
            numEmotionClasses = self.allEmotionClasses[validEmotionInd]  # The number of classes (intensity labels) per emotion.

            # Calculate the weight/significance of each emotion class.
            validEmotionLabels = self.dataInterface.getEmotionLabels(validEmotionInd, allLabels, validLabelsMask)
            emotionClassWeight = self.assignClassWeights(validEmotionLabels, numEmotionClasses)
            allEmotionClassWeights[validEmotionInd] = emotionClassWeight

        # Assign the class weights for the activities.      
        activityLabels = self.dataInterface.getActivityLabels(allLabels, validLabelsMask, self.activityLabelInd)
        activityClassWeights = self.assignClassWeights(activityLabels, numActivities)

        return allEmotionClassWeights, activityClassWeights

    @staticmethod
    def assignClassWeights(class_labels, num_classes):
        # Initialize the class counts to zero
        class_counts = torch.zeros(num_classes, device=class_labels.device)

        # Accumulate counts for each class
        for label in class_labels:
            lower_class = int(label)
            # Distribute weight for classes around the floating point
            if lower_class == label:
                # If the label is an integer, it gets full weight
                class_counts[lower_class] += 1
            else:
                # If the label is not an integer, distribute the weight
                upper_class = lower_class + 1
                lower_weight = upper_class - label
                upper_weight = label - lower_class
                class_counts[lower_class] += lower_weight
                class_counts[upper_class] += upper_weight

        # Handle classes with zero counts to avoid division by zero
        nonzero_counts_mask = class_counts > 0
        # Calculate weights for each class
        class_weights = torch.zeros_like(class_counts, device=class_labels.device)
        class_weights[nonzero_counts_mask] = 1.0 / class_counts[nonzero_counts_mask]

        # Normalize weights to sum to 1
        class_weights /= class_weights.sum()

        return class_weights
