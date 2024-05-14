import numpy as np
import itertools
import pickle
import random
import torch
import gzip
import os

# Import helper files.
from ..modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from ..modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from ..modelControl.Models.pyTorch.Helpers.modelMigration import modelMigration
from .dataPreparation import standardizeData


class compileModelDataHelpers:
    def __init__(self, submodel, userInputParams, accelerator=None):
        # General parameters
        self.compiledInfoLocation = os.path.dirname(__file__) + "/../../../_experimentalData/_compiledData/"
        self.compiledExtension = ".pkl.gz"
        self.userInputParams = userInputParams
        self.missingLabelValue = torch.nan
        self.accelerator = accelerator

        # Make Output Folder Directory if Not Already Created
        os.makedirs(self.compiledInfoLocation, exist_ok=True)

        # Initialize relevant classes.
        self.modelParameters = modelParameters(userInputParams,  accelerator)
        self.modelMigration = modelMigration(accelerator, debugFlag=False)
        self.dataInterface = emotionDataInterface

        # Submodel-specific parameters
        self.emotionPredictionModelInfo = None
        self.signalEncoderModelInfo = None
        self.autoencoderModelInfo = None
        self.numIndices_perShift = None
        self.maxClassPercentage = None
        self.dontShiftDatasets = None
        self.numSecondsShift = None
        self.minNumClasses = None
        self.minSeqLength = None
        self.maxSeqLength = None
        self.numShifts = None

        # Set the submodel-specific parameters
        if submodel is not None: self.addSubmodelParameters(submodel, userInputParams)

    def addSubmodelParameters(self, submodel, userInputParams):
        self.userInputParams = userInputParams

        # Exclusion criterion.
        self.minNumClasses, self.maxClassPercentage = self.modelParameters.getExclusionCriteria(submodel)
        self.minSeqLength, self.maxSeqLength = self.modelParameters.getSequenceLength(submodel, userInputParams['sequenceLength'])  # Seconds.

        # Data augmentation.
        samplingFrequency = 1
        self.dontShiftDatasets, self.numSecondsShift, numSeconds_perShift = self.modelParameters.getShiftInfo(submodel)
        # Data augmentation parameters calculations
        self.numShifts = 1 + self.numSecondsShift // numSeconds_perShift  # The first shift is the identity transformation.
        self.numIndices_perShift = (samplingFrequency * numSeconds_perShift)  # This must be an integer
        assert samplingFrequency == 1, "Check your code if samplingFrequency != 1 is okay."

    # ---------------------- Model Specific Parameters --------------------- #

    def embedInformation(self, submodel, trainingDate):
        if submodel == "signalEncoder":
            trainingDate = f"{trainingDate} {self.signalEncoderModelInfo}"
        elif submodel == "autoencoder":
            trainingDate = f"{trainingDate} {self.autoencoderModelInfo}"
        elif submodel == "emotionPrediction":
            trainingDate = f"{trainingDate} {self.emotionPredictionModelInfo}"
        else:
            raise Exception()
        print("trainingDate:", trainingDate, flush=True)

        return trainingDate

    # ---------------------- Saving/Loading Model Data --------------------- #
    def saveCompiledInfo(self, data_to_store, saveDataName):
        with gzip.open(filename=f'{self.compiledInfoLocation}{saveDataName}{self.compiledExtension}', mode='wb') as file:
            pickle.dump(data_to_store, file)

    def loadCompiledInfo(self, loadDataName):
        with gzip.open(filename=f'{self.compiledInfoLocation}{loadDataName}{self.compiledExtension}', mode='rb') as file:
            data_loaded = pickle.load(file)
        return data_loaded[f"{loadDataName}"]

    # ------------------------- Signal Organization ------------------------ #

    @staticmethod
    def organizeActivityLabels(activityNames, activityLabels):
        """
        Purpose: To remove any activityNames where the label is not present and 
                 reindex the activityLabels from 0 to number of unique activities after culling.
        Parameters:
        - activityNames: a list of all unique activity names (strings).
        - activityLabels: a 1D tensor of index hashes for the unique activityName.
        """
        # Convert activityNames to a numpy array for easier string handling
        activityNames = np.asarray(activityNames)

        # Find the unique activity labels
        uniqueActivityLabels, culledActivityLabels = torch.unique(activityLabels, return_inverse=True)

        # Get the corresponding unique activity names
        uniqueActivityNames = activityNames[uniqueActivityLabels.int()]

        assert len(activityLabels) == len(culledActivityLabels)
        return uniqueActivityNames, culledActivityLabels.double()

    @staticmethod
    def segmentSignals_toModel(numSignals, numSignalsCombine, random_state, metaTraining=True):
        random.seed(random_state)

        # If it's our data.
        if not metaTraining:
            signalCombinationInds = itertools.combinations(range(0, numSignals), numSignalsCombine)
            assert numSignalsCombine <= numSignals, f"You cannot make {numSignalsCombine} combination with only {numSignals} signals."
        else:
            signalCombinationInds = []
            allCombinations = list(range(numSignals))
            for _ in range(1):
                # Create a list of all possible combinations of indices
                random.shuffle(allCombinations)

                # Iterate over the shuffled list and create signalCombinationInds
                for groupInd in range(0, len(allCombinations), numSignalsCombine):
                    # Form a group of random signals
                    group = allCombinations[groupInd:groupInd + numSignalsCombine]
                    # If the group is too small, add some more.
                    if len(group) < numSignalsCombine:
                        group.extend(allCombinations[0:numSignalsCombine - len(group)])

                    # Organize the signal groupings.
                    signalCombinationInds.append(group)

        print(f"\tPotentially Initializing {len(signalCombinationInds)} models per emotion", flush=True)
        return signalCombinationInds

    def organizeSignals(self, allSignalData, signalInds):
        """
        Purpose: Create the final signal array (of correct length).
        --------------------------------------------
        allSignalData : A list of size (batchSize, numSignals, sequenceLength)
        signalInds : A list of size numSignalsCombine
        """
        featureData = []
        # Compile the feature's data across all experiments.
        for experimentInd in range(len(allSignalData)):
            signalData = np.asarray(allSignalData[experimentInd])
            data = signalData[signalInds, :]

            # Assertions about data integrity.
            numSignals, sequenceLength = data.shape
            assert self.minSeqLength <= sequenceLength, f"Expected {self.minSeqLength}, but received {data.shape[1]} "

            # Standardize the signals
            standardizeClass = standardizeData(data, axisDimension=1, threshold=0)
            data = standardizeClass.standardize(data)

            # Add buffer if needed.
            if sequenceLength < self.maxSeqLength + self.numSecondsShift:
                prependedBuffer = np.ones((numSignals, self.maxSeqLength + self.numSecondsShift - sequenceLength)) * data[:, 0:1]
                data = np.hstack((prependedBuffer, data))
            elif self.maxSeqLength + self.numSecondsShift < sequenceLength:
                data = data[:, -self.maxSeqLength - self.numSecondsShift:]

            # This is good data
            featureData.append(data.tolist())
        featureData = torch.tensor(featureData)

        # Assert the integrity.
        assert len(featureData) != 0

        return featureData

    def organizeLabels(self, allFeatureLabels, metaTraining, metaDatasetName, numSignals):
        # Convert to tensor and zero out the class indices
        allFeatureLabels = torch.tensor(np.asarray(allFeatureLabels))

        allSingleClassIndices = []
        # For each type of label recorded during the trial.
        for labelTypeInd in range(allFeatureLabels.shape[1]):
            featureLabels = allFeatureLabels[:, labelTypeInd]
            allSingleClassIndices.append([])

            # Remove unknown labels.
            goodLabelInds = 0 <= featureLabels  # The minimum label should be 0
            featureLabels[~goodLabelInds] = self.missingLabelValue

            # Count the classes
            unique_classes, class_counts = torch.unique(featureLabels[goodLabelInds], return_counts=True)
            if (class_counts < 2).any():
                # Find the bad experiments (batches) with only one sample.
                badClasses = unique_classes[torch.nonzero(class_counts < 2).reshape(1, -1)[0]]

                # Remove the datapoint as we cannot split the class between test/train
                badLabelInds = torch.isin(featureLabels, badClasses)
                allSingleClassIndices[-1].extend(badLabelInds)

                # Reassess the class counts
                finalMask = goodLabelInds & ~badLabelInds
                unique_classes, class_counts = torch.unique(featureLabels[finalMask], return_counts=True)

            # Ensure greater variability in the class rating system.
            if metaTraining and (len(unique_classes) < self.minNumClasses or len(featureLabels) * self.maxClassPercentage <= class_counts.max().item()):
                featureLabels[:] = self.missingLabelValue

            # Save the edits made to the featureLabels
            allFeatureLabels[:, labelTypeInd] = featureLabels

        # Report back the information from this dataset
        numExperiments, numAllLabels = allFeatureLabels.shape
        numGoodEmotions = torch.sum(~torch.all(torch.isnan(allFeatureLabels), dim=0)).item()
        print(f"\t{metaDatasetName.capitalize()}: Found {numGoodEmotions - 1} (out of {numAllLabels - 1}) well-labeled emotions across {numExperiments} experiments with {numSignals} signals.", flush=True)

        return allFeatureLabels, allSingleClassIndices

    def addDemographicInfo(self, allFeatureData, allSubjectInds, datasetInd):
        """
        Purpose: The same signal without the last few seconds still has the same label
        --------------------------------------------
        allFeatureData : A 3D list of all signals in each experiment (batchSize, numSignals, sequenceLength)
        allSubjectInds : A 1D numpy array of size batchSize
        """
        # Get the dimensions of the input arrays
        numExperiments, numSignals, totalLength = allFeatureData.shape
        demographicLength = 0
        numSubjectIdentifiers = 2

        # Create lists to store the new augmented data.
        updatedFeatureData = torch.zeros((numExperiments, numSignals, self.maxSeqLength + demographicLength + 2))

        # For each recorded experiment.
        for experimentInd in range(numExperiments):
            # Compile an array of subject indices.
            subjectInds = torch.full((numSignals, 1), allSubjectInds[experimentInd])
            datasetInds = torch.full((numSignals, 1), datasetInd)

            # Collect the demographic information.
            demographicContext = torch.hstack((subjectInds, datasetInds))
            assert demographicLength + numSubjectIdentifiers == demographicContext.shape[1], "Asserting I am self-consistent. Hardcoded assertion"

            # Add the demographic data to the feature array.
            updatedFeatureData[experimentInd] = torch.hstack((allFeatureData[experimentInd], demographicContext))

        return updatedFeatureData, numSubjectIdentifiers, demographicLength

    # ---------------------------------------------------------------------- #
    # -------------------------- Data Augmentation ------------------------- #

    def addShiftedSignals(self, allFeatureData, allFeatureLabels, currentTrainingMask, currentTestingMask, allSubjectInds):
        """
        Purpose: The same signal without the last few seconds still has the same label
        --------------------------------------------
        allFeatureData : A 3D list of all signals in each experiment (batchSize, numSignals, sequenceLength)
        allFeatureLabels : A numpy array of all labels per experiment of size (batchSize, numLabels)
        currentTestingMask : A boolean mask of testing data of size (batchSize, numLabels)
        currentTrainingMask : A boolean mask of training data of size (batchSize, numLabels)
        allSubjectInds : A 1D numpy array of size batchSize
        """
        # Get the dimensions of the input arrays
        numExperiments, numSignals, totalLength = allFeatureData.shape
        _, numLabels = allFeatureLabels.shape

        # Create lists to store the new augmented data, labels, and masks
        augmentedFeatureData = torch.zeros((numExperiments * self.numShifts, numSignals, self.maxSeqLength))
        augmentedFeatureLabels = torch.zeros((numExperiments * self.numShifts, numLabels))
        augmentedTrainingMask = torch.full(augmentedFeatureLabels.shape, False, dtype=torch.bool)
        augmentedTestingMask = torch.full(augmentedFeatureLabels.shape, False, dtype=torch.bool)
        augmentedSubjectInds = torch.zeros((numExperiments * self.numShifts))

        # For each recorded experiment.
        for experimentInd in range(numExperiments):

            # For each shift in the signals.
            for shiftInd in range(self.numShifts):
                # Create shifted signals
                shiftedSignals = allFeatureData[experimentInd, :, -self.maxSeqLength - shiftInd * self.numIndices_perShift:totalLength - shiftInd * self.numIndices_perShift]

                # Append the shifted data and corresponding labels and masks
                augmentedFeatureData[experimentInd * self.numShifts + shiftInd] = shiftedSignals
                augmentedFeatureLabels[experimentInd * self.numShifts + shiftInd] = allFeatureLabels[experimentInd]
                augmentedTrainingMask[experimentInd * self.numShifts + shiftInd] = currentTrainingMask[experimentInd]
                augmentedTestingMask[experimentInd * self.numShifts + shiftInd] = currentTestingMask[experimentInd]
                augmentedSubjectInds[experimentInd * self.numShifts + shiftInd] = allSubjectInds[experimentInd]

        # import matplotlib.pyplot as plt
        # plt.plot(allFeatureData[0][0][-self.maxSeqLength:], 'k', linewidth=3, label="Original Curve")
        # plt.plot(augmentedFeatureData[0][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[1][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[2][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[3][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[4][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[5][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[6][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[7][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[8][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.plot(augmentedFeatureData[9][0], 'tab:red', linewidth=1, label="Shifted Curve", alpha = 0.5)
        # plt.xlabel("Time (Seconds)")
        # plt.ylabel("AU")
        # # plt.plot(augmentedFeatureData[0][0], 'tab:blue', linewidth=2)
        # plt.show()

        return augmentedFeatureData, augmentedFeatureLabels, augmentedTrainingMask, augmentedTestingMask, augmentedSubjectInds

    # ---------------------------------------------------------------------- #
    # ---------------------------- Data Cleaning --------------------------- #

    def _removeBadExperiments(self, allSignalData, allLabels, subjectInds):
        """
        Purpose: Remove bad experiments from the data list.
        --------------------------------------------
        allSignalData : A list of size (batchSize, numSignals, sequenceLength)
        allLabels : A 2D numpy array of size (batchSize, numLabels)
        subjectInds : A 1D numpy array of size (batchSize,)
        """
        # Initialize data holders.
        goodBatchInds = []
        finalData = []

        # For each set of signals.
        for dataInd in range(len(allSignalData)):
            signalData = np.asarray(allSignalData[dataInd])
            numSignals, sequenceLength = signalData.shape

            # Assert that the signal is long enough.
            if sequenceLength < self.minSeqLength: continue

            # Compile the good batches.
            goodBatchInds.append(dataInd)
            finalData.append(signalData)

        return finalData, allLabels[goodBatchInds], subjectInds[goodBatchInds]

    @staticmethod
    def _removeBadSignals(allSignalData, featureNames):
        """
        Purpose: Remove poor signals from ALL data batches.
        --------------------------------------------
        allSignalData : A list of size (batchSize, numSignals, sequenceLength)
        featureNames : A numpy array of size numSignals
        """
        # Initialize data holders.
        goodFeatureInds = set(np.arange(0, len(featureNames)))
        assert len(allSignalData) == 0 or len(featureNames) == len(allSignalData[0]), allSignalData

        # For each set of signals.
        for dataInd in range(len(allSignalData)):
            signalData = np.asarray(allSignalData[dataInd])

            # Calculate time-series statistics for each signal.
            signalStandardDeviations = np.std(signalData, axis=1)

            # Remove any feature that doesn't vary enough.
            badIndices = np.where(signalStandardDeviations == 0)[0]
            goodFeatureInds.difference_update(badIndices)
        # Convert the set into a sorted list ofn indices.
        goodFeatureInds = np.array(list(goodFeatureInds))
        goodFeatureInds.sort()

        finalData = []
        # For each experimental data point.
        for experimentInd in range(len(allSignalData)):
            signalData = np.asarray(allSignalData[experimentInd])

            finalData.append(signalData[goodFeatureInds, :])

        # Only featureNames the labels of the good experiments.
        featureNames = featureNames[goodFeatureInds]

        return finalData, featureNames

    # ---------------------------------------------------------------------- #
    # -------------------------- Data Organization ------------------------- #

    @staticmethod
    def _selectSignals(allSignalData, signalInds):
        """
        Purpose: keep of the signalInds in allSignalData.
        --------------------------------------------
        allSignalData : A list of size (batchSize, numSignals, sequenceLength)
        featureInds : A list of size numSignalsKeep
        """
        # Base case, no indices selected.
        if signalInds is None:
            return allSignalData

        selectedData = []
        # For each set of signals.
        for dataInd in range(len(allSignalData)):
            # Select and store only the signalInds.
            signalData = np.asarray(allSignalData[dataInd])
            selectedData.append(signalData[signalInds, :])
        return selectedData

    # ---------------------------------------------------------------------- #
