# General
import gzip
import os
import copy
import pickle
import random
import itertools
import numpy as np
from sklearn.model_selection import train_test_split

# PyTorch
import torch

# Import files for training and testing the model
from ..modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionPipeline import emotionPipeline
from ..modelControl.modelSpecifications.compileModelInfo import compileModelInfo  # Functions with model information

# Import interfaces for the model's data
from ..modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from ...dataAcquisitionAndAnalysis.metadataAnalysis.globalMetaAnalysis import globalMetaAnalysis
from ..featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames  # Functions to extract feature names
from ..modelControl.Models.pyTorch.Helpers.dataLoaderPyTorch import pytorchDataInterface
from ..modelControl.Models.pyTorch.Helpers.modelMigration import modelMigration
from .dataPreparation import standardizeData


class compileModelData:
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
        self.modelMigration = modelMigration(accelerator, debugFlag=False)
        self.modelInfoClass = compileModelInfo("_.pkl", [0, 1, 2])
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
        self.minNumClasses, self.maxClassPercentage = self.getExclusionCriteria(submodel)
        self.minSeqLength, self.maxSeqLength = self.getSequenceLength(submodel, userInputParams['sequenceLength'])  # Seconds.

        # Data augmentation.
        samplingFrequency = 1
        self.dontShiftDatasets, self.numSecondsShift, numSeconds_perShift = self.getShiftInfo(submodel)
        # Data augmentation parameters calculations
        self.numShifts = 1 + self.numSecondsShift // numSeconds_perShift  # The first shift is the identity transformation.
        self.numIndices_perShift = (samplingFrequency * numSeconds_perShift)  # This must be an integer
        assert samplingFrequency == 1, "Check your code if samplingFrequency != 1 is okay."

        # Embedded information for each model.
        self.signalEncoderModelInfo = f"signalEncoder on {userInputParams['deviceListed']} at numLiftedChannels {userInputParams['numLiftedChannels']} at numExpandedSignals {userInputParams['numExpandedSignals']} at numEncodingLayers {userInputParams['numEncodingLayers']}"
        self.autoencoderModelInfo = f"autoencoder on {userInputParams['deviceListed']} at compressionFactor {str(userInputParams['compressionFactor']).replace('.', '')} expansionFactor {str(userInputParams['expansionFactor']).replace('.', '')}"
        self.emotionPredictionModelInfo = f"emotionPrediction on {userInputParams['deviceListed']} with seqLength {self.maxSeqLength}"

    # ---------------------------------------------------------------------- #
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

    @staticmethod
    def getModelInfo(submodel, specificInfo=None):
        if specificInfo is not None:
            return specificInfo

        elif submodel == "signalEncoder":
            # No model information to load.
            loadSubmodel = None
            loadSubmodelDate = None
            loadSubmodelEpochs = None

        elif submodel == "autoencoder":
            # Model loading information.
            loadSubmodelEpochs = -1  # The number of epochs the loading model was trained.
            loadSubmodel = "signalEncoder"  # The model's component we are loading.
            loadSubmodelDate = f"2024-04-06 Final signalEncoder on cuda at numExpandedSignals 4 at numEncodingLayers 4"  # The date the model was trained.

        elif submodel == "emotionPrediction":
            # Model loading information.
            loadSubmodelEpochs = -1  # The number of epochs the loading model was trained.
            loadSubmodel = "autoencoder"  # The model's component we are loading.
            loadSubmodelDate = f"2024-01-10 Final signalEncoder"  # The date the model was trained.
        else:
            raise Exception()

        return loadSubmodelDate, loadSubmodelEpochs, loadSubmodel

    def getBatchSize(self, submodel, metaDatasetName):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 75 experiments with 128 signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 99 signals.
        # Amigos: Found 12 (out of 12) well-labeled emotions across 178 experiments with 232 signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 41 signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1650 experiments with 87 signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 191 experiments with 183 signals.

        if submodel == "signalEncoder":
            totalMinBatchSize = 16 if self.userInputParams['deviceListed'].startswith("HPC") else 16
        elif submodel == "autoencoder":
            totalMinBatchSize = 16 if self.userInputParams['deviceListed'].startswith("HPC") else 16
        elif submodel == "emotionPrediction":
            totalMinBatchSize = 16 if self.userInputParams['deviceListed'].startswith("HPC") else 16
        else:
            raise Exception()

        # Adjust the batch size based on the number of gradient accumulations.
        gradientAccumulation = self.accelerator.gradient_accumulation_steps
        minimumBatchSize = totalMinBatchSize // gradientAccumulation
        # Assert that the batch size is divisible by the gradient accumulation steps.
        assert totalMinBatchSize % gradientAccumulation == 0, "The total batch size must be divisible by the gradient accumulation steps."
        assert gradientAccumulation <= totalMinBatchSize, "The gradient accumulation steps must be less than the total batch size."

        # Specify the small batch size datasets.
        if metaDatasetName in ['wesad']: return minimumBatchSize  # 1 times larger than the smallest dataset.

        # Specify the small-medium batch size datasets.
        if metaDatasetName in ['empatch']: return 2 * minimumBatchSize  # 2.5466 times larger than the smallest dataset.
        if metaDatasetName in ['amigos']: return 2 * minimumBatchSize  # 2.3733 times larger than the smallest dataset.

        # Specify the medium batch size datasets.
        if metaDatasetName in ['emognition']: return 4 * minimumBatchSize  # 5.4266 times larger than the smallest dataset.
        if metaDatasetName in ['dapper']: return 4 * minimumBatchSize  # 4.8533 times larger than the smallest dataset.

        # Specify the large batch size datasets.
        if metaDatasetName in ['case']: return 16 * minimumBatchSize  # 22 times larger than the smallest dataset.

        assert False, f"Dataset {metaDatasetName} not found for submodel {submodel}."

    def getMaxBatchSize(self, submodel, numSignals):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 75 experiments with 128 signals. 2.28125 times smaller than the largest signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 99 signals. 2.949 times smaller than the largest signals.
        # Amigos: Found 12 (out of 12) well-labeled emotions across 178 experiments with 232 signals. 1.0 times smaller than the largest signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 41 signals. 7.1219 times smaller than the largest signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1650 experiments with 87 signals. 3.356 times smaller than the largest signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 191 experiments with 183 signals. 1.5956 times smaller than the largest signals.

        if submodel == "signalEncoder":
            minimumBatchSize = 16 if self.userInputParams['deviceListed'].startswith("HPC") else 16
        elif submodel == "autoencoder":
            minimumBatchSize = 16 if self.userInputParams['deviceListed'].startswith("HPC") else 16
        elif submodel == "emotionPrediction":
            minimumBatchSize = 16 if self.userInputParams['deviceListed'].startswith("HPC") else 16
        else:
            raise Exception()

        maxNumSignals = 232
        # Adjust the batch size based on the number of signals used.
        return int(minimumBatchSize * maxNumSignals / numSignals)

    @staticmethod
    def getSequenceLength(submodel, sequenceLength):
        if submodel == "signalEncoder":
            return 90, 240
        elif submodel == "autoencoder":
            return 90, 240
        elif submodel == "emotionPrediction":
            return sequenceLength, sequenceLength
        else:
            raise Exception()

    @staticmethod
    def getShiftInfo(submodel):
        if submodel == "signalEncoder":
            return ["wesad", "emognition", "amigos", "dapper", "case"], 40, 50
        elif submodel == "autoencoder":
            return ["wesad", "emognition", "amigos", "dapper", "case"], 40, 50
        elif submodel == "emotionPrediction":
            return ['case', 'amigos'], 4, 2
        else:
            raise Exception()

    @staticmethod
    def getExclusionCriteria(submodel):
        if submodel == "signalEncoder":
            return -1, 2  # Emotion classes dont matter.
        elif submodel == "autoencoder":
            return -1, 2  # Emotion classes dont matter.
        elif submodel == "emotionPrediction":
            return 2, 0.8
        else:
            raise Exception()

    @staticmethod
    def getEpochInfo(submodel):
        if submodel == "signalEncoder":
            return 10, 1
        elif submodel == "autoencoder":
            return 10, 1
        elif submodel == "emotionPrediction":
            return 10, 1
        else:
            raise Exception()

    # ---------------------------------------------------------------------- #
    # ------------------------ Compile Analysis Data ------------------------ #

    def compileProjectAnalysis(self, loadCompiledData=False, compiledModelName="compiledProjectTrainingInfo"):
        print(f"Reading in data for empatch")

        # Base case: we are loading in data that was already compiled.
        if loadCompiledData and os.path.isfile(f'{self.compiledInfoLocation}{compiledModelName}{self.compiledExtension}'): return self.loadCompiledInfo(compiledModelName)

        trainingFolder = os.path.normpath(os.path.dirname(__file__) + "/../../../_experimentalData/allSensors/_finalDataset/") + "/"
        # Specify biomarker information.
        streamingOrder = ["eog", "eeg", "eda", "temp"]  # A List Representing the Order of the Sensors being Streamed in.
        extractFeaturesFrom = ["eog", "eeg", "eda", "temp"]  # A list with all the biomarkers from streamingOrder for feature extraction
        featureAverageWindows = [60, 30, 30, 30]  # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60
        plotTrainingData = False

        numQuestionOptions = [5] * 10
        numQuestionOptions.extend([4] * 20)

        # Compile feature names
        featureNames, biomarkerFeatureNames, biomarkerOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

        globalMetaAnalysisClass = globalMetaAnalysis(trainingFolder, surveyQuestions=[])
        # Extract the features from the training files and organize them.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
            subjectOrder, experimentalOrder, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
            = globalMetaAnalysisClass.trainingProtocolInterface(streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData, metaTraining=False)

        # Compile experimental information.
        userNames = np.unique([i.split(" ")[-2].lower() for i in subjectOrder])
        subjectOrder = np.array([np.where(userName.split(" ")[-2].lower() == userNames)[0][0] for userName in subjectOrder])
        activityNames, activityLabels = self.modelInfoClass.extractActivityInformation(experimentalOrder, distinguishBaselines=False)

        data_to_store = {
            # Compile the project data together
            f"{compiledModelName}": [allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
                                     allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
                                     subjectOrder, experimentalOrder, activityNames, activityLabels, allFinalFeatures, allFinalLabels, featureLabelTypes,
                                     featureNames, surveyQuestions, surveyAnswersList, surveyAnswerTimes, numQuestionOptions]
        }
        # Update the compiled data so the next person can use it.
        self.saveCompiledInfo(data_to_store, compiledModelName)

        return allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
            subjectOrder, experimentalOrder, activityNames, activityLabels, allFinalFeatures, allFinalLabels, featureLabelTypes, \
            featureNames, surveyQuestions, surveyAnswersList, surveyAnswerTimes, numQuestionOptions

    def compileMetaAnalyses(self, metaProtocolInterfaces, loadCompiledData=False, compiledModelName="compiledMetaTrainingInfo"):
        # Prepare to compile all the metadata analyses.
        metaSurveyQuestions, metaSurveyAnswersList, metaSurveyAnswerTimes, metaNumQuestionOptions = [], [], [], []
        metaSubjectOrder, metaExperimentalOrder, metaFinalFeatures, metaFinalLabels, metaFeatureLabelTypes = [], [], [], [], []
        metaRawFeatureTimesHolders, metaRawFeatureHolders, metaRawFeatureIntervals, metaRawFeatureIntervalTimes = [], [], [], []
        metaAlignedFeatureTimes, metaAlignedFeatureHolder, metaAlignedFeatureIntervals, metaAlignedFeatureIntervalTimes = [], [], [], []
        metaFeatureNames, metaActivityNames, metaActivityLabels, metaDatasetNames = [], [], [], []

        plotTrainingData = False
        # For each meta-analysis protocol
        for metaAnalysisProtocol in metaProtocolInterfaces:
            # Prepare the data to go through the training interface.
            streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames = metaAnalysisProtocol.compileTrainingInfo()
            compiledModelFinalName = compiledModelName + f"_{metaAnalysisProtocol.datasetName}"
            featureNames = np.asarray(list(itertools.chain(*biomarkerFeatureNames)))
            numQuestionOptions = metaAnalysisProtocol.numQuestionOptions
            datasetName = metaAnalysisProtocol.datasetName

            print(f"Reading in metadata for {datasetName}")
            # Base case: we are loading in data that was already compiled.
            if loadCompiledData and os.path.isfile(f'{self.compiledInfoLocation}{compiledModelFinalName}{self.compiledExtension}'):
                allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
                    allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, subjectOrder, \
                    experimentalOrder, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes, \
                    activityNames, activityLabels = self.loadCompiledInfo(compiledModelFinalName)
            else:
                # Collected the training data.
                allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
                    allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, subjectOrder, \
                    experimentalOrder, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
                    = metaAnalysisProtocol.trainingProtocolInterface(streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData, metaTraining=True)

                # Compile experimental information.
                activityNames, activityLabels = metaAnalysisProtocol.extractExperimentLabels(experimentalOrder)

                # Update the compiled data so the next person can use it.
                data_to_store = {f"{compiledModelFinalName}": [allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes,
                                                               allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, subjectOrder,
                                                               experimentalOrder, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes,
                                                               activityNames, activityLabels]}
                self.saveCompiledInfo(data_to_store, compiledModelFinalName)

            # Organize all the metadata analyses.
            metaDatasetNames.append(datasetName)
            metaSubjectOrder.append(subjectOrder)
            metaFeatureNames.append(featureNames)
            metaFinalLabels.append(allFinalLabels)
            metaActivityNames.append(activityNames)
            metaActivityLabels.append(activityLabels)
            metaFinalFeatures.append(allFinalFeatures)
            metaSurveyQuestions.append(surveyQuestions)
            metaExperimentalOrder.append(experimentalOrder)
            metaSurveyAnswersList.append(surveyAnswersList)
            metaSurveyAnswerTimes.append(surveyAnswerTimes)
            metaFeatureLabelTypes.append(featureLabelTypes)
            metaNumQuestionOptions.append(numQuestionOptions)
            metaRawFeatureHolders.append(allRawFeatureHolders)
            metaRawFeatureIntervals.append(allRawFeatureIntervals)
            metaAlignedFeatureTimes.append(allAlignedFeatureTimes)
            metaAlignedFeatureHolder.append(allAlignedFeatureHolder)
            metaRawFeatureTimesHolders.append(allRawFeatureTimesHolders)
            metaRawFeatureIntervalTimes.append(allRawFeatureIntervalTimes)
            metaAlignedFeatureIntervals.append(allAlignedFeatureIntervals)
            metaAlignedFeatureIntervalTimes.append(allAlignedFeatureIntervalTimes)

        return metaRawFeatureTimesHolders, metaRawFeatureHolders, metaRawFeatureIntervals, metaRawFeatureIntervalTimes, \
            metaAlignedFeatureTimes, metaAlignedFeatureHolder, metaAlignedFeatureIntervals, metaAlignedFeatureIntervalTimes, \
            metaSubjectOrder, metaExperimentalOrder, metaActivityNames, metaActivityLabels, metaFinalFeatures, metaFinalLabels, \
            metaFeatureLabelTypes, metaFeatureNames, metaSurveyQuestions, metaSurveyAnswersList, metaSurveyAnswerTimes, metaNumQuestionOptions, metaDatasetNames

    def saveCompiledInfo(self, data_to_store, saveDataName):
        with gzip.open(f'{self.compiledInfoLocation}{saveDataName}{self.compiledExtension}', 'wb') as file:
            pickle.dump(data_to_store, file)

    def loadCompiledInfo(self, loadDataName):
        with gzip.open(f'{self.compiledInfoLocation}{loadDataName}{self.compiledExtension}', 'rb') as file:
            data_loaded = pickle.load(file)
        return data_loaded[f"{loadDataName}"]

    # ---------------------------------------------------------------------- #
    # -------------------- Machine Learning Preparation -------------------- #

    def compileModels(self, metaAlignedFeatureIntervals, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions,
                      metaSubjectOrder, metaFeatureNames, metaDatasetNames, modelName, submodel, testSplitRatio, fullTest, metaTraining, specificInfo=None, random_state=42):
        # Initialize relevant holders.
        lossDataHolders = []
        allDataLoaders = []
        allModelPipelines = []

        # Specify model parameters
        loadSubmodelDate, loadSubmodelEpochs, loadSubmodel = self.getModelInfo(submodel, specificInfo)

        print(f"\nSplitting the data into {'meta-' if metaTraining else ''}models:", flush=True)
        # For each meta-information collected.
        for metadataInd in range(len(metaAlignedFeatureIntervals)):

            allAlignedFeatureIntervals = metaAlignedFeatureIntervals[metadataInd].copy()
            numQuestionOptions = metaNumQuestionOptions[metadataInd].copy()
            subjectOrder = np.asarray(metaSubjectOrder[metadataInd]).copy()
            surveyAnswersList = metaSurveyAnswersList[metadataInd].copy() - 1
            surveyQuestions = metaSurveyQuestions[metadataInd].copy()
            activityLabels = metaActivityLabels[metadataInd].copy()
            activityNames = metaActivityNames[metadataInd].copy()
            featureNames = metaFeatureNames[metadataInd].copy()
            metaDatasetName = metaDatasetNames[metadataInd]

            # Assert the assumptions made about the incoming data
            assert surveyAnswersList.min().item() >= -2, "All ratings must be greater than 0 (exception for -2, which is reserved for missing)."
            assert -1 not in surveyAnswersList, print("surveyAnswersList should contain ratings from 0 to n", flush=True)

            # ---------------------- Data Preparation ---------------------- #

            # Add the human activity recognition to the end.
            activityLabels = np.asarray(activityLabels, dtype=int).reshape(-1, 1)
            surveyAnswersList = np.hstack((surveyAnswersList, activityLabels))
            activityLabelInd = -1

            # Remove any experiments and signals that are bad.
            selectedSignalData = self._selectSignals(allAlignedFeatureIntervals, None)  # Cull bad signals (if any) from the data.
            allSignalData, allFeatureLabels, allSubjectInds = self._removeBadExperiments(selectedSignalData, surveyAnswersList, subjectOrder)
            allSignalData, featureNames = self._removeBadSignals(allSignalData, featureNames)
            if len(allSignalData) == 0: continue
            # allSignalData dimension: batchSize, numSignals, sequenceLength
            # allFeatureLabels dimension: batchSize, numLabels

            # Organize the feature labels and identify any missing labels.
            allFeatureLabels, allSingleClassIndices = self.organizeLabels(allFeatureLabels, metaTraining, metaDatasetName, len(allSignalData[0]))

            # Compile dataset-specific information.
            signalInds = np.arange(0, len(allSignalData[0]))
            numSubjects = max(allSubjectInds) + 1

            # Organize the signal indices and demographic information.
            allFeatureData = self.organizeSignals(allSignalData, signalInds)
            currentFeatureNames = np.asarray(featureNames)[signalInds]

            # Compile basic information about the data.
            numExperiments, numSignals, totalLength = allFeatureData.size()
            allExperimentalIndices = np.arange(0, numExperiments)
            sequenceLength = totalLength - self.numSecondsShift
            assert sequenceLength == self.maxSeqLength

            # ---------------------- Test/Train Split ---------------------- #

            # Initiate a mask for distinguishing between training and testing data.
            currentTestingMask = torch.full(allFeatureLabels.shape, False, dtype=torch.bool)
            currentTrainingMask = torch.full(allFeatureLabels.shape, False, dtype=torch.bool)

            # For each type of label recorded during the trial.
            for labelTypeInd in range(allFeatureLabels.shape[1]):
                currentFeatureLabels = copy.deepcopy(allFeatureLabels[:, labelTypeInd])

                # Temporarily remove the single classes.
                singleClassIndices = allSingleClassIndices[labelTypeInd]
                currentFeatureLabels[singleClassIndices] = self.missingLabelValue
                # Remove the missing data from the arrays.
                missingClassMask = torch.isnan(currentFeatureLabels)
                currentIndices = np.delete(allExperimentalIndices.copy(), torch.nonzero(missingClassMask))

                if len(currentIndices) != 0:
                    # print((~missingClassMask).sum() , testSplitRatio*len(currentIndices))
                    # if (~missingClassMask).sum() > testSplitRatio*len(currentIndices):
                    #     continue
                    try:
                        stratifyBy = currentFeatureLabels[~missingClassMask]
                        # Randomly split the data and labels, keeping a balance between testing/training.
                        Training_Indices, Testing_Indices = train_test_split(currentIndices, test_size=testSplitRatio, shuffle=True,
                                                                             stratify=stratifyBy, random_state=random_state)
                    except Exception as e:
                        continue
                    # Populate the training and testing mask                        
                    currentTestingMask[:, labelTypeInd][Testing_Indices] = True
                    currentTrainingMask[:, labelTypeInd][Training_Indices] = True
                    # Add in the single class values
                    currentTrainingMask[:, labelTypeInd][singleClassIndices] = True

            # ---------------------- Data Adjustments ---------------------- #

            # Remove any unused activity labels/names.
            goodActivityMask = (currentTestingMask[:, activityLabelInd]) | (currentTrainingMask[:, activityLabelInd])
            activityNames, allFeatureLabels[:, activityLabelInd][goodActivityMask] \
                = self.organizeActivityLabels(activityNames, allFeatureLabels[:, activityLabelInd][goodActivityMask])  # Expects inputs/outputs from 0 to n-1
            allFeatureLabels[~goodActivityMask] = -1  # Remove any unused activity indices (as the good indices were rehashed)

            # Data augmentation to increase the variations of datapoints.
            if metaDatasetName.lower() not in self.dontShiftDatasets:
                augmentedFeatureData, augmentedFeatureLabels, augmentedTrainingMask, augmentedTestingMask, augmentedSubjectInds \
                    = self.addShiftedSignals(allFeatureData, allFeatureLabels, currentTrainingMask, currentTestingMask, allSubjectInds)
            else:
                allFeatureData = self.dataInterface.getRecentSignalPoints(allFeatureData, self.maxSeqLength + self.numSecondsShift)

                augmentedFeatureData, augmentedFeatureLabels, augmentedTrainingMask, augmentedTestingMask, augmentedSubjectInds \
                    = self.dataInterface.getRecentSignalPoints(allFeatureData, self.maxSeqLength), allFeatureLabels, currentTrainingMask, currentTestingMask, allSubjectInds

            # Add the demographic information.
            augmentedFeatureData, numSubjectIdentifiers, demographicLength = self.addDemographicInfo(augmentedFeatureData, augmentedSubjectInds, metadataInd)

            # ---------------------- Create the Model ---------------------- #

            # Get the model parameters
            batch_size = self.getBatchSize(submodel, metaDatasetName)

            # Organize the training data into the expected pytorch format.
            pytorchDataClass = pytorchDataInterface(batch_size=batch_size, num_workers=0, shuffle=True, accelerator=self.accelerator)
            modelDataLoader = pytorchDataClass.getDataLoader(augmentedFeatureData, augmentedFeatureLabels, augmentedTrainingMask, augmentedTestingMask)

            # Initialize and train the model class.
            modelPipeline = emotionPipeline(accelerator=self.accelerator, modelID=metadataInd, datasetName=metaDatasetName, modelName=modelName, allEmotionClasses=numQuestionOptions.copy(),
                                            sequenceLength=sequenceLength, maxNumSignals=numSignals, numSubjectIdentifiers=numSubjectIdentifiers, demographicLength=demographicLength, numSubjects=numSubjects,
                                            userInputParams=self.userInputParams, emotionNames=surveyQuestions, activityNames=activityNames, featureNames=currentFeatureNames, submodel=submodel, fullTest=fullTest, metaTraining=metaTraining)

            # Hugging face integration.
            modelDataLoader = modelPipeline.acceleratorInterface(modelDataLoader)

            trainingInformation = modelPipeline.getDistributedModels(model=None, submodel="trainingInformation")
            trainingInformation.addSubmodel(submodel)

            # Store the information.
            allDataLoaders.append(modelDataLoader)
            allModelPipelines.append(modelPipeline)

        # Load in the previous model weights and attributes.
        self.modelMigration.loadModels(allModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True, loadModelWeights=True)

        # Prepare loss data holders.
        for modelInd in range(len(allModelPipelines)):
            dataLoader = allDataLoaders[modelInd]

            # Organize the training data into the expected pytorch format.
            numExperiments, numSignals, signalDimension = dataLoader.dataset.getSignalInfo()
            pytorchDataClass = pytorchDataInterface(batch_size=self.getMaxBatchSize(submodel, numSignals), num_workers=0, shuffle=False, accelerator=self.accelerator)
            modelDataLoader = pytorchDataClass.getDataLoader(*dataLoader.dataset.getAll())

            # Store the information.
            lossDataHolders.append(modelDataLoader)

        return allModelPipelines, allDataLoaders, lossDataHolders

    def onlyPreloadModelAttributes(self, modelName, datasetNames, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, allDummyModelPipelines=[]):
        # Initialize relevant holders.
        userInputParams = {'deviceListed': "cpu", 'submodel': 'emotionPrediction', 'numExpandedSignals': 2, 'numEncodingLayers': 4, 'numLiftedChannels': 4,
                           'compressionFactor': 1.5, 'expansionFactor': 1.5, 'numInterpreterHeads': 4, 'numBasicEmotions': 8, 'sequenceLength': 240}

        # Overwrite the location of the saved models.
        self.modelMigration.replaceFinalModelFolder("_finalModels/storedModels/")

        if len(allDummyModelPipelines) == 0:
            print(f"\nInitializing the models for {modelName}:", flush=True)
            # For each model we want.
            for metadataInd in range(len(datasetNames)):
                datasetName = datasetNames[metadataInd]

                # Initialize and train the model class.
                dummyModelPipeline = emotionPipeline(accelerator=self.accelerator, modelID=metadataInd, datasetName=datasetName, modelName=modelName, allEmotionClasses=[],
                                                     sequenceLength=240, maxNumSignals=500, numSubjectIdentifiers=1, demographicLength=0, numSubjects=1,
                                                     userInputParams=userInputParams, emotionNames=[], activityNames=[], featureNames=[], submodel=loadSubmodel, fullTest=True, metaTraining=False)
                # Hugging face integration.
                dummyModelPipeline.acceleratorInterface()

                # Store the information.
                allDummyModelPipelines.append(dummyModelPipeline)

        # Load in the model attributes.
        self.modelMigration.loadModels(allDummyModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True, loadModelWeights=False)

        return allDummyModelPipelines

    # ---------------------------------------------------------------------- #
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

        # If its our data.
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
        Purpose: Create the final signal array (of ccorrect length).
        --------------------------------------------
        allSignalData : A list of size (batchSize, numSignals, sequenceLength)
        signalInds : A list of size (numSignalsCombine,)
        """
        featureData = [];
        # Compile the feature's data across all experiments.
        for experimentInd in range(len(allSignalData)):
            signalData = np.asarray(allSignalData[experimentInd])
            data = signalData[signalInds, :]

            # Assertions about data integrity.
            numSignals, sequenceLength = data.shape
            assert self.minSeqLength <= sequenceLength, f"Expected {self.minSeqLength}, but recieved {data.shape[1]} "

            # Standardize the signals
            standardizeClass = standardizeData(data, axisDimension=1, threshold=0)
            data = standardizeClass.standardize(data)
            # Get the standardization information
            dataMeans, dataSTDs = standardizeClass.getStatistics()

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
        allSubjectInds : A 1D numpy array of size (batchSize,)
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
        allSubjectInds : A 1D numpy array of size (batchSize,)
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
        featureNames : A numpy array of size (numSignals,)
        """
        # Initialize data holders.
        goodFeatureInds = set(np.arange(0, len(featureNames)))
        assert len(allSignalData) == 0 or len(featureNames) == len(allSignalData[0]), allSignalData

        # For each set of signals.
        for dataInd in range(len(allSignalData)):
            signalData = np.asarray(allSignalData[dataInd])

            # Calculate time-series statistics for each signal.
            signalStandardDeviations = np.std(signalData, axis=1)

            # Remove any feature that doesnt varies enough.
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
        featureInds : A list of size (numSignalsKeep,)
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
