import os
import scipy
import numpy as np
import pandas as pd
from natsort import natsorted

# Import excel data interface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.globalMetaAnalysis import globalMetaAnalysis


class amigosInterface(globalMetaAnalysis):

    def __init__(self):
        # Specify the metadata file locations.
        self.subjectFolders = os.path.normpath(os.path.dirname(__file__) + "/../../../_experimentalData/_metaDatasets/AMIGOS/") + "/"

        # Initialize AMIGOS survey information.
        self.dimQuestions = ['arousal', 'valence', 'dominance', 'liking', 'familiarity']  # Rated 1 to 9
        self.emotionQuestions = ['neutral', 'disgust', 'happiness', 'surprise', 'anger', 'fear', 'sadness']  # Rated 0 to 1
        # Compile all survey questions.
        self.surveyQuestions = [question.lower() for question in self.dimQuestions]
        self.surveyQuestions.extend([question.lower() for question in self.emotionQuestions])
        # Compile the scoring ranges
        self.numQuestionOptions = [9] * len(self.dimQuestions)  # NOTE: this is a float from 1 to 9
        self.numQuestionOptions.extend([2] * len(self.emotionQuestions))  # NOTE: this is binary 0, 1 (which is why I add +1 to match others)
        # Specify the current dataset.
        self.datasetName = "amigos"

        # Define AMIGOS-specific parameters
        self.samplingFreq_processedData = 128

        # EEG Processing Notes:
        #       The data was processed at 128Hz.
        #       The data was averaged to the common reference.
        #       A bandpass frequency filter from 4.0-45.0Hz was applied.
        #       The trials were reordered from presentation order to video number (See video_list) order.
        self.streamingOrder = ["AF3", 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

        # GSR/ECG Processing Notes:
        #       The data was downsampled to 128Hz.
        #       ECG was low-pass filtered with 60Hz cut-off frequency.
        #       GSR was re-encoded to obtain skin conductance, and then GSR was calculated and low-pass filtered with 60Hz cut-off frequency.
        #       The trials were reordered from presentation order to video number (See video_list) order.
        self.streamingOrder.extend(['ECG Right', 'ECG Left', 'GSR'])

        # Specify which ones we are keeping
        self.streamingOrder_keeping = ["AF3", 'F7', 'F3', 'F4', 'F8', 'AF4', 'ECG Right', 'ECG Left', 'GSR']

        # Initialize the global meta protocol.
        super().__init__(self.subjectFolders, self.surveyQuestions)  # Initialize meta analysis.

    def getData(self):
        # Initialize data holders.
        allExperimentalTimesAmigos = []
        allExperimentalNamesAmigos = []
        allSurveyAnswerTimesAmigos = []
        allSurveyAnswersListAmigos = []
        allContextualInfoAmigos = []
        allCompiledDatasAmigos = []
        subjectOrderAmigos = []

        # Specify the metadata file locations.
        demographicFile = self.subjectFolders + "Metadata_xlsx/Participant_Questionnaires.xlsx"
        videoInfoFile = self.subjectFolders + "Metadata_xlsx/Experiment_Data.xlsx"

        # Extract all the metadata information.
        subjectDemographicInfo = pd.read_excel(demographicFile)
        # Extract specific demographic information's
        subjectIds = np.array(subjectDemographicInfo['UserID'].tolist())
        subjectAges = np.array(subjectDemographicInfo['Age'].tolist())
        subjectGenders = np.array(subjectDemographicInfo['Gender'].tolist())

        # Read in the video information file with experiment order
        shortVideoInfo = pd.read_excel(videoInfoFile, sheet_name='Short_Videos_Order')
        longVideoInfo = pd.read_excel(videoInfoFile, sheet_name='Long_Videos_Order')

        # Get the short video numbers (experiment names).
        shortVideoNumberHeaders = [col for col in shortVideoInfo.columns if 'Video_Number' in col]
        shortVideoNumbers = shortVideoInfo[shortVideoNumberHeaders].to_numpy()
        assert shortVideoNumbers.shape == (40, 16), "There are two video_number columns in the short video info file. Please ensure there is only one. This is an error with AMIGOS"
        # Get the long video numbers (experiment names).
        longVideoNumberHeaders = [col for col in longVideoInfo.columns if 'Video_Number' in col]
        longVideoNumbers_Grouped = longVideoInfo[longVideoNumberHeaders].to_numpy()
        userIds_forVideos = longVideoInfo["UserID(s)"]
        # Finalize the long video numbers (experiment names).
        longVideoNumbers = np.zeros((40, 4)) - 1
        for groupInd in range(len(userIds_forVideos)):
            userIds_forVideo = str(userIds_forVideos[groupInd]).split(",")

            for userId_forVideo in userIds_forVideo:
                longVideoNumbers[int(userId_forVideo) - 1] = longVideoNumbers_Grouped[groupInd]
        # Concatenate the video information together.
        videoNumbers = np.concatenate((shortVideoNumbers, longVideoNumbers), axis=1)

        # Remove the subjects with missing data.
        missingLongVideoInds = np.where(longVideoNumbers[:, 0] == -1)[0]

        keepingSensorInds = []
        for sensorInd in range(len(self.streamingOrder)):
            if self.streamingOrder[sensorInd] in self.streamingOrder_keeping:
                keepingSensorInds.append(sensorInd)

        if not os.path.exists(self.subjectFolders + "allProcessedData/"):
            raise ValueError(f"The processed data folder does not exist in {self.subjectFolders + 'allProcessedData/'}." +
                             f"Please ensure you have permission and downloaded the dataset from the AMIGOS website. " +
                             f"Then place the data in the correct folder.")

        subjectCounter = 0
        # For each subject in the training folder.
        for subjectFolderName in natsorted(os.listdir(self.subjectFolders + "allProcessedData/")):
            if not subjectFolderName.startswith("Data_Preprocessed_P"): continue
            print("\nExtracting data from AMIGOS subject:", subjectFolderName)

            # General file information
            subjectFolder = self.subjectFolders + "allProcessedData/" + subjectFolderName + "/"
            subjectFilename = subjectFolder + f"{subjectFolderName}.mat"
            subjectId = int(subjectFolderName.split("Data_Preprocessed_P")[1])
            assert subjectId == subjectIds[subjectCounter]

            # Extract the processed data file.
            subjectData = scipy.io.loadmat(subjectFilename)
            assert subjectData['VideoIDs'].shape == (1, 20)
            assert subjectData['joined_data'].shape == (1, 20)
            assert subjectData['labels_ext_annotation'].shape == (1, 20)
            assert subjectData['labels_selfassessment'].shape == (1, 20)
            # Remove bad subjects
            if subjectId - 1 in missingLongVideoInds:
                subjectCounter += 1
                continue

            # For each video shown to the participant.
            for videoInd in range(subjectData['joined_data'].shape[1]):
                surveyInfo = subjectData['labels_selfassessment'][0][videoInd]
                subjectVideoData = subjectData['joined_data'][0][videoInd]
                videoIds = subjectData['VideoIDs'][0][videoInd]  # NOTE: NOT IN ORDER
                # Remove ECG data
                subjectVideoData = subjectVideoData[:, keepingSensorInds]

                # Dont save the data if none collected.
                if subjectVideoData.shape == (0, 0): continue
                if np.isnan(subjectVideoData).sum() != 0: continue

                # Assert data integrity
                assert videoIds.shape == (1,), videoIds.shape
                assert surveyInfo.shape == (1, 12) or surveyInfo.shape == (1, 0), surveyInfo.shape
                assert subjectVideoData.shape[1] == len(self.streamingOrder_keeping), subjectVideoData.shape  # Dim: numPoints, numSignals

                # Convert to Sievert
                subjectVideoData[:, -1] = 1 / subjectVideoData[:, -1]  # EDA is the last signal!

                # Calculate the time axis of the data.
                numTotalPoints = subjectVideoData.shape[0]
                timepoints = self.universalMethods.getEvenlySampledArray(self.samplingFreq_processedData, numTotalPoints)

                # Compile the data into the expected format.
                assert timepoints.shape[0] == subjectVideoData.shape[0]
                compiledData = [[timepoints, subjectVideoData.T]]  # Dim: [numTimePoints, (numSignals, numTimePoints)]

                # Compile the experiment information.
                currentSurveyAnswersList = [[None] * 12] if surveyInfo.shape == (1, 0) else [surveyInfo[0]]
                experimentTimes = [[timepoints[0], timepoints[-1]]]
                currentSurveyAnswerTimes = [timepoints[-1]]
                experimentNames = [str(videoNumbers[subjectId - 1][videoInd])]
                assert "-1" not in experimentNames

                # Compile contextual information.
                subjectAge = subjectAges[subjectCounter]
                subjectGender = subjectGenders[subjectCounter]
                subjectGender = 0 if subjectGender[0] == 'f' else 1
                subjectContext = [["Age", "Gender"], [subjectAge, subjectGender]]

                # Account for emotion questions starting at 0.
                if surveyInfo.shape != (1, 0):
                    currentSurveyAnswersList[0][len(self.dimQuestions):] += 1

                # SubjectName
                SubjectName = subjectFolderName.split("Data_Preprocessed_")[1] + f"_trail{videoInd}"

                # Organize the data collected.
                subjectOrderAmigos.append(SubjectName)
                allCompiledDatasAmigos.append(compiledData)
                allContextualInfoAmigos.append(subjectContext)
                allExperimentalTimesAmigos.append(experimentTimes)
                allExperimentalNamesAmigos.append(experimentNames)
                allSurveyAnswerTimesAmigos.append(currentSurveyAnswerTimes)
                allSurveyAnswersListAmigos.append(currentSurveyAnswersList)

            subjectCounter += 1

        print("\tFinished data extraction")
        # Convert to numpy arrays
        subjectOrderAmigos = np.array(subjectOrderAmigos)
        allContextualInfoAmigos = np.array(allContextualInfoAmigos)
        allExperimentalTimesAmigos = np.array(allExperimentalTimesAmigos)
        allExperimentalNamesAmigos = np.array(allExperimentalNamesAmigos)
        allSurveyAnswerTimesAmigos = np.array(allSurveyAnswerTimesAmigos)
        allSurveyAnswersListAmigos = np.array(allSurveyAnswersListAmigos)

        return allCompiledDatasAmigos, subjectOrderAmigos, allExperimentalTimesAmigos, allExperimentalNamesAmigos, allSurveyAnswerTimesAmigos, allSurveyAnswersListAmigos, allContextualInfoAmigos

    def extractExperimentLabels(self, allExperimentalNamesAmigos):
        allExperimentalNamesAmigos = allExperimentalNamesAmigos.copy()

        # Read in and find the video names.
        videoNamesFile = self.subjectFolders + "Metadata_xlsx/Video_List.xlsx"
        videoNamesData = pd.read_excel(videoNamesFile, sheet_name='Video_List')
        videoNames = videoNamesData["Source_Movie"].to_numpy()

        return videoNames, (np.asarray(allExperimentalNamesAmigos, dtype=float) - 1).astype(int)

    def getStreamingInfo(self):
        # Specify EEG sensors.
        numEEGSensors = len(self.streamingOrder_keeping) - 3  # Remove ECG and GSR
        streamingOrderAmigos = ['eeg'] * numEEGSensors
        biomarkerOrderAmigos = ['eeg'] * numEEGSensors
        filteringOrdersAmigos = [[None, None]] * numEEGSensors  # Sampling Freq: 128 (Hz); Need 1/2 frequency at max.
        featureAverageWindowsAmigos = [30] * numEEGSensors  # ["EEG"]

        # Specify other sensors.
        streamingOrderAmigos.extend(['ecg', 'ecg', 'eda'])
        biomarkerOrderAmigos.extend(['ecg', 'ecg', 'eda'])
        filteringOrdersAmigos.extend([[None, None], [None, None], [None, None]])  # Sampling Freq: 128 (Hz); Need 1/2 frequency at max.
        featureAverageWindowsAmigos.extend([30, 30, 30])  # ["ECG", 'ECG', 'EDA']

        return streamingOrderAmigos, biomarkerOrderAmigos, featureAverageWindowsAmigos, filteringOrdersAmigos

    def compileTrainingInfo(self):
        # Compile the data: specific to the device worn.
        streamingOrderAmigos, biomarkerOrderAmigos, featureAverageWindowsAmigos, filteringOrderAmigos = self.getStreamingInfo()
        featureNamesAmigos, biomarkerFeatureNamesAmigos, biomarkerOrderAmigos = self.compileFeatureNames.extractFeatureNames(biomarkerOrderAmigos)

        return streamingOrderAmigos, biomarkerOrderAmigos, featureAverageWindowsAmigos, biomarkerFeatureNamesAmigos


if __name__ == "__main__":
    # Initialize metadata analysis class.
    amigosAnalysisClass = amigosInterface()

    analyzingData = True
    trainingData = False

    if analyzingData:
        # Extract the metadata
        allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, \
            allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo = amigosAnalysisClass.getData()
        # Compile the data: specific to the device worn.
        streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders = amigosAnalysisClass.getStreamingInfo()
        # Analyze and save the metadata features
        amigosAnalysisClass.extractFeatures(allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo,
                                            streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders, interfaceType='amigos', reanalyzeData=True, showPlots=True)

    if trainingData:
        # Prepare the data to go through the training interface.
        streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames = amigosAnalysisClass.compileTrainingInfo()

        plotTrainingData = False
        # Collected the training data.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, subjectOrder, \
            experimentOrder, activityNames, activityLabels, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
            = amigosAnalysisClass.trainingProtocolInterface(streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData, metaTraining=True)
