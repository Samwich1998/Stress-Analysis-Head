import os
import numpy as np
import pandas as pd
import scipy
from natsort import natsorted
import matplotlib.pyplot as plt

# Import excel data interface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.globalMetaAnalysis import globalMetaAnalysis


class caseInterface(globalMetaAnalysis):

    def __init__(self):
        self.debug = False
        # Specify the metadata file locations.
        self.subjectFolders = os.path.dirname(__file__) + "/../../../_experimentalData/_metaDatasets/CASE/data/interpolated/physiological/"
        self.subjectFoldersAnnotations = os.path.dirname(__file__) + "/../../../_experimentalData/_metaDatasets/CASE/data/interpolated/annotations/"
        self.subjectFoldersMetaData = os.path.dirname(__file__) + "/../../../_experimentalData/_metaDatasets/CASE/metadata/"
        # Initialize CASE survey information.
        self.surveyQuestions = ["valence", "arousal"]
        self.demographicsQuestions = ['Age', 'Gender']
        # Compile the scoring ranges
        self.numQuestionOptions = [10] * len(self.surveyQuestions)  # 0.5 through 9.5
        # Specify the current dataset.
        self.datasetName = "case"

        # Define CASE-specific parameters
        self.case_streamingOrder = ['ecg', 'bvp', 'gsr', 'rsp', 'skt']  # 'ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap'
        self.case_streamingConversions = [10 ** -3, 1, 10 ** -6, 1, 1]

        df_videos = pd.read_excel(self.subjectFoldersMetaData + 'videos.xlsx', engine='openpyxl')
        self.videoIDtoName = {row['Video-ID']: row['Source (Year)'] + ', ' + row['Video-label'] for index, row in df_videos.dropna(subset=['Video-ID']).iterrows()}
        self.activityNames = ['amusing', 'blue screen', 'boring', 'relaxed', 'scary', 'startVid']
        self.trialDuration = 5 * 60  # 5-minute sliding window per trial in experiment
        self.trialShift = 30  # 30-second sliding window shift
        self.videoStartBuffer = 60  # 60 seconds for emotional response from video
        self.surveyAggregation = 0.5  # emotion responses are averaged over the last 3 seconds of trial
        # Initialize the global meta protocol.
        super().__init__(self.subjectFolders, self.surveyQuestions)  # Initialize meta-analysis.

    @staticmethod
    def cleanFileData(synchronizedData, emotionAnnotations):
        synchronizedData['daqtime'] = synchronizedData['daqtime'] / 1000
        emotionAnnotations['jstime'] = emotionAnnotations['jstime'] / 1000

        return synchronizedData, emotionAnnotations

    def aggregateSurveysPerTrial(self, surveys):
        # surveys are pandas df only containing the responses per given trial
        return [scipy.stats.trim_mean(surveys[question], proportiontocut=0.1) for question in self.surveyQuestions]

    def getData(self, showPlots=False):
        # Initialize data holders.
        givenExperimentalTimes = []
        givenExperimentalNames = []
        givenSurveyAnswerTimes = []
        givenSurveyAnswersList = []
        givenSynchronizedData = []
        givenSubjectOrder = []
        givenContextualInfo = []

        df_metadata = pd.read_excel(self.subjectFoldersMetaData + 'participants.xlsx', engine='openpyxl')

        # For each subject in the training folder.
        for subjectSignalDataName in natsorted(os.listdir(self.subjectFolders)):
            # Do not analyze hidden files.
            if subjectSignalDataName.startswith((".", "~")): continue

            # Only look at subject folders, while ignoring other analysis folders.
            if os.path.exists(self.subjectFolders + subjectSignalDataName) and subjectSignalDataName.endswith('.csv'):
                subjectName = subjectSignalDataName.split(".")[0]
                # if subjectName not in ['sub_1']: continue
                print("\nExtracting data from CASE subject:", subjectName)

                # Get Contextual Info -----------------------------------------

                subjectInfo = df_metadata.iloc[int(subjectName.split("_")[-1]) - 1]
                givenContextualInfo.append([self.demographicsQuestions, [subjectInfo['Age-Group'], subjectInfo['Sex']]])
                if self.debug:
                    print(givenContextualInfo)

                # -------------------------------------------------------------

                df_subjectSynchronizedData = pd.read_csv(self.subjectFolders + subjectSignalDataName)
                df_subjectEmotionAnnotations = pd.read_csv(self.subjectFoldersAnnotations + subjectSignalDataName)

                df_subjectSynchronizedData, df_subjectEmotionAnnotations = self.cleanFileData(df_subjectSynchronizedData, df_subjectEmotionAnnotations)

                if self.debug:
                    print(df_subjectSynchronizedData.head())
                    print(df_subjectEmotionAnnotations.head())

                unique_videos = set(df_subjectSynchronizedData['video'].unique())
                assert unique_videos == set(df_subjectEmotionAnnotations['video'].unique())

                videoBufferTimestamps = {}
                for video in unique_videos:
                    videoStartTime = df_subjectSynchronizedData[df_subjectSynchronizedData['video'] == video].iloc[0]['daqtime']
                    videoBufferTimestamps[video] = [videoStartTime, videoStartTime + self.videoStartBuffer]

                # Get experiments ---------------------------------------------

                experimentTimes = []
                experimentNames = []
                trialKeep = []
                for i in range((int(df_subjectEmotionAnnotations['jstime'].max()) + 1 - self.trialDuration) // self.trialShift):
                    # For each Trial:
                    trialStart = i * self.trialShift
                    trialEnd = trialStart + self.trialDuration
                    df_currTrialEmotionAnnotations = df_subjectEmotionAnnotations[(df_subjectEmotionAnnotations['jstime'] > trialStart) &
                                                                                  (df_subjectEmotionAnnotations['jstime'] < trialEnd)]
                    trialVideos = df_currTrialEmotionAnnotations['video'].unique()
                    if trialEnd - videoBufferTimestamps[trialVideos[-1]][0] > self.videoStartBuffer:
                        # do not consider trial if survey response time (trial end) happens to soon after a video begins
                        experimentTimes.append([trialStart, trialEnd])
                        videoNames = [self.videoIDtoName[video] for video in trialVideos]
                        experimentNames.append('__'.join(videoNames))
                        trialKeep.append(i)
                    elif self.debug:
                        print('skipping', trialStart, trialEnd, videoBufferTimestamps[trialVideos[-1]])

                if showPlots:
                    experimentTimesPlotting = np.array(experimentTimes)
                    plt.vlines(trialKeep, experimentTimesPlotting[:, 0], experimentTimesPlotting[:, 1], label="Trials")
                    labelFlag = True
                    for timeWindow in videoBufferTimestamps.values():
                        if labelFlag:
                            plt.axhspan(timeWindow[0], timeWindow[1], color='grey', label='Video Start Buffer')
                        else:
                            plt.axhspan(timeWindow[0], timeWindow[1], color='grey')
                        labelFlag = False

                    plt.legend()
                    plt.xlabel('Trial #')
                    plt.ylabel('Time, seconds')
                    plt.title('Trial Time Visualization')
                    plt.show()

                print(len(trialKeep), 'of', (int(df_subjectEmotionAnnotations['jstime'].max()) + 1 - self.trialDuration) // self.trialShift, 'trials kept')

                # Get Valence + Arousal (Surveys) -----------------------------

                labelFlag = True
                currentSurveyAnswersList = []
                currentSurveyAnswerTimes = []
                for timeWindow in experimentTimes:
                    currentSurveyAnswerTimes.append(timeWindow[1])
                    # surveys aggregate the last few seconds of time window
                    df_currTrialEmotionAnnotations = df_subjectEmotionAnnotations[(df_subjectEmotionAnnotations['jstime'] >= timeWindow[1] - (self.surveyAggregation / 2)) &
                                                                                  (df_subjectEmotionAnnotations['jstime'] <= timeWindow[1] + (self.surveyAggregation / 2))]

                    trialValence, trialArousal = self.aggregateSurveysPerTrial(df_currTrialEmotionAnnotations)
                    currentSurveyAnswersList.append([trialValence + 0.5, trialArousal + 0.5])  # Account for the answers going from 0.5 to 9.5.
                    if showPlots:
                        if labelFlag:
                            plt.plot(df_currTrialEmotionAnnotations['jstime'] - df_currTrialEmotionAnnotations['jstime'].min(), df_currTrialEmotionAnnotations['valence'] - trialValence,
                                     marker='.', color='blue', label='Valence Error')
                            plt.plot(df_currTrialEmotionAnnotations['jstime'] - df_currTrialEmotionAnnotations['jstime'].min(), df_currTrialEmotionAnnotations['arousal'] - trialArousal,
                                     marker='.', color='orange', label='Arousal Error')
                        else:
                            plt.plot(df_currTrialEmotionAnnotations['jstime'] - df_currTrialEmotionAnnotations['jstime'].min(), df_currTrialEmotionAnnotations['valence'] - trialValence, color='blue', marker='.')
                            plt.plot(df_currTrialEmotionAnnotations['jstime'] - df_currTrialEmotionAnnotations['jstime'].min(), df_currTrialEmotionAnnotations['arousal'] - trialArousal, color='orange', marker='.')

                        labelFlag = False

                if showPlots:
                    plt.axvline(x=self.surveyAggregation / 2, color='black', linestyle='-', label='Trial End Time')
                    plt.legend()
                    plt.xlabel('Time from survey aggregation start, seconds')
                    plt.ylabel('Delta from Aggregated Value')
                    plt.title(f"Valence and Arousal vs. Aggregated Value Error t={self.surveyAggregation} sec")
                    plt.show()

                # Get Signal Data ---------------------------------------------

                timePoints = list(df_subjectSynchronizedData['daqtime'].dropna())
                experimentSignalData = []
                dataPerFreq = []
                for i, signalName in enumerate(self.case_streamingOrder):
                    signalData = list(df_subjectSynchronizedData[signalName].dropna() * self.case_streamingConversions[i])
                    assert len(timePoints) == len(signalData), f'Expected {len(timePoints)} number of datapoints. Got {len(signalData)}'
                    dataPerFreq.append(signalData)

                if showPlots:
                    for i, signalData in enumerate(dataPerFreq):
                        dataPlotting = signalData[20000:25000]
                        timePlotting = timePoints[20000:25000]
                        plt.plot(timePlotting, [(e - min(dataPlotting)) / (max(dataPlotting) - min(dataPlotting)) for e in dataPlotting],
                                 marker='.', linestyle='--', label=f'{self.case_streamingOrder[i]} [{min(dataPlotting)}, {max(dataPlotting)}]')
                    plt.xlabel('Time, seconds')
                    plt.ylabel('Value, Normalized')
                    plt.legend()
                    plt.title('All Signals')
                    plt.show()
                experimentSignalData.append([timePoints, dataPerFreq])

                # Organize the data collected ---------------------------------

                # Convert times to seconds: 
                experimentTimes = np.array(experimentTimes)
                currentSurveyAnswerTimes = np.array(currentSurveyAnswerTimes)

                givenSubjectOrder.append(subjectName)
                givenExperimentalTimes.append(experimentTimes)
                givenExperimentalNames.append(experimentNames)
                givenSynchronizedData.append(experimentSignalData)
                givenSurveyAnswerTimes.append(currentSurveyAnswerTimes)
                givenSurveyAnswersList.append(currentSurveyAnswersList)

                print("\tFinished data extraction")

                if self.debug:
                    print(self.extractExperimentLabels(givenExperimentalNames[0]))

                    print(np.array(givenSubjectOrder).shape)
                    print(np.array(givenExperimentalTimes).shape)
                    print(np.array(givenExperimentalNames).shape)
                    print(np.array(givenSurveyAnswerTimes).shape)
                    print(np.array(givenSurveyAnswersList).shape)

        return givenSynchronizedData, givenSubjectOrder, givenExperimentalTimes, givenExperimentalNames, givenSurveyAnswerTimes, givenSurveyAnswersList, givenContextualInfo

    def extractExperimentLabels(self, givenExperimentalNames):
        givenActivityNames = np.asarray(self.activityNames)

        givenActivityLabels = []
        # For each trial during the day.
        for trialActivityName in givenExperimentalNames:
            # Get the video information.
            videoType = trialActivityName.split('__')[-1]

            # Store the videos as a hashed index.
            activityHash = next((i for i, name in enumerate(givenActivityNames) if name in videoType))
            givenActivityLabels.append(activityHash)

        return givenActivityNames, givenActivityLabels

    @staticmethod
    def getStreamingInfo():
        # Feature information
        givenStreamingOrder = ['ecg', 'lowFreq', 'eda', 'lowFreq', 'temp']
        givenBiomarkerOrder = ['ecg', 'lowFreq', 'eda', 'lowFreq', 'temp']
        givenFilteringOrders = [[None, None], [None, 20], [None, 15], [None, 20], [None, 0.1]]  # Sampling Freq: 1000 (Hz); Need 1/2 frequency at max.
        givenFeatureAverageWindows = [30, 30, 30, 30, 30]  # ['ecg', 'bvp', 'gsr', 'rsp', 'skt']

        return givenStreamingOrder, givenBiomarkerOrder, givenFeatureAverageWindows, givenFilteringOrders

    def compileTrainingInfo(self):
        # Compile the data: specific to the device worn.
        givenStreamingOrder, givenBiomarkerOrder, givenFeatureAverageWindows, givenFilteringOrders = self.getStreamingInfo()
        featureNames, givenBiomarkerFeatureNames, givenBiomarkerOrder = self.compileFeatureNames.extractFeatureNames(givenBiomarkerOrder)

        return givenStreamingOrder, givenBiomarkerOrder, givenFeatureAverageWindows, givenBiomarkerFeatureNames


if __name__ == "__main__":
    # Initialize metadata analysis class.
    caseAnalysisClass = caseInterface()

    analyzingData = True
    trainingData = False

    if analyzingData:
        # Extract the metadata
        allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, \
            allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo = caseAnalysisClass.getData(showPlots=False)
        # Compile the data: specific to the device worn.
        streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders = caseAnalysisClass.getStreamingInfo()
        # # Analyze and save the metadata features
        caseAnalysisClass.extractFeatures(allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo,
                                          streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders, interfaceType='case', reanalyzeData=True, showPlots=False)

    if trainingData:
        # Prepare the data to go through the training interface.
        streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames = caseAnalysisClass.compileTrainingInfo()

        plotTrainingData = False
        # Collected the training data.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, subjectOrder, \
            experimentOrder, activityNames, activityLabels, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
            = caseAnalysisClass.trainingProtocolInterface(streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData, metaTraining=True)
