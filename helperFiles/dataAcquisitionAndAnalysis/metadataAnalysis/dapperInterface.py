import os
import numpy as np
import pandas as pd
from datetime import datetime
from natsort import natsorted
import matplotlib.pyplot as plt

# Import excel data interface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.globalMetaAnalysis import globalMetaAnalysis


class dapperInterface(globalMetaAnalysis):

    def __init__(self):
        # Specify the metadata file locations.
        self.debug = False
        self.subjectFolders = os.path.dirname(__file__) + "/../../../_experimentalData/_metaDatasets/DAPPER/"
        # Specify the current dataset.
        self.datasetName = "dapper"

        # Initialize DAPPER survey information.
        self.emotionQuestions = ['UPSET', 'HOSTILE', 'ALERT', 'ASHAMED', 'INSPIRED', 'NERVOUS', 'DETERMINED',
                                 'ATTENTIVE', 'AFRAID', 'ACTIVE']  # Rated 1 to 5
        self.dimQuestions = ['VALENCE', 'AROUSAL']  # Rated 1 to 5
        # # Compile all survey questions.
        self.surveyQuestions = [question.lower() for question in self.dimQuestions]
        self.surveyQuestions.extend([question.lower() for question in self.emotionQuestions])
        # # Compile the scoring ranges
        self.numQuestionOptions = [5] * len(self.surveyQuestions)
        self.trialDuration = 60 * 10  # duration of a trial, in seconds
        self.numPlaceOptions = {1: 'classroom', 2: 'library', 3: 'student dormitory',
                                4: 'playground', 5: 'gymnasium', 6: 'canteen', 7: 'department',
                                9: 'home', 10: 'internship'}  # omitting 'on-campus' and 'off-campus'
        self.numParticipatingPeopleOptions = {1: 'self', 2: 'teacher', 3: 'classmate', 4: 'family', 5: 'stranger'}
        self.numActivityTypeOptions = {1: 'major', 2: 'interest', 3: 'group', 4: 'personal'}

        # Sampling frequencies
        self.samplingFreq_GSR = 1
        self.samplingFreq_HR = 1

        # Initialize the global meta protocol.
        super().__init__(self.subjectFolders, self.surveyQuestions)  # Initialize meta analysis.

    @staticmethod
    def getSeconds(timestamp):
        time_format = '%Y%m%d%H%M%S'

        # Convert the time part to a datetime object (ignoring date information)
        time_obj = datetime.strptime(timestamp, time_format)
        # Get the total seconds from the time object
        total_seconds = time_obj.timestamp()

        return total_seconds

    def getActivityTypes(self, row):
        # given a survey response, get activities during the trial.
        place = self.numPlaceOptions.get(row['Place of the event'], "unknown")
        peopleIndex = str(row['Participating people']).replace(" ", "").split('|')
        peopleArr = []
        for i in peopleIndex:
            peopleArr.append(self.numParticipatingPeopleOptions.get(int(i), 'unknown'))
        peopleStr = '-'.join(peopleArr)
        activityType = self.numActivityTypeOptions.get(row['Activity type'], "unknown")

        return f'{place}_{peopleStr}_{activityType}'

    @staticmethod
    def cleanFileData(fileData, samplingFreq, timestampColumnName):
        def convert_to_timestamp(date_str):
            try:
                # Try to convert using the format "y/m/d h:m:s"
                return (pd.to_datetime(date_str, format='%Y/%m/%d %H:%M:%S')).timestamp()
            except ValueError:
                # If the above fails, try to convert using the format "y/m/d"
                print('date as y/m/d', date_str)
                return (pd.to_datetime(date_str + ' 00:00:00', format='%Y/%m/%d %H:%M:%S')).timestamp()

        fileData[timestampColumnName] = fileData[timestampColumnName].apply(convert_to_timestamp)
        timeOffset = fileData[timestampColumnName].min()
        # print(timeOffset)
        fileData['recompiledTime'] = fileData[timestampColumnName] - timeOffset
        if pd.isna(fileData['recompiledTime'].iloc[-1]):
            fileData.drop(fileData.index[-1], inplace=True)
        if len(fileData[fileData.isna().any(axis=1)]) != 0:
            print(fileData[fileData.isna().any(axis=1)])

        return fileData, timeOffset

    def recompileExperimentTimes(self, currExperimentTrialTimes, trialKeep, fileData, timeOffset, timestampColumnName='time'):
        def readjustTimeStamp(timestamp):
            row = fileData.loc[fileData[timestampColumnName] == timestamp]
            if len(row) == 0:  # Not in data time window?
                estimatedTime = int(timestamp - timeOffset)
                if estimatedTime > fileData['recompiledTime'].max() or estimatedTime < fileData['recompiledTime'].min():
                    return None
                return estimatedTime
            assert len(row) == 1, f"{timestamp}, {fileData[timestampColumnName].agg(['min', 'max'])}"
            return int(row['recompiledTime'].iloc[0])

        newTimes = []
        trialKeeps = []
        for i in trialKeep:
            unadjustedCurrTrialTimes = currExperimentTrialTimes[i]
            trialNewTimes = [readjustTimeStamp(unadjustedCurrTrialTimes[j]) for j in range(2)]
            if (len(trialNewTimes) != 2) or None in trialNewTimes:
                if self.debug:
                    print('skipping', trialNewTimes)
            else:
                newTimes.append([trialNewTimes])
                trialKeeps.append(i)
        return newTimes, trialKeeps

    def getData(self, showPlots=False):
        # Initialize data holders.
        allExperimentalTimes = []
        allExperimentalNames = []
        allSurveyAnswerTimes = []
        allSurveyAnswersList = []
        allCompiledDatas = []
        subjectOrder = []
        allContextualInfo = []

        # Information about what is in the DAPPER dataset.
        experimentSections = ['PANAS']
        surveyAnswersColumns = ['Valence', 'Arousal']

        for i in range(10):
            surveyAnswersColumns.append(f'PANAS_{i + 1}')

        # Information about the recorded signals.
        allExpectedSignalTypes = np.asarray(['GSR', 'Motion Intensity', 'PPG'])
        signalFreq = np.asarray([40, 1, 1])  # frequency of each signal type in Hz

        assert os.path.exists(self.subjectFolders + 'Psychol_Rec/ESM.xlsx'), "You need to download the DAPPER dataset and place in folder."
        ESMData = pd.read_excel(self.subjectFolders + 'Psychol_Rec/ESM.xlsx')
        ESMData[' StartTime '] = (pd.to_datetime(ESMData[' StartTime '])).map(pd.Timestamp.timestamp)

        # For each subject in the training folder.
        for subjectGroupFolderName in natsorted(os.listdir(self.subjectFolders)):
            subjectGroupFolder = self.subjectFolders + subjectGroupFolderName + "/"
            if not subjectGroupFolderName.startswith("Physiol_Rec"): continue

            # Only look at subject folders, while ignoring other analysis folders.
            for subjectFolderName in natsorted(os.listdir(subjectGroupFolder)):
                if subjectFolderName.startswith((".", "~", "README")): continue
                subjectFolder = subjectGroupFolder + subjectFolderName + "/"
                print("\nExtracting data from DAPPER subject:", subjectFolderName)

                # Get the survey start times for the subject
                subjectESM = ESMData[ESMData['Participant ID'] == int(subjectFolderName)]
                surveyStartTimes = subjectESM[' StartTime '].values

                experimentKeys = []
                for experimentDataFileName in natsorted(os.listdir(subjectFolder)):
                    if experimentDataFileName.count("_") == 1:  # get all experiment names
                        experimentKeys.append(experimentDataFileName[:-4])

                # dayInd = 0
                # experimentKeys = ['20191124092457_20191125220645']
                for experiment in experimentKeys:
                    # get all survey responses within experiment time
                    experimentTime = experiment.split('_')
                    assert len(experimentTime) == 2
                    start = self.getSeconds(experimentTime[0])
                    end = self.getSeconds(experimentTime[1])

                    experimentTrialEndTimes = [x for x in surveyStartTimes if start < x < end]

                    currExperimentTrialTimes = []
                    currExperimentTimes = []
                    currExperimentTrialNames = []
                    currExperimentSurveyTimes = []
                    currExperimentAnswersList = []

                    # get information by trial and combine by experiment
                    for trialEndTime in experimentTrialEndTimes:
                        # trial end time = survey start time
                        currExperimentSurveyTimes.append(trialEndTime)
                        # get trial start and end times
                        currExperimentTrialTimes.append([trialEndTime - self.trialDuration, trialEndTime])
                        currExperimentTimes.append([trialEndTime - (60 * 20), trialEndTime + (60 * 10)])
                        # get trial names
                        row = subjectESM[subjectESM[' StartTime '] == trialEndTime].iloc[0]
                        currExperimentSurveyAnswers = row[surveyAnswersColumns].values
                        trialActivityName = self.getActivityTypes(row)

                        currExperimentTrialNames.append(trialActivityName)
                        currExperimentAnswersList.append(currExperimentSurveyAnswers)

                    # get all files with same experiment key. Should be 1 Hz Heart Rate and GSR Data
                    experimentSignalData = []
                    # print(fileData['csv_time_GSR'].agg(['min', 'max']))

                    # 1 Hz for Heart Rate and GSR
                    assert os.path.exists(subjectFolder + experiment + '.csv')
                    fileData_1hz = pd.read_csv(subjectFolder + experiment + '.csv')
                    fileData_1hz, timeOffset = self.cleanFileData(fileData_1hz, 1, 'time')
                    fileData_1hz['GSR'] = fileData_1hz['GSR'] * (10 ** -6)
                    # print(fileData_1hz['GSR'].agg(['min', 'max']))
                    # timePoints_1hz = list(fileData_1hz['recompiledTime'])

                    # hrSignalData = fileData_1hz['heart_rate'].values
                    # experimentSignalData.append([timePoints_1hz, [hrSignalData]])

                    # GSRSignalData = fileData_1hz['GSR'].values
                    # experimentSignalData.append([timePoints_1hz, [GSRSignalData]])

                    # GSR data default = 6.10388818e-09. If this value occurs during trial, do not consider trial
                    # HR data default = 40. If this value occurs during trial, do not consider trial
                    trialKeep = []
                    for i, trialWindow in enumerate(currExperimentTrialTimes):

                        assert len(trialWindow) == 2
                        # GSR_trialData = fileData[(fileData['recompiledTime'] > trialWindow[0]) &
                        #                          (fileData['recompiledTime'] < trialWindow[1])]
                        experimentData = fileData_1hz[(fileData_1hz['recompiledTime'] > currExperimentTimes[i][0] - timeOffset) &
                                                      (fileData_1hz['recompiledTime'] < currExperimentTimes[i][1] - timeOffset)]

                        timePoints_1hz = list(experimentData['recompiledTime'])
                        GSRSignalData = experimentData['GSR'].values
                        hrSignalData = experimentData['heart_rate'].values
                        experimentSignalData.append([[timePoints_1hz, [GSRSignalData]], [timePoints_1hz, [hrSignalData]]])

                        trialData = fileData_1hz[(fileData_1hz['recompiledTime'] > trialWindow[0] - timeOffset) &
                                                 (fileData_1hz['recompiledTime'] < trialWindow[1] - timeOffset)]
                        if showPlots:
                            fig, ax1 = plt.subplots()

                            color = 'tab:green'
                            ax1.set_xlabel('time (s)')
                            ax1.set_ylabel('GSR, 1 Hz', color=color)
                            ax1.plot(experimentData['recompiledTime'], experimentData['GSR'], color=color)
                            ax1.tick_params(axis='y', labelcolor=color)

                            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                            color = 'tab:blue'
                            ax2.set_ylabel('Heart Rate, 1 Hz', color=color)  # we already handled the x-label with ax1
                            ax2.plot(experimentData['recompiledTime'], experimentData['heart_rate'], color=color)
                            ax2.tick_params(axis='y', labelcolor=color)

                            ax1.set_title(f'{experiment}, {subjectFolderName}')
                            fig.tight_layout()  # otherwise the right y-label is slightly clipped

                            plt.axvline(x=trialWindow[0] - timeOffset, color='black')
                            plt.axvline(x=trialWindow[1] - timeOffset, color='black')
                            plt.show()

                        if 0.0061038881767686 * (10 ** -6) >= trialData['GSR'].min() or len(trialData['GSR']) < (self.trialDuration * self.samplingFreq_GSR - 1):
                            if self.debug:
                                # print(0.0061038881767686 * (10**-6) in list(trialData['GSR']))
                                print('Bad GSR data. Pass Trial. \nNum Points vs Num Expected Points:', len(trialData['GSR']), self.trialDuration * self.samplingFreq_GSR - 1)

                        elif 40.0 >= trialData['heart_rate'].min() or len(trialData['heart_rate']) < (self.trialDuration * self.samplingFreq_HR - 1):
                            if self.debug:
                                # print(40.0 in list(trialData['heart_rate']))
                                print('Bad HR data. Pass Trial. \nNum Points vs Num Expected Points:', len(trialData['heart_rate']), self.trialDuration * self.samplingFreq_HR - 1)
                        else:
                            trialKeep.append(i)

                    cleanCurrExperimentTrialTimes, trialKeep = self.recompileExperimentTimes(currExperimentTrialTimes, trialKeep, fileData_1hz, timeOffset)
                    subjectOrder.extend([f'{subjectFolderName} {experiment} {i}' for i in trialKeep])
                    allExperimentalTimes.extend(cleanCurrExperimentTrialTimes)
                    allExperimentalNames.extend([[currExperimentTrialNames[i]] for i in trialKeep])
                    allSurveyAnswerTimes.extend([[e[0][1]] for e in cleanCurrExperimentTrialTimes])
                    allSurveyAnswersList.extend([[currExperimentAnswersList[i]] for i in trialKeep])
                    allCompiledDatas.extend([experimentSignalData[i] for i in trialKeep])
                    allContextualInfo.extend([[['Age', 'Gender'], [None, None]]] * len(trialKeep))

                    print(len(trialKeep), 'of', len(currExperimentTrialTimes), 'trials kept')

                # subjectOrder.extend([f'{subjectFolderName} {name}' for name in experimentKeys])
                # allExperimentalNames.extend(experimentNames)
                # allExperimentalTimes.extend(experimentTimes)
                # allCompiledDatas.extend(compiledData_eachFreq)
                # allSurveyAnswerTimes.extend(currentSurveyAnswerTimes)
                # allSurveyAnswersList.extend(currentSurveyAnswersList)

                # print(np.asarray(subjectOrder).shape)
                # print(np.asarray(allExperimentalNames).shape)
                # print(np.asarray(allSurveyAnswerTimes).shape)
                # print(np.asarray(allCompiledDatas).shape)
                # print(np.asarray(allSurveyAnswersList).shape)

                print("\tFinished data extraction")
                # print(subjectOrder)

        # Convert subject order to numpy array
        subjectOrder = np.asarray(subjectOrder)

        return allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo

    def extractExperimentLabels(self, allExperimentalNames):
        activityNames = np.asarray(list(self.numActivityTypeOptions.values()))

        activityLabels = []
        # For each trial during the day.
        for trialActivityName in allExperimentalNames:
            # Get the activity/person/place information.
            trialActivityPlace, trialActivityPerson, trialActivityType = trialActivityName.split("_")
            assert trialActivityType in activityNames, f"{trialActivityType} {activityNames}"

            # Store the activity as a hashed index.
            activityHash = np.where((activityNames == trialActivityType))[0][0]
            activityLabels.append(activityHash)

        return activityNames, activityLabels

    @staticmethod
    def getStreamingInfo():
        # Feature information
        streamingOrder = ['eda', 'lowFreq']
        biomarkerOrder = ['eda', 'lowFreq']
        filteringOrders = [[None, None], [None, None]]  # Sampling Freq: 1, 1 (Hz); Need 1/2 frequency at max.
        featureAverageWindows = [30, 30]  # ['EDA', 'Heart Rate']

        return streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders

    def compileTrainingInfo(self):
        # Compile the data: specific to the device worn.
        streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders = self.getStreamingInfo()
        featureNames, biomarkerFeatureNames, biomarkerOrder = self.compileFeatureNames.extractFeatureNames(biomarkerOrder)

        return streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames


if __name__ == "__main__":
    # Initialize metadata analysis class.
    dapperAnalysisClass = dapperInterface()

    analyzingData = True
    trainingData = False

    if analyzingData:
        # Extract the metadata
        dapperAllCompiledDatas, dapperSubjectOrder, dapperAllExperimentalTimes, dapperAllExperimentalNames, \
            dapperAllSurveyAnswerTimes, dapperAllSurveyAnswersList, dapperAllContextualInfo = dapperAnalysisClass.getData(showPlots=False)
        # Compile the data: specific to the device worn.
        dapperStreamingOrder, dapperBiomarkerOrder, dapperFeatureAverageWindows, dapperFilteringOrders = dapperAnalysisClass.getStreamingInfo()
        # Analyze and save the metadata features
        dapperAnalysisClass.extractFeatures(dapperAllCompiledDatas, dapperSubjectOrder, dapperAllExperimentalTimes, dapperAllExperimentalNames, dapperAllSurveyAnswerTimes, dapperAllSurveyAnswersList, dapperAllContextualInfo,
                                            dapperStreamingOrder, dapperBiomarkerOrder, dapperFeatureAverageWindows, dapperFilteringOrders, interfaceType='dapper', reanalyzeData=True, showPlots=False)

    if trainingData:
        # Prepare the data to go through the training interface.
        dapperStreamingOrder, dapperBiomarkerOrder, dapperFeatureAverageWindows, dapperBiomarkerFeatureNames = dapperAnalysisClass.compileTrainingInfo()

        plotTrainingData = False
        # Collected the training data.
        dapperAllRawFeatureTimesHolders, dapperAllRawFeatureHolders, dapperAllRawFeatureIntervals, dapperAllRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, dapperAllAlignedFeatureHolder, dapperAllAlignedFeatureIntervals, dapperAllAlignedFeatureIntervalTimes, dapperSubjectOrder, \
            experimentOrder, dapperActivityNames, dapperActivityLabels, dapperAllFinalFeatures, dapperAllFinalLabels, dapperFeatureLabelTypes, dapperSurveyQuestions, dapperSurveyAnswersList, dapperSurveyAnswerTimes \
            = dapperAnalysisClass.trainingProtocolInterface(dapperStreamingOrder, dapperBiomarkerOrder, dapperFeatureAverageWindows, dapperBiomarkerFeatureNames, plotTrainingData, metaTraining=True)
