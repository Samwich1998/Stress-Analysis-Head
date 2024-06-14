import os
import pickle
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt

# Import excel data interface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.globalMetaAnalysis import globalMetaAnalysis


class wesadInterface(globalMetaAnalysis):

    def __init__(self):
        # Specify the metadata file locations.
        self.subjectFolders = os.path.dirname(__file__) + "/../../../_experimentalData/_metaDatasets/WESAD/"

        # Initialize WESAD survey information.
        self.panasQuestions = ["Active", "Distressed", "Interested", "Inspired", "Annoyed", "Strong", "Guilty", "Scared",
                               "Hostile", "Excited", "Proud", "Irritable", "Enthusiastic", "Ashamed", "Alert", "Nervous",
                               "Determined", "Attentive", "Jittery", "Afraid", "Stressed", "Frustrated", "Happy", "Sad"]  # Rated 1 to 5
        self.panasQuestions_stressCondition = ["Active", "Distressed", "Interested", "Inspired", "Annoyed", "Strong", "Guilty", "Scared",
                                               "Hostile", "Excited", "Proud", "Irritable", "Enthusiastic", "Ashamed", "Alert", "Nervous",
                                               "Determined", "Attentive", "Jittery", "Afraid", "Stressed", "Frustrated", "Happy", "Angry", "Irritated", "Sad"]  # Rated 1 to 5
        self.staiQuestions = ["I feel at ease", "I feel nervous", "I am jittery", "I am relaxed", "I am worried", "I feel pleasant"]  # Rated 1 to 4
        self.dimQuestions = ["Valence", "Arousal"]  # Rated 1 to 9
        # Compile all survey questions. NOTE: Ignoring the extra 2 in the stress condition.
        self.surveyQuestions = self.panasQuestions.copy()
        self.surveyQuestions.extend(self.staiQuestions)
        self.surveyQuestions.extend(self.dimQuestions)
        # Compile the scoring ranges
        self.numQuestionOptions = [5] * len(self.panasQuestions)
        self.numQuestionOptions.extend([4] * len(self.staiQuestions))
        self.numQuestionOptions.extend([9] * len(self.dimQuestions))
        # Specify the current dataset.
        self.datasetName = "wesad"

        # Define WESAD-specific parameters
        self.empaticaEDA_Temp_samplingFreq = 4
        self.empaticaBVP_samplingFreq = 64
        self.empaticaACC_samplingFreq = 32
        self.respiBAN_samplingFreq = 700

        # Initialize the global meta protocol.
        super().__init__(self.subjectFolders, self.surveyQuestions)  # Initialize meta analysis.

    def getData(self):
        # Initialize data holders.
        allExperimentalTimes = []
        allExperimentalNames = []
        allSurveyAnswerTimes = []
        allSurveyAnswersList = []
        allSynchronizedData = []
        subjectOrder = []
        allContextualInfo = []

        # For each subject in the training folder.
        for subjectFolderName in natsorted(os.listdir(self.subjectFolders)):
            subjectFolder = self.subjectFolders + subjectFolderName + "/"
            # Only look at subject folders, while ignoring other analysis folders.
            if os.path.isdir(subjectFolder) and self.universalMethods.isNumber(subjectFolderName[1:]):
                print("\nExtracting data from WESAD subject:", subjectFolderName)
                labelsFile = subjectFolder + subjectFolderName + "_quest.csv"
                synchronizedDataFile = subjectFolder + subjectFolderName + ".pkl"
                demographicFile = subjectFolder + subjectFolderName + "_readme.txt"

                subjectDemographicInfo = []
                # Read in the demographic information line by line
                with open(demographicFile, "r") as file:
                    for line in file:
                        subjectDemographicInfo.append(line.rstrip())  # Remove newline character

                subjectAge = None
                subjectGender = None
                # Extract the subject demographic information
                for fileLine in subjectDemographicInfo:
                    if fileLine.startswith("Age: "):
                        subjectAge = int(fileLine.split("Age: ")[1])
                    if fileLine.startswith("Height (cm): "):
                        subjectHeight = int(fileLine.split("Height (cm): ")[1])
                    if fileLine.startswith("Weight (kg): "):
                        subjectWeight = int(fileLine.split("Weight (kg): ")[1])
                    if fileLine.startswith("Gender: "):
                        subjectGender = fileLine.split("Gender: ")[1]
                # Organize the contextual information.
                subjectGender = 0 if subjectGender[0] == 'f' else 1
                subjectContext = [["Age", "Gender"], [subjectAge, subjectGender]]

                # Load in synchronized data (compiled by researchers).
                with open(synchronizedDataFile, 'rb') as file:
                    synchronizedData = pickle.load(file, encoding='latin1')

                startTimes = []
                endTimes = []
                sssqAnswers = []
                dimAnswers = []
                panasAnswers = []
                staiAnswers = []
                experimentNames = None
                # Load in the label information.
                labelInfoRead = pd.read_csv(labelsFile).values
                # For each row in the CSV.
                for row in labelInfoRead:
                    row = row[0]

                    compiledNumbers = []
                    # Get all the numbers
                    for entry in row.split(";"):
                        if self.universalMethods.isNumber(entry):
                            compiledNumbers.append(float(entry))
                        elif entry == "Nan":
                            compiledNumbers.append(None)
                    # Remove the extra 2 from the PANAS stress.
                    if len(compiledNumbers) == 26:
                        sadAnswer = compiledNumbers.pop()
                        compiledNumbers.pop()
                        compiledNumbers.pop()
                        compiledNumbers.append(sadAnswer)

                    # Save the questionnaire data
                    if row.startswith("# PANAS"):
                        panasAnswers.append(compiledNumbers)
                    elif row.startswith("# STAI"):
                        staiAnswers.append(compiledNumbers)
                    elif row.startswith("# DIM"):
                        dimAnswers.append(compiledNumbers)
                    elif row.startswith("# SSSQ"):
                        sssqAnswers.append(compiledNumbers)
                    # Save the time data
                    elif row.startswith("# END"):
                        endTimes = compiledNumbers
                    elif row.startswith("# START"):
                        startTimes = compiledNumbers
                    # Save experiment names
                    elif row.startswith("# ORDER"):
                        experimentNames = row.split(";")[1:6]
                        assert len(experimentNames) == 5

                        # Assert that we have all the questionnaire data.
                assert len(panasAnswers) == len(staiAnswers) == len(dimAnswers)
                assert len(sssqAnswers) == 1
                # Assert that the first 5 are the sorted times.
                assert all(startTimes[i] <= startTimes[i + 1] for i in range(len(panasAnswers) - 1))
                assert all(endTimes[i] <= endTimes[i + 1] for i in range(len(panasAnswers) - 1))
                # Assert that the next time is the other times.
                assert startTimes[5] < startTimes[4]
                assert endTimes[5] < endTimes[4]

                # Prepare the data to be read in
                currentSurveyAnswersList = np.concatenate((panasAnswers, staiAnswers, dimAnswers), axis=1)
                currentSurveyAnswerTimes = np.array(endTimes[0:5]) * 60
                experimentTimes = [[startTimes[i] * 60, endTimes[i] * 60] for i in range(5)]

                # Organize the data collected.
                subjectOrder.append(subjectFolderName)
                allContextualInfo.append(subjectContext)
                allExperimentalTimes.append(experimentTimes)
                allExperimentalNames.append(experimentNames)
                allSynchronizedData.append(synchronizedData)
                allSurveyAnswerTimes.append(currentSurveyAnswerTimes)
                allSurveyAnswersList.append(currentSurveyAnswersList)
                print("\tFinished data extraction")

        # Convert to numpy arrays
        subjectOrder = np.array(subjectOrder)
        allContextualInfo = np.array(allContextualInfo)
        allExperimentalTimes = np.array(allExperimentalTimes)
        allExperimentalNames = np.array(allExperimentalNames)
        allSurveyAnswerTimes = np.array(allSurveyAnswerTimes)
        allSurveyAnswersList = np.array(allSurveyAnswersList)

        return allSynchronizedData, subjectOrder, allExperimentalTimes, allExperimentalNames, allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo

    @staticmethod
    def extractExperimentLabels(allExperimentalNames):
        allExperimentalNames = allExperimentalNames.copy()

        activityLabels = []
        activityNames = ["Base", "TSST", "Medi", "Fun"]  # Baseline, stress (public speaking + math), meditation (guided), amusement (videos)
        for experimentName in allExperimentalNames:
            experimentName = experimentName.split(" ")[0]

            labelInd = activityNames.index(experimentName)
            activityLabels.append(labelInd)

        return activityNames, activityLabels

    @staticmethod
    def getStreamingInfo():
        # respiBAN-specific information.
        streamingOrder = ['lowFreq', 'eda', 'temp', 'lowFreq']
        biomarkerFeatureOrder = ['lowFreq', 'eda', 'temp', 'lowFreq']
        filteringOrders = [[None, 150], [None, 15], [None, 0.1], [None, 20]]  # Sampling Freq: 700, 700, 700, 700, 700 (Hz); Need 1/2 frequency at max.
        featureAverageWindows = [30, 30, 30, 30]  # ['ECG', 'EDA', 'Temp', 'Resp']

        # Add empatica-specific information
        streamingOrder.extend(['lowFreq', 'eda', 'temp'])
        biomarkerFeatureOrder.extend(['lowFreq', 'eda', 'temp'])
        filteringOrders.extend([[None, 20], [None, None], [None, 0.1]])  # Sampling Freq: 64, 4, 4 (Hz); Need 1/2 frequency at max.
        featureAverageWindows.extend([30, 30, 30])  # ['BVP', 'EDA', 'Temp']

        return streamingOrder, biomarkerFeatureOrder, featureAverageWindows, filteringOrders

    def organizeSynchronizedData(self, synchronizedData, experimentTimes, showPlots=True):
        # Get the RespiBAN time points.
        respiBAN_numPoints = len(synchronizedData['signal']['chest']['EDA'])
        respiBAN_timePoints = self.universalMethods.getEvenlySampledArray(self.respiBAN_samplingFreq, respiBAN_numPoints)

        # Filter the data to keep only your signals.
        keys_to_keep = ['ECG', 'EDA', 'Temp', 'Resp']  # Options: ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
        filtered_dict = [synchronizedData['signal']['chest'][key] for key in keys_to_keep if key in synchronizedData['signal']['chest']]
        compiledSignalsRespiban = np.asarray(filtered_dict).squeeze()
        # Convert from uS to S for EDA.
        compiledSignalsRespiban[1] = compiledSignalsRespiban[1] * 1E-6

        # Get the Empatica number of points.
        empaticaACC_numPoints = len(synchronizedData['signal']['wrist']['ACC'])
        empaticaBVP_numPoints = len(synchronizedData['signal']['wrist']['BVP'])
        empaticaEDA_Temp_numPoints = len(synchronizedData['signal']['wrist']['EDA'])
        # Get the Empatica time points.
        empaticaACC_timePoints = self.universalMethods.getEvenlySampledArray(self.empaticaACC_samplingFreq, empaticaACC_numPoints)
        empaticaBVP_timePoints = self.universalMethods.getEvenlySampledArray(self.empaticaBVP_samplingFreq, empaticaBVP_numPoints)
        empaticaEDA_Temp_timePoints = self.universalMethods.getEvenlySampledArray(self.empaticaEDA_Temp_samplingFreq, empaticaEDA_Temp_numPoints)
        # Compile the time-series data into the format expected.
        compiledSignalsACC = np.asarray(list(synchronizedData['signal']['wrist'].values())[0]).squeeze().T
        compiledSignalsBVP = np.asarray(list(synchronizedData['signal']['wrist'].values())[1]).T
        compiledSignalsEDA_Temp = np.asarray(list(synchronizedData['signal']['wrist'].values())[2:4]).squeeze()
        # Convert from uS to S for EDA.
        compiledSignalsEDA_Temp[0] = compiledSignalsEDA_Temp[0] * 1E-6

        # Get the offset time
        classLabels = synchronizedData['label']
        offsetTime = self.calculateOffsetTime(respiBAN_timePoints, classLabels, experimentTimes)
        # Add the offset time
        respiBAN_timePoints += offsetTime
        empaticaACC_timePoints += offsetTime
        empaticaBVP_timePoints += offsetTime
        empaticaEDA_Temp_timePoints += offsetTime

        if showPlots:
            plt.plot(respiBAN_timePoints, synchronizedData['label'], label="Experiment Labels")
            plt.vlines(experimentTimes[:, 0], 0, 7, 'tab:red', label="Start Experiment")
            plt.vlines(experimentTimes[:, 1], 0, 7, 'black', label="End Experiment")
            plt.title("WESAD Experiment")
            plt.xlabel("Time (Seconds)")
            plt.ylabel("Experiment Label")
            plt.legend()
            plt.show()

        return respiBAN_timePoints, empaticaACC_timePoints, empaticaBVP_timePoints, empaticaEDA_Temp_timePoints, \
            compiledSignalsRespiban, compiledSignalsACC, compiledSignalsBVP, compiledSignalsEDA_Temp

    def compileAllData(self, allSynchronizedData, allExperimentalTimes, showPlots=True):
        allCompiledDatas = []
        # For each WESAD subject.
        for subjectInd in range(len(allSynchronizedData)):
            # Extract the data from the subject.
            experimentTimes = allExperimentalTimes[subjectInd]
            synchronizedData = allSynchronizedData[subjectInd]
            compiledData_eachFreq = []

            # Compile the data: specific to the device worn.
            respiBAN_timePoints, empaticaACC_timePoints, empaticaBVP_timePoints, empaticaEDA_Temp_timePoints, \
                compiledSignalsRespiban, compiledSignalsACC, compiledSignalsBVP, compiledSignalsEDA_Temp \
                = self.organizeSynchronizedData(synchronizedData, experimentTimes, showPlots=showPlots)

            # Organize the compiled data
            compiledData_eachFreq.append([respiBAN_timePoints, compiledSignalsRespiban])
            compiledData_eachFreq.append([empaticaBVP_timePoints, compiledSignalsBVP])
            compiledData_eachFreq.append([empaticaEDA_Temp_timePoints, compiledSignalsEDA_Temp])
            # Save the compiled data
            allCompiledDatas.append(compiledData_eachFreq)

        return allCompiledDatas

    @staticmethod
    def calculateOffsetTime(timepoints, classLabels, experimentTimes):
        # Get the time difference between the estimated and real.
        firstStartTime_EST = timepoints[(classLabels != 0).argmax()]
        # Calculate the offset time.
        offsetTime = experimentTimes[0][0] - firstStartTime_EST
        assert offsetTime >= 0

        return offsetTime

    def compileTrainingInfo(self):
        # Compile the data: specific to the device worn.
        streamingOrder, biomarkerFeatureOrder, featureAverageWindows, filteringOrders = self.getStreamingInfo()
        featureNames, biomarkerFeatureNames, biomarkerFeatureOrder = self.compileFeatureNames.extractFeatureNames(biomarkerFeatureOrder)

        return streamingOrder, biomarkerFeatureOrder, featureAverageWindows, featureNames, biomarkerFeatureNames


if __name__ == "__main__":
    # Initialize metadata analysis class.
    wesadAnalysisClass = wesadInterface()

    analyzingData = True
    trainingData = False

    if analyzingData:
        # Extract the metadata
        wesadAllSynchronizedData, wesadSubjectOrder, wesadAllExperimentalTimes, wesadAllExperimentalNames, \
            wesadAllSurveyAnswerTimes, wesadAllSurveyAnswersList, wesadAllContextualInfo = wesadAnalysisClass.getData()
        wesadAllCompiledDatas = wesadAnalysisClass.compileAllData(wesadAllSynchronizedData, wesadAllExperimentalTimes, showPlots=False)
        # Compile the data: specific to the device worn.
        wesadStreamingOrder, wesadBiomarkerFeatureOrder, wesadFeatureAverageWindows, wesadFilteringOrders = wesadAnalysisClass.getStreamingInfo()
        # Analyze and save the metadata features
        wesadAnalysisClass.extractFeatures(wesadAllCompiledDatas, wesadSubjectOrder, wesadAllExperimentalTimes, wesadAllExperimentalNames, wesadAllSurveyAnswerTimes, wesadAllSurveyAnswersList, wesadAllContextualInfo,
                                           wesadStreamingOrder, wesadBiomarkerFeatureOrder, wesadFeatureAverageWindows, wesadFilteringOrders, interfaceType='wesad', reanalyzeData=True, showPlots=False, analyzeSequentially=False)

    if trainingData:
        # Prepare the data to go through the training interface.
        wesadStreamingOrder, wesadBiomarkerFeatureOrder, wesadFeatureAverageWindows, wesadFeatureNames, wesadBiomarkerFeatureNames = wesadAnalysisClass.compileTrainingInfo()

        plotTrainingData = False
        # Collected the training data.
        wesadAllRawFeatureTimesHolders, wesadAllRawFeatureHolders, wesadAllRawFeatureIntervals, wesadAllRawFeatureIntervalTimes, \
            wesadAllAlignedFeatureTimes, wesadAllAlignedFeatureHolder, wesadAllAlignedFeatureIntervals, wesadAllAlignedFeatureIntervalTimes, wesadSubjectOrder, \
            wesadExperimentOrder, wesadActivityNames, wesadActivityLabels, wesadAllFinalLabels, wesadFeatureLabelTypes, wesadSurveyQuestions, wesadSurveyAnswersList, wesadSurveyAnswerTimes \
            = wesadAnalysisClass.trainingProtocolInterface(wesadStreamingOrder, wesadBiomarkerFeatureOrder, wesadFeatureAverageWindows, wesadFeatureNames, wesadBiomarkerFeatureNames, plotTrainingData, metaTraining=True)
