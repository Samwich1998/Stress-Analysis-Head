# General
from natsort import natsorted
import torch
import os

# Import files.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.dataInterface.dataInterface import dataInterface
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from helperFiles.dataAcquisitionAndAnalysis.excelProcessing.extractDataProtocols import extractData


class empatchProtocols(extractData):

    def __init__(self, predictionOrder, predictionBounds, modelParameterBounds):
        super().__init__()
        # General parameters.
        self.trainingFolder = os.path.dirname(__file__) + "/../../../../../../_experimentalData/allSensors/_finalTherapyData/"
        self.modelParameterBounds = modelParameterBounds
        self.predictionBounds = predictionBounds
        self.predictionOrder = predictionOrder

        # Collected data parameters.
        self.featureNames, self.biomarkerFeatureNames, self.biomarkerOrder = self.getFeatureInformation()

        # Initialize helper classes.
        self.modelInfoClass = compileModelInfo()

    @staticmethod
    def getFeatureInformation():
        # Specify biomarker information.
        extractFeaturesFrom = ["eog", "eeg", "eda", "temp"]  # A list with all the biomarkers from streamingOrder for feature extraction
        featureNames, biomarkerFeatureNames, biomarkerOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

        return featureNames, biomarkerFeatureNames, biomarkerOrder

    def getTherapyData(self):
        # Initialize holders.
        stateHolder = []  # The state values for each experiment. Dimensions: numExperiments, (T, PA, NA, SA); 2D array

        # For each file in the training folder.
        for excelFile in natsorted(os.listdir(self.trainingFolder)):
            # Only analyze Excel files with the training signals.
            if not excelFile.endswith(".xlsx") or excelFile.startswith(("~", ".")):
                continue
            # Only analyze the heating therapy data.
            if "HeatingPad" not in excelFile:
                continue
            # Get the full file information.
            savedFeaturesFile = self.trainingFolder + self.saveFeatureFolder + excelFile.split(".")[0] + self.saveFeatureFile_Appended
            print(savedFeaturesFile)

            # Extract the features from the Excel file.
            rawFeatureTimesHolder, rawFeatureHolder, _, experimentTimes, experimentNames, currentSurveyAnswerTimes, \
                currentSurveyAnswersList, surveyQuestions, currentSubjectInformationAnswers, subjectInformationQuestions \
                = self.getFeatures(self.biomarkerOrder, savedFeaturesFile, self.biomarkerFeatureNames, surveyQuestions=[], finalSubjectInformationQuestions=[])
            # currentSurveyAnswersList: The user answers to the survey questions during the experiment. Dimensions: numExperiments, numSurveyQuestions
            # surveyQuestions: The questions asked in the survey. Dimensions: numSurveyQuestions = numEmotionsRecorded
            # currentSurveyAnswerTimes: The times the survey questions were answered. Dimensions: numExperiments

            # Extract the mental health information.
            predictionOrder, finalLabels = self.modelInfoClass.extractFinalLabels(currentSurveyAnswersList, finalLabels=[])
            assert predictionOrder == self.predictionOrder, f"Expected prediction order: {self.predictionOrder}, but got {predictionOrder}"
            mean_temperature_list = torch.as_tensor(rawFeatureHolder[3])[:, 0]

            # Get the temperatures in the time window.
            allTemperatureTimes = torch.as_tensor(rawFeatureTimesHolder[3])
            allTemperatures = torch.as_tensor(mean_temperature_list)

            # For each experiment.
            for experimentalInd in range(len(experimentNames)):
                # Get the start and end times for the experiment.
                surveyAnswerTime = currentSurveyAnswerTimes[experimentalInd]
                startExperimentTime = experimentTimes[experimentalInd][0]
                experimentName = experimentNames[experimentalInd]

                # Extract the temperature used in the experiment.
                if "Heating" in experimentName:
                    experimentalTemp = int(experimentName.split("-")[-1])
                else:
                    # Get the temperature at the start and end of the experiment.
                    startTemperatureInd = torch.argmin(torch.abs(allTemperatureTimes - startExperimentTime))
                    surveyAnswerTimeInd = torch.argmin(torch.abs(allTemperatureTimes - surveyAnswerTime))

                    # Find the average temperature between the start and end times.
                    experimentalTemp = torch.mean(allTemperatures[startTemperatureInd:surveyAnswerTimeInd])

                # Store the state values.
                emotion_states = [finalLabels[0][experimentalInd], finalLabels[1][experimentalInd], finalLabels[2][experimentalInd]]
                emotion_states = dataInterface.normalizeParameters(currentParamBounds=self.predictionBounds, normalizedParamBounds=self.modelParameterBounds, currentParamValues=torch.tensor(emotion_states)).tolist()

                stateHolder.append([experimentalTemp] + emotion_states)
        stateHolder = torch.as_tensor(stateHolder)
        temperature = stateHolder[:, 0].view(1, -1)
        pa = stateHolder[:, 1].view(1, -1)
        na = stateHolder[:, 2].view(1, -1)
        sa = stateHolder[:, 3].view(1, -1)

        return temperature, pa, na, sa
