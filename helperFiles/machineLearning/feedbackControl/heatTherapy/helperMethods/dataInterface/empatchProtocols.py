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

    def __init__(self, predictionOrder, predictionBounds, modelParameterBounds, therapyMethod="HeatingPad"):
        super().__init__()
        # General parameters.
        self.modelParameterBounds = modelParameterBounds  # The bounds for the model parameters.
        self.predictionBounds = predictionBounds  # The bounds for the predictions.
        self.predictionOrder = predictionOrder  # The order of the predictions.
        self.therapyMethod = therapyMethod  # The therapy method to analyze.

        # Collected data parameters.
        self.featureNames, self.biomarkerFeatureNames, self.biomarkerFeatureOrder = self.getFeatureInformation()

        # Initialize helper classes.
        self.compileModelInfo = compileModelInfo()

        # Get the data folder.
        self.trainingFolder = "/../../../../../../" + self.compileModelInfo.getTrainingDataFolder(useTherapyData=True)

        # Assert the validity of the input parameters.
        self.compileModelInfo.assertValidTherapyMethod(therapyMethod=self.therapyMethod)

    def getFeatureInformation(self):
        # Specify biomarker information.
        extractFeaturesFrom = self.compileModelInfo.streamingOrder  # A list with all the biomarkers from streamingOrder for feature extraction
        featureNames, biomarkerFeatureNames, biomarkerFeatureOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

        return featureNames, biomarkerFeatureNames, biomarkerFeatureOrder

    def getTherapyData(self):
        # Initialize holders.
        paramStates = []  # The state values for each experiment. Dimensions: numExperiments, (P1, P2, ...); 2D array
        lossStates = []  # The state values for each experiment. Dimensions: numExperiments, (PA, NA, SA); 2D array

        # For each file in the training folder.
        for excelFile in natsorted(os.listdir(self.trainingFolder)):
            # Remove any files that are not part of the therapy data.
            if not excelFile.endswith(".xlsx") or excelFile.startswith(("~", ".")): continue  # Only analyze Excel files with the training signals.
            if self.therapyMethod not in excelFile: continue  # Only analyze the therapy data.

            # Get the full file information.
            savedFeaturesFile = self.trainingFolder + self.saveFeatureFolder + excelFile.split(".")[0] + self.saveFeatureFile_Appended
            # Extract the features from the Excel file.
            rawFeatureTimesHolder, rawFeatureHolder, _, experimentTimes, experimentNames, currentSurveyAnswerTimes, \
                currentSurveyAnswersList, surveyQuestions, currentSubjectInformationAnswers, subjectInformationQuestions \
                = self.getFeatures(self.biomarkerFeatureOrder, savedFeaturesFile, self.biomarkerFeatureNames, surveyQuestions=[], finalSubjectInformationQuestions=[])
            # currentSurveyAnswersList: The user answers to the survey questions during the experiment. Dimensions: numExperiments, numSurveyQuestions
            # surveyQuestions: The questions asked in the survey. Dimensions: numSurveyQuestions = numEmotionsRecorded
            # currentSurveyAnswerTimes: The times the survey questions were answered. Dimensions: numExperiments

            # Extract the mental health information.
            predictionOrder, finalLabels = self.compileModelInfo.extractFinalLabels(currentSurveyAnswersList, finalLabels=[])
            assert predictionOrder == self.predictionOrder, f"The given prediction order {predictionOrder} does not match the model's expected prediction order {self.predictionOrder}."
            # finalLabels: The final mental health labels for the experiment. Dimensions: numLabels, numExperiments

            currentPredictionStates = dataInterface.normalizeParameters(currentParamBounds=self.predictionBounds, normalizedParamBounds=self.modelParameterBounds, currentParamValues=torch.tensor(finalLabels)).tolist()

            # For each experiment.
            for experimentalInd in range(len(experimentNames)):
                # Get the mental state values and normalize them.
                emotion_states = [finalLabels[0][experimentalInd], finalLabels[1][experimentalInd], finalLabels[2][experimentalInd]]

                stateHolder.append([experimentalTemp] + emotion_states)
        stateHolder = torch.as_tensor(stateHolder)
        temperature = stateHolder[:, 0].view(1, -1)
        pa = stateHolder[:, 1].view(1, -1)
        na = stateHolder[:, 2].view(1, -1)
        sa = stateHolder[:, 3].view(1, -1)

        return temperature, pa, na, sa

    def temperatureInterface(self):
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
            if self.therapyMethod in experimentName:
                experimentalTemp = int(experimentName.split("-")[-1])
            else:
                # Get the temperature at the start and end of the experiment.
                startTemperatureInd = torch.argmin(torch.abs(allTemperatureTimes - startExperimentTime))
                surveyAnswerTimeInd = torch.argmin(torch.abs(allTemperatureTimes - surveyAnswerTime))

                # Find the average temperature between the start and end times.
                experimentalTemp = torch.mean(allTemperatures[startTemperatureInd:surveyAnswerTimeInd])
