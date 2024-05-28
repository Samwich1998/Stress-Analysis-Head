# General
import os
import re
import sys
import scipy
import numpy as np
import pandas as pd

# Module to Sort Files in Order
from natsort import natsorted

# Modules to plot
import seaborn as sns
import matplotlib.pyplot as plt

# Data interface modules
from openpyxl import load_workbook

# Import Files for Machine Learning
from .modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from .featureAnalysis.featurePlotting import featurePlotting

# Import excel data interface
from ..dataAcquisitionAndAnalysis.excelProcessing.extractDataProtocols import extractData
from ..dataAcquisitionAndAnalysis.excelProcessing.saveDataProtocols import saveExcelData


class trainingProtocols(extractData):

    def __init__(self, biomarkerFeatureNames, streamingOrder, biomarkerOrder, numberOfChannels, trainingFolder, readData):
        """
        Parameters
        ----------
        trainingFolder: The Folder with ONLY the Training Data Excel Files
        """
        super().__init__()
        # General parameters
        self.readData = readData
        self.trainingFolder = trainingFolder
        self.biomarkerOrder = biomarkerOrder
        self.streamingOrder = streamingOrder
        self.numberOfChannels = numberOfChannels
        self.biomarkerFeatureNames = biomarkerFeatureNames

        # Extract feature information
        self.featureNames = [item for sublist in self.biomarkerFeatureNames for item in sublist]

        # Initialize important classes
        self.saveInputs = saveExcelData()
        self.modelInfoClass = compileModelInfo()
        self.analyzeFeatures = featurePlotting(self.trainingFolder + "dataAnalysis/", overwrite=False)

    def streamTrainingData(self, featureAverageWindows, plotTrainingData=False, reanalyzeData=False, extendedTime=False, metaTraining=False):
        # Hold time series analysis of features.
        allRawFeatureIntervals = [];
        allRawFeatureIntervalTimes = []
        allAlignedFeatureIntervals = [];
        allAlignedFeatureIntervalTimes = []
        # Hold features extraction information.
        allRawFeatureHolders = [];
        allRawFeatureTimesHolders = []
        allAlignedFeatureHolder = [];
        allAlignedFeatureTimes = []
        # Hold survey information
        surveyQuestions = [];
        subjectInformationQuestions = []
        surveyAnswersList = [];
        surveyAnswerTimes = []
        # Hold experimental information.
        subjectOrder = [];
        experimentalOrder = []
        # Final parameters for ML;
        allFinalFeatures = []

        # ------------------------------------------------------------------ #
        # ---------------------- Collect Training Data --------------------- #
        # For each file in the training folder.
        for excelFile in natsorted(os.listdir(self.trainingFolder)):
            # Only analyze Excel files with the training signals.
            if not excelFile.endswith(".xlsx") or excelFile.startswith(("~", ".")):
                continue
            # Extract the file details
            trainingExcelFile = self.trainingFolder + excelFile
            excelFileName = excelFile.split(".")[0]

            # -------------------------------------------------------------- #
            # ---------------- Extract Raw Training Features --------------- #  
            print(f"\nCompiling training features from {excelFileName}")

            # If the file ends with {self.saveFeatureFile_Appended}, then the file is a feature file.
            if excelFile.endswith(self.saveFeatureFile_Appended):
                print("\tFound a metadata training set.")
                savedFeaturesFile = self.trainingFolder + excelFile
            else:
                savedFeaturesFile = self.trainingFolder + self.saveFeatureFolder + excelFile.split(".")[0] + self.saveFeatureFile_Appended

            # If you want to and can use previously extracted features
            if not reanalyzeData and os.path.isfile(savedFeaturesFile):
                print("\tUsing the previously analyzed features.")
                self.analyzeFeatures.overwrite = False
                rawFeatureTimesHolder, rawFeatureHolder, _, experimentTimes, experimentNames, currentSurveyAnswerTimes, \
                    currentSurveyAnswersList, surveyQuestions, currentSubjectInformationAnswers, subjectInformationQuestions \
                    = self.getFeatures(self.biomarkerOrder, savedFeaturesFile, self.biomarkerFeatureNames, surveyQuestions, subjectInformationQuestions)
            else:
                print("\tReanalyzing the feature file.")
                # Read in the training file with the raw data,
                WB = load_workbook(trainingExcelFile, data_only=True, read_only=True)

                self.readData.resetGlobalVariables()
                # Extract and analyze the raw data.
                compiledRawData, experimentTimes, experimentNames, currentSurveyAnswerTimes, currentSurveyAnswersList, surveyQuestions, currentSubjectInformationAnswers, subjectInformationQuestions \
                    = self.extractExperimentalData(WB.worksheets, self.numberOfChannels, surveyQuestions=surveyQuestions, finalSubjectInformationQuestions=subjectInformationQuestions)
                self.readData.streamExcelData(compiledRawData, experimentTimes, experimentNames, currentSurveyAnswerTimes, currentSurveyAnswersList, surveyQuestions, currentSubjectInformationAnswers, subjectInformationQuestions,
                                              excelFileName)
                # Extract information from the streamed data
                rawFeatureHolder = self.readData.rawFeatureHolder.copy()
                rawFeatureTimesHolder = self.readData.rawFeatureTimesHolder.copy()

                # Plot the signals
                self.analyzeFeatures.overwrite = True
                self.analyzeFeatures.plotRawData(self.readData, compiledRawData, currentSurveyAnswerTimes, experimentTimes, experimentNames, self.streamingOrder, folderName=excelFileName + "/rawSignals/")

                # Save the features to be analyzed in the future.
                self.saveInputs.saveRawFeatures(rawFeatureTimesHolder, rawFeatureHolder, self.biomarkerFeatureNames, self.biomarkerOrder, experimentTimes, experimentNames, currentSurveyAnswerTimes,
                                                currentSurveyAnswersList, surveyQuestions, currentSubjectInformationAnswers, subjectInformationQuestions, trainingExcelFile)

            noFeaturesFound = False
            # Assert consistency across training data.
            assert len(self.biomarkerOrder) == len(rawFeatureHolder), "Incorrect number of channels"
            # Checkpoint: are there features in ALL categories
            for biomarkerInd in range(len(rawFeatureHolder)):
                if not len(rawFeatureHolder[biomarkerInd]) > 1:
                    noFeaturesFound = True
                # assert len(rawFeatureHolder[biomarkerInd]) > 1, "\tMissing raw features in " + self.biomarkerOrder[biomarkerInd].upper() + " signal"
            if noFeaturesFound: print("No features found"); continue

            # Convert to numpy arrays.
            for biomarkerInd in range(len(rawFeatureHolder)):
                rawFeatureHolder[biomarkerInd] = np.asarray(rawFeatureHolder[biomarkerInd])
                rawFeatureTimesHolder[biomarkerInd] = np.asarray(rawFeatureTimesHolder[biomarkerInd])

            # -------------------------------------------------------------- #
            # -------------------- Compile Raw Features -------------------- #
            # Setup the compilation variables
            allRawFeatureTimesHolders.append(rawFeatureTimesHolder)
            allRawFeatureHolders.append(rawFeatureHolder);
            compiledFeatureHolders = []

            # Average the features across a sliding window at each timePoint
            for biomarkerInd in range(len(rawFeatureHolder)):
                rawFeatures = rawFeatureHolder[biomarkerInd]
                rawFeatureTimes = rawFeatureTimesHolder[biomarkerInd]
                averageWindow = featureAverageWindows[biomarkerInd]

                # # Standardize the raw data
                # standardizeClass_rawFeatures = standardizeData(rawFeatures, threshold = 0)
                # standardizedRawFeatures = standardizeClass_rawFeatures.standardize(rawFeatures)
                standardizedRawFeatures = rawFeatures

                # Perform the feature averaging
                compiledFeatures = self.readData.averageFeatures_DEPRECATED(rawFeatureTimes, standardizedRawFeatures, averageWindow)
                compiledFeatureHolders.append(compiledFeatures)

            # -------------------------------------------------------------- #
            # ------------------ Align and Predict Labels ------------------ #
            self.readData.resetGlobalVariables()
            self.readData.experimentNames = experimentNames
            self.readData.experimentTimes = experimentTimes
            self.readData.setUserName(excelFile)

            # Align all the features, removing boundaries without any alignment.
            lastTimePoint = min((rawFeatureTimes[-1] if len(rawFeatureTimes) != 0 else 0) for rawFeatureTimes in rawFeatureTimesHolder)
            self.readData.alignFeatures(lastTimePoint, 1, rawFeatureTimesHolder, compiledFeatureHolders)
            # If there are no aligned features.
            if len(self.readData.alignedFeatures) == 0:
                print("No aligned features found");
                allRawFeatureTimesHolders.pop()
                compiledFeatureHolders.pop()
                allRawFeatureHolders.pop()
                continue

            # Get the aligned feature information
            alignedFeatures = np.asarray(self.readData.alignedFeatures)
            alignedFeatureTimes = np.asarray(self.readData.alignedFeatureTimes)
            # Store the aligned feature information
            allAlignedFeatureHolder.append(alignedFeatures)
            allAlignedFeatureTimes.append(alignedFeatureTimes)

            if not metaTraining and False:
                # Get the predicted data
                self.readData.predictLabels()
                predictedLabels_List = self.readData.alignedFeatureLabels
                _, allTrueLabels = self.modelInfoClass.extractFinalLabels(currentSurveyAnswersList, [])

                # Plot the predicted stress of each person, if model provided
                for modelInd in range(len(self.modelInfoClass.predictionOrder)):
                    self.analyzeFeatures.plotPredictedScores(alignedFeatureTimes, allTrueLabels[modelInd][0] * np.ones(len(alignedFeatureTimes)), allTrueLabels[modelInd], currentSurveyAnswerTimes, experimentTimes, experimentNames,
                                                             predictionType=self.modelInfoClass.predictionOrder[modelInd], folderName=excelFileName + "/realTimePredictions/")
                    self.analyzeFeatures.plotPredictedScores(alignedFeatureTimes, predictedLabels_List[modelInd], allTrueLabels[modelInd], currentSurveyAnswerTimes, experimentTimes, experimentNames,
                                                             predictionType=self.modelInfoClass.predictionOrder[modelInd], folderName=excelFileName + "/realTimePredictions/")

            # -------------------------------------------------------------- #
            # ------------ Match Each Label to Each Feature ------------ #
            assert len(experimentTimes) == len(currentSurveyAnswerTimes), print(experimentTimes, currentSurveyAnswerTimes)

            ### TODO: Sort the features with their correct labels.
            currentFinalFeatures = [];
            badExperimentalInds = []
            # For each experiment performed in the trial.
            for experimentInd in range(len(experimentTimes)):
                startSurveyTime = currentSurveyAnswerTimes[experimentInd]
                startExperimentTime, endExperimentTime = experimentTimes[experimentInd]

                # Append a new data point.
                allRawFeatureIntervals.append([])
                allRawFeatureIntervalTimes.append([])
                allAlignedFeatureIntervals.append([])
                allAlignedFeatureIntervalTimes.append([])

                finalFeatures = []
                startAlignedIndex = 0
                # For each biomarker during that trial
                for biomarkerInd in range(len(rawFeatureHolder)):
                    rawFeatures = rawFeatureHolder[biomarkerInd]
                    rawFeatureTimes = rawFeatureTimesHolder[biomarkerInd]
                    # Get the training data's aligned features information
                    alignedFeatureSet = alignedFeatures[:, startAlignedIndex:startAlignedIndex + len(self.biomarkerFeatureNames[biomarkerInd])]
                    startAlignedIndex += len(self.biomarkerFeatureNames[biomarkerInd])

                    # Locate the experiment indices within the data
                    endStimuliInd = np.searchsorted(rawFeatureTimes, startSurveyTime, side='left')
                    startStimuliInd = np.searchsorted(rawFeatureTimes, startExperimentTime, side='left')

                    # Average the features within this region for all biomarkers.
                    startExperimentTime_model = max(0, startSurveyTime - 500)
                    finalFeature = self.readData.getFinalFeatures(rawFeatureTimes, rawFeatures, (startExperimentTime_model, startSurveyTime))
                    finalFeatures.extend(finalFeature)

                    # Save raw interval information
                    allRawFeatureIntervals[-1].extend(rawFeatures.T[:, startStimuliInd:endStimuliInd])
                    allRawFeatureIntervalTimes[-1].extend([rawFeatureTimes[startStimuliInd:endStimuliInd] for _ in range(len(rawFeatures[0]))])

                    # Locate the experiment indices within the data
                    endStimuliInd = np.searchsorted(alignedFeatureTimes, startSurveyTime, side='left')
                    startStimuliInd = np.searchsorted(alignedFeatureTimes, startExperimentTime_model, side='left')
                    # Save aligned interval information
                    allAlignedFeatureIntervals[-1].extend(alignedFeatureSet.T[:, startStimuliInd:endStimuliInd])
                    allAlignedFeatureIntervalTimes[-1].extend([alignedFeatureTimes[startStimuliInd:endStimuliInd] for _ in range(len(rawFeatures[0]))])

                if experimentInd == -10:
                    baseline = finalFeatures.copy()
                    badExperimentalInds.append(experimentInd)
                else:
                    # finalFeatures = np.asarray(finalFeatures) - np.asarray(baseline)
                    # Record the features
                    allFinalFeatures.append(finalFeatures)
                    currentFinalFeatures.append(finalFeatures)
                    # Save the analysis order 
                    experimentalOrder.append(experimentNames[experimentInd])

                    if metaTraining:
                        subjectOrder.append(int(re.search(r'\d+', excelFileName).group()))
                    else:
                        subjectOrder.append(" ".join(excelFileName.split(" ")[1:]))

            ### TODO:  REMOVE OTHER BAD DATA LIEK ALIGNED/RAW
            # currentSurveyAnswersList = np.asarray(currentSurveyAnswersList)
            # currentSurveyAnswersList -= currentSurveyAnswersList[0]
            # Remove indices where no features were collected.
            for experimentInd in sorted(badExperimentalInds, reverse=True):
                del currentSurveyAnswerTimes[experimentInd]
                currentSurveyAnswersList = np.delete(currentSurveyAnswersList, experimentInd, axis=0)

                # -------------------------------------------------------------- #
            # -------------------- Plot the features ------------------- #
            if plotTrainingData or False:  # or "Exercise Trial Hyunah" in excelFile or "Exercise Trial MQ" in excelFile:
                # Plot the feature correlation to each emotion/stress score.
                # if not metaTraining:
                #     self.analyzeFeatures.plotPsychCorrelation(currentFinalFeatures, currentSurveyAnswersList, self.featureNames, folderName = excelFileName + "/mentalStateCorrelation/")
                # self.analyzeFeatures.plotEmotionCorrelation(currentFinalFeatures, currentSurveyAnswersList, surveyQuestions, self.featureNames, folderName = excelFileName + "/mentalStateCorrelation/")
                if extendedTime or True:
                    # self.analyzeFeatures.plotEmotionCorrelation(currentFinalFeatures, currentSurveyAnswersList, surveyQuestions, self.featureNames, folderName = excelFileName + "/mentalStateCorrelation/")

                    startAlignedIndex = 0
                    # For all the biomarkers in the experiment.
                    for biomarkerInd in range(len(rawFeatureHolder)):
                        # Get the training data's raw features information
                        rawFeatures = np.asarray(rawFeatureHolder[biomarkerInd])
                        rawFeatureTimes = np.asarray(rawFeatureTimesHolder[biomarkerInd])
                        # Get the training data's aligned features information
                        alignedFeatureSet = alignedFeatures[:, startAlignedIndex:startAlignedIndex + len(self.biomarkerFeatureNames[biomarkerInd])]
                        startAlignedIndex += len(self.biomarkerFeatureNames[biomarkerInd])

                        # # Plot each biomarker's features from the training file.
                        self.analyzeFeatures.singleFeatureAnalysis(rawFeatureTimes, rawFeatures, self.biomarkerFeatureNames[biomarkerInd], preAveragingSeconds=0, averageIntervalList=[60, 75, 90, 120],
                                                                   surveyCollectionTimes=currentSurveyAnswerTimes, experimentTimes=experimentTimes, experimentNames=experimentNames,
                                                                   folderName=excelFileName + "/Feature Analysis/singleFeatureAnalysis - " + self.biomarkerOrder[biomarkerInd].upper() + "/")
                        self.analyzeFeatures.singleFeatureAnalysis(alignedFeatureTimes, alignedFeatureSet, self.biomarkerFeatureNames[biomarkerInd], preAveragingSeconds=featureAverageWindows[biomarkerInd], averageIntervalList=[0],
                                                                   surveyCollectionTimes=currentSurveyAnswerTimes, experimentTimes=experimentTimes, experimentNames=experimentNames,
                                                                   folderName=excelFileName + "/Feature Analysis/alignedFeatureAnalysis - " + self.biomarkerOrder[biomarkerInd].upper() + "/")
                        # Plot the correlation across features
                        # self.analyzeFeatures.correlationMatrix(alignedFeatures, self.featureNames, folderName = "correlationMatrix/") # Hurts Plotting Style

            # -------------------------------------------------------------- #
            # ------------------ Organize Information ------------------ #
            # Save the survey labels.
            surveyAnswersList.extend(currentSurveyAnswersList)
            surveyAnswerTimes.append(currentSurveyAnswerTimes)

            # -------------------------------------------------------------- #

        # ------------------------------------------------------------------ #
        # ---------------------- Compile Training Data --------------------- #
        # Organize the final labels for the features
        featureLabelTypes, allFinalLabels = [], []
        if not metaTraining:
            featureLabelTypes, allFinalLabels = self.modelInfoClass.extractFinalLabels(surveyAnswersList, allFinalLabels)

        # ------------------------------------------------------------------ #
        # -------------------- Plot Training Information ------------------- #
        if plotTrainingData or False:
            print("\nPlotting All Subject Information")
            # plot the feature correlation to each emotion/stress score.
            # if not metaTraining:
            #     self.analyzeFeatures.plotPsychCorrelation(allFinalFeatures, surveyAnswersList, self.featureNames, folderName = "Final Analysis/mentalStateCorrelation/", subjectOrder = subjectOrder)
            # if extendedTime or False:
            #     self.analyzeFeatures.plotEmotionCorrelation(allFinalFeatures, surveyAnswersList, surveyQuestions, self.featureNames, folderName = "Final Analysis/mentalStateCorrelation/", subjectOrder = subjectOrder)

            # # For each final label
            # for labelInd in range(len(allFinalLabels)):
            #     labelType = featureLabelTypes[labelInd]
            #     finalLabels = allFinalLabels[labelInd]

            #     # Plot the feature distribution across the final labels
            #     self.analyzeFeatures.featureDistribution(allFinalFeatures, finalLabels, self.featureNames, labelType, folderName = "Final Analysis/featureDistributions/")
        # ------------------------------------------------------------------ #
        allFinalFeatures, allFinalLabels = np.asarray(allFinalFeatures), np.asarray(allFinalLabels)
        surveyQuestions, surveyAnswersList = np.asarray(surveyQuestions), np.asarray(surveyAnswersList)
        print(surveyQuestions)

        assert len(allRawFeatureTimesHolders) == len(allRawFeatureHolders) == len(allAlignedFeatureTimes) == len(allAlignedFeatureHolder)
        assert len(allRawFeatureIntervals) == len(allRawFeatureIntervalTimes) == len(allAlignedFeatureIntervals) == len(allAlignedFeatureIntervalTimes)

        # Return Training Data and Labels
        return allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
            subjectOrder, experimentalOrder, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes

    def varyAnalysisParam(self, dataFile, featureAverageWindows, featureTimeWindows):
        print("\nLoading Excel File", dataFile)
        # Read in the training file with the raw data,
        WB = load_workbook(dataFile, data_only=True, read_only=True)

        allRawFeatureTimesHolders = [];
        allRawFeatureHolders = []

        # Extract and analyze the raw data.
        compiledRawData, experimentTimes, experimentNames, currentSurveyAnswerTimes, currentSurveyAnswersList, surveyQuestions, currentSubjectInformationAnswers, subjectInformationQuestions = self.extractExperimentalData(WB.worksheets,
                                                                                                                                                                                                                             self.numberOfChannels,
                                                                                                                                                                                                                             surveyQuestions=[],
                                                                                                                                                                                                                             finalSubjectInformationQuestions=[])
        # For each test parameter
        for featureTimeWindow in featureTimeWindows:
            print(featureTimeWindow)
            # Set the parameter in the analysis
            self.readData.setFeatureWindowEEG(featureTimeWindow)

            # Stream the data
            self.readData.streamExcelData(compiledRawData, experimentTimes, experimentNames, currentSurveyAnswerTimes, currentSurveyAnswersList, surveyQuestions, currentSubjectInformationAnswers, subjectInformationQuestions, dataFile)
            # Extract information from the streamed data
            allRawFeatureTimesHolders.append(self.readData.rawFeatureTimesHolder.copy())
            allRawFeatureHolders.append(self.readData.rawFeatureHolder.copy());
            # Remove all previous information from this trial
            self.readData.resetGlobalVariables()

            # Assert consistency across training data
            assert len(self.biomarkerOrder) == len(allRawFeatureHolders[-1]), "Incorrect number of channels"

        biomarkerInd = self.biomarkerOrder.index('eeg')
        # For each EEG Feature
        for featureInd in range(len(self.biomarkerFeatureNames[biomarkerInd])):

            heatMap = [];
            finalTimePoints = np.arange(max(allRawFeatureTimesHolders, key=lambda x: x[biomarkerInd][0])[biomarkerInd][0], min(allRawFeatureTimesHolders, key=lambda x: x[biomarkerInd][-1])[biomarkerInd][-1], 0.1)
            # For each feature-variation within a certain time window
            for trialInd in range(len(allRawFeatureHolders)):
                averageWindow = featureAverageWindows[biomarkerInd]
                rawFeatures = np.asarray(allRawFeatureHolders[trialInd][biomarkerInd])[:, featureInd]
                rawFeatureTimes = np.asarray(allRawFeatureTimesHolders[trialInd][biomarkerInd])

                # Perform the feature averaging
                compiledFeatures = self.readData.averageFeatures_DEPRECATED(rawFeatureTimes, rawFeatures, averageWindow)

                # Interpolate all the features within the same time-window
                featurePolynomial = scipy.interpolate.interp1d(rawFeatureTimes, compiledFeatures, kind='linear')
                finalFeatures = featurePolynomial(finalTimePoints)

                # Track the heatmap
                heatMap.append(finalFeatures)

            for finalFeatures in heatMap:
                plt.plot(finalTimePoints, finalFeatures, linewidth=1)

            plt.show()

            vMin = np.asarray(heatMap)[:, 100:].min()
            vMax = np.asarray(heatMap)[:, 100:].max()
            # Plot the heatmap
            ax = sns.heatmap(pd.DataFrame(heatMap, index=featureTimeWindows, columns=np.round(finalTimePoints, 2)), robust=True, vmin=vMin, vmax=vMax, cmap='icefire', yticklabels='auto')
            ax.set(title='Feature:' + self.biomarkerFeatureNames[biomarkerInd][featureInd])
            # Save the Figure
            sns.set(rc={'figure.figsize': (7, 9)})
            plt.xlabel("Time (Seconds)")
            plt.ylabel("Sliding Window Size")
            fig = ax.get_figure();
            fig.savefig(self.biomarkerFeatureNames[biomarkerInd][featureInd] + "Sliding Window.png", dpi=300)
            plt.show()

        return allRawFeatureTimesHolders, allRawFeatureHolders
