# General Modules
import os
import itertools
import numpy as np
import concurrent.futures
# Import interface for extracting feature names
from ...machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames  # Import Files for extracting feature names

# Import files for machine learning
from ...machineLearning.featureAnalysis.featurePlotting import featurePlotting
from ...machineLearning.trainingProtocols import trainingProtocols

# Import files for data analysis
from ..biolectricProtocols.helperMethods.filteringProtocols import filteringMethods
from ..biolectricProtocols.helperMethods.universalProtocols import universalMethods
from ..excelProcessing.excelFormatting import handlingExcelFormat
from ..excelProcessing.saveDataProtocols import saveExcelData
from ..streamingProtocols import streamingProtocols


class globalMetaAnalysis(handlingExcelFormat):

    def __init__(self, trainingFolder, surveyQuestions):
        super().__init__()
        self.trainingFolder = trainingFolder
        self.surveyQuestions = surveyQuestions
        self.savedFeatureFolder = self.trainingFolder + self.saveFeatureFolder

        # Define general classes to process data.
        self.saveInputs = saveExcelData()
        self.filteringMethods = filteringMethods()
        self.universalMethods = universalMethods()
        self.compileFeatureNames = compileFeatureNames()
        self.analyzeFeatures = featurePlotting(self.trainingFolder + "dataAnalysis/", overwrite=False)

    # ---------------------------------------------------------------------- #
    # ------------------ Interface with Training Protocols ----------------- #

    def extractFeatures(self, allCompiledDatas, subjectOrder, allExperimentalTimes, allExperimentalNames, allSurveyAnswerTimes, allSurveyAnswersList, allContextualInfo,
                        streamingOrder, biomarkerOrder, featureAverageWindows, filteringOrders, interfaceType="emognition", reanalyzeData=False, showPlots=True):
        # Prepare the data for each subject for parallel processing
        subjects_data = [
            (
                subjectOrder[i],
                allCompiledDatas[i],
                allExperimentalTimes[i],
                allExperimentalNames[i],
                allSurveyAnswerTimes[i],
                allSurveyAnswersList[i],
                allContextualInfo[i],
                streamingOrder,
                biomarkerOrder,
                featureAverageWindows,
                filteringOrders,
                interfaceType,
                reanalyzeData,
                showPlots
            ) for i in range(len(subjectOrder))
        ]

        # Use ProcessPoolExecutor to process each subject in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.processFeatures, subjects_data)

    def processFeatures(self, subject_data):
        # Unpack data
        (
            subjectName,
            compiledData_eachFreq,
            experimentTimes,
            experimentNames,
            currentSurveyAnswerTimes,
            currentSurveyAnswersList,
            contextualInfo,
            streamingOrder,
            biomarkerOrder,
            featureAverageWindows,
            filteringOrders,
            interfaceType,
            reanalyzeData,
            showPlots
        ) = subject_data

        # Organize the data for streaming.
        allFeatureAverageWindows = []
        biomarkerFeatureNames = []
        allStreamingOrders = []
        allBiomarkerOrders = []
        allFilteringOrders = []
        signalPointer = 0

        # Recompile considering the segmentation of the features.
        for segmentationInd in range(len(compiledData_eachFreq)):
            numSignals = len(compiledData_eachFreq[segmentationInd][1])

            # Separate out the relevant signals
            currentStreamingOrder = streamingOrder[signalPointer:signalPointer + numSignals]
            currentBiomarkerOrder = biomarkerOrder[signalPointer:signalPointer + numSignals]
            currentFilteringOrders = filteringOrders[signalPointer:signalPointer + numSignals]
            currentFeatureAverageWindows = featureAverageWindows[signalPointer:signalPointer + numSignals]

            # Organize the biomarker order
            currentFeatureNames, currentBiomarkerFeatureNames, currentBiomarkerOrder = self.compileFeatureNames.extractFeatureNames(currentBiomarkerOrder)

            # Organize the segmentation block
            allStreamingOrders.append(currentStreamingOrder)
            allBiomarkerOrders.append(currentBiomarkerOrder)
            allFilteringOrders.append(currentFilteringOrders)
            biomarkerFeatureNames.extend(currentBiomarkerFeatureNames)
            allFeatureAverageWindows.append(currentFeatureAverageWindows)

            # Reset for the next round
            signalPointer += numSignals

        # Stream the data.
        self.streamData(subjectName, allStreamingOrders, allBiomarkerOrders, allFeatureAverageWindows, allFilteringOrders, compiledData_eachFreq, experimentTimes,
                        experimentNames, currentSurveyAnswerTimes, currentSurveyAnswersList, contextualInfo, interfaceType, biomarkerFeatureNames, reanalyzeData, showPlots=showPlots)

    def streamData(self, subjectName, allStreamingOrders, allBiomarkerOrders, allFeatureAverageWindows, allFilteringOrders, compiledData_eachFreq, experimentTimes,
                   experimentNames, currentSurveyAnswerTimes, currentSurveyAnswersList, contextualInfo, interfaceType, biomarkerFeatureNames, reanalyzeData=False, showPlots=False):
        # Check if the features have already been extracted.        
        saveFeatureFile = self.savedFeatureFolder + subjectName + self.saveFeatureFile_Appended
        if os.path.exists(saveFeatureFile) and not reanalyzeData:
            print("\tNot reanalyzing file");
            return None

        # Get the flattened versions.
        biomarkerOrder = list(itertools.chain(*allBiomarkerOrders))

        rawFeatureHolder = []
        rawFeatureTimesHolder = []
        # Organize the data for streaming.
        for compiledDataInd in range(len(compiledData_eachFreq)):
            compiledRawData = compiledData_eachFreq[compiledDataInd]
            currentStreamingOrder = allStreamingOrders[compiledDataInd]
            currentBiomarkerOrder = allBiomarkerOrders[compiledDataInd]
            currentFilteringOrders = allFilteringOrders[compiledDataInd]
            currentFeatureAverageWindows = allFeatureAverageWindows[compiledDataInd]

            # Initialize instance to analyze the data
            readData = streamingProtocols(None, [], None, 2048576, 1048100, currentStreamingOrder, currentBiomarkerOrder, currentFeatureAverageWindows, False)
            readData.resetGlobalVariables()
            # Change filter of the analyses.
            for analysisTypeInd in range(len(currentBiomarkerOrder)):
                analysisType = currentBiomarkerOrder[analysisTypeInd]
                cutOffFreqs = currentFilteringOrders[analysisTypeInd]

                readData.analysisProtocols[analysisType].cutOffFreq = cutOffFreqs

            # Extract and analyze the raw data.
            readData.streamExcelData(compiledRawData, experimentTimes, experimentNames, currentSurveyAnswerTimes,
                                     currentSurveyAnswersList, self.surveyQuestions, [], [], interfaceType + " " + subjectName)
            # Compile all the features from each signal
            rawFeatureHolder.extend(readData.rawFeatureHolder.copy())
            rawFeatureTimesHolder.extend(readData.rawFeatureTimesHolder.copy())

            if showPlots:
                # Plot the signals
                self.analyzeFeatures.overwrite = True
                self.analyzeFeatures.plotRawData(readData, compiledRawData, currentSurveyAnswerTimes, experimentTimes, experimentNames, currentStreamingOrder, folderName=interfaceType + " " + subjectName + "/rawSignals/")

        # Assert the integrity of data combination
        assert len(rawFeatureHolder) == len(biomarkerOrder)
        assert len(rawFeatureTimesHolder) == len(biomarkerOrder)

        # Save the features to be analyzed in the future.
        trainingExcelFile = self.trainingFolder + subjectName + "/" + subjectName + ".xlsx"
        self.saveInputs.saveRawFeatures(rawFeatureTimesHolder, rawFeatureHolder, biomarkerFeatureNames, biomarkerOrder, experimentTimes, experimentNames, currentSurveyAnswerTimes,
                                        currentSurveyAnswersList, self.surveyQuestions, contextualInfo[1], contextualInfo[0], trainingExcelFile)

    def trainingProtocolInterface(self, streamingOrder, biomarkerOrder, featureAverageWindows, biomarkerFeatureNames, plotTrainingData=True, metaTraining=True):
        # Initialize instance to train the data
        readData = streamingProtocols(None, [], None, 2048576, 1048100, streamingOrder, biomarkerOrder, featureAverageWindows, False)
        trainingInterface = trainingProtocols(biomarkerFeatureNames, streamingOrder, biomarkerOrder, len(streamingOrder), self.savedFeatureFolder, readData)

        # Extract the features from the training files and organize them.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
            subjectOrder, experimentOrder, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
            = trainingInterface.streamTrainingData(featureAverageWindows, plotTrainingData=plotTrainingData, reanalyzeData=False, metaTraining=metaTraining)

        return allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
            subjectOrder, experimentOrder, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes

    # --------------------------- General Methods -------------------------- #

    @staticmethod
    def resampleSignal(timePoints, signalData, newSamplingFreq):
        signalData = np.asarray(signalData).T  # Getting signalData into the format: numSignals, numPoints
        # Create a time interval to interpolate the features.
        newInterpolatedTimes = np.arange(timePoints[0], timePoints[-1] + 1 / newSamplingFreq, 1 / newSamplingFreq)
        alignedData = np.zeros((len(signalData), len(newInterpolatedTimes)))

        interpolate = True
        alignmentPointer = 0
        # Check which feature times to add
        for dataInd in range(len(newInterpolatedTimes)):
            interpTime = newInterpolatedTimes[dataInd]

            # Find the time right before currentTime
            while interpolate and timePoints[alignmentPointer] <= interpTime:
                alignmentPointer += 1  # Look at the next point
                # If no time before AND after currentTime, you cant interpolate in between
                if alignmentPointer == len(timePoints):
                    print("Extrapolation is not a good option")
                    interpolate = False
                    break

            if interpolate:
                # Specify the variables
                t1, t2 = timePoints[alignmentPointer - 1], timePoints[alignmentPointer]
                y1, y2 = signalData[:, alignmentPointer - 1], signalData[:, alignmentPointer]

            # Linearly connect the line to get the next value
            alignedData[:, dataInd] = (y1 + ((y2 - y1) / (t2 - t1)) * (interpTime - t1))

        return newInterpolatedTimes, alignedData
