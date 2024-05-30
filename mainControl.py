""" Written by Samuel Solomon: https://scholar.google.com/citations?user=9oq12oMAAAAJ&hl=en """

# General
import numpy as np
import threading
import sys
import os

# Import helper files.
from helperFiles.dataAcquisitionAndAnalysis.excelProcessing import extractDataProtocols, saveDataProtocols  # Import interfaces for reading/writing data
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo  # Import files for machine learning
from helperFiles.machineLearning.dataInterface.dataPreparation import standardizeData  # Import interface for the data
from helperFiles.surveyInformation.questionaireGUI import stressQuestionnaireGUI  # Import file for GUI control
from helperFiles.dataAcquisitionAndAnalysis import streamingProtocols  # Import interfaces for reading/writing data
from helperFiles.machineLearning import trainingProtocols  # Import interfaces for reading/writing data
from adjustInputParameters import adjustInputParameters  # Import the class to adjust the input parameters

# Add the directory of the current file to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # Protocol switches: only the first true variably executes.
    readDataFromExcel = False  # Analyze Data from Excel File called 'currentFilename' on Sheet Number 'testSheetNum'
    streamData = True  # Stream in Data from the Board and Analyze.
    trainModel = False  # Train Model with ALL Data in 'collectedDataFolder'.

    # User options during the run: any number can be true.
    useModelPredictions = False or trainModel  # Apply the learning algorithm to decode the signals.
    plotStreamedData = False  # Graph the data to show incoming signals.
    useTherapyData = True  # Use the Therapy Data folder for any files.

    # Specify the user parameters.
    userName = "Josh".replace(" ", "")
    trialName = "HeatingPad"
    date = "2024-05-30"

    # Specify experimental parameters.
    boardSerialNum = '12ba4cb61c85ec11bc01fc2b19c2d21c'  # Board's Serial Number (port.serial_number). Only used if streaming data, else it gets reset to None.
    stopTimeStreaming = 60 * 300  # If Float/Int: The Number of Seconds to Stream Data; If String, it is the TimeStamp to Stop (Military Time) as "Hours:Minutes:Seconds:MicroSeconds"
    reanalyzeData = False  # Reanalyze training files: don't use saved features

    # ---------------------------------------------------------------------- #

    # Assert the proper use of the program
    assert sum((readDataFromExcel, streamData, trainModel)) == 1, "Only one protocol can be be executed."

    # Define helper classes.
    inputParameterClass = adjustInputParameters(plotStreamedData, streamData, readDataFromExcel, trainModel, useModelPredictions, useTherapyData)
    saveInputs = saveDataProtocols.saveExcelData()

    # Get the reading/saving information.
    numPointsPerBatch, moveDataFinger = inputParameterClass.getPlottingParams(analyzeBatches= plotStreamedData or useModelPredictions)
    collectedDataFolder, currentFilename = inputParameterClass.getSavingInformation(date, trialName, userName)

    # Compile all the protocol information.
    streamingOrder, biomarkerOrder, featureAverageWindows, featureNames, biomarkerFeatureNames = inputParameterClass.getGeneralParameters()
    performMachineLearning, modelClasses, actionControl, plotTrainingData, saveModel = inputParameterClass.getMachineLearningParams(featureNames, collectedDataFolder)
    boardSerialNum, maxVolt, adcResolution, saveRawSignals, recordQuestionnaire = inputParameterClass.getStreamingParams(boardSerialNum)
    soundInfoFile, dataFolder, playGenres = inputParameterClass.getModelParameters()
    saveRawFeatures, testSheetNum = inputParameterClass.getExcelParams()

    # Initialize instance to analyze the data
    readData = streamingProtocols.streamingProtocols(boardSerialNum, modelClasses, actionControl, numPointsPerBatch, moveDataFinger,
                                                     streamingOrder, biomarkerOrder, featureAverageWindows, plotStreamedData)

    # ---------------------------------------------------------------------- #

    # Stream in Data
    if streamData:
        if not recordQuestionnaire:
            # Stream in the data from the circuit board
            readData.streamArduinoData(maxVolt, adcResolution, stopTimeStreaming, currentFilename)
        else:
            # Stream in the data from the circuit board
            streamingThread = threading.Thread(target=readData.streamArduinoData, args=(maxVolt, adcResolution, stopTimeStreaming, currentFilename), daemon=True)
            streamingThread.start()
            # Open the questionnaire GUI.
            folderPath = "./helperFiles/surveyInformation/"
            stressQuestionnaire = stressQuestionnaireGUI(readData, folderPath)
            # When the streaming stops, close the GUI/Thread.
            stressQuestionnaire.finishedRun()
            streamingThread.join()

    # Take Data from Excel Sheet
    elif readDataFromExcel:
        # Collect the Data from Excel
        compiledRawData, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions = \
            extractDataProtocols.extractData().getData(currentFilename, numberOfChannels=len(streamingOrder), testSheetNum=testSheetNum)
        # Analyze the Data using the Correct Protocol
        readData.streamExcelData(compiledRawData, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList,
                                 surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, currentFilename)

    # Take Preprocessed (Saved) Features from Excel Sheet
    elif trainModel:
        # Initializing the training class.
        trainingInterface = trainingProtocols.trainingProtocols(biomarkerFeatureNames, streamingOrder, biomarkerOrder, len(streamingOrder), collectedDataFolder, readData)

        checkFeatureWindow_EEG = False
        if checkFeatureWindow_EEG:
            featureTimeWindows = np.arange(5, 25, 5)
            # # featureTimeWindows = [5, 30, 60, 90, 120, 150, 180]
            excelFile = collectedDataFolder + '2022-12-16 Full Dataset TV.xlsx'
            allRawFeatureTimesHolders, allRawFeatureHolders = trainingInterface.varyAnalysisParam(excelFile, featureAverageWindows, featureTimeWindows)

        # Extract the features from the training files and organize them.
        allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
            allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
            subjectOrder, experimentalOrder, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
            = trainingInterface.streamTrainingData(featureAverageWindows, plotTrainingData=plotTrainingData, reanalyzeData=reanalyzeData)
        # Assert the validity of the feature extraction
        assert len(allAlignedFeatureHolder[0][0]) == len(featureNames), "Incorrect number of compiled features extracted"
        for analysisInd in range(len(allRawFeatureHolders[0])):
            assert len(allRawFeatureHolders[0][analysisInd][0]) == len(biomarkerFeatureNames[analysisInd]), "Incorrect number of fraw eatures extracted"
        print("\nFinished Feature Extraction")

        import matplotlib.pyplot as plt
        bounds = compileModelInfo().predictionBounds

        colors = []
        currentSubjectName = ""
        subjectExperimentInds = []
        for experimentInd in range(len(experimentalOrder)):
            subjectName = subjectOrder[experimentInd]

            if (currentSubjectName != subjectName and len(subjectExperimentInds)) != 0 or experimentInd == len(experimentalOrder) - 1:
                if experimentInd == len(experimentalOrder) - 1:
                    subjectExperimentInds.append(experimentInd)
                    colors.append('#333333')

                for finalLabelInd in range(len(featureLabelTypes)):
                    finalLabel = featureLabelTypes[finalLabelInd]
                    experimentNames = [experimentalOrder[i] for i in subjectExperimentInds]

                    plt.figure(figsize=(12, 6))  # Increase the figure size for better readability
                    bar_positions = np.arange(len(experimentNames))
                    bars = plt.bar(bar_positions, [allFinalLabels[finalLabelInd][i] for i in subjectExperimentInds], color=colors)

                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

                    plt.xticks(ticks=bar_positions, labels=experimentNames, rotation=45, fontsize=12, ha='right')
                    plt.title(f'{currentSubjectName} - {finalLabel}', fontsize=16)
                    plt.xlabel("Experiment Number", fontsize=14)
                    plt.ylabel("Label Value", fontsize=14)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.ylim(bounds[finalLabelInd])
                    plt.tight_layout()
                    plt.show()

                colors = []
                subjectExperimentInds = []
            experimentName = experimentalOrder[experimentInd]
            subjectExperimentInds.append(experimentInd)
            currentSubjectName = subjectName

            colors.append('#333333')
            if 'cpt' in experimentName.lower():
                colors[-1] = 'skyblue'
            if 'heat' in experimentName.lower():
                colors[-1] = '#D62728'
        # exit()

        # Standardize data
        standardizeClass_Features = standardizeData(allFinalFeatures, threshold=0)
        standardizedFeatures = standardizeClass_Features.standardize(allFinalFeatures)
        # Standardize labels
        standardizeClass_Labels = []
        standardizedLabels = []
        scoreTransformations = []
        for modelInd in range(len(performMachineLearning.modelControl.modelClasses)):
            if modelInd == 2:
                standardizeClass_Labels.append(standardizeData(allFinalLabels[modelInd], threshold=0))
                standardizedLabels.append(standardizeClass_Labels[modelInd].standardize(allFinalLabels[modelInd]))

                scoreTransformation = np.diff(standardizedLabels[modelInd]) / np.diff(allFinalLabels[modelInd])
                scoreTransformations.append(scoreTransformation[~np.isnan(scoreTransformation)][0])
            else:
                oddLabels = allFinalLabels[modelInd]  # + (np.mod(allFinalLabels[modelInd],2)==0)
                standardizeClass_Labels.append(standardizeData(oddLabels, threshold=0))
                standardizedLabels.append(standardizeClass_Labels[modelInd].standardize(oddLabels))

                scoreTransformation = np.diff(standardizedLabels[modelInd]) / np.diff(oddLabels)
                scoreTransformations.append(scoreTransformation[~np.isnan(scoreTransformation)][0])

            # Compile information into the model class
            performMachineLearning.modelControl.modelClasses[modelInd].setStandardizationInfo(featureNames, standardizeClass_Features, standardizeClass_Labels[modelInd])
        standardizedLabels = np.array(standardizedLabels)

    # ---------------------------------------------------------------------- #
    # ------------------ Extract Data into this Namespace ------------------ #

    if streamData or readDataFromExcel:
        # Extract the data
        timePoints = np.array(readData.analysisList[0].data[0])
        eogReadings = np.array(readData.analysisProtocols['eog'].data[1][0])
        eegReadings = np.array(readData.analysisProtocols['eeg'].data[1][0])
        edaReadings = np.array(readData.analysisProtocols['eda'].data[1][0])
        tempReadings = np.array(readData.analysisProtocols['temp'].data[1][0])

        # # Extract raw features
        # eogFeatures, eegFeatures, edaFeatures, tempFeatures = readData.rawFeatureHolder
        # eogFeatureTimes, eegFeatureTimes, edaFeatureTimes, tempFeatureTimes = readData.rawFeatureTimesHolder

        # Extract the features
        alignedFeatures = np.array(readData.alignedFeatures)
        alignedFeatureTimes = np.array(readData.alignedFeatureTimes)
        alignedFeatureLabels = np.array(readData.alignedFeatureLabels)

        # Extract the feature labels.
        surveyAnswersList = np.array(readData.surveyAnswersList)  # A list of list of feature labels.
        surveyAnswerTimes = np.array(readData.surveyAnswerTimes)  # A list of times associated with each feature label.
        surveyQuestions = np.array(readData.surveyQuestions)  # A list of the survey questions asked to the user.
        # Extract the experiment information
        experimentTimes = np.array(readData.experimentTimes)
        experimentNames = np.array(readData.experimentNames)
        # Extract subject information
        subjectInformationAnswers = np.array(readData.subjectInformationAnswers)
        subjectInformationQuestions = np.array(readData.subjectInformationQuestions)

    # ---------------------------------------------------------------------- #
    # -------------------------- Save Input data --------------------------- #
    # Save the Data in Excel
    if saveRawSignals:
        # Double Check to See if User Wants to Save the Data
        # verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
        verifiedSave = "Y"
        if verifiedSave.upper() == "Y":
            # Get the streaming data
            streamingData = []
            for analysis in readData.analysisList:
                for analysisChannelInd in range(len(analysis.data[1])):
                    streamingData.append(np.array(analysis.data[1][analysisChannelInd]))
            # Initialize Class to Save the Data and Save
            saveInputs.saveData(timePoints, streamingData, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions,
                                subjectInformationAnswers, subjectInformationQuestions, streamingOrder, currentFilename)
        else:
            print("User Chose Not to Save the Data")
    elif saveRawFeatures:
        # Initialize Class to Save the Data and Save
        saveInputs.saveRawFeatures(readData.rawFeatureTimesHolder, readData.rawFeatureHolder, biomarkerFeatureNames, biomarkerOrder, experimentTimes,
                                   experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, currentFilename)

    # ----------------------------- End of Program ----------------------------- #
    # -------------------------------------------------------------------------- #

    import matplotlib.pyplot as plt

    # Replace 'path_to_arial.ttf' with the actual path to the Arial font on your system
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']  # Additional fallback option

    # Optional: Adjust other font-related settings if needed
    plt.rcParams['font.size'] = 12

    sys.exit()

    # Extract information from the streamed data
    alignedFeatures = np.asarray(readData.alignedFeatures.copy())[10:-30, :]  # signalLength, numSignals
    alignedFeatureTimes = np.asarray(readData.alignedFeatureTimes.copy())[10:-30]  # SignalLength

    # Standardize data
    standardizeClass_Features = standardizeData(alignedFeatures, axisDimension=0, threshold=0)
    standardizedFeatures = standardizeClass_Features.standardize(alignedFeatures)

    plottingFeatureNames = ["blinkDuration_EOG", "halfClosedTime_EOG",
                            "hjorthActivity_EEG", "engagementLevelEst_EEG",
                            "hjorthActivity_EDA", "firstDerivVariance_EDA",
                            "firstDerivativeMean_TEMP", "mean_TEMP"]
    shortenedNames = ["BD", "HCT", "HA", "EL", "HA", "FDV", "FDM", "M"]

    plottingColors = [
        '#3498db', '#2A4D7F',  # Blue shades
        '#9ED98F', '#38963E',  # Green shades
        '#918ae1', '#803F91',  # Purple shades
        '#fc827f', '#E63434'  # Red shades
    ]

    # plottingColors = [
    #     '#38c7e8', '#2A4D7F',  # Blue shades
    #     '#13d6b0', '#38963E',  # Green shades
    #     '#918ae1', '#803F91',  # Purple shades
    #     '#fc827f', '#E63434'   # Red shades
    # ]

    # plottingColors.reverse()
    # plottingFeatureNames.reverse()
    # shortenedNames.reverse()

    saveName = currentFilename.split("/")[-1].split(".")[0]

    plottingFeatureInds = [np.where(plottingFeatureNames[i] == featureNames)[0][0] for i in range(len(plottingFeatureNames))]
    yLim = [-3.5, 3.5]

    # fig, axes = plt.subplots(len(plottingFeatureNames), 1, figsize=(3, 6), sharex=True)
    fig, axes = plt.subplots(len(plottingFeatureNames), 1, figsize=(2, 6), sharex=True)

    for i in range(len(plottingFeatureNames)):
        featureInd = plottingFeatureInds[i]
        featureName = shortenedNames[i]
        color = plottingColors[i]

        axes[i].plot(alignedFeatureTimes, standardizedFeatures[:, featureInd], linewidth=1, color=color)
        axes[i].set_yticks([])  # Hide y-axis ticks
        # axes[i].set_ylabel(featureName)

    for i in range(len(experimentTimes)):
        for ax in axes:
            ax.axvline(experimentTimes[i][0], color='gray', linestyle='--', linewidth=0.3)
            ax.axvline(surveyAnswerTimes[i], color='gray', linestyle='--', linewidth=0.3)

            ax.fill_betweenx(np.array(yLim), experimentTimes[i][0], surveyAnswerTimes[i], color="lightblue", alpha=0.03)

    axes[-1].set_xlabel('Time')
    plt.ylim(yLim)
    plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
    plt.savefig(f"{saveName}.png", dpi=300, bbox_inches='tight')
    plt.show()

    for i in range(len(plottingFeatureNames)):
        fig, ax = plt.subplots(1, 1, figsize=(2, 1), sharex=True)

        featureInd = plottingFeatureInds[i]
        featureName = shortenedNames[i]
        color = plottingColors[i]

        ax.plot(alignedFeatureTimes, standardizedFeatures[:, featureInd], linewidth=1, color=color)
        ax.set_yticks([])  # Hide y-axis ticks
        ax.set_xticks([])  # Hide y-axis ticks
        ax.set_ylabel(featureName)

        for i in range(len(experimentTimes)):
            ax.axvline(experimentTimes[i][0], color='gray', linestyle='--', linewidth=0.3)
            ax.axvline(surveyAnswerTimes[i], color='gray', linestyle='--', linewidth=0.3)

            ax.fill_betweenx(np.array(yLim), experimentTimes[i][0], surveyAnswerTimes[i], color="lightblue", alpha=0.03)

        # ax.set_xlabel('Time')
        plt.ylim(yLim)
        plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
        plt.savefig(f"{featureName}_{color}.png", dpi=300, bbox_inches='tight')
        plt.show()

    data = np.array([eogReadings, eegReadings, edaReadings, tempReadings])
    # Standardize data
    standardizeClass_Features = standardizeData(data, axisDimension=1, threshold=0)
    standardizedFeatures = standardizeClass_Features.standardize(data)

    shortenedNames = ["EOG", "EEG", "EDA", "Temp"]
    plottingColors = [
        '#3498db',  # Blue shades
        '#9ED98F',  # Green shades
        '#918ae1',  # Purple shades
        '#fc827f',  # Red shades
    ]

    fig, axes = plt.subplots(len(shortenedNames), 1, figsize=(2, 3), sharex=True)

    for featureInd in range(len(shortenedNames)):
        featureName = shortenedNames[featureInd]
        color = plottingColors[featureInd]

        axes[featureInd].plot(timePoints, standardizedFeatures[featureInd], linewidth=1, color=color)
        axes[featureInd].set_yticks([])  # Hide y-axis ticks
        axes[featureInd].set_ylabel(featureName)

    for i in range(len(experimentTimes)):
        for ax in axes:
            ax.axvline(experimentTimes[i][0], color='gray', linestyle='--', linewidth=0.3)
            ax.axvline(surveyAnswerTimes[i], color='gray', linestyle='--', linewidth=0.3)

            ax.fill_betweenx(np.array(yLim), experimentTimes[i][0], surveyAnswerTimes[i], color="lightblue", alpha=0.03)

    axes[-1].set_xlabel('Time')
    plt.ylim(yLim)
    plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
    plt.show()
