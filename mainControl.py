""" Written by Samuel Solomon: https://scholar.google.com/citations?user=9oq12oMAAAAJ&hl=en """

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import threading
import numpy as np

# Import interfaces for reading/writing data
from helperFiles.dataAcquisitionAndAnalysis.excelProcessing import extractDataProtocols, saveDataProtocols
from helperFiles.dataAcquisitionAndAnalysis import streamingProtocols

# Import interface for extracting feature names
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames

# Import files for machine learning
from helperFiles.machineLearning import machineLearningInterface, trainingProtocols

# Import interface for the data
from helperFiles.machineLearning.dataInterface.dataPreparation import standardizeData

# Import file for GUI control
from helperFiles.surveyInformation.questionaireGUI import stressQuestionnaireGUI

# Import file for music therapy
# from helperFiles.machineLearning.feedbackControl. import musicTherapy

# Add the directory of the current file to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # Protocol switches: only the first true variable executes.
    readDataFromExcel = False  # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    streamData = True  # Stream in Data from the Board and Analyze.
    trainModel = False  # Train Model with ALL Data in 'trainingFolder'.
    metaTrainModel = False

    # User options during the run: any number can be true.
    plotStreamedData = False  # Graph the Data to Show Incoming Signals + Analysis.
    useModelPredictions = False  # Apply the Learning Algorithm to Decode the Signals.

    # ---------------------------------------------------------------------- #

    # Analyze the data in batches.
    numPointsPerBatch = 4000  # The Number of Data Points to Display to the User at a Time.
    moveDataFinger = 400  # The Minimum Number of NEW Data Points to Plot/Analyze in Each Batch;

    # Specify biomarker information.
    streamingOrder = ["eog", "eeg", "eda", "temp"]  # A List Representing the Order of the Sensors being Streamed in.
    extractFeaturesFrom = []  # "eog", "eeg", "eda", "temp"] # A list with all the biomarkers from streamingOrder for feature extraction
    allAverageIntervals = [60, 30, 30, 30]  # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60  Old: [120, 75, 90, 45]

    # Compile feature names
    featureNames, biomarkerFeatureNames, biomarkerOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

    featureAverageWindows = []
    # Compile feature average windows.
    for biomarker in biomarkerOrder:
        featureAverageWindows.append(allAverageIntervals[streamingOrder.index(biomarker)])

    # Stream in real-time incoming data.
    if streamData:
        # Arduino Streaming Parameters
        # boardSerialNum = '12ba4cb61c85ec11bc01fc2b19c2d21c'   # Board's Serial Number (port.serial_number)
        boardSerialNum = '12ba4cb61c85ec11bc01fc2b19c2d21c'  # Board's Serial Number (port.serial_number)
        stopTimeStreaming = 60 * 300  # If Float/Int: The Number of Seconds to Stream Data; If String, it is the TimeStamp to Stop (Military Time) as "Hours:Minutes:Seconds:MicroSeconds"
        adcResolution = 4096
        maxVolt = 3.3

        # Streaming flags
        saveRawSignals = True  # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'
        recordQuestionnaire = not plotStreamedData  # Only use one GUI: questionnaire or streaming

        # Save streaming data information as excel fileY
        saveExcelPath = "./_experimentalData/allSensors/_finalDataset/2024-05-16 HeatingPad Trial Ruixiao.xlsx"
        # saveExcelPath = "./_Supplementary/Blink Identification/_experimentalData/2023-05-22 Watching Blink 3.xlsx"
        # Extract filename
        fileName = os.path.basename(saveExcelPath).split(".")[0]
    else:
        # Specify flags when not streaming
        boardSerialNum, saveExcelPath, maxVolt, adcResolution, stopTimeStreaming = None, None, None, None, None
        saveRawSignals, recordQuestionnaire = False, False

    # Stream in excel data
    if readDataFromExcel:
        saveRawFeatures = False
        if not plotStreamedData:
            # If not displaying, read in all the excel data (max per sheet) at once
            numPointsPerBatch = 2048576
            moveDataFinger = 1048100

        # Specify the input file to analyze
        testSheetNum = 0  # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
        testDataExcelFile = "./_experimentalData/allSensors/_finalDataset/2024-03-18 HeatingPad Trial Shukun.xlsx"
    else:
        testDataExcelFile = None
        saveRawFeatures = False

    if saveRawSignals or saveRawFeatures:
        saveInputs = saveDataProtocols.saveExcelData()

    # Train or test the machine learning modules
    if trainModel or useModelPredictions:
        # ML Flags
        actionControl = None  # NOT IMPLEMENTED YET
        reanalyzeData = False  # Reanalyze training files: don't use saved features
        plotTrainingData = True  # Plot all training information
        # If training, read the data as quickly as possible

        # Specify the machine learning information
        modelFile = "predictionModel.pkl"  # Path to Model (Creates New if it Doesn't Exist)
        modelTypes = ["MF", "MF", "MF"]  # Model Options: linReg, logReg, ridgeReg, elasticNet, SVR_linear, SVR_poly, SVR_rbf, SVR_sigmoid, SVR_precomputed, SVC_linear, SVC_poly, SVC_rbf, SVC_sigmoid, SVC_precomputed, KNN, RF, ADA, XGB, XGB_Reg, lightGBM_Reg
        # Choose the Folder to Save ML Results
        if trainModel:
            # If not streaming real-time
            numPointsPerBatch = 2048576
            moveDataFinger = 1048100

            saveModel = True  # Save the Machine Learning Model for Later Use
            trainingFolder = "./_experimentalData/allSensors/_finalDataset/"  # Data Folder to Save the Excel Data; MUST END IN '/'
        else:
            saveModel = False
            trainingFolder = None

        # Get the Machine Learning Module
        performMachineLearning = machineLearningInterface.machineLearningHead(modelTypes, modelFile, featureNames, trainingFolder)
        modelClasses = performMachineLearning.modelControl.modelClasses
    else:
        actionControl, performMachineLearning = None, None
        modelClasses = []

    if True or useModelPredictions:
        # Specify the MTG-Jamendo dataset path
        soundInfoFile = 'raw_30s_cleantags_50artists.tsv'
        dataFolder = './helperFiles/machineLearning/_Feedback Control/Music Therapy/Organized Sounds/MTG-Jamendo/'
        # Initialize the classes
        # soundManager = musicTherapy.soundController(dataFolder, soundInfoFile)  # Controls the music playing
        # soundManager.loadSound(soundManager.soundInfo[0][3])
        playGenres = [None, 'pop', 'jazz', 'heavymetal', 'classical', None]
        # playGenres = [None, 'hiphop', 'blues', 'disco', 'ethno', None]
        # playGenres = [None, 'funk', 'reggae', 'rap', 'classicrock', None]

        # playGenres = [None, 'hiphop', 'blues', 'hardrock', 'african', None]
        # soundManager.pickSoundFromGenres(playGenres)
    # sys.exit()

    # Assert the proper use of the program
    assert sum((readDataFromExcel, streamData, trainModel)) == 1, "Only one protocol can be be executed."

    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #           Data Collection Program (Should Not Have to Edit)            #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # Initialize instance to analyze the data
    readData = streamingProtocols.streamingProtocols(boardSerialNum, modelClasses, actionControl, numPointsPerBatch, moveDataFinger,
                                                     streamingOrder, biomarkerOrder, featureAverageWindows, plotStreamedData)

    # Stream in Data
    if streamData:
        if not recordQuestionnaire:
            # Stream in the data from the circuit board
            readData.streamArduinoData(maxVolt, adcResolution, stopTimeStreaming, saveExcelPath)
        else:
            # Stream in the data from the circuit board
            streamingThread = threading.Thread(target=readData.streamArduinoData, args=(maxVolt, adcResolution, stopTimeStreaming, saveExcelPath), daemon=True)
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
            extractDataProtocols.extractData().getData(testDataExcelFile, numberOfChannels=len(streamingOrder), testSheetNum=testSheetNum)
        # Analyze the Data using the Correct Protocol
        readData.streamExcelData(compiledRawData, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList,
                                 surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, testDataExcelFile)

    # Take Preprocessed (Saved) Features from Excel Sheet
    elif trainModel:
        trainingInterface = trainingProtocols.trainingProtocols(biomarkerFeatureNames, streamingOrder, biomarkerOrder, len(streamingOrder), trainingFolder, readData)

        checkFeatureWindow_EEG = False
        if checkFeatureWindow_EEG:
            featureTimeWindows = np.arange(5, 25, 5)
            # # featureTimeWindows = [5, 30, 60, 90, 120, 150, 180]
            excelFile = trainingFolder + '2022-12-16 Full Dataset TV.xlsx'
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

        # Standardize data
        standardizeClass_Features = standardizeData(allFinalFeatures, threshold=0)
        standardizedFeatures = standardizeClass_Features.standardize(allFinalFeatures)
        # Standardize labels
        standardizeClass_Labels = [];
        standardizedLabels = [];
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
        verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
        if verifiedSave.upper() == "Y":
            # Get the streaming data
            streamingData = []
            for analysis in readData.analysisList:
                for analysisChannelInd in range(len(analysis.data[1])):
                    streamingData.append(np.array(analysis.data[1][analysisChannelInd]))
            # Initialize Class to Save the Data and Save
            saveInputs.saveData(timePoints, streamingData, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions,
                                subjectInformationAnswers, subjectInformationQuestions, streamingOrder, saveExcelPath)
        else:
            print("User Chose Not to Save the Data")
    if saveRawFeatures:
        # Initialize Class to Save the Data and Save
        saveInputs.saveRawFeatures(readData.rawFeatureTimesHolder, readData.rawFeatureHolder, biomarkerFeatureNames, biomarkerOrder, experimentTimes,
                                   experimentNames, surveyAnswerTimes, surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, testDataExcelFile)

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

    saveName = testDataExcelFile.split("/")[-1].split(".")[0]

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
