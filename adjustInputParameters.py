# Import helper files.
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from helperFiles.machineLearning.machineLearningInterface import machineLearningInterface


class adjustInputParameters:

    def __init__(self, plotStreamedData=True, streamData=False, readDataFromExcel=False, trainModel=False, useModelPredictions=False, useTherapyData=False):
        # Set the parameters for the program.
        self.useModelPredictions = useModelPredictions or trainModel  # Use the Machine Learning Model for Predictions
        self.readDataFromExcel = readDataFromExcel  # Read Data from an Excel File
        self.plotStreamedData = plotStreamedData  # Plot the Streamed Data in Real-Time
        self.useTherapyData = useTherapyData  # Use the Therapy Data for the Machine Learning Model
        self.streamData = streamData  # Stream Data from the Arduino

    def getGeneralParameters(self):
        # Specify biomarker information.
        streamingOrder = ["eog", "eeg", "eda", "temp"]  # A List Representing the Order of the Sensors being Streamed in: ["eog", "eeg", "eda", "temp"]
        extractFeaturesFrom = streamingOrder if self.useModelPredictions else []  # A list with all the biomarkers from streamingOrder for feature extraction
        allAverageIntervals = [60, 30, 30, 30]  # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60  Old: [120, 75, 90, 45]

        # Compile feature names
        featureNames, biomarkerFeatureNames, biomarkerOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

        featureAverageWindows = []
        # Compile feature average windows.
        for biomarker in biomarkerOrder:
            featureAverageWindows.append(allAverageIntervals[streamingOrder.index(biomarker)])

        return streamingOrder, biomarkerOrder, featureAverageWindows, featureNames, biomarkerFeatureNames

    def getSavingInformation(self, date, trialName, userName):
        # Specify the path to the collected data.
        collectedDataFolder = compileModelInfo.getTrainingDataFolder(self.useTherapyData)
        currentFilename = collectedDataFolder + f"{date} {trialName} Trial {userName}.xlsx"

        return collectedDataFolder, currentFilename

    def getStreamingParams(self, boardSerialNum):
        # Assert that you are using this protocol.
        if not self.streamData:
            return None, None, None, None, None
        print("\tSetting streaming parameters.")

        # Arduino Streaming Parameters.
        adcResolution = 4096
        maxVolt = 3.3

        # Streaming flags.
        recordQuestionnaire = not self.plotStreamedData  # Only use one GUI: questionnaire or streaming
        saveRawSignals = True  # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'

        return boardSerialNum, maxVolt, adcResolution, saveRawSignals, recordQuestionnaire

    @staticmethod
    def getPlottingParams(analyzeBatches=False):
        # Analyze the data in batches.
        numPointsPerBatch = 4000  # The Number of Data Points to Display to the User at a Time.
        moveDataFinger = 400  # The Minimum Number of NEW Data Points to Plot/Analyze in Each Batch;

        if analyzeBatches:
            # If displaying all data, read in all the Excel data (max per sheet) at once
            numPointsPerBatch = 2048576
            moveDataFinger = 1048100

        return numPointsPerBatch, moveDataFinger

    def getExcelParams(self):
        # Assert that you are using this protocol.
        if not self.readDataFromExcel:
            return False, None
        print("\tSetting reading parameters.")

        # Specify the Excel Parameters.
        saveRawFeatures = False  # Save the Raw Features to an Excel File
        testSheetNum = 0  # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document

        return saveRawFeatures, testSheetNum

    def getMachineLearningParams(self, featureNames, collectedDataFolder):
        # Train or test the machine learning modules
        if not self.useModelPredictions:
            return None, None, None, None, None

        print("\tSetting model parameters.")

        # Specify the Machine Learning Parameters
        plotTrainingData = True  # Plot all training information
        actionControl = None  # NOT IMPLEMENTED YET
        # If training, read the data as quickly as possible

        # Specify the machine learning information
        modelFile = "predictionModel.pkl"  # Path to Model (Creates New if it Doesn't Exist)
        modelTypes = ["MF", "MF", "MF"]  # Model Options: linReg, logReg, ridgeReg, elasticNet, SVR_linear, SVR_poly, SVR_rbf, SVR_sigmoid, SVR_precomputed, SVC_linear, SVC_poly, SVC_rbf, SVC_sigmoid, SVC_precomputed, KNN, RF, ADA, XGB, XGB_Reg, lightGBM_Reg

        # Choose the Folder to Save ML Results
        saveModel = not self.useModelPredictions  # Save the Machine Learning Model for Later Use

        # Get the Machine Learning Module
        performMachineLearning = machineLearningInterface(modelTypes, modelFile, featureNames, collectedDataFolder)
        modelClasses = performMachineLearning.modelControl.modelClasses

        return performMachineLearning, modelClasses, actionControl, plotTrainingData, saveModel

    def getModelParameters(self):
        # Train or test the machine learning modules
        if not self.useModelPredictions:
            return None, None, None
        print("\tSetting model parameters.")

        # Specify the MTG-Jamendo dataset path
        soundInfoFile = 'raw_30s_cleantags_50artists.tsv'
        dataFolder = './helperFiles/machineLearning/_Feedback Control/Music Therapy/Organized Sounds/MTG-Jamendo/'
        # Initialize the classes
        # soundManager = musicTherapy.soundController(dataFolder, soundInfoFile) # Controls the music playing
        # soundManager.loadSound(soundManager.soundInfo[0][3])
        playGenres = [None, 'pop', 'jazz', 'heavymetal', 'classical', None]
        # playGenres = [None, 'hiphop', 'blues', 'disco', 'ethno', None]
        # playGenres = [None, 'funk', 'reggae', 'rap', 'classicrock', None]

        # playGenres = [None, 'hiphop', 'blues', 'hardrock', 'african', None]
        # soundManager.pickSoundFromGenres(playGenres)

        return soundInfoFile, dataFolder, playGenres
