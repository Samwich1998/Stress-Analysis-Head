from compiledFeatureNames.compileFeatureNames import compileFeatureNames


class adjustInputParameters:

    def __init__(self, plotStreamedData=True, streamData=False, readDataFromExcel=False, trainModel=False, useModelPredictions=False, useTherapyData=False):
        # Set the parameters for the program.
        self.useModelPredictions = useModelPredictions  # Use the Machine Learning Model for Predictions
        self.readDataFromExcel = readDataFromExcel  # Read Data from an Excel File
        self.plotStreamedData = plotStreamedData  # Plot the Streamed Data in Real-Time
        self.useTherapyData = useTherapyData  # Use the Therapy Data for the Machine Learning Model
        self.streamData = streamData  # Stream Data from the Arduino
        self.trainModel = trainModel  # Train the Machine Learning Model

        # Specify flags when not streaming
        self.boardSerialNum, self.maxVolt, self.adcResolution, self.stopTimeStreaming = None, None, None, None
        self.saveRawSignals, self.recordQuestionnaire = False, False

    def resetParameters(self):
        # Specify flags when not streaming
        self.boardSerialNum, self.maxVolt, self.adcResolution, self.stopTimeStreaming = None, None, None, None
        self.saveRawSignals, self.recordQuestionnaire = False, False

    def getStreamingParams(self, boardSerialNum='12ba4cb61c85ec11bc01fc2b19c2d21c'):
        # Assert that you are using this protocol.
        assert self.streamData, "You must be reading data from an Excel file."

        # Arduino Streaming Parameters.
        boardSerialNum = boardSerialNum  # Board's Serial Number (port.serial_number)
        stopTimeStreaming = 60 * 300  # If Float/Int: The Number of Seconds to Stream Data; If String, it is the TimeStamp to Stop (Military Time) as "Hours:Minutes:Seconds:MicroSeconds"
        adcResolution = 4096
        maxVolt = 3.3

        # Streaming flags.
        recordQuestionnaire = not self.plotStreamedData  # Only use one GUI: questionnaire or streaming
        saveRawSignals = True  # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'

        return maxVolt, adcResolution, stopTimeStreaming, saveRawSignals, recordQuestionnaire

    @staticmethod
    def getPlottingParams(displayAllData=False):
        # Analyze the data in batches.
        numPointsPerBatch = 4000  # The Number of Data Points to Display to the User at a Time.
        moveDataFinger = 400  # The Minimum Number of NEW Data Points to Plot/Analyze in Each Batch;

        if displayAllData:
            # If displaying all data, read in all the Excel data (max per sheet) at once
            numPointsPerBatch = 2048576
            moveDataFinger = 1048100

        return numPointsPerBatch, moveDataFinger

    def getExcelParams(self):
        # Assert that you are using this protocol.
        assert self.readDataFromExcel, "You must be reading data from an Excel file."

        # Specify the Excel Parameters.
        numPointsPerBatch, moveDataFinger = self.getPlottingParams(displayAllData=not self.plotStreamedData)
        testSheetNum = 0  # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
        saveRawFeatures = False  # Save the Raw Features to an Excel File

        return saveRawFeatures, numPointsPerBatch, moveDataFinger, testSheetNum

    # ---------------------------------------------------------------------- #



    # Specify biomarker information.
    streamingOrder = ["eog", "eeg", "eda", "temp"]  # A List Representing the Order of the Sensors being Streamed in.
    extractFeaturesFrom = ["eog", "eeg", "eda", "temp"]  # ["eog", "eeg", "eda", "temp"] # A list with all the biomarkers from streamingOrder for feature extraction
    allAverageIntervals = [60, 30, 30, 30]  # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60  Old: [120, 75, 90, 45]

    # Compile feature names
    featureNames, biomarkerFeatureNames, biomarkerOrder = compileFeatureNames().extractFeatureNames(extractFeaturesFrom)

    # Specify the path to the collected data.
    collectedDataFolder = compileModelInfo.getTrainingDataFolder(useTherapyData)
    currentFilename = collectedDataFolder + f"{date} {trialName} Trial {userName}.xlsx"

    featureAverageWindows = []
    # Compile feature average windows.
    for biomarker in biomarkerOrder:
        featureAverageWindows.append(allAverageIntervals[streamingOrder.index(biomarker)])




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
        else:
            plotTrainingData, reanalyzeData, saveModel = False, False, False

        # Get the Machine Learning Module
        performMachineLearning = machineLearningInterface.machineLearningHead(modelTypes, modelFile, featureNames, collectedDataFolder)
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
