"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Data Aquisition:
        
    Virtual Reality:
        Vizard Must be Installed (See Below). Program MUST be run in Vizard IDE
        The Python Spyder IDE does NOT have 'import viz' (at least the correct viz)
        
    Plotting:
        If Plotting, You Need an GUI Backend -> In Spyder IDE Use: %matplotlib qt5
        Some IDEs (Spyder Included) may Naturally Plot in GUI.
    --------------------------------------------------------------------------
    
    Modules to Import Before Running the Program (Some May be Missing):
        $ conda install scikit-learn
        $ conda install matplotlib
        $ conda install tensorflow
        $ conda install openpyxl
        $ conda install pyserial
        $ conda install joblib
        $ conda install numpy
        $ conda install keras
    
    Programs to Install:
        Vizard: https://www.worldviz.com/virtual-reality-software-downloads
        
    --------------------------------------------------------------------------
"""

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import sys
import numpy as np

sys.path.append('./Helper Files/Data Aquisition and Analysis/Biolectric Protocols/')  # Folder with Data Aquisition Files
import ppgAnalysis as ppgAnalysis         # Functions to Analyze the EOG Data

# Import Data Aquisition and Analysis Files
sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import excelProcessing as excelDataProtocol       # Functions to Save/Read in Data from Excel
import readDataArduino as streamDataProtocol      # Functions to Handle Data from Arduino

# Import Files for Machine Learning
sys.path.append('./Helper Files/Machine Learning/')  # Folder with Machine Learning Files
import machineLearningMain  # Class Header for All Machine Learning
import featureAnalysis      # Functions for Feature Analysis

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # Protocol Switches: Only the First True Variable Excecutes
    readDataFromExcel = False      # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    streamData = True             # Stream in Data from the Board and Analyze;
    trainModel = False              # Train Model with ALL Data in 'trainDataExcelFolder'

    # User Options During the Run: Any Number Can be True
    plotStreamedData = True      # Graph the Data to Show Incoming Signals + Analysis
    testModel = False             # Apply the Learning Algorithm to Decode the Signals
    
    # Specify the Stressors/Sensors Used in this Experiment
    listOfStressors = ['cpt', 'exercise', 'music']         # This Keyword MUST be Present in the Filename
    listOfSensors = ["eog", "ppg", "gsr"]    # A List Representing the Order of the Sensors being Streamed in.
    
    featureLabels = ["Blink", "Movement", "Wire"]
    featureLabels = ["Afternoon"]
    

    # ---------------------------------------------------------------------- #
    
    # Analyze the Data in Batches
    numTimePoints = 1000          # The Number of Data Points to Display to the User at a Time;
    moveDataFinger = 100          # The Number (+ a few) of NEW Data Points to Plot/Analyze in Each Batch;
    
    # numTimePoints = 10000          # The Number of Data Points to Display to the User at a Time;
    # moveDataFinger = 1000          # The Number (+ a few) of NEW Data Points to Plot/Analyze in Each Batch;
    
    
    # Save the Data as an Excel File (For Later Use)
    if streamData:
        # Arduino Streaming Parameters
        boardSerialNum = '12ba4cb61c85ec11bc01fc2b19c2d21c'   # Board's Serial Number (port.serial_number)
        stopTimeStreaming = 60*60*4      # If Float/Int: The Number of Seconds to Stream Data; If String, it is the TimeStamp to Stop (Military Time) as "Hours:Minutes:Seconds:MicroSeconds"
    
        saveInputData = True      # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'
        saveExcelName = "2022-07-20 Blink.xlsx"  # The Name of the Saved File
        saveDataFolder = "./Data/allSensors/Industry Electrodes/_Blink Comparison/"   # Data Folder to Save the Excel Data; MUST END IN '/'
    else:
        boardSerialNum = None
        saveInputData = False
        
    # Instead of Arduino Data, Use Test Data from Excel File
    if readDataFromExcel:
        if not plotStreamedData:
            # If not displaying, read in all the excel data (max per sheet) at once
            numTimePoints = 2048576
            moveDataFinger = 1048100 
        
        testSheetNum = 0   # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
        testDataExcelFile = "./Data/allSensors/Industry Electrodes/2022-07-19 ESP32 WiFi Afternoon/Samuel Solomon 2022-07-19 Afternoon.xlsx" # Path to the Test Data
        
    # Train or Test the Data with the Machine Learning Model
    if trainModel or testModel:
        # If trainig, read the data as quickly as possible
        numTimePoints = 2048576
        moveDataFinger = 1048100 
        
        # Pick the Machine Learning Module to Use
        modelType = "SVR"  # Machine Learning Options: NN, RF, LR, KNN, SVM, RG, EN, SVR
        supportVectorKernel = "rbf"
        modelPath = "./Machine Learning/Models/predictionModelRF_12-13-2021.pkl" # Path to Model (Creates New if it Doesn't Exist)
        # Choos the Folder to Save ML Results
        if trainModel:
            saveModel = True  # Save the Machine Learning Model for Later Use
            trainDataExcelFolder = "./Data/allSensors/Industry Electrodes/_Blink Comparison/"
            trainDataExcelFolder = "./Data/allSensors/Industry Electrodes/2022-07-19 ESP32 WiFi Afternoon/"
            saveDataFolder_ML = trainDataExcelFolder + "Data Analysis/" + modelType + "/"
        else:
            saveDataFolder_ML = None
            
        # Compile Feature Names
        featureNames = []        # Compile Feature Names
        scoreFeatureLabels = []  # Compile Stress Scores
        # Folder with Feature Name Files
        compiledFeatureNamesFolder = "./Helper Files/Machine Learning/_Compiled Feature Names/All Features/"

        if 'ppsg' in listOfSensors:
            # Specify the Paths to the Pulse Feature Names
            pulseFeaturesFile_StressLevel = compiledFeatureNamesFolder + "ppgFeatureNames.txt"
            #pulseFeaturesFile_SignalIncrease = compiledFeatureNamesFolder + "pulseFeatureNames_SignalIncrease.txt"
            # Extract the Pulse Feature Names we are Using
            pulseFeatureNames_StressLevel = excelDataProtocol.getExcelData().extractFeatureNames(pulseFeaturesFile_StressLevel, prependedString = "pulseFeatures.extend([", appendToName = "_StressLevel")[1:]
            #pulseFeatureNames_SignalIncrease = excelDataProtocol.getExcelData().extractFeatureNames(pulseFeaturesFile_SignalIncrease, prependedString = "pulseFeatures.extend([", appendToName = "_SignalIncrease")[1:]
            # Combine all the Features
            pulseFeatureNames = []
            pulseFeatureNames.extend(pulseFeatureNames_StressLevel)
            #pulseFeatureNames.extend(pulseFeatureNames_SignalIncrease)
            # Get Pulse Names Without Anything Appended
            #pulseFeatureNamesFull = excelDataProtocol.getExcelData().extractFeatureNames(pulseFeaturesFile_SignalIncrease, prependedString = "pulseFeatures.extend([", appendToName = "")
            pulseFeatureNamesFull = excelDataProtocol.getExcelData().extractFeatureNames(pulseFeaturesFile_StressLevel, prependedString = "pulseFeatures.extend([", appendToName = "")
            # Create Data Structure to Hold the Features
            pulseFeatures = []
            pulseFeatureLabels = []  
            featureNames.extend(pulseFeatureNames)
            
        if 'eog' in listOfSensors:
            # Specify the Paths to the EOG Feature Names
            eogFeaturesFile_StressLevel = compiledFeatureNamesFolder + "eogFeatureNames.txt"
            # Extract the EOG Feature Names we are Using
            eogFeatureNames_StressLevel = excelDataProtocol.getExcelData().extractFeatureNames(eogFeaturesFile_StressLevel, prependedString = "peakFeatures.extend([", appendToName = "_StressLevel")
            #eogFeatureNames_SignalIncrease = excelDataProtocol.getExcelData().extractFeatureNames(eogFeaturesFile_SignalIncrease, prependedString = "eogFeatures.extend([", appendToName = "_SignalIncrease")
            # Combine all the Features
            eogFeatureNames = []
            eogFeatureNames.extend(eogFeatureNames_StressLevel)
            #eogFeatureNames.extend(eogFeatureNames_SignalIncrease)
            # Get EOG Names Without Anything Appended
            #eogFeatureNamesFull = excelDataProtocol.getExcelData().extractFeatureNames(eogFeaturesFile_SignalIncrease, prependedString = "eogFeatures.extend([", appendToName = "")
            eogFeatureNamesFull = excelDataProtocol.getExcelData().extractFeatureNames(eogFeaturesFile_StressLevel, prependedString = "peakFeatures.extend([", appendToName = "")
            # Create Data Structure to Hold the Features
            eogFeatures = []
            eogFeatureLabels = []  
            featureNames.extend(eogFeatureNames)
            
        # Get the Machine Learning Module
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureNames), machineLearningClasses = listOfStressors, saveDataFolder = saveDataFolder_ML, supportVectorKernel = supportVectorKernel)
        predictionModel = performMachineLearning.predictionModel
    else:
        predictionModel = None

    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #           Data Collection Program (Should Not Have to Edit)            #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # Initialize instance to analyze the data
    readData = streamDataProtocol.mainArduinoRead(boardSerialNum, numTimePoints, moveDataFinger, listOfSensors, plotStreamedData, guiApp = None)

    # Stream in Data from Arduino
    if streamData:
        readData.streamArduinoData(stopTimeStreaming, predictionModel = None, actionControl = None, calibrateModel = False)
    # Take Data from Excel Sheet
    elif readDataFromExcel:
        # Collect the Data from Excel
        compiledRawData = excelDataProtocol.getExcelData().getData(testDataExcelFile, numberOfChannels = len(listOfSensors), testSheetNum = testSheetNum)
        # Analyze the Data using the Correct Protocol
        readData.streamExcelData(compiledRawData, predictionModel = None, actionControl = None, calibrateModel = False)
    # Take Preprocessed (Saved) Features from Excel Sheet
    elif trainModel:
        # Extract the Data
        signalDataFull, signalLabels = excelDataProtocol.getExcelData().streamTrainingData(trainDataExcelFolder, featureLabels, len(listOfSensors), readData)
        signalData = np.array(signalDataFull)[:,1:]; signalLabels = np.array(signalLabels)
        print("\nCollected Signal Data")

        analyzeFeatures = True
        if analyzeFeatures:
            timePoints = np.array(signalDataFull)[:,0]
            blinkFeatures = np.array(eogFeatureNamesFull[1:])

            analyzeFeatures = featureAnalysis.featureAnalysis(blinkFeatures, [], trainDataExcelFolder + "Data Analysis/Feature Analysis/")
            #analyzeFeatures.correlationMatrix(signalData, folderName = "correlationMatrix/")
            analyzeFeatures.singleFeatureAnalysis(timePoints, signalData, averageIntervalList = [60, 2*60, 3*60], folderName = "singleFeatureAnalysis - Full/")
            analyzeFeatures.featureDistribution(signalData, signalLabels, featureLabels, folderName = "Feature Distribution/")
            # plt.hist(signalData[:, (blinkFeatures == "closingTime_Peak").argmax()][signalLabels == 2])
        
        sys.exit()
        # Train the Data on the Gestures
        performMachineLearning.trainModel(signalData, signalLabels, blinkFeatures)
        # Save Signals and Labels
        if saveData and performMachineLearning.map2D:
            saveInputs = excelData.saveExcel(numChannels)
            saveExcelNameMap = Path(saveExcelName).stem + "_mapedData.xlsx" #"Signal Features with Predicted and True Labels New.xlsx"
            saveInputs.saveLabeledPoints(performMachineLearning.map2D, signalLabels,  performMachineLearning.predictionModel.predictData(signalData), saveDataFolder, saveExcelNameMap, sheetName = "Signal Data and Labels")
        # Save the Neural Network (The Weights of Each Edge)
        if saveModel:
            modelPathFolder = os.path.dirname(modelPath)
            os.makedirs(modelPathFolder, exist_ok=True)
            performMachineLearning.predictionModel.saveModel(modelPath)

    # ---------------------------------------------------------------------- #
    # ------------------ Extract Data into this Namespace ------------------ #

    # Extract the EOG Data
    timeEOG = np.array(readData.eogAnalysis.data[0])
    eogVertical = np.array(readData.eogAnalysis.data[1][0])
    # Extract the PPG Data
    timePPG = np.array(readData.ppgAnalysis.data[0])
    ppgData = np.array(readData.ppgAnalysis.data[1][0])
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Save Input data --------------------------- #
    # Save the Data in Excel: PPG Channels (Cols 1-4); X-Peaks (Cols 5-8); Peak Features (Cols 9-12)
    if saveInputData:
        # Double Check to See if User Wants to Save the Data
        verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
        if verifiedSave.upper() == "Y":
            # Get Data on Same Time Axis
            fullDataPPG = []; ppgInd = 0
            for timePointInd in range(len(timeEOG)):
                timePoint = timeEOG[timePointInd]
                
                if ppgInd == len(timePPG) or timePPG[ppgInd] != timePoint:
                    fullDataPPG.append(0)
                else:
                    fullDataPPG.append(ppgData[ppgInd])
                    ppgInd += 1;
            fullDataSave = [eogVertical, fullDataPPG, eogVertical]
            
            # Initialize Class to Save the Data and Save
            saveInputs = excelDataProtocol.saveExcelData()
            saveInputs.saveData(timeEOG, fullDataSave, [], listOfSensors, saveDataFolder, saveExcelName, "Trial 1 - All Sensors")
        else:
            print("User Chose Not to Save the Data")
    

    sys.exit()



import matplotlib.pyplot as plt
timeEOG = np.array(readData.eogAnalysis.data[0])

fig = plt.figure();
ax = fig.add_axes([.1,.1,0.7,0.7]);

ax.plot(timeEOG[1:]/60, np.diff(timeEOG),'ko', label="delay_1ms");

plt.ylim(0,0.01)

plt.title("ESP-NOW Delay");
plt.ylabel("Delay Between Points (Sec)");
plt.xlabel("Minutes")
plt.legend(loc = "upper right")


timeEOG = np.array(readData.eogAnalysis.data[0])
eogVertical = np.array(readData.eogAnalysis.data[1][0])
eogHorizontal = np.array(readData.eogAnalysis.data[1][1])
# Extract the PPG Data
timePPG = np.array(readData.ppgAnalysis.data[0])
redLED = np.array(readData.ppgAnalysis.data[1][0])

plt.plot(timeEOG, eogVertical, 'tab:blue', alpha=0.2);
plt.plot(timeEOG, eogHorizontal, 'tab:red', alpha=0.2);
plt.plot(timeEOG, redLED, 'tab:green', alpha=0.2);

plt.hlines(np.average(eogVertical), timeEOG[0], timeEOG[-1], 'tab:blue', label="Channel 1");
plt.hlines(np.average(eogHorizontal), timeEOG[0], timeEOG[-1], 'tab:red', label="Channel 2");
plt.hlines(np.average(redLED), timeEOG[0], timeEOG[-1], 'tab:green', label="Channel 3");
plt.hlines(1.6, timeEOG[0], timeEOG[-1], 'k', label="Supplied Voltage");

plt.title("ADC Accuracy")
plt.xlabel("Seconds")
plt.ylabel("Voltage")
plt.legend(loc="upper right")


