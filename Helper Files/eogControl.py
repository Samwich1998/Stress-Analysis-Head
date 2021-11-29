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
        $ conda install -c conda-forge tensorflow
        $ conda install scikit-learn
        $ conda install matplotlib
        $ conda install openpyxl
        $ conda install pyserial
        $ conda install joblib
        $ conda install numpy
        $ conda install keras
    
    Programs to Install:
        Vizard: https://www.worldviz.com/virtual-reality-software-downloads
        
    --------------------------------------------------------------------------
"""
# Use '%matplotlib qt' to View Plot

# Basic Modules
import sys
import threading
import numpy as np
from pathlib import Path
# Import Data Aquisition and Analysis Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import readDataExcel as excelData         # Functions to Save/Read in Data from Excel
import readDataArduino as streamData      # Functions to Read in Data from Arduino
import eogAnalysis as eogAnalysis         # Functions to Analyze the EOG Data
# Import Virtual Reality Control Files
sys.path.append('./Execute Movements/')   # Folder with Virtual Reality Control Files

# Import Files for Machine Learning
sys.path.append('./Machine Learning/')  # Folder with Machine Learning Files
import machineLearningMain  # Class Header for All Machine Learning


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # General Data Collection Information (You Will Likely Not Edit These)
    eogSerialNum = '85035323234351D06052'#'85035323234351D06052'   # Arduino's Serial Number (port.serial_number)
    samplingFreq = None           # The Average Number of Points Steamed Into the Arduino Per Second; If NONE Given, Algorithm will Calculate Based on Initial Data
    numDataPoints = 40000         # The Number of Points to Stream into the Arduino
    numTimePoints = 20000          # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    moveDataFinger = 10000          # The Number of Data Points to Plot/Analyze at a Time; My Beta-Test Used 200 Points with Plotting; 10 Points Without
    numChannels = 2               # The Number of Arduino Channels with EOG Signals Read in; My Beta-Test Used 4 Channels
    # Specify the Type of Movements to Learn
    numFeatures = 10              # The Number of Features to Extract/Save/Train on
    gestureClasses = np.char.lower(['Spontaneous', 'Reflex', 'Voluntary', 'Double'])  # Define Labels as Array
    gestureClasses = np.char.lower(['Up', 'Down', 'Blink', 'Double Blink', 'Random'])  # Define Labels as Array
    gestureClasses = np.char.lower(['Blink', 'No Blink'])  # Define Labels as Array

    # Protocol Switches: Only the First True Variable Excecutes
    streamArduinoData = False      # Stream in Data from the Arduino and Analyze; Input 'controlVR' = True to Move VR
    readDataFromExcel = True       # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    trainModel = False             # Read in ALL Data Under 'neuralNetworkFolder', and Train the Data
    
    # User Options During the Run: Any Number Can be True
    plotStreamedData = False      # Graph the Data to Show Incoming Signals + Analysis
    calibrateModel = False         # Calibrate the EOG Voltage to Predict the Eye's Angle
    saveData = False         # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'
    testModel = False          # Apply the Learning Algorithm to Decode the Signals
    controlVR = False             # Apply the Algorithm to Control the Virtual Reality View

    
    # ---------------------------------------------------------------------- #
    
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveData:
        saveExcelName = "Samuel Solomon 2021-11-05 Movements.xlsx"  # The Name of the Saved File
        saveDataFolder = "../Data/EOG Data/All Data/Industry Electrodes/"   # Data Folder to Save the Excel Data; MUST END IN '/'
        # Speficy the eye Movement You Will Perform
        eyeMovement = "Random".lower() # Make Sure it is Lowercase
        if eyeMovement not in gestureClasses:
            print("The Gesture", "'" + eyeMovement + "'", "is Not in", gestureClasses)
            
    # Instead of Arduino Data, Use Test Data from Excel File
    if readDataFromExcel:
        testDataExcelFile = "../Data/EOG Data/All Data/Industry Electrodes/Samuel Solomon 2021-11-05 Movements.xlsx" # Path to the Test Data
        testSheetNum = 0   # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
    # Input Training Paramaters 
    if trainModel:
        saveModel = False   # Save the Machine Learning Model for Later Use
        trainDataExcelFolder = "../Input Data/EMG Data/"  # Path to the Training Data Folder; All .xlsx Data Used
    # Train or Test the Data with the Machine Learning Model
    if trainModel or testModel:
        # Pick the Machine Learning Module to Use
        modelType = "KNN"  # Machine Learning Options: NN, RF, LR, KNN, SVM
        modelPath = "./Machine Learning Modules/Models/predictionModelKNNFull_SamArm1.pkl" # Path to Model (Creates New if it Doesn't Exist)
        # Get the Machine Learning Module
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, dataDim = numChannels, gestureClasses = gestureClasses)
        predictionModel = performMachineLearning.predictionModel
    else:
        predictionModel = None
    
    if controlVR:
        # Import the VR File (MUST BE RUNNING INSIDE VIZARD!)
        import virtualRealityControl as vizardControl
        # Specify the VR File and Create the VR World
        virtualFile = "./Execute Movements/Virtual Reality Files/piazza.osgb"
        gazeControl = vizardControl.gazeControl(virtualFile)
    else:
        gazeControl = None
            
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #           Data Collection Program (Should Not Have to Edit)            #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    eogProtocol = eogAnalysis.eogProtocol(numTimePoints, moveDataFinger, numChannels, samplingFreq, plotStreamedData)
    readData = excelData.readExcel(eogProtocol)
    readData.streamExcelData(testDataExcelFile, plotStreamedData, testSheetNum, predictionModel = predictionModel, actionControl = None)
    sys.exit()
    def executeProtocol():
        # Stream in Data from Arduino
        if streamArduinoData:
            arduinoRead = streamData.arduinoRead(eogSerialNum = eogSerialNum, ppgSerialNum = None, emgSerialNum = None, eegSerialNum = None, handSerialNum = None)
            readData = streamData.eogArduinoRead(arduinoRead, numTimePoints, moveDataFinger, numChannels, samplingFreq, plotStreamedData, guiApp = None)
            readData.streamEOGData(numDataPoints, predictionModel = predictionModel, actionControl = gazeControl, calibrateModel = calibrateModel)
        # Take Data from Excel Sheet
        elif readDataFromExcel:
            eogProtocol = eogAnalysis.eogProtocol(numTimePoints, moveDataFinger, numChannels, samplingFreq, plotStreamedData)
            readData = excelData.readExcel(eogProtocol)
            readData.streamExcelData(testDataExcelFile, plotStreamedData, testSheetNum, predictionModel = predictionModel, actionControl = None)
        # Take Preprocessed (Saved) Features from Excel Sheet
        elif trainModel:
            # Extract the Data
            readData = excelData.readExcel(eogProtocol)
            signalData, signalLabels = readData.getTrainingData(trainDataExcelFolder, numFeatures, gestureClasses, mode='Train')
            print("\nCollected Signal Data")
            # Train the Data on the Gestures
            performMachineLearning.trainModel(signalData, signalLabels)
            # Save Signals and Labels
            if saveData and performMachineLearning.map2D:
                saveInputs = excelData.saveExcel(numChannels, numFeatures)
                saveExcelNameMap = Path(saveExcelName).stem + "_mapedData.xlsx" #"Signal Features with Predicted and True Labels New.xlsx"
                saveInputs.saveLabeledPoints(performMachineLearning.map2D, signalLabels,  performMachineLearning.predictionModel.predictData(signalData), saveDataFolder, saveExcelNameMap, sheetName = "Signal Data and Labels")
            # Save the Neural Network (The Weights of Each Edge)
            if saveModel:
                 performMachineLearning.predictionModel.saveModel(modelPath)
        
        return readData
            
    
    # The VR Requires Threading to Update the Game + Process the Biolectric Signals
    if controlVR:
        readData = threading.Thread(target = executeProtocol, args = (), daemon=True).start()
    else:
        readData = executeProtocol()
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Save Input data --------------------------- #
    # Save the Data in Excel: EOG Channels (Cols 1-4); X-Peaks (Cols 5-8); Peak Features (Cols 9-12)
    if saveData:
        # Double Check to See if User Wants to Save the Data
        verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
        sheetName = "Trial 1 - "  # If SheetName Already Exists, Increase Trial # by One
        sheetName = sheetName + eyeMovement
        if verifiedSave.upper() == "Y":
            # Initialize Class to Save the Data and Save
            saveInputs = excelData.saveExcel(numChannels, numFeatures = 0)
            saveInputs.saveData(readData.data, readData.featureList, saveDataFolder, saveExcelName, sheetName)
        else:
            print("User Chose Not to Save the Data")
    
    
"""
https://www.frontiersin.org/articles/10.3389/fnins.2017.00012/full
https://www.sciencedirect.com/science/article/pii/S221509861931403X#f0015

x = readData.analysisProtocol.data['timePoints']
y = readData.analysisProtocol.data['Channel1']

# Basic Modules
import sys
import math
import numpy as np
# Peak Detection
import scipy
import scipy.signal
from  itertools import chain
# High/Low Pass Filters
from scipy.signal import butter
# Calibration Fitting
from scipy.optimize import curve_fit
# Plotting
import matplotlib
import matplotlib.pyplot as plt

def butterParams(cutoffFreq = [0.1, 7], samplingFreq = 800, order=3, filterType = 'band'):
    nyq = 0.5 * samplingFreq
    if filterType == "band":
        normal_cutoff = [freq/nyq for freq in cutoffFreq]
    else:
        normal_cutoff = cutoffFreq / nyq
    sos = butter(order, normal_cutoff, btype = filterType, analog = False, output='sos')
    return sos

def butterFilter(data, cutoffFreq, samplingFreq, order = 3, filterType = 'band'):
    sos = butterParams(cutoffFreq, samplingFreq, order, filterType)
    return scipy.signal.sosfiltfilt(sos, data)
    
filteredData = butterFilter(y, 25, 1006, 3, 'low')

startInd = 0; stopInd = len(x)
xData = np.array(x[startInd:stopInd])
yData = np.array(filteredData[startInd:stopInd])
"""
