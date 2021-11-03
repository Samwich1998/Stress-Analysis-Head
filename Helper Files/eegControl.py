"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Data Aquisition:
    Each Channel Consists of 3 Electrodes: Two EEG Electrodes + 1 EEG Reference
    The Standard Setup Consists of Placing the Electrodes along a muscle group.
    The Reference Electrode Should be Placed in the Middle, And the Electrodes
    Should Line Up On the Axis From the hand to the Elbow (If Using Lower Arm).
    Provide Decent Spacing Between the Electrodes (Noticeable Gap)
    
    HardWare Processing:
    The Code Below Used the Following Electronic Material from Olimex:  
        Circuit Board: https://www.olimex.com/Products/Duino/Shields/SHIELD-EKG-EEG/open-source-hardware
        Electrodes: https://www.olimex.com/Products/Duino/Shields/SHIELD-EKG-EEG-PRO/
    
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
        
    --------------------------------------------------------------------------
"""

# Basic Modules
import sys
import numpy as np
from pathlib import Path

# Import Data Aquisition and Analysis Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import readDataExcel as excelData         # Functions to Save/Read in Data from Excel
import readDataArduino as streamData      # Functions to Read in Data from Arduino
import eegAnalysis as eegAnalysis         # Functions to Analyze the EEG Data

# Import Files for Machine Learning
sys.path.append('./Machine Learning/')  # Folder with Machine Learning Files
import machineLearningMain  # Class Header for All Machine Learning


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # General Data Collection Information (You Will Likely Not Edit These)
    eegSerialNum = '85735313333351E040A0' # Arduino Serial Number (port.serial_number) Collecting EEG Signals
    numDataPoints = 50000   # The Number of Points to Stream into the Arduino
    numTimePoints = 3000    # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    moveDataFinger = 200    # The Number of NEW Data Points to Analyze at a Time; My Beta-Test Used 200 Points with Plotting (100 Without). This CAN Change How SOME Peaks are Found (be Careful)
    samplingFreq = 800      # The Average Number of Points Steamed Into the Arduino Per Second; If NONE Given, Algorithm will Calculate Based on Initial Data
    numChannels = 4         # The Number of Arduino Channels with EEG Signals Read in; My Beta-Test Used 4 Channels
    numFeatures = 4         # The Number of Features to Extract/Save/Train on
    
    # Protocol Switches: Only One Can be True; Only the First True Variable Excecutes
    streamArduinoData = False  # Stream in Data from the Arduino and Analyze; Input 'testModel' = True to Apply Learning
    readDataFromExcel = True   # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    reAnalyzePeaks = False      # Read in ALL Data Under 'trainDataExcelFolder', and Reanalyze Peaks (THIS EDITS EXCEL DATA IN PLACE!; DONT STOP PROGRAM MIDWAY)
    trainModel = False         # Read in ALL Data Under 'neuralNetworkFolder', and Train the Data
    
    # User Options During the Run: Any Number Can be True
    plotStreamedData = True    # Graph the Data to Show Incoming Signals + Analysis
    saveModel = False          # Save the Machine Learning Model for Later Use
    testModel = False          # Apply the Learning Algorithm to Decode the Signals
    saveData = True           # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName' or map2D if Training
    
    # ---------------------------------------------------------------------- #
    
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveData:
        saveExcelName = "Samuel Solomon 2021-10-06 Circles.xlsx"  # The Name of the Saved File
        saveDataFolder = "../Output Data/EEG Data/"  # Data Folder to Save the Excel Data; MUST END IN '/'
            
    # Instead of Arduino Data, Use Test Data from Excel File
    if readDataFromExcel:
        testDataExcelFile = "../Input Data/EMG Data/Samuel Solomon (Pure; Robot Computer) 2021-03-24.xlsx" # Path to the Test Data
        testSheetNum = 0   # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
    # Use Previously Processed Data that was Saved; Extract Features for Training
    if reAnalyzePeaks or trainModel:
        gestureClasses = []
        trainDataExcelFolder = "../Input Data/EEG Data/"  # Path to the Training Data Folder; All .xlsx Data Used

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

    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #           Data Collection Program (Should Not Have to Edit)            #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    if not streamArduinoData:
        eegProtocol = eegAnalysis.eegProtocol(numTimePoints, moveDataFinger, numChannels, samplingFreq, plotStreamedData)
    # Stream in Data from Arduino
    if streamArduinoData:
        arduinoRead = streamData.arduinoRead(eogSerialNum = None, ppgSerialNum = None, emgSerialNum = None, eegSerialNum = eegSerialNum, handSerialNum = None)
        readData = streamData.eegArduinoRead(arduinoRead, numTimePoints, moveDataFinger, numChannels, samplingFreq, plotStreamedData, guiApp = None)
        readData.streamEEGData(numDataPoints, predictionModel = predictionModel, actionControl = None)
        featureList = readData.analysisProtocol.featureList
    # Take Data from Excel Sheet
    elif readDataFromExcel:
        readData = excelData.readExcel(eegProtocol)
        readData.streamExcelData(testDataExcelFile, plotStreamedData, testSheetNum, predictionModel = predictionModel, actionControl = None)
        featureList = eegProtocol.featureList
    # Redo Peak Analysis
    elif reAnalyzePeaks:
        readData = excelData.readExcel(eegProtocol)
        readData.getTrainingData(trainDataExcelFolder, numFeatures, gestureClasses, mode='reAnalyze')
    # Take Preprocessed (Saved) Features from Excel Sheet
    elif trainModel:
        # Extract the Data
        readData = excelData.readExcel(eegProtocol)
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
        
    
    # Save the Data in Excel
    if saveData and not trainModel and not reAnalyzePeaks:
        # Format Sheet Name
        sheetName = "Trial 1 - EEG"  # If SheetName Already Exists, Increase Trial # by One
        # Double Check to See if User Wants to Save the Data
        verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
        if verifiedSave.upper() == "Y":
            # Initialize Class to Save the Data and Save
            saveInputs = excelData.saveExcel(numChannels, numFeatures)
            saveInputs.saveData(eegProtocol.data, featureList, saveDataFolder, saveExcelName, sheetName)
        else:
            print("User Chose Not to Save the Data")

        


        

        
        
