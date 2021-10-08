"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Data Aquisition:
    
    --------------------------------------------------------------------------
    
    Modules to Import Before Running the Program (Some May be Missing):
        $ conda install matplotlib
        $ conda install tensorflow
        $ conda install openpyxl
        $ conda install sklearn
        $ conda install joblib
        $ conda install numpy
        $ conda install keras
        
    --------------------------------------------------------------------------
"""
# Use '%matplotlib qt' to View Plot

# Basic Modules
import sys
import numpy as np
import collections

# Neural Network Modules
from sklearn.model_selection import train_test_split

# Import Data Aquisition and Analysis Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import readDataExcel as excelData       # Functions to Save/Read in Data from Excel
import readDataArduino as streamData    # Functions to Read in Data from Arduino
import emgAnalysis as emgAnalysis

# Import Machine Learning Files
sys.path.append('./Machine Learning/')  # Folder with Machine Learning Files
import neuralNetwork as NeuralNet       # Functions for Neural Network Algorithm
import Linear_Regression as LR          # Functions for Linear Regression Algorithm
import KNN as KNN                       # Functions for K-Nearest Neighbors' Algorithm
import SVM as SVM                       # Functions for Support Vector Machine algorithm


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # General Data Collection Information (You Will Likely Not Edit These)
    handSerialNum = None
    emgSerialNum = '85735313333351E040A0' # Arduino's Serial Number (port.serial_number)
    numDataPoints = 50000  # The Number of Points to Stream into the Arduino
    moveDataFinger = 200    # The Number of Data Points to Plot/Analyze at a Time; My Beta-Test Used 200 Points
    numChannels = 4         # The Number of Arduino Channels with EMG Signals Read in; My Beta-Test Used 4 Channels
    numFeatures = 4         # The Number of Features to Extract/Save/Train on
    numTimePoints = 2000           # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    
    # Protocol Switches: Only One Can be True; Only the First True Variable Excecutes
    streamArduinoData = True   # Stream in Data from the Arduino and Analyze; Input 'testModel' = True to Apply Learning
    readDataFromExcel = True  # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    reAnalyzePeaks = False     # Read in ALL Data Under 'trainDataExcelFolder', and Reanalyze Peaks (THIS EDITS EXCEL DATA IN PLACE!; DONT STOP PROGRAM MIDWAY)
    trainModel = False         # Read in ALL Data Under 'neuralNetworkFolder', and Train the Data
    
    # User Options During the Run: Any Number Can be True
    plotStreamedData = True  # Graph the Data to Show Incoming Signals + Analysis
    saveInputData = False      # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'
    saveModel = False         # Save the Machine Learning Model for Later Use
    testModel = False         # Apply the Learning Algorithm to Decode the Signals
    
    # ---------------------------------------------------------------------- #
    
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveInputData:
        saveExcelName = "Samuel Solomon 2021-10-06 Circles.xlsx"  # The Name of the Saved File
        saveDataFolder = "../Output Data/All Data/Industry Electrodes/Sam/"   # Data Folder to Save the Excel Data; MUST END IN '/'
        eyeMovement = "Up"                          # Speficy the eye Movement You Will Perform
    
    # Instead of Arduino Data, Use Test Data from Excel File
    if readDataFromExcel:
        testDataExcelFile = "../Input Data/All Data/Industry Electrodes//Samuel Solomon 2021-09-20 Round 1.xlsx" # Path to the Test Data
        testSheetNum = 0   # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
    # Use Previously Processed Data that was Saved; Extract Features for Training
    if reAnalyzePeaks or trainModel:
        trainDataExcelFolder = "../Input Data/Full Training Data/Lab Electrodes/Sam/May11/"  # Path to the Training Data Folder; All .xlsx Data Used
    
    if trainModel or testModel:
        # Pick the Machine Learning Module to Use
        applyNN = False
        applyKNN = True
        applySVM = False
        applyLR = False
        # Initialize Machine Learning Parameters/Data
        modelPath = "./Machine Learning Modules/Models/myModelKNNFull_SamArm1.pkl"

    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #           Initiate Neural Network (Should Not Have to Edit)            #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    # Define Labels as Array
    movementOptions = np.array(["Up", "Down", "Left", "Right"])
    movementOptions = np.char.lower(movementOptions)
    # Edge Case: User Defines a Movement that Does not Exist, Return Error
    if saveInputData and eyeMovement.lower() not in movementOptions:
        print("\nUser Defined an Unknown eye Gesture")
        print("The Gesture", "'" + eyeMovement.lower() + "'", "is Not in", movementOptions)
        sys.exit()
    
    if trainModel or testModel:
        # Make the Neural   (dim = The dimensionality of one data point) 
        if applyNN:
            MLModel = NeuralNet.Neural_Network(modelPath = modelPath, dataDim = numChannels)
        elif applyKNN:
            MLModel = KNN.KNN(modelPath = modelPath, numClasses = len(movementOptions))
        elif applySVM:
            MLModel = SVM.SVM(modelPath = modelPath, modelType = "poly", polynomialDegree = 3)
        elif applyLR:
            MLModel = LR.logisticRegression(modelPath = modelPath)
    else:
        MLModel = None
        
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #           Data Collection Program (Should Not Have to Edit)            #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    emgProtocol = emgAnalysis.emgProtocol(numTimePoints, moveDataFinger, numChannels, movementOptions, plotStreamedData)
    # Stream in Data from Arduino
    if streamArduinoData:
        arduinoRead = streamData.arduinoRead(eogSerialNum = None, emgSerialNum = emgSerialNum, eegSerialNum = None, handSerialNum = handSerialNum)
        readData = streamData.emgArduinoRead(arduinoRead, numTimePoints, moveDataFinger, numChannels, movementOptions, plotStreamedData, guiApp = None)
        readData.streamEMGData(numDataPoints, myModel = MLModel)
    # Take Data from Excel Sheet
    elif readDataFromExcel:
        readData = excelData.readExcel(emgProtocol)
        readData.streamExcelData(testDataExcelFile, plotStreamedData, testSheetNum, myModel = MLModel)
    # Redo Peak Analysis
    elif reAnalyzePeaks:
        readData = excelData.readExcel(emgProtocol)
        readData.getTrainingData(trainDataExcelFolder, movementOptions, mode='reAnalyze')
    # Take Preprocessed (Saved) Features from Excel Sheet
    elif trainModel:
        readData = excelData.readExcel(emgProtocol)
        signalData, signalLabels = readData.getTrainingData(trainDataExcelFolder, movementOptions, mode='Train')
        print("\nCollected Signal Data")
    
    # Save the Data in Excel: EMG Channels (Cols 1-4); X-Peaks (Cols 5-8); Peak Features (Cols 9-12)
    if saveInputData:
        # Format Sheet Name
        sheetName = "Trial 1 - "  # If SheetName Already Exists, Increase Trial # by One
        sheetName = sheetName + eyeMovement
        # Double Check to See if User Wants to Save the Data
        verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
        if verifiedSave.upper() == "Y":
            # Initialize Class to Save the Data and Save
            saveInputs = excelData.saveExcel(numChannels, numFeatures)
            saveInputs.saveData(readData.data, readData.featureLocsX, readData.featureSetGrouping, saveDataFolder, saveExcelName, sheetName, eyeMovement)
        else:
            print("User Chose Not to Save the Data")
    
    
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #                    Train the Machine Learning Model                    #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    # Train the ML
    if trainModel:
        # Split the Data into Training and Validation Sets
        Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.2, shuffle= True, stratify=signalLabels)
        signalLabelsClass = [np.argmax(i) for i in signalLabels]
        
        if applyKNN or applySVM or applyLR:
            # Format Labels into 1D Array (Needed for KNN Setup)
            Training_LabelsClass = [np.argmax(i) for i in Training_Labels]
            Testing_LabelsClass= [np.argmax(i) for i in Testing_Labels]
            # Train the NN with the Training Data
            MLModel.trainModel(Training_Data, Training_LabelsClass, Testing_Data, Testing_LabelsClass)
            # Plot the training loss    
            #MLModel.plotModel(signalData, signalLabelsClass)
            #MLModel.plot3DLabels(signalData, signalLabelsClass)
            map2D = MLModel.mapTo2DPlot(signalData, signalLabelsClass)
            MLModel.plot3DLabelsMovie(signalData, np.array(signalLabelsClass))
            #MLModel.accuracyDistributionPlot(signalData, signalLabelsClass, MLModel.predictData(signalData), movementOptions)
            # Save Signals and Labels
            saveSignals = True
            if saveSignals:
                saveDataFolder = "../Output Data/"
                saveExcelName = "Maped Data.xlsx" #"Signal Features with Predicted and True Labels New.xlsx"
                saveInputs = excelData.saveExcel(numChannels, numFeatures)
                saveInputs.saveLabeledPoints(map2D, signalLabels, MLModel.predictData(signalData), saveDataFolder, saveExcelName, sheetName = "Signal Data and Labels")


        if applyNN:
            # Train the NN with the Training Data
            MLModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, 500, seeTrainingSteps = False)
            # Plot the training loss    
            MLModel.plotModel(signalData, signalLabelsClass)
            MLModel.plot3DLabels(signalData, signalLabelsClass)
            MLModel.accuracyDistributionPlot(signalData, signalLabelsClass, MLModel.predictData(signalData), movementOptions)
            MLModel.plotStats()

        # Save the Neural Network (The Weights of Each Edge)
        if saveModel:
            MLModel.saveModel(modelPath)

        # Find the Data Distribution
        classDistribution = collections.Counter(signalLabelsClass)
        print("Class Distribution:", classDistribution)
        print("Number of Data Points = ", len(classDistribution))
        
        
