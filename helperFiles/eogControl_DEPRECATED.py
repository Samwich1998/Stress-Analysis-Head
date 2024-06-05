"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Data Aquisition:
        
    Plotting:
        If Plotting, You Need an GUI Backend -> In Spyder IDE Use: %matplotlib qt5
        Some IDEs (Spyder Included) may Naturally Plot in GUI.
    --------------------------------------------------------------------------
    
    Modules to Import Before Running the Program (Some May be Missing):
        pip install -U numpy scikit-learn scipy matplotlib openpyxl pyserial joblib pandas
        pip install -U natsort pyfirmata2 shap ipywidgets seaborn pyqt6 lightgbm
        pip install -U ipython qdarkstyle pygame librosa xgboost sklearn
        pip install -U tensorflow pyexcel eeglib
    
    Programs to Install:
        Vizard (If using Virtual Reality): https://www.worldviz.com/virtual-reality-software-downloads
        
    --------------------------------------------------------------------------
"""

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import sys
import threading
import numpy as np
from pathlib import Path

sys.path.append('./Data Aquisition and Analysis/Biolectric Protocols/')  # Folder with Data Aquisition Files
import eogAnalysis as eogAnalysis         # Functions to Analyze the EOG Data

# Import Data Aquisition and Analysis Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import excelProcessing as excelData         # Functions to Save/Read in Data from Excel
import readDataArduino as streamData      # Functions to Read in Data from Arduino

# Import Virtual Reality Control Files
sys.path.append('./Execute Movements/Virtual Reality Control/')   # Folder with Virtual Reality Control Files

# Import Files for Machine Learning
sys.path.append('./Machine Learning/')  # Folder with Machine Learning Files
import machineLearningMain  # Class Header for All Machine Learning
import featureAnalysis      # Functions for Feature Analysis


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #
    #sys.exit()

    # General Data Collection Information (You Will Likely Not Edit These)
    eogSerialNum = '85035323234351D041B2'  # Arduino's Serial Number (port.serial_number)
    
    stopTimeStreaming = 600*10       # The Last Time to Stream into the Arduino. If Float, it is the Seconds from 12:00am; If String, it is the TimeStamp to Stop (Military Time) as "Hours:Minutes:Seconds:MicroSeconds"
    numTimePoints = 2048576          # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    moveDataFinger = 1048100         # The Number of Data Points to Plot/Analyze at a Time; My Beta-Test Used 200 Points with Plotting; 10 Points Without
    # numTimePoints = 5000           # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    # moveDataFinger = 2000          # The Number of Data Points to Plot/Analyze at a Time; My Beta-Test Used 200 Points with Plotting; 10 Points Without
    numChannels = 2                  # The Number of Arduino Channels with EOG Signals Read in;
    
    
    # Specify the Type of Movements to Learn
    gestureClasses = np.char.lower(['Spontaneous', 'Reflex', 'Voluntary', 'Double'])  # Define Labels as Array
    gestureClasses = np.char.lower(['Up', 'Down', 'Blink', 'Double Blink', 'Relaxed', 'Relaxed to Cold', 'Cold'])  # Define Labels as Array
    
    gestureClasses = np.char.lower(['Blink', 'Double Blink', 'Relaxed', 'Stroop Test', 'Exercise Weight', 'VR Roller Coaster'])  # Define Labels as Array
    machineLearningClasses = np.char.lower(['Relaxed', 'Stroop', 'Exercise', 'VR'])
    labelMap = [1, -1, 0, -1, -1, -1]
    
    gestureClasses = np.char.lower(['Morning', 'Night', 'Music'])
    machineLearningClasses = np.char.lower(['Morning', 'Night', 'Music'])
    labelMap = [-1, -1, 2]
    
    labelDict = dict(zip(labelMap, gestureClasses))
    
    # Protocol Switches: Only the First True Variable Excecutes
    streamArduinoData = False     # Stream in Data from the Arduino and Analyze; Input 'controlVR' = True to Move VR
    readDataFromExcel = False     # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    trainModel = True             # Read in ALL Data Under 'neuralNetworkFolder', and Train the Data
    
    # User Options During the Run: Any Number Can be True
    plotStreamedData = False      # Graph the Data to Show Incoming Signals + Analysis
    calibrateModel = False        # Calibrate the EOG Voltage to Predict the Eye's Angle
    saveData = False              # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'
    testModel = False             # Apply the Learning Algorithm to Decode the Signals
    controlVR = False             # Apply the Algorithm to Control the Virtual Reality View    

    # ------------------------ Dependant Parameters ------------------------- #
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveData:
        saveExcelName = "2022-03-02 Ruiain Mobb Box.xlsx"  # The Name of the Saved File
        saveDataFolder = "../Data/EOG Data/All Data/2022-03-02 Mobbs Lab Box/"   # Data Folder to Save the Excel Data; MUST END IN '/'
        # Speficy the eye Movement You Will Perform
        eyeMovement = "Full".lower() # Make Sure it is Lowercase
        if eyeMovement not in gestureClasses:
            print("The Gesture", "'" + eyeMovement + "'", "is Not in", gestureClasses)
            
    # Instead of Arduino Data, Use Test Data from Excel File
    if readDataFromExcel:
        #testDataExcelFile = "../Data/EOG Data/All Data/2021-12-01 First Cold Water Test/Jiahong 2021-12-1 Movements.xlsx" # Path to the Test Data
        #testDataExcelFile = "../Data/EOG Data/All Data/2021-12-10 First VR Test/Jose 2021-12-10 Movements.xlsx" # Path to the Test Data
       # testDataExcelFile = "../Data/EOG Data/All Data/2022-01-01 Sarah Music/2021-01-01 Sarah.xlsx" # Path to the Test Data
        testDataExcelFile = "../Data/EOG Data/All Data/2022-03-02 Mobbs Lab Box/2022-03-02 Ruiain Mobb Box.xlsx"   # Data Folder to Save the Excel Data; MUST END IN '/'
        testSheetNum = 0   # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
    # Input Training Paramaters 
    if trainModel:
        trainDataExcelFolder = "../Data/EOG Data/All Data/2021-12-10 First VR Test/"
        trainDataExcelFolder = "../Data/EOG Data/All Data/2021-12-11 Home Study/"  # Path to the Training Data Folder; All .xlsx Data Used
        trainDataExcelFolder = "../Data/EOG Data/All Data/2021-12-15 Music Study Home/"  # Path to the Training Data Folder; All .xlsx Data Used
        trainDataExcelFolder = "../Data/EOG Data/All Data/2022-01-01 Sarah Music/"  # Path to the Training Data Folder; All .xlsx Data Used

        trainDataExcelFolder = "../Data/EOG Data/All Data/2021-12-20 Combined Blinks/"  # Path to the Training Data Folder; All .xlsx Data Used
        trainDataExcelFolder = "../Data/EOG Data/All Data/2022-02-23 Sam Blink vs Noise/"   # Data Folder to Save the Excel Data; MUST END IN '/'
      
        trainDataExcelFolder = "../Data/EOG Data/All Data/2022-03-02 Mobbs Lab Box/"   # Data Folder to Save the Excel Data; MUST END IN '/'


    # Train or Test the Data with the Machine Learning Model
    if trainModel or testModel:
        # Pick the Machine Learning Module to Use
        modelType = "RF"  # Machine Learning Options: NN, RF, LR, KNN, SVM
        modelPath = "./Machine Learning/Models/predictionModelRF_12-13-2021.pkl" # Path to Model (Creates New if it Doesn't Exist)
        # Choos the Folder to Save ML Results
        if trainModel:
            saveModel = True  # Save the Machine Learning Model for Later Use
            saveDataFolder = trainDataExcelFolder + "Data Analysis/" #+ modelType + "/"
        else:
            saveDataFolder = None
        # Get the Machine Learning Module
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(blinkFeatures), gestureClasses = machineLearningClasses, saveDataFolder = saveDataFolder)
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
    
    def executeProtocol():
        global readData, eogProtocol, performMachineLearning, signalData, signalLabels, timePoints, modelPath, blinkFeatures, analyzeFeatures
        
        eogProtocol = eogAnalysis.eogProtocol(numTimePoints, moveDataFinger, numChannels, plotStreamedData)
        # Stream in Data from Arduino
        if streamArduinoData:
            arduinoRead = streamData.arduinoRead(eogSerialNum = eogSerialNum, ppgSerialNum = None, emgSerialNum = None, eegSerialNum = None, handSerialNum = None)
            readData = streamData.eogArduinoRead(arduinoRead, numTimePoints, moveDataFinger, numChannels, plotStreamedData, guiApp = None)
            readData.streamEOGData(stopTimeStreaming, predictionModel = predictionModel, actionControl = gazeControl, calibrateModel = calibrateModel)
        # Take Data from Excel Sheet
        elif readDataFromExcel:
            readData = excelData.readExcel(eogProtocol)
            readData.streamExcelData(testDataExcelFile, plotStreamedData, testSheetNum, predictionModel = predictionModel, actionControl = None)
        # Take Preprocessed (Saved) Features from Excel Sheet
        elif trainModel:
            # Extract the Data
            readData = excelData.readExcel(eogProtocol)
            signalData, signalLabels = readData.getTrainingData(trainDataExcelFolder, gestureClasses, blinkFeatures, labelMap)
            signalData = np.array(signalData); signalLabels = np.array(signalLabels)
            print("\nCollected Signal Data")
           # sys.exit()
            
            analyzeFeatures = True
            if analyzeFeatures:
                timePoints = signalData[:,0]
                signalData = signalData[:,1:]
                blinkFeatures = np.array(blinkFeatures[1:])

                analyzeFeatures = featureAnalysis.featureAnalysis(blinkFeatures, [], saveDataFolder)
                #analyzeFeatures.correlationMatrix(signalData, folderName = "correlationMatrix/")
                analyzeFeatures.singleFeatureAnalysis(timePoints[signalLabels == 3], signalData[signalLabels == 3], averageIntervalList = [60, 2*60, 3*60], folderName = "singleFeatureAnalysis - Full/")
                analyzeFeatures.featureDistribution(signalData, signalLabels, labelDict, folderName = "Feature Distribution/")
            
            
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
        
        return readData
            
    
    # The VR Requires Threading to Update the Game + Process the Biolectric Signals
    if controlVR:
        readData = threading.Thread(target = executeProtocol, args = (), daemon=True).start()
    else:
        readData = executeProtocol()
    
    os.system('say "Program Complete."')
    # ---------------------------------------------------------------------- #
    # -------------------------- Save Data --------------------------- #
    # Save the Data in Excel: EOG Channels (Cols 1-4); X-Peaks (Cols 5-8); Peak Features (Cols 9-12)
    if saveData:
        # Double Check to See if User Wants to Save the Data
        verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
        sheetName = "Trial 1 - "  # If SheetName Already Exists, Increase Trial # by One
        sheetName = sheetName + eyeMovement
        if verifiedSave.upper() == "Y":
            # Initialize Class to Save the Data and Save
            saveInputs = excelData.saveExcel()
            saveInputs.saveData(readData.data, [], [], saveDataFolder, saveExcelName, sheetName, eyeMovement)
        else:
            print("User Chose Not to Save the Data")
    
"""




# -------------------------------------------------------------------------- #
# ---------------------------- Reading EOG Data ---------------------------- #

# Deprecated Class: Keeping in case user wants to calibrate EOG sensor in the future
class eogArduinoRead(eogProtocol):

    def __init__(self, mainSerialNum, numPointsPerBatch, moveDataFinger, numChannels, plotStreamedData):
        # Get Variables from Peak Analysis File
        super().__init__(numPointsPerBatch, moveDataFinger, numChannels, plotStreamedData)

        # Create Pointer to Common Functions
        self.commonFunctions = streamingHead(mainSerialNum = mainSerialNum, therapySerialNum = None, numChannels = numChannels)

    def streamEOGData(self, stopTimeStreaming, modelClasses = None, actionControl = None, calibrateModel = False, numTrashReads=100, numPointsPerRead=300):
        print("Streaming in EOG Data from the Arduino")
        # Prepare the arduino to stream in data
        stopTimeStreaming = self.commonFunctions.setupArduinoStream(stopTimeStreaming)

        try:
            # If Needed Calibrate the Model
            if calibrateModel:
                self.askForCalibration(numTrashReads)

            dataFinger = 0
            # Loop Through and Read the Arduino Data in Real-Time
            while len(self.data[0]) == 0 or self.data[0][-1] < stopTimeStreaming:
                # Stream in the Latest Data
                self.commonFunctions.recordData(self.data, self.numChannels)
                
                # When Ready, Send Data Off for Analysis
                while len(self.data[0]) - dataFinger >= self.numPointsPerBatch:
                    # Analyze EOG Data
                    self.analyzeData(dataFinger, self.plotStreamedData, modelClasses = modelClasses, actionControl = actionControl, calibrateModel = calibrateModel)
                    # Move DataFinger to Analyze Next Section
                    dataFinger += self.moveDataFinger

                    # If You Need to Calibrate a Channel
                    if calibrateModel:
                        dataFinger = 0
                        calibrateModel = self.performCalibration(numTrashReads)
                        break
            # At the End, Analyze Any Data Left
            self.analyzeData(dataFinger, self.plotStreamedData, modelClasses = modelClasses, actionControl = actionControl, calibrateModel = calibrateModel)

        finally:
            self.mainArduino.close()

        print("Finished Streaming in Data; Closing Arduino\n")
        # Close the Arduinos at the End
        self.mainArduino.close()
    
    def performCalibration(self, numTrashReads, calibrateModel = True):
        self.channelCalibrationPointer += 1

        # See If Calibration of the Channel is Complete
        if len(self.calibrationVoltages[self.calibrateChannelNum]) == len(self.calibrationAngles[self.calibrateChannelNum]):
            # Get Data to Calibrate
            xData = self.calibrationVoltages[self.calibrateChannelNum]
            yData = self.calibrationAngles[self.calibrateChannelNum]
            # Calibrate the Data
            self.fitCalibration(xData, yData, channelIndexCalibrating = self.calibrateChannelNum, plotFit = False)
            # Move Onto the Next Channel
            self.calibrateChannelNum += 1
            self.channelCalibrationPointer = 0

        # Check if All Channel Calibrations are Complete
        if self.calibrateChannelNum == self.numChannels:
            # Reset Arduino and Stop Calibration
            self.initPlotPeaks()
            self.mainArduino = self.arduinoRead.resetArduino(self.mainArduino, numTrashReads)
            calibrateModel = False
        else:
            self.askForCalibration(numTrashReads)
        
        # Reset Stream 
        self.resetGlobalVariables()
        return calibrateModel

    def askForCalibration(self, numTrashReads):
        # Inform User of Next Angle; Then Flush Saved Outputs
        input("Orient Eye at " + str(self.calibrationAngles[self.calibrateChannelNum][self.channelCalibrationPointer]) + " Degrees For Channel " + str(self.calibrateChannelNum))
        # Reset Arduino
        self.mainArduino = self.arduinoRead.resetArduino(self.mainArduino, numTrashReads)
        print("Orient Now")
        






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

plt.plot(xData, yData)
plt.xlim(122.5,123)



personListBlink = [youFeaturesBlink]
personListDoubleBlink = [youFeaturesDoubleBlink]
personListRelaxed = [youFeaturesDoubleBlink]
personListCold = [youFeaturesCold]

personListBlink = [changhaoFeaturesBlink]
personListDoubleBlink = [changhaoFeaturesDoubleBlink]
personListRelaxed = [changhaoFeaturesRelaxed]
personListCold = [changhaoFeaturesCold]

personListBlink = [jiahongFeaturesBlink, benFeaturesBlink, youFeaturesBlink, changhaoFeaturesBlink]
personListDoubleBlink = [jiahongFeaturesDoubleBlink, benFeaturesDoubleBlink, youFeaturesDoubleBlink, changhaoFeaturesDoubleBlink]
personListRelaxed = [jiahongFeaturesRelaxed, benFeaturesRelaxed, youFeaturesRelaxed, changhaoFeaturesRelaxed]
personListCold = [jiahongFeaturesCold, benFeaturesCold, youFeaturesCold, changhaoFeaturesCold]



jiahongFeaturesBlink = np.array(readData.analysisProtocol.blinkFeatures)
benFeaturesBlink = np.array(readData.analysisProtocol.blinkFeatures)
youFeaturesBlink = np.array(readData.analysisProtocol.blinkFeatures)
changhaoFeaturesBlink = np.array(readData.analysisProtocol.blinkFeatures)
#personListBlink = [jiahongFeaturesBlink, benFeaturesBlink, youFeaturesBlink, changhaoFeaturesBlink]
personListBlink = [changhaoFeaturesBlink]

jiahongFeaturesDoubleBlink = np.array(readData.analysisProtocol.blinkFeatures)
benFeaturesDoubleBlink = np.array(readData.analysisProtocol.blinkFeatures)
youFeaturesDoubleBlink = np.array(readData.analysisProtocol.blinkFeatures)
changhaoFeaturesDoubleBlink = np.array(readData.analysisProtocol.blinkFeatures)
#personListDoubleBlink = [jiahongFeaturesDoubleBlink, benFeaturesDoubleBlink, youFeaturesDoubleBlink, changhaoFeaturesDoubleBlink]
personListDoubleBlink = [changhaoFeaturesDoubleBlink]

jiahongFeaturesRelaxed = np.array(readData.analysisProtocol.blinkFeatures)
benFeaturesRelaxed = np.array(readData.analysisProtocol.blinkFeatures)
youFeaturesRelaxed = np.array(readData.analysisProtocol.blinkFeatures)
changhaoFeaturesRelaxed = np.array(readData.analysisProtocol.blinkFeatures)
#personListRelaxed = [jiahongFeaturesRelaxed, benFeaturesRelaxed, youFeaturesRelaxed, changhaoFeaturesRelaxed]
personListRelaxed = [changhaoFeaturesRelaxed]

jiahongFeaturesCold = np.array(readData.analysisProtocol.blinkFeatures)
benFeaturesCold = np.array(readData.analysisProtocol.blinkFeatures)
youFeaturesCold = np.array(readData.analysisProtocol.blinkFeatures)
changhaoFeaturesCold = np.array(readData.analysisProtocol.blinkFeatures)
#personListCold = [jiahongFeaturesCold, benFeaturesCold, youFeaturesCold, changhaoFeaturesCold]
personListCold = [changhaoFeaturesCold]

colorList = ['b','k','r','m']
colorList1 = ['b','k','r','m']
for i in range(len(personListBlink[0][0])):
    for j in range(i+1, len(personListBlink[0][0])):
        fig = plt.figure()
        
        for personNum in range(len(personListBlink)):
            personFeatures = personListBlink[personNum]
            feature1 = personFeatures[:,i]
            feature2 = personFeatures[:,j]
            plt.plot(feature1, feature2, colorList[personNum]+'o')
        
        for personNum in range(len(personListDoubleBlink)):
            personFeatures = personListDoubleBlink[personNum]
            feature1 = personFeatures[:,i]
            feature2 = personFeatures[:,j]
            plt.plot(feature1, feature2, colorList[personNum]+'^')
                
        for personNum in range(len(personListRelaxed)):
            personFeatures = personListRelaxed[personNum]
            feature1 = personFeatures[:,i]
            feature2 = personFeatures[:,j]
            plt.plot(feature1, feature2, colorList[personNum]+'x', zorder = 150)
        
        for personNum in range(len(personListCold)):
            personFeatures = personListCold[personNum]
            feature1 = personFeatures[:,i]
            feature2 = personFeatures[:,j]
            plt.scatter(feature1, feature2, c='w',edgecolors=colorList[personNum], zorder = 100)
                
        plt.xlabel(blinkFeatures[i])
        plt.ylabel(blinkFeatures[j])
        #fig.savefig('../output/' + blinkFeatures[i] + ' VS ' + blinkFeatures[j] + ".png", dpi=300, bbox_inches='tight')
        plt.show()
        
for i in range(len(personListBlink[0][0])):
    fig = plt.figure()
    
    for personNum in range(len(personListBlink)):
        personFeatures = personListBlink[personNum]
        feature1 = personFeatures[:,i]
        plt.plot(feature1, colorList[personNum]+'o')
    
    for personNum in range(len(personListDoubleBlink)):
        personFeatures = personListDoubleBlink[personNum]
        feature1 = personFeatures[:,i]
        plt.plot(feature1, colorList[personNum]+'^')
            
    for personNum in range(len(personListRelaxed)):
        personFeatures = personListRelaxed[personNum]
        feature1 = personFeatures[:,i]
        plt.plot(feature1, colorList[personNum]+'x', zorder = 150)
    
    for personNum in range(len(personListCold)):
        personFeatures = personListCold[personNum]
        feature1 = personFeatures[:,i]
        plt.scatter(np.arange(0, len(feature1), 1), feature1, c='w',edgecolors=colorList[personNum], zorder = 100)
            
    plt.xlabel("Blinks")
    plt.ylabel(blinkFeatures[i])
    #fig.savefig('../outputSingle/' + blinkFeatures[i] + ".png", dpi=300, bbox_inches='tight')
    plt.show()
        
        
        
signalData = []; signalLabels = []
for personNum in range(len(personListBlink)):
    personFeatures = personListBlink[personNum]
    for personFeature in personFeatures:
        signalData.append(personFeature)
        signalLabels.append(0)

for personNum in range(len(personListDoubleBlink)):
    personFeatures = personListDoubleBlink[personNum]
    for personFeature in personFeatures:
        signalData.append(personFeature)
        signalLabels.append(0)

for personNum in range(len(personListRelaxed)):
    personFeatures = personListRelaxed[personNum]
    for personFeature in personFeatures:
        signalData.append(personFeature)
        signalLabels.append(0)

for personNum in range(len(personListCold)):
    personFeatures = personListCold[personNum]
    for personFeature in personFeatures:
        signalData.append(personFeature)
        signalLabels.append(1)
        
signalData = np.array(signalData); signalLabels = np.array(signalLabels)

model = neighbors.KNeighborsClassifier(n_neighbors = 2, weights = 'distance', algorithm = 'auto', 
                        leaf_size = 30, p = 1, metric = 'minkowski', metric_params = None, n_jobs = None)
Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.33, shuffle= True, stratify=signalLabels)
model.fit(Training_Data, Training_Labels)
model.score(signalData, signalLabels)











for featureInd in range(len(blinkFeatures)):
    fig = plt.figure()

    features = []
    allFeatures = signalData[:,featureInd]
    time = signalData[:,0]
    for pointInd in range(len(allFeatures)):
        feature = allFeatures[pointInd]
        label = signalLabels[pointInd]
        features.append(feature)
    
    fitLineParams = np.polyfit(time, features, 1)
    fitLine = np.polyval(fitLineParams, time);

    plt.plot(time, features, 'ko')
    plt.plot(time, fitLine, 'r')
    
    plt.xlabel("Time (Seconds)")
    plt.ylabel(blinkFeatures[featureInd])
    plt.title("Slope/average: " + str(fitLineParams[0]/np.mean(features)))
    #fig.savefig('../Time Graph/' + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    
    
    
from scipy import stats
featureDict = {}
signalData = np.array(signalData); signalLabels = np.array(signalLabels)

time = signalData[:,0][signalLabels == 2]
for featureInd in range(len(blinkFeatures)):

    allFeatures = signalData[:,featureInd][signalLabels == 2]
    for ind, averageTogether in enumerate([60*3]):
        features = []
        for pointInd in range(len(allFeatures)):
            featureInterval = allFeatures[time > time[pointInd] - averageTogether]
            timeMask = time[time > time[pointInd] - averageTogether]
            featureInterval = featureInterval[timeMask <= time[pointInd]]

            features.append(stats.trim_mean(featureInterval, 0.3))

        featureDict[blinkFeatures[featureInd]] = features
    
    
import matplotlib.pyplot as plt
time = signalData[:,0][signalLabels == 2]
for featureInd in range(len(blinkFeatures)):
    fig = plt.figure()

    allFeatures = signalData[:,featureInd][signalLabels == 2]
    colors = ['ko', 'ro', 'bo', 'go', 'mo']
    for ind, averageTogether in enumerate([60*3]):
        features = []
        for pointInd in range(len(allFeatures)):
            featureInterval = allFeatures[time > time[pointInd] - averageTogether]
            timeMask = time[time > time[pointInd] - averageTogether]
            featureInterval = featureInterval[timeMask <= time[pointInd]]
            
            featute = stats.trim_mean(featureInterval, 0.3)
            features.append(featute)
        
        features = np.array(features)
        #features -= np.array(featureDict[blinkFeatures[1]])
        plt.plot(time, features, colors[ind], markersize=5)


    plt.xlabel("Time (Seconds)")
    plt.ylabel(blinkFeatures[featureInd])
    plt.vlines(np.array([1, 2, 3, 4, 5])*60*6, min(features)*0.8, max(features)*1.2, 'g', zorder=100)
    #plt.ylim(min(feature1Min)*0.8, max(feature1Max)*1.2)
    plt.legend(['1 Min', '2 Min', '3 Min'])
    plt.title("Averaged Together: " + str(averageTogether/60) + " Min")
    fig.savefig('../Del Josh/' + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
from scipy import stats
import matplotlib.pyplot as plt
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
featureDict = {}
saveFolder = '../Time Graph Music/Average 1 2 3 Back/'
os.makedirs(saveFolder, exist_ok=True)
signalData = np.array(signalData); signalLabels = np.array(signalLabels)
for featureInd in range(len(blinkFeatures)):
    fig = plt.figure()


    allFeatures = signalData[:,featureInd][signalLabels == 2]
    time = signalData[:,0][signalLabels == 2]
    colors = ['ko', 'ro', 'bo', 'go', 'mo']
    for ind, averageTogether in enumerate([60*1, 60*2, 60*3]):
        features = []
        if ind == 1:
            feature1Min = [10E10]; feature1Max = [-10E10]
        for pointInd in range(len(allFeatures)):
            featureInterval = allFeatures[time > time[pointInd] - averageTogether]
            timeMask = time[time > time[pointInd] - averageTogether]
            featureInterval = featureInterval[timeMask <= time[pointInd]]

            #weight = [10E-20]
            #for i in range(int(-len(featureInterval)/2), int(len(featureInterval)/2)):
            #    weight.append(sigmoid(i/20))
            #weight = np.array(weight[-len(featureInterval):])
            #feature = np.average(featureInterval, axis=0, weights=weight)

            features.append(stats.trim_mean(featureInterval, 0.3))

            if ind == 1:
                feature1Min = min([features, feature1Min], key = lambda x: min(x))
                feature1Max = max([features, feature1Max], key = lambda x: max(x))

        #fitLineParams = np.polyfit(time, features, 1)
        #fitLine = np.polyval(fitLineParams, time);

        plt.plot(time, features, colors[ind], markersize=5)
        #plt.plot(time, fitLine, 'r')

        if ind == 0:
            featureDict[blinkFeatures[featureInd]] = features

    plt.xlabel("Time (Seconds)")
    plt.ylabel(blinkFeatures[featureInd])
    plt.vlines(np.array([1, 2, 3, 4, 5])*60*6, min(feature1Min)*0.8, max(feature1Max)*1.2, 'g', zorder=100)
    #plt.ylim(min(feature1Min)*0.8, max(feature1Max)*1.2)
    plt.legend(['1 Min', '2 Min', '3 Min'])
    plt.title("Averaged Together: " + str(averageTogether/60) + " Min")
    fig.savefig(saveFolder + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
    plt.show()






time = signalData[:,0][signalLabels == 2]
for compareFeatureInd in range(1,len(blinkFeatures)):
    saveFolder = '../Feature Comparison/' + blinkFeatures[compareFeatureInd] + "/"
    os.makedirs(saveFolder, exist_ok=True)

    compareFeature = signalData[:,compareFeatureInd][signalLabels == 2]
    for featureInd in range(1,len(blinkFeatures)):
        fig, ax = plt.subplots(figsize=(12, 6))
    
        allFeatures = signalData[:,featureInd][signalLabels == 2]
        colors = ['ko', 'ro', 'bo', 'go', 'mo']
        for ind, averageTogether in enumerate([60*10E-10]):
            features = []
            for pointInd in range(len(allFeatures)):
                featureInterval = allFeatures[time > time[pointInd] - averageTogether]
                timeMask = time[time > time[pointInd] - averageTogether]
                featureInterval = featureInterval[timeMask <= time[pointInd]]
    
                featute = stats.trim_mean(featureInterval, 0.3)
                features.append(featute)
    
            features = np.array(features)
            plt.scatter(time, features, c=compareFeature, ec='k')
            plt.plot(time, featureDict[blinkFeatures[featureInd]], 'r')
    
    
        plt.xlabel("Time (Seconds)")
        plt.ylabel(blinkFeatures[featureInd])
        vLines = np.array([1, 2, 3, 4, 5])*60*6
        for i,vLine in enumerate(vLines):
            ax.axvline(vLine, color = 'g')
        #plt.ylim(min(feature1Min)*0.8, max(feature1Max)*1.2)
        #plt.legend(['1 Min', '2 Min', '3 Min'])
        plt.title("Color: " + blinkFeatures[compareFeatureInd])
        fig.savefig(saveFolder + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
        plt.show()
        
        
import matplotlib.pyplot as plt
saveFolder = '../Vol Vs Invol Blinks/'
os.makedirs(saveFolder, exist_ok=True)
colors = ['ko', 'ro', 'bo', 'go', 'mo']
signalData = np.array(signalData); signalLabels = np.array(signalLabels)

for featureInd in range(1,len(blinkFeatures)):
    fig = plt.subplots()

    allFeatures0 = signalData[:,featureInd][signalLabels == 0]
    allFeatures1 = signalData[:,featureInd][signalLabels == 1]

    plt.plot(allFeatures0, 'bo', markersize=5, zorder=100)
    plt.plot(allFeatures1, 'ro', markersize=5)


    plt.xlabel("Time (Seconds)")
    plt.ylabel(blinkFeatures[featureInd])
    plt.legend(['Natural Blink', 'Forced Blink'])
    #fig.savefig(saveFolder + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    
    
# --------- Analyze Peak Distribution


from scipy import stats
import matplotlib.pyplot as plt
signalData = np.array(signalData); signalLabels = np.array(signalLabels)
time = signalData[:,0][signalLabels == 2]
for featureInd in range(len(blinkFeatures)):
    fig = plt.figure()

    allFeatures = signalData[:,featureInd][signalLabels == 2]
    colors = ['k-o', 'r-o', 'bo', 'go', 'mo']
    for ind, averageTogether in enumerate([60*3]):
        stopLines = np.array([0, 1, 2, 3, 4, 5, 6, 7])*60*6
        stopLineIndex = 1
        a = [[]]
        for pointInd in range(len(allFeatures)):
            featureInterval = allFeatures[time > time[pointInd] - averageTogether]
            timeMask = time[time > time[pointInd] - averageTogether]
            featureInterval = featureInterval[timeMask <= time[pointInd]]

            if stopLines[stopLineIndex - 1] < time[pointInd] - 60:
                a[-1].extend(featureInterval)
                

            if stopLines[stopLineIndex] < time[pointInd]:
                a.append([])
        for ai in enumerate(a):
            plt.hist(ai, bins=25, alpha=0.5, label=str(i))
        # plt.xlabel("Time (Seconds)")
        plt.ylabel(blinkFeatures[featureInd])
        #plt.ylim(min(feature1Min)*0.8, max(feature1Max)*1.2)
        #plt.legend(['3 Min'])
        plt.title("Interval: " + str(stopLineIndex-1))
        #fig.savefig('../Del Josh/Blink Distribution/' + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
        plt.show()
        stopLineIndex += 1

# -----------


# --------- Analyze Peaks

from scipy import stats
import matplotlib.pyplot as plt
signalData = np.array(signalData); signalLabels = np.array(signalLabels)
time = signalData[:,0][signalLabels == 2]
for featureInd in range(len(blinkFeatures)):
    fig = plt.figure()

    allFeatures = signalData[:,featureInd][signalLabels == 2]
    colors = ['k-o', 'r-o', 'bo', 'go', 'mo']
    for ind, averageTogether in enumerate([60*3]):
        features = []
        for pointInd in range(len(allFeatures)):
            featureInterval = allFeatures[time > time[pointInd] - averageTogether]
            timeMask = time[time > time[pointInd] - averageTogether]
            featureInterval = featureInterval[timeMask <= time[pointInd]]
            
            feature = stats.trim_mean(featureInterval, 0.3)
            features.append(feature)

        features = np.array(features)
        plt.plot(time, features, colors[ind], markersize=5)


    plt.xlabel("Time (Seconds)")
    plt.ylabel(blinkFeatures[featureInd])
    plt.vlines(np.array([1, 2, 3, 4, 5])*60*6, min(features)*0.9, max(features)*1.1, 'g', zorder=100)
    #plt.ylim(min(feature1Min)*0.8, max(feature1Max)*1.2)
    plt.legend(['3 Min'])
    plt.title("Averaged Together: " + str(averageTogether/60) + " Min")
    fig.savefig('../Del Josh/All Blinks/' + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
    plt.show()

# -----------


# ----------- PANAS

positive1 = [12, 14, 13, 10, 10, 10]
negative1 = [13, 10, 13, 10, 14, 10]
timeIntervals = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])*60*5

fig = plt.figure()
positive = []; negative = []
for i in range(len(positive1)):
    for _ in range(2):
        positive.append(positive1[i])
        negative.append(negative1[i])

plt.plot(timeIntervals, positive, 'b-')
plt.plot(timeIntervals, negative, 'r-')

plt.xlabel("Time (Seconds)")
plt.ylabel('PANAS Score')
plt.legend(['Positive', 'Negative'])
plt.vlines(np.array([1, 2, 3, 4, 5])*60*5, min(min(positive1, negative1))*0.9, 1.1*max(max(positive1, negative1)), 'g', zorder=100)
plt.title("PANAS Score")
fig.savefig('../Del Sarah/PANAS.png', dpi=300, bbox_inches='tight')
plt.show()
# -----------

# --------- Scale by allFeatures
from scipy import stats
import matplotlib.pyplot as plt
signalData = np.array(signalData); signalLabels = np.array(signalLabels)

signalDataJose = np.array(signalData); signalLabelsJose = np.array(signalLabels)
signalDataJiahong = np.array(signalData); signalLabelsJiahong = np.array(signalLabels)
signalDataSam = np.array(signalData); signalLabelsSam = np.array(signalLabels)

signalDataJose[:,0] = signalDataJose[:,0] - signalDataJose[:,0][0]
signalDataJiahong[:,0]  = signalDataJiahong[:,0] - signalDataJiahong[:,0][0]
signalDataSam[:,0] = signalDataSam[:,0] - signalDataSam[:,0][0]

time1 = np.array(signalDataJose[:,0] - signalDataJose[:,0][0])
time2 = np.array(signalDataJiahong[:,0] - signalDataJiahong[:,0][0])
time3 = np.array(signalDataSam[:,0] - signalDataSam[:,0][0])
time = [time1, time2, time3]

featureDict = [{}, {}, {}]

for featureIndDict in range(1,len(blinkFeatures)):
    saveDataFolder = '../Rest Data/featureScale/' + blinkFeatures[featureIndDict] + "/"
    os.makedirs(saveDataFolder, exist_ok=True)
    if np.mean(featureDict[i][blinkFeatures[featureIndDict]]) == 0:
        continue

    for featureInd in range(len(blinkFeatures)):
        fig = plt.figure()

        allFeatures1 = signalDataJose[:,featureInd]
        allFeatures2 = signalDataJiahong[:,featureInd]
        allFeatures3 = signalDataSam[:,featureInd]
        allFeatures = [allFeatures1, allFeatures2, allFeatures3]


        colors = ['ko', 'ro', 'bo', 'go', 'mo']
        for ind, averageTogether in enumerate([60*3]):
            features = [[] for _ in range(len(time))]
            for i, allFeaturesI in enumerate(allFeatures):
                for pointInd in range(len(allFeaturesI)):
                    featureInterval = allFeaturesI[time[i] > time[i][pointInd] - averageTogether]
                    timeMask = time[i][time[i] > time[i][pointInd] - averageTogether]
                    featureInterval = featureInterval[timeMask <= time[i][pointInd]]

                    feature = stats.trim_mean(featureInterval, 0.3)
                    features[i].append(feature)

            for i in range(len(time)):
                #features[i] = np.array(features[i])
                #featureDict[i][blinkFeatures[featureInd]] = features[i]

                features[i] = np.array(features[i])/featureDict[i][blinkFeatures[featureIndDict]]

                plt.plot(time[i], features[i], colors[i], markersize=5)


        plt.xlabel("Time (Seconds)")
        plt.ylabel(blinkFeatures[featureInd])
        plt.legend(['JO','JH','SM'])
        plt.title("Scaled by " + blinkFeatures[featureIndDict])
        fig.savefig(saveDataFolder + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
        plt.show()
        
        fig.clear()
        plt.close(fig)
        plt.cla()
        plt.clf()
            
# -----------

# --------- Scale by All Feature Point

from scipy import stats
import matplotlib.pyplot as plt
signalData = np.array(signalData); signalLabels = np.array(signalLabels)

signalDataJose = np.array(signalData); signalLabelsJose = np.array(signalLabels)
signalDataJiahong = np.array(signalData); signalLabelsJiahong = np.array(signalLabels)
signalDataSam = np.array(signalData); signalLabelsSam = np.array(signalLabels)

signalDataJose[:,0] = signalDataJose[:,0] - signalDataJose[:,0][0]
signalDataJiahong[:,0]  = signalDataJiahong[:,0] - signalDataJiahong[:,0][0]
signalDataSam[:,0] = signalDataSam[:,0] - signalDataSam[:,0][0]

time1 = np.array(signalDataJose[:,0] - signalDataJose[:,0][0])
time2 = np.array(signalDataJiahong[:,0] - signalDataJiahong[:,0][0])
time3 = np.array(signalDataSam[:,0] - signalDataSam[:,0][0])
time = [time1, time2, time3]


for featureIndDict in range(1,len(blinkFeatures)):
    saveDataFolder = '../Rest Data/featureScalePoint/' + blinkFeatures[featureIndDict] + "/"
    os.makedirs(saveDataFolder, exist_ok=True)

    for featureInd in range(len(blinkFeatures)):
        fig = plt.figure()

        allFeatures1 = signalDataJose[:,featureInd]/signalDataJose[:,featureIndDict]
        allFeatures2 = signalDataJiahong[:,featureInd]/signalDataJiahong[:,featureIndDict]
        allFeatures3 = signalDataSam[:,featureInd]/signalDataSam[:,featureIndDict]
        allFeatures = [allFeatures1, allFeatures2, allFeatures3]


        colors = ['ko', 'ro', 'bo', 'go', 'mo']
        for ind, averageTogether in enumerate([60*3]):
            features = [[] for _ in range(len(time))]
            for i, allFeaturesI in enumerate(allFeatures):
                for pointInd in range(len(allFeaturesI)):
                    featureInterval = allFeaturesI[time[i] > time[i][pointInd] - averageTogether]
                    timeMask = time[i][time[i] > time[i][pointInd] - averageTogether]
                    featureInterval = featureInterval[timeMask <= time[i][pointInd]]

                    feature = stats.trim_mean(featureInterval, 0.3)
                    features[i].append(feature)

            for i in range(len(time)):
                features[i] = np.array(features[i])
                plt.plot(time[i], features[i], colors[i], markersize=5)


        plt.xlabel("Time (Seconds)")
        plt.ylabel(blinkFeatures[featureInd])
        plt.legend(['JO','JH','SM'])
        plt.title("Scaled by " + blinkFeatures[featureIndDict])
        fig.savefig(saveDataFolder + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
        
        fig.clear()
        plt.close(fig)
        plt.cla()
        plt.clf()
    
# -----------


# --------- Analyze the Effect of Culling (Need Two Runs)

from scipy import stats
import matplotlib.pyplot as plt
signalData = np.array(signalData); signalLabels = np.array(signalLabels)

signalDataJose = np.array(signalData); signalLabelsJose = np.array(signalLabels)
signalDataJiahong = np.array(signalData); signalLabelsJiahong = np.array(signalLabels)
signalDataSam = np.array(signalData); signalLabelsSam = np.array(signalLabels)

signalDataJose[:,0] = signalDataJose[:,0] - signalDataJose[:,0][0]
signalDataJiahong[:,0]  = signalDataJiahong[:,0] - signalDataJiahong[:,0][0]
signalDataSam[:,0] = signalDataSam[:,0] - signalDataSam[:,0][0]

time1 = np.array(signalDataJose[:,0] - signalDataJose[:,0][0])
time2 = np.array(signalDataJiahong[:,0] - signalDataJiahong[:,0][0])
time3 = np.array(signalDataSam[:,0] - signalDataSam[:,0][0])
time = [time1, time2, time3]


a = []
b = []
c = []

for featureInd in range(len(blinkFeatures)):
    fig = plt.figure()

    allFeatures1 = signalDataJose[:,featureInd]
    allFeatures2 = signalDataJiahong[:,featureInd]
    allFeatures3 = signalDataSam[:,featureInd]
    allFeatures = [allFeatures1, allFeatures2, allFeatures3]


    colors = ['ko', 'ro', 'bo', 'go', 'mo']
    for ind, averageTogether in enumerate([60*3]):
        features = [[] for _ in range(len(time))]
        for i, allFeaturesI in enumerate(allFeatures):
            for pointInd in range(len(allFeaturesI)):
                featureInterval = allFeaturesI[time[i] > time[i][pointInd] - averageTogether]
                timeMask = time[i][time[i] > time[i][pointInd] - averageTogether]
                featureInterval = featureInterval[timeMask <= time[i][pointInd]]

                feature = stats.trim_mean(featureInterval, 0.3)
                features[i].append(feature)

        for i in range(len(time)):
            features[i] = np.array(features[i])
            plt.plot(time[i], features[i], colors[i], markersize=5)
            
    avJO = np.round(np.mean(features[0][time[0] > 200]), 7)
    avJH = np.round(np.mean(features[1][time[1] > 200]), 7)
    avSM = np.round(np.mean(features[2][time[2] > 200]), 7)
    
    a.append(avJO/avJH)
    b.append(avJO/avSM)
    c.append(avJH/avSM)
    
    plt.xlabel("Time (Seconds)")
    plt.ylabel(blinkFeatures[featureInd])
    plt.legend(['JO -> Av: ' + str(avJO),
        'JH- > Av: ' + str(avJH),
        'SM -> Av: ' + str(avSM) ])
    plt.title("JO/JH: " + str(np.round(avJO/avJH,6)) + " | JO/SM: " + str(np.round(avJO/avSM,6)) + " | JH/SM: " + str(np.round(avJH/avSM,6)))
    fig.savefig('../Rest Data/Compare/' + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
    plt.show()


a = [v for v in a if not (math.isinf(v) or math.isnan(v) or v == 0)]
b = [v for v in b if not (math.isinf(v) or math.isnan(v) or v == 0)]
c = [v for v in c if not (math.isinf(v) or math.isnan(v) or v == 0)]

plt.hist(a, bins=100, alpha=0.5, label="avJO/avJH")
plt.hist(b, bins=100, alpha=0.5, label="avJO/avSM")
plt.hist(c, bins=100, alpha=0.5, label="avJH/avSM")

# -----------

# --------- Correlation Matrix
import seaborn as sns;
from copy import copy, deepcopy
signalData = np.array(signalData); signalLabels = np.array(signalLabels)

signalDataSam = signalData; signalLabelsSam = signalLabels
signalDataJosh = signalData; signalLabelsJosh = signalLabels
signalDataSarah = signalData; signalLabelsSarah = signalLabels

signalData = deepcopy(signalDataSarah); signalLabels = deepcopy(signalLabelsSarah)
signalData.extend(signalDataJosh); signalLabels.extend(signalLabelsJosh)
signalData.extend(signalDataSam); signalLabels.extend(signalLabelsSam)


signalData = deepcopy(signalData); signalLabels = deepcopy(signalLabels)
# Standardize Feature
signalDataStandard = deepcopy(signalData[signalData[:,0] > 60*6])
blinkFeaturesX = np.array(blinkFeatures)
blinkFeaturesY = np.array(blinkFeatures)
# for i in range(len(signalDataStandard[0])):
#      signalDataStandard[:,i] = (signalDataStandard[:,i] - np.mean(signalDataStandard[:,i]))/np.std(signalDataStandard[:,i],ddof=1)

matrix = np.array(np.corrcoef(signalDataStandard.T)); 
#sns.set_theme(); ax = sns.heatmap(matrix, cmap='icefire', xticklabels=blinkFeaturesX, yticklabels=blinkFeaturesY)

# Cluster
for i in range(1,len(matrix)):
    blinkFeaturesX = blinkFeaturesX[matrix[:,i].argsort()]
    matrix = matrix[matrix[:,i].argsort()]
for i in range(1,len(matrix[0])):
    blinkFeaturesY = blinkFeaturesY[matrix[i].argsort()]
    matrix = matrix [ :, matrix[i].argsort()]

sns.set_theme(); ax = sns.heatmap(matrix, cmap='icefire', xticklabels=blinkFeaturesX, yticklabels=blinkFeaturesY)

sns.set(rc={'figure.figsize':(50,35)})
fig = ax.get_figure(); fig.savefig("../output.png", dpi=300)


for i in range(len(matrix)):
    for j in range(len(matrix)):
        if abs(matrix[i][j]) < 0.96:
            matrix[i][j] = 0
            
# -----------

"""


