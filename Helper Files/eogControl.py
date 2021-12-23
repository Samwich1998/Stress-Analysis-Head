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
import os
import sys
import threading
import numpy as np
from pathlib import Path
# Import Data Aquisition and Analysis Files
sys.path.append('./Data Aquisition and Analysis/Biolectric Protocols/')  # Folder with Data Aquisition Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import readDataExcel as excelData         # Functions to Save/Read in Data from Excel
import readDataArduino as streamData      # Functions to Read in Data from Arduino
import eogAnalysis as eogAnalysis         # Functions to Analyze the EOG Data
# Import Virtual Reality Control Files
sys.path.append('./Execute Movements/Virtual Reality Control/')   # Folder with Virtual Reality Control Files

# Import Files for Machine Learning
sys.path.append('./Machine Learning/')  # Folder with Machine Learning Files
import machineLearningMain  # Class Header for All Machine Learning


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # General Data Collection Information (You Will Likely Not Edit These)
    eogSerialNum = '85035323234351D06052'  # Arduino's Serial Number (port.serial_number)
    
    eogSerialNum = '85035323234351D06052'
    eogSerialNum = '044F979E50553133352E3120FF030F25'
    
    samplingFreq = None            # The Average Number of Points Steamed Into the Arduino Per Second; If NONE Given, Algorithm will Calculate Based on Initial Data
    numDataPoints = 1*50000         # The Number of Points to Stream into the Arduino
    numTimePoints = 15000 #2048576           # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    moveDataFinger = 200 #1048100         # The Number of Data Points to Plot/Analyze at a Time; My Beta-Test Used 200 Points with Plotting; 10 Points Without
    numChannels = 2                # The Number of Arduino Channels with EOG Signals Read in; My Beta-Test Used 4 Channels
    # Specify the Type of Movements to Learn
    gestureClasses = np.char.lower(['Spontaneous', 'Reflex', 'Voluntary', 'Double'])  # Define Labels as Array
    gestureClasses = np.char.lower(['Up', 'Down', 'Blink', 'Double Blink', 'Relaxed', 'Relaxed to Cold', 'Cold'])  # Define Labels as Array
    
    
    gestureClasses = np.char.lower(['Blink', 'Double Blink', 'Relaxed', 'Stroop Test', 'Exercise Weight', 'VR Roller Coaster'])  # Define Labels as Array
    machineLearningClasses = np.char.lower(['Relaxed', 'Stroop', 'Exercise', 'VR'])
    labelMap = [1, -1, 0, -1, -1, -1]
    
    gestureClasses = np.char.lower(['Morning', 'Night', 'Music'])
    machineLearningClasses = np.char.lower(['Morning', 'Night', 'Music'])
    labelMap = [-1, -1, 2]
    
    #gestureClasses = np.char.lower(['blink', 'relaxed'])
    #machineLearningClasses = np.char.lower(['blink', 'relaxed'])
    #labelMap = [0, 1]
    
    # Protocol Switches: Only the First True Variable Excecutes
    streamArduinoData = True     # Stream in Data from the Arduino and Analyze; Input 'controlVR' = True to Move VR
    readDataFromExcel = False     # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    reAnalyzePeaks = False        # Read in ALL Data Under 'trainDataExcelFolder', and Reanalyze Blinks (THIS EDITS EXCEL DATA IN PLACE!; DONT STOP PROGRAM MIDWAY)
    trainModel = False             # Read in ALL Data Under 'neuralNetworkFolder', and Train the Data
    
    # User Options During the Run: Any Number Can be True
    plotStreamedData = False      # Graph the Data to Show Incoming Signals + Analysis
    calibrateModel = False        # Calibrate the EOG Voltage to Predict the Eye's Angle
    saveData = False               # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'
    testModel = False             # Apply the Learning Algorithm to Decode the Signals
    controlVR = False             # Apply the Algorithm to Control the Virtual Reality View    
    
    
    # Finalize the Features
    blinkFeatures = ['peakLocX', 'blinkHeight', 'peakTentY', 'tentDeviationX', 'tentDeviationY', 'blinkAmpRatio', 'tentDeviationRatio']
    blinkFeatures.extend(['closingAmpDiff1', 'closingAmpDiff2', 'closingAmpDiff3', 'openingAmpDiff1', 'openingAmpDiff2', 'openingAmpDiff3'])
    blinkFeatures.extend(['closingAmpDiffRatio1', 'closingAmpDiffRatio2', 'closingAmpDiffRatio3', 'openingAmpDiffRatio1', 'openingAmpDiffRatio2', 'openingAmpDiffRatio3'])
    blinkFeatures.extend(['accel0ToVel0Ratio', 'accel1ToVel0Ratio', 'accel2ToVel1Ratio', 'accel3ToVel1Ratio'])
    blinkFeatures.extend(['riseTimePercent', 'dropTimePercent', 'velDiffPercent'])
    blinkFeatures.extend(['closingAmpRatio1', 'closingAmpRatio2', 'closingAmpRatio3', 'openingAmpRatio1', 'openingAmpRatio2', 'openingAmpRatio3'])

    blinkFeatures.extend(['blinkDuration', 'closingTime', 'openingTime', 'closingFraction', 'openingFraction', 'halfClosedTime', 'eyesClosedTime', 'percentTimeClosed'])
    blinkFeatures.extend(['closingSlope0', 'closingSlope1', 'closingSlope2', 'openingSlope1', 'openingSlope2', 'openingSlope3'])
    blinkFeatures.extend(['closingSlopeDiff10', 'closingSlopeDiff12', 'openingSlopeDiff21', 'openingSlopeDiff23'])
    blinkFeatures.extend(['closingSlopeRatio0', 'closingSlopeRatio1', 'closingSlopeRatio2', 'openingSlopeRatio1', 'openingSlopeRatio2', 'openingSlopeRatio3'])
    blinkFeatures.extend(['peakAverage', 'peakAverageRatio', 'peakIntegral', 'peakIntergralRatio', 'peakEntropy', 'peakSkew', 'peakKurtosis', 'peakSTD', 'maxCurvature'])
    # Compile the Features
    blinkFeatures.extend(['peakClosingVel', 'peakOpeningVel', 'peakClosingAccel1', 'peakClosingAccel2', 'peakopeningAccel1', 'peakopeningAccel2'])
    blinkFeatures.extend(['velOpenRatio', 'velClosedRatio', 'accelClosedRatio1', 'accelClosedRatio2', 'accelOpenRatio1', 'accelOpenRatio2'])
    blinkFeatures.extend(['velClosedVal', 'velOpenVal', 'accelClosedVal1', 'accelClosedVal2', 'accelOpenVal1', 'accelOpenVal2'])
    blinkFeatures.extend(['velRatio', 'accelRatio1', 'accelRatio2', 'velRatioYData', 'accelRatioYData1', 'accelRatioYData2'])
    blinkFeatures.extend(['durationByVel1', 'durationByVel2', 'durationByAccel1', 'durationByAccel2', 'durationByAccel3', 'midDurationRatio'])
    blinkFeatures.extend(['startToAccel', 'accelCloseingPeakDuration', 'accelToPeak', 'peakToAccel', 'accelOpeningPeakDuration', 'accelToEnd'])
    blinkFeatures.extend(['velPeakDuration', 'startToVel', 'velToPeak', 'peakToVel', 'velToEnd'])
    blinkFeatures.extend(['curvatureYDataAccel0', 'curvatureYDataAccel1', 'curvatureYDataAccel2', 'curvatureYDataAccel3', 'curvatureYDataVel0', 'curvatureYDataVel1'])
    blinkFeatures.extend(['velFullSTD', 'accelFullSTD', 'thirdDerivFullSTD', 'velSTD', 'accelSTD', 'thirdDerivSTD'])
    blinkFeatures.extend(['velFullEntropy', 'accelFullEntropy', 'thirdDerivFullEntropy', 'velEntropy', 'accelEntropy', 'thirdDerivEntropy'])
    # ------------------------ Dependant Parameters ------------------------- #
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveData:
        saveExcelName = "Blink Types.xlsx"  # The Name of the Saved File
        saveDataFolder = "../Data/EOG Data/All Data/Industry Electrodes/2021-12-20 Combined Blinks/"   # Data Folder to Save the Excel Data; MUST END IN '/'
        # Speficy the eye Movement You Will Perform
        eyeMovement = "Blink".lower() # Make Sure it is Lowercase
        if eyeMovement not in gestureClasses:
            print("The Gesture", "'" + eyeMovement + "'", "is Not in", gestureClasses)
            
    # Instead of Arduino Data, Use Test Data from Excel File
    if readDataFromExcel:
        #testDataExcelFile = "../Data/EOG Data/All Data/Industry Electrodes/2021-12-01 First Cold Water Test/Jiahong 2021-12-1 Movements.xlsx" # Path to the Test Data
        #testDataExcelFile = "../Data/EOG Data/All Data/Industry Electrodes/2021-12-10 First VR Test/Jose 2021-12-10 Movements.xlsx" # Path to the Test Data
        testDataExcelFile = "../Data/EOG Data/All Data/Industry Electrodes/2021-12-15 Music Study Lab/Emil 2021-12-11 Lab Music.xlsx" # Path to the Test Data
        testSheetNum = 1   # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
    # Input Training Paramaters 
    if reAnalyzePeaks or trainModel:
        trainDataExcelFolder = "../Data/EOG Data/All Data/Industry Electrodes/2021-12-10 First VR Test/"
        #trainDataExcelFolder = "../Data/EOG Data/All Data/Industry Electrodes/2021-12-11 Home Study/"  # Path to the Training Data Folder; All .xlsx Data Used
        trainDataExcelFolder = "../Data/EOG Data/All Data/Industry Electrodes/2021-12-15 Music Study Home/"  # Path to the Training Data Folder; All .xlsx Data Used
        #trainDataExcelFolder = "../Data/EOG Data/All Data/Industry Electrodes/2021-12-20 Combined Blinks/"  # Path to the Training Data Folder; All .xlsx Data Used

    # Train or Test the Data with the Machine Learning Model
    if trainModel or testModel and not reAnalyzePeaks:
        # Pick the Machine Learning Module to Use
        modelType = "RF"  # Machine Learning Options: NN, RF, LR, KNN, SVM
        modelPath = "./Machine Learning/Models/predictionModelRF_12-13-2021.pkl" # Path to Model (Creates New if it Doesn't Exist)
        # Choos the Folder to Save ML Results
        if trainModel:
            saveModel = True  # Save the Machine Learning Model for Later Use
            saveDataFolder = trainDataExcelFolder + "Data Analysis/" + modelType + "/"
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
        global readData, eogProtocol, performMachineLearning, signalData, signalLabels, modelPath
        
        eogProtocol = eogAnalysis.eogProtocol(numTimePoints, moveDataFinger, numChannels, samplingFreq, plotStreamedData)
        # Stream in Data from Arduino
        if streamArduinoData:
            arduinoRead = streamData.arduinoRead(eogSerialNum = eogSerialNum, ppgSerialNum = None, emgSerialNum = None, eegSerialNum = None, handSerialNum = None)
            readData = streamData.eogArduinoRead(arduinoRead, numTimePoints, moveDataFinger, numChannels, samplingFreq, plotStreamedData, guiApp = None)
            readData.streamEOGData(numDataPoints, predictionModel = predictionModel, actionControl = gazeControl, calibrateModel = calibrateModel)
        # Take Data from Excel Sheet
        elif readDataFromExcel:
            readData = excelData.readExcel(eogProtocol)
            readData.streamExcelData(testDataExcelFile, plotStreamedData, testSheetNum, predictionModel = predictionModel, actionControl = None)
        # Redo Peak Analysis
        elif reAnalyzePeaks:
            readData = excelData.readExcel(eogProtocol)
            readData.getTrainingData(trainDataExcelFolder, gestureClasses, blinkFeatures, labelMap, mode='reAnalyze')
        # Take Preprocessed (Saved) Features from Excel Sheet
        elif trainModel:
            # Extract the Data
            readData = excelData.readExcel(eogProtocol)
            signalData, signalLabels = readData.getTrainingData(trainDataExcelFolder, gestureClasses, blinkFeatures, labelMap, mode='Train')
            print("\nCollected Signal Data")
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
            saveInputs = excelData.saveExcel(numChannels)
            saveInputs.saveData(readData.data, [], [], saveDataFolder, saveExcelName, sheetName, eyeMovement)
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


plt.plot(xData, yData)
plt.xlim(122.5,123)

# Calculate Derivatives
dx_dt = np.gradient(xData); dy_dt = np.gradient(yData)
d2x_dt2 = np.gradient(dx_dt); d2y_dt2 = np.gradient(dy_dt,2)
d2y_dt2_ABS = abs(d2y_dt2)
# Calculate Peak Shape parameters
speed = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
acceleration = np.sqrt(d2x_dt2 * d2x_dt2 + d2y_dt2 * d2y_dt2)
curvature = np.abs((d2x_dt2 * dy_dt - dx_dt * d2y_dt2)) / speed**3
# Pull Out Peak Shape Features
velInds = scipy.signal.find_peaks(speed, prominence=.000001, width=20, height=np.mean(speed))[0];
accelInds = scipy.signal.find_peaks(d2y_dt2_ABS, prominence=.000001, width=10, height=np.mean(d2y_dt2_ABS))[0];

plt.plot(xData[accelInds], (d2y_dt2_ABS*2/max(d2y_dt2_ABS))[accelInds], 'o')
plt.plot(xData, d2y_dt2_ABS*2/max(d2y_dt2_ABS))
plt.plot(xData, speed*2/max(speed))
#plt.plot(xData[speedInds], (speed*2/max(speed))[speedInds], 'o')
#plt.plot(xData, d2y_dt2*2/max(d2y_dt2))
#plt.plot(xData[accelInds], (d2y_dt2*2/max(d2y_dt2))[accelInds])


importantFeatures = np.array(readData.analysisProtocol.importantArrays)
importantFeatures = np.array(readData.analysisProtocol.importantArrays)
i = 0
for curvature in importantFeatures[:,2]:
    i += 1
    plt.plot(curvature[int(len(curvature)/2)-30:int(len(curvature)/2)+31], 'o')
    if i == 10:
        break


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



Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.33, shuffle= True, stratify=signalLabels)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=len(Training_Data[0]), activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

epochs = 500; seeTrainingSteps = False
opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
loss = ['mean_squared_error', 'categorical_crossentropy']
metric = ['accuracy']

model.compile(optimizer = opt, loss = loss, metrics = list([metric]))

# For mini-batch gradient decent we want it small (not full batch) to better generalize data
max_batch_size = 32  # Keep Batch sizes relatively small (no more than 64 or 128)
mini_batch_gd = min(len(Training_Data)//4, max_batch_size)
mini_batch_gd = max(1, mini_batch_gd)  # For really small data samples at least take 1 data point
# For every Epoch (loop), run the Neural Network by:
    # With uninitialized weights, bring data through network
    # Calculate the loss based on the data
    # Perform optimizer to update the weights
history = model.fit(Training_Data, Training_Labels, validation_split=0.33, epochs=int(epochs), shuffle=True, batch_size = int(mini_batch_gd), verbose = seeTrainingSteps)
# Score the Model
results = model.evaluate(Testing_Data, Testing_Labels, batch_size=mini_batch_gd, verbose = seeTrainingSteps)
score = results[0]; accuracy = results[1]; 
print('Test score:', score)
print('Test accuracy:', accuracy)

pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')

from keras.utils.vis_utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True)

from ann_visualizer.visualize import ann_viz
ann_viz(model)




from sklearn.feature_selection import RFE
from sklearn.svm import SVR

blinkFeatures=np.array(blinkFeatures)
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(Training_Data, Training_Labels)
print(blinkFeatures[selector.support_])
selector.ranking_



mms = MinMaxScaler()
mms.fit(signalData)
data_transformed = mms.transform(signalData)

Sum_of_squared_distances = []
K = range(1,68)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# ----------- Histogram of Labels
import collections
classDistribution = collections.Counter(signalLabels)
print("Class Distribution:", classDistribution)
print("Number of Data Points = ", len(classDistribution))

import matplotlib.pyplot as plt
signalData = np.array(signalData); signalLabels = np.array(signalLabels)
for featureInd in range(len(blinkFeatures)):
    fig = plt.figure()

    blinkFeatures1 = signalData[:,featureInd][signalLabels == 0]
    relaxedFeatures = signalData[:,featureInd][signalLabels == 1]
    

    plt.hist(blinkFeatures1, bins=100, alpha=0.5, label="blinkFeatures")
    plt.hist(relaxedFeatures, bins=100, alpha=0.5, label="relaxedFeatures")
    plt.legend()
    plt.ylabel(blinkFeatures[featureInd])
    fig.savefig('../Blink Type Comparison/' + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
    plt.show()
# ----------------------



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
    

from scipy import stats
import matplotlib.pyplot as plt
signalData = np.array(signalData); signalLabels = np.array(signalLabels)
time = signalData[:,0][signalLabels == 2]
for featureInd in range(len(blinkFeatures)):
    fig = plt.figure()

    allFeatures1 = signalData[:,featureInd][signalLabels == 2]
    colors = ['ko', 'ro', 'bo', 'go', 'mo']
    for ind, averageTogether in enumerate([60*0.00001, 60*3]):
        entropy = allFeatures = signalData[:,54][signalLabels == 2]
        allFeatures = allFeatures1.copy()
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
    plt.legend(['10 Min', '3 Min'])
    plt.title("Averaged Together: " + str(averageTogether/60) + " Min")
    fig.savefig('../Del Josh/Culled Winks/' + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
    plt.show()
    

# --------- Analyze the Effect of Culling (Need Two Runs)

from scipy import stats
import matplotlib.pyplot as plt
signalData = np.array(signalData); signalLabels = np.array(signalLabels)

time1 = signalData[:,0][signalLabels == 2]
time2 = signalDataCull[:,0][signalLabelsCull == 2]

for featureInd in range(len(blinkFeatures)):
    fig = plt.figure()

    allFeatures1 = signalData[:,featureInd][signalLabels == 2]
    allFeatures2 = signalDataCull[:,featureInd][signalLabelsCull == 2]
    allFeatures = [allFeatures1, allFeatures2]
    
    time = [time1, time2]
    
    colors = ['ko', 'ro', 'bo', 'go', 'mo']
    for ind, averageTogether in enumerate([60*3]):
        features = [[],[]]
        for i, allFeaturesI in enumerate(allFeatures):
            for pointInd in range(len(allFeaturesI)):
                featureInterval = allFeaturesI[time[i] > time[i][pointInd] - averageTogether]
                timeMask = time[i][time[i] > time[i][pointInd] - averageTogether]
                featureInterval = featureInterval[timeMask <= time[i][pointInd]]
    
                featute = stats.trim_mean(featureInterval, 0.3)
                features[i].append(featute)

        plt.plot(time1, features[0], colors[0], markersize=5)
        plt.plot(time2, features[1], colors[1], markersize=5)


    plt.xlabel("Time (Seconds)")
    plt.ylabel(blinkFeatures[featureInd])
    plt.vlines(np.array([1, 2, 3, 4, 5])*60*6, min(min(features))*0.8, max(max(features))*1.2, 'g', zorder=100)
    plt.legend(['All', 'Cull'])
    plt.title("Averaged Together: " + str(averageTogether/60) + " Min")
    fig.savefig('../Del Josh/Compare/' + blinkFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
    plt.show()
# -----------
"""
