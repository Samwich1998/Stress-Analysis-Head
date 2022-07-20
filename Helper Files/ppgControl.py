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

sys.path.append('./Data Aquisition and Analysis/Biolectric Protocols/')  # Folder with Data Aquisition Files
import ppgAnalysis as ppgAnalysis         # Functions to Analyze the EOG Data

# Import Data Aquisition and Analysis Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import excelProcessing as excelData         # Functions to Save/Read in Data from Excel
import readDataArduino as streamData      # Functions to Handle Data from Arduino

# Import Files for Machine Learning
sys.path.append('./Machine Learning/')  # Folder with Machine Learning Files
import machineLearningMain  # Class Header for All Machine Learning
import featureAnalysis      # Functions for Feature Analysis

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # Arduino Streaming Parameters
    arduinoSerialNum = '044F979E50553133352E3120FF030F25'         # Arduino's Serial Number (port.serial_number)
    stopTimeStreaming = 60*2      # The Last Time to Stream data into the arduino. If Float, it is the total seconds; If String, it is the TimeStamp to Stop (Military Time) as "Hours:Minutes:Seconds:MicroSeconds"
    
    # Protocol Switches: Only the First True Variable Excecutes
    streamArduinoData = False      # Stream in Data from the Arduino and Analyze; Input 'controlVR' = True to Move VR
    readDataFromExcel = True       # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    
    # User Options During the Run: Any Number Can be True
    plotStreamedData = True       # Graph the Data to Show Incoming Signals + Analysis
    saveInputData = True          # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'
    
    # ---------------------------------------------------------------------- #
    
    streamingMap = ["ppg", "ppg"]           # A List Representing the Order of the Sensors being Streamed in.
    # Define Batches to Analyze and Plot the Data
    numTimePoints = 1000                    # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    moveDataFinger = 100                    # The Number of Data Points to Plot/Analyze at a Time; My Beta-Test Used 200 Points with Plotting; 10 Points Without
    
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveInputData:
        saveExcelName = "Samuel Solomon 2022-05-30 Finger 411.xlsx"  # The Name of the Saved File
        saveDataFolder = "../Data/PPG Data/All Data/Industry Electrodes/General Data Collection/"   # Data Folder to Save the Excel Data; MUST END IN '/'
    
    # Instead of Arduino Data, Use Test Data from Excel File
    if readDataFromExcel:
        testDataExcelFile = "../Data/PPG Data/All Data/Industry Electrodes/General Data Collection/Samuel Solomon 2022-05-30 Second Attempt.xlsx" # Path to the Test Data
        testSheetNum = 0   # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
        saveInputData = False
            
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #           Data Collection Program (Should Not Have to Edit)            #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    # Stream in Data from Arduino
    if streamArduinoData:
        readData = streamData.mainArduinoRead(arduinoSerialNum, numTimePoints, moveDataFinger, streamingMap, plotStreamedData, guiApp = None)
        readData.streamData(stopTimeStreaming, predictionModel = None, actionControl = None, calibrateModel = False)
    # Take Data from Excel Sheet
    elif readDataFromExcel:
        # Collect the Data from Excel
        ppgData = excelData.getExcelData().getData(testDataExcelFile, numberOfChannels = len(streamingMap), testSheetNum = testSheetNum)
        # Analyze the Data using the Correct Protocol
        readData = excelData.processExcelData(numTimePoints, moveDataFinger, plotStreamedData, streamingMap)
        readData.streamExcelData(ppgData)
    
    # Extract the PPG Data
    timePPG = np.array(readData.ppgAnalysis.data[0])
    redLED = np.array(readData.ppgAnalysis.data[1][0])
    irLED = np.array(readData.ppgAnalysis.data[1][1])
        
    # ---------------------------------------------------------------------- #
    # -------------------------- Save Input data --------------------------- #
    # Save the Data in Excel: PPG Channels (Cols 1-4); X-Peaks (Cols 5-8); Peak Features (Cols 9-12)
    if saveInputData:
        # Double Check to See if User Wants to Save the Data
        verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
        if verifiedSave.upper() == "Y":
            # Initialize Class to Save the Data and Save
            saveInputs = excelData.saveExcel()
            readData.ppgAnalysis.filteredData[0].extend([0]*(len(timePPG) - len(readData.ppgAnalysis.filteredData[1])))
            readData.ppgAnalysis.filteredData[1].extend([0]*(len(timePPG) - len(readData.ppgAnalysis.filteredData[1])))
            saveInputs.saveData(timePPG, readData.ppgAnalysis.data[1], readData.ppgAnalysis.filteredData, streamingMap, saveDataFolder, saveExcelName, "Trial 1 - PPG")
        else:
            print("User Chose Not to Save the Data")
    
    
    
from scipy.fftpack import ifft, idct, dct, dctn


import matplotlib.pyplot as plt
from scipy.fft import rfft,rfftfreq
from scipy.fft import irfft
import time

# Basic Modules
import numpy as np
# Filtering Modules
import scipy
from scipy.signal import butter
import math


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

irLED = np.max(irLED) - irLED

t1 = time.time()
irLED_Filt = butterFilter(eogVertical, 25, 383, order = 3, filterType = 'low')
t2 = time.time()
print(t2-t1)

t = timePPG
f_noise = list(eogVertical)

t1 = time.time()

# Pad the Data with Zeros
closestPowerOfTwo = 2**(math.ceil(math.log(len(f_noise))/math.log(2)))
numZerosToPad = closestPowerOfTwo - len(f_noise)
f_noisePadded = [0]*numZerosToPad
f_noisePadded.extend(f_noise)
# Pad the Data with Mirror
f_noisePadded.extend(f_noisePadded[::-1])
# Tranform the Data into the Frequency Domain
n    = len(f_noisePadded)
yf   = rfft(f_noisePadded)
xf   = rfftfreq(n, 1/105)

plt.plot(xf,np.abs(yf))    
plt.ylim(0,50)
plt.show()

freqCutoff = [0.04, 15]
yf_abs      = np.abs(yf) 
indices     = np.logical_and(freqCutoff[0] < xf, xf < freqCutoff[1])   # filter out those value under 300
yf_clean    = indices * yf # noise frequency will be set to 0
plt.plot(xf,np.abs(yf_clean))  
plt.ylim(0,100)
plt.show()  

f_noise = f_noise[0:len(timePPG)]
new_f_clean = irfft(yf_clean)[numZerosToPad:numZerosToPad+len(timePPG)]
# plt.plot(timePPG,f_noise, 'k', linewidth=2)
# plt.plot(timePPG,new_f_clean, 'tab:red', linewidth=2)
# plt.show()
    

t2 = time.time()
print(t2-t1)
    
    
    
    

    

