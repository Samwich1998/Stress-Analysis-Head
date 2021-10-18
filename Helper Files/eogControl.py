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
# Import Data Aquisition and Analysis Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Data Aquisition Files
import readDataExcel as excelData       # Functions to Save/Read in Data from Excel
import readDataArduino as streamData    # Functions to Read in Data from Arduino
import eogAnalysis as eogAnalysis
# Import Virtual Reality Control Files
sys.path.append('./Execute Movements/')  # Folder with Virtual Reality Control Files
import virtualRealityControl as vizardControl

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # General Data Collection Information (You Will Likely Not Edit These)
    eogSerialNum = '85035323234351D06052'#'85035323234351D06052'   # Arduino's Serial Number (port.serial_number)
    samplingFreq = 800       # The Average Number of Points Steamed Into the Arduino Per Second
    numDataPoints = 100000   # The Number of Points to Stream into the Arduino
    moveDataFinger = 5      # The Number of Data Points to Plot/Analyze at a Time; My Beta-Test Used 200 Points
    numChannels = 2          # The Number of Arduino Channels with EOG Signals Read in; My Beta-Test Used 4 Channels
    numTimePoints = 3000     # The Number of Data Points to Display to the User at a Time; My beta-Test Used 2000 Points
    
    # Protocol Switches: Only the First True Variable Excecutes
    streamArduinoData = False   # Stream in Data from the Arduino and Analyze; Input 'controlVR' = True to Move VR
    readDataFromExcel = True    # Analyze Data from Excel File called 'testDataExcelFile' on Sheet Number 'testSheetNum'
    calibrateModel = False      # Read in ALL Data Under 'neuralNetworkFolder', and Train the Data
    
    # User Options During the Run: Any Number Can be True
    saveCalibration = True   # Save the Calibration Model for Later Use
    plotStreamedData = False   # Graph the Data to Show Incoming Signals + Analysis
    saveInputData = False      # Saves the Data in 'readData.data' in an Excel Named 'saveExcelName'
    controlVR = False         # Apply the Algorithm to Control the Virtual Reality View
    
    # ---------------------------------------------------------------------- #
    
    # Take Data from the Arduino and Save it as an Excel (For Later Use)
    if saveInputData:
        saveExcelName = "Samuel Solomon 2021-10-08 BAD.xlsx"  # The Name of the Saved File
        saveDataFolder = "../Input Data/All Data/Industry Electrodes/"   # Data Folder to Save the Excel Data; MUST END IN '/'
    
    # Instead of Arduino Data, Use Test Data from Excel File
    if readDataFromExcel:
        testDataExcelFile = "../Input Data/All Data/Industry Electrodes/Samuel Solomon 2021-10-08.xlsx" # Path to the Test Data
        testSheetNum = 0   # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
    if controlVR:
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
    # Stream in Data from Arduino
    if streamArduinoData:
        arduinoRead = streamData.arduinoRead(eogSerialNum = eogSerialNum, emgSerialNum = None, eegSerialNum = None, handSerialNum = None)
        readData = streamData.eogArduinoRead(arduinoRead, numTimePoints, moveDataFinger, numChannels, samplingFreq, plotStreamedData, calibrateModel, guiApp = None)
        readData.streamEOGData(numDataPoints, calibrateModel = calibrateModel, actionControl = gazeControl)
    # Take Data from Excel Sheet
    elif readDataFromExcel:
        readData = excelData.readExcel(eogProtocol)
        readData.streamExcelData(testDataExcelFile, plotStreamedData, testSheetNum, predictionModel = None, actionControl = None)

    # ---------------------------------------------------------------------- #
    # -------------------------- Save Input data --------------------------- #
    # Save the Data in Excel: EOG Channels (Cols 1-4); X-Peaks (Cols 5-8); Peak Features (Cols 9-12)
    if saveInputData:
        # Double Check to See if User Wants to Save the Data
        verifiedSave = input("Are you Sure you Want to Save the Data (Y/N): ")
        if verifiedSave.upper() == "Y":
            # Initialize Class to Save the Data and Save
            saveInputs = excelData.saveExcel(numChannels, numFeatures = 0)
            saveInputs.saveData(eogProtocol.data, None, saveDataFolder, saveExcelName, "Trial 1 - EOG")
        else:
            print("User Chose Not to Save the Data")
    

    

