#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:44:26 2021

@author: samuelsolomon
"""

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General Modules
import os
import sys
import time
import math
import threading
import numpy as np
import matplotlib.pyplot as plt
# Modules for Time
from datetime import datetime
# Modules for reading excel
import openpyxl as xl

sys.path.append('./Data Aquisition and Analysis/Biolectric Protocols/')  # Folder with Data Aquisition Files
# Import Bioelectric Analysis Files
from emgAnalysis import emgProtocol
from eogAnalysis import eogProtocol
from ppgAnalysis import ppgProtocol

# Import Modules to Read in Data
import arduinoInterface as arduinoInterface      # Functions to Read in Data from Arduino


# -------------------------------------------------------------------------- #
# ---------------------------- Global Function ----------------------------- #

class streamingFunctions():
    
    def __init__(self, mainSerialNum, handSerialNum, numTimePoints, moveDataFinger, plotStreamedData, streamingMap):
        # Store the arduinoRead Instance
        if mainSerialNum != None:
            self.arduinoRead = arduinoInterface.arduinoRead(mainSerialNum = mainSerialNum, handSerialNum = handSerialNum)
            self.mainArduino = self.arduinoRead.mainArduino
        
        # Store General Streaming Parameters
        self.numTimePoints = numTimePoints
        self.moveDataFinger = moveDataFinger
        self.plotStreamedData = plotStreamedData
        self.streamingMap = np.array(streamingMap)
        # Segment the Channels by Their Sensor
        self.numChannels = len(streamingMap)
        self.numChannels_EOG = streamingMap.count('eog')
        self.numChannels_PPG = streamingMap.count('ppg')
        
        # Associate Each Channel with a Sensor
        self.eogIndices = np.where(self.streamingMap == 'eog')[0]
        self.ppgIndices = np.where(self.streamingMap == 'ppg')[0]
        # Check That We Segmented the Channels Correctly
        assert self.numChannels_EOG == len(self.eogIndices)
        assert self.numChannels_PPG == len(self.ppgIndices)
        assert self.numChannels_EOG + self.numChannels_PPG + 1 == self.numChannels
                
        # Create Pointer to the Analysis Classes
        self.eogAnalysis = eogProtocol(self.numTimePoints, self.moveDataFinger, self.numChannels_EOG, self.plotStreamedData)
        self.ppgAnalysis = ppgProtocol(self.numTimePoints, self.moveDataFinger, self.numChannels_PPG, self.plotStreamedData)
            
    def setupArduinoStream(self, stopTimeStreaming, usingTimestamps = False):
        # Read and throw out first few reads
        rawReadsList = []
        while (int(self.mainArduino.in_waiting) > 0 or len(rawReadsList) < 2000):
            rawReadsList.append(self.arduinoRead.readline(ser=self.mainArduino))
        
        if usingTimestamps:
            # Calculate the Stop Time
            timeBuffer = 0
            if type(stopTimeStreaming) in [float, int]:
                # Save Time Buffer
                timeBuffer = stopTimeStreaming
                # Get the Current Time as a TimeStamp
                currentTime = datetime.now().time()
                stopTimeStreaming = str(currentTime).replace(".",":")
            # Get the Final Time in Seconds (From 12:00am of the Current Day) to Stop Streaming
            stopTimeStreaming = self.convertToTime(stopTimeStreaming) + timeBuffer
        
        return stopTimeStreaming
    
    def recordData(self, maxVolt = 3.3, adcResolution = 4096):
        # Read in at least one point
        rawReadsList = []
        while (int(self.mainArduino.in_waiting) > 0 or len(rawReadsList) == 0):
            rawReadsList.append(self.arduinoRead.readline(ser=self.mainArduino))
            
        # Parse the Data
        Voltages, timePoints = self.arduinoRead.parseCompressedRead(rawReadsList,  self.numChannels,  maxVolt, adcResolution)
        # Organize the Data for Processing
        self.organizeData(timePoints, Voltages)
        
    def organizeData(self, timePoints, Voltages):
        # Update the EOG Data if Present
        if self.numChannels_EOG != 0:
            self.eogAnalysis.data[0].extend(timePoints)            
            for channelIndex in range(len(self.eogIndices)):
                # Compile the Voltage Data
                streamingDataIndex = self.eogIndices[channelIndex]
                newVoltageData = Voltages[streamingDataIndex]
                # Add the Data to the Correct Channel
                self.eogAnalysis.data[1][channelIndex].extend(newVoltageData)
        
        # Update the PPG Data if Present
        if self.numChannels_PPG != 0:
            # Compile the Voltages and Remove Zeros
            for dataInd in range(len(timePoints)):
                # Remove Zeros: Data That is Skipped
                if Voltages[self.ppgIndices[0]][dataInd] == 0:
                    continue
                # If the Data is Good, Add the Time
                self.ppgAnalysis.data[0].append(timePoints[dataInd])
                # Compile the Voltage Points in Each Channel
                for channelIndex in range(len(self.ppgIndices)):
                    # Compile the Voltage Data
                    streamingDataIndex = self.ppgIndices[channelIndex]
                    newVoltageData = Voltages[streamingDataIndex][dataInd]
                    # Add the Data to the Correct Channel
                    self.ppgAnalysis.data[1][channelIndex].append(newVoltageData)
            
    def convertToTime(self, timeStamp):
        if type(timeStamp) == str:
            timeStamp = timeStamp.split(":")
        timeStamp.reverse()
        
        currentTime = 0
        orderOfInput = [1E-6, 1, 60, 60*60, 60*60*24]
        for i, timeStampVal in enumerate(timeStamp):
            currentTime += orderOfInput[i]*int(timeStampVal)
        return currentTime
    
    def convertToTimeStamp(self, timeSeconds):
        hours = timeSeconds//3600
        remainingTime = timeSeconds%3600
        minutes = remainingTime//60
        remainingTime %=60
        seconds = math.floor(remainingTime)
        microSeconds = remainingTime - seconds
        microSeconds = np.round(microSeconds, 6)
        return hours, minutes, seconds, microSeconds

# -------------------------------------------------------------------------- #
# ---------------------------- Reading All Data ---------------------------- #

class mainArduinoRead(streamingFunctions):

    def __init__(self, mainSerialNum, numTimePoints, moveDataFinger, streamingMap, plotStreamedData, guiApp = None):
        # Create Pointer to Common Functions
        super().__init__(mainSerialNum, None, numTimePoints, moveDataFinger, plotStreamedData, streamingMap)

    def streamArduinoData(self, stopTimeStreaming, predictionModel = None, actionControl = None, calibrateModel = False, numTrashReads=100, numPointsPerRead=300):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Streaming in Data from the Arduino")
        # Prepare the arduino to stream in data
        stopTimeStreaming = self.setupArduinoStream(stopTimeStreaming)
            
        try:
            dataFinger_EOG = 0; dataFinger_PPG = 0
            # Loop Through and Read the Arduino Data in Real-Time
            while len(self.eogAnalysis.data[0]) == 0 or self.eogAnalysis.data[0][-1] < stopTimeStreaming:
                # Stream in the Latest Data
                self.recordData()

                # When Ready, Send Data Off for Analysis
                while len(self.eogAnalysis.data[0]) - dataFinger_EOG >= self.numTimePoints:
                    # Analyze Data
                    self.eogAnalysis.analyzeData(dataFinger_EOG, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)
                    # Move DataFinger to Analyze Next Section
                    dataFinger_EOG += self.moveDataFinger
                
                while len(self.ppgAnalysis.data[0]) - dataFinger_PPG >= self.numTimePoints:
                    # Analyze Data
                    #self.ppgAnalysis.analyzeData(dataFinger_PPG, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)
                    # Move DataFinger to Analyze Next Section
                    dataFinger_PPG += self.moveDataFinger

            # At the End, Analyze Any Data Left
            self.eogAnalysis.analyzeData(dataFinger_EOG, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)
            #self.ppgAnalysis.analyzeData(dataFinger_PPG, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)

        finally:
             # Close the Arduinos at the End
            print("Finished Streaming in Data; Closing Arduino\n")
            self.mainArduino.close();
        
    def streamExcelData(self, compiledData, predictionModel = None, actionControl = None, calibrateModel = False):
        print("Analyzing Data from Excel")
        # Reset Global Variable in Case it Was Previously Populated
        self.eogAnalysis.resetGlobalVariables()
        self.ppgAnalysis.resetGlobalVariables()
        # Extract the Time and Voltage Data
        timePoints, Voltages = compiledData
        Voltages = np.array(Voltages)
            
        generalDataFinger = 0;
        dataFinger_EOG = 0; dataFinger_PPG = 0
        # Loop Through and Read the Excel Data in Pseudo-Real-Time
        while generalDataFinger < len(timePoints): 
            # Organize the Input Data
            self.organizeData(timePoints[generalDataFinger:generalDataFinger+self.moveDataFinger], Voltages[:,generalDataFinger:generalDataFinger+self.moveDataFinger])

            # When Ready, Send Data Off for Analysis
            while len(self.eogAnalysis.data[0]) - dataFinger_EOG >= self.numTimePoints:
                # Analyze Data
                self.eogAnalysis.analyzeData(dataFinger_EOG, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)
                # Move DataFinger to Analyze Next Section
                dataFinger_EOG += self.moveDataFinger
            
            while len(self.ppgAnalysis.data[0]) - dataFinger_PPG >= self.numTimePoints:
                # Analyze Data
                #self.ppgAnalysis.analyzeData(dataFinger_PPG, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)
                # Move DataFinger to Analyze Next Section
                dataFinger_PPG += self.moveDataFinger
            
            # Move onto the Next Batch
            generalDataFinger += self.moveDataFinger
            
        # At the End, Analyze Any Data Left
        self.eogAnalysis.analyzeData(dataFinger_EOG, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)
        #self.ppgAnalysis.analyzeData(dataFinger_PPG, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)

        # Finished Analyzing the Data
        print("Finished Analyzing Excel Data\n")
            
# -------------------------------------------------------------------------- #
# ---------------------------- Reading EOG Data ---------------------------- #

class eogArduinoRead(eogProtocol):

    def __init__(self, mainSerialNum, numTimePoints, moveDataFinger, numChannels, plotStreamedData, guiApp = None):
        # Get Variables from Peak Analysis File
        super().__init__(numTimePoints, moveDataFinger, numChannels, plotStreamedData)

        # Create Pointer to Common Functions
        self.commonFunctions = streamingFunctions(mainSerialNum = mainSerialNum, handSerialNum = None, numChannels = numChannels)

    def streamEOGData(self, stopTimeStreaming, predictionModel = None, actionControl = None, calibrateModel = False, numTrashReads=100, numPointsPerRead=300):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Streaming in EOG Data from the Arduino")
        # Prepare the arduino to stream in data
        stopTimeStreaming = self.commonFunctions.setupArduinoStream(stopTimeStreaming)

        try:
            # If Needed Calibrate the Model
            if calibrateModel:
                plt.close()
                self.askForCalibration(numTrashReads)

            dataFinger = 0
            # Loop Through and Read the Arduino Data in Real-Time
            while len(self.data[0]) == 0 or self.data[0][-1] < stopTimeStreaming:
                # Stream in the Latest Data
                self.commonFunctions.recordData(self.data, self.numChannels)
                
                # When Ready, Send Data Off for Analysis
                while len(self.data[0]) - dataFinger >= self.numTimePoints:
                    # Analyze EOG Data
                    self.analyzeData(dataFinger, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl, calibrateModel = calibrateModel)
                    # Move DataFinger to Analyze Next Section
                    dataFinger += self.moveDataFinger

                    # If You Need to Calibrate a Channel
                    if calibrateModel:
                        dataFinger = 0
                        calibrateModel = self.performCalibration(numTrashReads)
                        break
            # At the End, Analyze Any Data Left
            self.analyzeData(dataFinger, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl, calibrateModel = calibrateModel)

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
        
# -------------------------------------------------------------------------- #
# ---------------------------- Reading PPG Data ---------------------------- #

class ppgArduinoRead(streamingFunctions):

    def __init__(self, mainSerialNum, numTimePoints, moveDataFinger, streamingMap, plotStreamedData, guiApp = None):
        # Create Pointer to Common Functions
        super().__init__(mainSerialNum, None, numTimePoints, moveDataFinger, plotStreamedData, streamingMap)
        
    def streamPPGData(self, stopTimeStreaming, predictionModel = None, actionControl = None, calibrateModel = False, numTrashReads=100, numPointsPerRead=300):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Streaming in PPG Data from the Arduino")
        # Prepare the arduino to stream in data
        stopTimeStreaming = self.setupArduinoStream(stopTimeStreaming)
            
        try:
            dataFinger_PPG = 0
            # Loop Through and Read the Arduino Data in Real-Time
            while len(self.ppgAnalysis.data[0]) == 0 or self.ppgAnalysis.data[0][-1] < stopTimeStreaming:
                # Stream in the Latest Data
                self.recordData()
                
                while len(self.ppgAnalysis.data[0]) - dataFinger_PPG >= self.numTimePoints:
                    # Analyze Data
                    self.ppgAnalysis.analyzeData(dataFinger_PPG, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)
                    # Move DataFinger to Analyze Next Section
                    dataFinger_PPG += self.moveDataFinger

            # At the End, Analyze Any Data Left
            self.ppgAnalysis.analyzeData(dataFinger_PPG, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)

        finally:
             # Close the Arduinos at the End
            print("Finished Streaming in Data; Closing Arduino\n")
            self.mainArduino.close();
            
# -------------------------------------------------------------------------- #
# ---------------------------- Reading EMG Data ---------------------------- #

class emgArduinoRead(emgProtocol, streamingFunctions):

    def __init__(self, arduinoRead, numTimePoints, moveDataFinger, numChannels, gestureClasses, plotStreamedData, guiApp = None):
        # Get Variables from Peak Analysis File
        super().__init__(numTimePoints, moveDataFinger, numChannels, gestureClasses, plotStreamedData)

        # Store the arduinoRead Instance
        self.arduinoRead = arduinoRead
        self.mainArduino = arduinoRead.mainArduino
        self.handArduino = arduinoRead.handArduino
        # Create Pointer to Common Functions
        self.commonFunctions = streamingFunctions(self.mainArduino, self.arduinoRead)

        # Variables for Hand Arduino's DistanceRead Funcion
        self.speed_x = 1 # speed_x = 1 when the arm is in fast mode, otherwise, speed_x = 0
        self.stop_x = 0  # stop_x = 1 when robot hand is stopped by the pressure sensor
        self.distance_slow = 120 # robot arm will slow down if the distance is less than this number,
        self.speed_slow = 0.05 # speed of arm in slow mode
        self.speed_fast = 0.15 # speed of arm in fast mode
        self.STOP = 9999 # when hand touch something, int(9999) will be sent to computer
        self.MOVE = 8888 # when hand does not touch anything, int(8888) will be sent to computer
        self.killDistanceRead = False

        # Initiate the GUI: a Copy of the UI Window
        self.guiApp = guiApp
        if self.guiApp:
            self.guiApp.handArduino = self.handArduino
            self.guiApp.initiateRoboticMovement()

    def streamEMGData(self, stopTimeStreaming, predictionModel = None, actionControl=None, numTrashReads=1000, numPointsPerRead=100):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Streaming in EMG Data from the Arduino")
        # Prepare the arduino to stream in data
        stopTimeStreaming = self.commonFunctions.setupArduinoStream(stopTimeStreaming)
        
        # Set Up Hand Arduino if Needed
        if self.handArduino:
            self.handArduino.readAll() # Throw Out Initial Readings
            # Set Up Laser Reading
            threading.Thread(target = self.distanceRead, args = (actionControl, stopTimeStreaming), daemon=True).start()

        try:
            dataFinger = 0
            # Loop Through and Read the Arduino Data in Real-Time
            while len(self.data[0]) == 0 or self.data[0][-1] < stopTimeStreaming:
                # Stream in the Latest Data
                self.commonFunctions.recordData(self.data, self.numChannels)
                
                # When Ready, Send Data Off for Analysis
                while len(self.data[0]) - dataFinger >= self.numTimePoints:
                    # Analyze EOG Data
                    self.analyzeData(dataFinger, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)
                    # Move DataFinger to Analyze Next Section
                    dataFinger += self.moveDataFinger

            # At the End, Analyze Any Data Left
            if dataFinger < len(self.data[0]):
                self.analyzeData(dataFinger, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl)

        finally:
            self.mainArduino.close()

        print("Finished Streaming in Data; Closing Arduino\n")
        # Close the Arduinos at the End
        self.mainArduino.close()
        if self.handArduino:
            self.handArduino.write(str.encode('s0')) # turn off the laser
            self.handArduino.close()
        if self.guiApp:
            self.guiApp.handArduino = None
            self.guiApp.resetButton()


    def distanceRead(self, RoboArm, stopTimeStreaming):
        print("In Distance Read")
        for _ in range(5):
            self.handArduino.read_until()
        l_time = 0
        while self.data[0][-1] < stopTimeStreaming and not self.killDistanceRead:
            if self.handArduino.in_waiting > 0:
                d_laser = self.handArduino.read_until()
                distance = d_laser.decode()

                # Update Gui App Text
                if self.guiApp:
                    self.guiApp.Number_distance.setText(self.guiApp.translate("MainWindow", str(distance)))
                distance = int(distance)
                l_time = l_time + 100
                if distance < self.distance_slow and self.speed_x == 1:
                    self.handArduino.read_until()
                    RoboArm.updateMovementParams([self.speed_slow]*5, 'speed')
                    self.handArduino.read_until()
                    self.speed_x = 0
                    print('slow')
                elif distance >= self.distance_slow and self.speed_x == 0 and distance <= 2000:
                    self.handArduino.read_until()
                    RoboArm.updateMovementParams([self.speed_fast]*5, 'speed')
                    self.handArduino.read_until()
                    self.speed_x = 1
                    print('fast')
                elif distance == self.STOP and self.stop_x == 0:
                    print('stop!!')
                    RoboArm.stopRobot()
                    self.stop_x = 1
                elif distance == self.MOVE:
                    self.stop_x =0
            time.sleep(0.05)

