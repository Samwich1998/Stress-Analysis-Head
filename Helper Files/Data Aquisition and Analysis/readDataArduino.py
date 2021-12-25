#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:44:26 2021

@author: samuelsolomon
"""

# General Modules
import sys
import time
import threading
import matplotlib.pyplot as plt
# Stream Data from Arduino
import serial
import serial.tools.list_ports
import pyfirmata2
# Import Bioelectric Analysis Files
from eegAnalysis import eegProtocol
from emgAnalysis import emgProtocol
from eogAnalysis import eogProtocol
from ppgAnalysis import ppgProtocol


# --------------------------------------------------------------------------- #
# ----------------- Stream Data from Arduino Can Edit ----------------------- #

class arduinoRead():
    def __init__(self, eogSerialNum = None, ppgSerialNum = None, emgSerialNum = None, eegSerialNum = None, handSerialNum = None):
        # Save Arduino Serial Numbers
        self.eogSerialNum = eogSerialNum
        self.ppgSerialNum = ppgSerialNum
        self.emgSerialNum = emgSerialNum
        self.eegSerialNum = eegSerialNum
        self.handSerialNum = handSerialNum

        # Connect to the Arduinos
        self.eogArduino = self.initiateArduino(self.eogSerialNum)
        self.ppgArduino = self.initiateArduino(self.ppgSerialNum)
        self.emgArduino = self.initiateArduino(self.emgSerialNum)
        self.eegArduino = self.initiateArduino(self.eegSerialNum)
        self.handArduino = self.initiateArduino(self.handSerialNum)
        
        # Initialize Arduino Buffer
        self.arduinoBuffer = bytearray()
        
        self.printPortNums()
    
    def initiateArduinoFirmata(self):
        # Find and Connect to the Arduino Board
        PORT =  pyfirmata2.Arduino.AUTODETECT
        board = pyfirmata2.Arduino(PORT)
        # Set Sampling Rate
        board.samplingOn(1)
        
        # Initialize Analog Pins
        A0 = board.get_pin('a:0:i')
        A0.register_callback(myCallback = 1)  # Unsure for Callback
        A0.enable_reporting()
        # Save the Pins as a List  
        A0.read()
    
    def convertToTime(self, timeStamp):
        if type(timeStamp) == str:
            timeStamp = timeStamp.split(":")
        currentTime = int(timeStamp[0])*60*60 + int(timeStamp[1])*60 + int(timeStamp[2]) + int(timeStamp[3])/1E6
        return currentTime
    
    def convertToTimeStamp(self, timeSeconds):
        import math; import numpy as np
        hours = timeSeconds//3600
        remainingTime = timeSeconds%3600
        minutes = remainingTime//60
        remainingTime %=60
        seconds = math.floor(remainingTime)
        microSeconds = remainingTime - seconds
        microSeconds = np.round(microSeconds, 6)
        return hours, minutes, seconds, microSeconds

    def printPortNums(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(port.serial_number)

    def initiateArduino(self, arduinoSerialNum):
        arduinoControl = None
        if arduinoSerialNum:
            try:
                # Try to Connect to the Arduino
                arduinoPort = self.findArduino(serialNum = arduinoSerialNum)
                arduinoControl = serial.Serial(arduinoPort, baudrate=115200, timeout=1)

            except Exception as e:
                # If No Connection Established, Exit Program and Inform User
                print("Cannot Connect to Arudino", arduinoSerialNum);
                print("Error Message:", e)
                sys.exit()
        # Retun the Arduino actionControl
        return arduinoControl


    def findArduino(self, serialNum):
        """Get the name of the port that is connected to the Arduino."""
        port = None  # Initialize Blank Port
        # Get all Ports Connected to the Computer
        ports = serial.tools.list_ports.comports()
        # Loop Through Ports Until you Find the One you Want
        for p in ports:
            if p.serial_number == serialNum:
                port = p.device
        return port
    
    def resetArduino(self, arduino, numTrashReads):
        # Toss any data already received, see
        arduino.flushInput()
        arduino.flush()
        
        # Read and throw out first few reads
        for i in range(numTrashReads):
            self.readAll(arduino)
            arduino.read_until()
        arduino.flushInput()
        arduino.flush()
        arduino.read_until(); arduino.read_until()
        return arduino

    def handshakeArduino(self, arduino, sleep_time=1, print_handshake_message=False, handshake_code=0):
        """Make sure connection is established by sending
        and receiving bytes."""
        # Close and Reopen the Arduino
        arduino.close(); arduino.open()
        # Give the Arudino Some Time to Settle
        time.sleep(sleep_time)
        # Set a long timeout to complete handshake
        timeout = arduino.timeout
        arduino.timeout = 2

        # Read and discard everything that may be in the input buffer
        arduino.readAll()
        # Send request to Arduino
        arduino.write(bytes([handshake_code]))
        # Read in what Arduino sent
        handshake_message = arduino.read_until()
        # Send and receive request again
        arduino.write(bytes([handshake_code]))
        handshake_message = arduino.read_until()

        # Print the handshake message, if desired
        if print_handshake_message:
            print("Handshake message: " + handshake_message.decode())

        # Reset the timeout
        arduino.timeout = timeout


    def readAll(self, ser, readBuffer=b"", **args):
        """Read all available bytes from the serial port
        and append to the read buffer.

        Parameters
        ----------
        ser : serial.Serial() instance
            The device we are reading from.
        readBuffer : bytes, default b''
            Previous read buffer that is appended to.

        Returns
        -------
        output : bytes
            Bytes object that contains readBuffer + read.

        Notes
        -----
        .. `**args` appears, but is never used. This is for
           compatibility with `readAllNewlines()` as a
           drop-in replacement for this function.
        """
        # Set timeout to None to make sure we read all bytes
        previous_timeout = ser.timeout
        ser.timeout = None

        in_waiting = ser.in_waiting
        read = ser.read(size=in_waiting)

        # Reset to previous timeout
        ser.timeout = previous_timeout

        return readBuffer + read


    def readAllNewlines(self, ser, readBuffer=b"", n_reads=400):
        """Read data in until encountering newlines.

        Parameters
        ----------
        ser : serial.Serial() instance
            The device we are reading from.
        n_reads : int
            The number of reads up to newlines
        readBuffer : bytes, default b''
            Previous read buffer that is appended to.

        Returns
        -------
        output : bytes
            Bytes object that contains readBuffer + read.

        Notes
        -----
        .. This is a drop-in replacement for readAll().
        """
        raw = readBuffer
        for _ in range(n_reads):
            raw += ser.read_until()
        return raw
    
    def readline(self, ser):
        i = self.arduinoBuffer.find(b"\n")
        if i >= 0:
            r = self.arduinoBuffer[:i+1]
            self.arduinoBuffer = self.arduinoBuffer[i+1:]
            return r
        while True:
            i = max(1, min(2048, ser.in_waiting))
            data = ser.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.arduinoBuffer + data[:i+1]
                self.arduinoBuffer[0:] = data[i+1:]
                return r
            else:
                self.arduinoBuffer.extend(data)


    def parseRead(self, byteArrayList, numChannels):
        """Parse a read with time, volage data

        Parameters
        ----------
        read : byte string
            Byte string with comma delimited time/voltage
            measurements.

        Returns a List of:
        -------
        voltage : list of floats; Voltages in volts.
        time: x-axis data
        remaining_bytes : byte string remaining, unparsed bytes.
        """
        # Initiate Variables to Hold [[Voltages (Y) -> ...], Time (X), Buffer]
        arduinoData = [ [[] for channel in range(numChannels)], [] ]

        for byteArray in byteArrayList:
            byteObject = bytes(byteArray)
            rawRead = str(byteObject)[2:-5]
            try:
                # Seperate the Arduino Data
                arduinoValues = rawRead.split(",")

                if len(arduinoValues) == numChannels + 3:
                    
                    # Store the Current Time
                    segmentedTime = arduinoValues[0].split("-")
                    # If There is Only One Time, its a General Counter in MicroSeconds
                    if len(segmentedTime) == 1:
                        arduinoData[1].append(int(segmentedTime[0])/1E6)
                    # If Multiple Segments, We Have "Hour:Minute:Second:MicroSecond"
                    else:
                        currentTime = self.convertToTime(segmentedTime)
                        arduinoData[1].append(currentTime)
                    
                    # Add the Voltage Data
                    for channelIndex in range(numChannels):
                        # Convert Arduino Data to Voltage Before Storing
                        arduinoData[0][channelIndex].append(int(arduinoValues[channelIndex+1]) * 3.3/4096)
                else:
                    print("Bad Arduino Reading:", arduinoValues)
                    print("You May Want to Inrease 'moveDataFinger' to Not Fall Behind in Reading Points")
            except:
                print("Cannot Read Arduino Value:", rawRead)
                pass
        # Return the Values
        return arduinoData

class emgArduinoRead(emgProtocol):

    def __init__(self, arduinoRead, numTimePoints, moveDataFinger, numChannels, gestureClasses, plotStreamedData, guiApp = None):
        # Get Variables from Peak Analysis File
        super().__init__(numTimePoints, moveDataFinger, numChannels, gestureClasses, plotStreamedData)

        # Store the arduinoRead Instance
        self.arduinoRead = arduinoRead
        self.emgArduino = arduinoRead.emgArduino
        self.handArduino = arduinoRead.handArduino

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
        
        # Read and throw out first few reads
        self.eogArduino.read_until(b'')
        for i in range(numTrashReads):
            self.eogArduino.read_until()
        
        # Calculate the Stop Time
        timeBuffer = 0
        if type(stopTimeStreaming) in [float, int]:
            from datetime import datetime
            # Save Time Buffer
            timeBuffer = stopTimeStreaming
            # Get the Current Time as a TimeStamp
            currentTime = datetime.now().time()
            stopTimeStreaming = str(currentTime).replace(".",":")
        # Get the Final Time in Seconds (From 12:00am of the Current Day) to Stop Streaming
        stopTimeStreaming = self.arduinoRead.convertToTime(stopTimeStreaming) + timeBuffer
        
        # Set Up Hand Arduino if Needed
        if self.handArduino:
            self.handArduino.readAll() # Throw Out Initial Readings
            # Set Up Laser Reading
            threading.Thread(target = self.distanceRead, args = (actionControl, stopTimeStreaming), daemon=True).start()

        try:
            dataFinger = 0
            # Loop Through and Read the Arduino Data in Real-Time
            while len(self.data["timePoints"]) == 0 or self.data["timePoints"][-1] < stopTimeStreaming:


                # Read in chunk of data
                rawReadsList = []
                while (int(self.eogArduino.in_waiting) > 0):
                    rawReadsList.append(self.arduinoRead.readline(ser=self.eogArduino))
                # Parse it, passing if it is gibberish
                Voltages, timePoints = self.arduinoRead.parseRead(rawReadsList, self.numChannels)

                # Update data dictionary
                self.data["timePoints"].extend(timePoints)
                for channelIndex in range(self.numChannels):
                    self.data['Channel' + str(channelIndex+1)].extend(Voltages[channelIndex])

                # When Ready, Send Data Off for Analysis
                pointNum = len(self.data["timePoints"])
                while pointNum - dataFinger >= self.numTimePoints:
                    self.analyzeData(dataFinger, self.plotStreamedData, predictionModel, actionControl)
                    dataFinger += self.moveDataFinger
            # At the End, Analyze Any Data Left
            if dataFinger < len(self.analysisProtocol.data["timePoints"]):
                self.analysisProtocol.analyzeData(dataFinger, self.plotStreamedData, predictionModel, actionControl)

        finally:
            self.emgArduino.close()

        print("Finished Streaming in Data; Closing Arduino\n")
        # Close the Arduinos at the End
        self.emgArduino.close()
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
        while self.data["timePoints"][-1] < stopTimeStreaming and not self.killDistanceRead:
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


class eegArduinoRead(eegProtocol):

    def __init__(self, arduinoRead, numTimePoints, moveDataFinger, numChannels, plotStreamedData, guiApp = None):
        # Get Variables from Peak Analysis File
        super().__init__(numTimePoints, moveDataFinger, numChannels, plotStreamedData)

        # Store the arduinoRead Instance
        self.arduinoRead = arduinoRead
        self.eegArduino = arduinoRead.eegArduino


    def streamEEGData(self, stopTimeStreaming, predictionModel = None, actionControl=None, numTrashReads=1000, numPointsPerRead=100):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Streaming in EMG Data from the Arduino")

        # Read and throw out first few reads
        self.eogArduino.read_until(b'')
        for i in range(numTrashReads):
            self.eogArduino.read_until()
        
        # Calculate the Stop Time
        timeBuffer = 0
        if type(stopTimeStreaming) in [float, int]:
            from datetime import datetime
            # Save Time Buffer
            timeBuffer = stopTimeStreaming
            # Get the Current Time as a TimeStamp
            currentTime = datetime.now().time()
            stopTimeStreaming = str(currentTime).replace(".",":")
        # Get the Final Time in Seconds (From 12:00am of the Current Day) to Stop Streaming
        stopTimeStreaming = self.arduinoRead.convertToTime(stopTimeStreaming) + timeBuffer
        
        try:
            dataFinger = 0
            # Loop Through and Read the Arduino Data in Real-Time
            while len(self.data["timePoints"]) == 0 or self.data["timePoints"][-1] < stopTimeStreaming:

                # Read in chunk of data
                rawReadsList = []
                while (int(self.eogArduino.in_waiting) > 0):
                    rawReadsList.append(self.arduinoRead.readline(ser=self.eogArduino))
                # Parse it, passing if it is gibberish
                Voltages, timePoints = self.arduinoRead.parseRead(rawReadsList, self.numChannels)

                # Update data dictionary
                self.data["timePoints"].extend(timePoints)
                for channelIndex in range(self.numChannels):
                    self.data['Channel' + str(channelIndex+1)].extend(Voltages[channelIndex])

                # When Ready, Send Data Off for Analysis
                while len(self.data["timePoints"]) - dataFinger >= self.numTimePoints:
                    self.analyzeData(dataFinger, self.plotStreamedData, predictionModel, actionControl)
                    dataFinger += self.moveDataFinger
            # At the End, Analyze Any Data Left
            if dataFinger < len(self.data["timePoints"]):
                self.analyzeData(dataFinger, self.plotStreamedData, predictionModel, actionControl)

        finally:
            self.eegArduino.close()

        print("Finished Streaming in Data; Closing Arduino\n")
        # Close the Arduinos at the End
        self.eegArduino.close()


class eogArduinoRead(eogProtocol):

    def __init__(self, arduinoRead, numTimePoints, moveDataFinger, numChannels, plotStreamedData, guiApp = None):
        # Get Variables from Peak Analysis File
        super().__init__(numTimePoints, moveDataFinger, numChannels, plotStreamedData)

        # Store the arduinoRead Instance
        self.arduinoRead = arduinoRead
        self.eogArduino = arduinoRead.eogArduino
        self.eogSerialNum = arduinoRead.eogSerialNum

    def streamEOGData(self, stopTimeStreaming, predictionModel = None, actionControl = None, calibrateModel = False, numTrashReads=100, numPointsPerRead=300):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Streaming in EOG Data from the Arduino")
        
        # Read and throw out first few reads
        self.eogArduino.read_until(b'')
        for i in range(numTrashReads):
            self.arduinoRead.readline(ser=self.eogArduino)
        
        # Calculate the Stop Time
        timeBuffer = 0
        if type(stopTimeStreaming) in [float, int]:
            from datetime import datetime
            # Save Time Buffer
            timeBuffer = stopTimeStreaming
            # Get the Current Time as a TimeStamp
            currentTime = datetime.now().time()
            stopTimeStreaming = str(currentTime).replace(".",":")
        # Get the Final Time in Seconds (From 12:00am of the Current Day) to Stop Streaming
        stopTimeStreaming = self.arduinoRead.convertToTime(stopTimeStreaming) + timeBuffer

        try:
            # If Needed Calibrate the Model
            if calibrateModel:
                plt.close()
                self.askForCalibration(numTrashReads)

            dataFinger = 0
            # Loop Through and Read the Arduino Data in Real-Time
            while len(self.data["timePoints"]) == 0 or self.data["timePoints"][-1] < stopTimeStreaming:

                # Read in chunk of data
                rawReadsList = []
                while (int(self.eogArduino.in_waiting) > 0):
                    rawReadsList.append(self.arduinoRead.readline(ser=self.eogArduino))
                # Parse it, passing if it is gibberish
                Voltages, timePoints = self.arduinoRead.parseRead(rawReadsList, self.numChannels)

                # Update data dictionary
                self.data["timePoints"].extend(timePoints)
                for channelIndex in range(self.numChannels):
                    self.data['Channel' + str(channelIndex+1)].extend(Voltages[channelIndex])
                
                # When Ready, Send Data Off for Analysis
                pointNum = len(self.data["timePoints"])
                while pointNum - dataFinger >= self.numTimePoints:
                    self.analyzeData(dataFinger, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl, calibrateModel = calibrateModel)
                    dataFinger += self.moveDataFinger

                    # If You Need to Calibrate a Channel
                    if calibrateModel:
                        dataFinger = 0
                        calibrateModel = self.performCalibration(numTrashReads)
                        break
            # At the End, Analyze Any Data Left
            if dataFinger < len(self.data["timePoints"]):
                self.analyzeData(dataFinger, self.plotStreamedData, predictionModel = predictionModel, actionControl = actionControl, calibrateModel = calibrateModel)

        finally:
            self.eogArduino.close()

        print("Finished Streaming in Data; Closing Arduino\n")
        # Close the Arduinos at the End
        self.eogArduino.close()
    
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
            self.eogArduino = self.arduinoRead.resetArduino(self.eogArduino, numTrashReads)
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
        self.eogArduino = self.arduinoRead.resetArduino(self.eogArduino, numTrashReads)
        print("Orient Now")

class ppgArduinoRead(ppgProtocol):

    def __init__(self, arduinoRead, numTimePoints, moveDataFinger, numChannels, gestureClasses, plotStreamedData, guiApp = None):
        # Get Variables from Peak Analysis File
        super().__init__(numTimePoints, moveDataFinger, numChannels, plotStreamedData)

        # Store the arduinoRead Instance
        self.arduinoRead = arduinoRead
        self.ppgArduino = arduinoRead.ppgArduino

    def streamPPGData(self, stopTimeStreaming, predictionModel = None, actionControl=None, numTrashReads=1000, numPointsPerRead=400):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Streaming in EMG Data from the Arduino")

        # Read and throw out first few reads
        self.eogArduino.read_until(b'')
        for i in range(numTrashReads):
            self.eogArduino.read_until()
        
        # Calculate the Stop Time
        timeBuffer = 0
        if type(stopTimeStreaming) in [float, int]:
            from datetime import datetime
            # Save Time Buffer
            timeBuffer = stopTimeStreaming
            # Get the Current Time as a TimeStamp
            currentTime = datetime.now().time()
            stopTimeStreaming = str(currentTime).replace(".",":")
        # Get the Final Time in Seconds (From 12:00am of the Current Day) to Stop Streaming
        stopTimeStreaming = self.arduinoRead.convertToTime(stopTimeStreaming) + timeBuffer
            
        try:
            dataFinger = 0
            # Loop Through and Read the Arduino Data in Real-Time
            while len(self.data["timePoints"]) == 0 or self.data["timePoints"][-1] < stopTimeStreaming:


                # Read in chunk of data
                rawReadsList = []
                while (int(self.eogArduino.in_waiting) > 0):
                    rawReadsList.append(self.arduinoRead.readline(ser=self.eogArduino))
                # Parse it, passing if it is gibberish
                Voltages, timePoints = self.arduinoRead.parseRead(rawReadsList, self.numChannels)

                # Update data dictionary
                self.data["timePoints"].extend(timePoints)
                for channelIndex in range(self.numChannels):
                    self.data['Channel' + str(channelIndex+1)].extend(Voltages[channelIndex])

                # When Ready, Send Data Off for Analysis
                pointNum = len(self.data["timePoints"])
                while pointNum - dataFinger >= self.numTimePoints:
                    self.analyzeData(dataFinger, self.plotStreamedData, predictionModel, actionControl)
                    dataFinger += self.moveDataFinger
            # At the End, Analyze Any Data Left
            if dataFinger < len(self.analysisProtocol.data["timePoints"]):
                self.analysisProtocol.analyzeData(dataFinger, self.plotStreamedData, predictionModel, actionControl)
        finally:
            self.ppgArduino.close()

        print("Finished Streaming in Data; Closing Arduino\n")
        # Close the Arduinos at the End
        self.ppgArduino.close()
        if self.handArduino:
            self.handArduino.write(str.encode('s0')) # turn off the laser
            self.handArduino.close()
        if self.guiApp:
            self.guiApp.handArduino = None
            self.guiApp.resetButton()


