#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:44:26 2021

@author: samuelsolomon
"""

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General Modules
import sys
import time
# Stream Data from Arduino
import serial
import pyfirmata2
import serial.tools.list_ports

# --------------------------------------------------------------------------- #
# ----------------- Stream Data from Arduino Can Edit ----------------------- #

class arduinoRead():
    def __init__(self, mainSerialNum = None, handSerialNum = None):
        # Save Arduino Serial Numbers
        self.mainSerialNum = mainSerialNum
        self.handSerialNum = handSerialNum

        # Connect to the Arduinos
        self.mainArduino = self.initiateArduino(self.mainSerialNum)
        self.handArduino = self.initiateArduino(self.handSerialNum)
        
        # Initialize Arduino Buffer
        self.arduinoBuffer = bytearray()
        
        self.currentTime = 0;
        
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
                arduinoControl.close(); arduinoControl.open();

            except Exception as e:
                # If No Connection Established, Exit Program and Inform User
                print("Cannot Connect to Arudino", arduinoSerialNum);
                print("Error Message:", e)
                print("Availible Ports are:\n")
                self.printPortNums()
                sys.exit()
        # Retun the Arduino actionControl
        return arduinoControl

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
        # Close and Reopen the Arduino
        arduino.close(); arduino.open()
        # Give the Arudino Some Time to Settle
        time.sleep(2)
        
        # Toss any data already received, see
        arduino.flushInput()
        arduino.flush()
        # Read and discard everything that may be in the input buffer
        arduino.readAll()
        
        # Read and throw out first few reads
        for i in range(numTrashReads):
            self.readAll(arduino)
            arduino.read_until()
        arduino.flushInput()
        arduino.flush()
        arduino.read_until(); arduino.read_until()
        return arduino

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


    def parseRead(self, byteArrayList, numChannels, maxVolt = 3.3, adcResolution = 4096, verbose = True):
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
            rawRead = str(byteObject)[2:-3]
            try:
                # Seperate the Arduino Data
                arduinoValues = rawRead.split(",")

                if len(arduinoValues) == numChannels + 1:
                    
                    # Store the Current Time
                    arduinoData[1].append(float(arduinoValues[0]))
                    
                    # Add the Voltage Data
                    for channelIndex in range(numChannels):
                        # Convert Arduino Data to Voltage Before Storing
                        arduinoData[0][channelIndex].append(int(arduinoValues[channelIndex+1]) * maxVolt/adcResolution)
                elif verbose:
                    print("Bad Arduino Reading:", arduinoValues, len(arduinoValues), numChannels + 1)
                    print("You May Want to Inrease 'moveDataFinger' to Not Fall Behind in Reading Points")
                    print("Alternatively, you are reading more/less channels than expected")
            except:
                if verbose:
                    print("Cannot Read Arduino Value:", rawRead)
                pass
        # Return the Values
        return arduinoData
    
    
    def decompressByte(self, compressedByte):
        # Split into Two Bytes
        leftInt = 0x0F >> 4 | compressedByte >> 4;
        rightInt = 0x0F >> 4 | (0x0F & compressedByte);
        # Return the Final Characters
        return str(leftInt) + str(rightInt);
    
    def parseCompressedRead(self, byteArrayList, numChannels, maxVolt = 3.3, adcResolution = 4096, verbose = True):
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
        assert numChannels == 3

        for byteArray in byteArrayList:
            rawReadCompressed = bytes(byteArray)[0:-2]

            if len(rawReadCompressed) == 11:
                
                    # Store the Current Time
                    self.currentTime += float( 
                        self.decompressByte(rawReadCompressed[0]) + "." + \
                        self.decompressByte(rawReadCompressed[1]) + self.decompressByte(rawReadCompressed[2]) + self.decompressByte(rawReadCompressed[3])
                    )
                    arduinoData[1].append(self.currentTime)
                    
                    # Add the Voltage Data
                    arduinoData[0][0].append(int( 
                        self.decompressByte(rawReadCompressed[4]) + self.decompressByte(rawReadCompressed[5])
                    )*maxVolt/adcResolution)
                    
                    arduinoData[0][1].append(int( 
                        self.decompressByte(rawReadCompressed[6]) + self.decompressByte(rawReadCompressed[7]) + self.decompressByte(rawReadCompressed[8])
                    )*maxVolt/adcResolution)
                                        
                    arduinoData[0][2].append(int( 
                        self.decompressByte(rawReadCompressed[9]) + self.decompressByte(rawReadCompressed[10])
                    )*maxVolt/adcResolution)                    
                    
            elif verbose:
                print("Cannot Read Arduino Value:", byteArray)
        # Return the Values
        return arduinoData

