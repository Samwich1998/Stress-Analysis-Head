import sys
import time
import serial
import pyfirmata2
import serial.tools.list_ports


class arduinoRead:
    
    def __init__(self, mainSerialNum=None, therapySerialNum=None):
        # Save Arduino Serial Numbers
        self.mainSerialNum = mainSerialNum
        self.therapySerialNum = therapySerialNum

        # Connect to the Arduino's
        self.mainArduino = self.initiateArduino(self.mainSerialNum)
        self.therapyArduino = self.initiateArduino(self.therapySerialNum)

        # Initialize Arduino Buffer
        self.arduinoBuffer = bytearray()

        self.currentTime = 0

    @staticmethod
    def printPortNums():
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(port.serial_number)

    def initiateArduino(self, arduinoSerialNum):
        arduinoControl = None
        if arduinoSerialNum:
            try:
                # Try to Connect to the Arduino
                arduinoPort = self.findArduino(serialNum=arduinoSerialNum)
                arduinoControl = serial.Serial(arduinoPort, baudrate=115200, timeout=1)
                arduinoControl.close()
                arduinoControl.open()

            except Exception as e:
                # If No Connection Established, Exit Program and Inform User
                print("Cannot Connect to Arduino", arduinoSerialNum)
                print("Error Message:", e)
                print("Available Ports are:\n")
                self.printPortNums()
                sys.exit()
                
        # Return the Arduino actionControl
        return arduinoControl

    @staticmethod
    def initiateArduinoFirmata():
        # Find and Connect to the Arduino Board
        PORT = pyfirmata2.Arduino.AUTODETECT
        board = pyfirmata2.Arduino(PORT)
        # Set Sampling Rate
        board.samplingOn(1)

        # Initialize Analog Pins
        A0 = board.get_pin('a:0:i')
        A0.register_callback(myCallback=1)  # Unsure for Callback
        A0.enable_reporting()
        # Save the Pins as a List  
        A0.read()

    @staticmethod
    def findArduino(serialNum):
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
        arduino.close()
        arduino.open()
        # Give the Arduino Some Time to Settle
        time.sleep(2)

        # Toss any data already received, see
        arduino.flushInput()
        arduino.flush()
        # Read and discard everything that may be in the input buffer
        # arduino.readAll()

        # Read and throw out the first few reads
        for i in range(numTrashReads):
            self.readAll(arduino)
            arduino.read_until()
        arduino.flushInput()
        arduino.flush()
        arduino.read_until()
        arduino.read_until()
        return arduino

    @staticmethod
    def readAll(ser, readBuffer=b"", **args):
        previous_timeout = ser.timeout
        ser.timeout = None

        in_waiting = ser.in_waiting
        read = ser.read(size=in_waiting)

        # Reset to previous timeout
        ser.timeout = previous_timeout

        return readBuffer + read

    @staticmethod
    def readAllNewlines(ser, readBuffer=b"", n_reads=400):
        raw = readBuffer
        for _ in range(n_reads):
            raw += ser.read_until()
        return raw

    def readline(self, ser):
        i = self.arduinoBuffer.find(b"\n")
        if i >= 0:
            r = self.arduinoBuffer[:i + 1]
            self.arduinoBuffer = self.arduinoBuffer[i + 1:]
            return r
        while True:
            i = max(1, min(2048, ser.in_waiting))
            data = ser.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.arduinoBuffer + data[:i + 1]
                self.arduinoBuffer[0:] = data[i + 1:]
                return r
            else:
                self.arduinoBuffer.extend(data)

    def sendToArduinoSerial(self, sendingValue):
        self.therapyArduino.write(bytes(sendingValue, 'utf-8'))

    @staticmethod
    def parseRead(byteArrayList, numChannels, maxVolt=3.3, adcResolution=4096, verbose=True):
        arduinoData = [[[] for channel in range(numChannels)], [[]]]

        for byteArray in byteArrayList:
            byteObject = bytes(byteArray)
            rawRead = str(byteObject)[2:-3]
            try:
                # Separate the Arduino Data
                arduinoValues = rawRead.split(",")

                if len(arduinoValues) == numChannels + 1:

                    # Store the Current Time
                    arduinoData[1][0].append(float(arduinoValues[0]))

                    # Add the Voltage Data
                    for channelIndex in range(numChannels):
                        # Convert Arduino Data to Voltage Before Storing
                        arduinoData[0][channelIndex].append(int(arduinoValues[channelIndex + 1]) * maxVolt / adcResolution)
                elif verbose:
                    print("Bad Arduino Reading:", arduinoValues, len(arduinoValues), numChannels + 1)
                    print("You May Want to Increase 'moveDataFinger' to Not Fall Behind in Reading Points")
                    print("Alternatively, you are reading more/less channels than expected")
            except Exception as e:
                if verbose:
                    print("Cannot Read Arduino Value:", rawRead)
                    print("\tError Report:", e)
                pass
        # Return the Values
        return arduinoData

    @staticmethod
    def decompressByte(compressedByte):
        # Split into Two Bytes
        leftInt = 0x0F >> 4 | compressedByte >> 4
        rightInt = 0x0F >> 4 | (0x0F & compressedByte)
        # Return the Final Characters
        return str(leftInt) + str(rightInt)

    def parseCompressedRead(self, byteArrayList, numChannels, maxVolt=3.3, adcResolution=4095, verbose=True):
        # Initialize the Arduino Data
        Voltages = [[] for _ in range(numChannels)]
        assert numChannels == 4
        timePoints = []

        for byteArray in byteArrayList:
            rawReadCompressed = bytes(byteArray)[0:-2]

            if len(rawReadCompressed) == 16:
                # Store the Current Time
                self.currentTime += float(
                    self.decompressByte(rawReadCompressed[0]) + "." + \
                    self.decompressByte(rawReadCompressed[1]) + self.decompressByte(rawReadCompressed[2]) + self.decompressByte(rawReadCompressed[3])
                )
                timePoints.append(self.currentTime)

                # Add the Voltage Data
                Voltages[0].append(int(
                    self.decompressByte(rawReadCompressed[4]) + self.decompressByte(rawReadCompressed[5])
                ) * maxVolt / adcResolution)

                Voltages[1].append(int(
                    self.decompressByte(rawReadCompressed[6]) + self.decompressByte(rawReadCompressed[7])
                ) * maxVolt / adcResolution)

                Voltages[2].append(1 / int(
                    self.decompressByte(rawReadCompressed[8]) + self.decompressByte(rawReadCompressed[9]) + self.decompressByte(rawReadCompressed[10]) + self.decompressByte(rawReadCompressed[11])
                ))

                Voltages[3].append(int(
                    self.decompressByte(rawReadCompressed[12]) + self.decompressByte(rawReadCompressed[13]) + self.decompressByte(rawReadCompressed[14])
                ) * maxVolt / adcResolution)

            elif verbose:
                print("Cannot Read Arduino Value (Bad Length, " + str(len(rawReadCompressed)) + "):", byteArray)
                a = ""
                for byteVal in rawReadCompressed:
                    a += self.decompressByte(byteVal)
                print(a)

        return timePoints, Voltages
