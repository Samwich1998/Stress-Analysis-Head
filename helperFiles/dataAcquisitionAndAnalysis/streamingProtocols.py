# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General Modules
import time
import math
import collections
import numpy as np
# Modules for Time
from datetime import datetime

# Import Bioelectric Analysis Files
from .biolectricProtocols.eogAnalysis import eogProtocol
from .biolectricProtocols.eegAnalysis import eegProtocol
from .biolectricProtocols.ecgAnalysis import ecgProtocol
from .biolectricProtocols.edaAnalysis import edaProtocol
from .biolectricProtocols.emgAnalysis import emgProtocol
from .biolectricProtocols.temperatureAnalysis import tempProtocol
from .biolectricProtocols.generalAnalysis_lowFreq import generalProtocol_lowFreq
from .biolectricProtocols.generalAnalysis_highFreq import generalProtocol_highFreq

# Import Modules to Read in Data
from humanMachineInterface.arduinoInterface import arduinoRead  # Functions to Read in Data from Arduino
from humanMachineInterface.featureOrganization import featureOrganization

# Import plotting protocols
from dataVisualization.biolectricPlottingProtocols import plottingProtocols


# -------------------------------------------------------------------------- #
# ---------------------------- Global Function ----------------------------- #

class streamingHead(featureOrganization):

    def __init__(self, mainSerialNum, therapySerialNum, modelClasses, actionControl, numPointsPerBatch, moveDataFinger, streamingOrder, biomarkerOrder, featureAverageWindows, plotStreamedData):
        """
        ----------
        mainSerialNum : String or None. The serial number of the board WITH the recording sensors being read through pyserial. If None, no board is present.
        therapySerialNum : String or None. The serial number of a potential second board being read through pyserial. If None, no board is present.
        modelClasses : List of classes. A list of all model classes. The top of the class is _globalModel.py
        actionControl : TYPE
        numPointsPerBatch : TYPE
        moveDataFinger : TYPE
        streamingOrder : List of strings. A list of strings representing the order the sensors are being read in.
        biomarkerOrder : List of strings. A list of strings representing the order of feature extraction for the sensor data.
        featureAverageWindows : List of floats/ints. A list of numbers representing the sliding window for averaging features per each biomarker in biomarkerOrder.
        plotStreamedData : Boolean. A flag for whether we are plotting the data in real-time. See self.plottingClass for more details.
        """
        # Store the arduinoRead Instance
        if mainSerialNum is not None:
            self.arduinoRead = arduinoRead(mainSerialNum=mainSerialNum, therapySerialNum=therapySerialNum)
            self.mainArduino = self.arduinoRead.mainArduino

        # Variables that specify biomarker order.
        streamingOrder = np.char.lower(streamingOrder)  # The names of each recorded signal in order. Ex: ['eog', 'eog', 'eeg', 'eda']
        self.analysisOrder = list(collections.OrderedDict.fromkeys(streamingOrder))  # The set of biomarkers, maintaining the order they will be analyzed. Ex: ['eog', 'eeg', 'eda']
        # General streaming parameters.
        self.moveDataFinger = moveDataFinger  # The minimum number of NEW points to analyze in each batch.
        self.plotStreamedData = plotStreamedData  # Boolean: whether to graph the incoming signals + analysis.
        self.numPointsPerBatch = numPointsPerBatch  # The number of points to analyze in each batch.
        self.numChannels = len(streamingOrder)  # The total number of signals being recorded.
        # Specify the channel indices for each biomarker.
        counter = collections.Counter()
        self.streamingChannelIndices = np.array([counter.update({item: 1}) or counter[item] - 1 for item in streamingOrder])  # The channel index of each biomarker.
        biomarkerChannelIndices = self.streamingChannelIndices[np.isin(streamingOrder, biomarkerOrder)]
        # NOTE: WE ARE TAKING FEATURES FROM ALL CHANNELS OF A BIOMARKER IN THIS CASE.
        assert len(biomarkerChannelIndices) == len(biomarkerOrder), f"You cannot extract only one channel's features yet: {biomarkerOrder}"

        # Variables that rely on the sensor data's order.
        self.numChannelDist = [0 for _ in range(len(self.analysisOrder))]  # Track the number of channels used by each sensor
        self.channelDist = [[] for _ in range(len(self.analysisOrder))]  # Track the order (its index) each sensor comes in
        # Populate the variables, accounting for the order.
        for analysisInd in range(len(self.analysisOrder)):
            biomarkerType = self.analysisOrder[analysisInd]
            # Organize the streaming channels by their respective biomarker.
            self.numChannelDist[analysisInd] = (streamingOrder == biomarkerType).sum()
            self.channelDist[analysisInd] = np.where(streamingOrder == biomarkerType)[0]

            # Check that we segmented the channels correctly.
            assert self.numChannelDist[analysisInd] == len(self.channelDist[analysisInd]), "numChannelDist: " + str(self.numChannelDist) + "; channelDist: " + str(self.channelDist)
        assert sum(self.numChannelDist) == self.numChannels, "The streaming map is missing values: " + str(streamingOrder)

        # Initialize global plotting class.
        self.plottingClass = plottingProtocols(self.numChannels, self.channelDist, self.analysisOrder) if self.plotStreamedData else None
        # Pointer to analysis classes.
        self.analysisProtocols = {
            'eog': eogProtocol(self.numPointsPerBatch, self.moveDataFinger, self.numChannelDist[self.analysisOrder.index('eog')], self.plottingClass, self) if 'eog' in self.analysisOrder else None,
            'eeg': eegProtocol(self.numPointsPerBatch, self.moveDataFinger, self.numChannelDist[self.analysisOrder.index('eeg')], self.plottingClass, self) if 'eeg' in self.analysisOrder else None,
            'ecg': ecgProtocol(self.numPointsPerBatch, self.moveDataFinger, self.numChannelDist[self.analysisOrder.index('ecg')], self.plottingClass, self) if 'ecg' in self.analysisOrder else None,
            'eda': edaProtocol(self.numPointsPerBatch, self.moveDataFinger, self.numChannelDist[self.analysisOrder.index('eda')], self.plottingClass, self) if 'eda' in self.analysisOrder else None,
            'emg': emgProtocol(self.numPointsPerBatch, self.moveDataFinger, self.numChannelDist[self.analysisOrder.index('emg')], self.plottingClass, self) if 'emg' in self.analysisOrder else None,
            'temp': tempProtocol(self.numPointsPerBatch, self.moveDataFinger, self.numChannelDist[self.analysisOrder.index('temp')], self.plottingClass, self) if 'temp' in self.analysisOrder else None,
            'lowfreq': generalProtocol_lowFreq(self.numPointsPerBatch, self.moveDataFinger, self.numChannelDist[self.analysisOrder.index('lowfreq')], self.plottingClass, self) if 'lowfreq' in self.analysisOrder else None,
            'highfreq': generalProtocol_highFreq(self.numPointsPerBatch, self.moveDataFinger, self.numChannelDist[self.analysisOrder.index('highfreq')], self.plottingClass, self) if 'highfreq' in self.analysisOrder else None,
        }

        self.analysisList = []
        # Generate a list of all analyses, keeping the streaming order.
        for biomarkerType in self.analysisOrder:
            self.analysisList.append(self.analysisProtocols[biomarkerType])

        super().__init__(modelClasses, actionControl, self.analysisProtocols, biomarkerOrder, biomarkerChannelIndices, featureAverageWindows)
        # Initialize mutable variables.
        self.resetGlobalVariables()

    def resetGlobalVariables(self):
        self.resetFeatureInformation()
        # Reset the analysis information
        for analysis in self.analysisList:
            analysis.resetGlobalVariables()

        # Experimental information
        # self.experimentTimes = []       # A list of lists of [start, stop] times of each experiment, where each element represents the times for one experiment. None means no time recorded.
        # self.experimentNames = []       # A list of names for each experiment, where len(experimentNames) == len(experimentTimes).
        # Survey Information
        self.surveyAnswersList = []  # A list of lists of survey answers, where each element represents a list of answers to surveyQuestions.
        self.surveyAnswerTimes = []  # A list of times when each survey was collected, where the len(surveyAnswerTimes) == len(surveyAnswersList).
        self.surveyQuestions = []  # A list of survey questions asked to the user, where each element in surveyAnswersList corresponds to this question order.
        # Subject Information
        self.subjectInformationAnswers = []  # A list of subject background answers, where each element represents an answer to subjectInformationQuestions.
        self.subjectInformationQuestions = []  # A list of subject background questions asked to the user, such as race, age, and gender.

    def getCurrentTime(self):
        # Return the last streaming timePoint
        while len(self.analysisList[0].data[0]) == 0:
            print("\tWaiting for arduino to initialize!");
            time.sleep(1)
        return self.analysisList[0].data[0][-1]

    def setupArduinoStream(self, stopTimeStreaming, usingTimestamps=False):
        # self.arduinoRead.resetArduino(self.mainArduino, 10)
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
                stopTimeStreaming = str(currentTime).replace(".", ":")
            # Get the Final Time in Seconds (From 12:00am of the Current Day) to Stop Streaming
            stopTimeStreaming = self.convertToTime(stopTimeStreaming) + timeBuffer

        return stopTimeStreaming

    def recordData(self, maxVolt=3.3, adcResolution=4096):
        # Read in at least one point
        rawReadsList = []
        while (int(self.mainArduino.in_waiting) > 0 or len(rawReadsList) == 0):
            rawReadsList.append(self.arduinoRead.readline(ser=self.mainArduino))

        # Parse the Data
        Voltages, timePoints = self.arduinoRead.parseCompressedRead(rawReadsList, self.numChannels, maxVolt, adcResolution)
        # Organize the Data for Processing
        self.organizeData(timePoints, Voltages)

    def organizeData(self, timePoints, Voltages):
        if len(timePoints[0]) == 0:
            print("\t !!! NO POINTS FOUND !!!")

        # Update the data (if present) for each sensor
        for analysisInd in range(len(self.analysisList)):
            analysis = self.analysisList[analysisInd]
            # Check if this sensor is streaming data
            if self.numChannelDist[analysisInd] != 0:
                analysis.data[0].extend(timePoints[0])
                for channelIndex in range(len(self.channelDist[analysisInd])):
                    # Compile the Voltage Data
                    streamingDataIndex = self.channelDist[analysisInd][channelIndex]
                    newVoltageData = Voltages[streamingDataIndex]
                    # Add the Data to the Correct Channel
                    analysis.data[1][channelIndex].extend(newVoltageData)

    @staticmethod
    def convertToTime(timeStamp):
        if type(timeStamp) == str:
            timeStamp = timeStamp.split(":")
        timeStamp.reverse()

        currentTime = 0
        orderOfInput = [1E-6, 1, 60, 60 * 60, 60 * 60 * 24]
        for i, timeStampVal in enumerate(timeStamp):
            currentTime += orderOfInput[i] * int(timeStampVal)
        return currentTime

    @staticmethod
    def convertToTimeStamp(timeSeconds):
        hours = timeSeconds // 3600
        remainingTime = timeSeconds % 3600
        minutes = remainingTime // 60
        remainingTime %= 60
        seconds = math.floor(remainingTime)
        microSeconds = remainingTime - seconds
        microSeconds = np.round(microSeconds, 6)
        return hours, minutes, seconds, microSeconds


# -------------------------------------------------------------------------- #
# ---------------------------- Reading All Data ---------------------------- #

class streamingProtocols(streamingHead):

    def __init__(self, mainSerialNum, modelClasses, actionControl, numPointsPerBatch, moveDataFinger, streamingOrder, biomarkerOrder, featureAverageWindows, plotStreamedData):
        # Create Pointer to Common Functions
        super().__init__(mainSerialNum, None, modelClasses, actionControl, numPointsPerBatch, moveDataFinger, streamingOrder, biomarkerOrder, featureAverageWindows, plotStreamedData)

    def analyzeBatchData(self, dataFinger, lastTimePoint):
        # Analyze the current data
        for analysis in self.analysisList:
            analysis.analyzeData(dataFinger)

        # Organize the new features
        self.organizeRawFeatures()
        self.alignFeatures(lastTimePoint, secondsPerPoint=1)
        # self.predictLabels()

        # Plot the Data
        if self.plotStreamedData: self.plottingClass.displayData()

        # Move the dataFinger pointer to analyze the next batch of data
        return dataFinger + self.moveDataFinger

    def streamArduinoData(self, maxVolt, adcResolution, stopTimeStreaming, filePath):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Streaming in Data from the Arduino")
        # Reset Global Variable in Case it Was Previously Populated
        self.resetGlobalVariables()

        # Get user information
        self.setUserName(filePath)

        # Prepare the arduino to stream in data
        self.stopTimeStreaming = self.setupArduinoStream(stopTimeStreaming)
        timePoints = self.analysisList[0].data[0]
        dataFinger = 0

        try:
            print("\tBeginning to loop through streamed data")
            # Loop Through and Read the Arduino Data in Real-Time
            while len(timePoints) == 0 or (timePoints[-1] - timePoints[0]) < self.stopTimeStreaming:
                # Stream in the Latest Data
                self.recordData(maxVolt, adcResolution)

                # When enough data has been collected, analyze the new data in batches.
                while len(timePoints) - dataFinger >= self.numPointsPerBatch:
                    dataFinger = self.analyzeBatchData(dataFinger, timePoints[-1])

            # At the end, analyze all remaining data
            dataFinger = self.analyzeBatchData(dataFinger, timePoints[-1])

        except Exception as error:
            self.mainArduino.close();
            print(error)

        finally:
            # Set the final experimental time to the end of the experiment
            if len(self.experimentTimes) != 0 and self.experimentTimes[-1][1] is None:
                self.experimentTimes[-1][1] == timePoints[-1]

            # Close the Arduinos at the End
            print("\nFinished Streaming in Data; Closing Arduino\n")
            self.mainArduino.close()

    def streamExcelData(self, compiledRawData, experimentTimes, experimentNames, surveyAnswerTimes,
                        surveyAnswersList, surveyQuestions, subjectInformationAnswers, subjectInformationQuestions, filePath):
        print("\tAnalyzing the Excel Data")
        # Reset Global Variable in Case it Was Previously Populated
        self.resetGlobalVariables()

        # Add Experimental information
        self.surveyQuestions = surveyQuestions
        self.subjectInformationAnswers = subjectInformationAnswers
        self.subjectInformationQuestions = subjectInformationQuestions
        # Experiment information parameters
        self.experimentInfoPointerStart = 0;
        self.experimentInfoPointerEnd = 0;
        self.featureInfoPointer = 0;
        # Get user information
        self.setUserName(filePath)

        # Extract the Time and Voltage Data
        timePoints, Voltages = compiledRawData;
        Voltages = np.array(Voltages)

        dataFinger = 0;
        generalDataFinger = 0;
        # Loop Through and Read the Excel Data in Pseudo-Real-Time
        while generalDataFinger < len(timePoints):
            # Organize the Input Data
            self.organizeData([timePoints[generalDataFinger:generalDataFinger + self.moveDataFinger]], Voltages[:, generalDataFinger:generalDataFinger + self.moveDataFinger])

            # When enough data has been collected, analyze the new data in batches.
            while generalDataFinger + self.moveDataFinger - dataFinger >= self.numPointsPerBatch:
                lastTimePoint = timePoints[min(generalDataFinger + self.moveDataFinger, len(timePoints)) - 1]
                dataFinger = self.analyzeBatchData(dataFinger, lastTimePoint)
            # Organize experimental information.
            self.organizeExperimentalInformation(timePoints, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, generalDataFinger)

            # Move onto the next batch of data
            generalDataFinger += self.moveDataFinger

        # At the end, analyze all remaining data
        lastTimePoint = timePoints[min(generalDataFinger + self.moveDataFinger, len(timePoints)) - 1]
        dataFinger = self.analyzeBatchData(dataFinger, lastTimePoint)
        # Organize experimental information.
        self.organizeExperimentalInformation(timePoints, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, generalDataFinger)

        # Assert that experimental information was read in correctly.
        assert np.array_equal(experimentTimes, self.experimentTimes), f"{experimentTimes} \n {self.experimentTimes}"
        assert (np.array(experimentNames) == np.array(self.experimentNames)).all(), experimentNames
        # Assert that experimental information was read in correctly.
        assert np.array_equal(surveyAnswerTimes, self.surveyAnswerTimes), print(surveyAnswerTimes, self.surveyAnswerTimes)
        assert np.array_equal(surveyAnswersList, self.surveyAnswersList), print(surveyAnswersList, self.surveyAnswersList)

        # Finished Analyzing the Data
        print("\n\tFinished Analyzing Excel Data")

    def organizeExperimentalInformation(self, timePoints, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, generalDataFinger):
        # Add the experiment information when the timepoint is reached.
        while self.experimentInfoPointerStart != len(experimentTimes) and experimentTimes[self.experimentInfoPointerStart][0] <= timePoints[min(len(timePoints) - 1, generalDataFinger + self.moveDataFinger - 1)]:
            self.experimentTimes.append([experimentTimes[self.experimentInfoPointerStart][0], None])
            self.experimentNames.append(experimentNames[self.experimentInfoPointerStart])
            self.experimentInfoPointerStart += 1
        while self.experimentInfoPointerEnd != len(experimentTimes) and experimentTimes[self.experimentInfoPointerEnd][1] <= timePoints[min(len(timePoints) - 1, generalDataFinger + self.moveDataFinger - 1)]:
            self.experimentTimes[self.experimentInfoPointerEnd][1] = experimentTimes[self.experimentInfoPointerEnd][1]
            self.experimentInfoPointerEnd += 1
        # Add the feature information when the timepoint is reached.
        while self.featureInfoPointer != len(surveyAnswerTimes) and surveyAnswerTimes[self.featureInfoPointer] <= timePoints[min(len(timePoints) - 1, generalDataFinger + self.moveDataFinger - 1)]:
            self.surveyAnswerTimes.append(surveyAnswerTimes[self.featureInfoPointer])
            self.surveyAnswersList.append(surveyAnswersList[self.featureInfoPointer])
            self.featureInfoPointer += 1

# -------------------------------------------------------------------------- #
