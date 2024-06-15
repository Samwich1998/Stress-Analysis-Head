import math
import time
from datetime import datetime

import numpy as np

# Import helper files.
from .streamingProtocolHelpers import streamingProtocolHelpers


class streamingProtocols(streamingProtocolHelpers):

    def __init__(self, mainSerialNum, modelClasses, actionControl, numPointsPerBatch, moveDataFinger, streamingOrder, extractFeaturesFrom, featureAverageWindows, voltageRange, plotStreamedData):
        # Create Pointer to Common Functions
        super().__init__(mainSerialNum=mainSerialNum, therapySerialNum=None, modelClasses=modelClasses, actionControl=actionControl, numPointsPerBatch=numPointsPerBatch, moveDataFinger=moveDataFinger, streamingOrder=streamingOrder,
                         extractFeaturesFrom=extractFeaturesFrom, featureAverageWindows=featureAverageWindows, voltageRange=voltageRange, plotStreamedData=plotStreamedData)
        # Holder parameters.
        self.experimentInfoPointerStart = None
        self.experimentInfoPointerEnd = None
        self.featureInfoPointer = None
        self.stopTimeStreaming = None

    def resetGlobalVariables(self):
        self.resetStreamingInformation()
        # Reset the Experiment Information Pointers
        self.experimentInfoPointerStart = None
        self.experimentInfoPointerEnd = None
        self.featureInfoPointer = None
        self.stopTimeStreaming = None

    def setupArduinoStream(self, stopTimeStreaming, usingTimestamps=False):
        # self.arduinoRead.resetArduino(self.mainArduino, 10)
        # Read and throw out the first few reads
        rawReadsList = []
        while int(self.mainArduino.in_waiting) > 0 or len(rawReadsList) < 2000:
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

    def streamArduinoData(self, adcResolution, stopTimeStreaming, filePath):
        """Stop Streaming When we Obtain `stopTimeStreaming` from Arduino"""
        print("Streaming in Data from the Arduino")
        # Reset Global Variable in Case it Was Previously Populated
        self.resetGlobalVariables()

        # Get user information
        self.setUserName(filePath)

        # Prepare the arduino to stream in data
        self.stopTimeStreaming = self.setupArduinoStream(stopTimeStreaming)
        timePoints = self.analysisList[0].timePoints
        dataFinger = 0

        try:
            print("\tBeginning to loop through streamed data")
            # Loop Through and Read the Arduino Data in Real-Time
            while len(timePoints) == 0 or (timePoints[-1] - timePoints[0]) < self.stopTimeStreaming:
                # Stream in the Latest Data
                self.recordData(self.voltageRange[1], adcResolution)

                # When enough data has been collected, analyze the new data in batches.
                while len(timePoints) - dataFinger >= self.numPointsPerBatch:
                    dataFinger = self.analyzeBatchData(dataFinger)

            # At the end, analyze all remaining data
            self.analyzeBatchData(dataFinger)

        except Exception as error:
            self.mainArduino.close()
            print(error)

        finally:
            # Set the final experimental time to the end of the experiment
            if len(self.experimentTimes) != 0 and self.experimentTimes[-1][1] is None:
                self.experimentTimes[-1][1] = timePoints[-1]

            # Close the Arduino's at the End
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
        self.experimentInfoPointerStart = 0
        self.experimentInfoPointerEnd = 0
        self.featureInfoPointer = 0
        # Get user information
        self.setUserName(filePath)

        # Extract the Time and Voltage Data
        timePoints, Voltages = compiledRawData
        Voltages = np.asarray(Voltages)

        dataFinger = 0
        generalDataFinger = -self.moveDataFinger
        # Loop Through and Read the Excel Data in Pseudo-Real-Time
        while generalDataFinger != len(timePoints) - 1:
            generalDataFinger = min(len(timePoints) - 1, generalDataFinger + self.moveDataFinger)

            # Organize the Input Data
            self.organizeData(timePoints=timePoints[generalDataFinger:generalDataFinger + self.moveDataFinger], Voltages=Voltages[:, generalDataFinger:generalDataFinger + self.moveDataFinger])

            # When enough data has been collected, analyze the new data in batches.
            while self.numPointsPerBatch <= generalDataFinger + self.moveDataFinger - dataFinger:
                dataFinger = self.analyzeBatchData(dataFinger)
            # Organize experimental information.
            self.organizeExperimentalInformation(timePoints, experimentTimes, experimentNames, surveyAnswerTimes, surveyAnswersList, generalDataFinger)

        # Assert that experimental information was read in correctly.
        assert np.array_equal(experimentTimes, self.experimentTimes), f"{experimentTimes} \n {self.experimentTimes}"
        assert np.all(np.asarray(experimentNames) == np.asarray(self.experimentNames)), experimentNames
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

    def getCurrentTime(self):
        # Return the last streaming timePoint
        while len(self.analysisList[0].timePoints) == 0:
            print("\tWaiting for arduino to initialize!")
            time.sleep(1)
        return self.analysisList[0].timePoints[-1]

    @staticmethod
    def convertToTime(timeStamp):
        if isinstance(timeStamp, str):
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
