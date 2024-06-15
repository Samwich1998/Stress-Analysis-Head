import collections
import numpy as np

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
from .humanMachineInterface.arduinoInterface import arduinoRead  # Functions to Read in Data from Arduino
from .humanMachineInterface.featureOrganization import featureOrganization

# Import plotting protocols
from .dataVisualization.biolectricPlottingProtocols import plottingProtocols

# Parameters for the streamingProtocolHelpers class:
#     Biomarker information:
#         streamingOrder: a list of incoming biomarkers in the order they appear; Dim: numStreamedSignals
#         analysisOrder: a unique list of biomarkers in streamingOrder; Dim: numUniqueSignals
#
#     Channel information:
#         numChannelDist: The integer number of channels used in each analysis; Dim: numChannels_perAnalysis
#         channelDist: A dictionary of arrays, specifying each biomarker's indices in streamingOrder; Dim: numUniqueSignals, numChannels_perAnalysis


class streamingProtocolHelpers(featureOrganization):

    def __init__(self, mainSerialNum, therapySerialNum, modelClasses, actionControl, numPointsPerBatch, moveDataFinger, streamingOrder, extractFeaturesFrom, featureAverageWindows, voltageRange, plotStreamedData):
        # General streaming parameters.
        self.streamingOrder = np.char.lower(streamingOrder)  # The names of each recorded signal in order. Ex: ['eog', 'eog', 'eeg', 'eda']
        self.numStreamedSignals = len(streamingOrder)  # The total number of signals being recorded.
        self.numPointsPerBatch = numPointsPerBatch  # The number of points to analyze in each batch.
        self.plotStreamedData = plotStreamedData  # Boolean: whether to graph the incoming signals + analysis.
        self.moveDataFinger = moveDataFinger  # The minimum number of NEW points to analyze in each batch.
        self.voltageRange = voltageRange  # The voltage range of the incoming signals.

        # Store the arduinoRead Instance
        if mainSerialNum is not None:
            self.arduinoRead = arduinoRead(mainSerialNum=mainSerialNum, therapySerialNum=therapySerialNum)
            self.mainArduino = self.arduinoRead.mainArduino

        # Specify the analysis order: a unique list of biomarkers in streamingOrder.
        self.analysisOrder = list(collections.OrderedDict.fromkeys(self.streamingOrder))  # The set of unique biomarkers, maintaining the order they will be analyzed. Ex: ['eog', 'eeg', 'eda']
        self.numUniqueSignals = len(self.analysisOrder)  # The number of unique biomarkers being recorded.

        # Variables that rely on the incoming data's order.
        self.numChannelDist = np.zeros(self.numUniqueSignals)  # Track the number of channels used by each sensor.
        self.channelDist = {}  # Track the order (its index) each sensor comes in. Dim: numUniqueSignals. numChannelsPerBiomarker

        # Populate the variables, accounting for the order.
        for analysisInd in range(len(self.analysisOrder)):
            biomarkerType = self.analysisOrder[analysisInd]

            # Find the locations where each biomarker appears.
            streamingChannelIndices = self.streamingOrder == biomarkerType

            # Organize the streaming channels by their respective biomarker.
            self.numChannelDist[analysisInd] = np.sum(streamingChannelIndices)
            self.channelDist[biomarkerType] = np.where(streamingChannelIndices)[0]
        assert np.sum(self.numChannelDist) == self.numStreamedSignals, f"The number of channels per biomarker ({self.numChannelDist}) does not align with the streaming order ({self.streamingOrder})"

        # Initialize global plotting class.
        self.plottingClass = plottingProtocols(self.numStreamedSignals, self.channelDist, self.analysisOrder) if self.plotStreamedData else None
        self.analysisProtocols = {
            'highfreq': generalProtocol_highFreq(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['highfreq'], self.plottingClass, self) if 'highfreq' in self.analysisOrder else None,
            'lowfreq': generalProtocol_lowFreq(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['lowfreq'], self.plottingClass, self) if 'lowfreq' in self.analysisOrder else None,
            'temp': tempProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['temp'], self.plottingClass, self) if 'temp' in self.analysisOrder else None,
            'eog': eogProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['eog'], self.plottingClass, self, voltageRange) if 'eog' in self.analysisOrder else None,
            'eeg': eegProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['eeg'], self.plottingClass, self) if 'eeg' in self.analysisOrder else None,
            'ecg': ecgProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['ecg'], self.plottingClass, self) if 'ecg' in self.analysisOrder else None,
            'eda': edaProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['eda'], self.plottingClass, self) if 'eda' in self.analysisOrder else None,
            'emg': emgProtocol(self.numPointsPerBatch, self.moveDataFinger, self.channelDist['emg'], self.plottingClass, self) if 'emg' in self.analysisOrder else None,
        }

        self.analysisList = []
        # Generate a list of all analyses, keeping the streaming order.
        for biomarkerType in self.analysisOrder:
            self.analysisList.append(self.analysisProtocols[biomarkerType])

        # Holder parameters.
        self.subjectInformationQuestions = None  # A list of subject background questions
        self.subjectInformationAnswers = None  # A list of subject background answers, where each element represents an answer to subjectInformationQuestions.
        self.surveyAnswersList = None  # A list of lists of survey answers, where each element represents an answer to surveyQuestions.
        self.surveyAnswerTimes = None  # A list of times when each survey was collected, where the len(surveyAnswerTimes) == len(surveyAnswersList).
        self.surveyQuestions = None  # A list of survey questions, where each element in surveyAnswersList corresponds to this question order.
        self.experimentTimes = None  # A list of lists of [start, stop] times of each experiment, where each element represents the times for one experiment. None means no time recorded.
        self.experimentNames = None  # A list of names for each experiment, where len(experimentNames) == len(experimentTimes).

        # Finish setting up the class.
        super().__init__(modelClasses, actionControl, self.analysisProtocols, extractFeaturesFrom, featureAverageWindows)
        self.resetStreamingInformation()

    def resetStreamingInformation(self):
        self.resetFeatureInformation()
        # Reset the analysis information
        for analysis in self.analysisList:
            analysis.resetAnalysisVariables()

        # Subject Information
        self.subjectInformationQuestions = []  # A list of subject background questions, such as race, age, and gender.
        self.subjectInformationAnswers = []  # A list of subject background answers, where each element represents an answer to subjectInformationQuestions.

        # Survey Information
        self.surveyAnswersList = []  # A list of lists of survey answers, where each element represents a list of answers to surveyQuestions.
        self.surveyAnswerTimes = []  # A list of times when each survey was collected, where the len(surveyAnswerTimes) == len(surveyAnswersList).
        self.surveyQuestions = []  # A list of survey questions, where each element in surveyAnswersList corresponds to this question order.

        # Experimental information
        self.experimentTimes = []  # A list of lists of [start, stop] times of each experiment, where each element represents the times for one experiment. None means no time recorded.
        self.experimentNames = []  # A list of names for each experiment, where len(experimentNames) == len(experimentTimes).

    def analyzeBatchData(self, streamingDataFinger):
        # Analyze the current data
        for analysis in self.analysisList:
            analysis.analyzeData(streamingDataFinger)

        # Organize the new features
        self.organizeRawFeatures()
        self.alignFeatures()
        # self.predictLabels()

        # Plot the Data
        if self.plotStreamedData: self.plottingClass.displayData()

        # Move the streamingDataFinger pointer to analyze the next batch of data
        return streamingDataFinger + self.moveDataFinger

    def recordData(self, maxVolt=3.3, adcResolution=4096):
        # Read in at least one point
        rawReadsList = []
        while int(self.mainArduino.in_waiting) > 0 or len(rawReadsList) == 0:
            rawReadsList.append(self.arduinoRead.readline(ser=self.mainArduino))

        # Parse the Data
        timePoints, Voltages = self.arduinoRead.parseCompressedRead(rawReadsList, self.numStreamedSignals, maxVolt, adcResolution)
        self.organizeData(timePoints, Voltages)  # Organize the data for further processing

    def organizeData(self, timePoints, Voltages):
        if len(timePoints) == 0:
            print("\tNO NEW TIMEPOINTS ADDED")

        # Update the data (if present) for each sensor
        for analysisInd in range(len(self.analysisOrder)):
            analysis = self.analysisList[analysisInd]

            # Skip if no data is present for this sensor
            if analysis.numChannels == 0: continue

            # Update the timepoints.
            analysis.timePoints.extend(timePoints)

            # For each channel, update the voltage data.
            for channelIndex in range(analysis.numChannels):
                # Compile the voltages for each of the sensor's channels.
                streamingDataIndex = analysis.streamingChannelInds[channelIndex]
                newVoltageData = Voltages[streamingDataIndex]

                # Add the Data to the Correct Channel
                analysis.channelData[channelIndex].extend(newVoltageData)
