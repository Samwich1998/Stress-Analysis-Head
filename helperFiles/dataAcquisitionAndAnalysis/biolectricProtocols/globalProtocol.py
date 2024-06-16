import abc
import numpy as np
import matplotlib.pyplot as plt

# Import Files
from .helperMethods.universalProtocols import universalMethods  # Import files with general analysis methods
from .helperMethods.filteringProtocols import filteringMethods  # Import files with filtering methods
from .helperMethods.plottingMethods import plottingMethods  # Import files with plotting methods
from .helperMethods.mneInterface import mneInterface  # Import files with MNE-features methods


class globalProtocol(abc.ABC):

    def __init__(self, analysisType, numPointsPerBatch=3000, moveDataFinger=10, channelIndices=(), plottingClass=None, readData=None):
        # General input parameters
        self.plotStreamedData = plottingClass is not None  # Plot the Data
        self.numPointsPerBatch = numPointsPerBatch  # The X-With of the Plot (Number of Data-Points Shown)
        self.streamingChannelInds = channelIndices  # The Indices of the Channels to Analyze in streamingOrder
        self.moveDataFinger = moveDataFinger  # The Amount of Data to Stream in Before Finding Peaks
        self.numChannels = len(channelIndices)  # Number of Bioelectric Signals
        self.plottingClass = plottingClass
        self.analysisType = analysisType
        self.collectFeatures = False  # User will change this flag if desired (NOT here).
        self.readData = readData

        # General mutable parameters: unifying GENERIC protocols.
        self.featureTimeWindow_highFreq = 12  # The duration of time that each feature considers.
        self.featureTimeWindow_lowFreq = 15  # The duration of time that each feature considers.
        self.dataPointBuffer = 5000  # A Prepended Buffer in the Filtered Data that Represents BAD Filtering; Units: Points
        self.secondsPerFeature = 1  # The duration of time that passes between each feature.

        # Feature parameters.
        self.featureChannelIndices = None  # The Indices of the Channels to Analyze in biomarkerFeatureOrder
        self.featureAverageWindow = None  # The number of seconds before each feature to average the features together. Set in streamData if collecting features.
        self.lastAnalyzedDataInd = None
        self.compiledFeatures = None
        self.rawFeatureTimes = None
        self.samplingFreq = None
        self.rawFeatures = None
        self.channelData = None
        self.timePoints = None

        # Prepare the Program to Begin Data Analysis
        self.resetGlobalVariables()  # Start with Fresh Inputs (Clear All Arrays/Values)
        self.checkGlobalParams()  # Check to See if the User's Input Parameters Make Sense

        # Define general classes to process data.
        self.filteringMethods = filteringMethods()
        self.universalMethods = universalMethods()
        self.mneInterface = mneInterface()

        # If Plotting, Define Class for Plotting Peaks
        if self.plotStreamedData and self.numChannels != 0:
            # Initialize the methods for plotting this analysis,
            self.plottingMethods = plottingMethods(analysisType, self.numChannels, plottingClass)
            # Initialize the plots for the analysis.
            self.plottingMethods.initPlotPeaks()

    def resetGlobalVariables(self):
        # Reset the data parameters.
        self.channelData = [[] for _ in range(self.numChannels)]  # The Data Streamed in from the Arduino
        self.samplingFreq = None  # The Average Number of Points Steamed Into the Arduino Per Second; If NONE Given, Algorithm will Calculate Based on Initial Data
        self.timePoints = []  # The Time Points of the Data

        # Reset Feature Extraction
        self.lastAnalyzedDataInd = np.array([0 for _ in range(self.numChannels)])  # The index of the last point analyzed.
        self.compiledFeatures = [[] for _ in range(self.numChannels)]  # FINAL compiled features at the current timepoint. Could be average of last x features.
        self.rawFeatureTimes = [[] for _ in range(self.numChannels)]  # The time of each feature.
        self.rawFeatures = [[] for _ in range(self.numChannels)]  # Raw features extraction at the current timepoint.

        # Close Any Opened Plots
        if self.plotStreamedData:
            plt.close('all')

        # Reset the analysis variables.
        self.resetAnalysisVariables()

    def checkGlobalParams(self):
        assert self.moveDataFinger < self.numPointsPerBatch, "You are Analyzing Too Much Data in a Batch. 'moveDataFinger' MUST be Less than 'numPointsPerBatch'"

        # Check to see if the user has set the correct parameters for the analysis.
        self.checkParams()

    def setSamplingFrequency(self, startFilterPointer):
        # Calculate the Sampling Frequency
        self.samplingFreq = len(self.timePoints[startFilterPointer:-1]) / (self.timePoints[-1] - self.timePoints[startFilterPointer])
        print(f"\n\tSetting {self.analysisType} Sampling Frequency to {self.samplingFreq}")
        print("\tIf this protocol runs longer than", self.moveDataFinger / self.samplingFreq, ", the analysis will NOT be in real-time")

        self.setSamplingFrequencyParams()

    def setFeatureCollectionParams(self, featureAverageWindow, featureChannelIndices):
        if self.featureAverageWindow is not None:
            assert featureAverageWindow == self.featureAverageWindow, "Feature Average Window is not the same as the one set in the protocol"
        
        # Set the feature collection parameters.
        self.featureChannelIndices = featureChannelIndices  # Add the feature channel indices to the analysis protocol
        self.featureAverageWindow = featureAverageWindow  # Add the feature average window to the analysis protocol
        self.collectFeatures = True

        # Assert that if you are collecting features on a biomarker, you are collecting features across ALL channels.
        assert len(self.featureChannelIndices) == len(self.streamingChannelInds), \
            f"Only taking features from a single biomarker channel is NOT supported! I would not be able to distinguish which channel you wanted given my setup: {self.featureChannelIndices}. {self.streamingChannelInds}"

    # ------------------------ Child Class Contract ------------------------ #

    @abc.abstractmethod
    def resetAnalysisVariables(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")

    @abc.abstractmethod
    def checkParams(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")

    @abc.abstractmethod
    def setSamplingFrequencyParams(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")

    @abc.abstractmethod
    def analyzeData(self, dataFinger):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")

    @abc.abstractmethod
    def filterData(self, timePoints, data):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")

        # ---------------------------------------------------------------------- #

# -------------------------------------------------------------------------- #
