import abc
import numpy as np
import matplotlib.pyplot as plt

# Import Files
from .helperMethods.universalProtocols import universalMethods  # Import files with general analysis methods
from .helperMethods.filteringProtocols import filteringMethods  # Import files with filtering methods
from .helperMethods.plottingMethods import plottingMethods  # Import files with plotting methods
from .helperMethods.mneInterface import mneInterface  # Import files with MNE-features methods


class globalProtocol(abc.ABC):

    def __init__(self, analysisType, numPointsPerBatch=3000, moveDataFinger=10, numChannels=2, plottingClass=None, readData=None):
        # General input parameters
        self.plotStreamedData = plottingClass is not None  # Plot the Data
        self.numPointsPerBatch = numPointsPerBatch  # The X-With of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger  # The Amount of Data to Stream in Before Finding Peaks
        self.plottingClass = plottingClass
        self.featureAverageWindow = None  # The number of seconds before each feature to average the features together. Set in streamData if collecting features.
        self.analysisType = analysisType
        self.numChannels = numChannels  # Number of Bioelectric Signals
        self.collectFeatures = False  # This flag will be changed by user if desired (NOT here).
        self.readData = readData

        # Prepare the Program to Begin Data Analysis
        self.checkAllParams()  # Check to See if the User's Input Parameters Make Sense
        self.resetGlobalVariables()  # Start with Fresh Inputs (Clear All Arrays/Values)

        # Define general classes to process data.
        self.filteringMethods = filteringMethods()
        self.universalMethods = universalMethods()
        self.mneInterface = mneInterface()

        # If Plotting, Define Class for Plotting Peaks
        if self.plotStreamedData and numChannels != 0:
            # Initialize the methods for plotting this analysis,
            self.plottingMethods = plottingMethods(analysisType, numChannels, plottingClass)
            # Initialize the plots for the analysis.
            self.plottingMethods.initPlotPeaks()

    def resetGlobalVariables(self):
        # Data to Read in
        self.data = [[], [[] for channelIndex in range(self.numChannels)]]
        # Reset Feature Extraction
        self.rawFeatures = [[] for channelIndex in range(self.numChannels)]  # Raw features extraction at the current timepoint.
        self.featureTimes = [[] for channelIndex in range(self.numChannels)]  # The time of each feature.
        self.compiledFeatures = [[] for channelIndex in range(self.numChannels)]  # FINAL compiled features at the current timepoint. Could be average of last x features.
        self.lastAnalyzedDataInd = np.array([0 for channelIndex in range(self.numChannels)])  # The index of the last point analyzed. 

        # General parameters
        self.samplingFreq = None  # The Average Number of Points Steamed Into the Arduino Per Second; Depends on the User's Hardware; If NONE Given, Algorithm will Calculate Based on Initial Data

        # Close Any Opened Plots
        if self.plotStreamedData:
            plt.close('all')

        self.resetAnalysisVariables()

    def checkAllParams(self):
        assert self.moveDataFinger < self.numPointsPerBatch, "You are Analyzing Too Much Data in a Batch. 'moveDataFinger' MUST be Less than 'numPointsPerBatch'"
        self.checkParams()

    def setSamplingFrequency(self, startFilterPointer):
        # Caluclate the Sampling Frequency
        self.samplingFreq = len(self.data[0][startFilterPointer:-1]) / (self.data[0][-1] - self.data[0][startFilterPointer])
        print(f"\n\tSetting {self.analysisType} Sampling Frequency to {self.samplingFreq}")
        print("\tIf this protocol runs longer than", self.moveDataFinger / self.samplingFreq, ", the analysis will NOT be in real-time")

        self.setSamplingFrequencyParams()

    def setFeatureCollectionParams(self, featureAverageWindow):
        self.featureAverageWindow = featureAverageWindow  # Add the feature average window to the analysis protocol
        self.collectFeatures = True

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
