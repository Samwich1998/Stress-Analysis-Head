import numpy as np
import scipy

# Import Files
from .globalProtocol import globalProtocol


class tempProtocol(globalProtocol):

    def __init__(self, numPointsPerBatch=2000, moveDataFinger=200, numChannels=1, plottingClass=None, readData=None):
        # Feature collection parameters
        self.secondsPerFeature = 1  # The duration of time that passes between each feature.
        self.featureTimeWindow = 60  # The duration of time that each feature considers; 5 - 15
        # High Pass Filter Parameters
        self.dataPointBuffer = 5000  # A Prepended Buffer in the Filtered Data that Represents BAD Filtering; Units: Points
        self.cutOffFreq = [None, 0.1]  # Optimal LPF Cutoff in Literature is 6-8 or 20 Hz (Max 35 or 50); I Found 25 Hz was the Best, but can go to 15 if noisy (small amplitude cutoff)

        # Holder parameters.
        self.startFeatureTimePointer = None  # The start pointer of the feature window interval.
        self.minPointsPerBatch = None  # The minimum number of points per batch.

        # Initialize common model class
        super().__init__("temp", numPointsPerBatch, moveDataFinger, numChannels, plottingClass, readData)

    def resetAnalysisVariables(self):
        # General parameters 
        self.startFeatureTimePointer = [0 for _ in range(self.numChannels)]  # The start pointer of the feature window interval.

    def checkParams(self):
        assert self.featureTimeWindow < self.dataPointBuffer, "The buffer does not include enough points for the feature window"

    def setSamplingFrequencyParams(self):
        maxBufferSeconds = max(self.featureTimeWindow, 15)
        # Set Parameters
        self.lastAnalyzedDataInd[:] = int(self.samplingFreq * self.featureTimeWindow)
        self.minPointsPerBatch = int(self.samplingFreq * self.featureTimeWindow / 2)
        self.dataPointBuffer = max(self.dataPointBuffer, int(self.samplingFreq * maxBufferSeconds))

    # ----------------------------------------------------------------------- #
    # ------------------------- Data Analysis Begins ------------------------ #

    def analyzeData(self, dataFinger):

        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):

            # ---------------------- Filter the Data ----------------------- #    
            # Find the starting/ending points of the data to analyze
            startFilterPointer = max(dataFinger - self.dataPointBuffer, 0)
            dataBuffer = np.array(self.data[1][channelIndex][startFilterPointer:dataFinger + self.numPointsPerBatch])
            timePoints = np.array(self.data[0][startFilterPointer:dataFinger + self.numPointsPerBatch])

            # Get the Sampling Frequency from the First Batch (If Not Given)
            if not self.samplingFreq:
                self.setSamplingFrequency(startFilterPointer)

            # Filter the data and remove bad indices
            filteredTime, filteredData, goodIndicesMask = self.filterData(timePoints, dataBuffer, removePoints=True)
            # --------------------------------------------------------------- #

            # ---------------------- Feature Extraction --------------------- #
            if self.collectFeatures:
                # Extract features across the dataset
                while self.lastAnalyzedDataInd[channelIndex] < len(self.data[0]):
                    featureTime = self.data[0][self.lastAnalyzedDataInd[channelIndex]]

                    # Find the start window pointer
                    self.startFeatureTimePointer[channelIndex] = self.findStartFeatureWindow(self.startFeatureTimePointer[channelIndex], featureTime, self.featureTimeWindow)
                    # Compile the good data in the feature interval.
                    intervalTimes, intervalData = self.compileBatchData(filteredTime, filteredData, goodIndicesMask, startFilterPointer, self.startFeatureTimePointer[channelIndex], channelIndex)

                    # Only extract features if enough information is provided.
                    if self.minPointsPerBatch < len(intervalTimes):
                        # Calculate and save the features in this window.
                        finalFeatures = self.extractFeatures(intervalTimes, intervalData)
                        self.readData.averageFeatures([featureTime], [finalFeatures], self.featureTimes[channelIndex],
                                                      self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex], self.featureAverageWindow)

                    # Keep track of which data has been analyzed 
                    self.lastAnalyzedDataInd[channelIndex] += int(self.samplingFreq * self.secondsPerFeature)
            # -------------------------------------------------------------- #  

            # ------------------- Plot Biolectric Signals ------------------- #
            if self.plotStreamedData:
                # Format the raw data:.
                timePoints = timePoints[dataFinger - startFilterPointer:]  # Shared axis for all signals
                rawData = dataBuffer[dataFinger - startFilterPointer:]
                # Format the filtered data
                filterOffset = (goodIndicesMask[0:dataFinger - startFilterPointer]).sum(axis=0, dtype=int)

                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.plottingMethods.bioelectricDataPlots[channelIndex].set_data(timePoints, rawData)
                self.plottingMethods.bioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1])

                # Plot the Filtered + Digitized Data
                self.plottingMethods.filteredBioelectricDataPlots[channelIndex].set_data(filteredTime[filterOffset:], filteredData[filterOffset:])
                self.plottingMethods.filteredBioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1])

                # Plot a single feature.
                if len(self.compiledFeatures[channelIndex]) != 0:
                    self.plottingMethods.featureDataPlots[channelIndex].set_data(self.featureTimes[channelIndex], np.array(self.compiledFeatures[channelIndex])[:, 9])
                    self.plottingMethods.featureDataPlotAxes[channelIndex].legend(["Signal Slope"], loc="upper left")

            # --------------------------------------------------------------- #   

    def filterData(self, timePoints, data, removePoints=False):
        # Filter the data
        filteredData = self.filteringMethods.bandPassFilter.butterFilter(data, self.cutOffFreq[1], self.samplingFreq, order=1, filterType='low')

        if removePoints:
            # Find the bad points associated with motion artifacts
            deriv = abs(np.gradient(filteredData, timePoints))
            motionIndices = deriv > 1
            motionIndices_Broadened = scipy.signal.savgol_filter(motionIndices, max(3, int(self.samplingFreq * 20)), 1, mode='nearest', deriv=0)
            goodIndicesMask = motionIndices_Broadened < 0.01
        else:
            goodIndicesMask = np.full_like(data, True, dtype=bool)

        # Remove the bad points from the data
        filteredTime = timePoints[goodIndicesMask]
        filteredData = filteredData[goodIndicesMask]

        # Finish filtering the data
        filteredData = scipy.signal.savgol_filter(filteredData, max(3, int(self.samplingFreq * 15)), 1, mode='nearest', deriv=0)

        return filteredTime, filteredData, goodIndicesMask

    def findStartFeatureWindow(self, timePointer, currentTime, timeWindow):
        # Loop through until you find the first time in the window 
        while self.data[0][timePointer] < currentTime - timeWindow:
            timePointer += 1

        return timePointer

    def compileBatchData(self, filteredTime, filteredData, goodIndicesMask, startFilterPointer, startFeatureTimePointer, channelIndex):
        assert len(goodIndicesMask) >= len(filteredData) == len(filteredTime), print(len(goodIndicesMask), len(filteredData), len(filteredTime))

        # Accounts for the missing points (count the number of viable points within each pointer).
        startReferenceFinger = (goodIndicesMask[0:startFeatureTimePointer - startFilterPointer]).sum(axis=0, dtype=int)
        endReferenceFinger = startReferenceFinger + (goodIndicesMask[startFeatureTimePointer - startFilterPointer:self.lastAnalyzedDataInd[channelIndex] + 1 - startFilterPointer]).sum(axis=0, dtype=int)
        # Compile the information in the interval.
        intervalTimes = filteredTime[startReferenceFinger:endReferenceFinger]
        intervalData = filteredData[startReferenceFinger:endReferenceFinger]

        return intervalTimes, intervalData

    # ---------------------------------------------------------------------- #
    # --------------------- Feature Extraction Methods --------------------- #

    def extractFeatures(self, timePoints, data):
        # ------------------------------------------------------------------ #  
        # ----------------------- Data Preprocessing ----------------------- #

        # Normalize the data
        standardized_data = self.universalMethods.standardizeData(data)
        if all(standardized_data == 0):
            return [0 for _ in range(23)]

        # Get the baseline data
        baselineX = timePoints - timePoints[0]

        # Calculate the derivatives
        firstDerivative = np.gradient(standardized_data, timePoints)

        # ------------------------------------------------------------------ #
        # ----------------------- Features from Data ----------------------- #

        # General Shape Parameters
        signalPower = scipy.integrate.simpson(data ** 2, timePoints) / (baselineX[-1] - baselineX[0])
        signalArea = scipy.integrate.simpson(data, timePoints) / (baselineX[-1] - baselineX[0])
        signalRange = max(data) - min(data)
        standardDeviation = np.std(data, ddof=1)
        mean = np.mean(data)

        # -------------------- Features from Derivatives ------------------- #

        # First derivative features
        firstDerivativeMean = np.mean(firstDerivative)
        firstDerivativeStdDev = np.std(firstDerivative, ddof=1)
        firstDerivativePower = scipy.integrate.simpson(firstDerivative ** 2, timePoints) / (baselineX[-1] - baselineX[0])

        # ----------------------- Organize Features ------------------------ #

        finalFeatures = []
        # Add peak shape parameters
        finalFeatures.extend([mean, standardDeviation])
        finalFeatures.extend([signalRange, signalPower, signalArea])
        # Add derivative features
        finalFeatures.extend([firstDerivativeMean, firstDerivativeStdDev, firstDerivativePower])

        return finalFeatures
