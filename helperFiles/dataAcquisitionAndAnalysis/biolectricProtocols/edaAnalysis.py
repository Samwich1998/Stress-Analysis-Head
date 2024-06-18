import numpy as np
import antropy
import scipy

# Import Files
from .globalProtocol import globalProtocol


class edaProtocol(globalProtocol):

    def __init__(self, numPointsPerBatch=3000, moveDataFinger=10, channelIndices=(), plottingClass=None, readData=None):
        # Feature collection parameters.
        self.startFeatureTimePointer_Phasic = None  # The start pointer of the feature window interval.
        self.startFeatureTimePointer_Tonic = None  # The start pointer of the feature window interval.
        self.featureTimeWindow_Phasic = None  # The duration of time that each feature considers.
        self.featureTimeWindow_Tonic = None  # The duration of time that each feature considers.
        self.minPointsPerBatchPhasic = None  # The minimum number of points required to extract features.
        self.minPointsPerBatchTonic = None  # The minimum number of points required to extract features.

        # Filter Parameters.
        self.tonicFrequencyCutoff = 0.05  # Maximum tonic component frequency.
        self.cutOffFreq = [None, 15]  # Filter cutoff frequencies: [HPF, LPF].

        # Reset analysis variables
        super().__init__("eda", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)
        self.resetAnalysisVariables()

    def resetAnalysisVariables(self):
        # General parameters
        self.startFeatureTimePointer_Phasic = [0 for _ in range(self.numChannels)]  # The start pointer of the feature window interval.
        self.startFeatureTimePointer_Tonic = [0 for _ in range(self.numChannels)]  # The start pointer of the feature window interval.

        # Feature collection parameters.
        self.minPointsPerBatchPhasic = 0  # The minimum number of points required to extract features.
        self.minPointsPerBatchTonic = 0  # The minimum number of points required to extract features.

        # Confirm the feature time windows are set.
        self.featureTimeWindow_Phasic = self.featureTimeWindow_highFreq  # The duration of time that each feature considers.
        self.featureTimeWindow_Tonic = self.featureTimeWindow_lowFreq  # The duration of time that each feature considers.

    def checkParams(self):
        # Confirm the buffer is large enough for the feature window.
        assert self.featureTimeWindow_Phasic < self.dataPointBuffer, "The buffer does not include enough points for the feature window"
        assert self.featureTimeWindow_Tonic < self.dataPointBuffer, "The buffer does not include enough points for the feature window"

    def setSamplingFrequencyParams(self):
        maxFeatureTimeWindow = max(self.featureTimeWindow_Tonic, self.featureTimeWindow_Phasic, 15)
        # Set Parameters
        self.minPointsPerBatchTonic = int(self.samplingFreq * self.featureTimeWindow_Tonic / 2)
        self.minPointsPerBatchPhasic = int(self.samplingFreq * self.featureTimeWindow_Phasic * 3 / 4)
        self.lastAnalyzedDataInd[:] = int(self.samplingFreq * maxFeatureTimeWindow)
        self.dataPointBuffer = max(self.dataPointBuffer, int(self.samplingFreq * maxFeatureTimeWindow))

    # ----------------------------------------------------------------------- #
    # ------------------------- Data Analysis Begins ------------------------ #

    def analyzeData(self, dataFinger):

        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):

            # ---------------------- Filter the Data ----------------------- #    
            # Find the starting/ending points of the data to analyze
            startFilterPointer = max(dataFinger - self.dataPointBuffer, 0)
            dataBuffer = np.array(self.channelData[channelIndex][startFilterPointer:dataFinger + self.numPointsPerBatch])
            timePoints = np.array(self.timePoints[startFilterPointer:dataFinger + self.numPointsPerBatch])

            # Extract sampling frequency from the first batch of data
            if not self.samplingFreq:
                self.setSamplingFrequency(startFilterPointer)

            # Filter the data and remove bad indices
            filteredTime, filteredData, goodIndicesMask = self.filterData(timePoints, dataBuffer, removePoints=True)

            # Separate the tonic (baseline) from the phasic (peaks) data
            tonicComponent, phasicComponent = self.splitPhasicTonic(filteredData)
            # --------------------------------------------------------------- #

            # ---------------------- Feature Extraction --------------------- #
            if self.collectFeatures:
                # Confirm assumptions made about EDA feature extraction
                assert dataFinger <= self.lastAnalyzedDataInd[channelIndex], f"{dataFinger}; {self.lastAnalyzedDataInd[channelIndex]}"  # We are NOT analyzing data in the buffer region. self.startTimePointerSCL CAN be in the buffer region.

                # Initialize the new raw features and times.
                newFeatureTimes, newRawFeatures = [], []

                # Extract features across the dataset
                while self.lastAnalyzedDataInd[channelIndex] < len(self.timePoints):
                    featureTime = self.timePoints[self.lastAnalyzedDataInd[channelIndex]]

                    # Find the start window pointer and get the data.
                    self.startFeatureTimePointer_Tonic[channelIndex] = self.findStartFeatureWindow(self.startFeatureTimePointer_Tonic[channelIndex], featureTime, self.featureTimeWindow_Tonic)
                    self.startFeatureTimePointer_Phasic[channelIndex] = self.findStartFeatureWindow(self.startFeatureTimePointer_Phasic[channelIndex], featureTime, self.featureTimeWindow_Phasic)
                    # Compile the well-formed data in the feature interval.
                    intervalTimesTonic, intervalTonicData = self.compileBatchData(filteredTime, tonicComponent, goodIndicesMask, startFilterPointer, self.startFeatureTimePointer_Tonic[channelIndex], channelIndex)
                    intervalTimesPhasic, intervalPhasicData = self.compileBatchData(filteredTime, phasicComponent, goodIndicesMask, startFilterPointer, self.startFeatureTimePointer_Phasic[channelIndex], channelIndex)

                    # Only extract features if enough information is provided.
                    if self.minPointsPerBatchTonic < len(intervalTimesTonic) and self.minPointsPerBatchPhasic < len(intervalTimesPhasic):
                        # Calculate the features in this window.
                        finalFeatures = self.extractFinalFeatures(intervalTimesTonic, intervalTonicData)
                        finalFeatures.extend(self.extractPhasicFeatures(intervalTimesPhasic, intervalPhasicData))

                        # Keep track of the new features.
                        newRawFeatures.append(finalFeatures)
                        newFeatureTimes.append(featureTime)

                    # Keep track of which data has been analyzed 
                    self.lastAnalyzedDataInd[channelIndex] += int(self.samplingFreq * self.secondsPerFeature)

                # Compile the new raw features into a smoothened (averaged) feature.
                self.readData.compileContinuousFeatures(newFeatureTimes, newRawFeatures, self.rawFeatureTimes[channelIndex], self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex], self.featureAverageWindow)

            # -------------------------------------------------------------- #  

            # ------------------- Plot Biolectric Signals ------------------ #
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
                    self.plottingMethods.featureDataPlots[channelIndex].set_data(self.rawFeatureTimes[channelIndex], np.array(self.compiledFeatures[channelIndex])[:, 19])
                    self.plottingMethods.featureDataPlotAxes[channelIndex].legend(["Hjorth Activity"], loc="upper left")

            # -------------------------------------------------------------- #   

    def filterData(self, timePoints, data, removePoints=False):
        # Filter the data: LPF and moving average (Savgol) filter
        filteredData = self.filteringMethods.bandPassFilter.butterFilter(data, self.cutOffFreq[1], self.samplingFreq, order=1, filterType='low')
        goodIndicesMask = np.full_like(data, True, dtype=bool)
        filteredTime = timePoints.copy()

        return filteredTime, filteredData, goodIndicesMask

    def splitPhasicTonic(self, data):
        # Isolate the tonic component (baseline) of the EDA
        tonicComponent = self.filteringMethods.bandPassFilter.butterFilter(data, self.tonicFrequencyCutoff, self.samplingFreq, order=1, filterType='low')
        # Extract the phasic component (peaks) of the EDA
        phasicComponent = tonicComponent - data

        return tonicComponent, phasicComponent

    def findStartFeatureWindow(self, timePointer, currentTime, timeWindow):
        # Loop through until you find the first time in the window 
        while self.timePoints[timePointer] < currentTime - timeWindow:
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

    def extractFinalFeatures(self, timePoints, data):

        # ----------------------- Data Preprocessing ----------------------- #

        # Normalize the data
        standardized_data = self.universalMethods.standardizeData(data)
        if all(standardized_data == 0):
            return [0 for _ in range(9)]

        # Calculate the derivatives
        firstDerivative = np.gradient(standardized_data, timePoints)

        # ----------------------- Features from Data ----------------------- #

        # General Shape Parameters
        signalRange = max(standardized_data) - min(standardized_data)
        standardDeviation = np.std(data, ddof=1)
        mean = np.mean(data)

        # -------------------- Features from Derivatives ------------------- #

        # First derivative features
        firstDerivativeStdDev = np.std(firstDerivative, ddof=1)
        firstDerivativeMean = np.mean(firstDerivative)

        # ------------------- Feature Extraction: Fractal ------------------- #

        # Fractal analysis
        petrosian_fd = antropy.petrosian_fd(standardized_data, axis=-1)  # Amplitude-invariant. Averages 25 μs.
        higuchi_fd = antropy.higuchi_fd(standardized_data, kmax=6)  # Amplitude-invariant. Same if standardized or not
        DFA = antropy.detrended_fluctuation(standardized_data)  # Amplitude-invariant. Same if standardized or not
        katz_fd = antropy.katz_fd(standardized_data, axis=-1)  # Amplitude-invariant. Same if standardized or not

        # ----------------------- Organize Features ------------------------ #

        finalFeatures = []
        # Add peak shape parameters
        finalFeatures.extend([mean, standardDeviation, signalRange])
        # Add derivative features
        finalFeatures.extend([firstDerivativeMean, firstDerivativeStdDev])
        # Feature Extraction: Fractal
        finalFeatures.extend([petrosian_fd, higuchi_fd, DFA, katz_fd])

        return finalFeatures

    def extractPhasicFeatures(self, timePoints, data):

        # ----------------------- Data Preprocessing ----------------------- #

        # Normalize the data
        standardized_data = self.universalMethods.standardizeData(data)
        if all(standardized_data == 0):
            return [0 for _ in range(6)]

        # Calculate the power spectral density (PSD) of the signal. USE NORMALIZED DATA
        powerSpectrumDensityFreqs, powerSpectrumDensity, powerSpectrumDensityNormalized = self.universalMethods.calculatePSD(standardized_data, self.samplingFreq)
        # powerSpectrumDensityNormalized is amplitude-invariant to the original data UNLIKE powerSpectrumDensity.
        # Note: we are removing the DC component from the power spectrum density.

        # ------------------- Feature Extraction: Hjorth ------------------- #

        # Calculate the hjorth parameters
        hjorthActivity, hjorthMobility, hjorthComplexity, firstDerivVariance, secondDerivVariance = self.universalMethods.hjorthParameters(timePoints, data, firstDeriv=None, secondDeriv=None, standardized_data=standardized_data)

        # ------------------- Feature Extraction: Entropy ------------------ #

        # Entropy calculation
        spectral_entropy = self.universalMethods.spectral_entropy(powerSpectrumDensityNormalized, normalizePSD=False)  # Spectral entropy: amplitude-independent if using normalized PSD
        perm_entropy = antropy.perm_entropy(standardized_data, order=3, delay=1, normalize=True)  # Permutation entropy: same if standardized or not
        # sample_entropy = antropy.sample_entropy(data, order=2, metric="chebyshev")       # Sample entropy
        # app_entropy = antropy.app_entropy(data, order=2, metric="chebyshev")             # Approximate sample entropy

        # ------------------- Feature Extraction: Fractals ------------------ #

        finalFeatures = []
        # Feature Extraction: Hjorth
        finalFeatures.extend([hjorthActivity, hjorthMobility, hjorthComplexity, firstDerivVariance])
        # Feature Extraction: Entropy
        finalFeatures.extend([spectral_entropy, perm_entropy])

        return finalFeatures
