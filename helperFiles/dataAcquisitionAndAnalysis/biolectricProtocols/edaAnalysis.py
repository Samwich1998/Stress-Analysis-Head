import numpy as np
import antropy
import scipy

# Import Files
from .globalProtocol import globalProtocol


# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------#

class edaProtocol(globalProtocol):

    def __init__(self, numPointsPerBatch=3000, moveDataFinger=10, numChannels=2, plottingClass=None, readData=None):
        # Feature collection parameters
        self.secondsPerFeature = 1  # The duration of time that passes between each feature.
        self.featureTimeWindow_Tonic = 60  # The duration of time that each feature considers.
        self.featureTimeWindow_Phasic = 15  # The duration of time that each feature considers.
        self.startFeatureTimePointer_Phasic = []  # The start pointer of the feature window interval.
        self.startFeatureTimePointer_Tonic = []  # The start pointer of the feature window interval.
        self.minPointsPerBatchPhasic = 0  # The minimum number of points required to extract features.
        self.minPointsPerBatchTonic = 0  # The minimum number of points required to extract features.

        # Filter Parameters
        self.tonicFrequencyCutoff = 0.05  # Maximum tonic component frequency.
        self.dataPointBuffer = 5000  # A Prepended Buffer in the Filtered Data that Represents BAD Filtering; Units: Points
        self.cutOffFreq = [None, 15]  # Filter cutoff frequencies: [HPF, LPF].

        # Initialize common model class
        super().__init__("eda", numPointsPerBatch, moveDataFinger, numChannels, plottingClass, readData)

    def resetAnalysisVariables(self):
        # General parameters
        self.startFeatureTimePointer_Phasic = [0 for _ in range(self.numChannels)]  # The start pointer of the feature window interval.
        self.startFeatureTimePointer_Tonic = [0 for _ in range(self.numChannels)]  # The start pointer of the feature window interval.

    def checkParams(self):
        assert self.featureTimeWindow_Tonic < self.dataPointBuffer, "The buffer does not include enough points for the feature window"
        assert self.featureTimeWindow_Phasic < self.dataPointBuffer, "The buffer does not include enough points for the feature window"

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
            dataBuffer = np.array(self.data[1][channelIndex][startFilterPointer:dataFinger + self.numPointsPerBatch])
            timePoints = np.array(self.data[0][startFilterPointer:dataFinger + self.numPointsPerBatch])

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
                assert dataFinger <= self.lastAnalyzedDataInd[channelIndex], str(dataFinger) + "; " + str(
                    self.lastAnalyzedDataInd[channelIndex])  # We are NOT analyzing data in the buffer region. self.startTimePointerSCL CAN be in the buffer region.

                # Extract features across the dataset
                while self.lastAnalyzedDataInd[channelIndex] < len(self.data[0]):
                    featureTime = self.data[0][self.lastAnalyzedDataInd[channelIndex]]

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
                        self.readData.averageFeatures([featureTime], [finalFeatures], self.featureTimes[channelIndex],
                                                      self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex], self.featureAverageWindow)

                    # Keep track of which data has been analyzed 
                    self.lastAnalyzedDataInd[channelIndex] += int(self.samplingFreq * self.secondsPerFeature)
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
                    self.plottingMethods.featureDataPlots[channelIndex].set_data(self.featureTimes[channelIndex], np.array(self.compiledFeatures[channelIndex])[:, 19])
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

    @staticmethod
    def extractFinalFeatures(timePoints, data):

        # ----------------------- Data Preprocessing ----------------------- #

        standardDeviation = np.std(data, ddof=1)

        # Normalize the data
        if standardDeviation <= 1e-10:
            standardized_data = (data - np.mean(data))
        else:
            standardized_data = (data - np.mean(data)) / standardDeviation

        # Calculate the derivatives
        firstDerivative = np.gradient(standardized_data, timePoints)
        secondDerivative = np.gradient(firstDerivative, timePoints)

        # Get the baseline data
        baselineX = timePoints - timePoints[0]
        baselineY = data - data[0]

        # ----------------------- Features from Data ----------------------- #

        # General Shape Parameters
        mean = np.mean(data)
        if standardDeviation <= 1e-10:
            skewness = 0
            kurtosis = 0
        else:
            skewness = scipy.stats.skew(data, bias=False)
            kurtosis = scipy.stats.kurtosis(data, fisher=True, bias=False)

        # Other Parameters
        signalRange = max(data) - min(data)
        signalArea = scipy.integrate.simpson(data, timePoints) / (baselineX[-1] - baselineX[0])

        # ------------------------------------------------------------------ #  
        # -------------------- Features from Derivatives ------------------- #

        # First derivative features
        firstDerivativeMean = np.mean(firstDerivative)
        firstDerivativeStdDev = np.std(firstDerivative, ddof=1)
        firstDerivativePower = scipy.integrate.simpson(firstDerivative ** 2, timePoints) / (baselineX[-1] - baselineX[0])

        # Second derivative features
        secondDerivativeMean = np.mean(secondDerivative)
        secondDerivativeStdDev = np.std(secondDerivative, ddof=1)
        secondDerivativePower = scipy.integrate.simpson(secondDerivative ** 2, timePoints) / (baselineX[-1] - baselineX[0])

        # ----------------------- Organize Features ------------------------ #

        finalFeatures = []
        # Add peak shape parameters
        finalFeatures.extend([mean, standardDeviation, skewness, kurtosis])
        finalFeatures.extend([signalRange, signalArea])

        # Add derivative features
        finalFeatures.extend([firstDerivativeMean, firstDerivativeStdDev, firstDerivativePower])
        finalFeatures.extend([secondDerivativeMean, secondDerivativeStdDev, secondDerivativePower])

        return finalFeatures

    def extractPhasicFeatures(self, timePoints, data):

        # ----------------------- Data Preprocessing ----------------------- #

        # Normalize the data
        standardized_data = self.universalMethods.standardizeData(data)
        if all(standardized_data == 0):
            return [0 for _ in range(23)]

        # Calculate the power spectral density (PSD) of the signal. USE NORMALIZED DATA
        powerSpectrumDensityFreqs, powerSpectrumDensity, powerSpectrumDensityNormalized = self.universalMethods.calculatePSD(standardized_data, self.samplingFreq, int(self.samplingFreq * 10))
        # powerSpectrumDensityNormalized is amplitude-invariant to the original data UNLIKE powerSpectrumDensity.
        # Note: we are removing the DC component from the power spectrum density.

        # ------------------- Feature Extraction: MNE ------------------- #

        decorr_time, higuchi_fd, _, line_length, ptp_amp = self.mneInterface.extractFeatures(standardized_data, self.samplingFreq)

        # ------------------- Feature Extraction: Hjorth ------------------- #

        # Calculate the hjorth parameters
        hjorthActivity, hjorthMobility, hjorthComplexity, firstDerivVariance, secondDerivVariance = self.universalMethods.hjorthParameters(timePoints, data, firstDeriv=None, secondDeriv=None, standardized_data=standardized_data)
        hjorthActivityPSD, hjorthMobilityPSD, hjorthComplexityPSD, firstDerivVariancePSD, secondDerivVariancePSD = self.universalMethods.hjorthParameters(powerSpectrumDensityFreqs, powerSpectrumDensityNormalized, firstDeriv=None, secondDeriv=None, standardized_data=powerSpectrumDensityNormalized)

        # ------------------- Feature Extraction: Entropy ------------------ #

        # Entropy calculation
        spectral_entropy = self.universalMethods.spectral_entropy(powerSpectrumDensityNormalized, normalizePSD=False)  # Spectral entropy: amplitude-independent if using normalized PSD
        perm_entropy = antropy.perm_entropy(standardized_data, order=3, delay=1, normalize=True)  # Permutation entropy: same if standardized or not
        # sample_entropy = antropy.sample_entropy(data, order=2, metric="chebyshev")       # Sample entropy
        # app_entropy = antropy.app_entropy(data, order=2, metric="chebyshev")             # Approximate sample entropy

        # ------------------- Feature Extraction: Fractals ------------------ #

        # Fractal analysis
        DFA = antropy.detrended_fluctuation(data)  # Numba. Same if standardized or not
        LZC = antropy.lziv_complexity(data)

        # -------------------- Feature Extraction: Other ------------------- #

        # Number of zero-crossings
        num_zerocross = antropy.num_zerocross(data)
        meanFrequency = np.sum(powerSpectrumDensityFreqs * powerSpectrumDensityNormalized)

        # ------------------------------------------------------------------ #

        finalFeatures = []
        # Feature Extraction: MNE
        finalFeatures.extend([decorr_time, higuchi_fd, line_length, ptp_amp])
        # Feature Extraction: Hjorth
        finalFeatures.extend([hjorthActivity, hjorthMobility, hjorthComplexity, firstDerivVariance, secondDerivVariance])
        finalFeatures.extend([hjorthActivityPSD, hjorthMobilityPSD, hjorthComplexityPSD, firstDerivVariancePSD, secondDerivVariancePSD])
        # Feature Extraction: Entropy
        finalFeatures.extend([spectral_entropy, perm_entropy])
        # Feature Extraction: Fractal
        finalFeatures.extend([DFA, LZC])
        # Feature Extraction: Other
        finalFeatures.extend([num_zerocross, meanFrequency])

        return finalFeatures
