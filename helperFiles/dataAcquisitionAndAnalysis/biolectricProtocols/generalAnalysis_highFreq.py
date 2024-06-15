import numpy as np
import antropy

# Import Files
from .globalProtocol import globalProtocol


class generalProtocol_highFreq(globalProtocol):

    def __init__(self, numPointsPerBatch=3000, moveDataFinger=10, channelIndices=(), plottingClass=None, readData=None):
        super().__init__("general_hf", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)
        # Feature collection parameters
        self.featureTimeWindow = self.featureTimeWindow_highFreq  # The duration of time that each feature considers
        self.startFeatureTimePointer = []  # The start pointer of the feature window interval.
        self.minPointsPerBatch = 0  # The minimum number of points per batch.

        # Filter parameters.
        self.cutOffFreq = [None, None]  # Optimal LPF Cutoff in Literature is 6-8 or 20 Hz (Max 35 or 50); I Found 25 Hz was the Best, but can go to 15 if noisy (small amplitude cutoff)

        # High-pass filter parameters.
        self.stopband_edge = 1  # Common values for EEG are 1 Hz and 2 Hz. If you need to remove more noise, you can choose a higher stopband-edge frequency. If you need to preserve the signal more, you can choose a lower stopband-edge frequency.
        self.passband_ripple = 0.1  # Common values for EEG are 0.1 dB and 0.5 dB. If you need to remove more noise, you can choose a lower passband ripple. If you need to preserve the signal more, you can choose a higher passband ripple.
        self.stopband_attenuation = 60  # Common values for EEG are 40 dB and 60 dB. If you need to remove more noise, you can choose a higher stopband attenuation. If you need to preserve the signal more, you can choose a lower stopband attenuation.

        # Finalize the protocol parameters.
        self.resetAnalysisVariables()

    def resetAnalysisVariables(self):
        # General parameters
        self.startFeatureTimePointer = [0 for _ in range(self.numChannels)]    # The start pointer of the feature window interval.
        self.minPointsPerBatch = 0  # The minimum number of points per batch.

        # Finalize the protocol parameters.
        self.resetGlobalVariables()
        self.checkParams()

    def checkParams(self):
        self.checkGlobalParams()

        # Assert that the buffer is large enough for the feature window.
        assert self.featureTimeWindow < self.dataPointBuffer, "The buffer does not include enough points for the feature window"

    def setSamplingFrequencyParams(self):
        maxBufferSeconds = max(self.featureTimeWindow, 100)
        # Set Parameters
        self.lastAnalyzedDataInd[:] = int(self.samplingFreq * self.featureTimeWindow)
        self.minPointsPerBatch = int(self.samplingFreq * self.featureTimeWindow * 3 / 4)
        self.dataPointBuffer = max(self.dataPointBuffer, int(self.samplingFreq * maxBufferSeconds))  # cutOffFreq = 0.1, use 70 seconds; cutOffFreq = 0.01, use 400 seconds; cutOffFreq = 0.05, use 100 seconds

        # Set the sampling frequency for the MNE interface
        self.mneInterface.setSamplingFrequencyParams(self.samplingFreq)

    # ------------------------- Data Analysis Begins ----------------------- #

    def analyzeData(self, dataFinger):

        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):

            # ---------------------- Filter the Data ----------------------- #

            # Find the starting/ending points of the data to analyze
            startFilterPointer = max(dataFinger - self.dataPointBuffer, 0)
            dataBuffer = np.array(self.channelData[channelIndex][startFilterPointer:dataFinger + self.numPointsPerBatch])
            timePoints = np.array(self.timePoints[startFilterPointer:dataFinger + self.numPointsPerBatch])

            # Get the Sampling Frequency from the First Batch (If Not Given)
            if not self.samplingFreq:
                self.setSamplingFrequency(startFilterPointer)

            # Filter the data and remove bad indices
            filteredTime, filteredData, goodIndicesMask = self.filterData(timePoints, dataBuffer)

            # ---------------------- Feature Extraction -------------------- #

            if self.collectFeatures:
                # Initialize the new raw features and times.
                newFeatureTimes, newRawFeatures = [], []

                # Extract features across the dataset
                while self.lastAnalyzedDataInd[channelIndex] < len(self.timePoints):
                    featureTime = self.timePoints[self.lastAnalyzedDataInd[channelIndex]]

                    # Find the start window pointer.
                    self.startFeatureTimePointer[channelIndex] = self.findStartFeatureWindow(self.startFeatureTimePointer[channelIndex], featureTime, self.featureTimeWindow)
                    # Compile the good data in the feature interval.
                    intervalTimes, intervalData = self.compileBatchData(filteredTime, filteredData, goodIndicesMask, startFilterPointer, self.startFeatureTimePointer[channelIndex], channelIndex)

                    # Only extract features if enough information is provided.
                    if self.minPointsPerBatch < len(intervalTimes):
                        # Calculate and save the features in this window.
                        finalFeatures = self.extractFeatures(intervalTimes, intervalData)

                        # Keep track of the new features.
                        newRawFeatures.append(finalFeatures)
                        newFeatureTimes.append(featureTime)

                    # Keep track of which data has been analyzed 
                    self.lastAnalyzedDataInd[channelIndex] += int(self.samplingFreq * self.secondsPerFeature)

                # Compile the new raw features into a smoothened (averaged) feature.
                self.readData.compileContinuousFeatures(newFeatureTimes, newRawFeatures, self.rawFeatureTimes[channelIndex], self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex], self.featureAverageWindow)

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
                    self.plottingMethods.featureDataPlots[channelIndex].set_data(self.rawFeatureTimes[channelIndex], np.array(self.compiledFeatures[channelIndex])[:, 0])
                    self.plottingMethods.featureDataPlotAxes[channelIndex].legend(["Hjorth Activity"], loc="upper left")

            # -------------------------------------------------------------- #   

    def filterData(self, timePoints, data):
        # Filter the Data: Low pass Filter and Savgol Filter
        filteredData = self.filteringMethods.bandPassFilter.butterFilter(data, self.cutOffFreq[1], self.samplingFreq, order=3, filterType='low', fastFilt=True)
        filteredData = self.filteringMethods.bandPassFilter.high_pass_filter(filteredData, self.samplingFreq, self.cutOffFreq[0], self.stopband_edge, self.passband_ripple, self.stopband_attenuation, fastFilt=True)
        filteredTime = timePoints.copy()

        return filteredTime, filteredData, np.ones(len(filteredTime))

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

    # ---------------------------------------------------------------------- #
    # --------------------- Feature Extraction Methods --------------------- #

    def extractFeatures(self, timePoints, data):

        # ----------------------- Data Preprocessing ----------------------- #

        # Normalize the data
        standardized_data = self.universalMethods.standardizeData(data)
        if all(standardized_data == 0):
            return [0 for _ in range(27)]

        # Calculate the power spectral density (PSD) of the signal. USE NORMALIZED DATA
        powerSpectrumDensityFreqs, powerSpectrumDensity, powerSpectrumDensityNormalized = self.universalMethods.calculatePSD(standardized_data, self.samplingFreq)
        # powerSpectrumDensityNormalized is amplitude-invariant to the original data UNLIKE powerSpectrumDensity.
        # Note: we are removing the DC component from the power spectrum density.

        # ------------------- Feature Extraction: MNE ------------------- #

        higuchi_fd, katz_fd, ptp_amp = self.mneInterface.extractFeatures(standardized_data)

        # ------------------- Feature Extraction: Hjorth ------------------- #

        # Calculate the hjorth parameters
        hjorthActivity, hjorthMobility, hjorthComplexity, firstDerivVariance, secondDerivVariance = self.universalMethods.hjorthParameters(timePoints, data, firstDeriv=None, secondDeriv=None, standardized_data=standardized_data)
        hjorthActivityPSD, hjorthMobilityPSD, hjorthComplexityPSD, firstDerivVariancePSD, secondDerivVariancePSD = self.universalMethods.hjorthParameters(powerSpectrumDensityFreqs, powerSpectrumDensityNormalized, firstDeriv=None, secondDeriv=None, standardized_data=powerSpectrumDensityNormalized)

        # ------------------- Feature Extraction: Entropy ------------------ #

        # Entropy calculation
        spectral_entropy = self.universalMethods.spectral_entropy(powerSpectrumDensityNormalized, normalizePSD=False)  # Spectral entropy: amplitude-independent if using normalized PSD
        perm_entropy = antropy.perm_entropy(standardized_data, order=3, delay=1, normalize=True)  # Permutation entropy: same if standardized or not
        svd_entropy = antropy.svd_entropy(standardized_data, order=3, delay=1, normalize=True)  # Singular value decomposition entropy: same if standardized or not
        # sample_entropy = antropy.sample_entropy(data, order=2, metric="chebyshev")       # Sample entropy
        # app_entropy = antropy.app_entropy(data, order=2, metric="chebyshev")             # Approximate sample entropy

        # ------------------- Feature Extraction: Fractals ------------------ #

        # Fractal analysis
        petrosian_fd = antropy.petrosian_fd(standardized_data, axis=-1)  # Amplitude-invariant. Averages 25 μs.
        DFA = antropy.detrended_fluctuation(data)  # Numba. Same if standardized or not

        # -------------------- Feature Extraction: Other ------------------- #

        # Calculate the band wave powers
        deltaPower, thetaPower, alphaPower, betaPower, gammaPower = self.universalMethods.bandPower(powerSpectrumDensity, powerSpectrumDensityFreqs, bands=[(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100)], relative=True)
        muPower, beta1Power, beta2Power, beta3Power, smrPower = self.universalMethods.bandPower(powerSpectrumDensity, powerSpectrumDensityFreqs, bands=[(8, 13), (13, 16), (16, 20), (20, 28), (13, 15)], relative=True)
        # Calculate band wave power ratios
        engagementLevelEst = betaPower / (alphaPower + thetaPower)

        # ------------------------------------------------------------------ #

        finalFeatures = []
        # Feature Extraction: MNE
        finalFeatures.extend([higuchi_fd, katz_fd, ptp_amp])
        # Feature Extraction: Hjorth
        finalFeatures.extend([hjorthActivity, hjorthMobility, hjorthComplexity, firstDerivVariance])
        finalFeatures.extend([hjorthActivityPSD, hjorthMobilityPSD, hjorthComplexityPSD, firstDerivVariancePSD])
        # Feature Extraction: Entropy
        finalFeatures.extend([spectral_entropy, perm_entropy, svd_entropy])
        # Feature Extraction: Fractal
        finalFeatures.extend([petrosian_fd, DFA])
        # Feature Extraction: Other
        finalFeatures.extend([deltaPower, thetaPower, alphaPower, betaPower, gammaPower])
        finalFeatures.extend([muPower, beta1Power, beta2Power, beta3Power, smrPower])
        finalFeatures.extend([engagementLevelEst])

        return finalFeatures
