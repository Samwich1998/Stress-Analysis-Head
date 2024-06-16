# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import scipy
import numpy as np
import matplotlib.pyplot as plt
# Feature Extraction Modules
from scipy.signal import savgol_filter

# Import Files
from .globalProtocol import globalProtocol


# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------#

class ecgProtocol(globalProtocol):

    def __init__(self, numPointsPerBatch=3000, moveDataFinger=10, channelIndices=(), plottingClass=None, readData=None):
        # Feature collection parameters
        self.initializingTimeWindow = 5
        self.secondsPerFeature = 1  # The duration of time that passes between each feature.
        self.featureTimeWindow = 2  # The duration of time that each feature considers;

        # Filter parameters.
        self.cutOffFreq = [None, 150]  # Band pass filter frequencies.
        self.debug = False

        # Holder parameters.
        self.startFeatureTimePointer = None  # The start pointer of the feature window interval.
        self.minPointsPerBatch = None  # The minimum number of points required to analyze a batch of data.
        self.lastRPeak = None

        # Reset analysis variables
        super().__init__("ecg", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)
        self.resetAnalysisVariables()

    def resetAnalysisVariables(self):
        # General parameters 
        self.startFeatureTimePointer = [0 for _ in range(self.numChannels)]  # The start pointer of the feature window interval.

        # Holder parameters.
        self.minPointsPerBatch = None  # The minimum number of points required to analyze a batch of data.
        self.lastRPeak = None

    def checkParams(self):
        assert self.numChannels == 1, "The ECG protocol now only supports one channel of data due tp feature alignment issues."

    def setSamplingFrequencyParams(self):
        maxBufferSeconds = max(self.featureTimeWindow, int(self.samplingFreq * 0.1))
        # Set Parameters
        self.lastAnalyzedDataInd[:] = int(self.samplingFreq * self.featureTimeWindow)
        self.minPointsPerBatch = int(self.samplingFreq * self.featureTimeWindow * 3 / 4)
        self.dataPointBuffer = max(self.dataPointBuffer, int(self.samplingFreq * maxBufferSeconds))  # cutOffFreq = 0.1, use 70 seconds; cutOffFreq = 0.01, use 400 seconds; cutOffFreq = 0.05, use int(self.samplingFreq * 0.1) seconds

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

            # Get the Sampling Frequency from the First Batch (If Not Given)
            if not self.samplingFreq:
                self.setSamplingFrequency(startFilterPointer)

            # Filter the data and remove bad indices
            filteredTime, filteredData, goodIndicesMask = self.filterData(timePoints, dataBuffer, removePoints=True)
            # -------------------------------------------------------------- #

            # The first seconds of data are used to compute R peak derivative threshold.
            endIndex = int(self.initializingTimeWindow * self.samplingFreq)
            self.lastRPeak, self.RPeakThreshold = self.getRThreshold(filteredTime[:endIndex], filteredData[:endIndex])

            # Feature Time Pointer becomes index of the last R peak in the initialization period.
            self.startFeatureTimePointer[channelIndex] = int(self.lastRPeak * self.samplingFreq)

            # ---------------------- Feature Extraction --------------------- #
            if self.collectFeatures:
                # Initialize the new raw features and times.
                newFeatureTimes, newRawFeatures = [], []

                # Extract features across the dataset
                while self.lastAnalyzedDataInd[channelIndex] < len(self.timePoints):
                    # while self.lastAnalyzedDataInd[channelIndex] < 200*self.samplingFreq:
                    featureTime = self.timePoints[self.lastAnalyzedDataInd[channelIndex]]

                    # Find the start window pointer.
                    self.startFeatureTimePointer[channelIndex] = self.findStartFeatureWindow(self.startFeatureTimePointer[channelIndex], featureTime, self.featureTimeWindow)
                    # Compile the good data in the feature interval.
                    intervalTimes, intervalData = self.compileBatchData(filteredTime, filteredData, goodIndicesMask, startFilterPointer, self.startFeatureTimePointer[channelIndex], channelIndex)

                    # Only extract features if enough information is provided.
                    if self.minPointsPerBatch < len(intervalTimes):
                        # Calculate and save the features in this window.
                        self.lastRPeak, featureTimes, finalFeatures = self.extractFeatures(intervalTimes, intervalData, self.lastRPeak)

                        # Keep track of the new features.
                        newRawFeatures.extend(finalFeatures)
                        newFeatureTimes.extend(featureTimes)

                        # Feature Time Pointer becomes index of the last R peak found in the time window
                        self.startFeatureTimePointer[channelIndex] = int(self.lastRPeak)

                    # Keep track of which data has been analyzed 
                    self.lastAnalyzedDataInd[channelIndex] += int(self.samplingFreq * self.secondsPerFeature)

                # Compile the new raw features into a smoothened (averaged) feature.
                self.readData.compileContinuousFeatures(newFeatureTimes, newRawFeatures, self.rawFeatureTimes[channelIndex], self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex], self.featureAverageWindow)

            # -------------------------------------------------------------- #   
            if self.debug:
                features = ['QRSWaveLength', 'QLength', 'SLength', 'PreRWavePeakTime', 'PostRWavePeakTime',
                            'QRDer', 'RSDer', 'QRSWaveDer', 'QRSVariance', 'QRSWaveSkew', 'QRSWaveKurt',
                            'netDirectionQRS', 'QAreaBelow', 'SAreaBelow', 'ratioDirectionQRS', 'ratioQSArea']

                for i, feature in enumerate(features):
                    plt.scatter(self.rawFeatureTimes[channelIndex], [row[i] for row in self.compiledFeatures[channelIndex]], marker='o')
                    plt.title(f'{feature}, N={len(self.rawFeatureTimes[channelIndex])}')
                    plt.xlabel('Time (s)')
                    plt.show()

            # ------------------- Plot Biolectric Signals ------------------ #
            if self.plotStreamedData:
                # Format the raw data:
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
                    self.plottingMethods.featureDataPlotAxes[channelIndex].legend(["QRSWaveLength"], loc="upper left")

            # -------------------------------------------------------------- #   

    def filterData(self, timePoints, data, removePoints=False):
        # Find the bad points associated with motion artifacts
        if removePoints and self.cutOffFreq[0] is not None:
            motionIndices = np.logical_or(data < 0.1, data > 3.15)
            motionIndices_Broadened = scipy.signal.savgol_filter(motionIndices, max(3, int(self.samplingFreq * 10)), 1, mode='nearest', deriv=0)
            goodIndicesMask = motionIndices_Broadened < 0.01
        else:
            goodIndicesMask = np.full_like(data, True, dtype=bool)

        # Filtering the whole dataset
        filteredData = self.filteringMethods.bandPassFilter.butterFilter(data, self.cutOffFreq[1], self.samplingFreq, order=3, filterType='low', fastFilt=True)
        # Remove the bad points from the filtered data
        filteredTime = timePoints[goodIndicesMask]
        filteredData = filteredData[goodIndicesMask]

        return filteredTime, filteredData, goodIndicesMask

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

    def getRThreshold(self, timePoints, data):
        firstDer = savgol_filter(data, 9, 2, mode='nearest', delta=1 / self.samplingFreq, deriv=1)
        secondDer = savgol_filter(data, 9, 2, mode='nearest', delta=1 / self.samplingFreq, deriv=2)

        # make mask for datapoints above mean
        above_mean_indices = np.where(data > np.mean(data))[0]
        # for all of the datapoints above mean, find indices with largest first derivatives
        # This is found by thresholding for first derivative values greater than 0.66 of maximum first derivative
        close_to_max_derivative_indices = above_mean_indices[firstDer[above_mean_indices] / np.max(firstDer[above_mean_indices]) >= 0.66]
        maxDerIndices = [close_to_max_derivative_indices[0]]

        # Clean up derivative indicies found by thresholding to only include R peaks
        maxDer = [firstDer[close_to_max_derivative_indices[0]]]
        for i in range(len(close_to_max_derivative_indices) - 1):
            # 1. Make sure the indices are at least half a second apart
            if timePoints[close_to_max_derivative_indices[i + 1]] - timePoints[maxDerIndices[-1]] > 0.5:
                maxDerIndices.append(close_to_max_derivative_indices[i + 1])
                maxDer.append(firstDer[close_to_max_derivative_indices[i + 1]])

        # 2. Take local maximum around each index found to ensure correct index of R peak
        peakIndices = [np.argmax(data[i:i + int(self.samplingFreq * 0.25)]) + i for i in maxDerIndices]

        if self.debug:
            fig, ax1 = plt.subplots()

            color = 'tab:green'
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('ECG', color=color)
            ax1.plot(timePoints, data, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim(bottom=np.min(data), top=np.max(data))

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('ECG First Derivative', color=color)  # we already handled the x-label with ax1
            ax2.plot(timePoints, firstDer, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(bottom=np.min(firstDer), top=np.max(firstDer))

            fig.tight_layout()  # otherwise the right y-label is slightly clipped

            ax1.vlines([timePoints[i] for i in close_to_max_derivative_indices], np.min(data), np.max(data), color='black', label=f'high derivative detected, n={len(close_to_max_derivative_indices)}')
            ax1.vlines([timePoints[i] for i in peakIndices], np.min(data), np.max(data), color='red', linestyles='--', label=f'peaks detected, n={len(maxDerIndices)}')

            ax1.set_title(f'Baseline R Peaks, N={len(peakIndices)}, R Threshold = {np.max(firstDer[above_mean_indices]) * 0.66}')
            plt.show()

        # RPeakThreshold is set to 0.66 of mean of all R peaks found in the time window
        return timePoints[peakIndices[-1]], np.mean(maxDer) * 0.66

    def updateRThreshold(self, peak, peakDer):
        self.RPeakThreshold = self.RPeakThreshold + 0.1 * (peakDer - self.RPeakThreshold)

    def PQRSTDetection(self, timePoints, data, firstDer, secondDer):

        # We are searching through 0.25 seconds before R peak to 0.5 seconds after R peak
        R = timePoints[int(self.samplingFreq * 0.25)]

        # Finding Q, S, and QRS Wave
        # 1. binary search on second derivative to estimate Q and S
        estimatedQIndex = self.universalMethods.findNearbyMaximum(secondDer, int(self.samplingFreq * 0.25), binarySearchWindow=-5, maxPointsSearch=int(self.samplingFreq * 0.1))
        estimatedSIndex = self.universalMethods.findNearbyMaximum(secondDer, int(self.samplingFreq * 0.25), binarySearchWindow=5, maxPointsSearch=int(self.samplingFreq * 0.1))

        # 2. Taking local minimum around each index found to ensure correct index of Q and S
        QIndex = self.universalMethods.findNearbyMinimum(data, estimatedQIndex, binarySearchWindow=-5, maxPointsSearch=int(self.samplingFreq * 0.02))
        SIndex = self.universalMethods.findNearbyMinimum(data, estimatedSIndex, binarySearchWindow=5, maxPointsSearch=int(self.samplingFreq * 0.02))

        # 3. Using Q and S Index to find bounds of QRS Wave
        QRS0EstimatedIndex = self.universalMethods.findNearbyMinimum(firstDer, QIndex - 1, binarySearchWindow=-10, maxPointsSearch=int(self.samplingFreq * 0.05))
        QRS0Index = self.universalMethods.findNearbyMinimum(secondDer, QRS0EstimatedIndex, binarySearchWindow=-10, maxPointsSearch=int(self.samplingFreq * 0.025))

        QRS1EstimatedIndex = self.universalMethods.findNearbyMaximum(firstDer, SIndex + 1, binarySearchWindow=10, maxPointsSearch=int(self.samplingFreq * 0.05))
        QRS1Index = self.universalMethods.findNearbyMaximum(secondDer, QRS1EstimatedIndex, binarySearchWindow=10, maxPointsSearch=int(self.samplingFreq * 0.025))
        QRSWave = [timePoints[QRS0Index], timePoints[QRS1Index]]

        # 4. if R peak does not deviate from baseline significantly, do not consider
        # height of R peak must be at least 2x difference between baseline before/after R peak
        if 0.5 * (data[int(self.samplingFreq * 0.25)] - data[QRS0Index]) < abs(data[QRS1Index] - data[QRS0Index]):
            return None

        return timePoints[QIndex], R, timePoints[SIndex], QRSWave

    '''
    def PQRSTDetection_DEBUG(self, timePoints, data, firstDer, secondDer):
        R = timePoints[int(self.samplingFreq *0.25)]
        
        # binary search on first deriv?
        QIndex = self.universalMethods.findNearbyMaximum(secondDer, int(self.samplingFreq *0.25), binarySearchWindow = -10, maxPointsSearch = int(self.samplingFreq * 0.1))
        SIndex = self.universalMethods.findNearbyMaximum(secondDer, int(self.samplingFreq *0.25), binarySearchWindow = 10, maxPointsSearch = int(self.samplingFreq * 0.1))
        
        QRS0Index = self.universalMethods.findNearbyMinimum(secondDer, QIndex - 1, binarySearchWindow = -10, maxPointsSearch = int(self.samplingFreq * 0.05))
        QRS1Index = self.universalMethods.findNearbyMinimum(secondDer, SIndex + 1, binarySearchWindow = 10, maxPointsSearch = int(self.samplingFreq * 0.05))
        # QRS1Index = np.where(firstDer[SIndex + 1:] < self.RPeakThreshold)[0][0] + SIndex + 1
        QRSWave = [timePoints[QRS0Index], timePoints[QRS1Index]]
        # smoothedData = scipy.signal.savgol_filter(data, window_length=25, polyorder=3)
        PIndex = np.argmax(data[max(QRS0Index - int(self.samplingFreq * 0.25), 0):QRS0Index]) + max(QRS0Index - self.samplingFreq * 0.25, 0)
        P0Index = self.universalMethods.findNearbyMaximum(secondDer, np.argmax(firstDer[:PIndex]), binarySearchWindow = -10, maxPointsSearch = int(self.samplingFreq * 0.1))
        
        return timePoints[PIndex], timePoints[QIndex], R, timePoints[SIndex], QRSWave, timePoints[P0Index]
    '''

    def extractPeakFeatures(self, timePoints, data, firstDer, secondDer, Q, R, S, QRSWave):
        # TODO: T not done, P needs improvement
        finalFeatures = []

        # PIndex = np.where(timePoints == P)[0][0] # P NOT IMPLEMENTED
        QIndex = np.where(timePoints == Q)[0][0]
        RIndex = np.where(timePoints == R)[0][0]
        SIndex = np.where(timePoints == S)[0][0]
        QRS0Index = np.where(timePoints == QRSWave[0])[0][0]
        QRS1Index = np.where(timePoints == QRSWave[1])[0][0]
        # TIndex = np.where(timePoints == T)[0][0] # T NOT IMPLEMENTED

        # Wave Time Durations
        QRSWaveLength = QRSWave[1] - QRSWave[0]
        QLength = Q - QRSWave[0]
        SLength = QRSWave[1] - S

        PreRWavePeakTime = R - QRSWave[0]
        PostRWavePeakTime = QRSWave[1] - R
        PRbaseline = np.mean(data[:QRS0Index])

        finalFeatures.extend([QRSWaveLength, QLength, SLength, PreRWavePeakTime, PostRWavePeakTime])
        # QRS Wave Feature Extraction

        if (RIndex - QIndex <= 0) or (SIndex - RIndex <= 0):
            return None

        QRDer = np.mean(firstDer[QIndex:RIndex])
        RSDer = np.mean(firstDer[RIndex:SIndex])
        QRSWaveDer = np.mean(firstDer[QRS0Index:QRS1Index])

        QRSWaveData = data[QRS0Index:QRS1Index]
        QRSVariance = np.var(QRSWaveData) / (data[RIndex] - PRbaseline)
        QRSWaveSkew = scipy.stats.skew(QRSWaveData)
        QRSWaveKurt = scipy.stats.kurtosis(QRSWaveData)

        # net direction is amplitude specific but used in literature
        netDirectionQRS = np.trapz([pt - PRbaseline for pt in QRSWaveData])

        QRSWaveData_above_baseline = np.where(QRSWaveData > PRbaseline)[0]
        QRSWaveData_below_baseline = np.where(QRSWaveData < PRbaseline)[0]
        QData_below_baseline = np.where(data[QRS0Index:RIndex] < QRS0Index)[0]
        SData_below_baseline = np.where(data[RIndex:QRS1Index] < QRS1Index)[0]

        # divided by height of peak to normalize
        areaAbove = np.trapz(QRSWaveData_above_baseline - PRbaseline) / (data[RIndex] - PRbaseline)
        areaBelow = np.trapz(QRSWaveData_below_baseline - PRbaseline) / (data[RIndex] - PRbaseline)

        QAreaBelow = np.trapz(QData_below_baseline - PRbaseline) / (data[RIndex] - PRbaseline)
        SAreaBelow = np.trapz(SData_below_baseline - PRbaseline) / (data[RIndex] - PRbaseline)

        if areaBelow == 0 or QAreaBelow == 0 or SAreaBelow == 0:
            return None
        else:
            ratioDirectionQRS = np.abs(areaAbove / areaBelow)
            ratioQSArea = np.abs(QAreaBelow / SAreaBelow)

        finalFeatures.extend([QRDer, RSDer, QRSWaveDer, QRSVariance, QRSWaveSkew, QRSWaveKurt,
                              netDirectionQRS])

        finalFeatures.extend([QAreaBelow, SAreaBelow, ratioDirectionQRS, ratioQSArea])

        return finalFeatures

    def extractFeatures(self, timePoints, data, lastPeak):

        # ----------------------- Data Preprocessing ----------------------- #

        # find R
        firstDer = savgol_filter(data, 9, 2, mode='nearest', delta=1 / self.samplingFreq, deriv=1)
        secondDer = savgol_filter(data, 9, 2, mode='nearest', delta=1 / self.samplingFreq, deriv=2)

        # above_mean_indices = np.where(data > np.mean(data))[0]
        # close_to_max_derivative_indices = above_mean_indices[firstDer[above_mean_indices] >= self.RPeakThreshold]    
        close_to_max_derivative_indices = np.where(firstDer > self.RPeakThreshold)[0]
        firstIndex = None
        for i in range(len(close_to_max_derivative_indices)):
            if timePoints[close_to_max_derivative_indices[i]] >= lastPeak + 0.5 \
                    and len(timePoints) > close_to_max_derivative_indices[i] + int(self.samplingFreq * 0.5) and int(self.samplingFreq * 0.25) < close_to_max_derivative_indices[i]:
                firstIndex = i
                maxDerIndices = [close_to_max_derivative_indices[i]]
                break

        if firstIndex == None:
            self.updateRThreshold(0, 0)
            if self.debug:
                print('no full peaks in time window', close_to_max_derivative_indices)

                fig, ax1 = plt.subplots()

                color = 'tab:green'
                ax1.set_xlabel('time (s)')
                ax1.set_ylabel('ECG', color=color)
                ax1.plot(timePoints, data, color=color)
                for e in close_to_max_derivative_indices:
                    ax1.axvline(timePoints[e], color='black')
                ax1.axvspan(lastPeak, lastPeak + 0.5, color='red')
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_ylim(bottom=np.min(data), top=np.max(data))

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:blue'
                ax2.set_ylabel('ECG First Derivative', color=color)  # we already handled the x-label with ax1
                ax2.plot(timePoints, firstDer, color=color)
                ax2.axhline(self.RPeakThreshold, color='black')
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(bottom=np.min(firstDer), top=np.max(firstDer))

                fig.tight_layout()  # otherwise the right y-label is slightly clipped

                plt.title('No Peaks?')
                plt.show()
            return lastPeak + 0.5, [], []

        for i in range(len(close_to_max_derivative_indices) - 1):
            if timePoints[close_to_max_derivative_indices[i + 1]] - timePoints[maxDerIndices[-1]] > 0.5 \
                    and len(timePoints) > close_to_max_derivative_indices[i + 1] + int(self.samplingFreq * 0.5) and int(self.samplingFreq * 0.25) < close_to_max_derivative_indices[i + 1]:
                if min(firstDer[close_to_max_derivative_indices[i + 1]:close_to_max_derivative_indices[i + 1] + int(self.samplingFreq * 0.5)]) < -0.75 * self.RPeakThreshold:
                    maxDerIndices.append(close_to_max_derivative_indices[i + 1])

        peakIndices = [np.argmax(data[i:i + int(self.samplingFreq * 0.1)]) + i for i in maxDerIndices]

        # ------------------------------------------------------------------ #
        finalFeatures = []
        featureTimes = []

        for peak in peakIndices:  # look around R to find P, Q, S, T
            PQRST = self.PQRSTDetection(timePoints[peak - int(self.samplingFreq * 0.25): peak + int(self.samplingFreq * 0.5)], data[peak - int(self.samplingFreq * 0.25): peak + int(self.samplingFreq * 0.5)], \
                                        firstDer[peak - int(self.samplingFreq * 0.25): peak + int(self.samplingFreq * 0.5)], secondDer[peak - int(self.samplingFreq * 0.25): peak + int(self.samplingFreq * 0.5)])
            if PQRST != None:
                Q, R, S, QRSWave = PQRST
                features = self.extractPeakFeatures(timePoints, data, firstDer, secondDer, Q, R, S, QRSWave)
                if features != None:
                    finalFeatures.append(features)
                    featureTimes.append(timePoints[peak])
                else:
                    if self.debug:
                        plt.plot(timePoints[peak - int(self.samplingFreq * 0.25): peak + int(self.samplingFreq * 0.5)], data[peak - int(self.samplingFreq * 0.25): peak + int(self.samplingFreq * 0.5)])
                        plt.axvline(timePoints[peak], color='black')
                        plt.axvline(Q, color='red', label='Q')
                        plt.axvline(R, color='black', label='R')
                        plt.axvline(S, color='green', label='S')
                        plt.axvspan(QRSWave[0], QRSWave[1], color='grey', label='QRS')
                        plt.title('No Features Recorded')
                        plt.show()
                        self.updateRThreshold(timePoints[peak], firstDer[peak])

            elif self.debug:
                plt.plot(timePoints[peak - int(self.samplingFreq * 0.25): peak + int(self.samplingFreq * 0.5)], data[peak - int(self.samplingFreq * 0.25): peak + int(self.samplingFreq * 0.5)])
                plt.axvline(timePoints[peak], color='black')
                plt.title('Invalid Peak?')
                plt.show()
                self.updateRThreshold(timePoints[peak], firstDer[peak])

            if self.debug:
                # PLOTTING DATA 0.25 SEC AROUND AN R PEAK
                if PQRST is not None and features is not None:
                    peak = np.where(timePoints == featureTimes[-1])[0][0]
                    plt.plot(timePoints[peak - int(self.samplingFreq * 0.25): peak + int(self.samplingFreq * 0.5)], data[peak - int(self.samplingFreq * 0.25): peak + int(self.samplingFreq * 0.5)], label='ECG Signal')
                    plt.axvline(Q, color='red', label='Q')
                    plt.axvline(R, color='black', label='R')
                    plt.axvline(S, color='green', label='S')
                    plt.axvspan(QRSWave[0], QRSWave[1], color='grey', label='QRS')
                    plt.legend()
                    plt.show()
        assert len(featureTimes) == len(finalFeatures)
        return timePoints[peakIndices[-1]], featureTimes, finalFeatures
