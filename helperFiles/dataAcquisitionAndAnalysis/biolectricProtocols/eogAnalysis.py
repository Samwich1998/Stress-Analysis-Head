# Basic Modules
import scipy
import numpy as np
# Plotting
import matplotlib
import matplotlib.pyplot as plt

# Import Files
from .globalProtocol import globalProtocol


class eogProtocol(globalProtocol):

    def __init__(self, numPointsPerBatch=3000, moveDataFinger=10, channelIndices=2, plottingClass=None, readData=None, voltageRange=(0, 3.3)):
        # Initialize the global protocol.
        super().__init__("eog", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)
        # Filter parameters.
        self.cutOffFreq = [.5, 15]  # Optimal LPF Cutoff in Literature is 6-8 or 20 Hz (Max 35 or 50); I Found 20 Hz was the Best, but can go to 15 if noisy (small amplitude cutoff)

        # High-pass filter parameters.
        self.stopband_edge = 10  # Common values for EEG are 1 Hz and 2 Hz. If you need to remove more noise, you can choose a higher stopband-edge frequency. If you need to preserve the signal more, you can choose a lower stopband-edge frequency.
        self.passband_ripple = 0.1  # Common values for EEG are 0.1 dB and 0.5 dB. If you need to remove more noise, you can choose a lower passband ripple. If you need to preserve the signal more, you can choose a higher passband ripple.
        self.stopband_attenuation = 20  # Common values for EEG are 40 dB and 60 dB. If you need to remove more noise, you can choose a higher stopband attenuation. If you need to preserve the signal more, you can choose a lower stopband attenuation.

        # Blink Parameters
        self.voltageRange = voltageRange  # The Voltage Range of the System; Units: Volts
        self.minPeakHeight_Volts = 0.05  # The Minimum Peak Height in Volts; Removes Small Oscillations
        self.debugBlinkDetection = False  # Debugging for Blink Detection

        # Eye Angle Determination Parameters
        self.voltagePositionBuffer = 100  # A Prepended Buffer to Find the Current Average Voltage; Units: Points
        self.minVoltageMovement = 0.05  # The Minimum Voltage Change Required to Register an Eye Movement; Units: Volts
        self.predictEyeAngleGap = 5  # The Number of Points Between Each Eye Gaze Prediction; Will Back-calculate if moveDataFinger > predictEyeAngleGap; Units: Points

        # Calibration Angles
        self.calibrationAngles = [[-45, 0, 45] for _ in range(self.numChannels)]
        self.calibrationVoltages = [[] for _ in range(self.numChannels)]
        # Pointers for Calibration
        self.calibrateChannelNum = 0  # The Current Channel We are Calibrating
        self.channelCalibrationPointer = 0  # A Pointer to the Current Angle Being Calibrated (A Pointer for Angle in self.calibrationAngles)
        # Calibration Function for Eye Angle
        self.steadyStateEye = voltageRange[0] + (voltageRange[1] - voltageRange[0]) / 2  # The Steady State Voltage of the System (With No Eye Movement); Units: Volts
        self.predictEyeAngle = [lambda x: (x - self.steadyStateEye) * 30] * self.numChannels

        # Holder parameters.
        self.minPoints_halfBaseline = None
        self.trailingAverageData = None
        self.currentEyeVoltages = None
        self.multipleBlinksX = None
        self.singleBlinksX = None
        self.currentState = None
        self.culledBlinkY = None
        self.culledBlinkX = None
        self.blinksXLocs = None
        self.blinksYLocs = None
        self.blinkTypes = None

        # Finalize the protocol parameters.
        self.resetAnalysisVariables()
        self.checkParams()

    def resetAnalysisVariables(self):
        # Hold Past Information
        self.trailingAverageData = {}
        for channelIndex in range(self.numChannels):
            self.trailingAverageData[channelIndex] = [0] * self.numPointsPerBatch

        # Reset Last Eye Voltage (Volts)
        self.currentEyeVoltages = [self.steadyStateEye for _ in range(self.numChannels)]
        # Reset Blink Indices
        self.singleBlinksX = []
        self.multipleBlinksX = []
        self.blinksXLocs = []
        self.blinksYLocs = []
        self.culledBlinkX = []
        self.culledBlinkY = []

        # Blink Classification
        self.blinkTypes = ['Relaxed', 'Stroop', 'Exercise', 'VR']
        self.currentState = self.blinkTypes[0]

    def checkParams(self):
        self.checkGlobalParams()
        assert self.numChannels == 1, "The EOG protocol now only supports one channel of data due tp feature alignment issues."

    def setSamplingFrequencyParams(self):
        # Set Blink Parameters
        self.minPoints_halfBaseline = max(1, int(self.samplingFreq * 0.015))  # The Minimum Points in the Left/Right Baseline
        self.dataPointBuffer = max(self.dataPointBuffer, int(self.samplingFreq * 20))  # cutOffFreq = 0.1, use 20 seconds; cutOffFreq = 0.01, use 60 seconds; cutOffFreq = 0.05, use 20 seconds

    # ---------------------------------------------------------------------- #
    # ------------------------- Data Analysis Begins ----------------------- #

    def analyzeData(self, dataFinger, calibrateModel=False):

        eyeAngles = []
        # Analyze all incoming EOG channels.
        for channelIndex in range(self.numChannels):

            # ---------------------- Filter the Data ----------------------- #    
            # Find the starting/ending points of the data to analyze.
            startFilterPointer = max(dataFinger - self.dataPointBuffer, 0)
            dataBuffer = np.array(self.channelData[channelIndex][startFilterPointer:dataFinger + self.numPointsPerBatch])
            timePoints = np.array(self.timePoints[startFilterPointer:dataFinger + self.numPointsPerBatch])

            # Assert that the data is within the expected voltage range.
            assert np.max(dataBuffer) <= self.voltageRange[1] + 0.1 and self.voltageRange[0] - 0.1 <= np.min(dataBuffer), f"Data is not within the expected voltage range: {np.max(dataBuffer)} {np.min(dataBuffer)}"

            # Extract sampling frequency from the first batch of data.
            if not self.samplingFreq:
                self.setSamplingFrequency(startFilterPointer)

            # Filter the data and remove bad indices.
            filteredTime, filteredData, goodIndicesMask = self.filterData(timePoints, dataBuffer, removePoints=True)
            # --------------------------------------------------------------- #

            # ------------------- Extract Blink Features  ------------------- #
            # Extract EOG peaks only from the vertical channel (channelIndex = 0).
            if channelIndex == 0 and self.collectFeatures:
                # Get a pointer for the unanalyzed data in filteredData (Account for missing points!).
                unAnalyzedDataPointer_NoCulling = max(0, self.lastAnalyzedDataInd[channelIndex] - startFilterPointer)
                unAnalyzedDataPointer = (goodIndicesMask[0:unAnalyzedDataPointer_NoCulling]).sum(axis=0, dtype=int)
                # Only analyze EOG data where no blinks have been recorded.
                unAnalyzedData = filteredData[unAnalyzedDataPointer:]
                unAnalyzedTimes = filteredTime[unAnalyzedDataPointer:]

                # Find and record the blinks in the EOG data.
                self.findBlinks(unAnalyzedTimes, unAnalyzedData, channelIndex, self.debugBlinkDetection)
            # --------------------------------------------------------------- #

            # --------------------- Calibrate Eye Angle --------------------- #
            if calibrateModel and self.calibrateChannelNum == channelIndex:
                argMax = np.argmax(filteredData)
                argMin = np.argmin(filteredData)
                earliestExtrema = argMax if argMax < argMin else argMin

                plt.plot(filteredTime, filteredData)
                plt.plot(filteredTime[earliestExtrema], filteredData[earliestExtrema], 'o', linewidth=3)
                plt.show()

                self.calibrationVoltages[self.calibrateChannelNum].append(np.average(filteredData[earliestExtrema:earliestExtrema + 20]))
            # --------------------------------------------------------------- #

            # --------------------- Predict Eye Movement  ------------------- #
            # Discretize Voltages (Using an Average Around the Point)
            channelVoltages = []
            for segment in range(self.moveDataFinger - self.predictEyeAngleGap, -self.predictEyeAngleGap, -self.predictEyeAngleGap):
                endPos = -segment if -segment != 0 else len(filteredData)
                currentEyeVoltage = self.findTraileringAverage(filteredData[-segment - self.voltagePositionBuffer:endPos], deviationThreshold=self.minVoltageMovement)
                # Compare Voltage Difference to Remove Small Shakes
                if abs(currentEyeVoltage - self.currentEyeVoltages[channelIndex]) > self.minVoltageMovement:
                    self.currentEyeVoltages[channelIndex] = currentEyeVoltage
                channelVoltages.append(self.currentEyeVoltages[channelIndex])

            # Predict the Eye's Degree
            if self.predictEyeAngle[channelIndex]:
                eyeAngle = self.predictEyeAngle[channelIndex](self.currentEyeVoltages[channelIndex])
                eyeAngles.append(eyeAngle)
            # --------------------------------------------------------------- #

            # ------------------- Plot Biolectric Signals ------------------- #
            if self.plotStreamedData and not calibrateModel:
                # Format the raw data:.
                timePoints = timePoints[dataFinger - startFilterPointer:]  # Shared axis for all signals
                rawData = dataBuffer[dataFinger - startFilterPointer:]
                # Format the filtered data
                filterOffset = (goodIndicesMask[0:dataFinger - startFilterPointer]).sum(axis=0, dtype=int)

                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.plottingMethods.bioelectricDataPlots[channelIndex].set_data(timePoints, rawData)
                self.plottingMethods.bioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1])

                # Keep Track of Recently Digitized Data
                # for voltageInd in range(len(channelVoltages)):
                #     self.trailingAverageData[channelIndex].extend([channelVoltages[voltageInd]]*self.predictEyeAngleGap)
                # self.trailingAverageData[channelIndex] = self.trailingAverageData[channelIndex][len(channelVoltages)*self.predictEyeAngleGap:]
                # self.plottingMethods.trailingAveragePlots[channelIndex].set_data(filteredTime, self.trailingAverageData[channelIndex][-len(timePoints):])
                # Plot the Filtered + Digitized Data
                self.plottingMethods.filteredBioelectricDataPlots[channelIndex].set_data(filteredTime[filterOffset:], filteredData[filterOffset:])
                self.plottingMethods.filteredBioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1])
                # Plot the Eye's Angle if Electrodes are Calibrated
                # if self.predictEyeAngle[channelIndex]:
                #     self.plottingMethods.filteredBioelectricPlotAxes[channelIndex].legend(["Eye's Angle: " + "%.3g"%eyeAngle, "Current State: " + self.currentState], loc="upper left")
                # Add Eye Blink Peaks
                if channelIndex == 0:
                    self.plottingMethods.eyeBlinkLocPlots[channelIndex].set_data(self.blinksXLocs, self.blinksYLocs)
                    self.plottingMethods.eyeBlinkCulledLocPlots[channelIndex].set_data(self.culledBlinkX, self.culledBlinkY)

                # Plot a single feature.
                if len(self.compiledFeatures[channelIndex]) != 0:
                    self.plottingMethods.featureDataPlots[channelIndex].set_data(self.rawFeatureTimes[channelIndex], np.array(self.compiledFeatures[channelIndex])[:, 24])
                    self.plottingMethods.featureDataPlotAxes[channelIndex].legend(["Blink Duration"], loc="upper left")

        # -------------------- Update Virtual Reality  ---------------------- #

        # Use eyes to move around in the virtual environment.
        # If actionControl and self.predictEyeAngle[channelIndex] and not calibrateModel:
        #     actionControl.setGaze(eyeAngles)

        # ------------------------------------------------------------------- #

    def filterData(self, timePoints, data, removePoints=False):
        if removePoints:
            # Find the bad points associated with motion artifacts
            motionIndices = np.logical_or(data < 0.1, data > 3.18)
            motionIndices_Broadened = scipy.signal.savgol_filter(motionIndices, max(3, int(self.samplingFreq * 2)), 1, mode='nearest', deriv=0)
            goodIndicesMask = motionIndices_Broadened < 0.01
        else:
            goodIndicesMask = np.full_like(data, True, dtype=bool)

        # Filtering the whole dataset
        filteredData = self.filteringMethods.bandPassFilter.butterFilter(data, self.cutOffFreq[1], self.samplingFreq, order=5, filterType='low')
        filteredData = scipy.signal.savgol_filter(filteredData, max(3, int(self.samplingFreq * 0.05)), 2, mode='nearest', deriv=0)
        # Remove the bad points from the filtered data
        filteredTime = timePoints[goodIndicesMask]
        filteredData = filteredData[goodIndicesMask]

        return filteredTime, filteredData, goodIndicesMask

    # ------------------------- Signal Analysis --------------------------------- #

    def findBlinks(self, xData, yData, channelIndex, debugBlinkDetection=False):

        # Initialize the new raw features and times.
        newFeatureTimes, newRawFeatures = [], []

        # --------------------- Find and Analyze Blinks --------------------- #

        # Find all potential blinks (peaks) in the data.
        peakIndices = scipy.signal.find_peaks(yData, prominence=0.1, width=max(1, int(self.samplingFreq * 0.04)))[0]

        # For each potential blink peak.
        for peakInd in peakIndices:
            peakTimePoint = xData[peakInd]

            # Do not reanalyze a peak (waste of time).
            if peakTimePoint <= self.timePoints[self.lastAnalyzedDataInd[channelIndex]]: continue

            # ------------------ Find the Peak's Baselines ------------------ #
            # Calculate the baseline of the peak.
            leftBaselineIndex = self.universalMethods.findNearbyMinimum(yData, peakInd, binarySearchWindow=-max(1, int(self.samplingFreq * 0.005)), maxPointsSearch=max(1, int(self.samplingFreq * 0.5)))
            rightBaselineIndex = self.universalMethods.findNearbyMinimum(yData, peakInd, binarySearchWindow=max(1, int(self.samplingFreq * 0.005)), maxPointsSearch=max(1, int(self.samplingFreq * 0.5)))

            # Wait to analyze peaks that are not fully formed. Additionally, filtering could affect boundary points.
            if rightBaselineIndex >= len(xData) - max(1, int(self.samplingFreq * 0.1)): break

            # --------------------- Initial Peak Culling --------------------- #

            # All peaks after this point will be evaluated only once.
            self.lastAnalyzedDataInd[channelIndex] = self.findStartFeatureWindow(self.lastAnalyzedDataInd[channelIndex], peakTimePoint, 0)

            # Cull peaks that are too small to be a blink
            if yData[peakInd] - max(yData[leftBaselineIndex], yData[rightBaselineIndex]) < self.minPeakHeight_Volts:
                if debugBlinkDetection: print("\t\tThe Peak Height is too Small; Time = ", peakTimePoint)
                continue
            # If no baseline is found, ignore the blink (If it's too noisy, It's probably not a blink)
            elif leftBaselineIndex >= peakInd - self.minPoints_halfBaseline or rightBaselineIndex <= peakInd + self.minPoints_halfBaseline:
                if debugBlinkDetection: print("\t\tPeak Width too Small; Time = ", peakTimePoint)
                continue

            baseLineSkewed_Large = abs(yData[rightBaselineIndex] - yData[leftBaselineIndex]) > 0.4 * (yData[peakInd] - max(yData[rightBaselineIndex], yData[leftBaselineIndex]))
            # A large skew likely indicates eye movement + blink.
            if baseLineSkewed_Large:
                if debugBlinkDetection: print("\t\tBaseline is WAY Too Skewed. Probably Eye Movement + Blink. Time = ", xData[rightBaselineIndex], yData[rightBaselineIndex], yData[peakInd], yData[leftBaselineIndex])
                continue

            # Add a buffer to the baseline
            leftBaselineIndex -= min(leftBaselineIndex, int(self.samplingFreq * 0.05))
            rightBaselineIndex += int(self.samplingFreq * 0.05)

            # -------------------- Extract Blink Features ------------------- #

            peakInd = self.universalMethods.findLocalMax(yData, peakInd, binarySearchWindow=max(1, int(self.samplingFreq * 0.005)), maxPointsSearch=len(yData))
            newFeatures = self.extractFeatures(xData[leftBaselineIndex:rightBaselineIndex + 1], yData[leftBaselineIndex:rightBaselineIndex + 1].copy(), peakInd - leftBaselineIndex, debugBlinkDetection)

            # Remove Peaks that are not Blinks
            if len(newFeatures) == 0: continue
            # Label/Remove Possible Winks
            elif len(newFeatures) == 1:
                if debugBlinkDetection: print("\t\tWink")
                self.culledBlinkX.append(peakTimePoint)
                self.culledBlinkY.append(yData[peakInd])
                continue

            # Record the Blink's Location
            self.blinksXLocs.append(peakTimePoint)
            self.blinksYLocs.append(yData[peakInd])
            # Store the Blink's Features
            newFeatureTimes.append(peakTimePoint)
            newRawFeatures.append(newFeatures)

            # ----------------- Singular or Multiple Blinks? ---------------- #

            multPeakSepMax = 0.5  # No Less Than 0.25
            # Check if the Blink is a Part of a Multiple Blink Sequence
            if self.singleBlinksX and peakTimePoint - self.singleBlinksX[-1] < multPeakSepMax:
                # If So, Remove the Last Single Blink as it's a Multiple
                lastBlinkX = self.singleBlinksX.pop()
                # Check if Other Associated Multiples Have Been Found
                if self.multipleBlinksX and peakTimePoint - self.multipleBlinksX[-1][-1] < multPeakSepMax:
                    self.multipleBlinksX[-1].append(peakTimePoint)
                else:
                    self.multipleBlinksX.append([lastBlinkX, peakTimePoint])
            else:
                self.singleBlinksX.append(peakTimePoint)

            # ----------------------- Plot Tent Shape ----------------------- #

            if debugBlinkDetection:
                peakTentX, peakTentY = newFeatures[0], newFeatures[1]
                xData = np.array(xData)
                yData = np.array(yData)
                # Plot the Peak
                plt.plot(xData, yData)
                plt.plot(xData[peakInd], yData[peakInd], 'ko')
                plt.plot(xData[leftBaselineIndex], yData[leftBaselineIndex], 'go')
                plt.plot(xData[rightBaselineIndex], yData[rightBaselineIndex], 'ro')
                plt.plot(peakTentX, peakTentY, 'kx')
                # Figure Aesthetics
                plt.xlim([xData[leftBaselineIndex], xData[rightBaselineIndex]])
                plt.show()

            # --------------------------------------------------------------- #

        # Even if no blinks found, we know that blinks wnt reappear in old data.
        if len(xData) != 0:
            self.lastAnalyzedDataInd[channelIndex] = self.findStartFeatureWindow(self.lastAnalyzedDataInd[channelIndex], xData[-1], 10)

        # Compile the new raw features into a smoothened (averaged) feature.
        self.readData.compileContinuousFeatures(newFeatureTimes, newRawFeatures, self.rawFeatureTimes[channelIndex], self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex], self.featureAverageWindow)

    def findStartFeatureWindow(self, timePointer, currentTime, timeWindow):
        # Loop through until you find the first time in the window.
        while self.timePoints[timePointer] < currentTime - timeWindow:
            timePointer += 1

        return timePointer

    def quantifyPeakShape(self, xData, yData):
        # Calculate the derivatives
        firstDeriv = scipy.signal.savgol_filter(yData, max(3, int(self.samplingFreq * 0.01)), 2, delta=1 / self.samplingFreq, mode='interp', deriv=1)
        secondDeriv = scipy.signal.savgol_filter(yData, max(3, int(self.samplingFreq * 0.01)), 2, delta=1 / self.samplingFreq, mode='interp', deriv=2)

        # Calculate the time derivatives
        dx_dt = scipy.signal.savgol_filter(xData, max(3, int(self.samplingFreq * 0.01)), 2, delta=1 / self.samplingFreq, mode='interp', deriv=1)
        dx_dt2 = scipy.signal.savgol_filter(xData, max(3, int(self.samplingFreq * 0.01)), 2, delta=1 / self.samplingFreq, mode='interp', deriv=2)
        # Calculate Peak Shape parameters
        speed = np.sqrt(dx_dt ** 2 + firstDeriv ** 2)
        curvature = np.abs((dx_dt2 * firstDeriv - dx_dt * secondDeriv)) / speed ** 3  # Units 1/Volts

        # Return the Peak Analysis Equations
        return firstDeriv, secondDeriv, curvature

    def getDerivPeaks(self, firstDeriv, secondDeriv, peakInd):
        # Get velocity peaks
        leftVelMax = np.argmax(firstDeriv[:peakInd])
        rightVelMin = self.universalMethods.findNearbyMinimum(firstDeriv, peakInd, binarySearchWindow=max(1, int(self.samplingFreq * 0.005)), maxPointsSearch=len(firstDeriv))
        # Organize velocity peaks
        velInds = [leftVelMax, rightVelMin]

        # Find Acceleration peaks
        leftAccelMax = self.universalMethods.findNearbyMaximum(secondDeriv, leftVelMax, binarySearchWindow=-max(1, int(self.samplingFreq * 0.005)), maxPointsSearch=len(secondDeriv))
        leftAccelMin = self.universalMethods.findNearbyMinimum(secondDeriv, leftVelMax, binarySearchWindow=max(1, int(self.samplingFreq * 0.005)), maxPointsSearch=len(secondDeriv))
        rightAccelMax = self.universalMethods.findNearbyMaximum(secondDeriv, rightVelMin, binarySearchWindow=max(1, int(self.samplingFreq * 0.005)), maxPointsSearch=len(secondDeriv))
        # Organize acceleration peaks
        accelInds = [leftAccelMax, leftAccelMin, rightAccelMax]

        # Find the end of peak
        startEndSearchInd = self.universalMethods.findNearbyMinimum(secondDeriv, rightAccelMax, binarySearchWindow=max(1, int(self.samplingFreq * 0.005)), maxPointsSearch=len(secondDeriv))
        peakEndInd = startEndSearchInd + np.argmax(secondDeriv[startEndSearchInd:])
        # Organize peak boundaries
        peakBoundaries = [0, peakEndInd]

        return velInds, accelInds, peakBoundaries

    def extractFeatures(self, xData, yData, peakInd, debugBlinkDetection=False):
        if len(xData) < int(self.samplingFreq * 0.01):
            if debugBlinkDetection: print("\t\tPeak was around noisy data", len(xData), xData[peakInd])
            return []

        # ---------------------- Baseline Subtraction ---------------------- #

        # Shift the data to zero
        firstDeriv = scipy.signal.savgol_filter(yData, max(3, int(self.samplingFreq * 0.01)), polyorder=2, delta=1 / self.samplingFreq, mode='interp', deriv=1)
        secondDeriv = scipy.signal.savgol_filter(yData, max(3, int(self.samplingFreq * 0.01)), polyorder=2, delta=1 / self.samplingFreq, mode='interp', deriv=2)

        firstDeriv_zeroCrossings = self.universalMethods.findPointCrossing(firstDeriv[0:peakInd] / max(firstDeriv[0:peakInd]), threshold=0.01)
        secondDeriv_zeroCrossings = self.universalMethods.findPointCrossing(secondDeriv[0:peakInd] / max(secondDeriv[0:peakInd]), threshold=0.01)

        # Does the first derivative cross zero -> baseline peak
        if len(firstDeriv_zeroCrossings) != 0:
            peakStartInd = firstDeriv_zeroCrossings[-1] + 1
        # Does the second derivative cross zero -> Closest first derivative to zero
        elif len(secondDeriv_zeroCrossings) != 0:
            peakStartInd = secondDeriv_zeroCrossings[-1] + 1
        # Else, keep the current baseline
        else:
            peakStartInd = 0

        yData = yData[peakStartInd:]
        xData = xData[peakStartInd:]
        peakInd = peakInd - peakStartInd
        # print("Data Start: " + str(xData[0]))
        yData -= yData[0]

        # Normalization to remove amplitudes drift and sensor variations.
        peakHeight = yData[peakInd]
        yData = yData / peakHeight

        # ---------------- Calculate the Peak's Derivatives ---------------- #

        # Calculate the Derivatives and Curvature
        firstDeriv, secondDeriv, curvature = self.quantifyPeakShape(xData, yData)
        # Calculate the Peak Derivatives
        velInds, accelInds, peakBoundaries = self.getDerivPeaks(firstDeriv, secondDeriv, peakInd)

        # Cull peaks with bad derivatives: malformed peaks
        if velInds[1] < velInds[0]:
            if debugBlinkDetection: print("\t\tBad Velocity Inds", velInds, xData[peakInd])
            return []
        elif not accelInds[0] < accelInds[1] < accelInds[2]:
            if debugBlinkDetection: print("\t\tBad Acceleration Inds: ", accelInds, xData[peakInd])
            return []
        elif not accelInds[0] < velInds[0] < accelInds[1]:
            if debugBlinkDetection: print("\t\tBad Derivative Inds Order: First Half")
            if debugBlinkDetection: print("\t\t\t", peakInd, velInds, accelInds, xData[peakInd])
            return []
        elif not peakInd < velInds[1] < accelInds[2]:
            if debugBlinkDetection: print("\t\tBad Derivative Inds Order: Second Half")
            if debugBlinkDetection: print("\t\t\t", peakInd, velInds, accelInds, xData[peakInd])
            return []

        # --------------------- Find the Blink's Endpoints ------------------ #

        # Linearly Fit the Peak's Slope
        upSlopeTangentParams = [firstDeriv[velInds[0]], yData[velInds[0]] - firstDeriv[velInds[0]] * xData[velInds[0]]]
        downSlopeTangentParams = [firstDeriv[velInds[1]], yData[velInds[1]] - firstDeriv[velInds[1]] * xData[velInds[1]]]

        # Calculate the New Endpoints of the Peak
        startBlinkX, _ = self.universalMethods.findLineIntersectionPoint([0, 0], upSlopeTangentParams)
        endBlinkX, _ = self.universalMethods.findLineIntersectionPoint(downSlopeTangentParams, [0, 0])
        # Calculate the New Endpoint Locations of the Peak
        startBlinkInd = np.argmin(abs(xData - startBlinkX))
        endBlinkInd = np.argmin(abs(xData - endBlinkX))

        # -------------------- Extract Amplitude Features ------------------- #

        # Find Peak's Tent
        peakTentX, peakTentY = self.universalMethods.findLineIntersectionPoint(upSlopeTangentParams, downSlopeTangentParams)
        # Calculate Tent Deviation Features
        tentDeviationX = peakTentX - xData[peakInd]
        tentDeviationY = peakTentY - yData[peakInd]
        tentDeviationRatio = tentDeviationX / tentDeviationY

        # Closing Amplitudes
        maxClosingAccel_Loc = yData[accelInds[0]]
        maxClosingVel_Loc = yData[velInds[0]]
        minBlinkAccel_Loc = yData[accelInds[1]]
        # Opening Amplitudes
        openingAmpVel_Loc = yData[velInds[1]]
        maxOpeningAccel_secondHalfLoc = yData[peakBoundaries[1]]

        # Closing Amplitude Intervals
        closingAmpSegmentFull = minBlinkAccel_Loc - maxClosingAccel_Loc
        # Opening Amplitude Intervals
        openingAmplitudeFull = openingAmpVel_Loc - maxOpeningAccel_secondHalfLoc

        # Mixed Amplitude Intervals
        velocityAmpInterval = openingAmpVel_Loc - maxClosingVel_Loc

        # Amplitude Ratio
        velAmpRatio = yData[velInds[0]] / yData[velInds[1]]

        # -------------------- Extract Duration Features -------------------- #

        # Find the Standard Blink Durations
        blinkDuration = endBlinkX - startBlinkX  # The Total Time of the Blink
        closingTime_Tent = peakTentX - startBlinkX  # The Time for the Eye to Close
        openingTime_Tent = endBlinkX - peakTentX  # The Time for the Eye to Open
        closingTime_Peak = xData[peakInd] - startBlinkX
        openingTime_Peak = endBlinkX - xData[peakInd]  # The Time for the Eye to Open
        # Calculate the Duration Ratios
        closingFraction = closingTime_Peak / blinkDuration
        openingFraction = openingTime_Peak / blinkDuration

        # Calculate the Half Amplitude Duration
        blinkAmp50Right = xData[peakInd + np.argmin(abs(yData[peakInd:] - yData[peakInd] / 2))]
        blinkAmp50Left = xData[np.argmin(abs(yData[0:peakInd] - yData[peakInd] / 2))]
        halfClosedTime = blinkAmp50Right - blinkAmp50Left
        # Calculate Time the Eyes are Closed
        blinkAmp90Right = xData[peakInd + np.argmin(abs(yData[peakInd:] - yData[peakInd] * 0.9))]
        blinkAmp90Left = xData[np.argmin(abs(yData[0:peakInd] - yData[peakInd] * 0.9))]
        eyesClosedTime = blinkAmp90Right - blinkAmp90Left
        # Calculate Percent Closed
        percentTimeEyesClosed = eyesClosedTime / halfClosedTime

        # Divide the Peak by Velocity
        startToVel = xData[velInds[0]] - startBlinkX
        velToPeak = xData[peakInd] - xData[velInds[0]]

        # Calculate the Blink's Duration
        condensedDuration2 = xData[peakBoundaries[1]] - xData[accelInds[0]]

        # New Half-Duration
        durationByVel1 = xData[peakInd:][np.argmin(abs(yData[peakInd:] - yData[velInds[0]]))] - xData[velInds[0]]

        # ------------------- Extract Derivative Features ------------------- #

        # Extract Closing Slopes
        closingSlope_MaxVel = firstDeriv[velInds[0]]
        # Extract Opening Slopes
        openingSlope_MinVel = firstDeriv[velInds[1]]

        # Ratios with Vel0
        velRatio = firstDeriv[velInds[0]] / firstDeriv[velInds[1]]

        # ------------------- Extract Integral Features ------------------- #

        portion5Integral = scipy.integrate.simpson(yData[peakInd:accelInds[2]], xData[peakInd:accelInds[2]])

        # Other Integrals
        closingIntegral = scipy.integrate.simpson(yData[startBlinkInd:peakInd], xData[startBlinkInd:peakInd])
        openingIntegral = scipy.integrate.simpson(yData[peakInd:endBlinkInd], xData[peakInd:endBlinkInd])
        closingSlopeIntegral = scipy.integrate.simpson(yData[accelInds[0]:accelInds[1]], xData[accelInds[0]:accelInds[1]])
        peakToVel0Integral = scipy.integrate.simpson(yData[velInds[0]:peakInd], xData[velInds[0]:peakInd])
        peakToVel1Integral = scipy.integrate.simpson(yData[peakInd:velInds[1]], xData[peakInd:velInds[1]])

        # ---------------------- Extract Shape Features --------------------- #

        fullBlink = yData[startBlinkInd:endBlinkInd].copy()
        fullBlink -= min(fullBlink)
        fullBlink = fullBlink / max(fullBlink)
        # Calculate Peak Shape Parameters
        peakAverage = np.mean(fullBlink)
        peakSkew = scipy.stats.skew(fullBlink, bias=False)

        # Standard Deviation
        accelFullSTD = np.std(secondDeriv[startBlinkInd:endBlinkInd], ddof=1)

        # scipy.stats.entropy
        velFullEntropy = scipy.stats.entropy(firstDeriv[startBlinkInd:endBlinkInd] - min(firstDeriv[startBlinkInd:endBlinkInd]) + 10E-10)
        accelFullEntropy = scipy.stats.entropy(secondDeriv[startBlinkInd:endBlinkInd] - min(secondDeriv[startBlinkInd:endBlinkInd]) + 10E-10)

        # ------------------------- Cull Bad Blinks ------------------------- #
        # Blinks are on average 100-400 ms. They can be in the range of 50-500 ms.

        if not 0.008 < blinkDuration < 0.5:
            if debugBlinkDetection: print("\t\tBad Blink Duration:", blinkDuration, xData[peakInd])
            return [None]
        if tentDeviationX < -0.2:
            if debugBlinkDetection: print("\t\tBad Blink tentDeviationX:", tentDeviationX, xData[peakInd])
            return [None]
        if not -0.2 < tentDeviationY < 0.6:
            if debugBlinkDetection: print("\t\tBad Blink tentDeviationY:", tentDeviationY, xData[peakInd])
            return [None]
        # If the Closing Time is Longer Than 150ms, it is Probably Not an Involuntary Blink
        elif not 0.04 < closingTime_Peak < 0.3:
            if debugBlinkDetection: print("\t\tBad Closing Time:", closingTime_Peak, xData[peakInd])
            return [None]
        elif not 0.04 < openingTime_Peak < 0.4:
            if debugBlinkDetection: print("\t\tBad Opening Time:", openingTime_Peak, xData[peakInd])
            return [None]
        elif -1 < velRatio:
            if debugBlinkDetection: print("\t\tBad velRatio:", velRatio, xData[peakInd])
            return [None]
        elif peakSkew < -0.75:
            if debugBlinkDetection: print("\t\tBad peakSkew:", peakSkew, xData[peakInd])
            return [None]

        # --------------------- Cull Voluntary Blinks  ---------------------- #

        # if tentDeviationX < -0.1 or tentDeviationX > 0.1:
        #     if debugBlinkDetection: print("\t\tWINK! tentDeviationX = ", tentDeviationX, " xLoc = ", xData[peakInd]); return ["Wink"]  
        # if not 0 < tentDeviationY < 0.6: # (max may be 0.25)
        #     if debugBlinkDetection: print("\t\tWINK! tentDeviationY = ", tentDeviationY); return ["Wink"]  
        # if blinkAmpRatio < 0.7:
        #     if debugBlinkDetection: print("\t\tWINK! blinkAmpRatio = ", blinkAmpRatio); return ["Wink"]
        # If the Closing Time is Longer Than 150ms, it is Probably Not an Involuntary Blink
        # elif closingTime > 0.2 or closingTime < 0.04:
        #      if debugBlinkDetection: print("\t\tWINK! closingTime:", closingTime); return ["Wink"]
        #     return []

        # ------------------ Consolidate the Blink Features ----------------- #

        finalFeatures = []
        # Organize Amplitude Features        
        finalFeatures.extend([peakHeight, tentDeviationX, tentDeviationY, tentDeviationRatio])
        finalFeatures.extend([closingAmpSegmentFull, openingAmplitudeFull])
        finalFeatures.extend([velocityAmpInterval])
        finalFeatures.extend([velAmpRatio])

        # Organize Duration Features
        finalFeatures.extend([blinkDuration, closingTime_Tent, openingTime_Tent, closingFraction, openingFraction])
        finalFeatures.extend([halfClosedTime, eyesClosedTime, percentTimeEyesClosed])
        finalFeatures.extend([condensedDuration2, durationByVel1])
        finalFeatures.extend([startToVel, velToPeak])

        # Organize Derivative Features
        finalFeatures.extend([closingSlope_MaxVel, openingSlope_MinVel, velRatio])

        # Organize Integral Features
        finalFeatures.extend([portion5Integral, closingIntegral, openingIntegral, closingSlopeIntegral, peakToVel0Integral, peakToVel1Integral])

        # Organize Shape Features
        finalFeatures.extend([peakAverage, accelFullSTD, velFullEntropy, accelFullEntropy])

        # ------------------------------------------------------------------- #

        if debugBlinkDetection:
            sepInds = [startBlinkInd, accelInds[0], accelInds[1], peakInd, accelInds[2], endBlinkInd]
            self.plotData(xData, yData, peakInd, velInds=velInds, accelInds=accelInds, sepInds=sepInds, title="Dividing the Blink")

        return finalFeatures

        # ------------------------------------------------------------------ #

    def findTraileringAverage(self, recentData, deviationThreshold=0.08):
        # Base Case in No Points Came in
        if len(recentData) == 0:
            return self.steadyStateEye

        # Keep Track of the trailingAverage
        trailingAverage = recentData[-1]
        for dataPointInd in range(2, len(recentData) - 1, -2):
            # Get New dataPoint from the Back of the List
            dataPoint = recentData[len(recentData) - dataPointInd]
            # If the dataPoint is Different from the trailingAverage by some Threshold, return the trailingAverage
            if abs(dataPoint - trailingAverage) > deviationThreshold:
                return trailingAverage
            else:
                trailingAverage = (trailingAverage * (dataPointInd - 1) + dataPoint) / dataPointInd
        # Else Return the Average
        return trailingAverage

    @staticmethod
    def plotData(xData, yData, peakInd, velInds=(), accelInds=(), sepInds=(), title="", peakSize=5, lineWidth=2, lineColor="black", ax=None, axisLimits=()):
        xData = np.array(xData)
        yData = np.array(yData)
        # Create Figure
        showFig = False
        if ax is None:
            plt.figure()
            ax = plt.gca()
            showFig = True
        # Plot the Data
        ax.plot(xData, yData, linewidth=lineWidth, color=lineColor)
        ax.plot(xData[peakInd], yData[peakInd], 'o', c='tab:purple', markersize=int(peakSize * 1.5))
        ax.plot(xData[velInds], yData[velInds], 'o', c='tab:red', markersize=peakSize)
        ax.plot(xData[accelInds], yData[accelInds], 'o', c='tab:blue', markersize=peakSize)
        if len(sepInds) > 0:
            sectionColors = ['red', 'orange', 'blue', 'green', 'black']
            for groupInd in range(len(sectionColors)):
                if sepInds[groupInd] in [np.nan, None] or sepInds[groupInd + 1] in [np.nan, None]:
                    continue
                ax.fill_between(xData[sepInds[groupInd]:sepInds[groupInd + 1] + 1], min(yData), yData[sepInds[groupInd]:sepInds[groupInd + 1] + 1], color=sectionColors[groupInd], alpha=0.15)
        # Add Axis Labels and Figure Title
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(title)
        # Change Axis Limits If Given
        if axisLimits:
            ax.set_xlim(axisLimits)
        # Increase DPI (Clarity); Will Slow Down Plotting
        matplotlib.rcParams['figure.dpi'] = 300
        # Show the Plot
        if showFig:
            plt.show()
