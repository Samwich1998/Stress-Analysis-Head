# Basic Modules
import scipy
import numpy as np
# Plotting
import matplotlib
import matplotlib.pyplot as plt

# Import Files
from .globalProtocol import globalProtocol


class eogProtocol(globalProtocol):
    
    def __init__(self, numPointsPerBatch = 3000, moveDataFinger = 10, numChannels = 2, plottingClass = None, readData = None):
        # Filter parameters.
        self.dataPointBuffer = 0          # A prepended buffer of data for filtering. Represents possible bad filtering points; Units: Points
        self.cutOffFreq = [.5, 15]        # Optimal LPF Cutoff in Literature is 6-8 or 20 Hz (Max 35 or 50); I Found 20 Hz was the Best, but can go to 15 if noisy (small amplitude cutoff)
        # High-pass filter parameters.
        self.stopband_edge = 10          # Common values for EEG are 1 Hz and 2 Hz. If you need to remove more noise, you can choose a higher stopband edge frequency. If you need to preserve the signal more, you can choose a lower stopband edge frequency.
        self.passband_ripple = 0.1       # Common values for EEG are 0.1 dB and 0.5 dB. If you need to remove more noise, you can choose a lower passband ripple. If you need to preserve the signal more, you can choose a higher passband ripple.
        self.stopband_attenuation = 20   # Common values for EEG are 40 dB and 60 dB. If you need to remove more noise, you can choose a higher stopband attenuation. If you need to preserve the signal more, you can choose a lower stopband attenuation.

        # Blink Parameters
        self.minPeakHeight_Volts = 0.05    # The Minimum Peak Height in Volts; Removes Small Oscillations

        # Eye Angle Determination Parameters
        self.voltagePositionBuffer = 100  # A Prepended Buffer to Find the Current Average Voltage; Units: Points
        self.minVoltageMovement = 0.05    # The Minimum Voltage Change Required to Register an Eye Movement; Units: Volts
        self.predictEyeAngleGap = 5       # The Number of Points Between Each Eye Gaze Prediction; Will Back-calculate if moveDataFinger > predictEyeAngleGap; Units: Points
        self.steadyStateEye = 3.3/2       # The Steady State Voltage of the System (With No Eye Movement); Units: Volts

        # Calibration Angles
        self.calibrationAngles = [[-45, 0, 45] for _ in range(numChannels)]
        self.calibrationVoltages = [[] for _ in range(numChannels)]
        # Pointers for Calibration
        self.calibrateChannelNum = 0           # The Current Channel We are Calibrating
        self.channelCalibrationPointer = 0     # A Pointer to the Current Angle Being Calibrated (A Pointer for Angle in self.calibrationAngles)
        # Calibration Function for Eye Angle
        self.predictEyeAngle = [lambda x: (x - self.steadyStateEye)*30]*numChannels
        
        # Initialize common model class
        super().__init__("eog", numPointsPerBatch, moveDataFinger, numChannels, plottingClass, readData)
    
    # TODO: [0 for _ in range(self.numChannels)] FOR CERTAIN VARIABLES
    def resetAnalysisVariables(self):
        # Hold Past Information
        self.trailingAverageData = {}
        for channelIndex in range(self.numChannels):
            self.trailingAverageData[channelIndex] = [0]*self.numPointsPerBatch
                    
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
        pass
            
    def setSamplingFrequencyParams(self):
        # Set Blink Parameters
        self.minPoints_halfBaseline = max(1, int(self.samplingFreq*0.015))  # The Minimum Points in the Left/Right Baseline
        self.dataPointBuffer = max(self.dataPointBuffer, int(self.samplingFreq*20))  # cutOffFreq = 0.1, use 20 seconds; cutOffFreq = 0.01, use 60 seconds; cutOffFreq = 0.05, use 20 seconds
    
    # ---------------------------------------------------------------------- #
    # ------------------------- Data Analysis Begins ----------------------- #
    
    def analyzeData(self, dataFinger, calibrateModel = False):     
        
        eyeAngles = []
        # Analyze all incoming EOG channels.
        for channelIndex in range(self.numChannels):
            
            # ---------------------- Filter the Data ----------------------- #    
            # Find the starting/ending points of the data to analyze.
            startFilterPointer = max(dataFinger - self.dataPointBuffer, 0)
            dataBuffer = np.array(self.data[1][channelIndex][startFilterPointer:dataFinger + self.numPointsPerBatch])
            timePoints = np.array(self.data[0][startFilterPointer:dataFinger + self.numPointsPerBatch])
            
            # Extract sampling frequency from the first batch of data.
            if not self.samplingFreq:
                self.setSamplingFrequency(startFilterPointer)
                
            # Filter the data and remove bad indices.
            filteredTime, filteredData, goodIndicesMask = self.filterData(timePoints, dataBuffer, removePoints = True)
            # --------------------------------------------------------------- #
            
            # ------------------- Extract Blink Features  ------------------- #
            # Extract EOG peaks only from the vertical channel (channelIndex = 0).
            if channelIndex == 0 and self.collectFeatures:
                # Get a pointer for the unanalyzed data in filteredData (Account for missing points!).
                unAnalyzedDataPointer_NoCulling = max(0, self.lastAnalyzedDataInd[channelIndex] - startFilterPointer)
                unAnalyzedDataPointer = (goodIndicesMask[0:unAnalyzedDataPointer_NoCulling]).sum(axis = 0, dtype=int)
                # Only analyze EOG data where no blinks have been recorded.
                unAnalyzedData = filteredData[unAnalyzedDataPointer:] 
                unAnalyzedTimes = filteredTime[unAnalyzedDataPointer:]
                
                # Find and record the blinks in the EOG data.
                self.findBlinks(unAnalyzedTimes, unAnalyzedData, channelIndex)
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
            for segment in range(self.moveDataFinger-self.predictEyeAngleGap, -self.predictEyeAngleGap, -self.predictEyeAngleGap):
                endPos = -segment if -segment != 0 else len(filteredData)
                currentEyeVoltage = self.findTraileringAverage(filteredData[-segment - self.voltagePositionBuffer:endPos], deviationThreshold = self.minVoltageMovement)
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
                timePoints = timePoints[dataFinger - startFilterPointer:] # Shared axis for all signals
                rawData = dataBuffer[dataFinger - startFilterPointer:]
                # Format the filtered data
                filterOffset = (goodIndicesMask[0:dataFinger - startFilterPointer]).sum(axis = 0, dtype=int)

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
                    self.plottingMethods.featureDataPlots[channelIndex].set_data(self.featureTimes[channelIndex], np.array(self.compiledFeatures[channelIndex])[:, 24])
                    self.plottingMethods.featureDataPlotAxes[channelIndex].legend(["Blink Duration"], loc="upper left")
                    
            # --------------------------------------------------------------- #   
            
        # -------------------- Update Virtual Reality  ---------------------- #
        # Use eyes to move arund in the virtual enviroment.
        # if actionControl and self.predictEyeAngle[channelIndex] and not calibrateModel:
        #     actionControl.setGaze(eyeAngles)
        # ------------------------------------------------------------------- #

    def filterData(self, timePoints, data, removePoints = False):
        if removePoints:
            # Find the bad points associated with motion artifacts
            motionIndices = np.logical_or(data < 0.1, data > 3.18)
            motionIndices_Broadened = scipy.signal.savgol_filter(motionIndices, max(3, int(self.samplingFreq*2)), 1, mode='nearest', deriv=0)
            goodIndicesMask = motionIndices_Broadened < 0.01 
        else:
            goodIndicesMask = np.full_like(data, True, dtype = bool)
        
        # Filtering the whole dataset
        filteredData = self.filteringMethods.bandPassFilter.butterFilter(data, self.cutOffFreq[1], self.samplingFreq, order = 5, filterType = 'low')
        filteredData = scipy.signal.savgol_filter(filteredData, max(3, int(self.samplingFreq*0.05)), 2, mode='nearest', deriv=0)
        # Remove the bad points from the filtered data
        filteredTime = timePoints[goodIndicesMask]
        filteredData = filteredData[goodIndicesMask]
        
        return filteredTime, filteredData, goodIndicesMask

# --------------------------------------------------------------------------- #
# ------------------------- Signal Analysis --------------------------------- #
    
    def findBlinks(self, xData, yData, channelIndex, debugBlinkDetection = False):
        
        # --------------------- Find and Analyze Blinks --------------------- #
        # Find all potential blinks (peaks) in the data.
        peakIndices = scipy.signal.find_peaks(yData, prominence=0.1, width=max(1, int(self.samplingFreq*0.04)))[0];
        
        # For each potential blink peak.
        for peakInd in peakIndices:
            peakTimePoint = xData[peakInd]
            
            # Dont reanalyze a peak (waste of time).
            if peakTimePoint <= self.data[0][self.lastAnalyzedDataInd[channelIndex]]:
                continue
            
            # ------------------ Find the Peak's Baselines ------------------ #
            # Calculate the baseline of the peak.
            leftBaselineIndex = self.universalMethods.findNearbyMinimum(yData, peakInd, binarySearchWindow = -max(1, int(self.samplingFreq*0.005)), maxPointsSearch = max(1, int(self.samplingFreq*0.5)))
            rightBaselineIndex = self.universalMethods.findNearbyMinimum(yData, peakInd, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = max(1, int(self.samplingFreq*0.5)))
            
            # Wait to analyze peaks that are not fully formed. Additionally, filtering could effect boundary points.
            if rightBaselineIndex >= len(xData) - max(1, int(self.samplingFreq*0.1)):
                break
            # --------------------------------------------------------------- #
            
            # --------------------- Initial Peak Culling --------------------- #
            # All peaks after this point will been evaluated only once.
            self.lastAnalyzedDataInd[channelIndex] = self.findStartFeatureWindow(self.lastAnalyzedDataInd[channelIndex], peakTimePoint, 0)
    
            # Cull peaks that are too small to be a blink
            if yData[peakInd] - max(yData[leftBaselineIndex], yData[rightBaselineIndex]) < self.minPeakHeight_Volts:
                # if debugBlinkDetection: print("\t\tThe Peak Height is too Small; Time = ", peakTimePoint)
                continue
            # If no baseline is found, ignore the blink (If its too noisy, its probably not a blink)
            elif leftBaselineIndex >= peakInd - self.minPoints_halfBaseline or rightBaselineIndex <= peakInd + self.minPoints_halfBaseline:
                if debugBlinkDetection: print("\t\tPeak Width too Small; Time = ", peakTimePoint)
                continue
            
            baseLineSkewed_Large = abs(yData[rightBaselineIndex] - yData[leftBaselineIndex]) > 0.4*(yData[peakInd] - max(yData[rightBaselineIndex], yData[leftBaselineIndex]))                        
            # A large skew likely indicates eye movement + blink.
            if baseLineSkewed_Large:
                if debugBlinkDetection: print("\t\tBaseline is WAY Too Skewed. Probably Eye Movement + Blink. Time = ", xData[rightBaselineIndex], yData[rightBaselineIndex], yData[peakInd], yData[leftBaselineIndex])
                continue
            
            # Add a buffer to the baseline
            leftBaselineIndex -= min(leftBaselineIndex, int(self.samplingFreq*0.05))
            rightBaselineIndex += int(self.samplingFreq*0.05)
            # --------------------------------------------------------------- #
            
            # -------------------- Extract Blink Features ------------------- #
            peakInd = self.universalMethods.findLocalMax(yData, peakInd, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(yData))
            newFeatures = self.extractFeatures(xData[leftBaselineIndex:rightBaselineIndex+1], yData[leftBaselineIndex:rightBaselineIndex+1].copy(), peakInd-leftBaselineIndex, debugBlinkDetection)

            # Remove Peaks that are not Blinks
            if len(newFeatures) == 0:
                #if debugBlinkDetection: print("\t\tNo Features; Time = ", peakTimePoint)
                continue
            # Label/Remove Possible Winks
            elif len(newFeatures) == 1:
                if debugBlinkDetection: print("\t\tWink")
                self.culledBlinkX.append(peakTimePoint)
                self.culledBlinkY.append(yData[peakInd])
                continue
            
            # Record the Blink's Location
            self.blinksXLocs.append(peakTimePoint)
            self.blinksYLocs.append(yData[peakInd])
            # Keep track of the new features
            self.readData.averageFeatures([peakTimePoint], [newFeatures], self.featureTimes[channelIndex], self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex], self.featureAverageWindow)
            # --------------------------------------------------------------- #
            
            # ----------------- Singular or Multiple Blinks? ---------------- #
            multPeakSepMax = 0.5   # No Less Than 0.25
            # Check if the Blink is a Part of a Multiple Blink Sequence
            if self.singleBlinksX and peakTimePoint - self.singleBlinksX[-1] < multPeakSepMax:
                # If So, Remove the Last Single Blink as its a Multiple
                lastBlinkX = self.singleBlinksX.pop()
                # Check if Other Associated Multiples Have Been Found
                if self.multipleBlinksX and peakTimePoint - self.multipleBlinksX[-1][-1] < multPeakSepMax:
                    self.multipleBlinksX[-1].append(peakTimePoint)
                else:
                    self.multipleBlinksX.append([lastBlinkX, peakTimePoint])
            else:
                self.singleBlinksX.append(peakTimePoint)
            # --------------------------------------------------------------- #
            
            # ----------------------- Plot Tent Shape ----------------------- #
            if False:
                peakTentX, peakTentY = newFeatures[0], newFeatures[1]
                xData = np.array(xData); yData = np.array(yData)
                # Plot the Peak
                plt.plot(xData, yData);
                plt.plot(xData[peakInd], yData[peakInd], 'ko');
                plt.plot(xData[leftBaselineIndex], yData[leftBaselineIndex], 'go');
                plt.plot(xData[rightBaselineIndex], yData[rightBaselineIndex], 'ro');
                plt.plot(peakTentX, peakTentY, 'kx')
                # Figure Aesthetics
                plt.xlim([xData[leftBaselineIndex], xData[rightBaselineIndex]])
                plt.show()
            # --------------------------------------------------------------- #
            
        # Even if no blinks found, we know that blinks wnt reappear in old data.
        if len(xData) != 0:
            self.lastAnalyzedDataInd[channelIndex] = self.findStartFeatureWindow(self.lastAnalyzedDataInd[channelIndex], xData[-1], 10)

    
            
    def findStartFeatureWindow(self, timePointer, currentTime, timeWindow):
        # Loop through until you find the first time in the window.
        while self.data[0][timePointer] < currentTime - timeWindow:
            timePointer += 1
            
        return timePointer
    
    def quantifyPeakShape(self, xData, yData, peakInd):
        # Caluclate the derivatives
        firstDeriv = scipy.signal.savgol_filter(yData, max(3, int(self.samplingFreq*0.01)), 2, delta=1/self.samplingFreq, mode='interp', deriv=1)
        secondDeriv = scipy.signal.savgol_filter(yData, max(3, int(self.samplingFreq*0.01)), 2, delta=1/self.samplingFreq, mode='interp', deriv=2)
        thirdDeriv = scipy.signal.savgol_filter(secondDeriv, max(3, int(self.samplingFreq*0.01)), 2, delta=1/self.samplingFreq, mode='interp', deriv=1)
        
        # Calculate the time derivatives
        dx_dt = scipy.signal.savgol_filter(xData, max(3, int(self.samplingFreq*0.01)), 2, delta=1/self.samplingFreq, mode='interp', deriv=1)
        dx_dt2 = scipy.signal.savgol_filter(xData, max(3, int(self.samplingFreq*0.01)), 2, delta=1/self.samplingFreq, mode='interp', deriv=2)
        # Calculate Peak Shape parameters
        speed = np.sqrt(dx_dt**2 + firstDeriv**2)
        curvature = np.abs((dx_dt2*firstDeriv - dx_dt*secondDeriv)) / speed**3  # Units 1/Volts
        
        # Return the Peak Analyis Equations
        return firstDeriv, secondDeriv, thirdDeriv, curvature
    
    def getDerivPeaks(self, firstDeriv, secondDeriv, thirdDeriv, peakInd):
        # Get velocity peaks
        leftVelMax = np.argmax(firstDeriv[:peakInd])
        rightVelMin = self.universalMethods.findNearbyMinimum(firstDeriv, peakInd, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(firstDeriv))
        # Organize velocity peaks
        velInds = [leftVelMax, rightVelMin]
        
        # Find Acceleration peaks
        leftAccelMax = self.universalMethods.findNearbyMaximum(secondDeriv, leftVelMax, binarySearchWindow = -max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(secondDeriv))
        leftAccelMin = self.universalMethods.findNearbyMinimum(secondDeriv, leftVelMax, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(secondDeriv))
        rightAccelMax = self.universalMethods.findNearbyMaximum(secondDeriv, rightVelMin, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(secondDeriv))            
        # Organize acceleration peaks
        accelInds = [leftAccelMax, leftAccelMin, rightAccelMax]
        
        # Find end of peak
        startEndSearchInd = self.universalMethods.findNearbyMinimum(secondDeriv, rightAccelMax, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(secondDeriv))            
        peakEndInd = startEndSearchInd + np.argmax(secondDeriv[startEndSearchInd:])
        # Organize peak boundaries
        peakBoundaries = [0, peakEndInd]

        # Find third derivative peaks
        thirdDeriv_leftMin = self.universalMethods.findNearbyMinimum(thirdDeriv, leftAccelMin, binarySearchWindow = -max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(thirdDeriv))
        thirdDeriv_rightMax = self.universalMethods.findNearbyMaximum(thirdDeriv, leftAccelMin, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(thirdDeriv))
        # Find Third Deriv Right Minimum
        thirdDeriv_rightMin = self.universalMethods.findNearbyMaximum(thirdDeriv, rightVelMin, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(thirdDeriv))
        thirdDeriv_rightMin = self.universalMethods.findNearbyMinimum(thirdDeriv, thirdDeriv_rightMin, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(thirdDeriv))
        # Organize third derivative peaks
        thirdDerivInds = [thirdDeriv_leftMin, thirdDeriv_rightMax, thirdDeriv_rightMin]

        return velInds, accelInds, thirdDerivInds, peakBoundaries
    
    def extractFeatures(self, xData, yData, peakInd, debugBlinkDetection = True):
        if len(xData) < int(self.samplingFreq*0.01):
            if debugBlinkDetection: print("\t\tPeak was around noisy data", len(xData), xData[peakInd])
            return []

        # ---------------------- Baseline Subtraction ---------------------- #
        # Shift the data to zero
        firstDeriv = scipy.signal.savgol_filter(yData, max(3, int(self.samplingFreq*0.01)), 2, delta=1/self.samplingFreq, mode='interp', deriv=1)
        secondDeriv = scipy.signal.savgol_filter(yData, max(3, int(self.samplingFreq*0.01)), 2, delta=1/self.samplingFreq, mode='interp', deriv=2)
        
        firstDeriv_zeroCrossings = self.universalMethods.findPointCrossing(firstDeriv[0:peakInd]/max(firstDeriv[0:peakInd]), threshold = 0.01)
        secondDeriv_zeroCrossings = self.universalMethods.findPointCrossing(secondDeriv[0:peakInd]/max(secondDeriv[0:peakInd]), threshold = 0.01)
            
        # Does the first derivative cross zero -> baseline peak
        if len(firstDeriv_zeroCrossings) != 0:
            peakStartInd = firstDeriv_zeroCrossings[-1] + 1
        # Does the second derivative cross zero -> Closest first derivative to zero
        elif len(secondDeriv_zeroCrossings) != 0:
            peakStartInd = secondDeriv_zeroCrossings[-1] + 1
        # Else, just keep the current baseline
        else:
            peakStartInd = 0
        
        yData = yData[peakStartInd:]
        xData = xData[peakStartInd:]
        peakInd = peakInd - peakStartInd
        # print("Data Start: " + str(xData[0]))
        yData -= yData[0]

        # Normalization to remove amplitudes drift and sensor variations.
        peakHeight = yData[peakInd]
        yData = yData/peakHeight
        # ------------------------------------------------------------------ #
        
        # ---------------- Calculate the Peak's Derivatives ---------------- #
        # Calculate the Derivatives and Curvature
        firstDeriv, secondDeriv, thirdDeriv, curvature = self.quantifyPeakShape(xData, yData, peakInd)
        # Calculate the Peak Derivatives
        velInds, accelInds, thirdDerivInds, peakBoundaries = self.getDerivPeaks(firstDeriv, secondDeriv, thirdDeriv, peakInd)

        debugBlinkDetection = True
        # Cull peaks with bad derivatives: malformed peaks
        if velInds[1] < velInds[0]:
            if debugBlinkDetection: print("\t\tBad Velocity Inds", velInds, xData[peakInd])
            return []
        elif not accelInds[0] < accelInds[1] < accelInds[2]:
            if debugBlinkDetection: print("\t\tBad Acceleration Inds: ", accelInds, xData[peakInd])
            return []
        elif thirdDerivInds[1] < thirdDerivInds[0]:
            if debugBlinkDetection: print("\t\tBad Third Derivative Inds: ", thirdDerivInds, xData[peakInd])
            return []
        elif not accelInds[0] < velInds[0] < accelInds[1]:
            if debugBlinkDetection: print("\t\tBad Derivative Inds Order: First Half")
            if debugBlinkDetection: print("\t\t\t", peakInd, velInds, accelInds, xData[peakInd])
            return []
        elif not peakInd < velInds[1] < accelInds[2]:
            if debugBlinkDetection: print("\t\tBad Derivative Inds Order: Second Half")
            if debugBlinkDetection: print("\t\t\t", peakInd, velInds, accelInds, xData[peakInd])
            return []
        # Cull peaks with improper baseline
        # if not -0.2 < yData[-1] < 0.2:
        #     if debugBlinkDetection: print("\t\tBad")
        #     return []        
        # Cull Noisy Peaks
        #accelInds_Trial = scipy.signal.find_peaks(thirdDeriv[thirdDerivInds[0]:], prominence=10E-20)[0];
        ####################elif
        
        # plt.plot(xData, yData/max(yData), 'k', linewidth=2)
        # plt.plot(xData, firstDeriv*0.8/max(firstDeriv), 'tab:red', linewidth=1)
        # plt.plot(xData, secondDeriv*0.8/max(secondDeriv), 'tab:blue', linewidth=1)
        # plt.plot(xData, thirdDeriv*0.8/max(thirdDeriv), 'purple', alpha = 0.5, linewidth=1)
        
        # secondDeriv = np.array(secondDeriv)
        # thirdDeriv = np.array(thirdDeriv)
        # xData = np.array(xData)
        
        # plt.plot(xData[velInds], (firstDeriv*0.8/max(firstDeriv))[velInds], 'ro', markersize=5)
        # plt.plot(xData[accelInds], (secondDeriv*0.8/max(secondDeriv))[accelInds], 'bo', markersize=5)
        # plt.plot(xData[thirdDerivInds], (thirdDeriv*0.8/max(thirdDeriv))[thirdDerivInds], 'mo', markersize=5)
        # plt.plot(xData[peakBoundaries], (yData/max(yData))[peakBoundaries], 'ko', markersize=5)
        # plt.legend(['Blink', 'firstDeriv', 'secondDeriv', 'thirdDeriv'])
        # # plt.title("Accel Inds = " + str(len(accelInds_Trial)))
        # plt.show()        
        # sepInds = [peakBoundaries[0], accelInds[0], accelInds[1], peakInd, accelInds[2], peakBoundaries[1]]
        # self.plotData(xData, yData, peakInd, velInds = velInds, accelInds = accelInds, sepInds = sepInds, title = "Dividing the Blink")

        # ------------------------------------------------------------------- #
        
        # --------------------- Find the Blink's Endpoints ------------------ #
        # Linearly Fit the Peak's Slope
        upSlopeTangentParams = [firstDeriv[velInds[0]], yData[velInds[0]] - firstDeriv[velInds[0]]*xData[velInds[0]]]
        downSlopeTangentParams = [firstDeriv[velInds[1]], yData[velInds[1]] - firstDeriv[velInds[1]]*xData[velInds[1]]]
        
        # Calculate the New Endpoints of the Peak
        startBlinkX, _ = self.universalMethods.findLineIntersectionPoint([0, 0], upSlopeTangentParams)
        endBlinkX, _ = self.universalMethods.findLineIntersectionPoint(downSlopeTangentParams, [0, 0])
        # Calculate the New Endpoint Locations of the Peak
        startBlinkInd = np.argmin(abs(xData - startBlinkX))
        endBlinkInd = np.argmin(abs(xData - endBlinkX))
        # ------------------------------------------------------------------- #
        
        # ------------------------------------------------------------------- #
        # -------------------- Extract Amplitude Features ------------------- #
        # Find Peak's Tent
        peakTentX, peakTentY = self.universalMethods.findLineIntersectionPoint(upSlopeTangentParams, downSlopeTangentParams)
        # Calculate Tent Deviation Features
        tentDeviationX = peakTentX - xData[peakInd]
        tentDeviationY = peakTentY - yData[peakInd]
        tentDeviationRatio = tentDeviationX/tentDeviationY
        
        # Closing Amplitudes
        maxClosingAccel_Loc = yData[accelInds[0]]
        maxClosingVel_Loc = yData[velInds[0]]
        minBlinkAccel_Loc = yData[accelInds[1]]
        # Opening Amplitudes
        openingAmpVel_Loc = yData[velInds[1]]
        maxOpeningAccel_firstHalfLoc = yData[accelInds[2]]
        maxOpeningAccel_secondHalfLoc = yData[peakBoundaries[1]]
        
        # Closing Amplitude Intervals
        closingAmpSegment1 = maxClosingVel_Loc - maxClosingAccel_Loc
        closingAmpSegment2 = minBlinkAccel_Loc - maxClosingVel_Loc
        closingAmpSegmentFull = minBlinkAccel_Loc - maxClosingAccel_Loc
        # Opening Amplitude Intervals
        openingAmpSegment1 = openingAmpVel_Loc - maxOpeningAccel_firstHalfLoc
        openingAmpSegment2 = maxOpeningAccel_firstHalfLoc - openingAmpVel_Loc
        openingAmplitudeFull = openingAmpVel_Loc - maxOpeningAccel_secondHalfLoc
        
        # Mixed Amplitude Intervals
        velocityAmpInterval = openingAmpVel_Loc - maxClosingVel_Loc
        accelAmpInterval1 = maxOpeningAccel_firstHalfLoc - maxClosingAccel_Loc
        accelAmpInterval2 = maxOpeningAccel_secondHalfLoc - maxClosingAccel_Loc
        
        # Amplitude Ratio
        accel01_AmpRatio = yData[accelInds[0]]/yData[accelInds[1]]
        accel0_Vel0_AmpRatio = yData[accelInds[0]]/yData[velInds[0]]
        velAmpRatio = yData[velInds[0]]/yData[velInds[1]]
        # More Amp Ratios
        accel2_Vel1_AmpRatio = yData[accelInds[2]]/yData[velInds[1]]
        # ------------------------------------------------------------------- #
        
        # -------------------- Extract Duration Features -------------------- #
        # Find the Standard Blink Durations
        blinkDuration = endBlinkX - startBlinkX      # The Total Time of the Blink
        closingTime_Tent = peakTentX - startBlinkX        # The Time for the Eye to Close
        openingTime_Tent = endBlinkX - peakTentX          # The Time for the Eye to Open
        closingTime_Peak = xData[peakInd] - startBlinkX
        openingTime_Peak = endBlinkX - xData[peakInd]          # The Time for the Eye to Open
        # Calculate the Duration Ratios
        closingFraction = closingTime_Peak/blinkDuration
        openingFraction = openingTime_Peak/blinkDuration
        
        # Calculate the Half Amplitude Duration
        blinkAmp50Right = xData[peakInd + np.argmin(abs(yData[peakInd:] - yData[peakInd]/2))]
        blinkAmp50Left = xData[np.argmin(abs(yData[0:peakInd] - yData[peakInd]/2))]
        halfClosedTime = blinkAmp50Right - blinkAmp50Left
        # Calculate Time the Eyes are Closed
        blinkAmp90Right = xData[peakInd + np.argmin(abs(yData[peakInd:] - yData[peakInd]*0.9))]
        blinkAmp90Left = xData[np.argmin(abs(yData[0:peakInd] - yData[peakInd]*0.9))]
        eyesClosedTime = blinkAmp90Right - blinkAmp90Left
        # Caluclate Percent Closed
        percentTimeEyesClosed = eyesClosedTime/halfClosedTime
        
        # Divide the Peak by Acceleration
        startToAccel = xData[accelInds[0]] - startBlinkX
        accelClosingDuration = xData[accelInds[1]] - xData[accelInds[0]]
        accelToPeak = xData[peakInd] - xData[accelInds[1]]
        peakToAccel = xData[accelInds[2]] - xData[peakInd]
        accelOpeningPeakDuration = xData[peakBoundaries[1]] - xData[accelInds[2]]
        accelToEnd = endBlinkX - xData[peakBoundaries[1]]
        # Divide the Peak by Velocity
        velocityPeakInterval = xData[velInds[1]] - xData[velInds[0]]
        startToVel = xData[velInds[0]] - startBlinkX
        velToPeak = xData[peakInd] - xData[velInds[0]]
        peakToVel = xData[velInds[1]] - xData[peakInd]
        velToEnd = endBlinkX - xData[velInds[1]]
        
        # Mixed Durations: Accel and Vel
        portion2Duration = xData[velInds[0]] - xData[accelInds[0]]
        portion3Duration = xData[accelInds[1]] - xData[velInds[0]]
        portion6Duration = xData[accelInds[2]] - xData[velInds[1]]
        # Mixed Accel Durations
        accel12Duration = xData[accelInds[1]] - xData[accelInds[2]]
        condensedDuration1 = xData[accelInds[2]] - xData[accelInds[0]]
        condensedDuration2 = xData[peakBoundaries[1]] - xData[accelInds[0]]
        
        # New Half Duration
        durationByVel1 = xData[peakInd:][np.argmin(abs(yData[peakInd:] - yData[velInds[0]]))] - xData[velInds[0]]
        durationByVel2 = xData[velInds[1]] - xData[0:peakInd][np.argmin(abs(yData[0:peakInd] - yData[velInds[1]]))]
        durationByAccel1 = xData[peakInd:][np.argmin(abs(yData[peakInd:] - yData[accelInds[0]]))] - xData[accelInds[0]]
        durationByAccel2 = xData[peakInd:][np.argmin(abs(yData[peakInd:] - yData[accelInds[1]]))] - xData[accelInds[1]]
        durationByAccel3 = xData[accelInds[2]] - xData[0:peakInd][np.argmin(abs(yData[0:peakInd] - yData[accelInds[2]]))]
        durationByAccel4 = xData[peakBoundaries[1]] - xData[0:peakInd][np.argmin(abs(yData[0:peakInd] - yData[peakBoundaries[1]]))]
        midDurationRatio = durationByVel1/durationByVel2
        # ------------------------------------------------------------------- #
        
        # ------------------- Extract Derivative Features ------------------- #
        # Extract Closing Slopes
        closingSlope_MaxAccel = firstDeriv[accelInds[0]]
        closingSlope_MaxVel = firstDeriv[velInds[0]]
        closingSlope_MinAccel = firstDeriv[accelInds[1]]
        # Extract Opening Slopes
        openingSlope_MinVel = firstDeriv[velInds[1]]
        openingSlope_MaxAccel1 = firstDeriv[accelInds[2]]
        
        # Extract Closing Accels
        closingAccel_MaxAccel = secondDeriv[accelInds[0]]
        closingAccel_MinAccel = secondDeriv[accelInds[1]]
        # Extract Opening Accels
        openingAccel_MaxAccel1 = secondDeriv[accelInds[2]]
        openingAccel_MaxAccel2 = secondDeriv[peakBoundaries[1]]
        
        # Ratios with Accel0
        accel0_Vel0_Ratio = secondDeriv[accelInds[0]]/firstDeriv[velInds[0]]
        accel0_Accel1_Ratio = secondDeriv[accelInds[0]]/secondDeriv[accelInds[1]]
        accel0_Peak_Ratio = secondDeriv[accelInds[0]]/yData[peakInd]
        accel0_Vel1_Ratio = secondDeriv[accelInds[0]]/firstDeriv[velInds[1]]
        accel0_Accel2_Ratio = secondDeriv[accelInds[0]]/secondDeriv[accelInds[2]]
        
        # Ratios with Vel0
        vel0_Accel1_Ratio = firstDeriv[velInds[0]]/secondDeriv[accelInds[2]]        
        vel0_Peak_Ratio = firstDeriv[velInds[0]]/yData[peakInd]
        velRatio = firstDeriv[velInds[0]]/firstDeriv[velInds[1]]
        
        # Ratios with Accel1
        accel1_Peak_Ratio = secondDeriv[accelInds[1]]/yData[peakInd]
        accel1_Vel1_Ratio = secondDeriv[accelInds[1]]/firstDeriv[velInds[1]]
        accel1_Accel2_Ratio = secondDeriv[accelInds[1]]/secondDeriv[accelInds[2]]
        
        # Ratios with Vel1
        vel1_Peak_Ratio = firstDeriv[velInds[1]]/yData[peakInd]
        vel1_Accel2_Ratio = firstDeriv[velInds[1]]/secondDeriv[accelInds[2]] 
        
        # Ratios with Accel2
        accelRatio2 = secondDeriv[accelInds[2]]/secondDeriv[peakBoundaries[1]]
        # ------------------------------------------------------------------- #
        
        # ------------------- Extract Integral Features ------------------- #
        # Peak Integral
        blinkIntegral = scipy.integrate.simpson(yData[startBlinkInd:endBlinkInd], xData[startBlinkInd:endBlinkInd])
        portion2Integral = scipy.integrate.simpson(yData[accelInds[0]:velInds[0]], xData[accelInds[0]:velInds[0]])
        portion3Integral = scipy.integrate.simpson(yData[velInds[0]:accelInds[1]], xData[velInds[0]:accelInds[1]])
        if peakInd == accelInds[1]:
            portion4Integral = 0
        else:
            indexOptions = [accelInds[1], peakInd]
            portion4Integral = (min(indexOptions) == accelInds[1]) * scipy.integrate.simpson(yData[min(indexOptions):max(indexOptions)], xData[min(indexOptions):max(indexOptions)])
        portion5Integral = scipy.integrate.simpson(yData[peakInd:accelInds[2]], xData[peakInd:accelInds[2]])
        portion6Integral = scipy.integrate.simpson(yData[velInds[1]: accelInds[2]], xData[velInds[1]: accelInds[2]])
        portion7Integral = scipy.integrate.simpson(yData[accelInds[2]:peakBoundaries[1]], xData[accelInds[2]:peakBoundaries[1]])
        
        # Other Integrals
        velToVelIntegral = scipy.integrate.simpson(yData[velInds[0]:velInds[1]], xData[velInds[0]:velInds[1]])
        closingIntegral = scipy.integrate.simpson(yData[startBlinkInd:peakInd], xData[startBlinkInd:peakInd])
        openingIntegral = scipy.integrate.simpson(yData[peakInd:endBlinkInd], xData[peakInd:endBlinkInd])
        closingSlopeIntegral = scipy.integrate.simpson(yData[accelInds[0]:accelInds[1]], xData[accelInds[0]:accelInds[1]])
        accel12Integral = scipy.integrate.simpson(yData[accelInds[1]:accelInds[2]], xData[accelInds[1]:accelInds[2]])
        openingAccelIntegral = scipy.integrate.simpson(yData[accelInds[2]:peakBoundaries[1]], xData[accelInds[2]:peakBoundaries[1]])
        condensedIntegral = scipy.integrate.simpson(yData[accelInds[0]:peakBoundaries[1]], xData[accelInds[0]:peakBoundaries[1]])
        peakToVel0Integral = scipy.integrate.simpson(yData[velInds[0]:peakInd], xData[velInds[0]:peakInd])
        peakToVel1Integral = scipy.integrate.simpson(yData[peakInd:velInds[1]], xData[peakInd:velInds[1]])
        # ------------------------------------------------------------------- #

        # ---------------------- Extract Shape Features --------------------- #
        fullBlink = yData[startBlinkInd:endBlinkInd].copy()
        fullBlink -= min(fullBlink)
        fullBlink = fullBlink/max(fullBlink)
        # Calculate Peak Shape Parameters
        peakAverage = np.mean(fullBlink)
        peakEntropy = scipy.stats.entropy(fullBlink +10E-10)
        peakSkew = scipy.stats.skew(fullBlink, bias=False)
        
        peakCurvature = curvature[peakInd]
        # Curvature Around Main Points
        curvatureYDataAccel0 = curvature[accelInds[0]]
        curvatureYDataAccel1 = curvature[accelInds[1]]
        curvatureYDataAccel2 = curvature[accelInds[2]]
        curvatureYDataAccel3 = curvature[peakBoundaries[1]]
        
        # Stanard Deviation
        velFullSTD = np.std(firstDeriv[startBlinkInd:endBlinkInd], ddof=1)
        accelFullSTD = np.std(secondDeriv[startBlinkInd:endBlinkInd], ddof=1)
        thirdDerivFullSTD = np.std(thirdDeriv[startBlinkInd:endBlinkInd], ddof=1)
        
        # scipy.stats.entropy
        velFullEntropy = scipy.stats.entropy(firstDeriv[startBlinkInd:endBlinkInd] - min(firstDeriv[startBlinkInd:endBlinkInd]) + 10E-10)
        accelFullEntropy = scipy.stats.entropy(secondDeriv[startBlinkInd:endBlinkInd] - min(secondDeriv[startBlinkInd:endBlinkInd]) + 10E-10)
        thirdDerivFullEntropy = scipy.stats.entropy(thirdDeriv[startBlinkInd:endBlinkInd] - min(thirdDeriv[startBlinkInd:endBlinkInd]) + 10E10)
        # ------------------------------------------------------------------- #

        # ------------------------- Cull Bad Blinks ------------------------- #
        debugBlinkDetection = True
        # Blinks are on average 100-400 ms. They can be on the range of 50-500 ms.
        
        # if not -226.08898051797456 < accel1_Accel2_Ratio:
        #     if debugBlinkDetection: print("		Bad accel1_Accel2_Ratio:", accel1_Accel2_Ratio, xData[peakInd])
        #     return [None]
        # if not 13.863705691193731 < closingSlope_MaxVel < 27.503764452400503:
        #     if debugBlinkDetection: print("		Bad closingSlope_MaxVel:", closingSlope_MaxVel, xData[peakInd])
        #     return [None]
        # if not eyesClosedTime < 0.2365096000000122:
        #     if debugBlinkDetection: print("		Bad eyesClosedTime:", eyesClosedTime, xData[peakInd])
        #     return [None]
        # if not 0.10229590503286148 < blinkDuration < 0.3931081285961759:
        #     if debugBlinkDetection: print("		Bad blinkDuration:", blinkDuration, xData[peakInd])
        #     return [None]
        # if not 3.469075904236875 < accelFullEntropy < 6.482192654566428:
        #     if debugBlinkDetection: print("		Bad accelFullEntropy:", accelFullEntropy, xData[peakInd])
        #     return [None]

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
        
        # if not 3.300450074015746 < velFullEntropy < 6.256531740100211:
        #     if debugBlinkDetection: print("\t\tBad Velocity Entropy:", velFullEntropy, xData[peakInd])
        #     return [None]
        
        # if not 0.04548315778346813 < closingTime_Peak < 0.08833519447480345:
        #     if debugBlinkDetection: print("\t\tBad closing Time:", closingTime_Peak, xData[peakInd])
        #     return [None]
        
        # if not 3.5154952272113817 < peakEntropy:
        #     if debugBlinkDetection: print("\t\tBad Peak Entropy:", peakEntropy, xData[peakInd])
        #     return [None]
        
        # if not 13.16455216938455 < closingSlope_MaxVel < 28.134315639215053:
        #     if debugBlinkDetection: print("\t\tBad Closing Slope:", closingSlope_MaxVel, xData[peakInd])
        #     return [None]
        
        # if not openingAccel_MaxAccel1 < 635.4681110053752:
        #     if debugBlinkDetection: print("\t\tBad Opening Accel", openingAccel_MaxAccel1, xData[peakInd])
        #     return [None]
    
        # if not -1614.7736375258792 < closingAccel_MinAccel < -232.38427184908272:
        #     if debugBlinkDetection: print("\t\tBad Closing Acceleration:", closingAccel_MinAccel, xData[peakInd])
        #     return [None]
        
        # if not -4.0152214059393785 < velRatio < -1.0509541558696633:
        #     if debugBlinkDetection: print("\t\tBad Velocity Ratio:", velRatio, xData[peakInd])
        #     return [None]
        
        # if not accel0_Vel0_Ratio < 42.5036011036511:
        #     if debugBlinkDetection: print("\t\tBad Acceleration_0 Velocity_0 Ratio:", accel0_Vel0_Ratio, xData[peakInd])
        #     return [None]
        
        # if not blinkDuration < 0.3326133725108635:
        #     if debugBlinkDetection: print("\t\tBad Blink Duration:", blinkDuration, xData[peakInd])
        #     return [None]
        
        # if not 0.07824406868662209 < tentDeviationY:
        #     if debugBlinkDetection: print("\t\tBad Tent Deviation Y:", tentDeviationY, xData[peakInd])
        #     return [None]
        
        # if not peakSkew < 0.2751274668208018:
        #     if debugBlinkDetection: print("\t\tBad Peak Skew", peakSkew, xData[peakInd])
        #     return [None]
        
        # if not accelFullSTD < 881.8338490596707:
        #     if debugBlinkDetection: print("\t\tBad Acceleration STD", accelFullSTD, xData[peakInd])
        #     return [None]
    
        # if not -1614.7736375258792 < closingAccel_MinAccel:
        #     if debugBlinkDetection: print("\t\tBad Closing Acceleration:", closingAccel_MinAccel, xData[peakInd])
        #     return []
        
        # if not -4.0152214059393785 < velRatio < -1.0509541558696633:
        #     if debugBlinkDetection: print("\t\tBad Velocity Ratio:", velRatio, xData[peakInd])
        #     return []
        
        # if not accel0_Vel0_Ratio < 42.5036011036511:
        #     if debugBlinkDetection: print("\t\tBad Acceleration_0 Velocity_0 Ratio:", accel0_Vel0_Ratio, xData[peakInd])
        #     return []
        
        # if not blinkDuration < 0.3326133725108635:
        #     if debugBlinkDetection: print("\t\tBad Blink Duration:", blinkDuration, xData[peakInd])
        #     return []
        
        # if not 0.07824406868662209 < tentDeviationY:
        #     if debugBlinkDetection: print("\t\tBad Tent Deviation Y:", tentDeviationY, xData[peakInd])
        #     return []
        
        # if not peakSkew < 0.2751274668208018:
        #     if debugBlinkDetection: print("\t\tBad Peak Skew", peakSkew, xData[peakInd])
        #     return []
        
        # if not 67.5513354051822 < accelFullSTD < 881.8338490596707:
        #     if debugBlinkDetection: print("\t\tBad Acceleration STD", accelFullSTD, xData[peakInd])
        #     return []
        
        # if not closingAccel_MinAccel > -2250:
        #     if debugBlinkDetection: print("\t\tBad Closing Acceleration:", closingAccel_MinAccel, xData[peakInd])
        #     return []
        
        # if not -3.25 < velRatio < -1:
        #     if debugBlinkDetection: print("\t\tBad Velocity Ratio:", velRatio, xData[peakInd])
        #     return []
        
        # if not 15 < accel0_Vel0_Ratio < 40:
        #     if debugBlinkDetection: print("\t\tBad Acceleration_0 Velocity_0 Ratio:", accel0_Vel0_Ratio, xData[peakInd])
        #     return []
        
        # if not 0.09 < blinkDuration < 0.3:
        #     if debugBlinkDetection: print("\t\tBad Blink Duration:", blinkDuration, xData[peakInd])
        #     return []
        
        # if not 12 < closingSlope_MaxVel < 27:
        #     if debugBlinkDetection: print("\t\tBad Closing Slope Velocity", closingSlope_MaxVel, xData[peakInd])
        #     return []
    
        # if not peakSkew < 0.25:
        #     if debugBlinkDetection: print("\t\tBad Peak Skew", peakSkew, xData[peakInd])
        #     return []
        
        # if not accelFullSTD < 1000:
        #     if debugBlinkDetection: print("\t\tBad Acceleration STD", accelFullSTD, xData[peakInd])
        #     return []
        
        # if not 0.008 < blinkDuration < 0.5:
        #     if debugBlinkDetection: print("\t\tBad Blink Duration:", blinkDuration, xData[peakInd])
        #     return []
        # if tentDeviationX < -0.2:
        #     if debugBlinkDetection: print("\t\tBad Blink tentDeviationX:", tentDeviationX, xData[peakInd])
        #     return []
        # if not -0.2 < tentDeviationY < 0.6:
        #     if debugBlinkDetection: print("\t\tBad Blink tentDeviationY:", tentDeviationY, xData[peakInd])
        #     return []
        # # If the Closing Time is Longer Than 150ms, it is Probably Not an Involuntary Blink
        # elif not 0.04 < closingTime_Peak < 0.3:
        #     if debugBlinkDetection: print("\t\tBad Closing Time:", closingTime_Peak, xData[peakInd])
        #     return []
        # elif not 0.04 < openingTime_Peak < 0.4:
        #     if debugBlinkDetection: print("\t\tBad Opening Time:", openingTime_Peak, xData[peakInd])
        #     return []
        # elif -1 < velRatio:
        #     if debugBlinkDetection: print("\t\tBad velRatio:", velRatio)
        #     return []
        # elif peakSkew < -0.75:
        #     if debugBlinkDetection: print("\t\tBad peakSkew:", peakSkew)
        #     return []
        
        
        # elif 15 < closingSlope_MinAccel or closingSlope_MinAccel < 0:
        #     if debugBlinkDetection: print("\t\tBad closingSlope2:", closingSlope_MinAccel)
        #     return []
        # elif 8 < closingSlopeDiff10:
        #     if debugBlinkDetection: print("\t\tBad closingSlopeDiff10:", closingSlopeDiff10)
        #     return []  
        # elif 15 < closingSlopeDiff12:
        #     if debugBlinkDetection: print("\t\tBad closingSlopeDiff12:", closingSlopeDiff12)
        #     return []
        # elif openingSlope1 < -8:
        #     if debugBlinkDetection: print("\t\tBad openingSlope1:", openingSlope1)
        #     return []     
        # elif openingSlope2 < -14:
        #     if debugBlinkDetection: print("\t\tBad openingSlope2:", openingSlope2)
        #     return [] 
        
        # accelClosedVal1: Most data past 0.00015 is bad. No Min threshold
        # elif 0.000175 < accelClosedVal1:
        #     if debugBlinkDetection: print("\t\tBad accelClosedVal1:", accelClosedVal1)
        #     return []
        # accelClosedVal2: Most data past 0.000175 is bad. No Min threshold
        # elif 0.000175 < accelClosedVal2:
        #     if debugBlinkDetection: print("\t\tBad accelClosedVal2:", accelClosedVal2)
        #     return []
        # accelOpenVal1: Most data past 0.000175 is bad. No Min threshold
        # elif 0.000175 < accelOpenVal1:
        #     if debugBlinkDetection: print("\t\tBad accelOpenVal1:", accelOpenVal1)
        #     return []
        # # accelOpenVal2: Most data past 0.000175 is bad. No Min threshold
        # elif 0.000175 < accelOpenVal2:
        #     if debugBlinkDetection: print("\t\tBad accelOpenVal2:", accelOpenVal2)
        #     return []
        # elif accelToPeak < 0:
        #     if debugBlinkDetection: print("\t\tBad accelToPeak:", accelToPeak)
        #     return []    
        # elif 0.7 < openingAmpAccel1:
        #     if debugBlinkDetection: print("\t\tBad openingAmpAccel1:", openingAmpAccel1)
        #     return []  
        # elif 0.7 < openingAmpVel:
        #     if debugBlinkDetection: print("\t\tBad openingAmpVel:", openingAmpVel)
        #     return [] 
        # elif 0.6 < openingAmpAccel2:
        #     if debugBlinkDetection: print("\t\tBad openingAmpAccel2:", openingAmpAccel2)
        #     return [] 

        
    
        # elif 0.005 < velOpenVal:
        #     if debugBlinkDetection: print("\t\tBad velOpenVal:", velOpenVal)
        #     return []
        # elif 0.0006 < peakClosingAccel1:
        #     if debugBlinkDetection: print("\t\tBad peakClosingAccel1:", peakClosingAccel1)
        #     return []
        # elif 0.0006 < peakClosingAccel2:
        #     if debugBlinkDetection: print("\t\tBad peakClosingAccel2:", peakClosingAccel2)
        #     return [] 
        # elif 0.016 < peakOpeningVel:
        #     if debugBlinkDetection: print("\t\tBad peakOpeningVel:", peakOpeningVel)
        #     return [] 

        # elif 1500 < peakCurvature:
        #     if debugBlinkDetection: print("\t\tBad maxCurvature:", peakCurvature)
        #     return []         
        

        # ------------------------------------------------------------------- #

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
        
        # elif 6E-4 < accelOpenVal1:
        #     if debugBlinkDetection: print("\t\tWINK! accelOpenVal1:", accelOpenVal1); return ["Wink"]
        # elif 3E-4 < accelOpenVal2:
        #     if debugBlinkDetection: print("\t\tWINK! accelOpenVal2:", accelOpenVal2); return ["Wink"]
        # elif closingSlope0 > 15:
        #     if debugBlinkDetection: print("\t\tWINK! closingSlope0 = ", closingSlope0); return ["Wink"]  
        # elif closingSlope1 > 15:
        #     if debugBlinkDetection: print("\t\tWINK! closingSlope1 = ", closingSlope1); return ["Wink"]  
        # elif closingSlope2 > 13:
        #     if debugBlinkDetection: print("\t\tWINK! closingSlope2 = ", closingSlope2); return ["Wink"] 
        # elif openingSlope1 < -7:
        #     if debugBlinkDetection: print("\t\tWINK! openingSlope1 = ", openingSlope1); return ["Wink"]  
        # elif openingSlope2 < -10:
        #     if debugBlinkDetection: print("\t\tWINK! openingSlope2 = ", openingSlope2); return ["Wink"]  
        # elif openingSlope3 < -7:
        #     if debugBlinkDetection: print("\t\tWINK! openingSlope3 = ", openingSlope3); return ["Wink"]  
        # elif 4 < closingSlopeDiff10:
        #     if debugBlinkDetection: print("\t\tWINK! closingSlopeDiff10 = ", closingSlopeDiff10); return ["Wink"]  
        # elif 31 < closingSlopeRatio1:
        #     if debugBlinkDetection: print("\t\tWINK! closingSlopeRatio1 = ", closingSlopeRatio1); return ["Wink"]  
        # elif -4.5 > openingSlopeDiff23:
        #     if debugBlinkDetection: print("\t\tWINK! openingSlopeDiff23:", openingSlopeDiff23); return ["Wink"]
        # elif -19 > openingSlopeRatio2:
        #     if debugBlinkDetection: print("\t\tWINK! openingSlopeRatio2:", openingSlopeRatio2); return ["Wink"]
        # elif maxCurvature > 750:
        #     if debugBlinkDetection: print("\t\tWINK! maxCurvature = ", maxCurvature); 
        # elif 0.001 < peakOpeningAccel2:
        #     if debugBlinkDetection: print("\t\tWINK! peakOpeningAccel2:", peakOpeningAccel2); return ["Wink"]
        # elif closingSlopeDiff12 > 12 or closingSlopeDiff12 < -1:
        #     if debugBlinkDetection: print("\t\tWINK! closingSlopeDiff12:", closingSlopeDiff12); return ["Wink"]
        # ------------------------------------------------------------------- #
        
        # ------------------ Consolidate the Blink Features ----------------- #
        finalFeatures = []
        # Organize Amplitude Features        
        finalFeatures.extend([peakHeight, tentDeviationX, tentDeviationY, tentDeviationRatio])
        finalFeatures.extend([maxClosingAccel_Loc, maxClosingVel_Loc, minBlinkAccel_Loc, openingAmpVel_Loc, maxOpeningAccel_firstHalfLoc, maxOpeningAccel_secondHalfLoc])
        finalFeatures.extend([closingAmpSegment1, closingAmpSegment2, closingAmpSegmentFull, openingAmpSegment1, openingAmpSegment2, openingAmplitudeFull])
        finalFeatures.extend([velocityAmpInterval, accelAmpInterval1, accelAmpInterval2])  
        finalFeatures.extend([accel01_AmpRatio, accel0_Vel0_AmpRatio, velAmpRatio, accel2_Vel1_AmpRatio])

        
        # Organize Duration Features
        finalFeatures.extend([blinkDuration, closingTime_Tent, openingTime_Tent, closingTime_Peak, openingTime_Peak, closingFraction, openingFraction])
        finalFeatures.extend([halfClosedTime, eyesClosedTime, percentTimeEyesClosed])
        finalFeatures.extend([startToAccel, accelClosingDuration, accelToPeak, peakToAccel, accelOpeningPeakDuration, accelToEnd])
        finalFeatures.extend([velocityPeakInterval, startToVel, velToPeak, peakToVel, velToEnd])
        finalFeatures.extend([portion2Duration, portion3Duration, portion6Duration])
        finalFeatures.extend([accel12Duration, condensedDuration1, condensedDuration2])
        finalFeatures.extend([durationByVel1, durationByVel1, durationByAccel1, durationByAccel2, durationByAccel3, durationByAccel4, midDurationRatio])

        # Organize Derivative Features
        finalFeatures.extend([closingSlope_MaxAccel, closingSlope_MaxVel, closingSlope_MinAccel, openingSlope_MinVel, openingSlope_MaxAccel1])
        finalFeatures.extend([closingAccel_MaxAccel, closingAccel_MinAccel, openingAccel_MaxAccel1, openingAccel_MaxAccel2])
        finalFeatures.extend([accel0_Vel0_Ratio, accel0_Accel1_Ratio, accel0_Peak_Ratio, accel0_Vel1_Ratio, accel0_Accel2_Ratio])
        finalFeatures.extend([vel0_Accel1_Ratio, vel0_Peak_Ratio, velRatio])
        finalFeatures.extend([accel1_Peak_Ratio, accel1_Vel1_Ratio, accel1_Accel2_Ratio])
        finalFeatures.extend([vel1_Peak_Ratio, vel1_Accel2_Ratio, accelRatio2])
        
        # Organize Integral Features
        finalFeatures.extend([blinkIntegral, portion2Integral, portion3Integral, portion4Integral, portion5Integral, portion6Integral, portion7Integral])
        finalFeatures.extend([velToVelIntegral, closingIntegral, openingIntegral, closingSlopeIntegral, accel12Integral, openingAccelIntegral, condensedIntegral, peakToVel0Integral, peakToVel1Integral])
        
        # Organize Shape Features
        finalFeatures.extend([peakAverage, peakEntropy, peakSkew])
        finalFeatures.extend([peakCurvature, curvatureYDataAccel0, curvatureYDataAccel1, curvatureYDataAccel2, curvatureYDataAccel3])
        finalFeatures.extend([velFullSTD, accelFullSTD, thirdDerivFullSTD, velFullEntropy, accelFullEntropy, thirdDerivFullEntropy])
        
        # ------------------------------------------------------------------- #
        
        if False:
            sepInds = [startBlinkInd, accelInds[0], accelInds[1], peakInd, accelInds[2], endBlinkInd]
            self.plotData(xData, yData, peakInd, velInds = velInds, accelInds = accelInds, sepInds = sepInds, title = "Dividing the Blink")

            # plt.plot(xData, yData/max(yData), 'k', linewidth=2)
            # plt.plot(xData, firstDeriv*0.8/max(firstDeriv), 'r', linewidth=1)
            # plt.plot(xData, secondDeriv*0.8/max(secondDeriv), 'b', linewidth=1)
            # plt.plot(xData, thirdDeriv*0.8/max(thirdDeriv), 'm', alpha = 0.5, linewidth=1)
            # plt.legend(['Blink', 'firstDeriv', 'secondDeriv', 'thirdDeriv'])
            # plt.show()
        
      
        return finalFeatures
        # ------------------------------------------------------------------ #

    def findTraileringAverage(self, recentData, deviationThreshold = 0.08):
        # Base Case in No Points Came in
        if len(recentData) == 0:
            return self.steadyStateEye
        
        # Keep Track of the trailingAverage
        trailingAverage = recentData[-1]
        for dataPointInd in range(2, len(recentData)-1, -2):
            # Get New dataPoint from the Back of the List
            dataPoint = recentData[len(recentData) - dataPointInd]
            # If the dataPoint is Different from the trailingAverage by some Threshold, return the trailingAverage
            if abs(dataPoint - trailingAverage) > deviationThreshold:
                return trailingAverage
            else:
                trailingAverage = (trailingAverage*(dataPointInd - 1) + dataPoint)/(dataPointInd)
        # Else Return the Average
        return trailingAverage

    def sigmoid(self, x, k, x0):       
        # Prediction Model
        return 1.0 / (1 + np.exp(-k * (x - x0)))

    def line(self, x, A, B):
        return A*x + B
    
    def fitCalibration(self, xData, yData, channelIndexCalibrating, plotFit = False):
        # Fit the curve
        popt, pcov = scipy.optimize.curve_fit(self.line, xData, yData)
        estimated_k, estimated_x0 = popt
        # Save Calibration
        self.predictEyeAngle[channelIndexCalibrating] = lambda x: self.line(x, estimated_k, estimated_x0)
        
        # Plot the Fit Results
        if plotFit:
            # Get Model's Data
            xTest = np.arange(min(xData) - 10, max(xData) + 10, 0.01)
            yTest = self.predictEyeAngle[channelIndexCalibrating](xTest)
        
            # Create the Plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # Plot the Data      
            ax.plot(xTest, yTest, '--', label='fitted')
            ax.plot(xData, yData, '-', label='true')  
            # Add Legend and Show
            ax.legend()
            plt.show()
        
        
    def plotData(self, xData, yData, peakInd, velInds = [], accelInds = [], sepInds = [], title = "", peakSize = 5, lineWidth = 2, lineColor = "black", ax = None, axisLimits = []):
        xData = np.array(xData); yData = np.array(yData)
        # Create Figure
        showFig = False
        if ax == None:
            plt.figure()
            ax = plt.gca()
            showFig = True
        # Plot the Data
        ax.plot(xData, yData, linewidth = lineWidth, color = lineColor)
        ax.plot(xData[peakInd], yData[peakInd], 'o', c='tab:purple', markersize=int(peakSize*1.5))
        ax.plot(xData[velInds], yData[velInds], 'o', c='tab:red', markersize=peakSize)
        ax.plot(xData[accelInds], yData[accelInds], 'o', c='tab:blue', markersize=peakSize)
        if len(sepInds) > 0:
            sectionColors = ['red','orange', 'blue','green', 'black']
            for groupInd in range(len(sectionColors)):
                if sepInds[groupInd] in [np.nan, None] or sepInds[groupInd+1] in [np.nan, None]: 
                    continue
                ax.fill_between(xData[sepInds[groupInd]:sepInds[groupInd+1]+1], min(yData), yData[sepInds[groupInd]:sepInds[groupInd+1]+1], color=sectionColors[groupInd], alpha=0.15)
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


