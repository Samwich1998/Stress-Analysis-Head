# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:17:05 2021
    conda install -c conda-forge ffmpeg

@author: Samuel Solomon
"""

# Basic Modules
import sys
import math
import numpy as np
# Peak Detection
import scipy
import scipy.signal
# High/Low Pass Filters
from scipy.signal import butter
# Calibration Fitting
from scipy.optimize import curve_fit
# Plotting
import matplotlib
import matplotlib.pyplot as plt
# Feature Extraction
from scipy.stats import skew
from scipy import interpolate
from scipy.stats import kurtosis
from BaselineRemoval import BaselineRemoval

from scipy.stats import entropy
import heapq

# --------------------------------------------------------------------------- #
# ------------------ User Can Edit (Global Variables) ----------------------- #

class eogProtocol:
    
    def __init__(self, numTimePoints = 3000, moveDataFinger = 10, numChannels = 2, samplingFreq = 800, plotStreamedData = True):
        
        # Input Parameters
        self.numChannels = numChannels            # Number of Bioelectric Signals
        self.numTimePoints = numTimePoints        # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger      # The Amount of Data to Stream in Before Finding Peaks
        self.plotStreamedData = plotStreamedData  # Plot the Data
        # Calibration Angles
        self.calibrationAngles = [[-45, 0, 45] for _ in range(self.numChannels)]
        self.calibrationVoltages = [[] for _ in range(self.numChannels)]
        
        # High Pass Filter Parameters
        self.samplingFreq = samplingFreq          # Depends on the User's Hardware
        self.cutOffFreq = 25                      # Optimal LPF 6-8 Hz (Max 35 or 50); literature Claimed 7 Hz is Best
        
        # Data Collection Parameters
        self.voltagePositionBuffer = 50   # Buffer to Find the Average Voltage
        self.minVoltageMovement = 0.05    # Min Voltage Change Threshold to Move the Gaze
        self.bandPassBuffer = 5000        # Buffer in the Filtered Data that Represented BAD Filtering
        
        # Eye Gesture Prediction Parameters
        self.predictEyeAngleGap = 5
        self.minVoltageThreshold = 0.05
        self.steadyStateEye = 3.3/2
        
        # Pointers for Calibration
        self.calibrateChannelNum = 0
        self.channelCalibrationPointer = 0
        # Calibration Function for Eye Angle
        self.predictEyeAngle = [lambda x: (x - self.steadyStateEye)*30]*self.numChannels
        
        # Check to See if Parameters Make Sense
        self.checkParams
                
        # Start with Fresh Inputs
        self.resetGlobalVariables()
        
        # Define Class for Plotting Peaks
        if plotStreamedData:
            # Initialize Plots
            matplotlib.use('Qt5Agg') # Set Plotting GUI Backend            
            self.initPlotPeaks()    # Create the Plots

    def resetGlobalVariables(self):
        # Data to Read in
        self.data = {'timePoints':[]}
        for channelIndex in range(self.numChannels):
            self.data['Channel'+str(1+channelIndex)] = []
        
        # Hold Past Information
        self.trailingAverageData = {}
        for channelIndex in range(self.numChannels):
            self.trailingAverageData[channelIndex] = [0]*self.numTimePoints
        
        # Reset Last Eye Voltage (Volts)
        self.currentEyeVoltages = [self.steadyStateEye for _ in range(self.numChannels)]
        # Reset Feature Extraction
        self.blinkFeatures = []
        # Reset Blink Indices
        self.singleBlinksX = []
        self.multipleBlinksX = []
        self.blinksXLocs = []
        self.blinksYLocs = []
        
        self.importantArrays = []
        
        # Close Any Opened Plots
        if self.plotStreamedData:
            plt.close()
    
    def checkParams(self):
        if self.moveDataFinger > self.numTimePoints:
            print("You are Analyzing Too Much Data in a Batch. 'moveDataFinger' MUST be Less than 'numTimePoints'")
            sys.exit()

    def initPlotPeaks(self): 
        
        # use ggplot style for more sophisticated visuals
        plt.style.use('seaborn-poster')
        
        # ------------------------------------------------------------------- #
        # --------- Plot Variables user Can Edit (Global Variables) --------- #

        # Specify Figure aesthetics
        figWidth = 14; figHeight = 10;
        self.fig, axes = plt.subplots(self.numChannels, 2, sharey=False, sharex = 'col', figsize=(figWidth, figHeight))
        
        # Plot the Raw Data
        yLimLow = 0; yLimHigh = 3.5; 
        self.bioelectricDataPlots = []; self.bioelectricPlotAxes = []
        for channelIndex in range(self.numChannels):
            # Create Plots
            self.bioelectricPlotAxes.append(axes[channelIndex, 0])
            
            # Generate Plot
            self.bioelectricDataPlots.append(self.bioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            
            # Set Figure Limits
            self.bioelectricPlotAxes[channelIndex].set_ylim(yLimLow, yLimHigh)
            # Label Axis + Add Title
            self.bioelectricPlotAxes[channelIndex].set_title("Bioelectric Signal in Channel " + str(channelIndex + 1))
            self.bioelectricPlotAxes[channelIndex].set_xlabel("Time (Seconds)")
            self.bioelectricPlotAxes[channelIndex].set_ylabel("Bioelectric Signal (Volts)")
            
        # Create the Data Plots
        self.eyeBlinks = []
        self.trailingAveragePlots = []
        self.filteredBioelectricDataPlots = []
        self.filteredBioelectricPlotAxes = [] 
        for channelIndex in range(self.numChannels):
            # Create Plot Axes
            self.filteredBioelectricPlotAxes.append(axes[channelIndex, 1])
            # Plot Flitered Peaks
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            self.trailingAveragePlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:blue", linewidth=1, alpha = 0.65)[0])
            self.eyeBlinks.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], 'o', c="tab:blue", markersize=7, alpha = 0.65)[0])

            # Set Figure Limits
            self.filteredBioelectricPlotAxes[channelIndex].set_ylim(yLimLow, yLimHigh)
            # Label Axis + Add Title
            self.filteredBioelectricPlotAxes[channelIndex].set_title("Filtered Bioelectric Signal in Channel " + str(channelIndex + 1))
            self.filteredBioelectricPlotAxes[channelIndex].set_xlabel("Time (Seconds)")
            self.filteredBioelectricPlotAxes[channelIndex].set_ylabel("Filtered Signal (Volts)")
            
        # Tighten Figure White Space (Must be After wW Add Fig Info)
        self.fig.tight_layout(pad=2.0);
        
    
    def analyzeData(self, dataFinger, plotStreamedData = False, predictionModel = None, actionControl = None, calibrateModel = False):     
        
        eyeAngles = []
        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):
            
            # ---------------------- Band Pass Filter ----------------------- #    
            # Band Pass Filter to Remove Noise
            startBPFindex = max(dataFinger - self.bandPassBuffer, 0)
            yDataBuffer = self.data['Channel' + str(channelIndex+1)][startBPFindex:dataFinger + self.numTimePoints].copy()
            
            # Get the Sampling Frequency from the First Batch (If Not Given)
            if not self.samplingFreq:
                self.samplingFreq = len(self.data['timePoints'][startBPFindex:])/(self.data['timePoints'][-1] - self.data['timePoints'][startBPFindex])
                print("Setting Sampling Frequency to", self.samplingFreq)
            
            filteredData = self.butterFilter(yDataBuffer, self.cutOffFreq, self.samplingFreq, order = 3, filterType = 'low')[-self.numTimePoints:]
            # --------------------------------------------------------------- #
            
            # --------------------- Predict Eye Movement  ------------------- #
            # Get the Current Voltage (Take Average)
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
            
            # ------------------- Extract Blink Features  ------------------- #
            # Extarct EOG Peaks from Up Channel
            if channelIndex == 0:
                filteredDataX = self.data['timePoints'][-len(filteredData):]
                self.findBlinks(filteredDataX, filteredData, channelIndex)
            # --------------------------------------------------------------- #
            
            # --------------------- Calibrate Eye Angle --------------------- #
            if calibrateModel and self.calibrateChannelNum == channelIndex:
                argMax = np.argmax(filteredData)
                argMin = np.argmin(filteredData)
                earliestExtrema = argMax if argMax < argMin else argMin
                
                self.timePoints = self.data['timePoints'][dataFinger:dataFinger + self.numTimePoints]
                plt.plot(self.timePoints, filteredData)
                plt.plot(self.timePoints[earliestExtrema], filteredData[earliestExtrema], 'o', linewidth=3)
                plt.show()
                
                self.calibrationVoltages[self.calibrateChannelNum].append(np.average(filteredData[earliestExtrema:earliestExtrema + 10]))
            # --------------------------------------------------------------- #
            
            # ------------------- Plot Biolectric Signals ------------------- #
            if plotStreamedData and not calibrateModel:
                # Get X Data: Shared Axis for All Channels
                self.timePoints = self.data['timePoints'][dataFinger:dataFinger + self.numTimePoints]

                # Get New Y Data
                newYData = self.data['Channel' + str(channelIndex+1)][dataFinger:dataFinger + self.numTimePoints]
                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.bioelectricDataPlots[channelIndex].set_data(self.timePoints, newYData)
                self.bioelectricPlotAxes[channelIndex].set_xlim(self.timePoints[0], self.timePoints[-1])
                            
                # Keep Track of Recently Digitized Data
                for voltageInd in range(len(channelVoltages)):
                    self.trailingAverageData[channelIndex].extend([channelVoltages[voltageInd]]*self.predictEyeAngleGap)
                self.trailingAverageData[channelIndex] = self.trailingAverageData[channelIndex][len(channelVoltages)*self.predictEyeAngleGap:]
                # Plot the Filtered + Digitized Data
                self.filteredBioelectricDataPlots[channelIndex].set_data(self.timePoints, filteredData[-len(self.timePoints):])
                self.trailingAveragePlots[channelIndex].set_data(self.timePoints, self.trailingAverageData[channelIndex][-len(self.timePoints):])
                self.filteredBioelectricPlotAxes[channelIndex].set_xlim(self.timePoints[0], self.timePoints[-1]) 
                # Plot the Eye's Angle if Electrodes are Calibrated
                if self.predictEyeAngle[channelIndex]:
                    self.filteredBioelectricPlotAxes[channelIndex].legend(["Eye's Angle: " + "%.3g"%eyeAngle], loc="upper left")
                # Add Eye Blink Peaks
                if channelIndex == 0:
                    self.eyeBlinks[channelIndex].set_data(self.blinksXLocs, self.blinksYLocs)
            # --------------------------------------------------------------- #   
            
        # -------------------- Update Virtual Reality  ---------------------- #
        if actionControl and self.predictEyeAngle[channelIndex] and not calibrateModel:
            actionControl.setGaze(eyeAngles)
        # ------------------------------------------------------------------- #

        # -------------------------- Update Plots --------------------------- #
        # Update to Get New Data Next Round
        if plotStreamedData and not calibrateModel:
            self.fig.show()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        # --------------------------------------------------------------------#  
        

    def analyzeFullBatch(self, channelIndex = 1):
        print("Printing Seperate test plots")
        # Get Data to Plot
        xData = self.data['timePoints']
        yData = self.data['Channel' + str(channelIndex)]
        
        # Get Data and Filter
        plt.figure()
        plt.plot(xData, yData, c='tab:red', alpha=0.7)
        plt.title("Bioelectric Data")
        
        plt.figure()
        filteredData = self.butterFilter(yData, self.cutOffFreq, self.samplingFreq, order = 3, filterType = 'low')
        plt.plot(xData,filteredData, c='tab:blue', alpha=0.7)
        plt.title("Filtered Data")
        
        # Get the Current Voltage (Take Average)
        eyeVoltages = []
        self.currentEyeVoltages[channelIndex] = self.steadyStateEye
        for dataBatchInd in range(0, len(filteredData), self.moveDataFinger):
            batchData = filteredData[dataBatchInd - self.voltagePositionBuffer:dataBatchInd + self.moveDataFinger]
            currentEyeVoltage = self.findTraileringAverage(batchData, deviationThreshold = self.minVoltageMovement)
            # Compare Voltage Difference to Remove Small Shakes
            if abs(currentEyeVoltage - self.currentEyeVoltages[channelIndex]) > self.minVoltageMovement:
                self.currentEyeVoltages[channelIndex] = currentEyeVoltage
            # Keep Track of Data
            eyeVoltages.extend([self.currentEyeVoltages[channelIndex]]*len(filteredData[dataBatchInd:dataBatchInd + self.moveDataFinger]))
        plt.plot(xData, eyeVoltages, c='k', alpha=0.7)
        

# --------------------------------------------------------------------------- #
# ------------------------- Signal Analysis --------------------------------- #

    def butterParams(self, cutoffFreq = [0.1, 7], samplingFreq = 800, order=3, filterType = 'band'):
        nyq = 0.5 * samplingFreq
        if filterType == "band":
            normal_cutoff = [freq/nyq for freq in cutoffFreq]
        else:
            normal_cutoff = cutoffFreq / nyq
        sos = butter(order, normal_cutoff, btype = filterType, analog = False, output='sos')
        return sos
    
    def butterFilter(self, data, cutoffFreq, samplingFreq, order = 3, filterType = 'band'):
        sos = self.butterParams(cutoffFreq, samplingFreq, order, filterType)
        return scipy.signal.sosfiltfilt(sos, data)
    
    def findBlinks(self, xData, yData, channelIndex):
        minBaselinePoints = 10
        minChi2 = 5*10E-5
        minPeakHeight = 0.1   # No Less Than 0.11
        multPeakSepMax = 0.5  # No Less Than 0.25
        
        # Find All Potential Blinks in the Data
        peakIndices = scipy.signal.find_peaks(yData, prominence=.01, width=20)[0];
        # Extract the True Blinks and Their Features
        for peakInd in peakIndices:
            peakLocX = xData[peakInd]
            
            # If the Peak Has Already Been Recorded, Don't ReAnalyze
            if self.blinksXLocs and peakLocX <= self.blinksXLocs[-1]+0.1:
                continue
            
            # ------------------ Find the Blink's Baselines ----------------- #
            # Calculate the Left and Right Baseline of the Peak
            leftBaselineIndex = self.findBaselineIndex(xData, yData, peakInd, searchDirection = -1)
            rightBaselineIndex = self.findBaselineIndex(xData, yData, peakInd, searchDirection = 1)
            
            # If No Baseline is Found, Ignore the Blink (Too Noisy, Probably Not a Blink)
            if leftBaselineIndex >= peakInd - minBaselinePoints or rightBaselineIndex <= peakInd + minBaselinePoints:
                continue
            # If the Peak is Too Small, Remove the Peak
            elif yData[peakInd] - yData[leftBaselineIndex] < minPeakHeight or yData[peakInd] - yData[rightBaselineIndex] < minPeakHeight:
                #print("Too Small", peakLocX)
                continue
            # --------------------------------------------------------------- #
            
            # ------------ Find leftStroke and rightStroke Lines ------------ #
            # Calculate the leftStroke and rightStroke Lines
            leftLineParams, startLeftLineInd, endLeftLineInd = self.findPeakLines_TWO(xData, yData, leftBaselineIndex, peakInd, minChi2, searchDirection = 1)
            rightLineParams, startRightLineInd, endRightLineInd = self.findPeakLines_TWO(xData, yData, rightBaselineIndex, peakInd, minChi2, searchDirection = -1)
        
            # Remove Peaks Without a Good Line
            if not startLeftLineInd or not startRightLineInd:
                print("No Line", peakLocX)
                continue
            # --------------------------------------------------------------- #
            
            # -------------------- Extract Blink Features ------------------- #
            newFeatures = self.extractFeatures(xData, yData, peakInd, leftBaselineIndex, rightBaselineIndex, leftLineParams, rightLineParams)
            
            # Remove Peaks with Bad Blink Features
            if len(newFeatures) == 0:
                continue
            # Else, Add the New Features
            self.blinkFeatures.append(newFeatures)
            # --------------------------------------------------------------- #
            
            # ----------------- Singular or Multiple Blinks? ---------------- #
            # Check if the Blink is a Part of a Multiple Blink Sequence
            if self.singleBlinksX and peakLocX - self.singleBlinksX[-1] < multPeakSepMax:
                # If So, Remove the Last Single Blink as its a Multiple
                lastBlinkX = self.singleBlinksX.pop()
                # Check if Other Associated Multiples Have Been Found
                if self.multipleBlinksX and peakLocX - self.multipleBlinksX[-1][-1] < multPeakSepMax:
                    self.multipleBlinksX[-1].append(peakInd)
                else:
                    self.multipleBlinksX.append([lastBlinkX, peakLocX])
            else:
                self.singleBlinksX.append(peakLocX)
            # Record the Blink's Location
            self.blinksXLocs.append(peakLocX)
            self.blinksYLocs.append(yData[peakInd])
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
                #plt.plot([leftBlinkBaselineX, rightBlinkBaselineX], [leftBlinkBaselineY, rightBlinkBaselineY], 'bo');
                plt.plot(xData[startLeftLineInd:endLeftLineInd], leftLineParams[0]*xData[startLeftLineInd:endLeftLineInd] + leftLineParams[1])
                plt.plot(xData[startRightLineInd:endRightLineInd], rightLineParams[0]*xData[startRightLineInd:endRightLineInd] + rightLineParams[1])
                # Figure Aesthetics
                plt.xlim([xData[leftBaselineIndex], xData[rightBaselineIndex]])
                plt.show()
            # --------------------------------------------------------------- #
    
    def findIntersectionPoint(self, leftLineParams, rightLineParams):
        xPoint = (rightLineParams[1] - leftLineParams[1])/(leftLineParams[0] - rightLineParams[0])
        yPoint = leftLineParams[0]*xPoint + leftLineParams[1]
        return xPoint, yPoint

    def findClosestInd(self, data, val):
        return data[np.argmin(abs(data - val))]
    
    def organizeVelInds(self, xData, speed, velInds):
        newInds = []
        
        
        return newInds

    def organizeDerivPeaks(self, dy_dt_ABS, peakInd, velIndsTotal, accelInds):    
        # Take the Highest Indices
        velInds = [max(velIndsTotal[velIndsTotal < peakInd], key = lambda velInd: dy_dt_ABS[velInd])]
        velInds.append(max(velIndsTotal[velIndsTotal > peakInd], key = lambda velInd: dy_dt_ABS[velInd]))
        # Verify That the Index is Correct, Else Remove the Peak (Improper Blink)
        if velInds[0] > peakInd or velInds[1] < peakInd:
            print("Bad Peak ... WHAT")
            return [], []
        # Get the Correct Accel Inds
        newIndsAccel = [accelInds[accelInds < velInds[0]][-1]]
        newIndsAccel.append(accelInds[accelInds > velInds[0]][0])
        
        return velInds, newIndsAccel
    
    def quantifyPeakShape(self, xData, yData, peakInd, peakAmp):
        # Calculate Derivatives
        dx_dt = np.gradient(xData); dy_dt = np.gradient(yData)
        d2x_dt2 = np.gradient(dx_dt); d2y_dt2 = np.gradient(dy_dt)
        d2y_dt2_ABS = abs(d2y_dt2); dy_dt_ABS = abs(dy_dt)
        # Calculate Peak Shape parameters
        speed = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        acceleration = np.sqrt(d2x_dt2 * d2x_dt2 + d2y_dt2 * d2y_dt2)
        curvature = np.abs((d2x_dt2 * dy_dt - dx_dt * d2y_dt2)) / speed**3
        # Pull Out Peak Shape Features
        velIndsTotal = scipy.signal.find_peaks(dy_dt_ABS, prominence=.000001, width=20, height=np.mean(dy_dt_ABS))[0];
        accelIndsTotal = scipy.signal.find_peaks(d2y_dt2_ABS, prominence=.000001, width=10, height=np.mean(d2y_dt2_ABS))[0];

        if len(velIndsTotal) >= 2 and len(accelIndsTotal) >= 2:
            # For Good Derivative, Find the Features
            velInds, accelInds = self.organizeDerivPeaks(dy_dt_ABS, peakInd, velIndsTotal, accelIndsTotal)
            # Extract Blink Velocities
            peakClosingVelRatio = dy_dt_ABS[velInds[0]]/(peakAmp - yData[velInds[0]])
            peakOpeningVelRatio = dy_dt_ABS[velInds[1]]/(peakAmp - yData[velInds[1]])
            # Extract Blink Acceleration
            peakClosingAccelRatio = d2y_dt2_ABS[accelInds[0]]/(peakAmp - yData[accelInds[0]])
            peakMidClosedAccelRatio = d2y_dt2_ABS[accelInds[1]]/(peakAmp - yData[accelInds[1]])
            # Ratio
            velRatio = peakOpeningVelRatio/peakClosingVelRatio
            accelRatio = peakClosingAccelRatio/peakMidClosedAccelRatio
            # New Half Duration
            halfAmpDuration2 = xData[int(len(yData)/2):][np.argmin(abs(yData[int(len(yData)/2):] - dy_dt_ABS[velInds[0]]))] - xData[velInds[0]]
            rightHalfAmpDuration = xData[velInds[1]] - xData[0:int(len(yData)/2)][np.argmin(abs(yData[int(len(yData)/2):] - dy_dt_ABS[velInds[1]]))]
            # Durations
            velPeakDuration = xData[velInds[1]] - xData[velInds[0]]
            accPeakDuration1 = xData[accelInds[1]] - xData[accelInds[0]]
            # Compile the Features
            indexParams = [peakClosingVelRatio, peakOpeningVelRatio, peakClosingAccelRatio, peakMidClosedAccelRatio]
            indexParams.extend([velRatio, accelRatio, halfAmpDuration2, velPeakDuration, accPeakDuration1, rightHalfAmpDuration])
            # Reduce the Derivatives
            return speed, acceleration, curvature, indexParams
        # Else, Remove the Blink
        else:
            return None, None, None, []
        
        

    def extractFeatures(self, xData, yData, peakInd, leftBaselineIndex, rightBaselineIndex, leftLineParams, rightLineParams):
        peakShapeBuffer = 15
        
        # ----------------------- Remove Peak Baseline ---------------------- #
        # Account for Skewed Baseline: Probably From Eye Movement Alongside Blink
        baseLinesSkewed = abs(yData[rightBaselineIndex] - yData[leftBaselineIndex]) > 0.5*(yData[peakInd] - max(yData[rightBaselineIndex], yData[leftBaselineIndex]))
        # Calculate Baseline of the Peak
        if baseLinesSkewed:
            # If Skewed, Ignore the Skewed Point, and Take the Minimum Baseline
            peakBaselineY = min(yData[rightBaselineIndex], yData[leftBaselineIndex])
        else:
            # If Proper, Take a Linear Fit of the Base Points
            delY = yData[rightBaselineIndex] - yData[leftBaselineIndex]
            delX = xData[rightBaselineIndex] - xData[leftBaselineIndex]
            peakBaselineY = yData[leftBaselineIndex] + (delY/delX)*(xData[peakInd] - xData[leftBaselineIndex])
        
        # Calculate New Baseline Points of the Peak
        leftBlinkBaselineX, _ = self.findIntersectionPoint([0, peakBaselineY], leftLineParams)
        rightBlinkBaselineX, _ = self.findIntersectionPoint(rightLineParams, [0, peakBaselineY])
        # Fit the Full Peak
        peakFunc = interpolate.interp1d(xData[leftBaselineIndex:rightBaselineIndex+1], yData[leftBaselineIndex:rightBaselineIndex+1])
        # ------------------------------------------------------------------- #
        
        # ---------------------- Extract Blink Features --------------------- #
        # Find Peak's Tent
        peakTentX, peakTentY = self.findIntersectionPoint(leftLineParams, rightLineParams)
        tentDeviationX = peakTentX - xData[peakInd]
        tentDeviationY = peakTentY - yData[peakInd]
        # Calculate Blink Amplitudes
        blinkAmpTent = peakTentY - peakBaselineY                  # Distance from the Tent to the Baseline
        blinkAmpPeak = yData[peakInd] - peakBaselineY             # Distance from the Peak to the Baseline
        # Calculate the Standard Blink Duration
        blinkDuration = rightBlinkBaselineX - leftBlinkBaselineX  # The Total Time of the Blink
        closingTime = peakTentX - leftBlinkBaselineX              # Eye's Closing Time
        openingTime = rightBlinkBaselineX - peakTentX             # Eye's Opening Time
        closingFraction = closingTime/blinkDuration
        openingFraction = openingTime/blinkDuration
        # Calculate the Half Amplitude Duration
        blinkAmp50Y = peakBaselineY + blinkAmpPeak*0.5        
        blinkAmp50Right = xData[peakInd:rightBaselineIndex][np.argmin(abs(yData[peakInd:rightBaselineIndex] - blinkAmp50Y))]
        blinkAmp50Left = xData[leftBaselineIndex:peakInd][np.argmin(abs(yData[leftBaselineIndex:peakInd] - blinkAmp50Y))]
        halfClosedTime = blinkAmp50Right - blinkAmp50Left
        # Calculate Time the Eyes are Closed
        blinkAmp90Y = peakBaselineY + blinkAmpPeak*0.9
        blinkAmp90Right = xData[peakInd:rightBaselineIndex][np.argmin(abs(yData[peakInd:rightBaselineIndex] - blinkAmp90Y))]
        blinkAmp90Left = xData[leftBaselineIndex:peakInd][np.argmin(abs(yData[leftBaselineIndex:peakInd] - blinkAmp90Y))]
        eyesClosedTime = blinkAmp90Right - blinkAmp90Left
        
        percentTimeClosed = eyesClosedTime/halfClosedTime
        
        amplitudeRatio_90_50 = blinkAmp50Y/blinkAmp90Y

        # Calculate Speed, Acceleration, Curvature
        speed, acceleration, curvature, indexParams = self.quantifyPeakShape(xData[leftBaselineIndex:rightBaselineIndex+1], yData[leftBaselineIndex:rightBaselineIndex+1], peakInd-leftBaselineIndex, peakTentY)
        if len(indexParams) == 0:
            return []
        scaledCurvature = curvature * (xData[peakInd + peakShapeBuffer] - xData[peakInd - peakShapeBuffer])  
        scaledCurvature = scaledCurvature/max(scaledCurvature)
        scaledCurvature = (1/curvature[peakShapeBuffer])*curvature * (-xData[peakInd-peakShapeBuffer] + xData[peakInd+peakShapeBuffer])
        #
        maxCurvature = max(curvature)
        maxAcceleration = max(acceleration)
        maxSpeed = max(speed)
        # Calculate Peak Shape Parameters
        peakEntropy = entropy(yData[leftBaselineIndex:rightBaselineIndex])
        peakSkew = skew(yData[leftBaselineIndex:rightBaselineIndex], bias=False)
        peakKurtosis = kurtosis(yData[leftBaselineIndex:rightBaselineIndex], fisher=False, bias = False)
        #peakFitX = np.arange(xData[peakInd - peakShapeBuffer], xData[peakInd + peakShapeBuffer], .001); peakFitY = peakFunc(peakFitX)
        peakShape = np.array(xData[peakInd - peakShapeBuffer:peakInd + peakShapeBuffer+1]); peakShape -= peakShape[0]

        # Finalize the Features
        blinkFeatures = [peakTentY, blinkAmpTent, blinkAmpPeak, blinkAmp50Y, blinkAmp90Y, amplitudeRatio_90_50]
        blinkFeatures.extend([blinkDuration, closingTime, openingTime, closingFraction, openingFraction])
        blinkFeatures.extend([tentDeviationX, tentDeviationY, halfClosedTime, eyesClosedTime, percentTimeClosed])
        blinkFeatures.extend([peakSkew, peakKurtosis, peakEntropy, maxCurvature, maxAcceleration, maxSpeed])
        blinkFeatures.extend(indexParams)
        self.importantArrays.append([speed, acceleration, curvature, scaledCurvature, peakShape])
        # ------------------------------------------------------------------- #
        
        # ---------------------- Cull Potential Blinks ---------------------- #
        # If the Blink is Shorter Than 50ms or Longer Than 500ms, Ignore the Blink (Probably Eye Movement)
        if 0.5 < blinkDuration < 0.05:
            print("Bad Blink Duration:", blinkDuration)
            return []
        # If the Closing Time is Longer Than 150ms, it is Probably Not an Involuntary Blink
        elif closingTime > 0.15:
            print("Bad Closing Time:", closingTime)
            return []
        # If the Tent Peak is Malformed, Ignore the Peak
        #elif abs((peakTentY - yData[peakInd]) + (peakTentX - xData[peakInd])) > 0.1:
        #    print("Improper Peak Tent", xData[peakInd])
        #    return []
    
        #currentShape = yData[peakInd - peakShapeBuffer:peakInd + peakShapeBuffer + 1]
        #peakDome = np.diff(currentShape)/np.diff(currentShape)[0]
        #if max(peakDome) > 1 or min(peakDome) < -1:
        #    print("Bad Peak Dome", xData[peakInd])
        #    return []
        
        # If the Blink is Good, Return the Features
        return blinkFeatures
        # ------------------------------------------------------------------- #


    def findNearbyMinimum(self, data, xPointer, binarySearchWindow = 50, maxPointsSearch = 1000):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        if abs(binarySearchWindow) <= 1:
            return xPointer
        
        maxHeight = data[xPointer]; searchDirection = int(binarySearchWindow/abs(binarySearchWindow))
        # Binary Search Data to the Left to Find Minimum (Skip Over Small Bumps)
        for dataPointer in range(max(xPointer, 0), max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
            # If the Point is Greater Than
            if data[dataPointer] > maxHeight:
                return self.findNearbyMinimum(data, dataPointer - binarySearchWindow, math.floor(binarySearchWindow/8), maxPointsSearch - (xPointer - dataPointer - binarySearchWindow))
            else:
                xPointer = dataPointer
                maxHeight = data[dataPointer]
    
        # If Your Binary Search is Too Small, Reduce it
        return self.findNearbyMinimum(data, xPointer, math.floor(binarySearchWindow/2), maxPointsSearch)
    
    def findBaselineIndex(self, xData, yData, xPointer, searchDirection = 1):
        
        if searchDirection == 1:
            endSearch = len(yData)
        elif searchDirection == -1:
            endSearch = max(-1, xPointer - 1000)
        else:
            print("Wrong Search Direction")
            sys.exit()
        
        addOn = 5; minimumPeakPoints = 10;
        foundDrop = False; maxSlope = 0
        # Caluclate the Running Slope of the Data
        for peakInd in range(xPointer + searchDirection*(addOn+minimumPeakPoints), endSearch, searchDirection):
            # Calculate the First Derivative
            deltaY = np.mean(yData[max(0,peakInd - addOn):peakInd+1]) - np.mean(yData[max(0,peakInd - 2*addOn - 1):peakInd-addOn+1])
            deltaX = max(xData[peakInd] - xData[max(0,peakInd-addOn - 1)], 10E-10)
            firstDeriv = deltaY/deltaX
            
            # Verify Major Slope Drop
            if abs(firstDeriv) > 0.5:
                foundDrop = True
                maxSlope = max(maxSlope, abs(firstDeriv))
            
            if foundDrop and abs(firstDeriv) < maxSlope/10:
                return self.findNearbyMinimum(yData, peakInd, binarySearchWindow = searchDirection*20, maxPointsSearch = 50) #peakInd
        return xPointer
    
    def normalizePulseBaseline(self, curveData, polynomialDegree):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            curveData:  y-Axis Data for Baseline Removal
            polynomialDegree: Polynomials Used in Baseline Subtraction
        Output Parameters:
            curveData: y-Axis Data for a Baseline-Normalized Curve
        Assumption in Function: curveData is Positive
        ----------------------------------------------------------------------
        Further API Information Can be Found in the Following Link:
        https://pypi.org/project/BaselineRemoval/
        ----------------------------------------------------------------------
        """
        # Perform Baseline Removal Twice to Ensure Baseline is Gone
        for _ in range(2):
            # Baseline Removal Procedure
            baseObj = BaselineRemoval(curveData)  # Create Baseline Object
            curveData = baseObj.ModPoly(polynomialDegree) # Perform Modified multi-polynomial Fit Removal
    
        # Return the Data With Removed Baseline
        return curveData
    
    
    def findPeakLines_TWO(self, xData, yData, baselineIndex, peakInd, minChi2, searchDirection = 1):
        # Find the Starting/Ending Points Representing the Inner 70% of the Peak Amplitude
        peakAmp = yData[peakInd] - yData[baselineIndex]
        startLineY = yData[baselineIndex] + peakAmp*0.3
        endLineY = yData[baselineIndex] + peakAmp*0.6
        
        # Find the Closest Starting Point in the Curve
        if searchDirection == 1:
            # Find Left Line
            startLineInd = baselineIndex + np.argmin(abs(yData[baselineIndex:peakInd+1] - startLineY))
            endLineInd = baselineIndex + np.argmin(abs(yData[baselineIndex:peakInd+1] - endLineY))
        else:
            # Find Right Line
            startLineInd = peakInd + np.argmin(abs(yData[peakInd:baselineIndex+1] - endLineY))
            endLineInd = peakInd + np.argmin(abs(yData[peakInd:baselineIndex+1] - startLineY))
        # If No Line Found, Return 0
        if endLineInd - startLineInd < 5:
            return [0,0],0,0
        
        # Calculate the Line Parameters: [Slope, Y-Cross]
        lineParams, residuals, _, _, _ = np.polyfit(xData[startLineInd:endLineInd+1], yData[startLineInd:endLineInd+1], 1, full=True)
        if len(residuals) != 0:
            # Calculate the Chi2
            lineCHI2 = residuals[0] / (endLineInd - startLineInd - 1)
            # 
            if lineCHI2 < minChi2:
                return lineParams, startLineInd, endLineInd
        return [0,0],0,0
                

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
        popt, pcov = curve_fit(self.line, xData, yData)
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


"""

def findBaselineIndex(xData, yData, xPointer, searchDirection = 1):
    
    if searchDirection == 1:
        endSearch = len(yData)
    elif searchDirection == -1:
        endSearch = max(-1, xPointer - 1000)
    else:
        print("Wrong Search Direction")
        sys.exit()
    
    addOn = 5; firstDer = [0]*addOn; minimumPeakPoints = 40;
    foundDrop = False; maxSlope = 0
    # Caluclate the Running Slope of the Data
    for peakInd in range(xPointer + searchDirection*(addOn+minimumPeakPoints), endSearch, searchDirection):
        # Calculate the First Derivative
        deltaY = np.mean(yData[max(0,peakInd - addOn):peakInd+1]) - np.mean(yData[max(0,peakInd - 2*addOn - 1):peakInd-addOn+1])
        deltaX = max(xData[peakInd] - xData[max(0,peakInd-addOn - 1)], 10E-10)
        firstDeriv = deltaY/deltaX
        firstDer.append(deltaY/deltaX)
        
        # Verify Major Slope Drop
        if abs(firstDeriv) > 0.5:
            foundDrop = True
            maxSlope = max(maxSlope, abs(firstDeriv))
        
        if foundDrop and abs(firstDeriv) < maxSlope/10:
            return peakInd
    return peakInd

yDiff4 = []
xDiff4 = []
blinkDurations = []
leftIndices = []
rightIndices = []
finalInds = []
toBaselines = 15;
fitPercent = 0.7
peakIndices = scipy.signal.find_peaks(yData, prominence=.3, width=50)[0];
for peakInd in peakIndices:
    leftBaselineIndex = findBaselineIndex(xData, yData, peakInd, searchDirection = -1)
    rightBaselineIndex = findBaselineIndex(xData, yData, peakInd, searchDirection = 1)

    if leftBaselineIndex == peakInd or rightBaselineIndex == peakInd:
        continue

    blinkDuration = xData[rightBaselineIndex] - xData[leftBaselineIndex]
  #  if xData[rightBaselineIndex] - xData[leftBaselineIndex] > 0.2:
  #      continue
    if blinkDuration < 2:

        # Find Place to Start
        baseIndexes = [leftBaselineIndex, rightBaselineIndex]
        closestBaseline = np.argmin([yData[leftBaselineIndex], yData[rightBaselineIndex]])
        distanceFromPeak = int(0.95*(abs(peakInd - baseIndexes[closestBaseline])))

        # Find Start Indices of the Line
        secondDerivLeft = np.gradient(np.gradient(yData[leftBaselineIndex:peakInd]))
        beginLeftLineInd = np.argmax(secondDerivLeft)
        endLeftLineInd = np.argmin(secondDerivLeft)

        secondDerivRight = np.gradient(np.gradient(yData[peakInd:rightBaselineIndex]))
        beginRightLineInd = np.argmin(secondDerivRight)
        endRightLineInd = np.argmax(secondDerivRight)

        # Define Left Line Bounds
        startLeftLine = leftBaselineIndex + beginLeftLineInd + 15
        endLeftLine = leftBaselineIndex + endLeftLineInd - 15
        # Define Right Line Bounds
        startRightLine = peakInd + beginRightLineInd + 15
        endRightLine = peakInd + endRightLineInd - 15

        if endRightLine <= startRightLine or endLeftLine <= startLeftLine:
            continue


        # Define Lines
        leftLine = np.polyfit(xData[startLeftLine:endLeftLine], yData[startLeftLine:endLeftLine], 1)
        rightLine = np.polyfit(xData[startRightLine:endRightLine], yData[startRightLine:endRightLine], 1)

        xIntersect = (rightLine[1] - leftLine[1])/(leftLine[0] - rightLine[0])
        yIntersect = leftLine[0]*xIntersect + leftLine[1]

        if yIntersect - yData[peakInd] < 0.3:
            yDiff4.append(yIntersect - yData[peakInd])
            xDiff4.append(xIntersect - xData[peakInd])
            blinkDurations.append(blinkDuration)
            finalInds.append(peakInd)
            leftIndices.append(leftBaselineIndex)
            rightIndices.append(rightBaselineIndex)

        if True:
            plt.plot(xData[peakInd], yData[peakInd], 'ro');
            plt.plot(xData, yData); plt.plot(xData[leftBaselineIndex], yData[leftBaselineIndex], 'go');
            plt.plot(xData[rightBaselineIndex], yData[rightBaselineIndex], 'ro');
            plt.plot(xIntersect, yIntersect, 'ko')
            plt.plot(xData[startLeftLine:endLeftLine], leftLine[0]*xData[startLeftLine:endLeftLine] + leftLine[1])
            plt.plot(xData[startRightLine:endRightLine], rightLine[0]*xData[startRightLine:endRightLine] + rightLine[1])
            plt.xlim([xData[leftBaselineIndex], xData[rightBaselineIndex]])
            plt.show()
plt.plot(xData, yData); plt.plot(xData[finalInds], yData[finalInds], 'o');
plt.plot(xData[leftIndices], yData[leftIndices], 'go');
plt.plot(xData[rightIndices], yData[rightIndices], 'ro');
plt.xlim([5, 20])
plt.show()
xDiff4 = np.array(xDiff4); yDiff4 = np.array(yDiff4)
#ax = plt.axes(projection='3d')
plt.plot(xDiff4, yDiff4, 'o'); #plt.xlim([-0.05, 0]); plt.ylim([-.15, 0.05])


# array([-5.18623727,  0.06458267])



# Find the Start Index of the Left Line
secondDerivLeft = np.diff(yData[leftBaselineIndex:peakInd], 2)
midLeftLine = int((np.argmax(secondDerivLeft) + np.argmin(secondDerivLeft))/2)
# Find the Start Index of the Right Line
secondDerivRight = np.diff(yData[peakInd:rightBaselineIndex], 2)
midRightLine = int((np.argmin(secondDerivRight) + np.argmax(secondDerivRight))/2)
"""
