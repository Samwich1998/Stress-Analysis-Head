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
        
        self.peakShapeBuffer = 15
        
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
        self.blinkTypes = ["Relaxed", "Stressed"]
        self.currentState = self.blinkTypes[0]
        
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
                self.findBlinks(filteredDataX, filteredData, channelIndex, predictionModel)
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
                    if predictionModel or True:
                        self.filteredBioelectricPlotAxes[channelIndex].legend(["Current State: " + self.currentState], loc="upper left")
                    else:
                        self.filteredBioelectricPlotAxes[channelIndex].legend(["Eye's Angle: " + "%.3g"%eyeAngle, "Current State: " + self.currentState], loc="upper left")
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
    
    def findBlinks(self, xData, yData, channelIndex, predictionModel):
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
       #     leftLineParams, startLeftLineInd, endLeftLineInd = self.findPeakLines_TWO(xData, yData, leftBaselineIndex, peakInd, minChi2, searchDirection = 1)
       #     rightLineParams, startRightLineInd, endRightLineInd = self.findPeakLines_TWO(xData, yData, rightBaselineIndex, peakInd, minChi2, searchDirection = -1)

            # Remove Peaks Without a Good Line
       #     if not startLeftLineInd or not startRightLineInd:
       #         print("No Line", peakLocX)
       #         continue
            # --------------------------------------------------------------- #
            
            # -------------------- Extract Blink Features ------------------- #
            newFeatures = self.extractFeatures(xData[leftBaselineIndex:rightBaselineIndex+1], yData[leftBaselineIndex:rightBaselineIndex+1], peakInd-leftBaselineIndex)
            
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
            
            # ----------------- Label the Stress Level/Type ----------------- #
            if predictionModel:
                # Predict the Blink Type
                self.predictMovement(newFeatures, predictionModel)
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
    
    def findIntersectionPoint(self, leftLineParams, rightLineParams):
        xPoint = (rightLineParams[1] - leftLineParams[1])/(leftLineParams[0] - rightLineParams[0])
        yPoint = leftLineParams[0]*xPoint + leftLineParams[1]
        return xPoint, yPoint

    def organizeDerivPeaks(self, dy_dt_ABS, peakInd, velIndsTotal, accelInds):    
        # Take the Highest Indices
        velInds = [max(velIndsTotal[velIndsTotal < peakInd], key = lambda velInd: dy_dt_ABS[velInd])]
        velInds.append(max(velIndsTotal[velIndsTotal > peakInd], key = lambda velInd: dy_dt_ABS[velInd]))
        # Verify That the Index is Correct, Else Remove the Peak (Improper Blink)
        if velInds[0] > peakInd or velInds[1] < peakInd:
            print("Bad Peak ... WHAT")
            return [], []
        try:
            # Get the Correct Accel Inds
            newIndsAccel = [accelInds[accelInds < velInds[0]][-1]]
            newIndsAccel.append(accelInds[accelInds > velInds[0]][0])
            newIndsAccel.append(accelInds[accelInds < velInds[1]][-1])
            newIndsAccel.append(accelInds[accelInds > velInds[1]][0])

            i = 1
            for accelInd in accelInds[accelInds > newIndsAccel[-1]]:
                newIndsAccel.append(accelInd)
                i += 1
                if i == 2:
                    break
        except Exception as e:
            print(e)
            return [],[]
        
        return velInds, newIndsAccel
    
    def quantifyPeakShape(self, xData, yData, peakInd):
        # Calculate Derivatives
        dx_dt = np.gradient(xData); dy_dt = np.gradient(yData)
        d2x_dt2 = np.gradient(dx_dt); d2y_dt2 = np.gradient(dy_dt)
        d2y_dt2_ABS = abs(d2y_dt2); dy_dt_ABS = abs(dy_dt)
        # Calculate Peak Shape parameters
        speed = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        acceleration = np.sqrt(d2x_dt2 * d2x_dt2 + d2y_dt2 * d2y_dt2)
        curvature = np.abs((d2x_dt2 * dy_dt - dx_dt * d2y_dt2)) / speed**3  # Units 1/Volts
        peakShape = np.array(xData[peakInd - self.peakShapeBuffer:peakInd + self.peakShapeBuffer+1]); peakShape -= peakShape[0]

        # Save Blink Shpaes
        scaledCurvature = curvature * (xData[peakInd + self.peakShapeBuffer] - xData[peakInd - self.peakShapeBuffer])  
        scaledCurvature = scaledCurvature/max(scaledCurvature)
        scaledCurvature = (1/curvature[self.peakShapeBuffer])*curvature * (-xData[peakInd-self.peakShapeBuffer] + xData[peakInd+self.peakShapeBuffer])
        self.importantArrays.append([speed, acceleration, curvature, scaledCurvature, peakShape])
        
        # Return the Velocity, Acceleration, and Curvature
        return dy_dt_ABS, d2y_dt2_ABS, curvature


    def extractDerivFeatures(self, xData, yData, dy_dt_ABS, d2y_dt2_ABS, peakInd, peakAmp, velInds, accelInds):
        # Extract Normalized Blink Velocities
        peakClosingVel = dy_dt_ABS[velInds[0]]
        peakOpeningVel = dy_dt_ABS[velInds[1]]
        # Extract Normalized Blink Acceleration
        peakClosingAccel1 = d2y_dt2_ABS[accelInds[0]]
        peakClosingAccel2 = d2y_dt2_ABS[accelInds[1]]
        peakopeningAccel1 = d2y_dt2_ABS[accelInds[2]]
        peakopeningAccel2 = d2y_dt2_ABS[accelInds[3]]
        peakTailAccel1 = d2y_dt2_ABS[accelInds[4]]
        peakTailAccel2 = d2y_dt2_ABS[accelInds[5]]
        # Extract Amplitude Ratios
        velClosedRatio = yData[velInds[0]]/peakAmp
        velOpenRatio = yData[velInds[1]]/peakAmp
        accelClosedRatio1 = yData[accelInds[0]]/peakAmp
        accelClosedRatio2 = yData[accelInds[1]]/peakAmp
        accelOpenRatio1 = yData[accelInds[2]]/peakAmp
        accelOpenRatio2 = yData[accelInds[3]]/peakAmp
        # Extract Amplitudes
        velClosedVal = dy_dt_ABS[velInds[0]]
        velOpenVal = dy_dt_ABS[velInds[1]]
        accelClosedVal1 = d2y_dt2_ABS[accelInds[0]]
        accelClosedVal2 = d2y_dt2_ABS[accelInds[1]]
        accelOpenVal1 = d2y_dt2_ABS[accelInds[2]]
        accelOpenVal2 = d2y_dt2_ABS[accelInds[3]]
        # Ratio
        velRatio = dy_dt_ABS[velInds[0]]/dy_dt_ABS[velInds[1]]
        accelRatio1 = d2y_dt2_ABS[accelInds[0]]/d2y_dt2_ABS[accelInds[1]]
        accelRatio2 = d2y_dt2_ABS[accelInds[2]]/d2y_dt2_ABS[accelInds[3]]
        # Ratio with yData
        velRatioYData = yData[velInds[0]]/yData[velInds[1]]
        accelRatioYData1 = yData[accelInds[0]]/yData[accelInds[1]]
        accelRatioYData2 = yData[accelInds[2]]/yData[accelInds[3]]
        
        # New Half Duration
        durationByVel1 = xData[peakInd:][np.argmin(abs(yData[peakInd:] - yData[velInds[0]]))] - xData[velInds[0]]
        durationByVel2 = xData[velInds[1]] - xData[0:peakInd][np.argmin(abs(yData[0:peakInd] - yData[velInds[1]]))]
        durationByAccel1 = xData[peakInd:][np.argmin(abs(yData[peakInd:] - yData[accelInds[0]]))] - xData[accelInds[0]]
        durationByAccel2 = xData[peakInd:][np.argmin(abs(yData[peakInd:] - yData[accelInds[1]]))] - xData[accelInds[1]]
        durationByAccel3 = xData[accelInds[2]] - xData[0:peakInd][np.argmin(abs(yData[0:peakInd] - yData[accelInds[2]]))]
        midDurationRatio = durationByVel1/durationByVel2
        # Diving the Peak by Acceleration
        startToAccel = xData[accelInds[0]] - leftBlinkBaselineX
        accelCloseingPeakDuration = xData[accelInds[1]] - xData[accelInds[0]]
        accelToPeak = xData[peakInd] - xData[accelInds[1]]
        peakToAccel = xData[accelInds[2]] - xData[peakInd]
        accelOpeningPeakDuration = xData[accelInds[3]] - xData[accelInds[2]]
        accelToEnd = rightBlinkBaselineX - xData[accelInds[3]]
        # Diving the Peak by Velocity
        velPeakDuration = xData[velInds[1]] - xData[velInds[0]]
        startToVel = xData[velInds[0]] - leftBlinkBaselineX
        velToPeak = xData[peakInd] - xData[velInds[0]]
        peakToVel = xData[velInds[1]] - xData[peakInd]
        velToEnd = rightBlinkBaselineX - xData[velInds[1]]
        # Compile the Features
        indexParams = [peakClosingVel, peakOpeningVel, peakClosingAccel1, peakClosingAccel2, peakopeningAccel1, peakopeningAccel2, peakTailAccel1, peakTailAccel2]
        indexParams.extend([velOpenRatio, velClosedRatio, accelClosedRatio1, accelClosedRatio2, accelOpenRatio1, accelOpenRatio2])
        indexParams.extend([velClosedVal, velOpenVal, accelClosedVal1, accelClosedVal2, accelOpenVal1, accelOpenVal2])
        indexParams.extend([velRatio, accelRatio1, accelRatio2, velRatioYData, accelRatioYData1, accelRatioYData2])
        
        indexParams.extend([durationByVel1, durationByVel2, durationByAccel2, midDurationRatio])
        indexParams.extend([velPeakDuration, startToVel, velToEnd, velToPeak, peakToVel])
        indexParams.extend([accelCloseingPeakDuration, accelOpeningPeakDuration, accelToPeak, peakToAccel, startToAccel, accelToEnd])
        indexParams.extend([maxCurvature, closingSlope1, closingSlope2, openingSlope1, openingSlope2, openingSlope3])
        # Save Blink Shpaes
        peakShape = np.array(xData[peakInd - self.peakShapeBuffer:peakInd + self.peakShapeBuffer+1]); peakShape -= peakShape[0]
        scaledCurvature = curvature * (xData[peakInd + self.peakShapeBuffer] - xData[peakInd - self.peakShapeBuffer])  
        scaledCurvature = scaledCurvature/max(scaledCurvature)
        scaledCurvature = (1/curvature[self.peakShapeBuffer])*curvature * (-xData[peakInd-self.peakShapeBuffer] + xData[peakInd+self.peakShapeBuffer])
        self.importantArrays.append([speed, acceleration, curvature, scaledCurvature, peakShape])
        # Reduce the Derivatives
        return indexParams, velInds, accelInds
        
        

    def extractFeatures(self, xData, yData, peakInd):
        
        # ------------------- Subtract the Blink's Baseline ----------------- #
        # Check if the Baseline is Skewed. A Skewed Peak is Probably Eye Movement Alongside a Blink
        baseLinesSkewed = abs(yData[-1] - yData[0]) > 0.25*(yData[peakInd] - max(yData[-1], yData[0]))
        if baseLinesSkewed:
            # If Skewed, Take the Minimum Point as the Baseline
            peakBaselineY = min(yData[-1], yData[0])
        else:
            # If Proper Baseline, Take a Linear Fit of the Base Points
            delY = yData[-1] - yData[0]; delX = xData[-1] - xData[0]
            peakBaselineY = yData[0] + (delY/delX)*(xData[peakInd] - xData[0])
        # Subtract the Baseline
        yData -= peakBaselineY
        # ------------------------------------------------------------------- #
        
        # ----------------- Calculate the Peak's Derivatives ---------------- #
        # Calculate Speed, Acceleration, Curvature
        dy_dt_ABS, d2y_dt2_ABS, curvature = self.quantifyPeakShape(xData, yData, peakInd)
        
        # Find the Derivatives' Peaks
        velIndsTotal = scipy.signal.find_peaks(dy_dt_ABS, prominence=10E-10, width=3)[0];
        accelIndsTotal = scipy.signal.find_peaks(d2y_dt2_ABS, prominence=10E-10, width=3)[0];
        
        # If Not Enough Velocty/Acceleration Peaks Found, Cull the Blink
        if len(velIndsTotal) >= 2 and len(accelIndsTotal) >= 6:
            # For Good Derivative, Pull Out the Important Peaks
            velInds, accelInds = self.organizeDerivPeaks(dy_dt_ABS, peakInd, velIndsTotal, accelIndsTotal)
            # If Not Enough Good Peaks, Cull the Blink
            if len(velInds) == 0:
                return []
        else:
            return []
        # ------------------------------------------------------------------- #
        
        # --------------------- Find the Blink's Endpoints ------------------ #
        # Linearly Fit the Peak's Base
        startBlinkLineParams = np.polyfit(xData[accelInds[0]: velInds[0]], yData[accelInds[0]: velInds[0]], 1)
        endBlinkLineparams1 = np.polyfit(xData[velInds[1]: accelInds[3]], yData[velInds[1]: accelInds[3]], 1)
        endBlinkLineparams2 = np.polyfit(xData[accelInds[4]: accelInds[5]], yData[accelInds[4]: accelInds[5]], 1)
        
        # Calculate the New Baseline of the Peak
        startBlinkX, _ = self.findIntersectionPoint([0, 0], startBlinkLineParams)
        endBlinkX, _ = self.findIntersectionPoint(endBlinkLineparams2, [0, 0])
        # Calculate the New Baseline's Index
        startBlinkInd = np.argmin(abs(xData - startBlinkX))
        endBlinkInd = np.argmin(abs(xData - endBlinkX))
        # ------------------------------------------------------------------- #
        
        # ------------------------------------------------------------------- #
        # -------------------- Extract Amplitude Features ------------------- #
        # Find Peak's Tent Features
        peakTentX, peakTentY = self.findIntersectionPoint(startBlinkLineParams, endBlinkLineparams1)
        tentDeviationX = peakTentX - xData[peakInd]
        tentDeviationY = peakTentY - yData[peakInd]
        # Find Blink Amplitude Features
        blinkHeight = yData[peakInd]             # Distance from the Peak to the Baseline
        blinkAmpRatio = blinkHeight/peakTentY
        # Calculate Tent Ratios
        tentRatio = peakTentY/blinkHeight
        tentDeviationRatio = peakTentY/peakTentX
        # ------------------------------------------------------------------- #
        
        # -------------------- Extract Duration Features -------------------- #
        # Find the Standard Blink Durations
        blinkDuration = startBlinkX - endBlinkX      # The Total Time of the Blink
        closingTime = peakTentX - startBlinkX        # The Time for the Eye to Close
        openingTime = endBlinkX - peakTentX          # The Time for the Eye to Open
        # Calculate the Duration Ratios
        closingFraction = closingTime/blinkDuration
        openingFraction = openingTime/blinkDuration

        # Calculate the Half Amplitude Duration
        blinkAmp50Right = xData[peakInd + np.argmin(abs(yData[peakInd:] - blinkHeight*0.5))]
        blinkAmp50Left = xData[np.argmin(abs(yData[0:peakInd] - blinkHeight*0.5))]
        halfClosedTime = blinkAmp50Right - blinkAmp50Left
        # Calculate Time the Eyes are Closed
        blinkAmp90Right = xData[peakInd + np.argmin(abs(yData[peakInd:] - blinkHeight*0.9))]
        blinkAmp90Left = xData[np.argmin(abs(yData[0:peakInd] - blinkHeight*0.9))]
        eyesClosedTime = blinkAmp90Right - blinkAmp90Left
        # Caluclate Percent Closed
        percentTimeClosed = eyesClosedTime/halfClosedTime
        # ------------------------------------------------------------------- #
        
        # ---------------------- Extract Slope Features --------------------- #
        # Extract Slopes
        closingSlope1 = startBlinkLineParams[0]
        closingSlope2 = np.polyfit(xData[accelInds[1]: peakInd], yData[accelInds[1]: peakInd], 1)[0]
        openingSlope1 = np.polyfit(xData[peakInd: velInds[1]], yData[peakInd: velInds[1]], 1)[0]
        openingSlope2 = endBlinkLineparams1[0]
        openingSlope3 = endBlinkLineparams2[0]
        # ------------------------------------------------------------------- #
        
        # ---------------------- Extract Shape Features --------------------- #
        # Calculate Peak Shape Parameters
        peakAverage = np.mean(yData[startBlinkInd:endBlinkInd])
        peakAverageRatio = peakAverage/blinkHeight
        peakEntropy = entropy(yData)
        peakSkew = skew(yData, bias=False)
        peakKurtosis = kurtosis(yData, fisher=True, bias = False)
        maxCurvature = max(curvature[peakInd - 15: peakInd + 15])
        
        # Extract Derivative Features
        blinkFeatures = self.extractDerivFeatures(xData, yData, dy_dt_ABS, d2y_dt2_ABS, peakInd, blinkHeight, velInds, accelInds)
        if len(blinkFeatures) == 0:
            return []
        # ------------------------------------------------------------------- #

        # ------------------ Consolidate the Blink Features ----------------- #
        # Finalize the Features
        blinkFeatures.extend([blinkHeight, peakTentY, tentDeviationX, tentDeviationY, blinkAmpRatio, tentRatio, tentDeviationRatio])
        blinkFeatures.extend([blinkDuration, closingTime, openingTime, closingFraction, openingFraction, halfClosedTime, eyesClosedTime, percentTimeClosed])
        blinkFeatures.extend([closingSlope1, closingSlope2, openingSlope1, openingSlope2, openingSlope3])
        blinkFeatures.extend([peakAverage, peakAverageRatio, peakEntropy, peakSkew, peakKurtosis, maxCurvature])
        # ------------------------------------------------------------------- #
        
        
        sepInds = [startBlinkX, accelInds[0], accelInds[1], peakInd, accelInds[3], endBlinkX]
        self.plotData(xData, yData, peakInd, velInds = velInds, accelInds = accelInds, sepInds = sepInds, title = "Dividing the Blink")
        
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
    
        #currentShape = yData[peakInd - self.peakShapeBuffer:peakInd + self.peakShapeBuffer + 1]
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
    
    def predictMovement(self, inputData, predictionModel): 
        # Predict Data
        predictedIndex = predictionModel.predictData(np.array([inputData]))[0]
        self.currentState = self.blinkTypes[predictedIndex]
        print("\tThe Blink is ", self.currentState)

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
        
        
    def plotData(self, xData, yData, peakInd, velInds = [], accelInds = [], sepInds = [], title = "", peakSize = 3, lineWidth = 2, lineColor = "black", ax = None, axisLimits = []):
        xData = np.array(xData); yData = np.array(yData)
        # Create Figure
        showFig = False
        if ax == None:
            plt.figure()
            ax = plt.gca()
            showFig = True
        # Plot the Data
        ax.plot(xData, yData, linewidth = lineWidth, color = lineColor)
        ax.plot(xData[peakInd], yData[peakInd], 'om', markersize=peakSize*2)
        ax.plot(xData[velInds], yData[velInds], 'or', markersize=peakSize)
        ax.plot(xData[accelInds], yData[accelInds], 'ob', markersize=peakSize)
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
