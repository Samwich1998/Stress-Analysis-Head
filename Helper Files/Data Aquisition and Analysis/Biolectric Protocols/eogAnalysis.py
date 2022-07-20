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
from scipy.signal import savgol_filter
# Calibration Fitting
from scipy.optimize import curve_fit
# Plotting
import matplotlib
import matplotlib.pyplot as plt
# Feature Extraction
from scipy.stats import skew
from scipy.stats import entropy
from scipy.stats import kurtosis

# Import Files
import _filteringProtocols as filteringMethods # Import Files with Filtering Methods

# --------------------------------------------------------------------------- #
# ------------------ User Can Edit (Global Variables) ----------------------- #

class eogProtocol:
    
    def __init__(self, numTimePoints = 3000, moveDataFinger = 10, numChannels = 2, plotStreamedData = True):
        
        # Input Parameters
        self.numChannels = numChannels            # Number of Bioelectric Signals
        self.numTimePoints = numTimePoints        # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger      # The Amount of Data to Stream in Before Finding Peaks
        self.plotStreamedData = plotStreamedData  # Plot the Data

        # Define the Class with all the Filtering Methods
        self.filteringMethods = filteringMethods.filteringMethods()
        # High Pass Filter Parameters
        self.samplingFreq = None          # The Average Number of Points Steamed Into the Arduino Per Second; Depends on the User's Hardware; If NONE Given, Algorithm will Calculate Based on Initial Data
        self.bandPassBuffer = 5000        # A Prepended Buffer in the Filtered Data that Represents BAD Filtering; Units: Points
        self.cutOffFreq = [.1, 15]        # Optimal LPF Cutoff in Literatrue is 6-8 or 20 Hz (Max 35 or 50); I Found 25 Hz was the Best, but can go to 15 if noisy (small amplitude cutoff)
        
        # Blink Parameters
        self.minPeakHeight_Volts = 0.1    # The Minimum Peak Height in Volts; Removes Small Oscillations

        # Eye Angle Determination Parameters
        self.voltagePositionBuffer = 100  # A Prepended Buffer to Find the Current Average Voltage; Units: Points
        self.minVoltageMovement = 0.05    # The Minimum Voltage Change Required to Register an Eye Movement; Units: Volts
        self.predictEyeAngleGap = 5       # The Number of Points Between Each Eye Gaze Prediction; Will Backcaluclate if moveDataFinger > predictEyeAngleGap; Units: Points
        self.steadyStateEye = 3.3/2       # The Steady State Voltage of the System (With No Eye Movement); Units: Volts

        # Calibration Angles
        self.calibrationAngles = [[-45, 0, 45] for _ in range(self.numChannels)]
        self.calibrationVoltages = [[] for _ in range(self.numChannels)]
        # Pointers for Calibration
        self.calibrateChannelNum = 0           # The Current Channel We are Calibrating
        self.channelCalibrationPointer = 0     # A Pointer to the Current Angle Being Calibrated (A Pointer for Angle in self.calibrationAngles)
        # Calibration Function for Eye Angle
        self.predictEyeAngle = [lambda x: (x - self.steadyStateEye)*30]*self.numChannels
        
        # Prepare the Program to Begin Data Analysis
        self.checkParams              # Check to See if the User's Input Parameters Make Sense
        self.resetGlobalVariables()   # Start with Fresh Inputs (Clear All Arrays/Values)
        
        # If Plotting, Define Class for Plotting Peaks
        if plotStreamedData and numChannels != 0:
            print("HERE")
            # Initialize Plots; NOTE: PLOTTING SLOWS DOWN PROGRAM!
            matplotlib.use('Qt5Agg') # Set Plotting GUI Backend            
            self.initPlotPeaks()     # Create the Plots
    
    def setSamplingFrequency(self, startBPFindex):
        # Caluclate the Sampling Frequency
        self.samplingFreq = len(self.data[0][startBPFindex:])/(self.data[0][-1] - self.data[0][startBPFindex])
        print("\tSetting EOG Sampling Frequency to", self.samplingFreq)
        print("\tFor Your Reference, If Data Analysis is Longer Than", self.moveDataFinger/self.samplingFreq, ", Then You Will NOT be Analyzing in Real Time")
        
        # Set Blink Parameters
        self.minPoints_halfBaseline = max(1, int(self.samplingFreq*0.015))  # The Minimum Points in the Left/Right Baseline

    def resetGlobalVariables(self):
        # Data to Read in
        self.data = [ [], [[] for channel in range(self.numChannels)] ]

        # Hold Past Information
        self.trailingAverageData = {}
        for channelIndex in range(self.numChannels):
            self.trailingAverageData[channelIndex] = [0]*self.numTimePoints
        
        # Reset Last Eye Voltage (Volts)
        self.currentEyeVoltages = [self.steadyStateEye for _ in range(self.numChannels)]
        # Reset Feature Extraction
        self.featureList = []
        self.featureListExact = []
        # Reset Blink Indices
        self.singleBlinksX = []
        self.multipleBlinksX = []
        self.blinksXLocs = []
        self.blinksYLocs = []
        self.culledBlinkX = []
        self.culledBlinkY = []
        self.importantArrays = []
        
        self.lastAnalyzedPeakInd = 0      # The Index of the Last Potential Blink Analyzed from the Start of Data
        self.averageBlinkWindow = 60*1.5  # Time Window to Average Blink Features Together (Prepended to Blink's Peak Time)
        
        # Blink Classification
        self.blinkTypes = ['Relaxed', 'Stroop', 'Exercise', 'VR']
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
        figWidth = 20; figHeight = 15;
        self.fig, axes = plt.subplots(self.numChannels, 2, sharey=False, sharex = 'col', figsize=(figWidth, figHeight))
        
        # Plot the Raw Data
        yLimLow = 0; yLimHigh = 3.5; 
        self.bioelectricDataPlots = []; self.bioelectricPlotAxes = []
        for channelIndex in range(self.numChannels):
            # Create Plots
            if self.numChannels == 1:
                self.bioelectricPlotAxes.append(axes[0])
            else:
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
        self.eyeBlinkLocPlots = []
        self.eyeBlinkCulledLocPlots = []
        self.trailingAveragePlots = []
        self.filteredBioelectricDataPlots = []
        self.filteredBioelectricPlotAxes = [] 
        for channelIndex in range(self.numChannels):
            # Create Plot Axes
            if self.numChannels == 1:
                self.filteredBioelectricPlotAxes.append(axes[1])
            else:
                self.filteredBioelectricPlotAxes.append(axes[channelIndex, 1])
            
            # Plot Flitered Peaks
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            self.trailingAveragePlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:blue", linewidth=1, alpha = 0.65)[0])
            self.eyeBlinkLocPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], 'o', c="tab:blue", markersize=7, alpha = 0.65)[0])
            self.eyeBlinkCulledLocPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], 'o', c="tab:red", markersize=7, alpha = 0.65)[0])

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
            
            # ---------------------- Filter the Data ----------------------- #    
            # Band Pass Filter to Remove Noise
            startBPFindex = max(dataFinger - self.bandPassBuffer, 0)
            yDataBuffer = self.data[1][channelIndex][startBPFindex:dataFinger + self.numTimePoints].copy()
            
            # Get the Sampling Frequency from the First Batch (If Not Given)
            if not self.samplingFreq:
                self.setSamplingFrequency(startBPFindex)

            # Filter the Data: Low pass Filter and Savgol Filter
            #filteredData = self.butterFilter(yDataBuffer, self.cutOffFreq, self.samplingFreq, order = 3, filterType = 'low')[-self.numTimePoints:]
            #filteredData = self.filteringMethods.fourierFilter.removeFrequencies(yDataBuffer, self.samplingFreq, self.cutOffFreq)
            filteredData = self.filteringMethods.bandPassFilter.butterFilter(yDataBuffer, self.cutOffFreq[1], self.samplingFreq, order = 3, filterType = 'low')
           # filteredData = self.filteringMethods.bandPassFilter.highPassFilter(filteredData, self.cutOffFreq[0], 1, 0.1, 30)
            filteredData = savgol_filter(filteredData, 21, 2, mode='nearest', deriv=0)[-self.numTimePoints:]
            # --------------------------------------------------------------- #
            
            # ------------------- Extract Blink Features  ------------------- #
            # Extarct EOG Peaks from Up Channel
            if channelIndex == 0:
                # Get the New Data Where No Blinks Have Been Found Yet
                findBlinkDataY = filteredData[max(0,self.lastAnalyzedPeakInd - dataFinger):] 
                findBlinkDataX = self.data[0][max(dataFinger, self.lastAnalyzedPeakInd):dataFinger+self.numTimePoints] 
                # Find the Blinks in the New Data
                self.findBlinks(findBlinkDataX, findBlinkDataY[-len(findBlinkDataX):], max(dataFinger, self.lastAnalyzedPeakInd), predictionModel)
            # --------------------------------------------------------------- #
            
            # --------------------- Calibrate Eye Angle --------------------- #
            if calibrateModel and self.calibrateChannelNum == channelIndex:
                argMax = np.argmax(filteredData)
                argMin = np.argmin(filteredData)
                earliestExtrema = argMax if argMax < argMin else argMin
                
                timePoints = self.data[0][dataFinger:dataFinger + self.numTimePoints]
                plt.plot(timePoints, filteredData)
                plt.plot(timePoints[earliestExtrema], filteredData[earliestExtrema], 'o', linewidth=3)
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
            if plotStreamedData and not calibrateModel:
                # Get X Data: Shared Axis for All Channels
                timePoints = np.array(self.data[0][dataFinger:dataFinger + self.numTimePoints])

                # Get New Y Data
                newYData = self.data[1][channelIndex][dataFinger:dataFinger + self.numTimePoints]
                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.bioelectricDataPlots[channelIndex].set_data(timePoints, newYData)
                self.bioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1])
                            
                # Keep Track of Recently Digitized Data
                for voltageInd in range(len(channelVoltages)):
                    self.trailingAverageData[channelIndex].extend([channelVoltages[voltageInd]]*self.predictEyeAngleGap)
                self.trailingAverageData[channelIndex] = self.trailingAverageData[channelIndex][len(channelVoltages)*self.predictEyeAngleGap:]
                # Plot the Filtered + Digitized Data
                self.filteredBioelectricDataPlots[channelIndex].set_data(timePoints, filteredData[-len(timePoints):])
                self.trailingAveragePlots[channelIndex].set_data(timePoints, self.trailingAverageData[channelIndex][-len(timePoints):])
                self.filteredBioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1]) 
                # Plot the Eye's Angle if Electrodes are Calibrated
                if self.predictEyeAngle[channelIndex]:
                    if predictionModel or True:
                        self.filteredBioelectricPlotAxes[channelIndex].legend(["Current State: " + self.currentState], loc="upper left")
                    else:
                        self.filteredBioelectricPlotAxes[channelIndex].legend(["Eye's Angle: " + "%.3g"%eyeAngle, "Current State: " + self.currentState], loc="upper left")
                # Add Eye Blink Peaks
                if channelIndex == 0:
                    self.eyeBlinkLocPlots[channelIndex].set_data(self.blinksXLocs, self.blinksYLocs)
                    self.eyeBlinkCulledLocPlots[channelIndex].set_data(self.culledBlinkX, self.culledBlinkY)
            # --------------------------------------------------------------- #   
            
        # -------------------- Update Virtual Reality  ---------------------- #
        if actionControl and self.predictEyeAngle[channelIndex] and not calibrateModel:
            actionControl.setGaze(eyeAngles)
        # ------------------------------------------------------------------- #

        # -------------------------- Update Plots --------------------------- #
        # Update to Get New Data Next Round
        if plotStreamedData and not calibrateModel and self.numChannels != 0:
            self.fig.show()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        # --------------------------------------------------------------------#  

# --------------------------------------------------------------------------- #
# ------------------------- Signal Analysis --------------------------------- #
    
    def findBlinks(self, xData, yData, lastAnalyzedBuffer, predictionModel, debugBlinkDetection = True):
        
        # --------------------- Find and Analyze Blinks --------------------- #
        # Find All Potential Blinks in the Data
        peakIndices = scipy.signal.find_peaks(yData, prominence=0.1, width=max(1, int(self.samplingFreq*0.04)))[0];
        for peakInd in peakIndices:
            peakLocX = xData[peakInd]
            
            # Dont Reanalyze a Peak (Waste of Time)
            if peakLocX <= self.data[0][self.lastAnalyzedPeakInd]:
                continue
            
            # ------------------ Find the Peak's Baselines ------------------ #
            # Calculate the Left and Right Baseline of the Peak
            leftBaselineIndex = self.findNearbyMinimum(yData, peakInd, binarySearchWindow = -max(1, int(self.samplingFreq*0.001)), maxPointsSearch = max(1, int(self.samplingFreq*0.75)))
            rightBaselineIndex = self.findNearbyMinimum(yData, peakInd, binarySearchWindow = max(1, int(self.samplingFreq*0.001)), maxPointsSearch = max(1, int(self.samplingFreq*0.75)))
            
            # Wait to Analyze Peaks that are not Fully Formed
            if rightBaselineIndex >= len(xData) - max(1, int(self.samplingFreq*0.03)):
                break
            # --------------------------------------------------------------- #
            
            # --------------------- Initial Peak Culling --------------------- #
            # All Peaks After this Point Will been Evaluated
            self.lastAnalyzedPeakInd = max(lastAnalyzedBuffer + peakInd, self.lastAnalyzedPeakInd)            
            
            # Cull Peaks that are Too Small to be a Blink
            if yData[peakInd] - yData[leftBaselineIndex] < self.minPeakHeight_Volts or yData[peakInd] - yData[rightBaselineIndex] < self.minPeakHeight_Volts:
                if debugBlinkDetection: print("The Peak Height is too Small; Time = ", peakLocX)
                continue
            # If No Baseline is Found, Ignore the Blink (Too Noisy, Probably Not a Blink)
            elif leftBaselineIndex >= peakInd - self.minPoints_halfBaseline or rightBaselineIndex <= peakInd + self.minPoints_halfBaseline:
                if debugBlinkDetection: print("Peak Width too Small; Time = ", peakLocX)
                continue
            # --------------------------------------------------------------- #
            
            # -------------------- Extract Blink Features ------------------- #
            newFeatures = self.extractFeatures(xData[leftBaselineIndex:rightBaselineIndex+1], yData[leftBaselineIndex:rightBaselineIndex+1].copy(), peakInd-leftBaselineIndex, debugBlinkDetection)

            # Remove Peaks that are not Blinks
            if len(newFeatures) == 0:
                if debugBlinkDetection: print("\tNo Features; Time = ", peakLocX)
                continue
            # Label/Remove Possible Winks
            elif len(newFeatures) == 1:
                if debugBlinkDetection: print("\tWink")
                self.culledBlinkX.append(peakLocX)
                self.culledBlinkY.append(yData[peakInd])
                continue
            
            # Record the Blink's Location
            self.blinksXLocs.append(peakLocX)
            self.blinksYLocs.append(yData[peakInd])
            # Keep Running List of Good Blink Features (Plus Average)
            self.featureListExact.append(newFeatures)
            self.featureList.extend(np.mean(np.array(self.featureListExact)[self.blinksXLocs[-1] > peakLocX - self.averageBlinkWindow], axis=0))
            # --------------------------------------------------------------- #
            
            # ----------------- Singular or Multiple Blinks? ---------------- #
            multPeakSepMax = 0.5   # No Less Than 0.25
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
            # --------------------------------------------------------------- #
            
            # ----------------- Label the Stress Level/Type ----------------- #
            if False and predictionModel:
                # Predict the Blink Type
                self.predictMovement(self.featureList[-1], predictionModel)
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
    
    def convertToOddInt(self, x):
        return max(1, 2*math.floor((int(x)+1)/2) - 1)
    
    def getDerivPeaks(self, firstDeriv, secondDeriv, thirdDeriv, fourthDeriv, peakInd):
        # Get velocity peaks
        leftVelMax = np.argmax(firstDeriv[:peakInd])
        rightVelMin = self.findNearbyMinimum(firstDeriv, peakInd, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(firstDeriv))
        # Organize velocity peaks
        velInds = [leftVelMax, rightVelMin]
        
        # Find Acceleration peaks
        leftAccelMax = np.argmax(secondDeriv[:leftVelMax])
        leftAccelMin = self.findNearbyMinimum(secondDeriv, leftVelMax, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(secondDeriv))
        # Find midPoint Acceleration
        peakAccelMin = self.findNearbyMinimum(secondDeriv, peakInd, binarySearchWindow = -max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(secondDeriv))
        peakAccelMin = self.findNearbyMinimum(secondDeriv, peakAccelMin, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(secondDeriv))
        peakAccelMin = max(peakInd, peakAccelMin)
        # Find First Half of Eye Opening
        rightAccelMax = self.findNearbyMaximum(secondDeriv, rightVelMin, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(secondDeriv))
        # Find Second Half of Eye Opening
        rightAccelMin = self.findNearbyMinimum(secondDeriv, rightAccelMax, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(secondDeriv))
        rightAccelMax_End = self.findNearbyMaximum(secondDeriv, rightAccelMin, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(secondDeriv))
        # Organize acceleration peaks
        accelInds = [leftAccelMax, leftAccelMin, rightAccelMax, rightAccelMax_End]
        
        # Find third derivative peaks
        thirdDeriv_leftMin = self.findNearbyMinimum(thirdDeriv, leftVelMax, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(thirdDeriv))
        thirdDeriv_leftMin = self.findNearbyMinimum(thirdDeriv, thirdDeriv_leftMin, binarySearchWindow = -max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(thirdDeriv))
        thirdDeriv_rightMax = self.findNearbyMaximum(thirdDeriv, peakAccelMin, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(thirdDeriv))
        # Find Third Deriv Right Minimum
        thirdDeriv_rightMin = self.findNearbyMaximum(thirdDeriv, rightVelMin, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(thirdDeriv))
        thirdDeriv_rightMin = self.findNearbyMinimum(thirdDeriv, thirdDeriv_rightMin, binarySearchWindow = max(1, int(self.samplingFreq*0.005)), maxPointsSearch = len(thirdDeriv))
        # Organize third derivative peaks
        thirdDerivInds = [thirdDeriv_leftMin, thirdDeriv_rightMax, thirdDeriv_rightMin]
        
        # Add Correction to the Indices
        # if thirdDeriv_rightMin < rightAccelMax:
        #     accelInds[3] = rightAccelMax
        #     accelInds[2] = thirdDeriv_rightMin

        return velInds, accelInds, thirdDerivInds
    
    def quantifyPeakShape(self, xData, yData, peakInd):
        
        # Calculate 1-D Derivatives
        dx_dt = np.gradient(xData); dx_dt2 = np.gradient(dx_dt); 
        dy_dt = np.gradient(yData); dy_dt2 = np.gradient(dy_dt);
        # Calculate Peak Shape parameters
        speed = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        curvature = np.abs((dx_dt2 * dy_dt - dx_dt * dy_dt2)) / speed**3  # Units 1/Volts

        # Caluclate the Peak Derivatives
        firstDeriv = savgol_filter(yData, self.convertToOddInt(self.samplingFreq*0.02), 2,delta=1/self.samplingFreq, mode='interp', deriv=1)
        secondDeriv = savgol_filter(yData, self.convertToOddInt(self.samplingFreq*0.01), 2,delta=1/self.samplingFreq, mode='interp', deriv=2)
        thirdDeriv = savgol_filter(secondDeriv, self.convertToOddInt(self.samplingFreq*0.01), 2,delta=1/self.samplingFreq, mode='interp', deriv=1)
        fourthDeriv = savgol_filter(secondDeriv, self.convertToOddInt(self.samplingFreq*0.02), 2,delta=1/self.samplingFreq, mode='interp', deriv=2)
        
        # Return the Peak Analyis Equations
        return firstDeriv, secondDeriv, thirdDeriv, fourthDeriv, curvature
    
    def findLineIntersectionPoint(self, leftLineParams, rightLineParams):
        xPoint = (rightLineParams[1] - leftLineParams[1])/(leftLineParams[0] - rightLineParams[0])
        yPoint = leftLineParams[0]*xPoint + leftLineParams[1]
        return xPoint, yPoint
    
    def extractFeatures(self, xData, yData, peakInd, debugBlinkDetection = True):

        # ----------------------- Normalize the Peak ------------------------ #
        # Check if the Baseline is Skewed.
        baseLinesSkewed_Large = abs(yData[-1] - yData[0]) > 0.4*(yData[peakInd] - max(yData[-1], yData[0]))
        baseLinesSkewed_Medium = abs(yData[-1] - yData[0]) > 0.2*(yData[peakInd] - max(yData[-1], yData[0]))
                    
        # A Large Skew is Probably Eye Movement Alongside a Blink.
        if baseLinesSkewed_Large:
            if debugBlinkDetection: print("\tBaseline is WAY Too Skewed. Probably Eye Movement + Blink")
            return []
        
        # A Medium Skew is Okay. Could be Two Blinks Overlapping (etc).
        elif baseLinesSkewed_Medium:
            # If Skewed, Take the Minimum Point as the Baseline.
            peakBaselineY = min(yData[-1], yData[0])
        
        # If a Good Baseline is Found
        else:
            # Take a Linear Fit of the Base Points
            delY = yData[-1] - yData[0]; delX = xData[-1] - xData[0]
            peakBaselineY = yData[0] + (delY/delX)*(xData[peakInd] - xData[0])
            
        # Subtract the Baseline
        yData = yData - peakBaselineY
        
        # Normalize the Peak: Required as Voltages Dont have Physical Meaning
        peakHeight = yData[peakInd]
        yData = yData/peakHeight
        # ------------------------------------------------------------------- #
        
        # ----------------- Calculate the Peak's Derivatives ---------------- #
        # Calculate the Derivatives and Curvature
        firstDeriv, secondDeriv, thirdDeriv, fourthDeriv, curvature = self.quantifyPeakShape(xData, yData, peakInd)
        # Calculate the Peak Derivatives
        velInds, accelInds, thirdDerivInds = self.getDerivPeaks(firstDeriv, secondDeriv, thirdDeriv, fourthDeriv, peakInd)
            
        # Cull peaks with bad derivatives: malformed peaks
        if velInds[1] < velInds[0]:
            if debugBlinkDetection: print("Bad Velocity Inds")
            return []
        elif not accelInds[0] < accelInds[1] < accelInds[2] < accelInds[3]:
            if debugBlinkDetection: print("Bad Acceleration Inds")
            return []
        elif thirdDerivInds[1] < thirdDerivInds[0]:
            if debugBlinkDetection: print("Bad Third Derivative Inds: ", thirdDerivInds)
            return []
        elif not accelInds[0] < velInds[0] < accelInds[1] < peakInd < velInds[1] < accelInds[2] < accelInds[3]:
            if debugBlinkDetection: print("Bad Derivative Inds Order")
            return []
        # Cull Peaks if Baseline Not Fully Formed
        # if len(yData) - 5 < accelInds[3]:
        #     if debugBlinkDetection: print("Right Baseline Should be Extended")
        #     return []
        # Cull Noisy Peaks
        #accelInds_Trial = scipy.signal.find_peaks(thirdDeriv[thirdDerivInds[0]:], prominence=10E-20)[0];
        ####################elif
        
        # plt.plot(xData, yData/max(yData), 'k', linewidth=2)
        # plt.plot(xData, firstDeriv*0.8/max(firstDeriv), 'r', linewidth=1)
        # plt.plot(xData, secondDeriv*0.8/max(secondDeriv), 'b', linewidth=1)
        # plt.plot(xData, thirdDeriv*0.8/max(thirdDeriv), 'm', alpha = 0.5, linewidth=1)
        
        # secondDeriv = np.array(secondDeriv)
        # thirdDeriv = np.array(thirdDeriv)
        # xData = np.array(xData)
        
        # plt.plot(xData[accelInds], (secondDeriv*0.8/max(secondDeriv))[accelInds], 'ko', markersize=5)
        # plt.plot(xData[thirdDerivInds], (thirdDeriv*0.8/max(thirdDeriv))[thirdDerivInds], 'mo', markersize=5)
        # plt.legend(['Blink', 'firstDeriv', 'secondDeriv', 'thirdDeriv'])
        # # plt.title("Accel Inds = " + str(len(accelInds_Trial)))
        # plt.show()        
        # sepInds = [0, accelInds[0], accelInds[1], peakInd, accelInds[2], accelInds[3]]
        # self.plotData(xData, yData, peakInd, velInds = velInds, accelInds = accelInds, sepInds = sepInds, title = "Dividing the Blink")

        # ------------------------------------------------------------------- #
        
        # --------------------- Find the Blink's Endpoints ------------------ #
        # Linearly Fit the Peak's Slope
        upSlopeTangentParams = [firstDeriv[velInds[0]], yData[velInds[0]] - firstDeriv[velInds[0]]*xData[velInds[0]]]
        downSlopeTangentParams = [firstDeriv[velInds[1]], yData[velInds[1]] - firstDeriv[velInds[1]]*xData[velInds[1]]]
        
        # Calculate the New Endpoints of the Peak
        startBlinkX, _ = self.findLineIntersectionPoint([0, 0], upSlopeTangentParams)
        endBlinkX, _ = self.findLineIntersectionPoint(downSlopeTangentParams, [0, 0])
        # Calculate the New Endpoint Locations of the Peak
        startBlinkInd = np.argmin(abs(xData - startBlinkX))
        endBlinkInd = np.argmin(abs(xData - endBlinkX))
        # ------------------------------------------------------------------- #
        
        # ------------------------------------------------------------------- #
        # -------------------- Extract Amplitude Features ------------------- #
        # Find Peak's Tent
        peakTentX, peakTentY = self.findLineIntersectionPoint(upSlopeTangentParams, downSlopeTangentParams)
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
        maxOpeningAccel_secondHalfLoc = yData[accelInds[3]]
        
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
        accelOpeningPeakDuration = xData[accelInds[3]] - xData[accelInds[2]]
        accelToEnd = endBlinkX - xData[accelInds[3]]
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
        condensedDuration2 = xData[accelInds[3]] - xData[accelInds[0]]
        # ------------------------------------------------------------------- #
        
        # ------------------- Extract Derivative Features ------------------- #
        # Extract Closing Slopes
        closingSlope_MaxAccel = firstDeriv[accelInds[0]]
        closingSlope_MaxVel = firstDeriv[velInds[0]]
        closingSlope_MinAccel = firstDeriv[accelInds[1]]
        # Extract Opening Slopes
        openingSlope_MinVel = firstDeriv[velInds[1]]
        openingSlope_MaxAccel1 = firstDeriv[accelInds[2]]
        openingSlope_MaxAccel2 = firstDeriv[accelInds[3]]
        
        # Extract Closing Accels
        closingAccel_MaxAccel = secondDeriv[accelInds[0]]
        closingAccel_MaxVel = secondDeriv[velInds[0]]
        closingAccel_MinAccel = secondDeriv[accelInds[1]]
        # Extract Opening Accels
        openingAccel_MinVel = secondDeriv[velInds[1]]
        openingAccel_MaxAccel1 = secondDeriv[accelInds[2]]
        openingAccel_MaxAccel2 = secondDeriv[accelInds[3]]
        
        # Derivaive Ratios
        velRatio = firstDeriv[velInds[0]]/firstDeriv[velInds[1]]
        accelRatio1 = secondDeriv[accelInds[0]]/secondDeriv[accelInds[1]]
        accelRatio2 = secondDeriv[accelInds[2]]/secondDeriv[accelInds[3]]
        
        # New Half Duration
        durationByVel1 = xData[peakInd:][np.argmin(abs(yData[peakInd:] - yData[velInds[0]]))] - xData[velInds[0]]
        durationByVel2 = xData[velInds[1]] - xData[0:peakInd][np.argmin(abs(yData[0:peakInd] - yData[velInds[1]]))]
        durationByAccel1 = xData[peakInd:][np.argmin(abs(yData[peakInd:] - yData[accelInds[0]]))] - xData[accelInds[0]]
        durationByAccel2 = xData[peakInd:][np.argmin(abs(yData[peakInd:] - yData[accelInds[1]]))] - xData[accelInds[1]]
        durationByAccel3 = xData[accelInds[2]] - xData[0:peakInd][np.argmin(abs(yData[0:peakInd] - yData[accelInds[2]]))]
        durationByAccel4 = xData[accelInds[3]] - xData[0:peakInd][np.argmin(abs(yData[0:peakInd] - yData[accelInds[3]]))]
        midDurationRatio = durationByVel1/durationByVel2
        # ------------------------------------------------------------------- #
        
        # ------------------- Extract Integral Features ------------------- #
        # Peak Integral
        blinkIntegral = scipy.integrate.simpson(yData[startBlinkInd:endBlinkInd], xData[startBlinkInd:endBlinkInd])
        # Portion of Blink Integrals
        portion1Integral = 0
        if startBlinkInd < accelInds[0]:
            portion1Integral = scipy.integrate.simpson(yData[startBlinkInd:accelInds[0]], xData[startBlinkInd:accelInds[0]])
        portion2Integral = scipy.integrate.simpson(yData[accelInds[0]:velInds[0]], xData[accelInds[0]:velInds[0]])
        portion3Integral = scipy.integrate.simpson(yData[velInds[0]:accelInds[1]], xData[velInds[0]:accelInds[1]])
        portion4Integral = 0
        if accelInds[1] < peakInd:
            portion4Integral = scipy.integrate.simpson(yData[accelInds[1]:peakInd], xData[accelInds[1]:peakInd])
        portion5Integral = scipy.integrate.simpson(yData[peakInd:accelInds[2]], xData[peakInd:accelInds[2]])
        portion6Integral = scipy.integrate.simpson(yData[velInds[1]: accelInds[2]], xData[velInds[1]: accelInds[2]])
        portion7Integral = scipy.integrate.simpson(yData[accelInds[2]:accelInds[3]], xData[accelInds[2]:accelInds[3]])
        portion8Integral = 0
        if accelInds[3] < endBlinkInd:
            portion8Integral = scipy.integrate.simpson(yData[accelInds[3]:endBlinkInd], xData[accelInds[3]:endBlinkInd])
        
        # Other Integrals
        velToVelIntegral = scipy.integrate.simpson(yData[velInds[0]:velInds[1]], xData[velInds[0]:velInds[1]])
        closingIntegral = scipy.integrate.simpson(yData[startBlinkInd:peakInd], xData[startBlinkInd:peakInd])
        openingIntegral = scipy.integrate.simpson(yData[peakInd:endBlinkInd], xData[peakInd:endBlinkInd])
        closingSlopeIntegral = scipy.integrate.simpson(yData[accelInds[0]:accelInds[1]], xData[accelInds[0]:accelInds[1]])
        accel12Integral = scipy.integrate.simpson(yData[accelInds[1]:accelInds[2]], xData[accelInds[1]:accelInds[2]])
        openingAccelIntegral = scipy.integrate.simpson(yData[accelInds[2]:accelInds[3]], xData[accelInds[2]:accelInds[3]])
        condensedIntegral = scipy.integrate.simpson(yData[accelInds[0]:accelInds[3]], xData[accelInds[0]:accelInds[3]])
        peakToVel0Integral = scipy.integrate.simpson(yData[velInds[0]:peakInd], xData[velInds[0]:peakInd])
        peakToVel1Integral = scipy.integrate.simpson(yData[peakInd:velInds[1]], xData[peakInd:velInds[1]])
        # ------------------------------------------------------------------- #

        # ---------------------- Extract Shape Features --------------------- #
        fullBlink = yData[startBlinkInd:endBlinkInd].copy()
        fullBlink -= min(fullBlink)
        fullBlink = fullBlink/max(fullBlink)
        # Calculate Peak Shape Parameters
        peakAverage = np.mean(fullBlink)
        peakEntropy = entropy(abs(fullBlink) + 0.01)
        peakSkew = skew(fullBlink, bias=False)
        peakKurtosis = kurtosis(fullBlink, fisher=True, bias = False)
        peakSTD = np.std(fullBlink, ddof=1)
        
        peakCurvature = curvature[peakInd]
        # Curvature Around Main Points
        curvatureYDataAccel0 = curvature[accelInds[0]]
        curvatureYDataAccel1 = curvature[accelInds[1]]
        curvatureYDataAccel2 = curvature[accelInds[2]]
        curvatureYDataAccel3 = curvature[accelInds[3]]
        curvatureYDataVel0 = curvature[velInds[0]]
        curvatureYDataVel1 = curvature[velInds[1]]
        
        # Stanard Deviation
        velFullSTD = np.std(firstDeriv[startBlinkInd:endBlinkInd], ddof=1)
        accelFullSTD = np.std(secondDeriv[startBlinkInd:endBlinkInd], ddof=1)
        thirdDerivFullSTD = np.std(thirdDeriv[startBlinkInd:endBlinkInd], ddof=1)
        
        # Entropy
        velFullEntropy = entropy(abs(firstDeriv[startBlinkInd:endBlinkInd]) + 0.01)
        accelFullEntropy = entropy(abs(secondDeriv[startBlinkInd:endBlinkInd]) + 0.01)
        thirdDerivFullEntropy = entropy(abs(thirdDeriv[startBlinkInd:endBlinkInd]) + 0.01)
        # ------------------------------------------------------------------- #

        # ------------------------- Cull Bad Blinks ------------------------- #
        # Blinks are on average 100-400 ms. They can be on the range of 50-500 ms.
        if not 0.008 < blinkDuration < 0.5:
            if debugBlinkDetection: print("\tBad Blink Duration:", blinkDuration, xData[peakInd])
            return []
        if tentDeviationX < -0.2:
            if debugBlinkDetection: print("\tBad Blink tentDeviationX:", tentDeviationX, xData[peakInd])
            return []    
        if not -0.2 < tentDeviationY < 0.5:
            if debugBlinkDetection: print("\tBad Blink tentDeviationY:", tentDeviationY, xData[peakInd])
            return []                
        # If the Closing Time is Longer Than 150ms, it is Probably Not an Involuntary Blink
        elif not 0.04 < closingTime_Peak < 0.3:
            if debugBlinkDetection: print("\tBad Closing Time:", closingTime_Peak, xData[peakInd])
            return []
        elif not 0.04 < openingTime_Peak < 0.4:
            if debugBlinkDetection: print("\tBad Opening Time:", openingTime_Peak, xData[peakInd])
            return []
        elif -1 < velRatio:
            if debugBlinkDetection: print("\tBad velRatio:", velRatio)
            return [] 
        elif peakSkew < -0.75:
            if debugBlinkDetection: print("\tBad peakSkew:", peakSkew)
            return [] 
        
        # elif 15 < closingSlope_MinAccel or closingSlope_MinAccel < 0:
        #     if debugBlinkDetection: print("\tBad closingSlope2:", closingSlope_MinAccel)
        #     return []
        # elif 8 < closingSlopeDiff10:
        #     if debugBlinkDetection: print("\tBad closingSlopeDiff10:", closingSlopeDiff10)
        #     return []   
        # elif 15 < closingSlopeDiff12:
        #     if debugBlinkDetection: print("\tBad closingSlopeDiff12:", closingSlopeDiff12)
        #     return [] 
        # elif openingSlope1 < -8:
        #     if debugBlinkDetection: print("\tBad openingSlope1:", openingSlope1)
        #     return []      
        # elif openingSlope2 < -14:
        #     if debugBlinkDetection: print("\tBad openingSlope2:", openingSlope2)
        #     return []  
        
        # accelClosedVal1: Most data past 0.00015 is bad. No Min threshold
        # elif 0.000175 < accelClosedVal1:
        #     if debugBlinkDetection: print("\tBad accelClosedVal1:", accelClosedVal1)
        #     return []
        # accelClosedVal2: Most data past 0.000175 is bad. No Min threshold
        # elif 0.000175 < accelClosedVal2:
        #     if debugBlinkDetection: print("\tBad accelClosedVal2:", accelClosedVal2)
        #     return []
        # accelOpenVal1: Most data past 0.000175 is bad. No Min threshold
        # elif 0.000175 < accelOpenVal1:
        #     if debugBlinkDetection: print("\tBad accelOpenVal1:", accelOpenVal1)
        #     return []
        # # accelOpenVal2: Most data past 0.000175 is bad. No Min threshold
        # elif 0.000175 < accelOpenVal2:
        #     if debugBlinkDetection: print("\tBad accelOpenVal2:", accelOpenVal2)
        #     return []
        # elif accelToPeak < 0:
        #     if debugBlinkDetection: print("\tBad accelToPeak:", accelToPeak)
        #     return []     
        # elif 0.7 < openingAmpAccel1:
        #     if debugBlinkDetection: print("\tBad openingAmpAccel1:", openingAmpAccel1)
        #     return []   
        # elif 0.7 < openingAmpVel:
        #     if debugBlinkDetection: print("\tBad openingAmpVel:", openingAmpVel)
        #     return []  
        # elif 0.6 < openingAmpAccel2:
        #     if debugBlinkDetection: print("\tBad openingAmpAccel2:", openingAmpAccel2)
        #     return []  

        
    
        # elif 0.005 < velOpenVal:
        #     if debugBlinkDetection: print("\tBad velOpenVal:", velOpenVal)
        #     return [] 
        # elif 0.0006 < peakClosingAccel1:
        #     if debugBlinkDetection: print("\tBad peakClosingAccel1:", peakClosingAccel1)
        #     return [] 
        # elif 0.0006 < peakClosingAccel2:
        #     if debugBlinkDetection: print("\tBad peakClosingAccel2:", peakClosingAccel2)
        #     return []  
        # elif 0.016 < peakOpeningVel:
        #     if debugBlinkDetection: print("\tBad peakOpeningVel:", peakOpeningVel)
        #     return []  

        # elif 1500 < peakCurvature:
        #     if debugBlinkDetection: print("\tBad maxCurvature:", peakCurvature)
        #     return []          
        

        # ------------------------------------------------------------------- #

        # --------------------- Cull Voluntary Blinks  ---------------------- #
        # if tentDeviationX < -0.1 or tentDeviationX > 0.1:
        #     if debugBlinkDetection: print("\tWINK! tentDeviationX = ", tentDeviationX, " xLoc = ", xData[peakInd]); return ["Wink"]  
        # if not 0 < tentDeviationY < 0.6: # (max may be 0.25)
        #     if debugBlinkDetection: print("\tWINK! tentDeviationY = ", tentDeviationY); return ["Wink"]  
        # if blinkAmpRatio < 0.7:
        #     if debugBlinkDetection: print("\tWINK! blinkAmpRatio = ", blinkAmpRatio); return ["Wink"]
        # If the Closing Time is Longer Than 150ms, it is Probably Not an Involuntary Blink
        # elif closingTime > 0.2 or closingTime < 0.04:
        #      if debugBlinkDetection: print("\tWINK! closingTime:", closingTime); return ["Wink"]
        #     return []
        
        # elif 6E-4 < accelOpenVal1:
        #     if debugBlinkDetection: print("\tWINK! accelOpenVal1:", accelOpenVal1); return ["Wink"]
        # elif 3E-4 < accelOpenVal2:
        #     if debugBlinkDetection: print("\tWINK! accelOpenVal2:", accelOpenVal2); return ["Wink"]
        # elif closingSlope0 > 15:
        #     if debugBlinkDetection: print("\tWINK! closingSlope0 = ", closingSlope0); return ["Wink"]  
        # elif closingSlope1 > 15:
        #     if debugBlinkDetection: print("\tWINK! closingSlope1 = ", closingSlope1); return ["Wink"]  
        # elif closingSlope2 > 13:
        #     if debugBlinkDetection: print("\tWINK! closingSlope2 = ", closingSlope2); return ["Wink"] 
        # elif openingSlope1 < -7:
        #     if debugBlinkDetection: print("\tWINK! openingSlope1 = ", openingSlope1); return ["Wink"]  
        # elif openingSlope2 < -10:
        #     if debugBlinkDetection: print("\tWINK! openingSlope2 = ", openingSlope2); return ["Wink"]  
        # elif openingSlope3 < -7:
        #     if debugBlinkDetection: print("\tWINK! openingSlope3 = ", openingSlope3); return ["Wink"]  
        # elif 4 < closingSlopeDiff10:
        #     if debugBlinkDetection: print("\tWINK! closingSlopeDiff10 = ", closingSlopeDiff10); return ["Wink"]  
        # elif 31 < closingSlopeRatio1:
        #     if debugBlinkDetection: print("\tWINK! closingSlopeRatio1 = ", closingSlopeRatio1); return ["Wink"]  
        # elif -4.5 > openingSlopeDiff23:
        #     if debugBlinkDetection: print("\tWINK! openingSlopeDiff23:", openingSlopeDiff23); return ["Wink"]
        # elif -19 > openingSlopeRatio2:
        #     if debugBlinkDetection: print("\tWINK! openingSlopeRatio2:", openingSlopeRatio2); return ["Wink"]
        # elif maxCurvature > 750:
        #     if debugBlinkDetection: print("\tWINK! maxCurvature = ", maxCurvature); 
        # elif 0.001 < peakOpeningAccel2:
        #     if debugBlinkDetection: print("\tWINK! peakOpeningAccel2:", peakOpeningAccel2); return ["Wink"]
        # elif closingSlopeDiff12 > 12 or closingSlopeDiff12 < -1:
        #     if debugBlinkDetection: print("\tWINK! closingSlopeDiff12:", closingSlopeDiff12); return ["Wink"]
        # ------------------------------------------------------------------- #
        
        # ------------------ Consolidate the Blink Features ----------------- #
        peakFeatures = []
        # Organize Amplitude Features        
        peakFeatures.extend([xData[peakInd], peakHeight, tentDeviationX, tentDeviationY, tentDeviationRatio])
        peakFeatures.extend([maxClosingAccel_Loc, maxClosingVel_Loc, minBlinkAccel_Loc, openingAmpVel_Loc, maxOpeningAccel_firstHalfLoc, maxOpeningAccel_secondHalfLoc])
        peakFeatures.extend([closingAmpSegment1, closingAmpSegment2, closingAmpSegmentFull, openingAmpSegment1, openingAmpSegment2, openingAmplitudeFull])
        peakFeatures.extend([velocityAmpInterval, accelAmpInterval1, accelAmpInterval2])        
        
        # Organize Duration Features
        peakFeatures.extend([blinkDuration, closingTime_Tent, openingTime_Tent, closingTime_Peak, openingTime_Peak, closingFraction, openingFraction])
        peakFeatures.extend([halfClosedTime, eyesClosedTime, percentTimeEyesClosed])
        peakFeatures.extend([startToAccel, accelClosingDuration, accelToPeak, peakToAccel, accelOpeningPeakDuration, accelToEnd])
        peakFeatures.extend([velocityPeakInterval, startToVel, velToPeak, peakToVel, velToEnd])
        peakFeatures.extend([portion2Duration, portion3Duration, portion6Duration])
        peakFeatures.extend([accel12Duration, condensedDuration1, condensedDuration2])
        
        # Organize Derivative Features
        peakFeatures.extend([closingSlope_MaxAccel, closingSlope_MaxVel, closingSlope_MinAccel, openingSlope_MinVel, openingSlope_MaxAccel1, openingSlope_MaxAccel2])
        peakFeatures.extend([closingAccel_MaxAccel, closingAccel_MaxVel, closingAccel_MinAccel, openingAccel_MinVel, openingAccel_MaxAccel1, openingAccel_MaxAccel2])
        peakFeatures.extend([velRatio, accelRatio1, accelRatio2])
        peakFeatures.extend([durationByVel1, durationByVel1, durationByAccel1, durationByAccel2, durationByAccel3, durationByAccel4, midDurationRatio])
        
        # Organize Integral Features
        peakFeatures.extend([blinkIntegral, portion1Integral, portion2Integral, portion3Integral, portion4Integral, portion5Integral, portion6Integral, portion7Integral, portion8Integral])
        peakFeatures.extend([velToVelIntegral, closingIntegral, openingIntegral, closingSlopeIntegral, accel12Integral, openingAccelIntegral, condensedIntegral, peakToVel0Integral, peakToVel1Integral])
        
        # Organize Shape Features
        peakFeatures.extend([peakAverage, peakEntropy, peakSkew, peakKurtosis, peakSTD])
        peakFeatures.extend([peakCurvature, curvatureYDataAccel0, curvatureYDataAccel1, curvatureYDataAccel2, curvatureYDataAccel3, curvatureYDataVel0, curvatureYDataVel1])
        peakFeatures.extend([velFullSTD, accelFullSTD, thirdDerivFullSTD, velFullEntropy, accelFullEntropy, thirdDerivFullEntropy])
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
        
      
        return peakFeatures
        # ------------------------------------------------------------------- #

    def findNearbyMinimum(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 500):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            return xPointer - min(2, xPointer) + np.argmin(data[max(0,xPointer-2):min(xPointer+3, len(data))]) 
        
        maxHeightPointer = xPointer
        maxHeight = data[xPointer]; searchDirection = binarySearchWindow//abs(binarySearchWindow)
        # Binary Search Data to Find the Minimum (Skip Over Minor Fluctuations)
        for dataPointer in range(max(xPointer, 0), max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
            # If the Next Point is Greater Than the Previous, Take a Step Back
            if data[dataPointer] > maxHeight:
                return self.findNearbyMinimum(data, dataPointer - binarySearchWindow, round(binarySearchWindow/8), maxPointsSearch - searchDirection*(abs(dataPointer - binarySearchWindow)) - xPointer)
            # Else, Continue Searching
            else:
                maxHeightPointer = dataPointer
                maxHeight = data[dataPointer]

        # If Your Binary Search is Too Large, Reduce it
        return self.findNearbyMinimum(data, maxHeightPointer, round(binarySearchWindow/2), maxPointsSearch-1)
    
    def findNearbyMaximum(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 500):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        xPointer = min(max(xPointer, 0), len(data)-1)
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            return xPointer - min(2, xPointer) + np.argmax(data[max(0,xPointer-2):min(xPointer+3, len(data))]) 
        
        minHeightPointer = xPointer; minHeight = data[xPointer];
        searchDirection = binarySearchWindow//abs(binarySearchWindow)
        # Binary Search Data to Find the Minimum (Skip Over Minor Fluctuations)
        for dataPointer in range(xPointer, max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
            # If the Next Point is Greater Than the Previous, Take a Step Back
            if data[dataPointer] < minHeight:
                return self.findNearbyMaximum(data, dataPointer - binarySearchWindow, round(binarySearchWindow/2), maxPointsSearch - searchDirection*(abs(dataPointer - binarySearchWindow)) - xPointer)
            # Else, Continue Searching
            else:
                minHeightPointer = dataPointer
                minHeight = data[dataPointer]

        # If Your Binary Search is Too Large, Reduce it
        return self.findNearbyMaximum(data, minHeightPointer, round(binarySearchWindow/2), maxPointsSearch-1)
    
    
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
        predictionProbs = predictionModel.predictData(np.reshape(inputData, (1,len(inputData))))[0]
        print(predictionProbs, int(predictionProbs + 0.5))
        predictedIndex = int(predictionProbs + 0.5)
        self.currentState = self.blinkTypes[predictedIndex] 
        # Predict Data
  #      predictedIndex = predictionModel.predictData(np.array([inputData]))[0]
  #      self.currentState = self.blinkTypes[predictedIndex]
  #      print("\tThe Blink is ", self.currentState)

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


