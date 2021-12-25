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
from scipy.stats import entropy
from scipy.stats import kurtosis


from scipy.signal import savgol_filter


# --------------------------------------------------------------------------- #
# ------------------ User Can Edit (Global Variables) ----------------------- #

class eogProtocol:
    
    def __init__(self, numTimePoints = 3000, moveDataFinger = 10, numChannels = 2, plotStreamedData = True):
        
        # Input Parameters
        self.numChannels = numChannels            # Number of Bioelectric Signals
        self.numTimePoints = numTimePoints        # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger      # The Amount of Data to Stream in Before Finding Peaks
        self.plotStreamedData = plotStreamedData  # Plot the Data
        
        # High Pass Filter Parameters
        self.samplingFreq = None          # The Average Number of Points Steamed Into the Arduino Per Second; Depends on the User's Hardware; If NONE Given, Algorithm will Calculate Based on Initial Data
        self.bandPassBuffer = 5000        # A Prepended Buffer in the Filtered Data that Represents BAD Filtering; Units: Points
        self.cutOffFreq = 25              # Optimal LPF Cutoff in Literatrue is 6-8 or 20 Hz (Max 35 or 50); I Found 25 Hz was the Best for My System
        
        # Eye Angle Determination Parameters
        self.voltagePositionBuffer = 100  # A Prepended Buffer to Find the Current Average Voltage; Units: Points
        self.minVoltageMovement = 0.06    # The Minimum Voltage Change Required to Register an Eye Movement; Units: Volts
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
        if plotStreamedData:
            # Initialize Plots; NOTE: PLOTTING SLOWS DOWN PROGRAM!
            matplotlib.use('Qt5Agg') # Set Plotting GUI Backend            
            self.initPlotPeaks()     # Create the Plots

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
        self.featureList = []
        self.featureListExact = []
        # Reset Blink Indices
        self.singleBlinksX = []
        self.multipleBlinksX = []
        self.blinksXLocs = []
        self.blinksYLocs = []
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
        self.eyeBlinkLocPlots = []
        self.trailingAveragePlots = []
        self.filteredBioelectricDataPlots = []
        self.filteredBioelectricPlotAxes = [] 
        for channelIndex in range(self.numChannels):
            # Create Plot Axes
            self.filteredBioelectricPlotAxes.append(axes[channelIndex, 1])
            # Plot Flitered Peaks
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            self.trailingAveragePlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:blue", linewidth=1, alpha = 0.65)[0])
            self.eyeBlinkLocPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], 'o', c="tab:blue", markersize=7, alpha = 0.65)[0])

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
                print("\tSetting Sampling Frequency to", self.samplingFreq)
            
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
                # Get the New Data Where No Blinks Have Been Found Yet
                findBlinkDataY = filteredData[- (len(self.data['timePoints']) - self.lastAnalyzedPeakInd):] 
                filteredDataX = self.data['timePoints'][-len(findBlinkDataY):] 
                # Find the Blinks in the New Data
                self.findBlinks(filteredDataX, findBlinkDataY, len(self.data['timePoints']) - len(findBlinkDataY), predictionModel)
            # --------------------------------------------------------------- #
            
            # --------------------- Calibrate Eye Angle --------------------- #
            if calibrateModel and self.calibrateChannelNum == channelIndex:
                argMax = np.argmax(filteredData)
                argMin = np.argmin(filteredData)
                earliestExtrema = argMax if argMax < argMin else argMin
                
                timePoints = self.data['timePoints'][dataFinger:dataFinger + self.numTimePoints]
                plt.plot(timePoints, filteredData)
                plt.plot(timePoints[earliestExtrema], filteredData[earliestExtrema], 'o', linewidth=3)
                plt.show()
                
                self.calibrationVoltages[self.calibrateChannelNum].append(np.average(filteredData[earliestExtrema:earliestExtrema + 20]))
            # --------------------------------------------------------------- #
            
            # ------------------- Plot Biolectric Signals ------------------- #
            if plotStreamedData and not calibrateModel:
                # Get X Data: Shared Axis for All Channels
                timePoints = np.array(self.data['timePoints'][dataFinger:dataFinger + self.numTimePoints])# - self.data['timePoints'][0]

                # Get New Y Data
                newYData = self.data['Channel' + str(channelIndex+1)][dataFinger:dataFinger + self.numTimePoints]
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
    
    def findBlinks(self, xData, yData, lastAnalyzedBuffer, predictionModel):
        minBaselinePoints = 20
        minPeakHeight = 0.11   # No Less Than 0.11
        multPeakSepMax = 0.5   # No Less Than 0.25
        
        # Find All Potential Blinks in the Data
        peakIndices = scipy.signal.find_peaks(yData, prominence=1E-8, width=50)[0];
        # Extract the True Blinks and Their Features
        for peakInd in peakIndices:
            peakLocX = xData[peakInd]
            
            # ------------------ Find the Blink's Baselines ----------------- #
            # Calculate the Left and Right Baseline of the Peak
            leftBaselineIndex = self.findBaselineIndex(xData, yData, peakInd, searchDirection = -1)
            rightBaselineIndex = self.findBaselineIndex(xData, yData, peakInd, searchDirection = 1)
            # Dont Analyze if Blink is Not Fully Formed
            if rightBaselineIndex == xData[-1]:
                print("Too Soon to Analyze ", peakLocX)
                continue
            
            
            # If the Peak is Too Small, Remove the Peak
            if yData[peakInd] - yData[leftBaselineIndex] < minPeakHeight or yData[peakInd] - yData[rightBaselineIndex] < minPeakHeight:
                #print("Too Small", peakLocX)
                continue
            # If No Baseline is Found, Ignore the Blink (Too Noisy, Probably Not a Blink)
            elif leftBaselineIndex >= peakInd - minBaselinePoints or rightBaselineIndex <= peakInd + minBaselinePoints:
                print("Bad Baseline", peakLocX)
                continue

            # --------------------------------------------------------------- #
            
            # -------------------- Extract Blink Features ------------------- #
            self.lastAnalyzedPeakInd = lastAnalyzedBuffer + peakInd
            print("Last Analyzed: ", self.data['timePoints'][self.lastAnalyzedPeakInd])
            newFeatures = self.extractFeatures(xData[leftBaselineIndex:rightBaselineIndex+1], yData[leftBaselineIndex:rightBaselineIndex+1].copy(), peakInd-leftBaselineIndex)
            
            # Remove Peaks with Bad Blink Features
            if len(newFeatures) == 0:
                print("\tNo Features", peakLocX)
                continue
            # Else, Add the New Features
            self.featureListExact.append(newFeatures)
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
            # Average the Last Few Blink Features
            self.featureList.append(newFeatures)
            #self.featureList.extend(np.mean(np.array(self.featureListExact)[self.blinksXLocs[-1] > peakLocX - self.averageBlinkWindow], axis=0))
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
    
    def findIntersectionPoint(self, leftLineParams, rightLineParams):
        xPoint = (rightLineParams[1] - leftLineParams[1])/(leftLineParams[0] - rightLineParams[0])
        yPoint = leftLineParams[0]*xPoint + leftLineParams[1]
        return xPoint, yPoint

    def organizeDerivPeaks(self, dy_dt_ABS, dy_dt3_ABS, peakInd, velIndsTotal, accelIndsTotal, thirdDerivIndsTotal):    
        # Take the Highest Indices
        try:
            velInds = [max(velIndsTotal[velIndsTotal < peakInd], key = lambda velInd: dy_dt_ABS[velInd])]
            velInds.append(max(velIndsTotal[velIndsTotal > peakInd], key = lambda velInd: dy_dt_ABS[velInd]))
        except:
            print("\tNo Velocity Inds on Both Sides of Peak")
            return [],[],[]
        # Verify That the Index is Correct, Else Remove the Peak (Improper Blink)
        if len(accelIndsTotal[accelIndsTotal < velInds[0]]) == 0:
            return [],[],[]
        
        # Get the Correct Accel Inds
        newIndsAccel = [accelIndsTotal[accelIndsTotal < velInds[0]][-1]]
        newIndsAccel.append(accelIndsTotal[accelIndsTotal > velInds[0]][0])
        
        accelIndsAfterPeak = accelIndsTotal[accelIndsTotal > peakInd]
        if len(accelIndsAfterPeak[accelIndsAfterPeak < velInds[1]]) != 0:
            newIndsAccel.append(accelIndsAfterPeak[accelIndsAfterPeak < velInds[1]][-1])
        else:
            newIndsAccel.append(peakInd + dy_dt3_ABS[peakInd:velInds[1]].argmin())
        if len(accelIndsTotal[accelIndsTotal > velInds[1]]) != 0:
            newIndsAccel.append(accelIndsTotal[accelIndsTotal > velInds[1]][0])
        else:
            return [], [], []
        
        # Get the Correct Third Deriv Inds
        thirdDerivBeforePeak = thirdDerivIndsTotal[thirdDerivIndsTotal < peakInd]
        thirdDerivBeforePeak = thirdDerivBeforePeak[thirdDerivBeforePeak > newIndsAccel[1]]
        
        thirdDerivInds = []
        if len(thirdDerivBeforePeak) == 0:
            thirdDerivInds.append(velInds[0])
            thirdDerivInds.append(newIndsAccel[1])
        elif len(thirdDerivBeforePeak) == 1:
            thirdDerivInds.extend(np.sort([thirdDerivBeforePeak[0], newIndsAccel[1]]))
        else:
            thirdDerivInds.extend(thirdDerivBeforePeak[0:2])
        
        return velInds, newIndsAccel, thirdDerivInds
    
    def quantifyPeakShape(self, xData, yData, peakInd):
        
        # Calculate Derivatives
        dx_dt = np.gradient(xData); dx_dt2 = np.gradient(dx_dt); 
        dy_dt = np.gradient(yData); dy_dt2 = np.gradient(dy_dt); dy_dt3 = np.gradient(dy_dt2); 
        dy_dt_ABS = abs(dy_dt); dy_dt2_ABS = abs(dy_dt2); dy_dt3_ABS = abs(dy_dt3)
        
        # Calculate Peak Shape parameters
        speed = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        acceleration = np.sqrt(dx_dt2 * dx_dt2 + dy_dt2 * dy_dt2)
        curvature = np.abs((dx_dt2 * dy_dt - dx_dt * dy_dt2)) / speed**3  # Units 1/Volts
        scaledCurvature = curvature/max(curvature)
        # Save Blink Shpaes
        self.importantArrays.append([speed, acceleration, curvature, scaledCurvature])
        
        # Return the Velocity, Acceleration, and Curvature
        return dy_dt_ABS, dy_dt2_ABS, dy_dt3_ABS, curvature


    def extractFeatures(self, xData, yData, peakInd):

        # ------------------- Subtract the Blink's Baseline ----------------- #
        # Check if the Baseline is Skewed. A Skewed Peak is Probably Eye Movement Alongside a Blink
        baseLinesSkewed = abs(yData[-1] - yData[0]) > 0.2*(yData[peakInd] - max(yData[-1], yData[0]))
        badBaseline = abs(yData[-1] - yData[0]) > 0.6*(yData[peakInd] - max(yData[-1], yData[0]))
        if baseLinesSkewed:
            # If Skewed, Take the Minimum Point as the Baseline
            peakBaselineY = min(yData[-1], yData[0])
            #print("\tSkewed Baseline")
            #return []
        elif badBaseline:
            print("\tBaseline is WAY Too Skewed ... Unsure What to Do")
            return []
        else:
            # If Proper Baseline, Take a Linear Fit of the Base Points
            delY = yData[-1] - yData[0]; delX = xData[-1] - xData[0]
            peakBaselineY = yData[0] + (delY/delX)*(xData[peakInd] - xData[0])
        # Subtract the Baseline
        yData -= peakBaselineY
        # Smoothen the Data Just a Little
        yData = savgol_filter(yData, 51, 2)
        # ------------------------------------------------------------------- #
        
        # ----------------- Calculate the Peak's Derivatives ---------------- #
        # Calculate Speed, Acceleration, Curvature
        dy_dt_ABS, dy_dt2_ABS, dy_dt3_ABS, curvature = self.quantifyPeakShape(xData, yData, peakInd)
        
        # Find the Derivatives' Peaks
        velIndsTotal = scipy.signal.find_peaks(dy_dt_ABS, prominence=10E-15, width=15, height=np.mean(dy_dt_ABS)/4)[0];
        accelIndsTotal = scipy.signal.find_peaks(dy_dt2_ABS, prominence=10E-20, width=10)[0];
        thirdDerivIndsTotal = scipy.signal.find_peaks(dy_dt3_ABS, prominence=10E-20, width=10)[0];

        # If Not Enough Velocty/Acceleration Peaks Found, Cull the Blink
        if len(velIndsTotal) >= 2 and len(accelIndsTotal) >= 4:
            # For Good Derivative, Pull Out the Important Peaks
            velInds, accelInds, thirdDerivInds = self.organizeDerivPeaks(dy_dt_ABS, dy_dt3_ABS, peakInd, velIndsTotal, accelIndsTotal, thirdDerivIndsTotal) 
            
            # If Not Enough Good Peaks, Cull the Blink
            if len(accelInds) < 4:
                print("\tNo Accel Inds")
                return []
        else:
            print("\tBad Deriv Inds", len(accelIndsTotal))
            return []
        # ------------------------------------------------------------------- #
        
        # --------------------- Find the Blink's Endpoints ------------------ #
        # Linearly Fit the Peak's Base
        startBlinkLineParams = np.polyfit(xData[accelInds[0]: velInds[0]], yData[accelInds[0]: velInds[0]], 1)
        endBlinkLineparams1 = np.polyfit(xData[velInds[1]: accelInds[3]], yData[velInds[1]: accelInds[3]], 1)
        
        # Calculate the New Baseline of the Peak
        startBlinkX, _ = self.findIntersectionPoint([0, 0], startBlinkLineParams)
        endBlinkX, _ = self.findIntersectionPoint(endBlinkLineparams1, [0, 0])
        # Calculate the New Baseline's Index
        startBlinkInd = np.argmin(abs(xData - startBlinkX))
        endBlinkInd = np.argmin(abs(xData - endBlinkX))
        #
        if accelInds[-1] >= endBlinkInd:
            print("\tCannot Find Acccel Ind")
            return []
        # ------------------------------------------------------------------- #
        
        # ------------------------------------------------------------------- #
        # -------------------- Extract Amplitude Features ------------------- #
        # Find Peak's Tent Features
        peakTentX, peakTentY = self.findIntersectionPoint(startBlinkLineParams, endBlinkLineparams1)
        tentDeviationX = peakTentX - xData[peakInd]
        tentDeviationY = peakTentY - yData[peakInd]
        # Find Blink Amplitude Features
        blinkHeight = yData[peakInd]                 # Distance from the Peak to the Baseline
        blinkAmpRatio = blinkHeight/peakTentY
        # Calculate Tent Ratios
        tentDeviationRatio = tentDeviationY/tentDeviationX
        # Other Ampltiude Ratios
        closingAmpDiff1 = yData[accelInds[0]] - blinkHeight
        closingAmpDiff2 = yData[velInds[0]] - blinkHeight
        closingAmpDiff3 = yData[accelInds[1]] - blinkHeight
        openingAmpDiff1 = yData[accelInds[2]] - blinkHeight
        openingAmpDiff2 = yData[velInds[1]] - blinkHeight
        openingAmpDiff3 = yData[accelInds[3]] - blinkHeight
        # Other Ampltiude Ratios
        closingAmpDiffRatio1 = (yData[accelInds[0]] - blinkHeight)/blinkHeight
        closingAmpDiffRatio2 = (yData[velInds[0]] - blinkHeight)/blinkHeight
        closingAmpDiffRatio3 = (yData[accelInds[1]] - blinkHeight)/blinkHeight
        openingAmpDiffRatio1 = (yData[accelInds[2]] - blinkHeight)/blinkHeight
        openingAmpDiffRatio2 = (yData[velInds[1]] - blinkHeight)/blinkHeight
        openingAmpDiffRatio3 = (yData[accelInds[3]] - blinkHeight)/blinkHeight
        # Other Deviation Ratios
        accel0ToVel0Ratio = (yData[velInds[0]] - yData[accelInds[0]])
        accel1ToVel0Ratio = (yData[accelInds[1]] - yData[velInds[0]])
        accel2ToVel1Ratio = (yData[accelInds[2]] - yData[velInds[1]])
        accel3ToVel1Ratio = (yData[velInds[1]] - yData[accelInds[3]])
        # Percent Amplitude Ratios
        riseTimePercent = (yData[accelInds[1]] - yData[accelInds[0]])
        dropTimePercent = (yData[accelInds[2]] - yData[accelInds[3]])
        velDiffPercent = (yData[velInds[1]] - yData[velInds[0]])
        # Pure Amplitude Ratios
        closingAmpRatio1 = (yData[accelInds[0]])
        closingAmpRatio2 = (yData[velInds[0]])
        closingAmpRatio3 = (yData[accelInds[1]])
        openingAmpRatio1 = (yData[accelInds[2]])
        openingAmpRatio2 = (yData[velInds[1]])
        openingAmpRatio3 = (yData[accelInds[3]])
        # ------------------------------------------------------------------- #
        
        # -------------------- Extract Duration Features -------------------- #
        # Find the Standard Blink Durations
        blinkDuration = endBlinkX - startBlinkX      # The Total Time of the Blink
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
        closingSlope0 = np.polyfit(xData[startBlinkInd: velInds[0]], yData[startBlinkInd: velInds[0]], 1)[0]
        closingSlope1 = startBlinkLineParams[0]
        closingSlope2 = np.polyfit(xData[thirdDerivInds[0]:thirdDerivInds[1]], yData[thirdDerivInds[0]:thirdDerivInds[1]], 1)[0]
        openingSlope1 = np.polyfit(xData[peakInd: velInds[1]], yData[peakInd: velInds[1]], 1)[0]
        openingSlope2 = endBlinkLineparams1[0]
        openingSlope3 = np.polyfit(xData[accelInds[3]: endBlinkInd], yData[accelInds[3]: endBlinkInd], 1)[0]
        
        # Slope Differences
        closingSlopeDiff10 = closingSlope1 - closingSlope0
        closingSlopeDiff12 = closingSlope1 - closingSlope2
        openingSlopeDiff21 = openingSlope2 - openingSlope1
        openingSlopeDiff23 = openingSlope2 - openingSlope3
        
        # Slope Ratios
        closingSlopeRatio0 = closingSlope0/blinkHeight
        closingSlopeRatio1 = closingSlope1/blinkHeight
        closingSlopeRatio2 = closingSlope2/blinkHeight
        openingSlopeRatio1 = openingSlope1/blinkHeight
        openingSlopeRatio2 = openingSlope2/blinkHeight
        openingSlopeRatio3 = openingSlope3/blinkHeight
        # ------------------------------------------------------------------- #
        
        # ---------------------- Extract Shape Features --------------------- #
        # Calculate Peak Shape Parameters
        peakAverage = np.mean(yData)
        peakAverageRatio = peakAverage/blinkHeight
        peakIntegral = np.sum(yData)
        peakIntergralRatio = peakIntegral/blinkHeight
        peakEntropy = entropy(yData-min(yData)+10E-10)
        peakSkew = skew(yData, bias=False)
        peakKurtosis = kurtosis(yData, fisher=True, bias = False)
        peakSTD = np.std(yData, ddof=1)
        maxCurvature = max(curvature[max(0, peakInd - 25): peakInd + 25])
        
        # Curvature Around Main Points
        curvatureYDataAccel0 = curvature[accelInds[0]]
        curvatureYDataAccel1 = curvature[accelInds[1]]
        curvatureYDataAccel2 = curvature[accelInds[2]]
        curvatureYDataAccel3 = curvature[accelInds[3]]
        curvatureYDataVel0 = curvature[velInds[0]]
        curvatureYDataVel1 = curvature[velInds[1]]
        
        # Stanard Deviation
        velFullSTD = np.std(dy_dt_ABS, ddof=1)
        accelFullSTD = np.std(dy_dt2_ABS, ddof=1)
        thirdDerivFullSTD = np.std(dy_dt3_ABS, ddof=1)
        velSTD = np.std(dy_dt_ABS[velInds[1]:], ddof=1)
        accelSTD = np.std(dy_dt2_ABS[velInds[1]:], ddof=1)
        thirdDerivSTD = np.std(dy_dt3_ABS[velInds[1]:], ddof=1)
        
        # Entropy
        velFullEntropy = entropy(dy_dt_ABS+10E-50)
        accelFullEntropy = entropy(dy_dt2_ABS+10E-50)
        thirdDerivFullEntropy = entropy(dy_dt3_ABS+10E-50)
        velEntropy = entropy(dy_dt_ABS[velInds[1]:]+10E-50)
        accelEntropy = entropy(dy_dt2_ABS[velInds[1]:]+10E-50)
        thirdDerivEntropy = entropy(dy_dt3_ABS[velInds[1]:]+10E-50)
        # ------------------------------------------------------------------- #
        
        # -------------------- Extract Derivative Features ------------------ #
        # Extract Normalized Blink Velocities
        peakClosingVel = dy_dt_ABS[velInds[0]]/blinkHeight
        peakOpeningVel = dy_dt_ABS[velInds[1]]/blinkHeight
        # Extract Normalized Blink Acceleration
        peakClosingAccel1 = dy_dt2_ABS[accelInds[0]]/blinkHeight
        peakClosingAccel2 = dy_dt2_ABS[accelInds[1]]/blinkHeight
        peakopeningAccel1 = dy_dt2_ABS[accelInds[2]]/blinkHeight
        peakopeningAccel2 = dy_dt2_ABS[accelInds[3]]/blinkHeight
        # Extract Amplitude Ratios
        velClosedRatio = yData[velInds[0]]/blinkHeight
        velOpenRatio = yData[velInds[1]]/blinkHeight
        accelClosedRatio1 = yData[accelInds[0]]/blinkHeight
        accelClosedRatio2 = yData[accelInds[1]]/blinkHeight
        accelOpenRatio1 = yData[accelInds[2]]/blinkHeight
        accelOpenRatio2 = yData[accelInds[3]]/blinkHeight
        # Extract Amplitudes
        velClosedVal = dy_dt_ABS[velInds[0]]
        velOpenVal = dy_dt_ABS[velInds[1]]
        accelClosedVal1 = dy_dt2_ABS[accelInds[0]]
        accelClosedVal2 = dy_dt2_ABS[accelInds[1]]
        accelOpenVal1 = dy_dt2_ABS[accelInds[2]]
        accelOpenVal2 = dy_dt2_ABS[accelInds[3]]
        # Ratio
        velRatio = dy_dt_ABS[velInds[0]]/dy_dt_ABS[velInds[1]]
        accelRatio1 = dy_dt2_ABS[accelInds[0]]/dy_dt2_ABS[accelInds[1]]
        accelRatio2 = dy_dt2_ABS[accelInds[2]]/dy_dt2_ABS[accelInds[3]]
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

        # Divide the Peak by Acceleration
        startToAccel = xData[accelInds[0]] - startBlinkX
        accelCloseingPeakDuration = xData[accelInds[1]] - xData[accelInds[0]]
        accelToPeak = xData[peakInd] - xData[accelInds[1]]
        peakToAccel = xData[accelInds[2]] - xData[peakInd]
        accelOpeningPeakDuration = xData[accelInds[3]] - xData[accelInds[2]]
        accelToEnd = endBlinkX - xData[accelInds[3]]
        # Divide the Peak by Velocity
        velPeakDuration = xData[velInds[1]] - xData[velInds[0]]
        startToVel = xData[velInds[0]] - startBlinkX
        velToPeak = xData[peakInd] - xData[velInds[0]]
        peakToVel = xData[velInds[1]] - xData[peakInd]
        velToEnd = endBlinkX - xData[velInds[1]]
        # ------------------------------------------------------------------- #

        # ------------------ Consolidate the Blink Features ----------------- #
        # Finalize the Features
        featureList = [xData[peakInd], blinkHeight, peakTentY, tentDeviationX, tentDeviationY, blinkAmpRatio, tentDeviationRatio]
        featureList.extend([closingAmpDiff1, closingAmpDiff2, closingAmpDiff3, openingAmpDiff1, openingAmpDiff2, openingAmpDiff3])
        featureList.extend([closingAmpDiffRatio1, closingAmpDiffRatio2, closingAmpDiffRatio3, openingAmpDiffRatio1, openingAmpDiffRatio2, openingAmpDiffRatio3])
        featureList.extend([accel0ToVel0Ratio, accel1ToVel0Ratio, accel2ToVel1Ratio, accel3ToVel1Ratio])
        featureList.extend([riseTimePercent, dropTimePercent, velDiffPercent])
        featureList.extend([closingAmpRatio1, closingAmpRatio2, closingAmpRatio3, openingAmpRatio1, openingAmpRatio2, openingAmpRatio3])


        featureList.extend([blinkDuration, closingTime, openingTime, closingFraction, openingFraction, halfClosedTime, eyesClosedTime, percentTimeClosed])
        featureList.extend([closingSlope0, closingSlope1, closingSlope2, openingSlope1, openingSlope2, openingSlope3])
        featureList.extend([closingSlopeDiff10, closingSlopeDiff12, openingSlopeDiff21, openingSlopeDiff23])
        featureList.extend([closingSlopeRatio0, closingSlopeRatio1, closingSlopeRatio2, openingSlopeRatio1, openingSlopeRatio2, openingSlopeRatio3])
        featureList.extend([peakAverage, peakAverageRatio, peakIntegral, peakIntergralRatio, peakEntropy, peakSkew, peakKurtosis, peakSTD, maxCurvature])
        # Compile the Features
        featureList.extend([peakClosingVel, peakOpeningVel, peakClosingAccel1, peakClosingAccel2, peakopeningAccel1, peakopeningAccel2])
        featureList.extend([velOpenRatio, velClosedRatio, accelClosedRatio1, accelClosedRatio2, accelOpenRatio1, accelOpenRatio2])
        featureList.extend([velClosedVal, velOpenVal, accelClosedVal1, accelClosedVal2, accelOpenVal1, accelOpenVal2])
        featureList.extend([velRatio, accelRatio1, accelRatio2, velRatioYData, accelRatioYData1, accelRatioYData2])
        featureList.extend([durationByVel1, durationByVel2, durationByAccel1, durationByAccel2, durationByAccel3, midDurationRatio])
        featureList.extend([startToAccel, accelCloseingPeakDuration, accelToPeak, peakToAccel, accelOpeningPeakDuration, accelToEnd])
        featureList.extend([velPeakDuration, startToVel, velToPeak, peakToVel, velToEnd])
        featureList.extend([curvatureYDataAccel0, curvatureYDataAccel1, curvatureYDataAccel2, curvatureYDataAccel3, curvatureYDataVel0, curvatureYDataVel1])
        featureList.extend([velFullSTD, accelFullSTD, thirdDerivFullSTD, velSTD, accelSTD, thirdDerivSTD])
        featureList.extend([velFullEntropy, accelFullEntropy, thirdDerivFullEntropy, velEntropy, accelEntropy, thirdDerivEntropy])
        # ------------------------------------------------------------------- #


        # ------------------------- Cull Bad Blinks ------------------------- #
        # If the Blink is Shorter Than 50ms or Longer Than 500ms, Ignore the Blink (Probably Eye Movement)
        if 0.6 < blinkDuration or blinkDuration < 0.05:
            print("\tBad Blink Duration:", blinkDuration)
            return []
        # If the Closing Time is Longer Than 150ms, it is Probably Not an Involuntary Blink
        elif closingTime > 0.25:
            print("\tBad Closing Time:", closingTime)
            return []
        elif 0.35 < openingTime:
            print("\tBad Opening Time:", closingTime)
            return []
        elif 4 < accelRatio1:
            print("\tBad accelRatio1:", accelRatio1)
            return []
        elif 0.5 < accelToEnd:
            print("\tBad accelToEnd:", accelToEnd)
            return []
        elif peakSkew < -1:
            print("\tBad peakSkew:", peakSkew)
            return []   
        elif 6 < velRatio:
            print("\tBad velRatio:", velRatio)
            return []     
        elif 0.5 < velToEnd:
            print("\tBad velToEnd:", velToEnd)
            return []    
        elif 5 < midDurationRatio:
            print("\tBad midDurationRatio:", midDurationRatio)
            return []    
        elif 60 < curvatureYDataAccel0:
            print("\tBad curvatureYDataAccel0:", curvatureYDataAccel0)
            return []  
        elif 500 < abs(tentDeviationRatio):
            print("\tBad tentDeviationRatio:", tentDeviationRatio)
            return []    
        # ------------------------------------------------------------------- #
        
        # --------------------- Cull Voluntary Blinks  ---------------------- #
        if tentDeviationX < -0.02 or tentDeviationX > 0.1:
            print("\tWINK! tentDeviationX = ", tentDeviationX, " xLoc = ", xData[peakInd]); return []  
        elif tentDeviationY > 0.2 or tentDeviationY < 0:
            print("\tWINK! tentDeviationY = ", tentDeviationY); return []  
        elif closingSlope0 > 11:
            print("\tWINK! closingSlope0 = ", closingSlope0); return []  
        elif closingSlope1 > 12:
            print("\tWINK! closingSlope1 = ", closingSlope1); return []  
        elif closingSlope2 > 12:
            print("\tWINK! closingSlope2 = ", closingSlope2); return []  
        elif openingSlope2 < -7:
            print("\tWINK! openingSlope2 = ", openingSlope2); return []  
        elif openingSlope3 < -7:
            print("\tWINK! openingSlope3 = ", openingSlope3); return []  
        elif accelToPeak > 0.06:
            print("\tWINK! accelToPeak = ", accelToPeak); return []  
        elif blinkAmpRatio < 0.75:
            print("\tWINK! blinkAmpRatio = ", blinkAmpRatio); return []
        elif maxCurvature > 750:
            print("\tWINK! maxCurvature = ", maxCurvature); return []
        elif 200 < abs(tentDeviationRatio):
            print("\WINK! tentDeviationRatio:", tentDeviationRatio)
            return []           # ------------------------------------------------------------------- #
        
        if 40 < tentDeviationRatio:
            print("\tBad tentDeviationRatio:", tentDeviationRatio)
            return []   
        elif tentDeviationX > 0:
            print("\tWINK! tentDeviationX > 0; tentDeviationX = ", tentDeviationX); return []
            return []
        elif closingSlopeDiff12 > 5 or closingSlopeDiff12 < -1:
            print("\tWINK! closingSlopeDiff12 > 5 or < -1; closingSlopeDiff12 = ", closingSlopeDiff12); return []
            return []
        
        
        # sepInds = [startBlinkInd, accelInds[0], accelInds[1], peakInd, accelInds[3], endBlinkInd]
        # self.plotData(xData, yData, peakInd, velInds = velInds, accelInds = accelInds, sepInds = sepInds, title = "Dividing the Blink")

        # plt.plot(xData, yData/max(yData), 'k', linewidth=2)
        # plt.plot(xData, dy_dt_ABS*0.8/max(dy_dt_ABS), 'r', linewidth=1)
        # plt.plot(xData, dy_dt2_ABS*0.8/max(dy_dt2_ABS), 'b', linewidth=1)
        # plt.plot(xData, dy_dt3_ABS*0.5/max(dy_dt3_ABS), 'm', linewidth=.3)
        # plt.legend(['Blink', 'Velocity ABS', 'Acceleration ABS'])
        # plt.show()
        
      
        return featureList
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
            try:
                deltaY = np.mean(yData[max(0,peakInd - addOn):peakInd+1]) - np.mean(yData[max(0,peakInd - 2*addOn - 1):peakInd-addOn+1])
            except:
                continue
            deltaX = max(xData[peakInd] - xData[max(0,peakInd-addOn - 1)], 10E-10)
            firstDeriv = deltaY/deltaX
            
            # Verify Major Slope Drop
            if abs(firstDeriv) > 0.5:
                foundDrop = True
                maxSlope = max(maxSlope, abs(firstDeriv))
            
            if foundDrop and abs(firstDeriv) < maxSlope/10:
                return self.findNearbyMinimum(yData, peakInd, binarySearchWindow = searchDirection*20, maxPointsSearch = 50) #peakInd
        return xPointer
    
    
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


