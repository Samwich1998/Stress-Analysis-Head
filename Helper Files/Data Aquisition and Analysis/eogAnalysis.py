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
from  itertools import chain
# High/Low Pass Filters
from scipy.signal import butter
# Calibration Fitting
from scipy.optimize import curve_fit
# Plotting
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import skew
from scipy.stats import kurtosis




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
        
        self.xPeaksListTop = [[0] for _ in range(self.numChannels)]
        self.xPeaksListBottom = [[0] for _ in range(self.numChannels)]
        self.featureList = [[] for _ in range(self.numChannels)]
        self.triggerHolders = [[] for _ in range(self.numChannels)]
        self.lastTriggersAnalyzed = [[0,0] for _ in range(self.numChannels)]
        
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
        self.filteredBioelectricPlotAxes = [] 
        self.filteredBioelectricDataPlots = []
        self.trailingAveragePlots = []
        for channelIndex in range(self.numChannels):
            # Create Plot Axes
            self.filteredBioelectricPlotAxes.append(axes[channelIndex, 1])
            # Plot Flitered Peaks
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            self.trailingAveragePlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:blue", linewidth=1, alpha = 0.65)[0])
            
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
            
            # -------------------- Extract Eye Gestures  -------------------- #
            # Extarct EOG Peaks
            filteredDataX = self.data['timePoints'][-len(filteredData):]
            xPeaksNew, yPeaksNew, peakInds = self.findBlinks(filteredDataX, filteredData, channelIndex)
            
            # Extract Features from the Peaks
            newFeatures = self.extractFeatures(filteredDataX, filteredData, peakInds, channelIndex)

            """
            # Read in New Points and Check for Features
            for pointInd in range(self.lastTriggersAnalyzed[channelIndex][0] + 1, len(self.data['timePoints'])):
                # Extract X,Y Data
                timePoint = self.data['timePoints'][pointInd]
                dataPoint = filteredData[pointInd - (len(self.data['timePoints']) - len(filteredData))]
                # Check to See if the Gesture has Just Started
                if abs(dataPoint - self.steadyStateEye) > self.minVoltageThreshold:
                    if not self.lastTriggersAnalyzed[channelIndex][0] or len(self.triggerHolders[channelIndex][-1]) == 2:
                        # Store the Start of the Gesture
                        self.lastTriggersAnalyzed[channelIndex][0] = pointInd
                        self.triggerHolders[channelIndex].append([(self.lastTriggersAnalyzed[channelIndex][0], timePoint, dataPoint)])
                # Else, Check if the Gesture Has Just Finished
                elif self.lastTriggersAnalyzed[channelIndex][0] and len(self.triggerHolders[channelIndex][-1]) == 1:
                    # If the Peak Only Touches Above the Trigger Slightly
                    if pointInd - self.lastTriggersAnalyzed[channelIndex][0] < 100:
                        # Remove the Trigger
                        self.triggerHolders[channelIndex].pop()
                        continue
                    # Save the New Last Trigger
                    self.lastTriggersAnalyzed[channelIndex][1] = pointInd
                    # Save the Endpoint if Enough of a Peak is Showing
                    self.triggerHolders[channelIndex][-1].append((pointInd, timePoint, dataPoint))
                    print("HERE", self.triggerHolders[channelIndex][-1][0][0],self.triggerHolders[channelIndex][-1][1][0])
                    plt.plot(self.data['timePoints'], self.data['Channel' + str(channelIndex+1)])
                    plt.plot(self.data['timePoints'][self.triggerHolders[channelIndex][-1][0][0]], self.data['Channel' + str(channelIndex+1)][self.triggerHolders[channelIndex][-1][0][0]], 'o', markersize=5)
                    plt.plot(self.data['timePoints'][self.triggerHolders[channelIndex][-1][1][0]], self.data['Channel' + str(channelIndex+1)][self.triggerHolders[channelIndex][-1][1][0]], 'o', markersize=5)
                    plt.plot([0, 6], [self.steadyStateEye + self.minVoltageThreshold, self.steadyStateEye + self.minVoltageThreshold])
                    plt.plot([0, 6], [self.steadyStateEye - self.minVoltageThreshold, self.steadyStateEye - self.minVoltageThreshold])

                    leftBaselineIndex = self.findNearbyMinimum(self.data['Channel' + str(channelIndex+1)], self.triggerHolders[channelIndex][-1][0][0], binarySearchWindow = -25, maxPointsSearch = 1000)
                    rightBaselineIndex = self.findNearbyMinimum(self.data['Channel' + str(channelIndex+1)], self.triggerHolders[channelIndex][-1][1][0], binarySearchWindow = 25, maxPointsSearch = 1000)
                    plt.plot(self.data['timePoints'][leftBaselineIndex], self.data['Channel' + str(channelIndex+1)][leftBaselineIndex], 'o', markersize=5)
                    plt.plot(self.data['timePoints'][rightBaselineIndex], self.data['Channel' + str(channelIndex+1)][rightBaselineIndex], 'o', markersize=5)
                    
                    timePoints1 = self.data['timePoints'][-len(filteredData):]
                    plt.plot(timePoints1, filteredData)

                    plt.show()
            """            
            # --------------- Calibrate Angle Prediction Model -------------- #
            if calibrateModel:
                if self.calibrateChannelNum == channelIndex:
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
            # --------------------------------------------------------------- #    
        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #
            
        # -------------------- Update Virtual Reality  ---------------------- #
        if actionControl and self.predictEyeAngle[channelIndex] and not calibrateModel:
            actionControl.setGaze(eyeAngles)
        # ------------------------------------------------------------------- #

        # ------------------------ Predict Movement ------------------------- #

        # ------------------------------------------------------------------- #

        # ------------------------------------------------------------------- #
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
        
        
        if False:
            # Find Blink Parameters
            findNewPeak = True; findRightBaseline = False
            highestRecorderedBlinkX = 0
            threshold = 0.1; 
            peakDistance = 0.05;    # 50 ms Interval Between Peaks. See "BLINKER: Automated Extraction of Ocular Indices from EEG Enabling Large-Scale Analysis"
            minRiseTime = 0.0;
            maxRiseTime = 0.54;
            peakIndices = []
            leftBaselineIndexes = []
            rightBaselineIndexes = []
            
            runningMean = np.mean(yData)
            runningSTD = np.std(yData)
                 
            doubleBlinkMaxSep = 200
            
            initBlink = [[]];
            blinkPoints = [[]]
            
            addOn = 0; firstDer = [0]*addOn
            # Caluclate the Running Slope of the Data
            for peakInd in range(addOn, len(yData)):
             # Calculate the First Derivative
             deltaY = np.mean(yData[max(0,peakInd - addOn):peakInd+1]) - np.mean(yData[max(0,peakInd - 2*addOn - 1):peakInd-addOn+1])
             deltaX = max(xData[peakInd] - xData[max(0,peakInd-addOn - 1)], 10E-10)
             firstDer.append(deltaY/deltaX)
             
             # --------------------------------------------------------------- #
             peakLoc = xData[peakInd]
             # Track Blink as it Rises
             if findNewPeak and firstDer[-1] > threshold and peakLoc > highestRecorderedBlinkX + peakDistance:
                 # If it is the First Point, Label it as the Baseline
                 if not initBlink[-1]:
                     leftBaselineIndexes.append(peakInd)
                 # Record the Peak
                 initBlink[-1].append(firstDer[-1])
                 blinkPoints[-1].append(peakInd)
                 
             # Remove Blink if Too Many or Little Points
             elif initBlink[-1] and (maxRiseTime < xData[peakInd] - xData[leftBaselineIndexes[-1]] or xData[peakInd] - xData[leftBaselineIndexes[-1]] < minRiseTime):
                 # Reset Blink Detection Parameters
                 initBlink.pop(); initBlink.append([])
                 blinkPoints.pop(); blinkPoints.append([])
                 if leftBaselineIndexes:
                     leftBaselineIndexes.pop()
                 
             # Once the Slope is Negative, the Peak Rise is Complete
             elif findNewPeak and len(firstDer) > 3 and firstDer[-1] < firstDer[-3]:
                 # If There is a Peak to Detect
                 if initBlink[-1]:
                     if runningMean + 1.5*runningSTD < yData[peakInd]:
                         # Check if it is a Dobule Peak (Associated with the Last)
                         if peakIndices and peakInd - peakIndices[-1] < doubleBlinkMaxSep and yData[leftBaselineIndexes[-2]+2] < yData[leftBaselineIndexes[-1]]:
                             print("Double Peak")
                         if peakIndices:
                             ax = plt.axes(projection='3d')
                             ax.scatter3D([max(initBlink[-1])], [xData[peakIndices[-1]]-xData[leftBaselineIndexes[-1]]], [yData[peakIndices[-1]]-yData[leftBaselineIndexes[-1]]])
                         # Record the Peak
                         highestRecorderedBlinkX = peakLoc
                         peakIndices.append(peakInd)
                         # Stop Looking for a Peak
                         findNewPeak = False
                     else:
                         # Reset Blink Detection Parameters
                         initBlink.pop(); initBlink.append([])
                         blinkPoints.pop(); blinkPoints.append([])
                         if leftBaselineIndexes:
                             leftBaselineIndexes.pop()
             elif not findRightBaseline and not findNewPeak and firstDer[-1] < -threshold:
                 findRightBaseline = True
                 
                 
             # Once the Derivative GOes Negative, We can Start Looking Aga
             elif findRightBaseline and not findNewPeak and firstDer[-1] > -.01:
                 # If There is a Peak to Detect
                 peakWidth = peakLoc - xData[leftBaselineIndexes[-1]]; 
                 if abs(yData[peakInd] - yData[leftBaselineIndexes[-1]]) < (yData[peakIndices[-1]] - np.mean([yData[peakInd], yData[leftBaselineIndexes[-1]]]))/3:
                     peakWidth = peakLoc - xData[leftBaselineIndexes[-1]]
                 elif yData[peakInd] > yData[leftBaselineIndexes[-1]]:
                     peakWidth = (xData[peakIndices[-1]] - xData[leftBaselineIndexes[-1]])*1.5
                 else:
                     peakWidth = (peakLoc - xData[peakIndices[-1]])*1.5
            
                     
                     
                 if initBlink[-1] and peakWidth < 0.3:
                     highestRecorderedBlinkX = peakLoc
                     initBlink.append([])
                     blinkPoints.append([])
                     rightBaselineIndexes.append(peakInd)
                 else:
                     # Reset Blink Detection Parameters
                     initBlink.pop(); initBlink.append([])
                     blinkPoints.pop(); blinkPoints.append([])
                     if leftBaselineIndexes:
                         leftBaselineIndexes.pop()
                         peakIndices.pop()
                     
                 findNewPeak = True 
                 findRightBaseline = False
                    
                    
             # --------------------------------------------------------------- #
             
             # --------------------------------------------------------------- #
             if findNewPeak and firstDer[-1] > threshold and peakInd > highestRecorderedBlinkX + peakDistance:
                 i = 1
                 
            firstDer = np.array(firstDer)
            if True:
                plt.plot(xData, yData); plt.plot(xData, firstDer/20 + 3.3/2, 'o', markersize = 2)
                plt.show()
                if peakIndices:
                    print(np.diff(peakIndices))
                    xData = np.array(xData); yData = np.array(yData)
                    plt.plot(xData, yData); plt.plot(xData[peakIndices], yData[peakIndices], 'o', markersize = 2)
                    plt.plot(xData[leftBaselineIndexes], yData[leftBaselineIndexes], 'ro', markersize = 2)
                    plt.plot(xData[rightBaselineIndexes], yData[rightBaselineIndexes], 'ko', markersize = 2)
                    plt.show()
                
        # Find New Peak Indices and Last Recorded Peak's xLocation   
        peakIndices = scipy.signal.find_peaks(yData, prominence=.1, width=40, distance = 30)[0]; 
   #     xData = np.array(xData); yData = np.array(yData)
   #     plt.plot(xData, yData); plt.plot(xData[peakIndices], yData[peakIndices], 'o', markersize = 2)
   #     plt.show()

        # Find Where the New Peaks Begin
        xPeaksNew = []; yPeaksNew = []; peakInds = []
        for peakInd in peakIndices:
            xPeakLoc = xData[peakInd]
            # If it is a New Peak NOT Seen in This Channel
            if xPeakLoc not in self.xPeaksListTop[channelIndex] and xPeakLoc > self.xPeaksListTop[channelIndex][-1]:
                # Add the Peak
                xPeaksNew.append(xPeakLoc)
                yPeaksNew.append(yData[peakInd])
                peakInds.append(peakInd)

        # Return New Peaks and Their Baselines
        return xPeaksNew, yPeaksNew, peakInds

    def extractFeatures(self, xData, yData, peakInds, channelIndex):
        peakFeatures = []
        for xPointer in peakInds:
            peakFeatures.append([])
            xPeakLoc = xData[xPointer]
            # Take Average of the Signal (Only Left Side As I Want to Decipher Motor Intention as Fast as I Can; Plus the Signal is generally Symmetric)       
            if yData[xPointer] > self.steadyStateEye:
                leftBaselineIndex = self.findNearbyMinimum(yData, xPointer, binarySearchWindow = -40, maxPointsSearch = 1000)
                rightBaselineIndex = self.findNearbyMinimum(yData, xPointer, binarySearchWindow = 40, maxPointsSearch = 1000)
                if xPeakLoc not in self.xPeaksListTop[channelIndex] and (len(yData) - rightBaselineIndex > 40 or leftBaselineIndex > 40):
                    self.xPeaksListTop[channelIndex].append(xPeakLoc)
                else:
                    continue
            else:
                leftBaselineIndex = self.findNearbyMinimum(-yData, xPointer, binarySearchWindow = -40, maxPointsSearch = 1000)
                rightBaselineIndex = self.findNearbyMinimum(-yData, xPointer, binarySearchWindow = 40, maxPointsSearch = 1000)
                if xPeakLoc not in self.xPeaksListBottom[channelIndex] and len(yData) - rightBaselineIndex > 40 or leftBaselineIndex > 40:
                    self.xPeaksListBottom[channelIndex].append(xPeakLoc)
                else:
                    continue
            
            if False:
                plt.plot(xData[leftBaselineIndex], yData[leftBaselineIndex], 'bo', markersize=5)
                plt.plot(xData[rightBaselineIndex], yData[rightBaselineIndex], 'ro', markersize=5)
                plt.plot(xData[xPointer], yData[xPointer], 'ko', markersize=5)
                plt.plot(xData, yData)
                plt.show()
            
            peakAverage = np.mean(yData[leftBaselineIndex:xPointer+1])
            peakFeatures[-1].append(peakAverage)
        #    plt.plot(peakAnalysisData)
        #    plt.plot(xPointer, peakAnalysisData[xPointer], 'o')
        #    plt.plot(leftBaselineIndex, peakAnalysisData[leftBaselineIndex], 'o')
        #    plt.show()
        #    print(leftBaselineIndex, xPointer)
        # Return Features
        return peakFeatures

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
        
        addOn = 5; firstDer = [0]*addOn; skipPoints = 40;
        foundDrop = False; maxSlope = 0
        # Caluclate the Running Slope of the Data
        for peakInd in range(xPointer + searchDirection*(addOn+skipPoints), endSearch, searchDirection):
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
    
    addOn = 5; firstDer = [0]*addOn; skipPoints = 40;
    foundDrop = False; maxSlope = 0
    # Caluclate the Running Slope of the Data
    for peakInd in range(xPointer + searchDirection*(addOn+skipPoints), endSearch, searchDirection):
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
