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
# Filtering Modules
from scipy.signal import savgol_filter
# Calibration Fitting
from scipy.optimize import curve_fit
# Baseline Subtraction
from BaselineRemoval import BaselineRemoval
# Plotting
import matplotlib
import matplotlib.pyplot as plt
# Feature Extraction
from scipy.stats import skew
from scipy.stats import entropy
from scipy.stats import kurtosis

# Import Files
import _filteringProtocols as filteringMethods # Import Files with Filtering Methods

# -------------------------------------------------------------------------- #
# -------------------- User Can Edit (Global Variables) -------------------- #

class ppgProtocol:
    
    def __init__(self, numTimePoints = 3000, moveDataFinger = 10, numChannels = 2, plotStreamedData = True, plotIndivisualPulses = True):
        
        # Input Parameters
        self.numChannels = numChannels            # Number of Bioelectric Signals
        self.numTimePoints = numTimePoints        # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger      # The Amount of Data to Stream in Before Finding Peaks
        self.plotStreamedData = plotStreamedData  # Plot the Data
        self.plotIndivisualPulses = plotIndivisualPulses  # Plot the Seperated Pulses
        
        # Define the Class with all the Filtering Methods
        self.filteringMethods = filteringMethods.filteringMethods()
        # Filtering Parameters
        self.samplingFreq = None                 # The Average Number of Points Steamed Into the Arduino Per Second; Depends on the User's Hardware; If NONE Given, Algorithm will Calculate Based on Initial Data
        self.bandPassBuffer = 2000               # I Found that [0.3-0.5, 5-10] as an optimal range. A Prepended Buffer in the Filtered Data that Represents BAD Filtering; Units: Points
        self.cutOffFreq_MultiPulse = [0.5, 10]   # I Found that 15-25 as an optimal range.
        self.cutOffFreq = 15                     # I Found that 15-25 as an optimal range.

        # Parameters that Define a Pulse
        self.maxBPM = 180                   # The Maximum Beats Per Minute for a Human; max = 480
        self.minBPM = 40                    # The Minimum Beats Per Minute for a Human; min = 27
        # Pulse Seperation Parameters
        self.bufferTime = 60/self.minBPM    # The Initial Wait Time Before we Start Labeling Peaks
        self.previousSystolicAmp = None
        
        # Feature Extarction Parameters
        self.averageFeatureWindow = 60*1.5  # Time Window to Average Features Together (Averaged with Past Features)

        # Prepare the Program to Begin Data Analysis
        self.checkParams                    # Check to See if the User's Input Parameters Make Sense
        self.resetGlobalVariables()         # Start with Fresh Inputs (Clear All Arrays/Values)
        
        # If Plotting, Define Class for Plotting Peaks
        if plotStreamedData and numChannels != 0:
            # Initialize Plots; NOTE: PLOTTING SLOWS DOWN PROGRAM!
            matplotlib.use('Qt5Agg')        # Set Plotting GUI Backend            
            self.initPlotPeaks()            # Create the Plots

    def resetGlobalVariables(self):
        # Data to Read in
        self.data = [ [], [[] for channel in range(self.numChannels)] ]
        
        # Pulse Seperation Parameters
        self.peakStandard = []
        for channelIndex in range(self.numChannels):
            self.peakStandard.append(0)     # The Max (negative) Second Deriviative of the Previous Pulse's Systolic Peak
        
        # Reset Feature Extraction
        self.featureList = []               # A List of Features Extracted
        self.featureListExact = []          # A List of Features Extracted Averaged Backwards
        
        # Peak Labeling Parameters
        self.lastAnalyzedPulseInd = [0 for channel in range(self.numChannels)]      # The Index of the Last Potential Pulse Analyzed from the Start of Data
        
        # Systolic and Diastolic References
        self.systolicPressure0 = None   # The Calibrated Systolic Pressure
        self.diastolicPressure0 = None  # The Calibrated Diastolic Pressure
        self.diastolicPressure = None   # The Current Diastolic Pressure
        
        # Blink Classification
        self.featureLabels = ['Relaxed', 'Stroop', 'Exercise', 'VR']
        self.currentState = self.featureLabels[0]
        
        # Save the Filtered Data
        self.filteredData = [[] for channel in range(self.numChannels)]
        
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
        
        # ------------------------------------------------------------------ #
        # --------- Plot Variables user Can Edit (Global Variables) -------- #

        # Specify Figure aesthetics
        figWidth = 20; figHeight = 15;
        self.fig, axes = plt.subplots(self.numChannels, 2, sharey=False, sharex = 'col', figsize=(figWidth, figHeight))
        
        # Plot the Raw Data
        yLimLow = 0; yLimHigh = 1; 
        self.bioelectricPlotAxes = []
        self.bioelectricDataPlots = []
        for channelIndex in range(self.numChannels):
            # Create Plots
            if self.numChannels == 1:
                self.bioelectricPlotAxes.append(axes[channelIndex])
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
        self.filteredBioelectricPlotAxes = [] 
        self.filteredBioelectricDataPlots = []
        for channelIndex in range(self.numChannels):
            # Create Plot Axes
            if self.numChannels == 1:
                self.filteredBioelectricPlotAxes.append(axes[channelIndex])
            else:
                self.filteredBioelectricPlotAxes.append(axes[channelIndex, 1])
            # Plot Flitered Peaks
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])

            # Set Figure Limits
            self.filteredBioelectricPlotAxes[channelIndex].set_ylim(yLimLow, yLimHigh)
            # Label Axis + Add Title
            self.filteredBioelectricPlotAxes[channelIndex].set_title("Filtered Bioelectric Signal in Channel " + str(channelIndex + 1))
            self.filteredBioelectricPlotAxes[channelIndex].set_xlabel("Time (Seconds)")
            self.filteredBioelectricPlotAxes[channelIndex].set_ylabel("Filtered Signal (Volts)")
            
        # Tighten Figure White Space (Must be After wW Add Fig Info)
        self.fig.tight_layout(pad=2.0);
    
    def analyzeData(self, dataFinger, plotStreamedData = False, predictionModel = None, actionControl = None, calibrateModel = False):     
        
        import time     
        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):
            t1 = time.time()
            
            # ---------------------- Filter the Data ----------------------- #    
            # Get the Data with a Buffer for the Filters
            startBPFindex = max(dataFinger - self.bandPassBuffer, 0)
            yDataBuffer = self.data[1][channelIndex][startBPFindex:dataFinger + self.numTimePoints].copy()
            # Invert the Y-Data for Reflection Spectroscopy
            filteredData = max(yDataBuffer) - np.array(yDataBuffer)
            
            # Get the Sampling Frequency from the First Batch (If Not Given)
            if not self.samplingFreq:
                self.samplingFreq = len(self.data[0][startBPFindex:])/(self.data[0][-1] - self.data[0][startBPFindex])
                print("\tSetting PPG Sampling Frequency to", self.samplingFreq)
                print("\tFor Your Reference, If Data Analysis is Longer Than", self.moveDataFinger/self.samplingFreq, ", Then You Will NOT be Analyzing in Real Time")
                
                # Estimate that Defines the Number of Points in a Pulse
                self.minPointsPerPulse = math.floor(self.samplingFreq*60/self.maxBPM)
                self.maxPointsPerPulse = math.ceil(self.samplingFreq*60/self.minBPM)

            # Filter the Data: Low pass Filter and Savgol Filter
            filteredData = self.filteringMethods.fourierFilter.removeFrequencies(filteredData, self.samplingFreq, self.cutOffFreq_MultiPulse)
            filteredData = savgol_filter(filteredData, 11, 2, mode='nearest', deriv=0)
            
            # Account for the Pointer Moving
            self.lastAnalyzedPulseInd[channelIndex] = max(self.lastAnalyzedPulseInd[channelIndex], startBPFindex)
            # Cut the Data to Only Analyze New Pulses
            filteredData = np.array(filteredData[self.lastAnalyzedPulseInd[channelIndex]-startBPFindex:])
            timePoints = np.array(self.data[0][self.lastAnalyzedPulseInd[channelIndex]:dataFinger + self.numTimePoints])
            
            # Add a Buffer of Zeros to the Filtered Data
            self.filteredData[channelIndex].extend([0]*(len(self.data[0]) - len(self.filteredData[channelIndex])))
            # -------------------------------------------------------------- #
            
            # --------------------- Seperate the Pulses -------------------- # 
            # Calculate Derivatives
            firstDeriv = savgol_filter(filteredData, 7, 2, mode='nearest', deriv=1)
            # Take First Derivative of Smoothened Data
            systolicPeaks = self.seperatePulses(timePoints, firstDeriv, channelIndex)
            
            # plt.plot(timePoints, max(yDataBuffer) - np.array(yDataBuffer)[-len(filteredData):], 'k', linewidth=2)
            # plt.plot(timePoints, filteredData, 'tab:red', linewidth=1)
            # plt.plot(timePoints[systolicPeaks], filteredData[systolicPeaks], 'o')
            # plt.show()
            # -------------------------------------------------------------- #

            # ----------------------- Pulse Analysis ----------------------- #
            # Seperate Peaks Based on the Minimim Before the R-Peak Rise
            if len(systolicPeaks) != 0:
                if self.lastAnalyzedPulseInd[channelIndex] == 0:
                    pulseStartInd = systolicPeaks[0]
                else:
                    pulseStartInd = 0
                startDataIndex = self.lastAnalyzedPulseInd[channelIndex]
                for pulseNum in range(len(systolicPeaks)):                    
                    pulseEndInd = self.findNearbyMinimum(filteredData, systolicPeaks[pulseNum], binarySearchWindow = -2, maxPointsSearch=self.maxPointsPerPulse)
                    # Keep track of which data was already analyzed
                    self.lastAnalyzedPulseInd[channelIndex] = startDataIndex + pulseEndInd
                    
                    # ------------------- Cull Bad Pulses ------------------ #
                    # Check if the Peak was Double Counted
                    if pulseEndInd == pulseStartInd:
                        # print("Found the Same Peak", pulseEndInd, pulseStartInd, self.maxPointsPerPulse, timePoints[pulseStartInd])
                        pulseStartInd = pulseEndInd; continue                    # Check if the Pulse is Too Big: Likely Double Pulse
                    elif pulseEndInd - pulseStartInd > self.maxPointsPerPulse:
                        # print("Pulse Too Big", pulseEndInd, pulseStartInd, self.maxPointsPerPulse, timePoints[pulseStartInd])
                        pulseStartInd = pulseEndInd; continue
                    # Check if the Pulse is Too Small; Likely Not an R-Peak
                    elif pulseEndInd - pulseStartInd < self.minPointsPerPulse:
                        # print("Pulse Too Small", pulseEndInd, pulseStartInd, self.minPointsPerPulse, timePoints[pulseStartInd])
                        pulseStartInd = pulseEndInd; continue
                    # ------------------------------------------------------ #
                                        
                    # ----------------- Pulse Preprocessing ---------------- #
                    # Extract Indivisual Pulse Data
                    pulseTime = timePoints[pulseStartInd:pulseEndInd+1]
                    pulseData = filteredData[pulseStartInd:pulseEndInd+1]
                    # Filter the Pulse
                    # pulseData = self.filteringMethods.bandPassFilter.butterFilter(pulseData, self.cutOffFreq, self.samplingFreq, order = 3, filterType = 'low')
                    pulseData = savgol_filter(pulseData, max(3, self.convertToOddInt(len(pulseData)/8)), 2, mode='nearest')
                    # Normalize the Pulse's Baseline to Zero
                    normalizedPulse = self.normalizePulseBaseline(pulseTime, pulseData, polynomialDegree = 1, fastBaselineRemoval = True)
                    
                    numWrongSideOfTangent = len(normalizedPulse[normalizedPulse < 0])
                    # Cull Pulses with Bad Normalization
                    if len(normalizedPulse)/5 < numWrongSideOfTangent:
                        pulseStartInd = pulseEndInd; continue
                    # Cull Pulses whose Amplitude Changes Abnormally. Likely Motion Artifact
                    elif self.previousSystolicAmp != None and 2*self.previousSystolicAmp < max(normalizedPulse) < 0.5*self.previousSystolicAmp:
                        pulseStartInd = pulseEndInd; continue
                    self.previousSystolicAmp = max(normalizedPulse)

                    # Calculate the Pulse Derivatives
                    pulseVelocity = savgol_filter(normalizedPulse, 3, 2, mode='nearest', deriv=1)
                    pulseAcceleration = savgol_filter(normalizedPulse, 3, 2, mode='nearest', deriv=2)
                    thirdDeriv = savgol_filter(pulseAcceleration, 3, 1, mode='nearest', deriv=1)
                    # ------------------------------------------------------ #
                    
                    # --------------- Extract Pulse Features --------------- #
                    # Save the Filtered Data
                    self.filteredData[channelIndex][startDataIndex+pulseStartInd:startDataIndex+pulseEndInd+1] = normalizedPulse
                    
                    # Extract Features from the Pulse Data
                    # self.extractPulsePeaks(pulseTime, normalizedPulse, pulseVelocity, pulseAcceleration, thirdDeriv)
                    # ------------------------------------------------------ #
                    # Reset the Pulse Data Finger
                    pulseStartInd = pulseEndInd
            elif 5 < timePoints[-1] - self.data[0][self.lastAnalyzedPulseInd[channelIndex]]:
                self.peakStandard[channelIndex] = self.peakStandard[channelIndex]/1.5

            # -------------------------------------------------------------- #

            t2 = time.time()
            # print("TIME", channelIndex, t2-t1)
            # ------------------- Plot Biolectric Signals ------------------ #
            if plotStreamedData and not calibrateModel:
                # Compile the Data to Show on the Plot
                timePoints = self.data[0][dataFinger:dataFinger + self.numTimePoints]
                newYData = np.array(self.data[1][channelIndex][dataFinger:dataFinger + self.numTimePoints])
                newFilteredData = np.array(self.filteredData[channelIndex][-len(timePoints):])

                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.bioelectricDataPlots[channelIndex].set_data(timePoints, (newYData - min(newYData))/((max(newYData) - min(newYData))))
                self.bioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1])
                            
                # Plot the Good Filtered Pulses
                self.filteredBioelectricDataPlots[channelIndex].set_data(timePoints, newFilteredData/max(newFilteredData))
                self.filteredBioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1]) 
            # -------------------------------------------------------------- #   

        # -------------------------- Update Plots -------------------------- #
        # Update to Get New Data Next Round
        if plotStreamedData and not calibrateModel and self.numChannels != 0:
            self.fig.show()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        # -------------------------------------------------------------------#

# -------------------------------------------------------------------------- #
# ----------------------------- Signal Analysis ---------------------------- #

    def convertToOddInt(self, x):
        return 2*math.floor((x+1)/2) - 1

    def seperatePulses(self, time, firstDeriv, channelIndex):
        self.peakStandardInd = 0
        # Take First Derivative of Smoothened Data
        peaks = [];
        for pointInd in range(self.minPointsPerPulse, len(firstDeriv)):
            # Calcuate the Derivative at pointInd
            firstDerivVal = firstDeriv[pointInd]
            
            # If the Derivative Stands Out, Its the Systolic Peak
            if self.peakStandard[channelIndex]*0.5 < firstDerivVal:
                # Use the First Few Peaks as a Standard
                if (self.bufferTime + self.data[0][0] < time[pointInd]) and self.minPointsPerPulse < pointInd:

                    # If the Point is Sufficiently Far Away, its a New R-Peak
                    if self.peakStandardInd + self.minPointsPerPulse < pointInd:
                        peaks.append(pointInd)
                    # Else, Find the Max of the Peak
                    elif len(peaks) != 0 and firstDeriv[peaks[-1]] < firstDeriv[pointInd]:
                        peaks[-1] = pointInd
                    # Else, Dont Update Pointer
                    else:
                        continue
                    self.peakStandardInd = pointInd
                    if firstDerivVal*0.5 < self.peakStandard[channelIndex]:
                        self.peakStandard[channelIndex] = firstDerivVal
                elif self.peakStandard[channelIndex] == 0 or firstDerivVal*0.5 < self.peakStandard[channelIndex]:
                    self.peakStandard[channelIndex] = max(self.peakStandard[channelIndex], firstDerivVal)

        return peaks
    
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
    
    def normalizePulseBaseline(self, xData, yData, polynomialDegree, fastBaselineRemoval = False):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            pulseData:  yData-Axis Data for a Single Pulse (Start-End)
            polynomialDegree: Polynomials Used in Baseline Subtraction
        Output Parameters:
            pulseData: yData-Axis Data for a Baseline-Normalized Pulse (Start, End = 0)
        Use Case: Shift the Pulse to the xData-Axis (Removing non-Horizontal Base)
        Assumption in Function: pulseData is Positive
        ----------------------------------------------------------------------
        Further API Information Can be Found in the Following Link:
        https://pypi.org/project/BaselineRemoval/
        ----------------------------------------------------------------------
        """
        if fastBaselineRemoval:
            # Draw a Linear Line Between the Points
            lineSlope = (yData[0] - yData[-1])/(xData[0] - xData[-1])
            slopeIntercept = yData[0] - lineSlope*xData[0]
            linearFit = lineSlope*xData + slopeIntercept
            # Remove the Baseline
            yData = yData - linearFit
        else:
            # Perform Baseline Removal Twice to Ensure Baseline is Gone
            for _ in range(2):
                # Baseline Removal Procedure
                baseObj = BaselineRemoval(yData)  # Create Baseline Object
                yData = baseObj.ModPoly(polynomialDegree) # Perform Modified multi-polynomial Fit Removal
            
        # Return the Data With Removed Baseline
        return yData
    

# -------------------------------------------------------------------------- #
# --------------------------- Feature Extraction --------------------------- #
    
    def extractPulsePeaks(self, pulseTime, normalizedPulse, pulseVelocity, pulseAcceleration, thirdDeriv):
        
        # ----------------------- Detect Systolic Peak ---------------------- #        
        # Find Systolic Peak
        systolicPeakInd = self.findNearbyMaximum(normalizedPulse, 0, binarySearchWindow = 4, maxPointsSearch = len(pulseTime))
        # Find UpStroke Peaks
        systolicUpstrokeVelInd = self.findNearbyMaximum(pulseVelocity, 0, binarySearchWindow = 1, maxPointsSearch = systolicPeakInd)
        systolicUpstrokeAccelMaxInd = self.findNearbyMaximum(pulseAcceleration, systolicUpstrokeVelInd, binarySearchWindow = -1, maxPointsSearch = systolicPeakInd)
        systolicUpstrokeAccelMinInd = self.findNearbyMinimum(pulseAcceleration, systolicUpstrokeVelInd, binarySearchWindow = 1, maxPointsSearch = systolicPeakInd)
        # ------------------------------------------------------------------- #
                
        # ---------------------- Detect Tidal Wave Peak --------------------- #     
        bufferToTidal = self.findNearbyMinimum(thirdDeriv, systolicPeakInd+1, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        # Find Tidal Peak Beginning
        tidalStartInd = self.findNearbyMaximum(thirdDeriv, bufferToTidal+2, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        tidalStartInd_OPTION2 = self.findNearbyMaximum(pulseAcceleration, systolicPeakInd, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        # Find Tidal Peak
        tidalPeakInd = self.findNearbyMinimum(thirdDeriv, min(tidalStartInd_OPTION2, tidalStartInd+1), binarySearchWindow = 4, maxPointsSearch = int(len(pulseTime)/2))
        # Find Tidal Peak Ending
        tidalEndInd = self.findNearbyMaximum(thirdDeriv, tidalPeakInd+1, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        # ------------------------------------------------------------------- #
        
        # ----------------------  Detect Dicrotic Peak ---------------------- #
        dicroticNotchInd = self.findNearbyMinimum(normalizedPulse, tidalEndInd, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        dicroticPeakInd = self.findNearbyMaximum(normalizedPulse, dicroticNotchInd, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        
        # Other Extremas Nearby
        dicroticInflectionInd = self.findNearbyMaximum(pulseVelocity, dicroticNotchInd, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        dicroticFallVelMinInd = self.findNearbyMinimum(pulseVelocity, dicroticInflectionInd, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        # ------------------------------------------------------------------- #
        
        def plotIt(badReason = ""):
            normalizedPulse1 = normalizedPulse/max(normalizedPulse)
            pulseVelocity1 = pulseVelocity/ max(pulseVelocity)
            pulseAcceleration1 = pulseAcceleration/max(pulseAcceleration)
            thirdDeriv1 = thirdDeriv/max(thirdDeriv)
            
            plt.plot(pulseTime, normalizedPulse1, linewidth = 2, color = "black")
            plt.plot(pulseTime, pulseVelocity1, alpha=0.5)
            plt.plot(pulseTime, pulseAcceleration1, alpha=0.5)
            plt.plot(pulseTime, thirdDeriv1, alpha=0.5)
            #plt.plot(pulseTime, fourthDeriv1, alpha=0.5)

            plt.plot(pulseTime[systolicPeakInd], normalizedPulse1[systolicPeakInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeVelInd], normalizedPulse1[systolicUpstrokeVelInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeAccelMaxInd], normalizedPulse1[systolicUpstrokeAccelMaxInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeAccelMinInd], normalizedPulse1[systolicUpstrokeAccelMinInd],  'ko')

            plt.plot(pulseTime[tidalStartInd], normalizedPulse1[tidalStartInd],  'ro')
            plt.plot(pulseTime[tidalPeakInd], normalizedPulse1[tidalPeakInd],  'go')
            plt.plot(pulseTime[tidalEndInd], normalizedPulse1[tidalEndInd],  'ro')
            
            plt.plot(pulseTime[dicroticNotchInd], normalizedPulse1[dicroticNotchInd],  'bo')
            plt.plot(pulseTime[dicroticPeakInd], normalizedPulse1[dicroticPeakInd],  'bo')
            
            plt.plot(pulseTime[[dicroticInflectionInd, dicroticFallVelMinInd]], normalizedPulse1[[dicroticInflectionInd, dicroticFallVelMinInd]],  'bo')
            
            plt.title("Time: " + str(pulseTime[-1]) + "; " + badReason)
            plt.show()

        # ------------------------- Cull Bad Pulses ------------------------- #
        # Check The Order of the Systolic Peaks
        if not systolicUpstrokeAccelMaxInd < systolicUpstrokeVelInd < systolicUpstrokeAccelMinInd < systolicPeakInd:
            print("\t\tBad Systolic Sequence. Time = ", pulseTime[-1]); 
            plotIt("SYSTOLIC")
            return None
        # Check The Order of the Tidal Peaks
        elif not tidalPeakInd < tidalEndInd:
            print("\t\tBad Tidal Sequence. Time = ", pulseTime[-1]); 
            plotIt("TIDAL")
            return None
        # Check The Order of the Dicrotic Peaks
        elif not dicroticNotchInd < dicroticInflectionInd < dicroticPeakInd < dicroticFallVelMinInd:
            print("\t\tBad Dicrotic Sequence. Time = ", pulseTime[-1]); 
            plotIt("DICROTIC")
            return None
        # Check The Order of the Peaks
        elif not systolicPeakInd < tidalEndInd < dicroticNotchInd - 2:
            print("\t\tBad Peak Sequence. Time = ", pulseTime[-1]); 
            plotIt("GENERAL")
            return None
        
        # Check If the Dicrotic Peak was Skipped
        if pulseTime[-1]*0.75 < pulseTime[dicroticPeakInd] - pulseTime[systolicUpstrokeAccelMaxInd]:
            print("\t\tDicrotic Peak Likely Skipped Over. Time = ", pulseTime[-1]);
            return None
        # ------------------------------------------------------------------- #
        
        # ----------------------- Feature Extraction ------------------------ #
        allSystolicPeaks = [systolicUpstrokeAccelMaxInd, systolicUpstrokeVelInd, systolicUpstrokeAccelMinInd, systolicPeakInd]
        allTidalPeaks = [tidalPeakInd, tidalEndInd]
        allDicroticPeaks = [dicroticNotchInd, dicroticInflectionInd, dicroticPeakInd, dicroticFallVelMinInd]
        
        # Extract the Pulse Features
        # self.extractFeatures(normalizedPulse, pulseTime, pulseVelocity, pulseAcceleration, allSystolicPeaks, allTidalPeaks, allDicroticPeaks)
        # ------------------------------------------------------------------- #

        if self.plotIndivisualPulses:
            normalizedPulse1 = normalizedPulse/max(normalizedPulse)
            pulseVelocity1 = pulseVelocity/ max(pulseVelocity)
            pulseAcceleration1 = pulseAcceleration/max(pulseAcceleration)
            thirdDeriv1 = thirdDeriv/max(thirdDeriv)
            
            plt.plot(pulseTime, normalizedPulse1, linewidth = 2, color = "black")
            plt.plot(pulseTime, pulseVelocity1, alpha=0.5)
            plt.plot(pulseTime, pulseAcceleration1, alpha=0.5)
            plt.plot(pulseTime, thirdDeriv1, alpha=0.5)
            
            plt.plot(pulseTime[systolicPeakInd], normalizedPulse1[systolicPeakInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeVelInd], normalizedPulse1[systolicUpstrokeVelInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeAccelMaxInd], normalizedPulse1[systolicUpstrokeAccelMaxInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeAccelMinInd], normalizedPulse1[systolicUpstrokeAccelMinInd],  'ko')

            plt.plot(pulseTime[tidalStartInd], normalizedPulse1[tidalStartInd],  'ro')
            plt.plot(pulseTime[tidalPeakInd], normalizedPulse1[tidalPeakInd],  'ro')
            plt.plot(pulseTime[tidalEndInd], normalizedPulse1[tidalEndInd],  'ro')

            plt.plot(pulseTime[dicroticNotchInd], normalizedPulse1[dicroticNotchInd],  'bo')
            plt.plot(pulseTime[dicroticPeakInd], normalizedPulse1[dicroticPeakInd],  'bo')
            
            plt.plot(pulseTime[[dicroticInflectionInd, dicroticFallVelMinInd]], normalizedPulse1[[dicroticInflectionInd, dicroticFallVelMinInd]],  'bo')
            
            plt.title("Time: " + str(pulseTime[-1]))
            plt.show()
            
            
            