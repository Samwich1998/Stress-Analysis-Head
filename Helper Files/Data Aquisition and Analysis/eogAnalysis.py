# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:17:05 2021
    conda install -c conda-forge ffmpeg

@author: Samuel Solomon
"""

# Basic Modules
import numpy as np
# Peak Detection
import scipy
import scipy.signal
# High/Low Pass Filters
from scipy.signal import butter
# Plotting
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# ------------------ User Can Edit (Global Variables) ----------------------- #

class eogProtocol:
    
    def __init__(self, numTimePoints = 2000, moveDataFinger = 200, numChannels = 4, samplingFreq = 800, movementOptions = [], plotStreamedData = True):
        
        # Input Parameters
        self.numChannels = numChannels            # Number of Bioelectric Signals
        self.numTimePoints = numTimePoints        # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger      # The Amount of Data to Stream in Before Finding Peaks
        self.movementOptions = movementOptions    # Gesture Movement Options
        self.plotStreamedData = plotStreamedData  # Plot the Data
        
        # High Pass Filter Parameters
        self.samplingFreq = samplingFreq          # Depends on the User's Hardware
        self.cutOffFreq = 8                       # Optimal LPF 6-8 Hz (Max 35 or 50); literature Claimed 7 Hz is Best
        
        # Data Collection Parameters
        self.minVoltageMovement = 0.1
        self.voltagePositionBuffer = 50   # Buffer to Find the Average Voltage
        self.peakDetectionBuffer = 500   # Buffer in Case Peaks are Only Half Formed at Edges
        self.bandPassBuffer = 1000        # Buffer in the Filtered Data that Represented BAD Filtering
        self.minGroupSep = 100            # Seperation that Defines a New Group
        
        # Start with Fresh Inputs
        self.resetGlobalVariables()
        
        # Define Class for Plotting Peaks
        if plotStreamedData:
            self.initPlotPeaks()

    def resetGlobalVariables(self):
        # Data to Read in
        self.data = {'timePoints':[]}
        for channelIndex in range(self.numChannels):
            self.data['Channel'+str(1+channelIndex)] = []
        
        # Reset Mutable Variables
        self.highestAnalyzedPeakX = 0
        
        # Reset Last Eye Voltage (Volts)
        self.currentEyeVoltages = [2.5 for _ in range(self.numChannels)]
        # Calibration Function for Eye Angle
        self.predictEyeAngle = [lambda inputVolts: inputVolts*12 - 30 for _ in range(self.numChannels)]
        
        # Close Any Opened Plots
        if self.plotStreamedData:
            plt.close()


    def initPlotPeaks(self): 

        # Specify Figure Asthetics
        self.peakCurrentRightColorOrder = {
            0: "tab:red",
            1: "tab:purple",
            2: "tab:orange",
            3: "tab:pink",
            4: "tab:brown",
            5: "tab:green",
            6: "tab:gray",
            7: "tab:cyan",
            }

        # use ggplot style for more sophisticated visuals
        plt.style.use('seaborn-poster') #sets the size of the charts
        #plt.style.use('ggplot')
        #plt.ion()

        # ------------------------------------------------------------------- #
        # --------- Plot Variables user Can Edit (Global Variables) --------- #

        # Specify Figure aesthetics
        figWidth = 14; figHeight = 10;
        self.fig, axes = plt.subplots(self.numChannels, 2, sharey=False, sharex = 'col', figsize=(figWidth, figHeight))
        
        # Plot the Raw Data
        yLimLow = 0; yLimHigh = 5; 
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
        
        # Hold Past Information
        self.trailingAverageData = {}
        for channelIndex in range(self.numChannels):
            self.trailingAverageData[channelIndex] = np.zeros(self.numTimePoints)
        
    
    def analyzeData(self, dataFinger, plotStreamedData = False, predictionModel = None, Controller=None):     
        
        # Get X Data: Shared Axis for All Channels
        self.timePoints = self.data['timePoints'][dataFinger:dataFinger + self.numTimePoints]
        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):
            
            # ------------------- Plot Biolectric Signal -------------------- #
            if plotStreamedData:
                # Get New Y Data
                newYData = self.data['Channel' + str(channelIndex+1)][dataFinger:dataFinger + self.numTimePoints]
                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.bioelectricDataPlots[channelIndex].set_data(self.timePoints, newYData)
                self.bioelectricPlotAxes[channelIndex].set_xlim(self.timePoints[0], self.timePoints[-1])
            # --------------------------------------------------------------- #
            
            # ---------------------- Band Pass Filter ----------------------- #    
            # Band Pass Filter to Remove Noise
            startBPFindex = max(dataFinger - self.bandPassBuffer, 0)
            yDataBuffer = self.data['Channel' + str(channelIndex+1)][startBPFindex:dataFinger + self.numTimePoints]
            filteredData = self.butterFilter(yDataBuffer, self.cutOffFreq, self.samplingFreq, order = 3, filterType = 'low')[-self.numTimePoints:]
            # --------------------------------------------------------------- #
            
            # --------------------- Predict Eye Movement  ------------------- #
            # Get the Current Voltage (Take Average)
            currentEyeVoltage = self.findTraileringAverage(filteredData[-self.voltagePositionBuffer:], deviationThreshold = self.minVoltageMovement)
            # Compare Voltage Difference to Remove Small Shakes
            if abs(currentEyeVoltage - self.currentEyeVoltages[channelIndex]) > self.minVoltageMovement:
                self.currentEyeVoltages[channelIndex] = currentEyeVoltage        
            
            # Predict the Eye's Degree
            eyeDegree = self.predictEyeAngle[channelIndex](self.currentEyeVoltages[channelIndex])
            
            # Plot the Eye's Angle
            if plotStreamedData:
                self.trailingAverageData[channelIndex] = np.append(self.trailingAverageData[channelIndex][self.moveDataFinger:], np.ones(self.moveDataFinger)*self.currentEyeVoltages[channelIndex])
                # Plot the Filtered Data
                self.filteredBioelectricDataPlots[channelIndex].set_data(self.timePoints, filteredData)
                self.trailingAveragePlots[channelIndex].set_data(self.timePoints, self.trailingAverageData[channelIndex])
                self.filteredBioelectricPlotAxes[channelIndex].set_xlim(self.timePoints[0], self.timePoints[-1]) 
                self.filteredBioelectricPlotAxes[channelIndex].legend(["Eye's Angle: " + "%.3g"%eyeDegree], loc="upper left")
            # --------------------------------------------------------------- #
            
            # --------------------- Move Virtual Reality  ------------------- #
            
            # --------------------------------------------------------------- #

        # Update to Get New Data Next Round
        if plotStreamedData:
            self.fig.show()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()

        # -------------------------------------------------------------------#
    

    def analyzeFullBatch(self, channelNum = 1):
        print("Printing Seperate test plots")
        # Get Data to Plot
        xData = self.data['timePoints']
        yData = self.data['Channel' + str(channelNum)]
        
        # Get Data and Filter
        plt.figure()
        plt.plot(xData,yData, c='tab:blue', alpha=0.7)
        plt.title("Bioelectric Data")
        
        plt.figure()
        filteredData = self.highPassFilter(yData)
        plt.plot(xData,filteredData, c='tab:blue', alpha=0.7)
        plt.title("Filtered Data")
        
        plt.figure()
        RMSData = self.RMSFilter(filteredData, self.window, self.step)
        plt.plot(xData[0:len(RMSData)],RMSData, c='tab:blue', alpha=0.7)
        plt.title("RMS Data")
        
        # Find Peaks
        batchTopPeaks = self.find_peaks(xData, RMSData)
        xPeakTop, yPeakTop, yBase = zip(*batchTopPeaks.items())
        featureLocsX, featureSet = self.featureDefinition(RMSData, xPeakTop, yBase, 0)
    
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

    def featureDefinition(self, RMSData, xTop, yBase, currentGroupNum, predictionModel = None, Controller = None):
        featureSet = []
        correctionTerm = 0
        # For Every Peak, Take the Average of the Points in Front of it
        for peakNum in range(len(xTop)):
            xTopLoc = xTop[peakNum]
            yBaseline = yBase[peakNum]
            # Get Peak's Points
            featureWindow = RMSData[xTopLoc - self.rmsEdgeBuffer:xTopLoc + self.rmsEdgeBuffer]
            # Take the Average of the Peak as the Feature
            if len(featureWindow) > 0:
                featureSet.append(np.mean(featureWindow) - yBaseline)
            # Edge Effect: np.mean([]) = NaN -> Ignore This Peak Until Next Round
            else:
                correctionTerm += 1
        
        # If No Features/Peaks This Round, Return Empty Dictionaries
        if featureSet == []:
            return {}, {}
            
        # Group the Feature Sets Peaks into One Bioelectric Signal/Group
        peakSeperation = np.diff(xTop[0:len(xTop) - correctionTerm]) # Identify New Signal by Peak Seperation
        peakGrouping = {currentGroupNum:[featureSet[0]]}  # Holder for the Features
        xGrouping = {currentGroupNum:[xTop[0]]}        # Holder for the Corresponding Peaks
        for i, peakSep in enumerate(peakSeperation):
            # A Part of Previous Group
            if peakSep < self.minGroupSep:
                peakGrouping[currentGroupNum].append(featureSet[i+1])
                xGrouping[currentGroupNum].append(xTop[i+1])
            # New Group
            else:
                goodData = self.createNewGroup(predictionModel, Controller)
                if goodData:
                    peakGrouping[currentGroupNum] = [featureSet[i+1]]
                    xGrouping[currentGroupNum] = [xTop[i+1]]
     
        return xGrouping, peakGrouping

    def findTraileringAverage(self, recentData, deviationThreshold = 0.08):
        # Base Case in No Points Came in
        if len(recentData) == 0:
            return 2.5
        
        # Keep Track of the trailingAverage
        trailingAverage = recentData[-1]
        for dataPointInd in range(0, len(recentData)-1, 5):
            # Get New dataPoint from the Back of the List
            dataPoint = recentData[len(recentData) - dataPointInd - 2]
            # If the dataPoint is Different from the trailingAverage by some Threshold, return the trailingAverage
            if abs(dataPoint - trailingAverage) > deviationThreshold:
                return trailingAverage
            else:
                numSamplesConsidered = dataPointInd + 2
                trailingAverage = (trailingAverage*(numSamplesConsidered) + dataPoint)/(numSamplesConsidered+1)
        # Else Return the Average
        return trailingAverage
            


    def findPeaks(self, xData, yData, channelIndex):
        # Find New Peak Indices and Last Recorded Peak's xLocation
        peakIndices = scipy.signal.find_peaks(yData, prominence=.1, height=0.01, rel_height=0.5)[0]
        
        # Find Where the New Peaks Begin
        xPeaksNew = []; yPeaksNew = []
        for peakInd in peakIndices:
            xPeakLoc = xData[peakInd]
            # A New Peak is AFTER the Last Recorded Peak + Buffer
            if xPeakLoc > self.highestAnalyzedPeakX + self.minGroupSep:
                xPeaksNew.append(xPeakLoc)
                yPeaksNew.append(yData[peakInd])

        # Return New Peaks and Their Baselines
        return xPeaksNew, yPeaksNew
    
    
    def removePeakBackground(self, xTop, yTop, RMSData):
        newY = []
        for pointNum in range(len(xTop)):
            baselineIndex = self.findLeftMinimum(RMSData, xTop[pointNum])
            yPoint = yTop[pointNum] - RMSData[baselineIndex]
            newY.append(yPoint)
        return newY
    
    def createNewGroup(self, predictionModel, Controller, goodData = True):
        # Get FeatureSet Point for Group
        groupFeatures = []
        for channel in range(self.numChannels):
            # Get the Features for the Group and Take the First One
            channelFeature = self.featureSetGrouping[channel][self.currentGroupNum]
            groupFeatures.append((channelFeature or [0])[0])
        featureArray = np.array([groupFeatures])
        
        # If the Feature is Bad, Throw it Away
        if len(featureArray[featureArray > 0]) <= 1 and np.sum(featureArray) <= 0.1:
            if predictionModel:
                print("Only One Small Signal Found; Not Moving Robot")
            self.currentGroupNum -= 1
            goodData = False
        # If it is Okay and We Have an ML Model, Predict the Movement
        elif predictionModel:
            self.predictMovement(predictionModel, featureArray, Controller)
            
        self.currentGroupNum += 1
        for channel in range(self.numChannels):
            self.featureLocsX[channel][self.currentGroupNum] = []
            self.featureSetGrouping[channel][self.currentGroupNum] = []
        
        return goodData
    
    def predictMovement(self, predictionModel, inputData, Controller = None):        
        # Predict Data
        predictedIndex = predictionModel.predictData(inputData)[0]
        predictedLabel = self.movementOptions[predictedIndex]
        print("The Predicted Label is", predictedLabel)
        if Controller:
            if predictedLabel == "left":
                Controller.moveLeft()
            elif predictedLabel == "right":
                Controller.moveRight()
            elif predictedLabel == "down":
                Controller.moveDown()
            elif predictedLabel == "up":
                Controller.moveUp()
            elif predictedLabel == "grab":
                Controller.grabHand()
            elif predictedLabel == "release":
                Controller.releaseHand()



 
