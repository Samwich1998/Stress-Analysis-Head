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
from scipy.signal import butter, lfilter
# Plotting
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# ------------------ User Can Edit (Global Variables) ----------------------- #

class emgProtocol:
    
    def __init__(self, numTimePoints = 2000, moveDataFinger = 200, numChannels = 4, samplingFreq = 800, movementOptions = [], plotStreamedData = False):
        
        # Input Parameters
        self.numChannels = numChannels        # Number of EMG Signals
        self.numTimePoints = numTimePoints                  # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger  # The Amount of Data to Stream in Before Finding Peaks
        self.movementOptions = movementOptions
        self.plotStreamedData = plotStreamedData
        # Data Holders
        self.previousDataRMS = {}
        
        # High Pass Filter Parameters
        f1 = 100; f3 = 50;
        self.samplingFreq = samplingFreq
        self.Rp = 0.1; self.Rs = 30;
        self.Wp = 2*math.pi*f1/self.samplingFreq
        self.Ws = 2*math.pi*f3/self.samplingFreq
        # Root Mean Squared (RMS) Parameters
        self.rmsWindow = 250; self.stepSize = 8;
        
        # Data Collection Parameters
        self.highPassBuffer = max(self.rmsWindow + self.stepSize, 5000)  # Must be > rmsWindow + stepSize
        self.peakDetectionBuffer = 500  # Buffer in Case Peaks are Only Half Formed at Edges
        self.numPointsRMS = 10000       # Number of Root Mean Squared Data (After HPF) to Plot
        self.minGroupSep = 100          # Seperation that Defines a New Group
        
        # Start with Fresh Inputs
        self.resetGlobalVariables()
        
        # Initialize Plots (If Displaying Plots)
        if plotStreamedData:
            self.initPlotPeaks()

    def resetGlobalVariables(self):
        # Data to Read in
        self.data = {'timePoints':[]}
        for channelIndex in range(self.numChannels):
            self.data['Channel'+str(1+channelIndex)] = []
            # Hold Analysis Values
            self.previousDataRMS[channelIndex] = []
        
        # Reset Mutable Variables
        self.highestAnalyzedPeakX = 0
        self.featureList = []
        
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
        plt.ion()

        # ------------------------------------------------------------------- #
        # --------- Plot Variables user Can Edit (Global Variables) --------- #

        # Specify Figure aesthetics
        figWidth = 14; figHeight = 10;
        self.fig, ax = plt.subplots(self.numChannels, 2, sharey=False, sharex = 'col', figsize=(figWidth, figHeight))
        
        # Plot the Raw Data
        yLimLow = 0; yLimHigh = 5; 
        xLimLow = 0; xLimHigh = xLimLow + self.numTimePoints;
        self.bioelectricDataPlots = []; self.bioelectricPlotAxes = []
        for channelNum in range(self.numChannels):
            # Create Plots
            self.bioelectricPlotAxes.append(ax[channelNum, 0])
            
            # Generate Plot
            self.bioelectricDataPlots.append(self.bioelectricPlotAxes[channelNum].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            
            # Set Figure Limits
            self.bioelectricPlotAxes[channelNum].set_xlim(xLimLow, xLimHigh)
            self.bioelectricPlotAxes[channelNum].set_ylim(yLimLow, yLimHigh)
            # Label Axis + Add Title
            self.bioelectricPlotAxes[channelNum].set_title("Bioelectric Signal in Channel " + str(channelNum + 1))
            self.bioelectricPlotAxes[channelNum].set_xlabel("Time (Seconds)")
            self.bioelectricPlotAxes[channelNum].set_ylabel("Bioelectric Signal (Volts)")
            
        yLimitHighFiltered = 0.5;
        # Create the Data Plots
        self.filteredBioelectricPlotAxes = [] 
        self.filteredBioelectricDataPlots = []
        for channelNum in range(self.numChannels):
            # Plot RMS Peaks
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelNum].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            
            # Create Plot Axes
            self.filteredBioelectricPlotAxes.append(ax[channelNum, 1])
            # Set Figure Limits
            self.filteredBioelectricPlotAxes[channelNum].set_ylim(yLimLow, yLimitHighFiltered)
            # Label Axis + Add Title
            self.filteredBioelectricPlotAxes[channelNum].set_title("Filtered Bioelectric Signal in Channel " + str(channelNum + 1))
            self.filteredBioelectricPlotAxes[channelNum].set_xlabel("Root Mean Squared Data Point")
            self.filteredBioelectricPlotAxes[channelNum].set_ylabel("Filtered Signal (Volts)")
            
        # Tighten Figure White Space (Must be After wW Add Fig Info)
        self.fig.tight_layout(pad=2.0); plt.show()
        
    
    def analyzeData(self, dataFinger, plotStreamedData = False, predictionModel = None, Controller=None):  
        
        xPeaksHolder = []; yPeaksHolder = []
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
            
            # ---------------------- High pass Filter ----------------------- #
            # Find New Points That Need Filtering
            totalPreviousPointsRMS = max(1 + math.floor((dataFinger + self.numTimePoints - self.moveDataFinger - self.rmsWindow) / self.stepSize), 0)
            dataPointerRMS = self.stepSize*totalPreviousPointsRMS
            # Add Buffer to New Points as HPF is Bad at Edge
            startHPF = max(dataPointerRMS - self.highPassBuffer, 0)
            
            # High Pass Filter to Remove Noise
            numNewDataForRMS = dataFinger + self.numTimePoints - dataPointerRMS
            yDataBuffer = self.data['Channel' + str(channelIndex+1)][startHPF:dataFinger + self.numTimePoints]
            filteredData = self.highPassFilter(yDataBuffer)[-(numNewDataForRMS):]   
            # --------------------------------------------------------------- #
    
            # --------------------- Root Mean Squared ----------------------- #
            # Calculated the RMS and Add the Data to the Stored Buffer from the Last Round
            totalCurrentPointsRMS = max(1 + math.floor((dataFinger + self.numTimePoints - self.rmsWindow) / self.stepSize), 0)
            dataRMS = self.RMSFilter(filteredData, self.previousDataRMS[channelIndex], self.rmsWindow, self.stepSize)            
            
            # Plot RMS Data
            xDataRMS = np.arange(max(totalCurrentPointsRMS - self.numPointsRMS,0), totalCurrentPointsRMS, 1)
            if plotStreamedData:
                savePointsRMS = max(self.numPointsRMS, self.peakDetectionBuffer + self.stepSize)
                self.filteredBioelectricDataPlots[channelIndex].set_data(xDataRMS, dataRMS[-self.numPointsRMS:])
                self.filteredBioelectricPlotAxes[channelIndex].set_xlim(xDataRMS[0], xDataRMS[0] + self.numPointsRMS)
            else:
                savePointsRMS = self.peakDetectionBuffer + self.stepSize
            # Store RMS Data Needed for Next Round
            self.previousDataRMS[channelIndex] = dataRMS[-savePointsRMS:]
            # --------------------------------------------------------------- #

            # ----------------------- Peak Detection ------------------------ #
            # Get Most Current RMS Data (Add Buffer in Case the peak is Cut Off)
            numNewPointsRMS = totalCurrentPointsRMS - totalPreviousPointsRMS
            bufferRMSData = dataRMS[-(numNewPointsRMS + self.peakDetectionBuffer):]
            bufferRMSDataX = xDataRMS[-(numNewPointsRMS + self.peakDetectionBuffer):]
            # Find Peaks from the New Data
            xPeaksNew, yPeaksNew = self.findPeaks(bufferRMSDataX, bufferRMSData, channelIndex)
            # Keep Track of Peaks
            
            xPeaksHolder.append(xPeaksNew); yPeaksHolder.append(yPeaksNew)
            # --------------------------------------------------------------- #
            
            # ---------------------- Feature Analysis  ---------------------- #
            # Extract Features from the Good Peaks 
            newFeatures = self.extractFeatures(bufferRMSDataX, xGoodPeaks)
            # Keep Track of the Features
            numFeatures = len(newFeatures[0])
            self.featureList[-1][numFeatures*channelIndex:numFeatures*(channelIndex+1)] = newFeatures
            # --------------------------------------------------------------- #
            
            
            self.highestAnalyzedPeakX
            self.featureList
            
            # -------------------- Group Peaks Together --------------------- #
            xGoodPeaks = [xPeaksNew[0]]; jumpDifference = 0
            # Group the Peaks and Take the FIRST Peak if Two are Nearby
            peakSeperation = np.diff(xPeaksNew); 
            # Identify New Movements by Peak Seperation
            for peakSepInd in range(len(peakSeperation)):
                peakSep = peakSeperation[peakSepInd]
                # Only Taking the First Peak, If the Peak Seperation is Big Enough
                if peakSep > self.minGroupSep - jumpDifference:
                    # Make a New Group
                    jumpDifference = 0
                    xGoodPeaks.append(xPeaksNew[peakSepInd + 1])
                else:
                    # Store Peak Seperation as We Look for the True Next Peak
                    jumpDifference += peakSep
            # --------------------------------------------------------------- #
            
            # ------------------------ Move Robot --------------------------- #
            # If New Peak Was Found with Enough Peak Seperation, Add Group 
            if abs(xDataRMS[-1] - self.highestAnalyzedPeakX) > self.minGroupSep and self.highestAnalyzedPeakX:
                self.createNewGroup(predictionModel, Controller)
            # --------------------------------------------------------------- #
            
            # ---------------------- Feature Analysis  ---------------------- #
            # Extract Features from the Good Peaks 
            newFeatures = self.extractFeatures(bufferRMSDataX, xGoodPeaks)
            # Keep Track of Features
            numFeatures = len(newFeatures)
            self.featureList[-1][numFeatures*channelIndex:numFeatures*(channelIndex+1)] = newFeatures
            
            goodData = self.createNewGroup(predictionModel, Controller)
            if goodData:
                self.featureList[channelIndex][self.currentGroupNum] = [featureSet[i+1]]
        
        
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
                self.featureList[channel][self.currentGroupNum] = []
            # --------------------------------------------------------------- #

            # ------------------------ Plot Peaks --------------------------- #
            if plotStreamedData:
                # Plot the Peaks; Colored by Grouping
                for groupNum in self.featureListLocX[channelIndex].keys():
                    # Check to See if the Group Has a Plot You Can Use
                    groupPeakPlot = self.movieGraphChannelTopPeaksList[channelIndex].get(groupNum, None)
                    # If None Availible, Create a New Plot to Add the Data
                    if not groupPeakPlot:
                        channelFiltered = self.filteredBioelectricPlotAxes[channelIndex]
                        # Color Code the Group Peaks. Wrap Around to First Index When Done
                        groupColor = (groupNum-1)%(len(self.peakCurrentRightColorOrder))
                        # Create a Plot for the Peaks Using its Respective Group's Color
                        groupPeakPlot = channelFiltered.plot([], [], 'o', c=self.peakCurrentRightColorOrder[groupColor], linewidth=1, alpha = 0.65)[0]
                        # Save the Plot for Later Use in the Group
                        self.movieGraphChannelTopPeaksList[channelIndex][groupNum] = groupPeakPlot
                    # Get Peak Points
                    if len(self.featureListLocX[channelIndex][groupNum]) != 0:
                        xPeaksNew = self.featureListLocX[channelIndex][groupNum][0]
                        if xDataRMS[0] <= xPeaksNew:
                            yPeaksNew = RMSData[-len(xDataRMS):][np.where(xDataRMS == xPeaksNew)[0][0]]
                            # Plot the Peaks in the Group
                            groupPeakPlot.set_data(xPeaksNew, yPeaksNew)
                        
        # Update to Get New Data Next Round
        if plotStreamedData:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        # -------------------------------------------------------------------#
        

    def analyzeFullBatch(self, channelNum = 1):
        print("Printing Seperate test plots")
        # Get Data to Plot
        xData = self.data['timePoints']
        yData = self.data['Channel' + str(channelNum)]
        
        # Get Data and Filter
        plt.figure()
        plt.plot(xData,yData, c='tab:blue', alpha=0.7)
        plt.title("EMG Data")
        
        plt.figure()
        filteredData = self.highPassFilter(yData)
        plt.plot(xData,filteredData, c='tab:blue', alpha=0.7)
        plt.title("Filtered Data")
        
        plt.figure()
        RMSData = self.RMSFilter(filteredData, [], self.window, self.step)
        plt.plot(xData[0:len(RMSData)],RMSData, c='tab:blue', alpha=0.7)
        plt.title("RMS Data")
        
        # Find Peaks
        batchTopPeaks = self.find_peaks(xData, RMSData)
        xPeaksNew, yPeaksNew, yBase = zip(*batchTopPeaks.items())
        featureListLocX, featureSet = self.extractFeatures(RMSData, xPeaksNew, yBase, 0)

# --------------------------------------------------------------------------- #
# ------------------------- Signal Analysis --------------------------------- #


    def highPassFilter(self, inputData):
        """
        data: Data to Filter
        f1: cutOffFreqPassThrough
        f3: cutOffFreqBand
        Rp: attDB (0.1)
        Rs: cutOffDB (30)
        samplingFreq: Frequecy You Take Data
        """
        [n, wn] = scipy.signal.cheb1ord(self.Wp/math.pi, self.Ws/math.pi, self.Rp, self.Rs)
        [bz1, az1] = scipy.signal.cheby1(n, self.Rp, self.Wp/math.pi, 'High')
        filteredData = lfilter(bz1, az1, inputData)
        return filteredData
    
    def RMSFilter(self, inputData, RMSData = [], rmsWindow=250, stepSize=8):
        """
        The Function loops through the given EMG Data, looking at batches of data
            of size rmsWindow at every interval seperated by stepSize.
        In Each Window, we take the magnitude of the data vector (sqrt[a^2+b^2]
            for [a,b] data point)
        A list of each root mean squared value is returned (in order)
        
        The Final List has a length of 1 + math.floor((len(inputData) - rmsWindow) / stepSize)
        --------------------------------------------------------------------------
        Input Variable Definitions:
            inputData: A List containing the  EMG Data
            rmsWindow: The Amount of Data in the Groups we Analyze via RMS
            stepSize: The Distance Between Data Groups
        --------------------------------------------------------------------------
        """
        # Initialize Starting Parameters
        normalization = math.sqrt(rmsWindow)
        numSteps = max(1 + math.floor((len(inputData) - rmsWindow) / stepSize), 0)
        # Take Root Mean Squared of Batch Data (numBatch = rmsWindow)
        for i in range(numSteps):
            # Get Data in the Window to take RMS
            inputWindow = inputData[i*stepSize:i*stepSize + rmsWindow]
            # Take RMS
            RMSData.append(np.linalg.norm(inputWindow, ord=2)/normalization)
        
        return RMSData     


    def findPeaks(self, xData, yData, channelIndex):
        # Find New Peak Indices and Last Recorded Peak's xLocation
        peakIndices = scipy.signal.find_peaks(yData, prominence=.03, height=0.01, width=15, rel_height=0.5, distance = 100)[0]
        
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
    
    def findNearbyMinimum(self, data, xPointer, binarySearchWindow = 50, maxPointsSearch = 2000):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        if abs(binarySearchWindow) < 1:
            return xPointer
        
        maxHeight = data[xPointer]; searchDirection = int(binarySearchWindow/abs(binarySearchWindow))
        # Binary Search Data to the Left to Find Minimum (Skip Over Small Bumps)
        for dataPointer in range(xPointer + binarySearchWindow, min(xPointer + searchDirection*maxPointsSearch, len(data)), -binarySearchWindow):
            # If the Point is Greater Than 9
            if data[dataPointer] > maxHeight:
                return self.findNearbyMinimum(data, dataPointer - binarySearchWindow, math.floor(binarySearchWindow/2), maxPointsSearch - (xPointer - dataPointer - binarySearchWindow))
            else:
                maxHeight = data[dataPointer]
        
        return xPointer
                
    def extractFeatures(self, peakAnalysisData, xPeaks):
        peakFeatures = []
        for xPeakInd in range(len(xPeaks)):
            peakFeatures.append([])
            xPeak = xPeaks[xPeakInd]
            # Take Average of the Signal (Only Left Side As I Want to Decipher Motor Intention as Fast as I Can; Plus the Signal is generally Symmetric)
            leftBaselineIndex = self.findNearbyMinimum(peakAnalysisData, xPeak, binarySearchWindow = -50, maxPeakSize = 2000)
            peakAverage = np.sum(peakAnalysisData[leftBaselineIndex:xPeakInd+1])/(xPeakInd + 1 - leftBaselineIndex)
            peakFeatures[-1].append(peakAverage)
        # Return Features
        return peakFeatures
    
    
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



 
