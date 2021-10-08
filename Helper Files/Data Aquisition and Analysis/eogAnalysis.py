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
# Filter
from scipy.signal import butter, lfilter
# Plotting
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# ------------------ User Can Edit (Global Variables) ----------------------- #

class eogProtocol:
    
    def __init__(self, numTimePoints = 2000, moveDataFinger = 200, numChannels = 4, movementOptions = [], plotStreamedData = True):
        
        # Input Parameters
        self.numChannels = numChannels        # Number of Bioelectric Signals
        self.numTimePoints = numTimePoints                  # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger  # The Amount of Data to Stream in Before Finding Peaks
        self.movementOptions = movementOptions
        self.plotStreamedData = plotStreamedData
        
        # Data to Stream in
        self.data = {}
        # Peak Finding and Feature Holders
        self.RMSDataList = {}
        self.featureLocsX = {}
        self.featureSetGrouping = {}
        self.badPeaks = {}
        
        # High Pass Filter Parameters
        self.cutOffFreqLPF = 7
        self.samplingFreq = 500
        # Root Mean Squared (RMS) Parameters
        self.rmsWindow = 100; self.stepSize = 5;
        
        # Data Collection Parameters
        self.numPointsRMS = 10000
        self.peakDetectionBufferSize = 500
        self.rmsEdgeBuffer = 50      # Buffer in the Last Points of the RMS Data for Determining Peaks
        self.minGroupSep = 100       # Seperation that Defines a New Group
        self.lowPassBuffer = max(self.rmsWindow + self.stepSize, 1000)  # Must be > rmsWindow + stepSize; Current Experimental lowest: numTimePoints*0.4 (Changes with numTimePoints)
        # Keep Track of Gestures Recorded
        self.currentGroupNum = -1
        
        # Start with Fresh Inputs
        self.resetGlobalVariables()
        
        # Define Class for Plotting Peaks
        if plotStreamedData:
            self.initPlotPeaks()

    def resetGlobalVariables(self):
        # Data to Read in
        self.data = {'timePoints':[]}
        for channel in range(self.numChannels):
            self.data['Channel'+str(1+channel)] = []
        
        # Peak Finding and Feature Holders
        for channelNum in range(self.numChannels):
            # Hold Analysis Values
            self.RMSDataList[channelNum] = []
            self.featureLocsX[channelNum] = {1:[]}
            self.featureSetGrouping[channelNum] = {1:[]}
            self.badPeaks[channelNum] = []
        
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
            self.bioelectricPlotAxes[channelNum].set_xlabel("Bioelectric Data Points")
            self.bioelectricPlotAxes[channelNum].set_ylabel("Bioelectric Signal (Volts)")
            
        # Create the Peak Data Plot
        yLimitHighFiltered = 5;
        self.filteredBioelectricPlotAxes = [] 
        # Plot the Peak Data
        self.filteredBioelectricDataPlots = []
        self.movieGraphChannelTopPeaksList = []
        for channelNum in range(self.numChannels):
            # Create Plots
            self.filteredBioelectricPlotAxes.append(ax[channelNum, 1])
            
            # Plot RMS Peaks
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelNum].plot([], [], '-', c="tab:red", linewidth=1, alpha = 0.65)[0])
            # Plot Top Peaks
            self.movieGraphChannelTopPeaksList.append({})
            
            # Set Figure Limits
            self.filteredBioelectricPlotAxes[channelNum].set_ylim(yLimLow, yLimitHighFiltered)
            # Label Axis + Add Title
            self.filteredBioelectricPlotAxes[channelNum].set_title("Filtered Bioelectric Signal in Channel " + str(channelNum + 1))
            self.filteredBioelectricPlotAxes[channelNum].set_xlabel("Root Mean Squared Data Point")
            self.filteredBioelectricPlotAxes[channelNum].set_ylabel("Filtered Signal (Volts)")
            
            # Hold Analysis Values
            self.RMSDataList[channelNum] = []
            self.featureLocsX[channelNum] = {1:[]}
            self.featureSetGrouping[channelNum] = {1:[]}
            self.badPeaks[channelNum] = []
            
        # Tighten Figure White Space (Must be After wW Add Fig Info)
        self.fig.tight_layout(pad=2.0); plt.show()
        
    
    def analyzeData(self, dataFinger, plotStreamedData = False, myModel = None, Controller=None):     
        
        if plotStreamedData:
            # Get X Data: Shared Axis for All Channels
            self.timePoints = self.data['timePoints'][dataFinger:dataFinger + self.numTimePoints]
            
        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):
            
            # ------------------- Plot Biolectric Signal ---------------------#
            if plotStreamedData:
                # Get New Y Data
                newYData = self.data['Channel' + str(channelIndex+1)][dataFinger:dataFinger + self.numTimePoints]
                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.bioelectricDataPlots[channelIndex].set_data(self.timePoints, newYData)
                self.bioelectricPlotAxes[channelIndex].set_xlim(self.timePoints[0], self.timePoints[-1])
            # ----------------------------------------------------------------#
            
            # ---------------------- Low pass Filter -------------------------#    
            # Low Pass Filter to Remove Noise
            startLPFindex = max(dataFinger - self.lowPassBuffer, 0)
            yDataBuffer = self.data['Channel' + str(channelIndex+1)][startLPFindex:dataFinger + self.numTimePoints]
            filteredData = self.butter_lowpass_filter(yDataBuffer, self.cutOffFreqLPF, self.samplingFreq, order=5)[-self.numTimePoints:]
    
            # Plot Filtered Data
            if plotStreamedData:
                self.filteredBioelectricDataPlots[channelIndex].set_data(self.timePoints, filteredData)
                self.filteredBioelectricPlotAxes[channelIndex].set_xlim(self.timePoints[0], self.timePoints[-1])
            # ----------------------------------------------------------------#

            # ----------------------- Peak Detection  ------------------------#
            # Get Most Current RMS Data (Add Buffer in Case the peak is Cut Off)
            bufferRMSData = filteredData[-self.peakDetectionBufferSize:]
            bufferRMSDataX = self.timePoints[-self.peakDetectionBufferSize:]
            # Find Peaks from the New Data
            newTopPeaks, yBase = self.find_peaks(bufferRMSDataX, bufferRMSData, channelIndex, filteredData)
            # If No New Peaks, Then No New Features
            if newTopPeaks == {}:
                continue
            # Split the Peaks into the X,Y Points
            xPeakTop, yPeakTop = zip(*newTopPeaks.items())
            # ----------------------------------------------------------------#
            
            # ------------------------ Move Robot ----------------------------#
            # If New Peak Was Found with Enough Peak Seperation, Add Group 
            self.currentGroupNum = max(self.featureLocsX[channelIndex].keys(), default=0)
            currentHighestXPeak = max([max(self.featureLocsX[i][self.currentGroupNum], default=0) for i in range(self.numChannels)])
            if abs(self.timePoints[-1] - currentHighestXPeak) > self.minGroupSep and currentHighestXPeak != 0:
                self.createNewGroup(myModel, Controller)
            # ----------------------------------------------------------------#
            
            # --------------------- Feature Extraction  ----------------------#
            # Features Analysis to Group Peaks Together 
            batchXGroups, featureSetTemp = self.featureDefinition(filteredData, xPeakTop, yBase, self.currentGroupNum, myModel, Controller)
            # Update Overall Grouping Dictionary
            for groupNum in batchXGroups.keys():
                # Get New Peaks/Features to Add
                updateXGroups = batchXGroups[groupNum]
                updateFeatures = featureSetTemp[groupNum]
                # Add Them
                if groupNum in self.featureLocsX[channelIndex].keys():
                    self.featureLocsX[channelIndex][groupNum].extend(updateXGroups)
                    self.featureSetGrouping[channelIndex][groupNum].extend(updateFeatures)
                else:
                    self.featureLocsX[channelIndex][groupNum] = updateXGroups
                    self.featureSetGrouping[channelIndex][groupNum] = updateFeatures
            # ----------------------------------------------------------------#

            # ------------------------ Plot Peaks ----------------------------#
            if plotStreamedData:
                # Plot the Peaks; Colored by Grouping
                for groupNum in self.featureLocsX[channelIndex].keys():
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
                    if len(self.featureLocsX[channelIndex][groupNum]) != 0:
                        xPeakTop = self.featureLocsX[channelIndex][groupNum][0]
                        if self.timePoints[0] <= xPeakTop:
                            yPeakTop = filteredData[np.where(self.timePoints == xPeakTop)]
                            # Plot the Peaks in the Group
                            groupPeakPlot.set_data(xPeakTop, yPeakTop)
                        
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

    def butter_lowpass(self, cutoff = 7, fs = 330, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def highPassFilter(self, inputData):
        """
        data: Data to Filter
        f1: cutOffFreqPassThrough
        f3: cutOffFreqBand
        Rp: attDB (0.1)
        Rs: cutOffDB (30)
        samplingFreq: Frequecy You Take Data
        """
        Wp = 2*math.pi*self.f1/self.samplingFreq
        Ws = 2*math.pi*self.f3/self.samplingFreq
        [n, wn] = scipy.signal.cheb1ord(Wp/math.pi, Ws/math.pi, self.Rp, self.Rs)
        [bz1, az1] = scipy.signal.cheby1(n, self.Rp, Wp/math.pi, 'High')
        filteredData = lfilter(bz1, az1, inputData)
        return filteredData
    
    def RMSFilter(self, inputData, rmsWindow=250, stepSize=8):
        """
        The Function loops through the given Bioelectric Data, looking at batches of data
            of size rmsWindow at every interval seperated by stepSize.
        In Each Window, we take the magnitude of the data vector (sqrt[a^2+b^2]
            for [a,b] data point)
        A list of each root mean squared value is returned (in order)
        
        The Final List has a length of 1 + math.floor((len(inputData) - rmsWindow) / stepSize)
        --------------------------------------------------------------------------
        Input Variable Definitions:
            inputData: A List containing the  Bioelectric Data
            rmsWindow: The Amount of Data in the Groups we Analyze via RMS
            stepSize: The Distance Between Data Groups
        --------------------------------------------------------------------------
        """
        normalization = math.sqrt(rmsWindow)
        # Take Root Mean Squared of Batch Data (numBatch = rmsWindow)
        numSteps = 1 + math.floor((len(inputData) - rmsWindow) / stepSize)
        #print("RMS:", numSteps, len(inputData))
        RMSData = np.zeros(numSteps)
        for i in range(numSteps):
            # Get Data in the Window to take RMS
            inputWindow = inputData[i*stepSize:i*stepSize + rmsWindow]
            # Take RMS
            RMSData[i] = np.linalg.norm(inputWindow, ord=2)/normalization
        
        return RMSData

    def featureDefinition(self, RMSData, xTop, yBase, currentGroupNum, myModel = None, Controller = None):
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
                goodData = self.createNewGroup(myModel, Controller)
                if goodData:
                    peakGrouping[currentGroupNum] = [featureSet[i+1]]
                    xGrouping[currentGroupNum] = [xTop[i+1]]
     
        return xGrouping, peakGrouping


    def find_peaks(self, xData, yData, channel, RMSData):
        # Convert to Numpy (For Faster Data Processing)
        numpyDataX = np.array(xData)
        numpyDataY = np.array(yData)
        # Find Peak Indices
        peakInfo = scipy.signal.find_peaks(yData, prominence=.1, height=0.01, rel_height=0.5)
        indicesTop = peakInfo[0]
        # Get X,Y Peaks
        xTop = numpyDataX[indicesTop]
        yTop = numpyDataY[indicesTop]
        
        yBases = []
        for top in indicesTop:
            yBases.append(min(numpyDataY[top-50:top], default=[]))
        #print(peakInfo)
        
        # Find the New Peaks
        newTopPeaks = {}; yBase = []
        for i, xLoc in enumerate(xTop):
            lastBottomPeak = max(max(self.featureLocsX[channel].values(), default = [0]), default = 0)
            if xLoc > lastBottomPeak and xLoc + self.rmsEdgeBuffer > len(RMSData):
                # Record New Peaks and Add New Peaks to Ongoing list
                newTopPeaks[xLoc] = yTop[i]
                yBase.append(yBases[i])
        # Return New Peaks and Update Peak Dictionary
        return newTopPeaks, yBase
    
    
    def removePeakBackground(self, xTop, yTop, RMSData):
        newY = []
        for pointNum in range(len(xTop)):
            baselineIndex = self.findLeftMinimum(RMSData, xTop[pointNum])
            yPoint = yTop[pointNum] - RMSData[baselineIndex]
            newY.append(yPoint)
        return newY
    
    def createNewGroup(self, myModel, Controller, goodData = True):
        # Get FeatureSet Point for Group
        groupFeatures = []
        for channel in range(self.numChannels):
            # Get the Features for the Group and Take the First One
            channelFeature = self.featureSetGrouping[channel][self.currentGroupNum]
            groupFeatures.append((channelFeature or [0])[0])
        featureArray = np.array([groupFeatures])
        
        # If the Feature is Bad, Throw it Away
        if len(featureArray[featureArray > 0]) <= 1 and np.sum(featureArray) <= 0.1:
            if myModel:
                print("Only One Small Signal Found; Not Moving Robot")
            for channelNum in range(self.numChannels):
                self.badPeaks[channelNum].extend(self.featureLocsX[channelNum][self.currentGroupNum])
            self.currentGroupNum -= 1
            goodData = False
        # If it is Okay and We Have an ML Model, Predict the Movement
        elif myModel:
            self.predictMovement(myModel, featureArray, Controller)
            
        self.currentGroupNum += 1
        for channel in range(self.numChannels):
            self.featureLocsX[channel][self.currentGroupNum] = []
            self.featureSetGrouping[channel][self.currentGroupNum] = []
        
        return goodData
    
    def predictMovement(self, myModel, inputData, Controller = None):        
        # Predict Data
        predictedIndex = myModel.predictData(inputData)[0]
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



 
