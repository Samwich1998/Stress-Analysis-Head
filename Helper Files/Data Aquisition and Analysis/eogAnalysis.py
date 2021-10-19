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
# Calibration Fitting
from scipy.optimize import curve_fit
# Plotting
import matplotlib
import matplotlib.pyplot as plt


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
        self.calibrationAngles = [[-30, 30] for _ in range(self.numChannels)]
        self.calibrationVoltages = [[] for _ in range(self.numChannels)]
        
        # High Pass Filter Parameters
        self.samplingFreq = samplingFreq          # Depends on the User's Hardware
        self.cutOffFreq = 8                       # Optimal LPF 6-8 Hz (Max 35 or 50); literature Claimed 7 Hz is Best
        
        # Data Collection Parameters
        self.voltagePositionBuffer = 50   # Buffer to Find the Average Voltage
        self.minVoltageMovement = 0.05    # Min Voltage Change Threshold to Move the Gaze
        self.bandPassBuffer = 1000        # Buffer in the Filtered Data that Represented BAD Filtering
        
        # Pointers for Calibration
        self.calibrateChannelNum = 0
        self.channelCalibrationPointer = 0
        # Calibration Function for Eye Angle
        self.predictEyeAngle = [lambda x: (x-2.5)*30]*self.numChannels
                
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
        self.currentEyeVoltages = [2.5 for _ in range(self.numChannels)]
        
        # Close Any Opened Plots
        if self.plotStreamedData:
            plt.close()        


    def initPlotPeaks(self): 
        
        # use ggplot style for more sophisticated visuals
        plt.style.use('seaborn-poster')
        #matplotlib.use( 'Qt5Agg' )

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
        
    
    def analyzeData(self, dataFinger, plotStreamedData = False, calibrateModel = False, actionControl = None):     
        
        eyeAngles = []
        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):
            
            # ---------------------- Band Pass Filter ----------------------- #    
            # Band Pass Filter to Remove Noise
            startBPFindex = max(dataFinger - self.bandPassBuffer, 0)
            yDataBuffer = self.data['Channel' + str(channelIndex+1)][startBPFindex:dataFinger + self.numTimePoints].copy()
            filteredData = self.butterFilter(yDataBuffer, self.cutOffFreq, self.samplingFreq, order = 3, filterType = 'low')[-self.numTimePoints:]
            # --------------------------------------------------------------- #
            
            # --------------------- Predict Eye Movement  ------------------- #
            # Get the Current Voltage (Take Average)
            currentEyeVoltage = self.findTraileringAverage(filteredData[-self.voltagePositionBuffer:], deviationThreshold = self.minVoltageMovement)
            # Compare Voltage Difference to Remove Small Shakes
            if abs(currentEyeVoltage - self.currentEyeVoltages[channelIndex]) > self.minVoltageMovement:
                self.currentEyeVoltages[channelIndex] = currentEyeVoltage        
            # Predict the Eye's Degree
            if self.predictEyeAngle[channelIndex]:
                eyeAngle = self.predictEyeAngle[channelIndex](self.currentEyeVoltages[channelIndex])
                eyeAngles.append(eyeAngle)
            # --------------------------------------------------------------- #
            
            # --------------- Calibrate Angle Prediction Model -------------- #
            if calibrateModel:
                if self.calibrateChannelNum == channelIndex:
                    self.calibrationVoltages[self.calibrateChannelNum].append(currentEyeVoltage)
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
                self.trailingAverageData[channelIndex].extend([self.currentEyeVoltages[channelIndex]]*self.moveDataFinger)
                self.trailingAverageData[channelIndex] = self.trailingAverageData[channelIndex][self.moveDataFinger:]
                # Plot the Filtered + Digitized Data
                self.filteredBioelectricDataPlots[channelIndex].set_data(self.timePoints, filteredData)
                self.trailingAveragePlots[channelIndex].set_data(self.timePoints, self.trailingAverageData[channelIndex])
                self.filteredBioelectricPlotAxes[channelIndex].set_xlim(self.timePoints[0], self.timePoints[-1]) 
                # Plot the Eye's Angle if Electrodes are Calibrated
                if self.predictEyeAngle[channelIndex]:
                    self.filteredBioelectricPlotAxes[channelIndex].legend(["Eye's Angle: " + "%.3g"%eyeAngle], loc="upper left")
            # --------------------------------------------------------------- #
            
            
        # -------------------- Update Virtual Reality  ---------------------- #
        if actionControl and self.predictEyeAngle[channelIndex] and not calibrateModel:
            actionControl.setGaze(eyeAngles)
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
        self.currentEyeVoltages[channelIndex] = 2.5
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


    def findTraileringAverage(self, recentData, deviationThreshold = 0.08):
        # Base Case in No Points Came in
        if len(recentData) == 0:
            return 2.5
        
        # Keep Track of the trailingAverage
        trailingAverage = recentData[-1]
        for dataPointInd in range(2, len(recentData)-1, -1):
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


 
