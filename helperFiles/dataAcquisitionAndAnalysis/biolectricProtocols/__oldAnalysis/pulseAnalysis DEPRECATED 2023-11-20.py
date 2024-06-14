
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import sys
import math
import numpy as np
from scipy import stats
from bisect import bisect
# Peak Detection
import scipy
import scipy.signal
# Filter the Data
from scipy.signal import savgol_filter
# Baseline Subtraction
from BaselineRemoval import BaselineRemoval
# Matlab Plotting API
import matplotlib.pyplot as plt

# Import Files
import _globalProtocol

# Import Files
sys.path.append(os.path.dirname(__file__) + "/../_dataVisualization/")
import pulsePlottingProtocols

# -------------------------------------------------------------------------- #
# ----------------------------- Pulse Analysis ----------------------------- #

class pulseAnalysis(_globalProtocol.globalProtocol):
    
    def __init__(self, numPointsPerBatch = 3000, moveDataFinger = 10, channelIndices = (), plottingClass = None, readData = None):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            alreadyFilteredData: Do Not Reprocess Data That has Already been Processed; Just Extract Features
            plotSeperation: Display the Indeces Identified as Around Mid-Sysolic Along with the Data
            plotGaussFit: Display the Gaussian Decomposition of Each Pulse
        ----------------------------------------------------------------------
        """        
        # Program Flags
        self.plotGaussFit = plotGaussFit                # Plot the Guassian Decomposition
        self.plotSeperation = plotSeperation            # Plot the First Derivative and Labeled Systolic Peak Location (General)
        self.alreadyFilteredData = alreadyFilteredData  # If the Data is Already Filtered and Normalize, Do NOT Filter Again
        
        # Data Processing Parameters
        self.minGaussianWidth = 10E-5   # THe Minimum Gaussian Width During Guassian Decomposition
        self.minPeakIndSep = 10         # The Minimum Points Between the Dicrotic and Tail Peak
        
        self.resetGlobalVariables()
        
        # Initialize common model class
        super().__init__("pulse", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)

    def checkParams(self):
        pass
    
    def setSamplingFrequencyParams(self):
        pass
    
    def resetAnalysisVariables(self):
        # Feature Tracking Parameters
        self.timeOffset = 0             # Store the Time Offset Between Files
        self.numSecondsAverage = 60     # The Number of Swconds to Consider When Taking the Averaging Data
        self.incomingPulseTimes = []    # An Ongoing List Representing the Times of Each Pulse's Peak
        self.heartRateListAverage = []  # An Ongoing List Representing the Heart Rate
        # Feature Lists
        self.featureListExact = []      # List of Lists of Features; Each Index Represents a Pulse; Each Pulse's List Represents its Features
        self.featureListAverage = []    # List of Lists of Features Averaged in Time by self.numSecondsAverage; Each Index Represents a Pulse; Each Pulse's List Represents its Features

        # Peak Seperation Parameters
        self.peakStandard = 0;          # The Max First Deriviative of the Previous Pulse's Systolic Peak
        self.peakStandardInd = 0        # The Index of the Max Derivative in the Previous Pulse's Systolic Peak
        # Peak Filtering Parameters
        self.lowPassCutoff = 18         # Low Pass Filter Cutoff; Used on SEPERATED Pulses
        
        # Systolic and Diastolic References
        self.systolicPressure0 = None   # The Calibrated Systolic Pressure
        self.diastolicPressure0 = None  # The Calibrated Diastolic Pressure
        self.diastolicPressure = None   # The Current Diastolic Pressure
        self.systolicPressure = None   # The Current Systolic Pressure
        self.diastolicPressureList = []
        self.systolicPressureList = []
        # Systolic and Diastolic Calibration
        self.calibratedSystolicAmplitude = None    # The Average Amplitude for the Calibrated Systolic/Diastolic Pressure
        self.calibratedSystolicAmplitudeList = []  # A List of Systolic Amplitudes for the Calibration
        self.calibratedZero = None
        self.conversionSlope = None
        self.diastolicPressureInitialList = []
        
        # Save Each Filtered Pulse
        self.time = []
        self.signalData = []
        self.filteredData = []

    def setPressureCalibration(self, systolicPressure0, diastolicPressure0):
        self.systolicPressure0 = systolicPressure0    # The Calibrated Systolic Pressure
        self.diastolicPressure0 = diastolicPressure0  # The Calibrated Diastolic Pressure

    def seperatePulses(self, time, firstDer):
        self.peakStandardInd = 0
        # Take First Derivative of Smoothened Data
        systolicPeaks = [];
        for pointInd in range(len(firstDer)):
            # Calcuate the Derivative at pointInd
            firstDerVal = firstDer[pointInd]
            
            # If the Derivative Stands Out, Its the Systolic Peak
            if firstDerVal > self.peakStandard*0.5:
                
                # Use the First Few Peaks as a Standard
                if (self.timeOffset != 0 or 1.5 < time[pointInd]) and self.minPointsPerPulse < pointInd:
                    # If the Point is Sufficiently Far Away, its a New R-Peak
                    if self.peakStandardInd + self.minPointsPerPulse < pointInd:
                        systolicPeaks.append(pointInd)
                    # Else, Find the Max of the Peak
                    elif firstDer[systolicPeaks[-1]] < firstDer[pointInd]:
                        systolicPeaks[-1] = pointInd
                    # Else, Dont Update Pointer
                    else:
                        continue
                    self.peakStandardInd = pointInd
                    self.peakStandard = firstDerVal
                else:
                    self.peakStandard = max(self.peakStandard, firstDerVal)

        return systolicPeaks
    
    # ----------------------------------------------------------------------- #
    # ------------------------- Data Analysis Begins ------------------------ #
    
    def analyzeData(self, time, signalData, minBPM = 27, maxBPM = 480):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            time: xData-Axis Data for the Blood Pulse (Seconds)
            signalData:  yData-Axis Data for Blood Pulse (Capacitance)
            minBPM = Minimum Beats Per Minute Possible. 27 BPM is the lowest recorded; 30 is a good threshold
            maxBPM: Maximum Beats Per Minute Possible. 480 is the maximum recorded. 220 is a good threshold
        Use Case: Seperate the Pulses, Gaussian Decompositions, Feature Extraction
        ----------------------------------------------------------------------
        """       
        print("\nSeperating Pulse Data")
        # ------------------------- Set Up Analysis ------------------------- #
        # Calculate the Sampling Frequency, if None Present
        self.samplingFreq = len(signalData)/(time[-1]-time[0])
        print("\tSampling Frequency: " + str(self.samplingFreq))
  
        # Estimate that Defines the Number of Points in a Pulse
        self.minPointsPerPulse = math.floor(self.samplingFreq*60/maxBPM)
        self.maxPointsPerPulse = math.ceil(self.samplingFreq*60/minBPM)
        
        # Save the Data
        previousData = len(self.time)
        self.time.extend(time + self.timeOffset)
        self.signalData.extend(signalData)
        self.filteredData.extend([0]*len(time))
        # ------------------------------------------------------------------- #

        # ------------------------- Seperate Pulses ------------------------- #
        # Calculate Derivatives
        firstDer = savgol_filter(signalData, 9, 2, mode='nearest', delta=1/self.samplingFreq, deriv=1)
        # Take First Derivative of Smoothened Data
        systolicPeaks = self.seperatePulses(time, firstDer)
        # If no Systolic peaks found, it is likely there was a noise artifact with a high derivative
        while len(systolicPeaks) == 0:
            self.peakStandard = self.peakStandard/2;
            systolicPeaks = self.seperatePulses(time, firstDer)
        
        # If Questioning: Plot to See How the Pulses Seperated
        if self.plotSeperation:
            systolicPeaks = np.array(systolicPeaks); firstDer = np.array(firstDer)
            scaledData = signalData*max(np.abs(firstDer))/(max(signalData) - min(signalData))
            plt.figure()
            plt.plot(time, scaledData - np.mean(scaledData), label = "Centered + Scaled Signal Data", zorder = 3)
            plt.plot(time, firstDer, label = "First Derivative of Signal Data", zorder = 2)
            plt.plot(time[systolicPeaks], firstDer[systolicPeaks], 'o', label = "Mid-Pulse Rise Identification")
            plt.legend(loc=9, bbox_to_anchor=(1.35, 1));
            plt.hlines(0,time[0], time[-1])
            #plt.xlim(3,5)
            plt.show()
        # ------------------------------------------------------------------- #
                    
        # -------------------------- Pulse Analysis ------------------------- #
        print("\tAnalyzing Pulses")
        # Seperate Peaks Based on the Minimim Before the R-Peak Rise
        pulseStartInd = self.universalMethods.findNearbyMinimum(signalData, systolicPeaks[0], binarySearchWindow=-1, maxPointsSearch=self.maxPointsPerPulse)
        for pulseNum in range(1, len(systolicPeaks)):
            pulseEndInd = self.universalMethods.findNearbyMinimum(signalData, systolicPeaks[pulseNum], binarySearchWindow=-1, maxPointsSearch=self.maxPointsPerPulse)
            self.timePoint = time[pulseEndInd] + self.timeOffset
            
            # -------------------- Calculate Heart Rate --------------------- #
            # Save the Pulse's Time
            self.incomingPulseTimes.append(self.timePoint)
                        
            # Average Heart Rate in Time
            numPulsesAverage = len(self.incomingPulseTimes) - bisect(self.incomingPulseTimes, self.timePoint - self.numSecondsAverage)
            self.heartRateListAverage.append(numPulsesAverage*60/self.numSecondsAverage)
            # --------------------------------------------------------------- #
            
            # ---------------------- Cull Bad Pulses ------------------------ #
            # Check if the Pulse is Too Big: Likely Double Pulse
            if pulseEndInd - pulseStartInd > self.maxPointsPerPulse:
                print("Pulse Too Big; THIS SHOULDNT HAPPEN")
                pulseStartInd = pulseEndInd; continue
            # Check if the Pulse is Too Small; Likely Not an R-Peak
            elif pulseEndInd - pulseStartInd < self.minPointsPerPulse:
                print("Pulse Too Small; THIS SHOULDNT HAPPEN")
                pulseStartInd = pulseEndInd; continue
            # --------------------------------------------------------------- #
            
            # ----------------------- Filter the Pulse ---------------------- #
            # Extract Indivisual Pulse Data
            pulseData = signalData[pulseStartInd:pulseEndInd+1]
            # Filter the pulse, if not already filtered
            if not self.alreadyFilteredData:
                # Apply Low Pass Filter and then Smoothing Function
                pulseData = self.filteringMethods.bandPassFilter.butterFilter(pulseData, self.lowPassCutoff, self.samplingFreq, order = 3, filterType = 'low', fastFilt = True)
                pulseData = savgol_filter(pulseData, min(3, (len(pulseData)/8)), 2, mode='nearest')
            # --------------------------------------------------------------- #

            # ------------------ PreProcess the Pulse Data ------------------ #
            # Calculate the Pulse Derivatives
            pulseTime = time[pulseStartInd:pulseEndInd+1] - time[pulseStartInd]
            pulseVelocity = savgol_filter(pulseData, 3, 2, mode='interp', delta=1/self.samplingFreq, deriv=1)
            pulseAcceleration = savgol_filter(pulseData, 3, 2, mode='interp', delta=1/self.samplingFreq, deriv=2)
            thirdDeriv = savgol_filter(pulseAcceleration, 3, 1, mode='interp', delta=1/self.samplingFreq, deriv=1)

            # Normalize the Pulse's Baseline to Zero
            normalizedPulse = pulseData.copy()
            if not self.alreadyFilteredData:
                normalizedPulse = self.normalizePulseBaseline(normalizedPulse, polynomialDegree = 1)
            
            # Calculate Diastolic and Systolic Reference of the First Pulse (IF NO REFERENCE GIVEN)
            if not self.diastolicPressure0:
                diastolicPressure0 = pulseData[0]
                systolicPressure0 = self.universalMethods.findNearbyMaximum(signalData, systolicPeaks[pulseNum-1], binarySearchWindow=1, maxPointsSearch=self.maxPointsPerPulse)
                self.setPressureCalibration(systolicPressure0, diastolicPressure0)
            # --------------------------------------------------------------- #
            
            # -------------------- Extract Pulse Features ------------------- #
            if self.calibratedSystolicAmplitude != None:
                # Calculate the Diastolic Pressure
                self.diastolicPressure = self.calibratePressure(pulseData[0])
                self.systolicPressure = self.calibratePressure(max(pulseData))
                self.systolicPressureList.append(self.systolicPressure)
                self.diastolicPressureList.append(self.diastolicPressure)
                
                normalizedPulse = self.calibrateAmplitude(normalizedPulse)
                self.filteredData[previousData+pulseStartInd:previousData+pulseEndInd+1] = normalizedPulse
                # Label Systolic, Tidal Wave, Dicrotic, and Tail Wave Peaks Using Gaussian Decomposition   
                self.extractPulsePeaks(pulseTime, normalizedPulse, pulseVelocity, pulseAcceleration, thirdDeriv)
            else:
                self.diastolicPressureInitialList.append(pulseData[0])
                self.calibratedSystolicAmplitudeList.append(max(normalizedPulse) - normalizedPulse[0])
            # --------------------------------------------------------------- #
            
            # Reste for Next Pulse
            pulseStartInd = pulseEndInd
        # ------------------------------------------------------------------- #
        self.timeOffset += time[-1]
        
        if self.calibratedSystolicAmplitude == None:
            self.calibratedSystolicAmplitude = np.mean(self.calibratedSystolicAmplitudeList)
            self.conversionSlope = (self.systolicPressure0 - self.diastolicPressure0)/self.calibratedSystolicAmplitude
            self.calibratedZero = self.diastolicPressure0 - self.conversionSlope*np.mean(self.diastolicPressureInitialList)
        
        # plt.plot(self.heartRateListAverage, 'k-', linewidth=2)
        # plt.ylim(60, 100)
        # plt.show()
    
    def calibrateAmplitude(self, normalizedPulse):
        return normalizedPulse*self.conversionSlope
    
    def calibratePressure(self, capacitancePoint):
        return self.conversionSlope*capacitancePoint + self.calibratedZero
    
    def extractPulsePeaks(self, pulseTime, normalizedPulse, pulseVelocity, pulseAcceleration, thirdDeriv):
        
        # ----------------------- Detect Systolic Peak ---------------------- #        
        # Find Systolic Peak
        systolicPeakInd = self.universalMethods.findNearbyMaximum(normalizedPulse, 0, binarySearchWindow = 4, maxPointsSearch = len(pulseTime))
        # Find UpStroke Peaks
        systolicUpstrokeVelInd = self.universalMethods.findNearbyMaximum(pulseVelocity, 0, binarySearchWindow = 1, maxPointsSearch = systolicPeakInd)
        systolicUpstrokeAccelMaxInd = self.universalMethods.findNearbyMaximum(pulseAcceleration, systolicUpstrokeVelInd, binarySearchWindow = -1, maxPointsSearch = systolicPeakInd)
        systolicUpstrokeAccelMinInd = self.universalMethods.findNearbyMinimum(pulseAcceleration, systolicUpstrokeVelInd, binarySearchWindow = 1, maxPointsSearch = systolicPeakInd)
        # ------------------------------------------------------------------- #
                
        # ---------------------- Detect Tidal Wave Peak --------------------- #     
        bufferToTidal = self.universalMethods.findNearbyMaximum(thirdDeriv, systolicPeakInd+2, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        # Find Tidal Peak Boundaries
        tidalStartInd = bufferToTidal + np.where(np.diff(np.sign(thirdDeriv[bufferToTidal:])))[0][0]
        tidalEndInd_Estimate = self.universalMethods.findNearbyMaximum(pulseAcceleration, tidalStartInd, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        tidalEndInd_Estimate = self.universalMethods.findNearbyMinimum(pulseAcceleration, tidalEndInd_Estimate, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        tidalEndInd_Estimate = self.universalMethods.findNearbyMaximum(thirdDeriv, tidalEndInd_Estimate, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))        
        # Find Tidal Peak
        tidalVelocity_ZeroCrossings = tidalStartInd + np.where(np.diff(np.sign(pulseVelocity[tidalStartInd:tidalEndInd_Estimate])))[0]
        tidalAccel_ZeroCrossings = tidalStartInd + np.where(np.diff(np.sign(pulseAcceleration[tidalStartInd:tidalEndInd_Estimate])))[0]
        # Does the First Derivative Cross Zero -> Tidal Peak
        if len(tidalVelocity_ZeroCrossings) != 0:
            tidalPeakInd = tidalVelocity_ZeroCrossings[-1] + 1
        # Does the Second Derivative Cross Zero -> Closest First Derivative to Zero
        elif len(tidalAccel_ZeroCrossings) != 0:
            tidalPeakInd = tidalAccel_ZeroCrossings[0] + 1
        # Find Third Derivative Minimum -> Closest First Derivative to Zero
        else:
            tidalEndInd_Estimate = self.universalMethods.findNearbyMinimum(thirdDeriv, tidalEndInd_Estimate, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
            tidalEndInd_Estimate = self.universalMethods.findNearbyMaximum(thirdDeriv, tidalEndInd_Estimate, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
            dicroticNotchInd_Estimate = self.universalMethods.findNearbyMinimum(normalizedPulse, tidalEndInd_Estimate, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))   
            tidalEndInd_Estimate = self.universalMethods.findNearbyMinimum(thirdDeriv, dicroticNotchInd_Estimate, binarySearchWindow = -2, maxPointsSearch = int(len(pulseTime)/2))
            tidalEndInd_Estimate = self.universalMethods.findNearbyMaximum(thirdDeriv, tidalEndInd_Estimate, binarySearchWindow = -2, maxPointsSearch = int(len(pulseTime)/2))
            tidalPeakInd = self.universalMethods.findNearbyMinimum(thirdDeriv, tidalEndInd_Estimate, binarySearchWindow = -4, maxPointsSearch = int(len(pulseTime)/2))
        # Find Tidal Peak Ending
        tidalEndInd = self.universalMethods.findNearbyMinimum(thirdDeriv, tidalPeakInd, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        tidalEndInd = self.universalMethods.findNearbyMaximum(thirdDeriv, tidalEndInd, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        # ------------------------------------------------------------------- #
        
        # ----------------------  Detect Dicrotic Peak ---------------------- #
        dicroticNotchInd = self.universalMethods.findNearbyMinimum(normalizedPulse, tidalEndInd, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        dicroticPeakInd = self.universalMethods.findNearbyMaximum(normalizedPulse, dicroticNotchInd, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        
        # Other Extremas Nearby
        dicroticInflectionInd = self.universalMethods.findNearbyMaximum(pulseVelocity, dicroticNotchInd, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        dicroticFallVelMinInd = self.universalMethods.findNearbyMinimum(pulseVelocity, dicroticInflectionInd, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
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
            
            plt.title("Time: " + str(self.timePoint) + "; " + badReason)
            plt.show()

        # ------------------------- Cull Bad Pulses ------------------------- #
        # Check The Order of the Systolic Peaks
        if not systolicUpstrokeAccelMaxInd < systolicUpstrokeVelInd < systolicUpstrokeAccelMinInd < systolicPeakInd:
            print("\t\tBad Systolic Sequence. Time = ", self.timePoint)
            # plotIt("SYSTOLIC")
            return None
        # Check The Order of the Tidal Peaks
        elif not tidalPeakInd < tidalEndInd:
            print("\t\tBad Tidal Sequence. Time = ", self.timePoint); 
            # plotIt("TIDAL")
            return None
        # Check The Order of the Dicrotic Peaks
        elif not dicroticNotchInd < dicroticInflectionInd < dicroticPeakInd < dicroticFallVelMinInd:
            print("\t\tBad Dicrotic Sequence. Time = ", self.timePoint); 
            # plotIt("DICROTIC")
            return None
        # Check The Order of the Peaks
        elif not systolicPeakInd < tidalEndInd < dicroticNotchInd - 2:
            print("\t\tBad Peak Sequence. Time = ", self.timePoint); 
            # plotIt("GENERAL")
            return None
        elif pulseTime[dicroticNotchInd] - pulseTime[0] <= 0.25:
            print("\t\tToo Early Dicrotic. You Probably Missed the Tidal. Time = ", self.timePoint); 
            return None
        
        # Check If the Dicrotic Peak was Skipped
        if pulseTime[-1]*0.75 < pulseTime[dicroticPeakInd] - pulseTime[systolicUpstrokeAccelMaxInd]:
            print("\t\tDicrotic Peak Likely Skipped Over. Time = ", self.timePoint);
            return None
        # ------------------------------------------------------------------- #

        # ----------------------- Feature Extraction ------------------------ #
        allSystolicPeaks = [systolicUpstrokeAccelMaxInd, systolicUpstrokeVelInd, systolicUpstrokeAccelMinInd, systolicPeakInd]
        allTidalPeaks = [tidalPeakInd, tidalEndInd]
        allDicroticPeaks = [dicroticNotchInd, dicroticInflectionInd, dicroticPeakInd, dicroticFallVelMinInd]
        
        # Extract the Pulse Features
        self.extractFeatures(normalizedPulse, pulseTime, pulseVelocity, pulseAcceleration, allSystolicPeaks, allTidalPeaks, allDicroticPeaks)
        # ------------------------------------------------------------------- #
        
        # plotClass = pulsePlottingProtocols.pulsePlottingProtocols()
        # plotClass.plotPulseInfo(pulseTime, normalizedPulse/max(normalizedPulse), pulseVelocity/max(pulseVelocity), pulseAcceleration/max(pulseAcceleration), thirdDeriv/max(thirdDeriv), allSystolicPeaks, allTidalPeaks, allDicroticPeaks)
        # plotClass.plotPulseInfo_Amps(pulseTime, normalizedPulse/max(normalizedPulse), pulseVelocity/max(pulseVelocity), pulseAcceleration/max(pulseAcceleration), thirdDeriv/max(thirdDeriv), allSystolicPeaks, allTidalPeaks, allDicroticPeaks, tidalVelocity_ZeroCrossings, tidalAccel_ZeroCrossings)

        if self.plotGaussFit:
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
            plt.plot(pulseTime[tidalPeakInd], pulseAcceleration1[tidalPeakInd],  'ro')
            plt.plot(pulseTime[tidalEndInd], normalizedPulse1[tidalEndInd],  'ro')

            plt.plot(pulseTime[dicroticNotchInd], normalizedPulse1[dicroticNotchInd],  'bo')
            plt.plot(pulseTime[dicroticPeakInd], normalizedPulse1[dicroticPeakInd],  'bo')
            
            plt.plot(pulseTime[[dicroticInflectionInd, dicroticFallVelMinInd]], normalizedPulse1[[dicroticInflectionInd, dicroticFallVelMinInd]],  'bo')
            
            plt.axhline(y=0, color='k', linestyle='-', alpha = 0.5)
            
            plt.title("Time: " + str(self.timePoint))
            plt.show()


    def extractFeatures(self, normalizedPulse, pulseTime, pulseVelocity, pulseAcceleration, allSystolicPeaks, allTidalPeaks, allDicroticPeaks):
     
        # ------------------- Extract Data from Peak Inds ------------------- #        
        # Unpack All Peak Inds
        systolicUpstrokeAccelMaxInd, systolicUpstrokeVelInd, systolicUpstrokeAccelMinInd, systolicPeakInd = allSystolicPeaks
        tidalPeakInd, tidalEndInd = allTidalPeaks
        dicroticNotchInd, dicroticInflectionInd, dicroticPeakInd, dicroticFallVelMinInd = allDicroticPeaks
        
        # Find TimePoints of All Peaks
        systolicUpstrokeAccelMaxTime, systolicUpstrokeVelTime, systolicUpstrokeAccelMinTime, systolicPeakTime = pulseTime[allSystolicPeaks]
        tidalPeakTime, tidalEndTime = pulseTime[allTidalPeaks]
        dicroticNotchTime, maxVelDicroticRiseTime, dicroticPeakTime, minVelDicroticFallTime = pulseTime[allDicroticPeaks]
        # Find Amplitude of All Peaks
        systolicUpstrokeAccelMaxAmp, systolicUpstrokeVelAmp, systolicUpstrokeAccelMinAmp, pulsePressure = normalizedPulse[allSystolicPeaks]
        tidalPeakAmp, tidalEndAmp = normalizedPulse[allTidalPeaks]
        dicroticNotchAmp, dicroticRiseVelMaxAmp, dicroticPeakAmp, dicroticFallVelMinAmp = normalizedPulse[allDicroticPeaks]
        # Find Velocity of All Peaks
        systolicUpstrokeAccelMaxVel, systolicUpstrokeVelVel, systolicUpstrokeAccelMinVel, systolicPeakVel = pulseVelocity[allSystolicPeaks]
        tidalPeakVel, tidalEndVel = pulseVelocity[allTidalPeaks]
        dicroticNotchVel, dicroticRiseVelMaxVel, dicroticPeakVel, dicroticFallVelMinVel = pulseVelocity[allDicroticPeaks]
        # Find Acceleration of All Peaks
        systolicUpstrokeAccelMaxAccel, systolicUpstrokeVelAccel, systolicUpstrokeAccelMinAccel, systolicPeakAccel = pulseAcceleration[allSystolicPeaks]
        tidalPeakAccel, tidalEndAccel = pulseAcceleration[allTidalPeaks]
        dicroticNotchAccel, dicroticRiseVelMaxAccel, dicroticPeakAccel, dicroticFallVelMinAccel = pulseAcceleration[allDicroticPeaks]
        # ------------------------------------------------------------------- #
        
        # -------------------------- Time Features -------------------------- #        
        # Diastole and Systole Parameters
        pulseDuration = pulseTime[-1]
        systoleDuration = dicroticNotchTime
        diastoleDuration = pulseDuration - dicroticNotchTime
        leftVentricularPerformance = systoleDuration/diastoleDuration
        
        # Time from Systolic Peak
        maxDerivToSystolic = systolicPeakTime - systolicUpstrokeVelTime
        systolicToTidal = tidalPeakTime - systolicPeakTime
        systolicToDicroticNotch = dicroticNotchTime - systolicPeakTime
        # Time from Dicrotic Notch
        dicroticNotchToTidalDuration = tidalPeakTime - dicroticNotchTime
        dicroticNotchToDicrotic = dicroticPeakTime - dicroticNotchTime
        
        # General Times
        systolicRiseDuration = systolicUpstrokeAccelMinTime - systolicUpstrokeAccelMaxTime
        midToEndTidal = tidalEndTime - tidalPeakTime
        tidalToDicroticVelPeakInterval = maxVelDicroticRiseTime - tidalPeakTime
        # ------------------------------------------------------------------- #

        # --------------------- Under the Curve Features -------------------- #        
        # Calculate the Area Under the Curve
        pulseArea = scipy.integrate.simpson(normalizedPulse, pulseTime)
        pulseAreaSquared = scipy.integrate.simpson(normalizedPulse**2, pulseTime)
        leftVentricleLoad = scipy.integrate.simpson(normalizedPulse[0:dicroticNotchInd+1], pulseTime[0:dicroticNotchInd+1])
        diastolicArea = pulseArea - leftVentricleLoad
        
        # General Areas
        systolicUpSlopeArea = scipy.integrate.simpson(normalizedPulse[systolicUpstrokeAccelMaxInd:systolicUpstrokeAccelMinInd+1], pulseTime[systolicUpstrokeAccelMaxInd:systolicUpstrokeAccelMinInd+1])
        velToTidalArea = scipy.integrate.simpson(normalizedPulse[systolicUpstrokeVelInd:tidalPeakInd+1], pulseTime[systolicUpstrokeVelInd:tidalPeakInd+1])

        # Average of the Pulse
        pulseAverage = np.mean(normalizedPulse)
        # ------------------------------------------------------------------- #

        # -------------------------- Ratio Features ------------------------- #        
        # # Systole and Diastole Ratios
        systoleDiastoleAreaRatio = leftVentricleLoad/diastolicArea
        systolicDicroticNotchAmpRatio = dicroticNotchAmp/pulsePressure
        systolicDicroticNotchVelRatio = dicroticNotchVel/systolicPeakVel
        systolicDicroticNotchAccelRatio = dicroticNotchAccel/systolicPeakAccel
        
        # Other Diastole Ratios
        dicroticNotchTidalAmpRatio = tidalPeakAmp/dicroticNotchAmp
        dicroticNotchDicroticAmpRatio = dicroticPeakAmp/dicroticNotchAmp
        
        # Systolic Velocty Ratios
        systolicTidalVelRatio = tidalPeakVel/systolicPeakVel
        systolicDicroticVelRatio = dicroticPeakVel/systolicPeakVel
        # Diastole Velocity Ratios
        dicroticNotchTidalVelRatio = tidalPeakVel/dicroticNotchVel
        dicroticNotchDicroticVelRatio = dicroticPeakVel/dicroticNotchVel
        
        # Systolic Acceleration Ratios
        systolicTidalAccelRatio = tidalPeakAccel/systolicPeakAccel
        systolicDicroticAccelRatio = dicroticPeakAccel/systolicPeakAccel
        # Diastole Acceleration Ratios
        dicroticNotchTidalAccelRatio = tidalPeakAccel/dicroticNotchAccel
        dicroticNotchDicroticAccelRatio = dicroticPeakAccel/dicroticNotchAccel
        # ------------------------------------------------------------------- #

        # -------------------------- Slope Features --------=---------------- #        
        # Systolic Slopes
        systolicSlopeUp = pulseVelocity[systolicUpstrokeVelInd]

        # Tidal Slopes
        tidalSlope = np.polyfit(pulseTime[tidalPeakInd:tidalEndInd], normalizedPulse[tidalPeakInd:tidalEndInd], 1)[0]
        
        # Dicrotic Slopes
        dicroticSlopeUp = pulseVelocity[dicroticInflectionInd]
        
        # Tail Slopes
        endSlope = pulseVelocity[dicroticFallVelMinInd]
        # ------------------------------------------------------------------- #
        
        # ----------------------- Biological Features ----------------------- #        
        # Find the Diastolic and Systolic Pressure
        diastolicPressure = self.diastolicPressure
        systolicPressure = self.systolicPressure
        pressureRatio = systolicPressure/diastolicPressure

        momentumDensity = 2*pulseTime[-1]*pulseArea
        meanArterialBloodPressure = diastolicPressure + pulsePressure/3 # Dias + PP/3
        pseudoCardiacOutput = pulseArea/pulseTime[-1]
        pseudoSystemicVascularResistance = meanArterialBloodPressure/pulseTime[-1]
        pseudoStrokeVolume = pseudoCardiacOutput/pulseTime[-1]
                
        maxSystolicVelocity = max(pulseVelocity)
        valveCrossSectionalArea = pseudoCardiacOutput/maxSystolicVelocity
        velocityTimeIntegral = scipy.integrate.simpson(pulseVelocity, pulseTime)
        velocityTimeIntegralABS = scipy.integrate.simpson(abs(pulseVelocity), pulseTime)
        velocityTimeIntegral_ALT = pseudoStrokeVolume/valveCrossSectionalArea

        # Add Index Parameters: https://www.vitalscan.com/dtr_pwv_parameters.htm
        pAIx = tidalPeakAmp/pulsePressure  # Tidal Peak / Systolic Max Vel yData
        reflectionIndex = dicroticPeakAmp/pulsePressure  # Dicrotic Peak / Systolic Peak
        stiffensIndex = 1/(dicroticPeakTime - systolicPeakTime)  # 1/ Time from the Systolic to Dicrotic Peaks
        # ------------------------------------------------------------------- #
        
        # ------------------------ Organize Features ------------------------ #        
        pulseFeatures = [self.timePoint]
        # Saving Features from Section: Extract Data from Peak Inds
        pulseFeatures.extend([systolicUpstrokeAccelMaxTime, systolicUpstrokeVelTime, systolicUpstrokeAccelMinTime, systolicPeakTime])
        pulseFeatures.extend([tidalPeakTime, tidalEndTime])
        pulseFeatures.extend([maxVelDicroticRiseTime, dicroticPeakTime, minVelDicroticFallTime])
        pulseFeatures.extend([systolicUpstrokeAccelMaxAmp, systolicUpstrokeVelAmp, systolicUpstrokeAccelMinAmp, pulsePressure])
        pulseFeatures.extend([tidalPeakAmp, tidalEndAmp])
        pulseFeatures.extend([dicroticNotchAmp, dicroticRiseVelMaxAmp, dicroticPeakAmp, dicroticFallVelMinAmp])
        pulseFeatures.extend([systolicUpstrokeAccelMaxVel, systolicUpstrokeVelVel, systolicUpstrokeAccelMinVel, systolicPeakVel])
        pulseFeatures.extend([tidalPeakVel, tidalEndVel])
        pulseFeatures.extend([dicroticNotchVel, dicroticRiseVelMaxVel, dicroticPeakVel, dicroticFallVelMinVel])
        pulseFeatures.extend([systolicUpstrokeAccelMaxAccel, systolicUpstrokeVelAccel, systolicUpstrokeAccelMinAccel, systolicPeakAccel])
        pulseFeatures.extend([tidalPeakAccel, tidalEndAccel])
        pulseFeatures.extend([dicroticNotchAccel, dicroticRiseVelMaxAccel, dicroticPeakAccel])
        
        # Saving Features from Section: Time Features
        pulseFeatures.extend([pulseDuration, systoleDuration, diastoleDuration, leftVentricularPerformance])
        pulseFeatures.extend([maxDerivToSystolic, systolicToTidal, systolicToDicroticNotch, dicroticNotchToTidalDuration, dicroticNotchToDicrotic])
        pulseFeatures.extend([systolicRiseDuration, midToEndTidal, tidalToDicroticVelPeakInterval])
        
        # Saving Features from Section: Under the Curve Features
        pulseFeatures.extend([pulseArea, pulseAreaSquared, leftVentricleLoad, diastolicArea])
        pulseFeatures.extend([systolicUpSlopeArea, velToTidalArea, pulseAverage])
        
        # Saving Features from Section: Ratio Features
        pulseFeatures.extend([systoleDiastoleAreaRatio, systolicDicroticNotchAmpRatio, systolicDicroticNotchVelRatio, systolicDicroticNotchAccelRatio])
        pulseFeatures.extend([dicroticNotchTidalAmpRatio, dicroticNotchDicroticAmpRatio])
        pulseFeatures.extend([systolicTidalVelRatio, systolicDicroticVelRatio, dicroticNotchTidalVelRatio, dicroticNotchDicroticVelRatio])
        pulseFeatures.extend([systolicTidalAccelRatio, systolicDicroticAccelRatio, dicroticNotchTidalAccelRatio, dicroticNotchDicroticAccelRatio])
        
        # Saving Features from Section: Slope Features
        pulseFeatures.extend([systolicSlopeUp, tidalSlope, dicroticSlopeUp, endSlope])
        
        # Saving Features from Section: Biological Features
        pulseFeatures.extend([momentumDensity, pseudoCardiacOutput, pseudoStrokeVolume])
        pulseFeatures.extend([diastolicPressure, systolicPressure, pressureRatio, meanArterialBloodPressure, pseudoSystemicVascularResistance])
        pulseFeatures.extend([maxSystolicVelocity, valveCrossSectionalArea, velocityTimeIntegral, velocityTimeIntegralABS, velocityTimeIntegral_ALT])
        pulseFeatures.extend([pAIx, reflectionIndex, stiffensIndex])
        
        pulseFeatures.extend(pulseFeatures[1:])
        
        # Save the Pulse Features
        pulseFeatures = np.array(pulseFeatures)
        self.featureListExact.append(pulseFeatures)
        self.featureListAverage.append(stats.trim_mean(np.array(self.featureListExact)[:,1:][ np.array(self.featureListExact)[:,0] >= self.timePoint - self.numSecondsAverage ], 0.3))
    
    def normalizePulseBaseline(self, pulseData, polynomialDegree):
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
        # Perform Baseline Removal Twice to Ensure Baseline is Gone
        for _ in range(2):
            # Baseline Removal Procedure
            baseObj = BaselineRemoval(pulseData)  # Create Baseline Object
            pulseData = baseObj.ModPoly(polynomialDegree) # Perform Modified multi-polynomial Fit Removal
            
        # Return the Data With Removed Baseline
        return pulseData
    
    
    