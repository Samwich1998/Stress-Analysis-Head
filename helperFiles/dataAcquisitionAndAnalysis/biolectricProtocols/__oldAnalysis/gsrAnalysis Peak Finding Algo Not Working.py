

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import scipy
import numpy as np
# Matlab Plotting Modules
import matplotlib.pyplot as plt

# Import Files
import _filteringProtocols # Import files with filtering methods
import _universalProtocols # Import files with general analysis methods
        
# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------#

class gsrProtocol:
    
    def __init__(self, numPointsPerBatch = 3000, moveDataFinger = 10, numChannels = 2, plottingClass = None, readData = None):
        # Input Parameters
        self.numChannels = numChannels            # Number of Bioelectric Signals
        self.numPointsPerBatch = numPointsPerBatch        # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger      # The Amount of Data to Stream in Before Finding Peaks
        self.plotStreamedData = plottingClass != None  # Plot the Data
        self.plottingClass = plottingClass
        self.collectFeatures = False  # This flag will be changed by user if desired (NOT here).
        self.featureAverageWindow = None # The number of seconds before each feature to avaerage the features together.
        self.readData = readData
        
        # Feature collection parameters
        self.featureTimeWindowSCL = 60   
        self.featureTimeWindowSCR = 60             # The duration of time that each feature considers
        
        # Define general classes to process data.
        self.filteringMethods = _filteringProtocols.filteringMethods()
        self.universalMethods = _universalProtocols.universalMethods()
        # High Pass Filter Parameters
        self.dataPointBuffer = 5000        # A Prepended Buffer in the Filtered Data that Represents BAD Filtering; Units: Points
        self.cutOffFreq = [.01, 1]        # Optimal LPF Cutoff in Literatrue is 6-8 or 20 Hz (Max 35 or 50); I Found 25 Hz was the Best, but can go to 15 if noisy (small amplitude cutoff)
        
        self.increaseDerivThreshold = 0.004
        self.decreaseDerivThreshold = -0.00225
        
        # Prepare the Program to Begin Data Analysis
        self.checkParams()              # Check to See if the User's Input Parameters Make Sense
        self.resetGlobalVariables()     # Start with Fresh Inputs (Clear All Arrays/Values)
                
        # If Plotting, Define Class for Plotting Peaks
        if self.plotStreamedData and numChannels != 0:
            self.initPlotPeaks()
        
    def resetGlobalVariables(self):
        # Data to Read in
        self.data = [ [], [[] for channel in range(self.numChannels)] ]
        # Reset Feature Extraction
        self.rawFeatures = []           # Raw features extraction at the current timepoint.
        self.featureTimes = []          # The time of each feature.
        self.compiledFeatures = []      # FINAL compiled features at the current timepoint. Could be average of last x features.
        
        # General parameters
        self.samplingFreq = None        # The Average Number of Points Steamed Into the Arduino Per Second; Depends on the User's Hardware; If NONE Given, Algorithm will Calculate Based on Initial Data
        self.lastAnalyzedDataInd = 100  # The index of the last point analyzed  
        self.startFeatureTimePointer = 0
        self.startPeakTimePointer = 0
        self.startSubpeakTimePointer = 0
        
        self.peakRising = False  
        self.peakHeight_Threshold = 0.1
        
        # Close Any Opened Plots
        if self.plotStreamedData:
            plt.close()
            
    def checkParams(self):
        assert self.moveDataFinger < self.numPointsPerBatch, "You are Analyzing Too Much Data in a Batch. 'moveDataFinger' MUST be Less than 'numPointsPerBatch'"
        assert self.featureTimeWindowSCL < self.dataPointBuffer, "The buffer does not include enough points for the feature window"
        
    def setSamplingFrequency(self, startBPFindex):
        # Caluclate the Sampling Frequency
        self.samplingFreq = len(self.data[0][startBPFindex:])/(self.data[0][-1] - self.data[0][startBPFindex])
        print("\n\tSetting GSR Sampling Frequency to", self.samplingFreq)
        print("\tFor Your Reference, If Data Analysis is Longer Than", self.moveDataFinger/self.samplingFreq, ", Then You Will NOT be Analyzing in Real Time")
        
        # Set Blink Parameters
        self.minPoints_halfBaseline = max(1, int(self.samplingFreq*0.015))  # The Minimum Points in the Left/Right Baseline
        self.dataPointBuffer = int(self.samplingFreq*120)

    def initPlotPeaks(self): 
        # Establish pointers to the figure
        self.fig = self.plottingClass.fig
        axes = self.plottingClass.axes['gsr'][0]

        # Plot the Raw Data
        yLimLow = 1E-6; yLimHigh = 1E-5; 
        self.bioelectricDataPlots = []; self.bioelectricPlotAxes = []
        for channelIndex in range(self.numChannels):
            # Create Plots
            if self.numChannels == 1:
                self.bioelectricPlotAxes.append(axes[0])
            else:
                self.bioelectricPlotAxes.append(axes[channelIndex, 0])
            
            # Generate Plot
            self.bioelectricDataPlots.append(self.bioelectricPlotAxes[channelIndex].plot([], [], '-', c="purple", linewidth=1, alpha = 0.65)[0])
            
            # Set Figure Limits
            self.bioelectricPlotAxes[channelIndex].set_ylim(yLimLow, yLimHigh)
            # Label Axis + Add Title
            self.bioelectricPlotAxes[channelIndex].set_ylabel("GSR (Siemens)", fontsize=13, labelpad = 10)
            
        # Create the Data Plots
        self.filteredBioelectricDataPlots = []
        self.filteredBioelectricPlotAxes = [] 
        for channelIndex in range(self.numChannels):
            # Create Plot Axes
            if self.numChannels == 1:
                self.filteredBioelectricPlotAxes.append(axes[1])
            else:
                self.filteredBioelectricPlotAxes.append(axes[channelIndex, 1])
            
            # Plot Flitered Peaks
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="purple", linewidth=1, alpha = 0.65)[0])

            # Set Figure Limits
            self.filteredBioelectricPlotAxes[channelIndex].set_ylim(yLimLow, yLimHigh)
            
        # Tighten figure's white space (must be at the end)
        self.plottingClass.fig.tight_layout(pad=2.0);
        
    # ----------------------------------------------------------------------- #
    # ------------------------- Data Analysis Begins ------------------------ #
    
    def analyzeData(self, dataFinger, predictionModel = None, actionControl = None):
        
        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):
            
            # ---------------------- Filter the Data ----------------------- #    
            # Find the starting/ending points of the data to analyze
            startBPFindex = max(dataFinger - self.dataPointBuffer, 0)
            yDataBuffer = self.data[1][channelIndex][startBPFindex:dataFinger + self.numPointsPerBatch].copy()
            
            # Extract sampling frequency from the first batch of data
            if not self.samplingFreq:
                self.setSamplingFrequency(startBPFindex)

            # Filter the data: LPF and moving average (Savgol) filter
            filteredData = self.filteringMethods.bandPassFilter.butterFilter(yDataBuffer, self.cutOffFreq[1], self.samplingFreq, order = 1, filterType = 'low')
            filteredData = scipy.signal.savgol_filter(filteredData, max(3, int(self.samplingFreq*5)), 1, mode='nearest', deriv=0)
            
            # Format the data and timepoints
            timePoints = np.array(self.data[0])[-len(filteredData):]
            filteredData = np.array(filteredData)
            assert len(filteredData) == len(timePoints), "Incorrect sampling of batch"
            # --------------------------------------------------------------- #
            
            # ---------------------- Feature Extraction --------------------- #
            if self.collectFeatures:
                # Confirm assumptions made about GSR feature extraction
                assert dataFinger <= self.lastAnalyzedDataInd, str(dataFinger) + " " + str(self.lastAnalyzedDataInd) # We are NOT analyzing data in the buffer region. self.startFeatureTimePointer CAN be in the buffer region.
                
                extraFilteredData = scipy.signal.savgol_filter(filteredData, max(3, int(self.samplingFreq*5)), 1, mode='nearest', deriv=0)
                extremeFilteredNormalizedData = (extraFilteredData - np.mean(extraFilteredData))/np.std(extraFilteredData, ddof=1)
                # Calculate the derivatives
                firstDeriv_Extra = scipy.signal.savgol_filter(extraFilteredData, max(3, int(self.samplingFreq*60)), 1, mode='nearest', delta=1/self.samplingFreq, deriv=1)
                secondDeriv_Extra = scipy.signal.savgol_filter(firstDeriv_Extra, max(3, int(self.samplingFreq*60)), 1, mode='nearest', delta=1/self.samplingFreq, deriv=1)
                
                # Z-score normalization
                normalizedData = (filteredData - np.mean(extraFilteredData))/np.std(filteredData, ddof=1)
                # Calculate the derivatives
                firstDeriv = scipy.signal.savgol_filter(normalizedData, max(3, int(self.samplingFreq*60)), 1, mode='nearest', delta=1/self.samplingFreq, deriv=1)
                secondDeriv = scipy.signal.savgol_filter(firstDeriv, max(3, int(self.samplingFreq*60)), 1, mode='nearest', delta=1/self.samplingFreq, deriv=1)

                # plt.show()
                # plt.plot(timePoints, extremeFilteredNormalizedData/max(extremeFilteredNormalizedData), 'k', linewidth=2)
                # plt.plot(timePoints, firstDeriv_Extra/max(firstDeriv_Extra), 'b', linewidth=2)
                # plt.plot(timePoints, secondDeriv_Extra/max(secondDeriv_Extra), 'r', linewidth=2)
                # plt.show()
                
                # plt.plot(timePoints, normalizedData, 'k', linewidth=2)
                # plt.show()
                
                lastExtremaFound = 0  #  THIS SHOULD BE GLOBAL
                baselinePointers, peakMaxPointers = self.findAllPeaks(extremeFilteredNormalizedData, lastExtremaFound, binarySearchWindow = int(self.samplingFreq*5), maxPointsSearch = len(extremeFilteredNormalizedData))
                # print(timePoints[baselinePointers], timePoints[peakMaxPointers])
                
                if peakMaxPointers[0] < baselinePointers[0]:
                    lastbaselinePointer = self.findPrevBaselinePointer(extremeFilteredNormalizedData, peakMaxPointers[0], int(self.samplingFreq * -5))
                    baselinePointers.insert(0, lastbaselinePointer)
                                
                optimizedBaselinePointers = self.localOptimization(normalizedData, baselinePointers, 'min', int(self.samplingFreq * 1), int(self.samplingFreq * 30))
                optimizedpeakMaxPointers = self.localOptimization(normalizedData, peakMaxPointers, 'max', int(self.samplingFreq * 1), int(self.samplingFreq * 30))

                # if optimizedpeakMaxPointers[0] < optimizedBaselinePointers[0]:
                #     lastbaselinePointer = self.findPrevBaselinePointer(extremeFilteredNormalizedData, optimizedpeakMaxPointers[0], int(self.samplingFreq * -5))
                #     optimizedBaselinePointers.insert(0, lastbaselinePointer)
                                
                # plt.plot(timePoints, normalizedData, 'k', linewidth=2)
                # plt.plot(timePoints, extremeFilteredNormalizedData, 'tab:brown', linewidth=2)
                
                # plt.plot(timePoints[peakMaxPointers], normalizedData[peakMaxPointers], 'or')
                # plt.plot(timePoints[baselinePointers], normalizedData[baselinePointers], 'ob')
                
                # plt.plot(timePoints[optimizedpeakMaxPointers], normalizedData[optimizedpeakMaxPointers], 'om')
                # plt.plot(timePoints[optimizedBaselinePointers], normalizedData[optimizedBaselinePointers], 'og')
                
                # plt.plot(timePoints[peakMaxPointers], extremeFilteredNormalizedData[peakMaxPointers], 'or')
                # plt.plot(timePoints[baselinePointers], extremeFilteredNormalizedData[baselinePointers], 'ob')
                # plt.show()
                
                self.lastAnalyzedDataInd = len(self.data[0])
                return None
            
                ## LOCAL OPTIMAZATION FOR FINAL MIN/MAX
                # DISTINGUISH SUBPEAK AND OVERALL PEAK
                # EXTRACR FEATURES FROM SCRs

                
                # Compile the features together
                peakPresent = False
                subpeakPresent = False
                peakStage = 0 # 1 = increasing, 0 = peak, -1 = decreasing
                subpeakStage = 0
                
                peakPointer = 0
                
                self.peakAmp = 0
                while self.lastAnalyzedDataInd < len(self.timePoints):
                    featureTime = self.timePoints[self.lastAnalyzedDataInd]
                    self.findStartFeatureWindow(featureTime, peakPresent, subpeakPresent) # logic for startFeatureTimePointer and startPeakTimePointer
                    
                    intervalTimesSCL = timePoints[self.startFeatureTimePointer - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    intervalDataSCL = normalizedData[self.startFeatureTimePointer - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    
                    intervalTimesSCR = timePoints[self.startPeakTimePointer - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    intervalDataSCR = normalizedData[self.startPeakTimePointer  - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    
                    # Get the derivatives in the feature window
                    intervalFirstDerivSCL = firstDeriv[self.startFeatureTimePointer - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    intervalSecondDerivSCL = secondDeriv[self.startFeatureTimePointer - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    
                    intervalFirstDerivSCR = firstDeriv[self.startPeakTimePointer - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    intervalSecondDerivSCR = secondDeriv[self.startPeakTimePointer - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    
                    # Get the features in this window
                    gsrFeatures = self.extractFeaturesSCL(intervalTimesSCL, intervalDataSCL, intervalFirstDerivSCL, intervalSecondDerivSCL)
                    gsrFeatures.extend(self.extractPeakFeaturesSCR(intervalTimesSCR, intervalDataSCR, intervalFirstDerivSCR, intervalSecondDerivSCR))
                    
                    #intervalTimesSCR_sub = timePoints[self.startSubpeakTimePointer - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    #intervalDataSCR_sub = normalizedData[self.startSubpeakTimePointer  - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    
                    #intervalFirstDerivSCR_sub = firstDeriv[self.startSubpeakTimePointer - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    #intervalSecondDerivSCR_sub = secondDeriv[self.startSubpeakTimePointer - dataFinger:self.lastAnalyzedDataInd+1 - dataFinger]
                    

                    peakPointer, peakType = self.findRightMinMax(intervalDataSCL, peakPointer, binarySearchWindow = int(self.samplingFreq*5), maxPointsSearch = len(intervalDataSCL))
                    
                    plt.show()
                    print(abs(len(intervalDataSCL) - peakPointer))
                    print(int(self.samplingFreq*1))
                    print(peakPointer, peakType)
                    print("")
                    
                    plt.plot(intervalTimesSCL, intervalDataSCL, 'k', linewidth=2)
                    plt.plot(intervalTimesSCL[peakPointer], intervalDataSCL[peakPointer], 'or')
                    plt.show()
                    
                    
                    prevPeakPresent = peakPresent # only necessary for plotting
                    prevSubpeakPresent = subpeakPresent # only necessary for plotting
                    
                    peakPresent, peakStage, subpeakPresent = self.checkPeakPresentLogic(intervalFirstDerivSCL, peakPresent, peakStage, intervalDataSCR)
                    
                    self.plotPeaks(peakPresent, prevPeakPresent, subpeakPresent, prevSubpeakPresent, timePoints, normalizedData)
                    
                    # Keep track of the new features
                    self.readData.compileContinuousFeatures([featureTime], [gsrFeatures], self.rawFeatureTimes, self.rawFeatures, self.compiledFeatures, self.featureAverageWindow)
                
                    # Keep track of which data has been analyzed 
                    self.lastAnalyzedDataInd += int(self.samplingFreq*1)
            # -------------------------------------------------------------- #  
        
            # ------------------- Plot Biolectric Signals ------------------ #
            if self.plotStreamedData:
                # Get X Data: Shared Axis for All Channels
                timePoints = np.array(self.data[0][dataFinger:dataFinger + self.numPointsPerBatch])
    
                # Get New Y Data
                newYData = self.data[1][channelIndex][dataFinger:dataFinger + self.numPointsPerBatch]
                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.bioelectricDataPlots[channelIndex].set_data(timePoints, newYData)
                self.bioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1])
                            
                # Plot the Filtered + Digitized Data
                self.filteredBioelectricDataPlots[channelIndex].set_data(timePoints, filteredData[-len(timePoints):])
                self.filteredBioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1]) 
            # -------------------------------------------------------------- #   
    
    def plotPeaks(self, peakPresent, prevPeakPresent, subpeakPresent, prevSubpeakPresent, timePoints, normalizedData):
        if peakPresent and not prevPeakPresent:
            plt.plot(timePoints[self.startPeakTimePointer], normalizedData[self.startPeakTimePointer], 'ro')
            
        elif prevPeakPresent and not peakPresent:
            plt.plot(timePoints[self.startFeatureTimePointer], normalizedData[self.startFeatureTimePointer], 'mo')
        
        elif subpeakPresent and not prevSubpeakPresent:
            plt.plot(timePoints[self.startSubpeakTimePointer], normalizedData[self.startSubpeakTimePointer], 'bo')
            
        elif prevSubpeakPresent and not subpeakPresent:
            plt.plot(timePoints[self.startFeatureTimePointer], normalizedData[self.startFeatureTimePointer], 'co')
    
    def checkPeakPresentLogic(self, intervalFirstDerivSCL, peakPresent, peakStage, intervalDataSCR):
        avg_slope = scipy.stats.trim_mean(intervalFirstDerivSCL, 0.3)
        # print(peakPresent, avg_slope, self.peakAmp)
        if peakPresent:
            # print(self.startFeatureTimePointer, self.startPeakTimePointer, self.startSubpeakTimePointer)
            self.peakAmp = np.max([np.max(intervalDataSCR), self.peakAmp])
            if peakStage == 1:
                if avg_slope > self.increaseDerivThreshold * 0.5:
                    self.increaseDerivThreshold = np.max(intervalFirstDerivSCL)
                    return True, 1, True
                elif avg_slope < self.decreaseDerivThreshold:
                    return True, -1, True
                else:
                    return True, 0, True
                
            if peakStage == -1:
                if avg_slope > self.increaseDerivThreshold * 0.5:
                    if (self.peakAmp - intervalDataSCR[0]) * 0.5 <= (self.peakAmp - intervalDataSCR[-1]) and (self.startFeatureTimePointer - self.startPeakTimePointer > self.samplingFreq*30):
                        print("start: ", intervalDataSCR[0])
                        print("peak: ", self.peakAmp)
                        print("end: ", intervalDataSCR[-1])
                        self.increaseDerivThreshold = max(self.increaseDerivThreshold, np.max(intervalFirstDerivSCL))
                        return False, 0, False
                    elif self.startFeatureTimePointer - self.startSubpeakTimePointer > self.samplingFreq*30 and (self.peakAmp - intervalDataSCR[0])* 0.1 < np.max(intervalDataSCR[self.startSubpeakTimePointer - self.startPeakTimePointer:]) - intervalDataSCR[-1]:
                        print(np.max(intervalDataSCR[self.startSubpeakTimePointer - self.startPeakTimePointer:]) - intervalDataSCR[-1])
                        return True, -1, False
                    else:
                        return True, -1, True
                elif avg_slope < self.decreaseDerivThreshold:
                    return True, -1, True
                else:
                    if (self.peakAmp - intervalDataSCR[0]) * 0.5 <= (self.peakAmp - intervalDataSCR[-1]) and (self.startFeatureTimePointer - self.startPeakTimePointer > self.samplingFreq*30):
                        print("start: ", intervalDataSCR[0])
                        print("peak: ", self.peakAmp)
                        print("end: ", intervalDataSCR[-1])
                        return False, 0, False
                    elif self.startFeatureTimePointer - self.startSubpeakTimePointer > self.samplingFreq*30 and (self.peakAmp - intervalDataSCR[0])* 0.1 < np.max(intervalDataSCR[self.startSubpeakTimePointer - self.startPeakTimePointer:]) - intervalDataSCR[-1]:
                        self.increaseDerivThreshold = np.max(intervalFirstDerivSCL)
                        return True, -1, False
                    else:
                        return True, -1, True
                
            if peakStage == 0:
                if avg_slope > self.increaseDerivThreshold * 0.5:
                    self.increaseDerivThreshold = np.max(intervalFirstDerivSCL)
                    return True, 1, True
                elif avg_slope < self.decreaseDerivThreshold:
                    return True, -1, True
                else:
                    return True, 0, True
        else:
            # print(avg_slope, self.increaseDerivThreshold)
            if avg_slope > self.increaseDerivThreshold * 0.5:
                self.increaseDerivThreshold = np.max(intervalFirstDerivSCL)
                return True, 1, True
        
        return False, 0, False
    
    def findStartFeatureWindow(self, featureTime, peakPresent, subpeakPresent):
        while self.data[0][self.startFeatureTimePointer] < featureTime - self.featureTimeWindowSCL:
            self.startFeatureTimePointer += 1
            
        if not peakPresent:
            self.startPeakTimePointer = self.startFeatureTimePointer
        
        if not subpeakPresent:
            self.startSubpeakTimePointer = self.startFeatureTimePointer
    
    def isExtremaGood(self, data, oldExtremaPointer, newExtremaPointer):
        if abs(newExtremaPointer - oldExtremaPointer) < int(self.samplingFreq*5):
            print()
            return False
        elif abs(data[oldExtremaPointer] - data[newExtremaPointer]) < self.peakHeight_Threshold:
            return False
        
        return True
    
    def findAllPeaks(self, data, extremaPointer, binarySearchWindow, maxPointsSearch):
        baselinePointers = []; peakMaxPointers = []
        
        lastbaselinePointer = -1
        lastpeakMaxPointer = -1
        
        # Loop through all the points and label the top/bottom of the peaks.
        while binarySearchWindow*2 < abs(extremaPointer - len(data)):
            
            if self.peakRising == True:
                # Find the maximum to the right of the current extrema.
                nextExtremaPointer = self.universalMethods.findNearbyMaximum(data, extremaPointer, binarySearchWindow, maxPointsSearch)
                # If we truely found the peak maximum, record it; If not, ignore this extrema.
                if self.isExtremaGood(data, extremaPointer, nextExtremaPointer):
                    
                    ## TODO:CLEAN UP WHATEVER THIS IS
                    if lastpeakMaxPointer <= lastbaselinePointer:
                        peakMaxPointers.append(nextExtremaPointer) 
                        lastpeakMaxPointer = nextExtremaPointer
                        # if len(baselinePointers) == 0:
                        #     lastbaselinePointer = self.findPrevBaselinePointer(data, lastpeakMaxPointer, int(self.samplingFreq * -5))
                        #     baselinePointers.append(lastbaselinePointer)
                            # if lastpeakMaxPointer - lastbaselinePointer > self.samplingFreq * 15:
                            #     baselinePointers.append(lastbaselinePointer)
                            # else:
                            #     peakMaxPointers.pop()
                    # else:
                    #     peakMaxPointers[-1] = nextExtremaPointer
                        
                        
                    '''
                    if len(peakMaxPointers) == 0:
                        peakMaxPointers.append(nextExtremaPointer)  
                    elif len(baselinePointers) == 0:
                        peakMaxPointers[-1] = nextExtremaPointer  
                    elif peakMaxPointers[-1] < baselinePointers[-1]:
                        peakMaxPointers.append(nextExtremaPointer)   
                    else:
                        peakMaxPointers[-1] = nextExtremaPointer  
                    '''
                 
            else:
                nextExtremaPointer = self.universalMethods.findNearbyMinimum(data, extremaPointer, binarySearchWindow, maxPointsSearch)
                # If we truely found the peak minimum, record it; If not, ignore this extrema.
                if self.isExtremaGood(data, extremaPointer, nextExtremaPointer):
                    # If no maximum occured during the peak, update the last baseline.
                    
                    if lastbaselinePointer <= lastpeakMaxPointer:
                        baselinePointers.append(nextExtremaPointer) 
                        lastbaselinePointer = nextExtremaPointer
                    # else:
                    #     baselinePointers[-1] = nextExtremaPointer
                    
                    '''
                    if len(baselinePointers) == 0:
                        baselinePointers.append(nextExtremaPointer)  
                    elif len(peakMaxPointers) == 0:
                        baselinePointers[-1] = nextExtremaPointer  
                    elif baselinePointers[-1] < peakMaxPointers[-1]:
                        baselinePointers.append(nextExtremaPointer)   
                    else:
                        baselinePointers[-1] = nextExtremaPointer  
                    '''
                        
            # Reset the baselinePointers to find the minimum
            extremaPointer = nextExtremaPointer
            self.peakRising = not self.peakRising  
            
        # filteredBaselinePointers = []; filteredpeakMaxPointers = []
        
        # for index in range(len(peakMaxPointers)):
        #     if peakMaxPointers[index] - baselinePointers[index] > self.samplingFreq * 15:
        #         filteredpeakMaxPointers.append(peakMaxPointers[index])
        #         filteredBaselinePointers.append(baselinePointers[index])
            
        
        return baselinePointers, peakMaxPointers
        # return filteredBaselinePointers, filteredpeakMaxPointers
    
    def findRightMinMax(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 10000):
        rightMinumum = self.universalMethods.findNearbyMinimum(data, xPointer, binarySearchWindow, maxPointsSearch)
        rightMaximum = self.universalMethods.findNearbyMaximum(data, xPointer, binarySearchWindow, maxPointsSearch)
        
        # We are at a 
        # if abs(rightMinumum - rightMaximum) < self.samplingFreq*1:
        #     import sys
        #     sys.exit("findRightMinMax is not differentiating. rightMinumum: " + str(rightMinumum) + "; rightMaximum: " + str(rightMaximum))
            
        if rightMinumum < rightMaximum: 
            return rightMaximum, 1
        else:
            return rightMinumum, -1
    
    def localOptimization(self, data, points, localType, binarySearchWindow, maxPointsSearch):
        optimizedPoints = []
        if localType == 'min':
            for point in points:
                left = self.universalMethods.findNearbyMinimum(data, point, binarySearchWindow * -1, maxPointsSearch)
                right = self.universalMethods.findNearbyMinimum(data, point, binarySearchWindow, maxPointsSearch)
                if data[left] < data[right]:
                    optimizedPoints.append(left)
                else:
                    optimizedPoints.append(right)
        elif localType == 'max':
            for point in points:
                left = self.universalMethods.findNearbyMaximum(data, point, binarySearchWindow * -1, maxPointsSearch)
                right = self.universalMethods.findNearbyMaximum(data, point, binarySearchWindow, maxPointsSearch)
                if data[left] > data[right]:
                    optimizedPoints.append(left)
                else:
                    optimizedPoints.append(right)
        else:
            return points
        
        return optimizedPoints
    
    def findPrevBaselinePointer(self, data, xPointer, binarySearchWindow):
        currBaseLinePointer = xPointer
        newBaseLinePointer = self.universalMethods.findNearbyMinimum(data, currBaseLinePointer, binarySearchWindow, maxPointsSearch = int(self.samplingFreq * 30))
        print(binarySearchWindow)
        print(currBaseLinePointer, newBaseLinePointer)
        print(data[currBaseLinePointer] - data[newBaseLinePointer])
        
        while newBaseLinePointer < currBaseLinePointer and data[currBaseLinePointer] - data[newBaseLinePointer] > self.peakHeight_Threshold and newBaseLinePointer > 0:
            currBaseLinePointer = newBaseLinePointer
            newBaseLinePointer = self.universalMethods.findNearbyMinimum(data, currBaseLinePointer, binarySearchWindow, maxPointsSearch = int(self.samplingFreq * 30))   
            print('walking')
            print(data[currBaseLinePointer] - data[newBaseLinePointer])
        return currBaseLinePointer
        
    
                    
    def extractFeaturesSCL(self, xData, yData, firstDeriv, secondDeriv):
        
        # ------------------------------------------------------------------ #  
        # ----------------------- Features from Data ----------------------- #
        
        # General Shape Parameters
        meanSignal = np.mean(yData)
        signalEntropy = scipy.stats.entropy(abs(yData + 10E-50))
        standardDeviation = np.std(yData, ddof = 1)
        signalSkew = scipy.stats.skew(yData, bias=False)
        signalKurtosis = scipy.stats.kurtosis(yData, fisher=True, bias = False)
        normalizedArea = scipy.integrate.simpson(yData, xData)/len(xData)
        
        # Other pamaeters
        signalChange = yData[-1] - yData[0]
        averageNoise = np.mean(abs(np.diff(yData)))
        averageSquaredNoise = np.mean(np.diff(yData)**2)/len(xData)
        signalPower = scipy.integrate.simpson(yData**2, xData)/len(xData)
        
        centralMoment = scipy.stats.moment(yData, moment=1)
        arcLength = np.sqrt(1 + np.diff(yData))
        rootMeaSquared = np.sqrt(signalPower)
        
        # areaPerimeter =  
        
        # ------------------------------------------------------------------ #  
        # -------------------- Features from Derivatives ------------------- #
        
        # First derivative features
        firstDerivMean = np.mean(firstDeriv)
        firstDerivSTD = np.std(firstDeriv, ddof = 1)
        firstDerivPower = scipy.integrate.simpson(firstDeriv**2, xData)/len(xData)
        
        # Second derivative features
        secondDerivMean = np.mean(secondDeriv)
        secondDerivSTD = np.std(secondDeriv, ddof = 1)
        secondDerivPower = scipy.integrate.simpson(secondDeriv**2, xData)/len(xData)
        
        # ------------------------------------------------------------------ #  
        # ----------------- Features from Normalized Data ------------------ #
        baselineDataX = xData - xData[0]
        baselineDataY = yData - meanSignal
                
        signalSlope, slopeIntercept = np.polyfit(baselineDataX, baselineDataY, 1)     
        
        # fequencyProfile = scipy.fft.fft(yData)
        
        # 0.1 to 0.2 (F1SC), 0.2 to 0.3 (F2SC) and 0.3 to 0.4 (F3SC)

        
        # ------------------------------------------------------------------ #  
        # ----------------------- Organize Features ------------------------ #
        
        gsrFeatures = []
        # Add peak shape parameters
        gsrFeatures.extend([meanSignal, signalEntropy, standardDeviation, signalSkew, signalKurtosis, normalizedArea])
        gsrFeatures.extend([signalChange, averageNoise, averageSquaredNoise, signalPower])
        
        # # Add derivative features
        gsrFeatures.extend([firstDerivMean, firstDerivSTD, firstDerivPower])
        gsrFeatures.extend([secondDerivMean, secondDerivSTD, secondDerivPower])
        
        # Add normalized features
        gsrFeatures.extend([signalSlope, slopeIntercept])

        return gsrFeatures
            
    def extractPeakFeaturesSCR(self, xData, yData, firstDeriv, secondDeriv):
        # ------------------------------------------------------------------ #  
        # ----------------------- Features from Data ----------------------- #
        
        # General Shape Parameters
        valueAboveMean = np.mean(yData)
        
        signalEntropy = scipy.stats.entropy(abs(yData + 10E-50))
        standardDeviation = np.std(yData, ddof = 1)
        signalSkew = scipy.stats.skew(yData, bias=False)
        signalKurtosis = scipy.stats.kurtosis(yData, fisher=True, bias = False)
        normalizedArea = scipy.integrate.simpson(yData, xData)/len(xData)
        
        # Other pamaeters
        signalChange = yData[-1] - yData[0]
        averageNoise = np.mean(abs(np.diff(yData)))
        averageSquaredNoise = np.mean(np.diff(yData)**2)/len(xData)
        signalPower = scipy.integrate.simpson(yData**2, xData)/len(xData)
        
        centralMoment = scipy.stats.moment(yData, moment=1)
        arcLength = np.sqrt(1 + np.diff(yData))
        rootMeaSquared = np.sqrt(signalPower)
        
        test_feature = np.max(yData) - np.min(yData)
        
        # areaPerimeter =  
        
        # ------------------------------------------------------------------ #  
        # -------------------- Features from Derivatives ------------------- #
        
        # First derivative features
        #firstDerivMean = np.mean(firstDeriv)
        # firstDerivSTD = np.std(firstDeriv, ddof = 1)
        # firstDerivPower = scipy.integrate.simpson(firstDeriv**2, xData)/len(xData)
        
        # # Second derivative features
        # secondDerivMean = np.mean(secondDeriv)
        # secondDerivSTD = np.std(secondDeriv, ddof = 1)
        # secondDerivPower = scipy.integrate.simpson(secondDeriv**2, xData)/len(xData)
        
        # ------------------------------------------------------------------ #  
        # ----------------- Features from Normalized Data ------------------ #
        baselineDataX = xData - xData[0]
        baselineDataY = yData - yData[0]
                
        signalSlope, slopeIntercept = np.polyfit(baselineDataX, baselineDataY, 1)     
        
        # fequencyProfile = scipy.fft.fft(yData)
        
        # 0.1 to 0.2 (F1SC), 0.2 to 0.3 (F2SC) and 0.3 to 0.4 (F3SC)

        
        # ------------------------------------------------------------------ #  
        # ----------------------- Organize Features ------------------------ #
        
        gsrFeatures = []
        gsrFeatures.extend([test_feature])
        # # Add peak shape parameters
        # gsrFeatures.extend([meanSignal, signalEntropy, standardDeviation, signalSkew, signalKurtosis, normalizedArea])
        # gsrFeatures.extend([signalChange, averageNoise, averageSquaredNoise, signalPower])
        
        # # # Add derivative features
        # gsrFeatures.extend([firstDerivMean, firstDerivSTD, firstDerivPower])
        # gsrFeatures.extend([secondDerivMean, secondDerivSTD, secondDerivPower])
        
        # # Add normalized features
        # gsrFeatures.extend([signalSlope, slopeIntercept])

        return gsrFeatures




    
    
    
    
    