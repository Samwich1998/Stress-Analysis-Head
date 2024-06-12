# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General Modules
import scipy
import numpy as np

# Import files.
from .humanMachineInterface import humanMachineInterface


# -------------------------------------------------------------------------- #
# ---------------------------- Global Function ----------------------------- #

class featureOrganization(humanMachineInterface):

    def __init__(self, modelClasses, actionControl, analysisProtocols, biomarkerOrder, biomarkerChannelIndices, featureAverageWindows):
        super().__init__(modelClasses, actionControl)
        # General parameters.
        self.biomarkerOrder = biomarkerOrder  # The biomarker order for feature analysis.
        self.biomarkerChannelIndices = biomarkerChannelIndices  # The channel index of each biomarker in biomarkerOrder.

        # Assert the integrity of feature organization.
        assert len(featureAverageWindows) == len(biomarkerOrder), \
            f"Found {featureAverageWindows} windows for {biomarkerOrder} biomarkers. These must to be the same length."

        self.analysisList_WithFeatures = []  # A list of all analyses that will have features, keeping the order they are streamed in.
        # Loop through each analysis requiring feature collection.
        for biomarkerInd in range(len(self.biomarkerOrder)):
            biomarkerType = self.biomarkerOrder[biomarkerInd]

            # Specify the parameters to collect features.
            analysisProtocols[biomarkerType].setFeatureCollectionParams(featureAverageWindows[biomarkerInd])
            self.analysisList_WithFeatures.append(analysisProtocols[biomarkerType])

            # Initialize mutable variables.
        self.resetFeatureInformation()

    def resetFeatureInformation(self):
        self.resetVariables_HMI()
        # Experimental information
        self.experimentTimes = []  # A list of lists of [start, stop] times of each experiment, where each element represents the times for one experiment. None means no time recorded.
        self.experimentNames = []  # A list of names for each experiment, where len(experimentNames) == len(experimentTimes).

        # Raw feature data structure
        self.rawFeatureHolder = [[] for _ in range(len(self.biomarkerOrder))]  # A list (in biomarkerOrder) of lists of raw features extraction at the current timepoint.
        self.rawFeatureTimesHolder = [[] for _ in range(len(self.biomarkerOrder))]  # A list (in biomarkerOrder) of lists of each features' times in biomarkerOrder.

        # Feature collection parameters
        self.alignmentPointers = [0 for _ in range(len(self.biomarkerOrder))]  # A list of pointers indicating the last seen aligned feature index for each analysis
        self.rawFeaturePointers = [0 for _ in range(len(self.biomarkerOrder))]  # A list of pointers indicating the last seen raw feature index for each analysis

    def unifyFeatureTimeWindows(self, featureTimeWindow):
        # Set the time window for EEG analysis.
        self.analysisProtocols['eeg'].featureTimeWindow = featureTimeWindow
        # Set the time window for EDA analysis.
        self.analysisProtocols['eda'].featureTimeWindow_Phasic = featureTimeWindow
        self.analysisProtocols['eda'].featureTimeWindow_Tonic = featureTimeWindow
        # Set the time window for temperature analysis.
        self.analysisProtocols['temp'].featureTimeWindow = featureTimeWindow

    # ---------------------------------------------------------------------- #
    # --------------------- Organize Incoming Features --------------------- #

    def organizeRawFeatures(self):
        # Loop through and compile each analysis' raw features
        for analysisInd in range(len(self.biomarkerOrder)):
            rawFeaturePointer = self.rawFeaturePointers[analysisInd]
            channelIndex = self.biomarkerChannelIndices[analysisInd]
            analysis = self.analysisList_WithFeatures[analysisInd]

            # Organize the raw features
            self.rawFeatureHolder[analysisInd].extend(analysis.rawFeatures[channelIndex][rawFeaturePointer:])
            self.rawFeatureTimesHolder[analysisInd].extend(analysis.featureTimes[channelIndex][rawFeaturePointer:])
            # Update the raw pointers.
            self.rawFeaturePointers[analysisInd] += len(analysis.rawFeatures[channelIndex][rawFeaturePointer:])

    def alignFeatures(self, lastTimePoint, secondsPerPoint, rawFeatureTimesHolder=None, compiledFeatureHolders=None):
        # Create a time interval to interpolate the features.
        startTime = self.alignedFeatureTimes[-1] + secondsPerPoint if len(self.alignedFeatureTimes) != 0 else 0
        newInterpolatedTimes = np.arange(startTime, lastTimePoint + secondsPerPoint, secondsPerPoint)
        # If the time interval is empty, then we need to read in more points.
        if len(newInterpolatedTimes) == 0:
            return None

        # Check which feature times to add
        for currentTime in newInterpolatedTimes:

            alignedFeatures = []
            # Loop through and compile each analysis' features
            for analysisInd in range(len(self.biomarkerOrder)):
                alignmentPointer = self.alignmentPointers[analysisInd]

                # Get the feature information
                if rawFeatureTimesHolder is None:
                    # Get the features from the streaming analsyis.
                    channelIndex = self.biomarkerChannelIndices[analysisInd]
                    alignedFeatureTimes = self.analysisList_WithFeatures[analysisInd].featureTimes[channelIndex]
                    compiledFeatures = self.analysisList_WithFeatures[analysisInd].compiledFeatures[channelIndex]
                else:
                    # Get the features from the information passed in.
                    alignedFeatureTimes = rawFeatureTimesHolder[analysisInd]
                    compiledFeatures = compiledFeatureHolders[analysisInd]

                # Check: is there a feature below the currentTime for inteprolation.
                if len(alignedFeatureTimes) == 0 or alignedFeatureTimes[alignmentPointer] > currentTime:
                    break
                # Find the time right before currentTime
                while alignedFeatureTimes[alignmentPointer] <= currentTime:
                    # Look at the next feature
                    alignmentPointer += 1
                    # If no time before AND after currentTime, you cant interpolate in between
                    if alignmentPointer == len(alignedFeatureTimes):
                        return None
                self.alignmentPointers[analysisInd] = alignmentPointer - 1

                # Specify the variables
                t1, t2 = alignedFeatureTimes[alignmentPointer - 1], alignedFeatureTimes[alignmentPointer]
                compiledFeatureLeft, compiledFeatureRight = compiledFeatures[alignmentPointer - 1], compiledFeatures[alignmentPointer]

                # Interpolate the feature
                alignedFeaturePoint = []
                for featureInd in range(len(compiledFeatureLeft)):
                    y1, y2 = compiledFeatureLeft[featureInd], compiledFeatureRight[featureInd]
                    # Interpolate the features
                    alignedFeaturePoint.append(y1 + ((y2 - y1) / (t2 - t1)) * (currentTime - t1))

                # Smooth data using convolution
                alignedFeaturePoint = scipy.signal.savgol_filter(alignedFeaturePoint, window_length=5, polyorder=3, mode='interp', deriv=0)

                # Compile the features from each analysis
                alignedFeatures.extend(alignedFeaturePoint)
            else:
                # If ALL the features are found for currentTime, add the point
                self.alignedFeatures.append(alignedFeatures)
                self.alignedFeatureTimes.append(currentTime)
                # Record the item being shown to the user
                self.alignedUserNames.append(self.userName)

                # itemName = "Baseline"
                # for experimentInd in range(len(self.experimentNames)):
                #     startTime, endTime = self.experimentTimes[experimentInd]

                #     if startTime <= currentTime:
                #         itemName = self.experimentNames[experimentInd]
                #         if itemName.isdigit(): itemName = "Music"
                #         elif itemName == "Recovery": itemName = "Baseline"
                #         elif "VR" in itemName.split(" - "): itemName = "VR"
                #     if endTime == None or currentTime <= endTime:
                #         break
                #     itemName = "Baseline"
                # self.alignedItemNames.append(itemName)

                # if len(self.experimentTimes) == 0 or self.experimentTimes[-1][1] != None:
                #     self.alignedItemNames.append("Baseline")
                # else:
                #     self.alignedItemNames.append(self.experimentNames[experimentInd])

    # ---------------------------------------------------------------------- #
    # ---------------------- Compile Incoming Features --------------------- #

    @staticmethod
    def getFinalFeatures(rawFeatureTimes, rawFeatures, timeInterval):
        startTime, endTime = timeInterval
        # Locate the experiment indices within the data
        endExperimentInd = np.searchsorted(rawFeatureTimes, endTime, side='left')
        startExperimentInd = np.searchsorted(rawFeatureTimes, startTime, side='left')
        # Take the inner portion of the interval
        featureIntervals = rawFeatures[startExperimentInd:endExperimentInd, :]
        # Ensure enough points in the interval.
        while len(featureIntervals) < 1:
            endExperimentInd = endExperimentInd + 1
            startExperimentInd = max(0, startExperimentInd - 1);
            featureIntervals = rawFeatures[startExperimentInd:endExperimentInd, :]

        # Average each feature across ALL biomarkers.
        finalFeatures = scipy.stats.trim_mean(featureIntervals, 0.2, axis=0)
        return finalFeatures

    @staticmethod
    def averageFeatures(newFeatureTimes, newRawFeatures, rawFeatureTimes, rawFeatures, compiledFeatures, averageWindow):
        # For each new raw feature
        for newFeatureInd in range(len(newFeatureTimes)):
            newRawFeature = newRawFeatures[newFeatureInd]
            newFeatureTime = newFeatureTimes[newFeatureInd]

            # Keep track of the new features
            rawFeatures.append(newRawFeature)
            rawFeatureTimes.append(newFeatureTime)

            # Track the running average of the features
            featureInterval = np.array(rawFeatures)[np.array(rawFeatureTimes) >= newFeatureTime - averageWindow]
            newCompiledFeature = scipy.stats.trim_mean(featureInterval, 0.2, axis=0)
            compiledFeatures.append(newCompiledFeature)

    @staticmethod
    def averageFeatures_DEPRECATED(rawFeatureTimes, rawFeatures, averageWindow):
        compiledFeatures = []
        # Average the Feature Together at Each Point
        for featureInd in range(len(rawFeatures)):
            # Get the interval of features to average
            featureMask = np.logical_and(
                rawFeatureTimes <= rawFeatureTimes[featureInd],
                rawFeatureTimes >= rawFeatureTimes[featureInd] - averageWindow
            )
            featureInterval = rawFeatures[featureMask]

            # Take the trimmed average
            compiledFeature = scipy.stats.trim_mean(featureInterval, 0.3)
            compiledFeatures.append(compiledFeature)

        return compiledFeatures
