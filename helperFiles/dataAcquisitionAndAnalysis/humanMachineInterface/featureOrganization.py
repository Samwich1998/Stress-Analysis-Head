from scipy.interpolate import Akima1DInterpolator
import collections
import numpy as np
import scipy

# Import files.
from .humanMachineInterface import humanMachineInterface

# Parameters for the streamingProtocolHelpers class:
#     Biomarker information:
#         biomarkerFeatureOrder: A list of biomarkers that require features in streamingOrder; Dim: numFeatureSignals
#         featureAnalysisOrder: A list of unique biomarkers that require features in streamingOrder; Dim: numUniqueFeatureSignals
#
#     Analysis information:
#         featureAnalysisList: A list of unique analysis protocols in featureAnalysisOrder; Dim: numUniqueFeatureSignals
#         analysisList: A list of unique analysis protocols; Dim: numUniqueSignals
#
#     Feature information:
#         rawFeatureTimesHolder: A list (in biomarkerFeatureOrder) of lists of raw feature's times; Dim: numFeatureSignals, numPoints
#         rawFeatureHolder: A list (in biomarkerFeatureOrder) of lists of raw features; Dim: numFeatureSignals, numPoints, numBiomarkerFeatures


class featureOrganization(humanMachineInterface):

    def __init__(self, modelClasses, actionControl, analysisProtocols, extractFeaturesFrom, featureAverageWindows):
        super().__init__(modelClasses, actionControl, extractFeaturesFrom)
        # General parameters.
        self.featureAnalysisOrder = list(collections.OrderedDict.fromkeys(self.biomarkerFeatureOrder))  # The set of unique feature biomarkers, maintaining the order they will be analyzed. Ex: ['eog', 'eeg', 'eda']
        self.newSamplingFreq = self.compileModelInfo.newSamplingFreq  # The new sampling frequency for the data for the machine learning model.
        self.minNumExperimentalPoints = 10  # The minimum number of points required for an experimental interval.
        self.biomarkerFeatureInds = []  # The indices of the biomarker features in the featureNames; Dim: numBiomarkers, numBiomarkerFeatures
        self.featureAnalysisList = []  # A list of unique analyses that will have features, keeping the order they are streamed in.
        self.interpBufferPoints = 8  # The buffer for the edge of the feature interpolation.
        self.trimMeanCut = 0.25  # The proportion to cut for the trimmed mean.

        # Assert the integrity of feature organization.
        assert len(featureAverageWindows) == len(self.biomarkerFeatureOrder), f"Found {featureAverageWindows} windows for {self.biomarkerFeatureOrder} biomarkers. These must to be the same length."

        # Loop through each analysis requiring collecting features.
        for biomarkerInd in range(len(self.featureAnalysisOrder)):
            biomarkerType = self.featureAnalysisOrder[biomarkerInd]

            # Get the channel indices for the biomarker.
            featureChannelIndices = np.where(self.biomarkerFeatureOrder == biomarkerType)[0]

            # Specify the parameters to collect features.
            analysisProtocols[biomarkerType].setFeatureCollectionParams(featureAverageWindows[biomarkerInd], featureChannelIndices)
            self.featureAnalysisList.append(analysisProtocols[biomarkerType])

        startBiomarkerInd = 0; endBiomarkerInd = 0
        # Loop through each biomarker's list of features names.
        for biomarkerInd in range(len(self.biomarkerFeatureNames)):
            numBiomarkerFeatures = len(self.biomarkerFeatureNames[biomarkerInd])
            endBiomarkerInd = startBiomarkerInd + numBiomarkerFeatures

            # Store the feature indices for this biomarker.
            self.biomarkerFeatureInds.append(np.arange(startBiomarkerInd, endBiomarkerInd, 1))
            startBiomarkerInd = endBiomarkerInd
        assert endBiomarkerInd == len(self.featureNames), f"Found {endBiomarkerInd} biomarker features and {len(self.featureNames)} features. These must be the same length."

        # Holder parameters.
        self.rawFeatureTimesHolder = None  # A list (in biomarkerFeatureOrder) of lists of raw feature's times; Dim: numFeatureSignals, numPoints
        self.alignmentDataPointers = None  # A list of pointers indicating the last seen aligned feature index for each analysis
        self.rawFeaturePointers = None  # A list of pointers indicating the last seen raw feature index for each analysis and each channel.
        self.rawFeatureHolder = None  # A list (in biomarkerFeatureOrder) of lists of raw features; Dim: numFeatureSignals, numPoints, numBiomarkerFeatures

        # Initialize mutable variables.
        self.resetFeatureInformation()

    def resetFeatureInformation(self):
        self.resetVariables_HMI()
        # Reset the analysis information
        for featureAnalysis in self.featureAnalysisList:
            featureAnalysis.resetAnalysisVariables()

        # Raw feature data structure
        self.rawFeatureTimesHolder = [[] for _ in range(len(self.biomarkerFeatureOrder))]  # A list (in biomarkerFeatureOrder) of lists of raw feature's times; Dim: numFeatureSignals, numPoints
        self.rawFeatureHolder = [[] for _ in range(len(self.biomarkerFeatureOrder))]  # A list (in biomarkerFeatureOrder) of lists of raw features; Dim: numFeatureSignals, numPoints, numBiomarkerFeatures

        # Feature collection parameters
        self.alignmentDataPointers = np.zeros(len(self.biomarkerFeatureOrder), dtype=int)  # A list of pointers indicating the last seen aligned feature index for each analysis.
        self.rawFeaturePointers = np.zeros(len(self.biomarkerFeatureOrder), dtype=int)  # A list of pointers indicating the last seen raw feature index for each analysis.

    def unifyFeatureTimeWindows(self, featureTimeWindow):
        for featureAnalysis in self.featureAnalysisList:
            featureAnalysis.featureTimeWindow = featureTimeWindow

    # --------------------- Organize Incoming Features --------------------- #

    def organizeRawFeatures(self):
        # For each unique analysis with features.
        for analysis in self.featureAnalysisList:

            # For each channel in the analysis.
            for featureChannelInd in range(len(analysis.featureChannelIndices)):
                biomarkerInd = analysis.featureChannelIndices[featureChannelInd]
                rawFeaturePointer = self.rawFeaturePointers[biomarkerInd]

                # Organize the raw features; NOTE: I am assuming that the raw features are in order of the featureChannelIndices.
                self.rawFeatureTimesHolder[biomarkerInd].extend(analysis.rawFeatureTimes[featureChannelInd][rawFeaturePointer:])
                self.rawFeatureHolder[biomarkerInd].extend(analysis.rawFeatures[featureChannelInd][rawFeaturePointer:])

                # Update the raw pointers.
                self.rawFeaturePointers[biomarkerInd] = len(analysis.rawFeatureTimes[featureChannelInd])

                # Assert the integrity of the raw feature organization.
                assert len(self.rawFeatureTimesHolder[biomarkerInd]) == len(analysis.rawFeatureTimes[featureChannelInd]), \
                    f"Found {len(self.rawFeatureTimesHolder[biomarkerInd])} raw times and {len(self.rawFeatureHolder[biomarkerInd])} raw features. These must be the same length."
                assert len(self.rawFeatureHolder[biomarkerInd]) == len(analysis.rawFeatures[featureChannelInd]), \
                    f"Found {len(self.rawFeatureHolder[biomarkerInd])} raw features and {len(analysis.rawFeatures[featureChannelInd])} raw features. These must be the same length."

    def findCommonTimeRange(self):
        # Set up the parameters.
        rightAlignedTime = np.inf
        leftAlignedTime = 0

        # For each biomarker with features.
        for biomarkerInd in range(len(self.biomarkerFeatureOrder)):
            rawFeatureTimes = self.rawFeatureTimesHolder[biomarkerInd]

            # Update the min/max times.
            if len(rawFeatureTimes) != 0:
                leftAlignedTime = max(leftAlignedTime, rawFeatureTimes[0])
                rightAlignedTime = min(rightAlignedTime, rawFeatureTimes[-1])

        # Check if the time range is valid.
        if np.isinf(rightAlignedTime):
            return None, None

        return leftAlignedTime, rightAlignedTime

    def alignFeatures(self):
        # Create a time interval for interpolation.
        leftAlignedTime, rightAlignedTime = self.findCommonTimeRange()
        if rightAlignedTime is None: return None

        # Do not re-interpolate all the data.
        if len(self.alignedFeatureTimes) != 0:
            leftAlignedTime = self.alignedFeatureTimes[-1] + 1 / self.newSamplingFreq

        # Create the new time interval for interpolation.
        newInterpolatedTimes = np.arange(leftAlignedTime, rightAlignedTime, 1 / self.newSamplingFreq)

        # Store the new time points.
        self.alignedFeatureTimes.extend(newInterpolatedTimes.tolist())

        # Calculate the number of new points.
        numNewAlignedPoints = len(newInterpolatedTimes)
        newInterpolatedFeatureTimes = self.alignedFeatureTimes[-numNewAlignedPoints - self.interpBufferPoints:]
        numBufferPoints = len(newInterpolatedFeatureTimes) - numNewAlignedPoints

        # For each unique analysis with features.
        for analysis in self.featureAnalysisList:
            for featureChannelInd in range(len(analysis.featureChannelIndices)):

                # Extract the feature information.
                compiledFeatureData = analysis.compiledFeatures[featureChannelInd]  # Dim: numTimePoints, numBiomarkerFeatures
                biomarkerInd = analysis.featureChannelIndices[featureChannelInd]  # Dim: 1
                biomarkerFeatureInds = self.biomarkerFeatureInds[biomarkerInd]  # Dim: numBiomarkerFeatures
                rawFeatureTimes = analysis.rawFeatureTimes[featureChannelInd]  # Dim: numTimePoints

                # Get the feature information.
                alignmentDataPointer = max(0, self.alignmentDataPointers[biomarkerInd] - self.interpBufferPoints)
                compiledFeatures = compiledFeatureData[alignmentDataPointer:]  # Dim: numTimePoints, numBiomarkerFeatures
                compiledFeatureTimes = rawFeatureTimes[alignmentDataPointer:]  # Dim: numCompiledPoints

                # Interpolate the features.
                makimaInterpFunc = Akima1DInterpolator(compiledFeatureTimes, compiledFeatures, method='makima', axis=0)
                alignedFeatures = makimaInterpFunc(newInterpolatedFeatureTimes)
                # alignedFeatures dim: numTimePoints, numBiomarkerFeatures

                # Store the aligned features.
                for compiledFeatureInd, featureInd in enumerate(biomarkerFeatureInds):
                    self.alignedFeatures[featureInd][-numBufferPoints:] = alignedFeatures[0:numBufferPoints, compiledFeatureInd].tolist()
                    self.alignedFeatures[featureInd].extend(alignedFeatures[numBufferPoints:, compiledFeatureInd].tolist())

                # Update the alignment pointers.
                while rawFeatureTimes[self.alignmentDataPointers[biomarkerInd]] <= self.alignedFeatureTimes[-1]:
                    self.alignmentDataPointers[biomarkerInd] += 1

                    # Check if the alignment pointer is at the end of the data.
                    if self.alignmentDataPointers[biomarkerInd] == len(rawFeatureTimes): break
                # Do not overshoot the final aligned time.
                self.alignmentDataPointers[biomarkerInd] -= 1

    # ---------------------- Compile Incoming Features --------------------- #

    def compileModelFeatures(self, featureTimes, features, startTime, endTime):
        # Locate the experiment indices within the data
        endStimuliInd = np.searchsorted(featureTimes, endTime, side='left')
        startStimuliInd = np.searchsorted(featureTimes, startTime, side='right')

        # Check if the time range is valid.
        if endStimuliInd - startStimuliInd <= self.minNumExperimentalPoints:
            return None, None

        # Save raw interval information
        featureIntervalTimes = [featureTimes[startStimuliInd:endStimuliInd] for _ in range(len(features[0]))]
        featureIntervals = features[startStimuliInd:endStimuliInd]

        return featureIntervals, featureIntervalTimes

    def averageFeatures_static(self, rawFeatureTimes, rawFeatures, averageWindow, startTimeInd=0):
        # rawFeatures dim: numTimePoints, numBiomarkerFeatures
        # rawFeatureTimes dim: numTimePoints

        # Ensure proper initial data structure.
        rawFeatureTimes = np.asarray(rawFeatureTimes)
        rawFeatures = np.asarray(rawFeatures)

        compiledFeatures = []
        # Average the Feature Together at Each Point
        for timePointInd in range(startTimeInd, len(rawFeatureTimes)):
            currentTimepoint = rawFeatureTimes[timePointInd]

            # Get the interval of features to average
            timeMask = np.logical_and(
                currentTimepoint - averageWindow <= rawFeatureTimes,
                rawFeatureTimes <= currentTimepoint,
            )
            featureInterval = rawFeatures[timeMask]

            # Take the trimmed average
            compiledFeature = scipy.stats.trim_mean(featureInterval, proportiontocut=self.trimMeanCut, axis=0)
            compiledFeatures.append(compiledFeature)
        # compiledFeatures dim: numTimePoints, numBiomarkerFeatures

        return list(compiledFeatures)

    def compileContinuousFeatures(self, newFeatureTimes, newRawFeatures, rawFeatureTimes, rawFeatures, compiledFeatures, averageWindow):
        # newRawFeatures dim: numNewTimePoints, numBiomarkerFeatures
        # compiledFeatures dim: numTimePoints, numBiomarkerFeatures
        # rawFeatures dim: numTimePoints, numBiomarkerFeatures
        # newFeatureTimes dim: numNewTimePoints
        # rawFeatureTimes dim: numTimePoints

        # Assert the integrity of the feature compilation.
        assert len(newFeatureTimes) == len(newRawFeatures), f"Found {len(newFeatureTimes)} new times and {len(newRawFeatures)} new features. These must be the same length."
        assert len(rawFeatureTimes) == len(rawFeatures), f"Found {len(rawFeatureTimes)} times and {len(rawFeatures)} features. These must be the same length."
        if len(newFeatureTimes) == 0: return None

        startTimeInd = len(rawFeatures)
        # Append the new features to the raw features
        rawFeatureTimes.extend(list(newFeatureTimes))
        rawFeatures.extend(list(newRawFeatures))

        # Perform the feature averaging
        newCompiledFeatures = self.averageFeatures_static(rawFeatureTimes, rawFeatures, averageWindow, startTimeInd=startTimeInd)
        compiledFeatures.extend(newCompiledFeatures)

        # Assert the integrity of the feature compilation.
        assert len(rawFeatures) == len(compiledFeatures), f"Found {len(rawFeatures)} raw features and {len(compiledFeatures)} compiled features. {len(rawFeatures[0])} {len(compiledFeatures[0])}"

    def compileStaticFeatures(self, rawFeatureTimesHolder, rawFeatureHolder, featureAverageWindows):
        # rawFeatureHolder dim: numBiomarkers, numTimePoints, numBiomarkerFeatures
        # rawFeatureTimesHolder dim: numBiomarkers, numTimePoints
        # featureAverageWindows dim: numBiomarkers

        # Assert the integrity of the feature compilation.
        assert len(rawFeatureTimesHolder) == len(rawFeatureHolder), \
            f"Found {len(rawFeatureTimesHolder)} times and {len(rawFeatureHolder)} features. These must be the same length."

        compiledFeatureHolders = []
        # Average the features across a sliding window at each timePoint
        for biomarkerInd in range(len(rawFeatureTimesHolder)):
            rawFeatureTimes = rawFeatureTimesHolder[biomarkerInd]
            averageWindow = featureAverageWindows[biomarkerInd]
            rawFeatures = rawFeatureHolder[biomarkerInd]

            # Assert the integrity of the feature compilation.
            assert len(rawFeatureTimes) == len(rawFeatures), \
                f"Found {len(rawFeatureTimes)} times and {len(rawFeatures)} features. These must be the same length."

            # Perform the feature averaging
            compiledFeatures = self.averageFeatures_static(rawFeatureTimes, rawFeatures, averageWindow, startTimeInd=0)
            compiledFeatureHolders.append(compiledFeatures)
        # compiledFeatures dim: numTimePoints, numBiomarkerFeatures

        return compiledFeatureHolders
