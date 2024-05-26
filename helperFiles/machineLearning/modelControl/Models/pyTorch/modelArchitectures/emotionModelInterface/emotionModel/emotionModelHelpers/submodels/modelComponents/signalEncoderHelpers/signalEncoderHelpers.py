# General
import math
from fractions import Fraction

# PyTorch
import torch
from torch import nn

# Import files for machine learning
from .channelPositionalEncoding import channelPositionalEncoding
from ....emotionDataInterface import emotionDataInterface
from .channelEncoding import channelEncoding
from .changeVariance import changeVariance
from .denoiser import denoiser


class signalEncoderHelpers(nn.Module):
    def __init__(self, sequenceBounds=(90, 240), signalMinMaxScale=2, numExpandedSignals=2, numSigEncodingLayers=5, numSigLiftedChannels=48, waveletType='bior3.7', debuggingResults=False):
        super(signalEncoderHelpers, self).__init__()
        # General
        self.numSigEncodingLayers = numSigEncodingLayers          # The number of layers to encode the signals.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool

        # Compression/Expansion parameters.
        self.sequenceBounds = sequenceBounds                # The minimum and maximum number of signals in any expansion/compression.
        self.numExpandedSignals = numExpandedSignals        # The final number of signals in any expansion
        self.numCompressedSignals = numExpandedSignals - 1  # The final number of signals in any compression.
        self.expansionFactor = Fraction(self.numExpandedSignals, self.numCompressedSignals)  # The percent expansion.
        # Assert the integrity of the input parameters.
        assert self.numExpandedSignals - self.numCompressedSignals == 1, "You should only gain 1 channel when expanding or else you may overshoot."

        # Initialize signal encoder helper classes.
        self.channelEncodingInterface = channelEncoding(waveletType=waveletType, numCompressedSignals=self.numCompressedSignals, numExpandedSignals=self.numExpandedSignals, expansionFactor=self.expansionFactor, numSigEncodingLayers=numSigEncodingLayers, sequenceBounds=self.sequenceBounds, numSigLiftedChannels=numSigLiftedChannels)
        self.positionalEncodingInterface = channelPositionalEncoding(waveletType=waveletType, sequenceBounds=self.sequenceBounds)
        self.denoiseSignals = denoiser(waveletType=waveletType, sequenceBounds=sequenceBounds)
        self.finalVarianceInterface = changeVariance(signalMinMaxScale=signalMinMaxScale)
        self.dataInterface = emotionDataInterface

    # ----------------------- Signal Pairing Methods ----------------------- #

    @staticmethod
    def roundNextSignal(compressingSignalFlag, numActiveSignals, expansionFactor):
        if compressingSignalFlag:
            return math.ceil(numActiveSignals / expansionFactor)
        else:
            return math.floor(numActiveSignals * expansionFactor)

    def simulateSignalPath(self, numSignals, targetNumSignals):
        # Initialize parameters for the simulated compression/expansion.
        numUnchangedSignals = numSignals  # Start off with no signals modified.
        numSignalsPath = [numSignals]

        # Calculate the multiplicative factor for expanding/compressing.
        compressingSignalFlag = False
        if targetNumSignals < numSignals:
            compressingSignalFlag = True

        # While we are compressing/expanding.
        while numSignals != targetNumSignals:
            # Simulate the next active/frozen batch.
            numActiveSignals = self.getNumActiveSignals(numSignals, targetNumSignals)
            numFrozenSignals = numSignals - numActiveSignals

            # Update how many initial signals have now been modified.
            numUnchangedSignals = min(numUnchangedSignals, numFrozenSignals)

            # Update the number of signals for the next round.
            numSignals = numFrozenSignals + self.roundNextSignal(compressingSignalFlag, numActiveSignals, self.expansionFactor)
            numSignalsPath.append(numSignals)

            # Assert you didnt miss the value
            if compressingSignalFlag: assert targetNumSignals <= numSignals, f"{targetNumSignals}, {numSignals}"
            else: assert targetNumSignals <= numSignals, f"{targetNumSignals}, {numSignals}"

        return numSignalsPath, int(numUnchangedSignals)

    def getNumActiveSignals(self, numSignals, targetNumSignals):
        # If we are upsampling the signals as much as I can.
        if numSignals * self.expansionFactor <= targetNumSignals:
            # Upsample the max number of signals.
            numActiveSignals = numSignals - (numSignals % self.numCompressedSignals)

        # If we are only slightly below the final number of signals.
        elif numSignals < targetNumSignals < numSignals * self.expansionFactor:
            # Find the number of signals to expand.
            numSignalsGained = targetNumSignals - numSignals
            numExpansions = numSignalsGained / (self.numExpandedSignals - self.numCompressedSignals)
            numActiveSignals = numExpansions * self.numCompressedSignals
            assert numActiveSignals <= numSignals, "This must be true if the logic is working."

        # If we are only slightly above the final number of signals.
        elif targetNumSignals < numSignals < targetNumSignals * self.expansionFactor:
            # Find the number of signals to reduce.
            numSignalsLost = numSignals - targetNumSignals
            numCompressions = numSignalsLost / (self.numExpandedSignals - self.numCompressedSignals)
            numActiveSignals = numCompressions * self.numExpandedSignals
            assert numActiveSignals <= numSignals, "This must be true if the logic is working."

        # If we are reducing the signals as much as I can.
        elif targetNumSignals * self.expansionFactor <= numSignals:
            # We can only pair up an even number.
            numActiveSignals = numSignals - (numSignals % self.numExpandedSignals)

        # Base case: numSignals == targetNumSignals
        else:
            numActiveSignals = 0
        numActiveSignals = int(numActiveSignals)

        return numActiveSignals

    def separateActiveData(self, inputData, targetNumSignals):
        # Extract the number opf signals.
        numSignals = inputData.size(1)
        numActiveSignals = self.getNumActiveSignals(numSignals, targetNumSignals)

        # Segment the tensor into its frozen and active components.
        activeData = inputData[:, 0:numActiveSignals].contiguous()  # Reducing these signals.
        frozenData = inputData[:, numActiveSignals:numSignals].contiguous()  # These signals are finalized.
        # Only the last rows of signals are frozen.

        return activeData, frozenData, numActiveSignals

    # ---------------------- Data Structure Interface ---------------------- #

    def pairSignals(self, inputData, targetNumSignals):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.shape

        # Separate out the active and frozen data.
        activeData, frozenData, numActiveSignals = self.separateActiveData(inputData, targetNumSignals)
        # activeData dimension: batchSize, numActiveSignals, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension

        # Pair up the signals.
        numSignalPairs = int(numActiveSignals / self.numExpandedSignals)
        pairedData = activeData.view(batchSize, numSignalPairs, self.numExpandedSignals, signalDimension)
        pairedData = pairedData.view(batchSize * numSignalPairs, self.numExpandedSignals, signalDimension)
        # pairedData dimension: batchSize*numSignalPairs, numExpandedSignals, signalDimension

        return pairedData, frozenData, numActiveSignals

    def unpairSignals(self, inputData, targetNumSignals):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.shape

        # Separate out the active and frozen data.
        activeData, frozenData, numActiveSignals = self.separateActiveData(inputData, targetNumSignals)
        # activeData dimension: batchSize, numActiveSignals, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension

        # Unpair up the signals.
        numUnpairedBatches = int(activeData.size(0) * numActiveSignals / self.numCompressedSignals)
        unpairedData = activeData.view(numUnpairedBatches, self.numCompressedSignals, signalDimension)  # Create a channel for the CNN.
        # unpairedData dimension: batchSize*numSignalPairs, numCompressedSignals, signalDimension

        return unpairedData, frozenData, numActiveSignals

    @staticmethod
    def recompileSignals(pairedData, frozenData):
        # Extract the incoming data's dimension.
        batchPairedSize, numChannels, signalDimension = pairedData.shape
        batchSize, numFrozenSignals, signalDimension = frozenData.shape

        # Separate out the paired data into its batches.
        unpairedData = pairedData.view(batchSize, int(numChannels * batchPairedSize / batchSize), signalDimension)
        # unpairedData dimension: batchSize, numSignalPairs, signalDimension

        # Recombine the paired and frozen data.
        recombinedData = torch.cat((unpairedData, frozenData), dim=1).contiguous()
        # recombinedData dimension: batchSize, numSignalPairs + numFrozenSignals, signalDimension

        return recombinedData
