# General
import time

# PyTorch
import torch
from torchsummary import summary

# Import helper models
from .signalEncoderHelpers.signalEncoderHelpers import signalEncoderHelpers


class signalEncoderBase(signalEncoderHelpers):
    def __init__(self, sequenceBounds=(90, 300), numExpandedSignals=2, numSigEncodingLayers=5, numSigLiftedChannels=48, waveletType='bior3.7', debuggingResults=False):
        super(signalEncoderBase, self).__init__(sequenceBounds=sequenceBounds, numExpandedSignals=numExpandedSignals, numSigEncodingLayers=numSigEncodingLayers,
                                                numSigLiftedChannels=numSigLiftedChannels, waveletType=waveletType, debuggingResults=debuggingResults)

    # ---------------------------- Loss Methods ---------------------------- #

    def calculateEncodingLoss(self, originalData, encodedData):
        # originalData  encodedDecodedOriginalData
        #          \         /
        #          encodedData

        # Set up the variables for signal encoding.
        originalNumSignals = originalData.size(1)
        numEncodedSignals = encodedData.size(1)

        # Calculate the number of active signals in each path.
        numActiveSignals = originalNumSignals - self.simulateSignalPath(originalNumSignals, numEncodedSignals)[1]

        # Reverse operation
        if numEncodedSignals < originalNumSignals:
            encodedDecodedOriginalData = self.expansionModel(encodedData, originalNumSignals)
        else:
            encodedDecodedOriginalData = self.compressionModel(encodedData, originalNumSignals)
        # Assert the integrity of the expansions/compressions.
        assert encodedDecodedOriginalData.size(1) == originalData.size(1)

        # Calculate the squared error loss for this layer of compression/expansion.
        squaredErrorLoss_forward = (originalData - encodedDecodedOriginalData)[:, :numActiveSignals, :].pow(2).mean(dim=2).mean(dim=1)
        if self.debuggingResults: print("\tSignal encoder reverse layer losses:", squaredErrorLoss_forward.mean().item())

        return squaredErrorLoss_forward

    def updateLossValues(self, originalData, encodedData, signalEncodingLayerLoss):
        # Keep tracking of the loss through each loop.
        layerLoss = self.calculateEncodingLoss(originalData, encodedData)

        # If the loss is significant, add it to the total loss.
        if 0.1 < signalEncodingLayerLoss.mean().item(): signalEncodingLayerLoss = 2*signalEncodingLayerLoss
        if 0.001 < layerLoss.mean().item(): signalEncodingLayerLoss = signalEncodingLayerLoss + layerLoss

        return signalEncodingLayerLoss

    # ---------------------------------------------------------------------- #
    # -------------------------- Data Organization ------------------------- #

    def expansionModel(self, originalData, targetNumSignals):
        # Unpair the signals with their neighbors.
        unpairedData, frozenData, numActiveSignals = self.unpairSignals(originalData, targetNumSignals)
        # activeData dimension: batchSize*numActiveSignals/numCompressedSignals, numCompressedSignals, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension

        # Increase the number of signals.
        expandedData = self.channelEncodingInterface.expansionAlgorithm(unpairedData)
        # expandedData dimension: batchSize*numSignalPairs, 2, signalDimension

        # Recompile the signals to their original dimension.
        signalData = self.recompileSignals(expandedData, frozenData)
        # signalData dimension: batchSize, 2*numSignalPairs + numFrozenSignals, signalDimension

        return signalData

    def compressionModel(self, originalData, targetNumSignals):
        # Pair up the signals with their neighbors.
        pairedData, frozenData, numActiveSignals = self.pairSignals(originalData, targetNumSignals)
        # pairedData dimension: batchSize*numActiveSignals/numExpandedSignals, numExpandedSignals, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension

        # Reduce the number of signals.
        reducedPairedData = self.channelEncodingInterface.compressionAlgorithm(pairedData)
        # reducedPairedData dimension: batchSize*numSignalPairs, 1, signalDimension

        # Recompile the signals to their original dimension.
        signalData = self.recompileSignals(reducedPairedData, frozenData)
        # signalData dimension: batchSize, numSignalPairs + numFrozenSignals, signalDimension

        return signalData


# -------------------------- Encoder Architecture -------------------------- #

class generalSignalEncoding(signalEncoderBase):
    def __init__(self, sequenceBounds=(90, 300), numExpandedSignals=2, numSigEncodingLayers=5, numSigLiftedChannels=48, waveletType="bior3.7", debuggingResults=False):
        super(generalSignalEncoding, self).__init__(sequenceBounds=sequenceBounds, numExpandedSignals=numExpandedSignals, numSigEncodingLayers=numSigEncodingLayers,
                                                    numSigLiftedChannels=numSigLiftedChannels, waveletType=waveletType, debuggingResults=debuggingResults)

    def forward(self, signalData, targetNumSignals=32, signalEncodingLayerLoss=None, calculateLoss=True, forward=True):
        """ The shape of signalData: (batchSize, numSignals, compressedLength) """
        # Initialize first time parameters for signal encoding.
        if signalEncodingLayerLoss is None: signalEncodingLayerLoss = torch.zeros((signalData.size(0),), device=signalData.device)

        # Set up the variables for signal encoding.
        batchSize, numSignals, signalDimension = signalData.size()
        numSignalPath = [numSignals]  # Keep track of the signal's at each iteration.

        # Assert that we have the expected data format.
        assert self.numCompressedSignals <= targetNumSignals, f"At the minimum, we cannot go lower than compressed signal batch. You provided {targetNumSignals} signals."
        assert self.sequenceBounds[0] <= signalDimension <= self.sequenceBounds[1], f"Can only process signals that are within the {self.sequenceBounds}. You provided a sequence of length {signalDimension}"
        assert self.numCompressedSignals <= numSignals, f"We cannot compress or expand if we dont have at least the compressed signal batch. You provided {numSignals} signals."

        # ------------- Signal Compression/Expansion Algorithm ------------- #

        # While we have the incorrect number of signals.
        while targetNumSignals != signalData.size(1):
            compressedDataFlag = targetNumSignals < signalData.size(1)
            originalData = signalData.clone()  # Keep track of the initial state

            # Compress the signals down to the targetNumSignals.
            if compressedDataFlag: signalData = self.compressionModel(signalData, targetNumSignals)

            # Expand the signals up to the targetNumSignals.
            else: signalData = self.expansionModel(signalData, targetNumSignals)

            # Apply smoothing to the signals in the forward direction.
            if forward: signalData = self.denoiseSignals.applySmoothing_forSigEnc(signalData)

            # Keep track of the error during each compression/expansion.
            if calculateLoss: signalEncodingLayerLoss = self.updateLossValues(originalData, signalData, signalEncodingLayerLoss)

            # Keep track of the signal's at each iteration.
            numSignalPath.append(signalData.size(1))

        # ------------------------------------------------------------------ #

        # Assert the integrity of the expansion/compression.
        if numSignals != targetNumSignals:
            assert all(numSignalPath[i] <= numSignalPath[i + 1] for i in range(len(numSignalPath) - 1)) \
                   or all(numSignalPath[i] >= numSignalPath[i + 1] for i in
                          range(len(numSignalPath) - 1)), "List is not sorted up or down"

        # Remove the target signal from the path.
        numSignalPath.pop()

        return signalData, numSignalPath, signalEncodingLayerLoss

    def printParams(self, numSignals=50, sequenceBounds=(90, 300)):
        # generalSignalEncoding(numExpandedSignals=3, sequenceBounds=(90, 300)).to('cpu').printParams(numSignals=100, sequenceBounds=(90, 300))
        t1 = time.time()
        summary(self, (numSignals, sequenceBounds[1]))
        t2 = time.time()
        if self.debuggingResults: print(t2 - t1)

        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.debuggingResults: print(f'The model has {numParams} trainable parameters.')
