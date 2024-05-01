# General
import time

# PyTorch
import torch
from torchsummary import summary

# Import helper models
from .signalEncoderHelpers.signalEncoderHelpers import signalEncoderHelpers
from ...emotionDataInterface import emotionDataInterface


class signalEncoderBase(signalEncoderHelpers):
    def __init__(self, sequenceBounds=(90, 300), numExpandedSignals=2, numEncodingLayers=5, numLiftedChannels=48, accelerator=None):
        super(signalEncoderBase, self).__init__(sequenceBounds, numExpandedSignals, numEncodingLayers, numLiftedChannels)
        # General parameters.
        self.accelerator = accelerator  # Hugging face model optimizations.

    # ---------------------------- Loss Methods ---------------------------- #

    def calculateEncodingLoss(self, originalData, encodedData, trainingFlag):
        # originalData  encodedDecodedOriginalData
        #          \         /
        #          encodedData

        # Set up the variables for signal encoding.
        originalNumSignals = originalData.size(1)
        numEncodedSignals = encodedData.size(1)

        # Add noise to the data to ensure that the latent space is continuous.
        noisyEncodedData = emotionDataInterface.addNoise(encodedData, trainingFlag, noiseSTD=0.001)

        # Calculate the number of active signals in each path.
        numActiveSignals = originalNumSignals - self.simulateSignalPath(originalNumSignals, numEncodedSignals)[1]

        # Reverse operation
        if numEncodedSignals < originalNumSignals:
            encodedDecodedOriginalData = self.expansionModel(noisyEncodedData, originalNumSignals)
        else:
            encodedDecodedOriginalData = self.compressionModel(noisyEncodedData, originalNumSignals)
        # Assert the integrity of the expansions/compressions.
        assert encodedDecodedOriginalData.size(1) == originalData.size(1)

        # Calculate the squared error loss for this layer of compression/expansion.
        squaredErrorLoss_forward = (originalData - encodedDecodedOriginalData)[:, :numActiveSignals, :].pow(2).mean(dim=2).mean(dim=1)
        print("\tSignal encoder reverse operation loss:", squaredErrorLoss_forward.mean().item(), flush=True)

        return squaredErrorLoss_forward

    def updateLossValues(self, originalData, encodedData, signalEncodingLayerLoss, trainingFlag):
        # Keep tracking of the loss through each loop.
        layerLoss = self.calculateEncodingLoss(originalData, encodedData, trainingFlag)

        # If the loss is significant, add it to the total loss.
        signalEncodingLayerLoss = 1.1*signalEncodingLayerLoss + layerLoss

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
    def __init__(self, sequenceBounds=(90, 300), numExpandedSignals=2, numEncodingLayers=5, numLiftedChannels=48, accelerator=None):
        super(generalSignalEncoding, self).__init__(sequenceBounds, numExpandedSignals, numEncodingLayers, numLiftedChannels, accelerator)

    def forward(self, signalData, targetNumSignals=32, signalEncodingLayerLoss=None, calculateLoss=True, trainingFlag=False):
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
            originalData = signalData.clone()  # Keep track of the initial stateF

            # Compress the signals down to the targetNumSignals.
            if compressedDataFlag: signalData = self.compressionModel(signalData, targetNumSignals)

            # Expand the signals up to the targetNumSignals.
            else: signalData = self.expansionModel(signalData, targetNumSignals)

            if calculateLoss:
                # Keep track of the error during each compression/expansion.
                signalEncodingLayerLoss = self.updateLossValues(originalData, signalData, signalEncodingLayerLoss, trainingFlag)

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
        print(t2 - t1)

        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
