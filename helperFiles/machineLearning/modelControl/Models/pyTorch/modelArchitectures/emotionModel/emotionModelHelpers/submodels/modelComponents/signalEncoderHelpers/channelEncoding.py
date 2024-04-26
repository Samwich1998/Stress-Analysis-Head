# PyTorch
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# Import machine learning files
from .signalEncoderModules import signalEncoderModules
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer


class channelEncoding(signalEncoderModules):

    def __init__(self, numCompressedSignals, numExpandedSignals, expansionFactor, numEncoderLayers, sequenceBounds, numLiftedChannels):
        super(channelEncoding, self).__init__()
        # General parameters
        self.numCompressedSignals = numCompressedSignals    # Number of compressed signals.
        self.numExpandedSignals = numExpandedSignals        # Number of expanded signals.
        self.numEncoderLayers = numEncoderLayers            # Number of encoder layers.
        self.expansionFactor = expansionFactor              # Expansion factor for the model.
        self.sequenceBounds = sequenceBounds                # The minimum and maximum sequence length.

        # Neural operator parameters.
        self.numLiftedChannels = numLiftedChannels  # Number of channels to lift the signal to.

        # Initialize initial lifting models.
        self.liftingCompressionModel = self.liftingOperator(inChannel=self.numExpandedSignals, outChannel=self.numLiftedChannels)
        self.liftingExpansionModel = self.liftingOperator(inChannel=self.numCompressedSignals, outChannel=self.numLiftedChannels)

        # Initialize the neural operator layer.
        self.compressedNeuralOperatorLayers = nn.ModuleList([])
        self.expandedNeuralOperatorLayers = nn.ModuleList([])

        # Initialize the processing layers.
        self.compressedProcessingLayers = nn.ModuleList([])
        self.expandedProcessingLayers = nn.ModuleList([])

        # For each encoder model.
        for modelInd in range(self.numEncoderLayers):
            # Create the spectral convolution layers.
            self.compressedNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=self.numLiftedChannels + self.numExpandedSignals, numOutputSignals=self.numLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=1, wavelet='db3', mode='zero', numLayers=1,  encodeLowFrequency=True, encodeHighFrequencies=True))
            self.expandedNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=self.numLiftedChannels + self.numCompressedSignals, numOutputSignals=self.numLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=1, wavelet='db3', mode='zero', numLayers=1, encodeLowFrequency=True, encodeHighFrequencies=True))
            # Notes: I found that dmey and db3 work well. coif3 is not that good. db6 is bad.

            # Create the processing layers.
            self.compressedProcessingLayers.append(self.signalPostProcessing(inChannel=self.numLiftedChannels))
            self.expandedProcessingLayers.append(self.signalPostProcessing(inChannel=self.numLiftedChannels))

        # Initialize final models.
        self.projectingCompressionModel = self.projectionOperator(inChannel=self.numLiftedChannels, outChannel=self.numCompressedSignals)
        self.projectingExpansionModel = self.projectionOperator(inChannel=self.numLiftedChannels, outChannel=self.numExpandedSignals)

    # ---------------------------------------------------------------------- #
    # ----------------------- Signal Encoding Methods ---------------------- #

    def compressionAlgorithm(self, inputData):
        # Learn the initial signal.
        processedData = self.liftingCompressionModel(inputData)
        # processedData dimension: batchSize, numLiftedChannels, signalDimension

        # For each encoder model.
        for modelInd in range(self.numEncoderLayers):
            # Keep attention to the initial signal.
            processedData = torch.cat(tensors=(processedData, inputData), dim=1)
            # processedData dimension: batchSize, numLiftedChannels + numExpandedSignals, signalDimension

            # Apply the neural operator and the skip connection.
            processedData = checkpoint(self.compressedNeuralOperatorLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numLiftedChannels, signalDimension

            # Apply non-linearity to the processed data.
            processedData = checkpoint(self.compressedProcessingLayers[modelInd], processedData, use_reentrant=False)

        # Learn the final signal.
        processedData = self.projectingCompressionModel(processedData)
        # processedData dimension: batchSize, numCompressedSignals, signalDimension

        return processedData

    def expansionAlgorithm(self, inputData):
        # Learn the initial signal.
        processedData = self.liftingExpansionModel(inputData)
        # processedData dimension: batchSize, numLiftedChannels, signalDimension

        # For each encoder model.
        for modelInd in range(self.numEncoderLayers):
            # Keep attention to the initial signal.
            processedData = torch.cat(tensors=(processedData, inputData), dim=1)
            # processedData dimension: batchSize, numLiftedChannels + numCompressedSignals, signalDimension

            # Apply the neural operator and the skip connection.
            processedData = checkpoint(self.expandedNeuralOperatorLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numLiftedChannels, signalDimension

            # Apply non-linearity to the processed data.
            processedData = checkpoint(self.expandedProcessingLayers[modelInd], processedData, use_reentrant=False)

        # Learn the final signal.
        processedData = self.projectingExpansionModel(processedData)
        # processedData dimension: batchSize, numExpandedSignals, signalDimension

        return processedData
