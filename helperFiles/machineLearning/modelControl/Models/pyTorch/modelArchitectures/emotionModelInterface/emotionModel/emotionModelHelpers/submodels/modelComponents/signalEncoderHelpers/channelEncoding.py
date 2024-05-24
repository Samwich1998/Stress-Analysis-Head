# PyTorch
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# Import machine learning files
from .signalEncoderModules import signalEncoderModules
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer


class channelEncoding(signalEncoderModules):

    def __init__(self, numCompressedSignals, numExpandedSignals, expansionFactor, numSigEncodingLayers, sequenceBounds, numSigLiftedChannels, debuggingResults=False):
        super(channelEncoding, self).__init__()
        # General parameters
        self.numSigLiftedChannels = numSigLiftedChannels  # The number of channels to lift to during signal encoding.
        self.numSigEncodingLayers = numSigEncodingLayers    # The number of operator layers during signal encoding.
        self.numCompressedSignals = numCompressedSignals    # Number of compressed signals.
        self.numExpandedSignals = numExpandedSignals        # Number of expanded signals.
        self.debuggingResults = debuggingResults            # Whether to print debugging results. Type: bool
        self.expansionFactor = expansionFactor              # Expansion factor for the model.
        self.sequenceBounds = sequenceBounds                # The minimum and maximum sequence length.

        # Neural operator parameters.
        self.numDecompositions = 2     # Number of decompositions for the wavelet transform.
        self.wavelet = 'db3'           # Wavelet type for the wavelet transform: bior3.7, db3, dmey
        self.mode = 'zero'             # Mode for the wavelet transform.

        # Initialize initial lifting models.
        self.liftingCompressionModel = self.liftingOperator(inChannel=self.numExpandedSignals, outChannel=self.numSigLiftedChannels)
        self.liftingExpansionModel = self.liftingOperator(inChannel=self.numCompressedSignals, outChannel=self.numSigLiftedChannels)

        # Initialize the neural operator layer.
        self.compressedNeuralOperatorLayers = nn.ModuleList([])
        self.expandedNeuralOperatorLayers = nn.ModuleList([])

        # Initialize the processing layers.
        self.compressedProcessingLayers = nn.ModuleList([])
        self.expandedProcessingLayers = nn.ModuleList([])

        # For each encoder model.
        for modelInd in range(self.numSigEncodingLayers):
            # Create the spectral convolution layers.
            self.compressedNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=self.numSigLiftedChannels + self.numExpandedSignals, numOutputSignals=self.numSigLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', independentChannels=False, skipConnectionProtocol='singleCNN'))
            self.expandedNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=self.numSigLiftedChannels + self.numCompressedSignals, numOutputSignals=self.numSigLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', independentChannels=False, skipConnectionProtocol='singleCNN'))

            # Create the processing layers.
            self.compressedProcessingLayers.append(self.signalPostProcessing(inChannel=self.numSigLiftedChannels))
            self.expandedProcessingLayers.append(self.signalPostProcessing(inChannel=self.numSigLiftedChannels))

        # Initialize final models.
        self.projectingCompressionModel = self.projectionOperator(inChannel=self.numSigLiftedChannels, outChannel=self.numCompressedSignals)
        self.projectingExpansionModel = self.projectionOperator(inChannel=self.numSigLiftedChannels, outChannel=self.numExpandedSignals)

    # ---------------------------------------------------------------------- #
    # ----------------------- Signal Encoding Methods ---------------------- #

    def compressionAlgorithm(self, inputData):
        # Learn the initial signal.
        processedData = self.liftingCompressionModel(inputData)
        # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

        # For each encoder model.
        for modelInd in range(self.numSigEncodingLayers):
            # Keep attention to the initial signal.
            processedData = torch.cat(tensors=(inputData, processedData), dim=1)
            # processedData dimension: batchSize, numSigLiftedChannels + numExpandedSignals, signalDimension

            # Apply the neural operator and the skip connection.
            processedData = checkpoint(self.compressedNeuralOperatorLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

            # Apply non-linearity to the processed data.
            processedData = checkpoint(self.compressedProcessingLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

        # Learn the final signal.
        processedData = self.projectingCompressionModel(processedData)
        # processedData dimension: batchSize, numCompressedSignals, signalDimension

        return processedData

    def expansionAlgorithm(self, inputData):
        # Learn the initial signal.
        processedData = self.liftingExpansionModel(inputData)
        # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

        # For each encoder model.
        for modelInd in range(self.numSigEncodingLayers):
            # Keep attention to the initial signal.
            processedData = torch.cat(tensors=(inputData, processedData), dim=1)
            # processedData dimension: batchSize, numSigLiftedChannels + numCompressedSignals, signalDimension

            # Apply the neural operator and the skip connection.
            processedData = checkpoint(self.expandedNeuralOperatorLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

            # Apply non-linearity to the processed data.
            processedData = checkpoint(self.expandedProcessingLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

        # Learn the final signal.
        processedData = self.projectingExpansionModel(processedData)
        # processedData dimension: batchSize, numExpandedSignals, signalDimension

        return processedData
