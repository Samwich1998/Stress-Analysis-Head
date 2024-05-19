# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Import machine learning files
from .signalEncoderModules import signalEncoderModules
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer


class channelEncoding(signalEncoderModules):

    def __init__(self, numCompressedSignals, numExpandedSignals, expansionFactor, numEncoderLayers, sequenceBounds, numLiftedChannels, debuggingResults=False):
        super(channelEncoding, self).__init__()
        # General parameters
        self.numCompressedSignals = numCompressedSignals    # Number of compressed signals.
        self.numExpandedSignals = numExpandedSignals        # Number of expanded signals.
        self.numEncoderLayers = numEncoderLayers            # Number of encoder layers.
        self.debuggingResults = debuggingResults            # Whether to print debugging results. Type: bool
        self.expansionFactor = expansionFactor              # Expansion factor for the model.
        self.sequenceBounds = sequenceBounds                # The minimum and maximum sequence length.

        # Neural operator parameters.
        self.numDecompositions = 2     # Number of decompositions for the wavelet transform.
        self.wavelet = 'db3'           # Wavelet type for the wavelet transform: bior3.7, db3, dmey
        self.mode = 'zero'             # Mode for the wavelet transform.

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
            self.compressedNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=self.numLiftedChannels + self.numExpandedSignals, numOutputSignals=self.numLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, numLayers=1, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', skipConnectionProtocol='CNN'))
            self.expandedNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=self.numLiftedChannels + self.numCompressedSignals, numOutputSignals=self.numLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, numLayers=1, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', skipConnectionProtocol='CNN'))

            # Create the processing layers.
            self.compressedProcessingLayers.append(self.signalPostProcessing(inChannel=self.numLiftedChannels))
            self.expandedProcessingLayers.append(self.signalPostProcessing(inChannel=self.numLiftedChannels))

        # Initialize final models.
        self.projectingCompressionModel = self.projectionOperator(inChannel=self.numLiftedChannels, outChannel=self.numCompressedSignals)
        self.projectingExpansionModel = self.projectionOperator(inChannel=self.numLiftedChannels, outChannel=self.numExpandedSignals)

        self.gaussian_kernel = nn.Parameter(torch.tensor([1., 2., 1.]).reshape(1, 1, 3) / 4, requires_grad=False)

    # ---------------------------------------------------------------------- #
    # ----------------------- Signal Encoding Methods ---------------------- #

    def compressionAlgorithm(self, inputData):
        # Learn the initial signal.
        processedData = self.liftingCompressionModel(inputData)
        # processedData dimension: batchSize, numLiftedChannels, signalDimension

        # For each encoder model.
        for modelInd in range(self.numEncoderLayers):
            # Keep attention to the initial signal.
            processedData = torch.cat(tensors=(inputData, processedData), dim=1)
            # processedData dimension: batchSize, numLiftedChannels + numExpandedSignals, signalDimension

            # Apply the neural operator and the skip connection.
            processedData = checkpoint(self.compressedNeuralOperatorLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numLiftedChannels, signalDimension

            # Apply non-linearity to the processed data.
            processedData = checkpoint(self.compressedProcessingLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numLiftedChannels, signalDimension

            processedData = F.conv1d(processedData, self.gaussian_kernel, padding=1)

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
            processedData = torch.cat(tensors=(inputData, processedData), dim=1)
            # processedData dimension: batchSize, numLiftedChannels + numCompressedSignals, signalDimension

            # Apply the neural operator and the skip connection.
            processedData = checkpoint(self.expandedNeuralOperatorLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numLiftedChannels, signalDimension

            # Apply non-linearity to the processed data.
            processedData = checkpoint(self.expandedProcessingLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numLiftedChannels, signalDimension

        # Learn the final signal.
        processedData = self.projectingExpansionModel(processedData)
        # processedData dimension: batchSize, numExpandedSignals, signalDimension

        return processedData


class Smoothing1DCNN(nn.Module):
    def __init__(self):
        super(Smoothing1DCNN, self).__init__()
        # Define a 1D Gaussian filter
        self.gaussian_kernel = nn.Parameter(torch.tensor([1., 2., 1.]).reshape(1, 1, 3) / 4, requires_grad=False)

    def forward(self, x):
        # Apply average filter
        x_avg = F.conv1d(x, self.average_kernel, padding=1)
        # Apply Gaussian filter
        x_gauss = F.conv1d(x, self.gaussian_kernel, padding=1)
        return x_avg, x_gauss