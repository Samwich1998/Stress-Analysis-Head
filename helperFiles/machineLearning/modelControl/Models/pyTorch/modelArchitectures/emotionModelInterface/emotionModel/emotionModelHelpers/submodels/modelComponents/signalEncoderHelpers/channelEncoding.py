from torch.utils.checkpoint import checkpoint
from torch import nn

from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from .signalEncoderModules import signalEncoderModules


class channelEncoding(signalEncoderModules):

    def __init__(self, waveletType, numCompressedSignals, numExpandedSignals, expansionFactor, numSigEncodingLayers, sequenceBounds, numSigLiftedChannels):
        super(channelEncoding, self).__init__()
        # General parameters
        self.numSigLiftedChannels = numSigLiftedChannels  # The number of channels to lift to during signal encoding.
        self.numSigEncodingLayers = numSigEncodingLayers  # The number of operator layers during signal encoding.
        self.numCompressedSignals = numCompressedSignals  # Number of compressed signals.
        self.numExpandedSignals = numExpandedSignals  # Number of expanded signals.
        self.expansionFactor = expansionFactor  # Expansion factor for the model.
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence length.

        # Neural operator parameters.
        self.numDecompositions = min(5, waveletNeuralOperatorLayer.max_decompositions(signal_length=self.sequenceBounds[0], wavelet_name=waveletType))  # Number of decompositions for the waveletType transform.
        self.activationMethod = self.getActivationMethod_channelEncoder()
        self.waveletType = waveletType  # wavelet type for the waveletType transform: bior, db3, dmey
        self.mode = 'zero'  # Mode for the waveletType transform.

        # initialize the heuristic method.
        self.heuristicCompressionModel = self.heuristicEncoding(inChannel=self.numExpandedSignals, outChannel=self.numCompressedSignals)
        self.heuristicExpansionModel = self.heuristicEncoding(inChannel=self.numCompressedSignals, outChannel=self.numExpandedSignals)

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
            self.addModelBlock()  # Add a block to build up the model.

        # Initialize final models.
        self.projectingCompressionModel = self.projectionOperator(inChannel=self.numSigLiftedChannels, outChannel=self.numCompressedSignals)
        self.projectingExpansionModel = self.projectionOperator(inChannel=self.numSigLiftedChannels, outChannel=self.numExpandedSignals)

    def addModelBlock(self):
        # Create the spectral convolution layers.
        compressedNeuralOperatorLayer, compressedPostProcessingLayer = self.initializeNeuralLayer(numInputSignals=self.numSigLiftedChannels, numOutputSignals=self.numSigLiftedChannels, bottleneckChannel=self.numCompressedSignals)
        expandedNeuralOperatorLayer, expandedPostProcessingLayer = self.initializeNeuralLayer(numInputSignals=self.numSigLiftedChannels, numOutputSignals=self.numSigLiftedChannels, bottleneckChannel=self.numExpandedSignals)

        # Create the spectral convolution layers.
        self.compressedNeuralOperatorLayers.append(compressedNeuralOperatorLayer)
        self.expandedNeuralOperatorLayers.append(expandedNeuralOperatorLayer)

        # Create the processing layers.
        self.compressedProcessingLayers.append(compressedPostProcessingLayer)
        self.expandedProcessingLayers.append(expandedPostProcessingLayer)

    def initializeNeuralLayer(self, numInputSignals, numOutputSignals, bottleneckChannel):
        # Create the spectral convolution layers.
        neuralOperatorLayers = waveletNeuralOperatorLayer(numInputSignals=numInputSignals, numOutputSignals=numOutputSignals, sequenceBounds=self.sequenceBounds, numDecompositions=self.numDecompositions, waveletType=self.waveletType,
                                                          mode=self.mode, addBiasTerm=False, smoothingKernelSize=3, activationMethod=self.activationMethod, encodeLowFrequencyProtocol='lowFreq',
                                                          encodeHighFrequencyProtocol='highFreq', useConvolutionFlag=True, independentChannels=False, skipConnectionProtocol='identity')

        # Create the post-processing layers.
        signalPostProcessing = self.signalPostProcessing(inChannel=numInputSignals, bottleneckChannel=bottleneckChannel)

        return neuralOperatorLayers, signalPostProcessing

    # ---------------------------------------------------------------------- #
    # ----------------------- Signal Encoding Methods ---------------------- #

    def compressionAlgorithm(self, inputData):
        return self.applyChannelEncoding(inputData, self.heuristicCompressionModel, self.liftingCompressionModel, self.compressedNeuralOperatorLayers, self.compressedProcessingLayers, self.projectingCompressionModel)

    def expansionAlgorithm(self, inputData):
        return self.applyChannelEncoding(inputData, self.heuristicExpansionModel, self.liftingExpansionModel, self.expandedNeuralOperatorLayers, self.expandedProcessingLayers, self.projectingExpansionModel)

    def applyChannelEncoding(self, inputData, heuristicModel, liftingModel, neuralOperatorLayers, processingLayers, projectingModel):
        # Learn the initial signal.
        processedData = liftingModel(inputData)
        # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

        # For each encoder model.
        for modelInd in range(self.numSigEncodingLayers):
            # Apply the neural operator and the skip connection.
            processedData = checkpoint(neuralOperatorLayers[modelInd], processedData, 0, use_reentrant=False)
            # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

            # Apply non-linearity to the processed data.
            processedData = checkpoint(processingLayers[modelInd], processedData, use_reentrant=False)
            # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

        # Learn the final signal.
        processedData = projectingModel(processedData)
        # processedData dimension: batchSize, numOutputSignals, signalDimension

        # Add the heuristic model as a baseline.
        processedData = processedData + heuristicModel(inputData)
        # processedData dimension: batchSize, numOutputSignals, signalDimension

        return processedData
