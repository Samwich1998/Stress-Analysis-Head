# PyTorch
from torch import nn
from torch.utils.checkpoint import checkpoint

# Import machine learning files
from .signalEncoderModules import signalEncoderModules
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer


class channelEncoding(signalEncoderModules):

    def __init__(self, waveletType, numCompressedSignals, numExpandedSignals, expansionFactor, numSigEncodingLayers, sequenceBounds, numSigLiftedChannels):
        super(channelEncoding, self).__init__()
        # General parameters
        self.numSigLiftedChannels = numSigLiftedChannels  # The number of channels to lift to during signal encoding.
        self.numSigEncodingLayers = numSigEncodingLayers    # The number of operator layers during signal encoding.
        self.numCompressedSignals = numCompressedSignals    # Number of compressed signals.
        self.numExpandedSignals = numExpandedSignals        # Number of expanded signals.
        self.expansionFactor = expansionFactor              # Expansion factor for the model.
        self.sequenceBounds = sequenceBounds                # The minimum and maximum sequence length.

        # Neural operator parameters.
        self.numDecompositions = min(5, waveletNeuralOperatorLayer.max_decompositions(signal_length=self.sequenceBounds[0], wavelet_name=waveletType))  # Number of decompositions for the waveletType transform.
        self.activationMethod = self.getActivationMethod_channelEncoder()
        self.waveletType = waveletType  # wavelet type for the waveletType transform: bior, db3, dmey
        self.mode = 'zero'              # Mode for the waveletType transform.

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
            # Create the spectral convolution layers.
            self.compressedNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=self.numSigLiftedChannels, numOutputSignals=self.numSigLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, waveletType=self.waveletType, mode=self.mode,
                                                                                  addBiasTerm=False,  activationMethod=self.activationMethod, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', useConvolutionFlag=True, independentChannels=False, skipConnectionProtocol='identity'))
            self.expandedNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=self.numSigLiftedChannels, numOutputSignals=self.numSigLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, waveletType=self.waveletType, mode=self.mode,
                                                                                addBiasTerm=False, activationMethod=self.activationMethod, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', useConvolutionFlag=True, independentChannels=False, skipConnectionProtocol='identity'))

            # Create the processing layers.
            self.compressedProcessingLayers.append(self.signalPostProcessing(inChannel=self.numSigLiftedChannels, bottleneckChannel=self.numCompressedSignals))
            self.expandedProcessingLayers.append(self.signalPostProcessing(inChannel=self.numSigLiftedChannels, bottleneckChannel=self.numExpandedSignals))

        # Initialize final models.
        self.projectingCompressionModel = self.projectionOperator(inChannel=self.numSigLiftedChannels, outChannel=self.numCompressedSignals)
        self.projectingExpansionModel = self.projectionOperator(inChannel=self.numSigLiftedChannels, outChannel=self.numExpandedSignals)

        # Smoothing kernel for the channel encoding.
        self.gausKernel_forChanEnc = self.smoothingKernel(kernelSize=3)

    # ---------------------------------------------------------------------- #
    # ----------------------- Signal Encoding Methods ---------------------- #

    def compressionAlgorithm(self, inputData):
        return self.applyChannelEncoding(inputData, self.heuristicCompressionModel, self.liftingCompressionModel, self.compressedNeuralOperatorLayers, self.compressedProcessingLayers, self.projectingCompressionModel)

    def expansionAlgorithm(self, inputData):
        return self.applyChannelEncoding(inputData, self.heuristicExpansionModel, self.liftingExpansionModel, self.expandedNeuralOperatorLayers, self.expandedProcessingLayers, self.projectingExpansionModel)

    def applyChannelEncoding(self, inputData, heuristicModel, liftingModel, neuralOperatorLayers, processingLayers, projectingModel):
        # Learn the initial signal.
        liftedData = liftingModel(inputData)
        liftedData = self.applySmoothing(liftedData, self.gausKernel_forChanEnc)
        # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

        # For each encoder model.
        processedData = liftedData.clone()
        for modelInd in range(self.numSigEncodingLayers):
            # Apply the neural operator and the skip connection.
            processedData = checkpoint(neuralOperatorLayers[modelInd], processedData, 0, use_reentrant=False)
            # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

            # Apply non-linearity to the processed data.
            processedData = checkpoint(processingLayers[modelInd], processedData, use_reentrant=False)
            processedData = self.applySmoothing(processedData, self.gausKernel_forChanEnc) + liftedData
            # processedData dimension: batchSize, numSigLiftedChannels, signalDimension

        # Learn the final signal.
        processedData = projectingModel(processedData)
        # processedData dimension: batchSize, numOutputSignals, signalDimension

        # Add the heuristic model as a baseline.
        processedData = processedData + heuristicModel(inputData)
        # processedData dimension: batchSize, numOutputSignals, signalDimension

        return processedData
