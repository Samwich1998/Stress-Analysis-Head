# PyTorch
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward

# Import machine learning files
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from .signalEncoderModules import signalEncoderModules


class channelPositionalEncoding(signalEncoderModules):
    def __init__(self, waveletType, sequenceBounds=(90, 300), signalMinMaxScale=1):
        super(channelPositionalEncoding, self).__init__()
        # General parameters.
        self.signalMinMaxScale = signalMinMaxScale  # The minimum and maximum signal values.
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence length.

        # Positional encoding parameters.
        self.numEncodingStamps = 8  # The number of binary bits in the encoding (010 = 2 signals; 3 encodings). Max: 256 signals -> 2**8.
        self.maxNumEncodedSignals = 2 ** self.numEncodingStamps  # The maximum number of signals that can be encoded.
        self.smoothingKernel_forPosEnc = self.getSmoothingKernel(kernelSize=11)

        # Neural operator parameters.
        self.activationMethod = self.getActivationMethod_posEncoder()
        self.waveletType = waveletType  # wavelet type for the waveletType transform: bior, db3, dmey
        self.numDecompositions = 1     # Number of decompositions for the wavelet transform.
        self.mode = 'zero'             # Mode for the wavelet transform.

        # Create the spectral convolution layers.
        self.unlearnNeuralOperatorLayers = waveletNeuralOperatorLayer(numInputSignals=1, numOutputSignals=1, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, waveletType=self.waveletType, mode=self.mode, addBiasTerm=False, smoothingKernelSize=11,
                                                                      activationMethod=self.activationMethod, encodeLowFrequencyProtocol='none', encodeHighFrequencyProtocol='none', useConvolutionFlag=False, independentChannels=True, skipConnectionProtocol='none')
        self.learnNeuralOperatorLayers = waveletNeuralOperatorLayer(numInputSignals=1, numOutputSignals=1, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, waveletType=self.waveletType, mode=self.mode, addBiasTerm=False, smoothingKernelSize=11,
                                                                    activationMethod=self.activationMethod, encodeLowFrequencyProtocol='none', encodeHighFrequencyProtocol='none', useConvolutionFlag=False, independentChannels=True, skipConnectionProtocol='none')
        self.minLowFrequencyShape = self.learnNeuralOperatorLayers.minLowFrequencyShape  # For me, this was 49
        self.lowFrequencyShape = self.learnNeuralOperatorLayers.lowFrequencyShape  # For me, this was 124

        # Positional encoding parameters.
        self.learnPosFrequencies = self.getFrequencyParams(self.numEncodingStamps)
        self.unlearnPosFrequencies = self.getFrequencyParams(self.numEncodingStamps)

        # Initialize the wavelet decomposition and reconstruction layers.
        self.dwt_indexPredictor = DWT1DForward(J=self.numDecompositions, wave=self.waveletType, mode=self.mode)
        self.posIndexPredictor = self.predictedPosEncodingIndex(numFeatures=self.lowFrequencyShape)

    # ---------------------------------------------------------------------- #
    # -------------------- Learned Positional Encoding --------------------- #

    def addPositionalEncoding(self, inputData):
        # Compile the encoding stamp.
        encodingStamp = self.compileStampWaveforms(self.learnPosFrequencies)
        posEncodedData = self.positionalEncoding(inputData, encodingStamp, self.learnNeuralOperatorLayers)

        return posEncodedData

    def removePositionalEncoding(self, inputData):
        # Compile the decoding stamp.
        decodingStamp = self.compileStampWaveforms(self.unlearnPosFrequencies)
        originalData = self.positionalEncoding(inputData, decodingStamp, self.unlearnNeuralOperatorLayers)

        return originalData

    def positionalEncoding(self, inputData, encodingStamp, learnNeuralOperatorLayers):
        # Initialize and learn an encoded stamp for each signal index.
        finalStamp = self.compileStampEncoding(inputData, encodingStamp)
        positionEncodedData = self.applyNeuralOperator(inputData, finalStamp, learnNeuralOperatorLayers)
        
        return positionEncodedData

    def applyNeuralOperator(self, inputData, finalStamp, learnNeuralOperatorLayers):
        # Set up the variables for signal encoding.
        batchSize, numSignals, signalDimension = inputData.size()

        # Assert the validity of the input parameters.
        assert numSignals <= self.maxNumEncodedSignals, "The number of signals exceeds the maximum encoding limit."

        # Apply the neural operator and the skip connection.
        positionEncodedData = learnNeuralOperatorLayers(torch.zeros_like(inputData), extraSkipConnection=0, lowFrequencyTerms=finalStamp, highFrequencyTerms=None) + inputData
        # positionEncodedData dimension: batchSize, numSignals, signalDimension

        return positionEncodedData

    def compileStampEncoding(self, inputData, encodingStamp):
        # Extract the input data dimensions.
        batchSize, numSignals, signalDimension = inputData.size()

        # Set up the variables for signal encoding.
        numStampsUsed = torch.zeros(numSignals, device=inputData.device)
        finalStamp = torch.zeros((batchSize, numSignals, self.lowFrequencyShape), device=inputData.device)

        # Extract the size of the input parameter.
        bitInds = torch.arange(self.numEncodingStamps).to(inputData.device)
        signalInds = torch.arange(numSignals).to(inputData.device)

        # Generate the binary encoding of signalInds in a batched manner
        binary_encoding = signalInds[:, None].bitwise_and(2 ** bitInds).bool()
        # binary_encoding dim: numSignals, numEncodingStamps

        # For each stamp encoding
        for stampInd in range(self.numEncodingStamps):
            # Check each signal if it is using this specific encoding.
            usingStampEncoding = binary_encoding[:, stampInd:stampInd + 1].float()
            encodingVector = usingStampEncoding * encodingStamp[stampInd]
            # encodingVector dim: numSignals, lowFrequencyShape

            # Keep track of the stamps added.
            numStampsUsed = numStampsUsed + usingStampEncoding.squeeze(1)

            # Add the stamp encoding to all the signals in all the batches.
            finalStamp = finalStamp + encodingVector.unsqueeze(0)
            # finalStamp dim: batchSize, numSignals, lowFrequencyShape. Note, the unused signals are essentially zero-padded.

        # Normalize the final stamp.
        finalStamp = finalStamp / numStampsUsed.clamp(min=1).unsqueeze(0).unsqueeze(-1)

        return finalStamp

    def compileStampWaveforms(self, learnPosFrequencies):
        # Initialize the encoding stamp.
        encodingStamp = []

        # For each encoding bit.
        for stampInd in range(self.numEncodingStamps):
            # Assign a learnable parameter to the signal.
            encodingStamp.append(self.positionalEncodingStamp(self.lowFrequencyShape, frequency=learnPosFrequencies[stampInd], signalMinMaxScale=self.signalMinMaxScale))

        return encodingStamp

    def predictSignalIndex(self, inputData):
        # Extract the input data dimensions.
        batchSize, numInputSignals, sequenceLength = inputData.size()

        # Pad the data to the maximum sequence length.
        inputData = torch.nn.functional.pad(inputData, pad=(self.sequenceBounds[1] - sequenceLength, 0), mode='constant', value=0)
        # inputData dimension: batchSize, numInputSignals, maxSequenceLength

        # Perform wavelet decomposition.
        lowFrequency, _ = self.dwt_indexPredictor(inputData)  # Note: each channel is treated independently here.
        # highFrequencies[decompositionLayer] dimension: batchSize, numInputSignals, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numInputSignals, lowFrequencyShape

        # Predict the signal index.
        predictedIndexProbabilities = self.posIndexPredictor(lowFrequency)
        # predictedIndexProbabilities dimension: batchSize, numInputSignals, maxNumEncodingSignals

        return predictedIndexProbabilities
