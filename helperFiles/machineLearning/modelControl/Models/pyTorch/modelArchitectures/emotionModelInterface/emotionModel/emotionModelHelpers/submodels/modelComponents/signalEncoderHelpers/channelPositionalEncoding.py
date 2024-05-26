# PyTorch
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward

# Import machine learning files
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from .signalEncoderModules import signalEncoderModules


class channelPositionalEncoding(signalEncoderModules):
    def __init__(self, waveletType, sequenceBounds=(90, 300)):
        super(channelPositionalEncoding, self).__init__()
        # General parameters.
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence length.

        # Positional encoding parameters.
        self.numEncodingStamps = 8  # The number of binary bits in the encoding (010 = 2 signals; 3 encodings). Max: 256 signals -> 2**8.
        self.maxNumEncodedSignals = 2 ** self.numEncodingStamps  # The maximum number of signals that can be encoded.

        # Neural operator parameters.
        self.activationMethod = self.getActivationMethod_posEncoder()
        self.waveletType = waveletType  # wavelet type for the waveletType transform: bior, db3, dmey
        self.numDecompositions = 1     # Number of decompositions for the wavelet transform.
        self.mode = 'zero'             # Mode for the wavelet transform.

        # Create the spectral convolution layers.
        self.unlearnNeuralOperatorLayers = waveletNeuralOperatorLayer(numInputSignals=1, numOutputSignals=1, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, waveletType=self.waveletType, mode=self.mode, addBiasTerm=False,
                                                                      activationMethod=self.activationMethod, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='none', useCNN=False, independentChannels=True, skipConnectionProtocol='identity')
        self.learnNeuralOperatorLayers = waveletNeuralOperatorLayer(numInputSignals=1, numOutputSignals=1, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, waveletType=self.waveletType, mode=self.mode, addBiasTerm=False,
                                                                    activationMethod=self.activationMethod, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='none', useCNN=False, independentChannels=True, skipConnectionProtocol='identity')
        self.lowFrequencyShape = self.learnNeuralOperatorLayers.lowFrequencyShape

        # A list of parameters to encode each signal.
        self.encodingStamp = nn.ParameterList()  # A list of learnable parameters for learnable signal positions.
        self.decodingStamp = nn.ParameterList()  # A list of learnable parameters for learnable signal positions.

        # For each encoding bit.
        for stampInd in range(self.numEncodingStamps):
            # Assign a learnable parameter to the signal.
            self.encodingStamp.append(self.positionalEncodingStamp(self.lowFrequencyShape))
            self.decodingStamp.append(self.positionalEncodingStamp(self.lowFrequencyShape))

        # Initialize the wavelet decomposition and reconstruction layers.
        self.dwt_indexPredictor = DWT1DForward(J=self.numDecompositions, wave=self.waveletType, mode=self.mode)
        self.posIndexPredictor = self.predictedPosEncodingIndex(numFeatures=self.lowFrequencyShape, numClasses=self.maxNumEncodedSignals)

    # ---------------------------------------------------------------------- #
    # -------------------- Learned Positional Encoding --------------------- #

    def addPositionalEncoding(self, inputData):
        return self.positionalEncoding(inputData, self.encodingStamp, self.learnNeuralOperatorLayers)

    def removePositionalEncoding(self, inputData):
        return self.positionalEncoding(inputData, self.decodingStamp, self.unlearnNeuralOperatorLayers)

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
        positionEncodedData = learnNeuralOperatorLayers(inputData, lowFrequencyTerms=finalStamp, highFrequencyTerms=None)
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
        # finalStamp dim: batchSize, numSignals, lowFrequencyShape

        return finalStamp

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
