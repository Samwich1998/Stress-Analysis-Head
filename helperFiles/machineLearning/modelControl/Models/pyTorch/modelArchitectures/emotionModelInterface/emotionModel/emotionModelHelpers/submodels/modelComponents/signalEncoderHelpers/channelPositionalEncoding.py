# PyTorch
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward

# Import machine learning files
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from .signalEncoderModules import signalEncoderModules


class channelPositionalEncoding(signalEncoderModules):
    def __init__(self, sequenceBounds=(90, 300), debuggingResults=False):
        super(channelPositionalEncoding, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence length.

        # Positional encoding parameters.
        self.numPosLiftedChannels = 1  # The number of channels to lift to during positional encoding.
        self.numPosEncodingLayers = 1  # The number of operator layers during positional encoding.
        self.numEncodingStamps = 8  # The number of binary bits in the encoding (010 = 2 signals; 3 encodings). Max: 256 signals -> 2**8.
        self.maxNumEncodedSignals = 2 ** self.numEncodingStamps  # The maximum number of signals that can be encoded.

        # Neural operator parameters.
        self.numDecompositions = 2     # Number of decompositions for the wavelet transform.
        self.wavelet = 'bior3.7'       # Wavelet type for the wavelet transform: bior3.7, db3, dmey
        self.mode = 'zero'             # Mode for the wavelet transform.

        # Create the spectral convolution layers.
        self.unlearnNeuralOperatorLayer = waveletNeuralOperatorLayer(numInputSignals=self.numPosLiftedChannels, numOutputSignals=self.numPosLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', independentChannels=True, skipConnectionProtocol='none')
        self.learnNeuralOperatorLayer = waveletNeuralOperatorLayer(numInputSignals=self.numPosLiftedChannels, numOutputSignals=self.numPosLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', independentChannels=True, skipConnectionProtocol='none')
        self.lowFrequencyShape = self.learnNeuralOperatorLayer.lowFrequencyShape

        # Create the post-processing layers.
        self.unlearnPostProcessingLayer = self.signalPostProcessing_forPosEnc()
        self.learnPostProcessingLayer = self.signalPostProcessing_forPosEnc()

        # A list of parameters to encode each signal.
        self.encodingStamp = nn.ParameterList()  # A list of learnable parameters for learnable signal positions.
        self.decodingStamp = nn.ParameterList()  # A list of learnable parameters for learnable signal positions.

        # For each encoding bit.
        for stampInd in range(self.numEncodingStamps):
            # Assign a learnable parameter to the signal.
            self.encodingStamp.append(self.positionalEncodingStamp(self.lowFrequencyShape))
            self.decodingStamp.append(self.positionalEncodingStamp(self.lowFrequencyShape))

        # Initialize the encoding parameters.
        self.unlearnStampEncodingFNN = self.learnEncodingStampFNN(numFeatures=self.lowFrequencyShape)
        self.learnStampEncodingFNN = self.learnEncodingStampFNN(numFeatures=self.lowFrequencyShape)

        # Smoothing kernels.
        self.gausKernel_forPosStamp = self.smoothingKernel(kernelSize=3)

        # Initialize the wavelet decomposition and reconstruction layers.
        self.dwt_indexPredictor = DWT1DForward(J=self.numDecompositions, wave=self.wavelet, mode=self.mode)
        self.posIndexPredictor = self.predictedPosEncodingIndex(numFeatures=self.lowFrequencyShape, numClasses=self.maxNumEncodedSignals)

    # ---------------------------------------------------------------------- #
    # -------------------- Learned Positional Encoding --------------------- #

    def addPositionalEncoding(self, inputData):
        return self.positionalEncoding(inputData, self.encodingStamp, self.learnStampEncodingFNN, self.learnNeuralOperatorLayer, self.learnPostProcessingLayer)

    def removePositionalEncoding(self, inputData):
        return self.positionalEncoding(inputData, self.decodingStamp, self.unlearnStampEncodingFNN, self.unlearnNeuralOperatorLayer, self.unlearnPostProcessingLayer)

    def positionalEncoding(self, inputData, encodingStamp, learnStampEncodingFNN, learnNeuralOperatorLayer, learnPostProcessingLayer):
        # Initialize and learn an encoded stamp for each signal index.
        finalStamp = self.compileStampEncoding(inputData, encodingStamp, learnStampEncodingFNN)
        positionEncodedData = self.applyNeuralOperator(inputData, finalStamp, learnNeuralOperatorLayer, learnPostProcessingLayer)
        
        return positionEncodedData

    def applyNeuralOperator(self, inputData, finalStamp, learnNeuralOperatorLayer, learnPostProcessingLayer):
        # Set up the variables for signal encoding.
        batchSize, numSignals, signalDimension = inputData.size()

        # Assert the validity of the input parameters.
        assert numSignals <= self.maxNumEncodedSignals, "The number of signals exceeds the maximum encoding limit."

        # Smoothen out the stamp.
        finalStamp = self.applySmoothing(finalStamp, kernelWeights=self.gausKernel_forPosStamp)
        # finalStamp dimension: batchSize, numSignals, lowFrequencyShape

        # Apply the neural operator and the skip connection.
        positionEncodedData = learnNeuralOperatorLayer(inputData, lowFrequencyTerms=finalStamp, highFrequencyTerms=None)
        # positionEncodedData dimension: batchSize, numSignals, signalDimension

        # Projection operators to compress signal information.
        positionEncodedData = positionEncodedData.view(batchSize * numSignals, 1, signalDimension)
        positionEncodedData = learnPostProcessingLayer(positionEncodedData)
        # positionEncodedData dimension: batchSize*numSignals, 1, signalDimension

        # Reshape the data back into the original format.
        positionEncodedData = positionEncodedData.view(batchSize, numSignals, signalDimension) + inputData
        # positionEncodedData dimension: batchSize, numSignals, signalDimension

        return positionEncodedData

    def compileStampEncoding(self, inputData, encodingStamp, learnStampEncodingFNN):
        # Set up the variables for signal encoding.
        batchSize, numSignals, signalDimension = inputData.size()
        finalStamp = torch.zeros((batchSize, numSignals, self.lowFrequencyShape), device=inputData.device)

        # Extract the size of the input parameter.
        bitInds = torch.arange(self.numEncodingStamps).to(inputData.device)
        signalInds = torch.arange(numSignals).to(inputData.device)

        # Generate the binary encoding of signalInds in a batched manner
        binary_encoding = signalInds[:, None].bitwise_and(2 ** bitInds).bool()
        # binary_encoding dim: numSignals, numEncodingStamps

        # For each stamp encoding
        for stampInd in range(self.numEncodingStamps):
            # Smooth the encoding stamp.
            currentStamp = encodingStamp[stampInd]

            # Check each signal if it is using this specific encoding.
            usingStampEncoding = binary_encoding[:, stampInd:stampInd + 1]
            encodingVector = usingStampEncoding.float() * currentStamp
            # encodingVector dim: numSignals, lowFrequencyShape

            # Add the stamp encoding to all the signals in all the batches.
            finalStamp = finalStamp + (stampInd % 2 == 0) * encodingVector.unsqueeze(0)
            # finalStamp dim: batchSize, numSignals, lowFrequencyShape. Note, the unused signals are essentially zero-padded.

        # Synthesize the signal encoding information.
        finalStamp = learnStampEncodingFNN(finalStamp)

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
