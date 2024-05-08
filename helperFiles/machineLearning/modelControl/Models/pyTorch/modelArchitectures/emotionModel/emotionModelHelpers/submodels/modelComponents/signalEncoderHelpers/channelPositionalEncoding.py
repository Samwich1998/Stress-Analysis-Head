# PyTorch
import torch
import torch.nn as nn

# Import machine learning files
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from .signalEncoderModules import signalEncoderModules


class channelPositionalEncoding(signalEncoderModules):
    def __init__(self, sequenceBounds=(90, 300), debuggingResults=False):
        super(channelPositionalEncoding, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence length.

        # Positional encoding parameters.
        self.numConvolutionalLayers = 2  # The number of convolutional layers to learn the encoding.
        self.numEncodingStamps = 10  # The number of binary bits in the encoding (010 = 2 signals; 3 encodings).
        self.numOperatorLayers = 1  # The number of layers to learn the encoding.

        # Neural operator parameters.
        self.numDecompositions = 2     # Number of decompositions for the wavelet transform.
        self.wavelet = 'bior3.7'       # Wavelet type for the wavelet transform: bior3.7, db3, dmey
        self.mode = 'zero'             # Mode for the wavelet transform.

        # Initialize the neural operator layer.
        self.learnNeuralOperatorLayers = nn.ModuleList([])
        self.unlearnNeuralOperatorLayers = nn.ModuleList([])

        # For each encoder model.
        for modelInd in range(self.numOperatorLayers):
            # Create the spectral convolution layers.
            self.learnNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=1, numOutputSignals=1, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, numLayers=1, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='none', skipConnectionProtocol='linearCNN'))
            self.unlearnNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=1, numOutputSignals=1, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, numLayers=1, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='none', skipConnectionProtocol='linearCNN'))
        self.lowFrequencyShape = self.learnNeuralOperatorLayers[0].lowFrequencyShape

        # A list of parameters to encode each signal.
        self.encodingStamp = nn.ParameterList()  # A list of learnable parameters for learnable signal positions.
        self.decodingStamp = nn.ParameterList()  # A list of learnable parameters for learnable signal positions.

        # For each encoding bit.
        for stampInd in range(self.numEncodingStamps):
            # Assign a learnable parameter to the signal.
            self.encodingStamp.append(self.positionalEncodingStamp(self.lowFrequencyShape))
            self.decodingStamp.append(self.positionalEncodingStamp(self.lowFrequencyShape))

        # Initialize the encoding parameters.
        self.learnStampEncodingCNN = nn.ModuleList()
        self.unlearnStampEncodingCNN = nn.ModuleList()

        # For each encoding matrix.
        for stampInd in range(self.numConvolutionalLayers):
            # Learn the encoding stamp for each signal.
            self.learnStampEncodingCNN.append(self.learnEncodingStampCNN())
            self.unlearnStampEncodingCNN.append(self.learnEncodingStampCNN())

        # Initialize the encoding parameters.
        self.learnStampEncodingFNN = self.learnEncodingStampFNN(numFeatures=self.lowFrequencyShape)
        self.unlearnStampEncodingFNN = self.learnEncodingStampFNN(numFeatures=self.lowFrequencyShape)

        # Initialize helper classes.
        self.dataInterface = emotionDataInterface

    # ---------------------------------------------------------------------- #
    # -------------------- Learned Positional Encoding --------------------- #

    def addPositionalEncoding(self, inputData):
        return self.positionalEncoding(inputData, self.encodingStamp, self.learnNeuralOperatorLayers, self.learnStampEncodingCNN, self.learnStampEncodingFNN)

    def removePositionalEncoding(self, inputData):
        return self.positionalEncoding(inputData, self.decodingStamp, self.unlearnNeuralOperatorLayers, self.unlearnStampEncodingCNN, self.unlearnStampEncodingFNN)

    def positionalEncoding(self, inputData, encodingStamp, learnNeuralOperatorLayers, learnStampEncodingCNN, learnStampEncodingFNN):
        # Apply a small network to learn the encoding.
        positionEncodedData = self.encodingInterface(inputData, learnStampEncodingCNN[0], useCheckpoint=False) + inputData

        # Initialize and learn an encoded stamp for each signal index.
        finalStamp = self.compileStampEncoding(inputData, encodingStamp, learnStampEncodingFNN)
        positionEncodedData = self.applyNeuralOperator(positionEncodedData, finalStamp, learnNeuralOperatorLayers)

        # Apply a small network to learn the encoding.
        positionEncodedData = self.encodingInterface(positionEncodedData, learnStampEncodingCNN[1], useCheckpoint=False) + inputData

        return positionEncodedData

    def applyNeuralOperator(self, positionEncodedData, finalStamp, learnNeuralOperatorLayers):
        # Set up the variables for signal encoding.
        batchSize, numSignals, signalDimension = positionEncodedData.size()

        # Reshape the data and add stamp encoding to process each signal separately.
        positionEncodedData = positionEncodedData.view(batchSize * numSignals, 1, signalDimension)
        finalStamp = finalStamp.view(batchSize * numSignals, 1, self.lowFrequencyShape)
        # positionEncodedData dimension: batchSize*numSignals, 1, signalDimension
        # finalStamp dimension: batchSize*numSignals, 1, lowFrequencyShape

        # For each neural operator layer.
        for modelInd in range(self.numOperatorLayers):
            # Apply the neural operator and the skip connection.
            positionEncodedData = learnNeuralOperatorLayers[modelInd](positionEncodedData, lowFrequencyTerms=finalStamp, highFrequencyTerms=None)
            finalStamp = None  # Only add the stamp encoding to the first layer.
            # positionEncodedData dimension: batchSize*numSignals, 1, signalDimension

        # Reshape the data back into the original format.
        positionEncodedData = positionEncodedData.view(batchSize, numSignals, signalDimension)

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
            # Check each signal if it is using this specific encoding.
            usingStampEncoding = binary_encoding[:, stampInd:stampInd + 1]
            encodingVector = usingStampEncoding.float() * encodingStamp[stampInd]
            # encodingVector dim: numSignals, lowFrequencyShape

            # Add the stamp encoding to all the signals in all the batches.
            finalStamp = finalStamp + (stampInd % 2 == 0) * encodingVector.unsqueeze(0)
            # finalStamp dim: batchSize, numSignals, lowFrequencyShape. Note, the unused signals are essentially zero-padded.

        # Synthesize the signal encoding information.
        finalStamp = learnStampEncodingFNN(finalStamp)

        return finalStamp
