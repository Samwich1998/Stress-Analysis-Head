# PyTorch
import torch
import torch.nn as nn

# Import machine learning files
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from .signalEncoderModules import signalEncoderModules


class channelPositionalEncoding(signalEncoderModules):
    def __init__(self, sequenceBounds=(90, 300), numPosEncodingLayers=2, numPosLiftedChannels=4, debuggingResults=False):
        super(channelPositionalEncoding, self).__init__()
        # General parameters.
        self.numPosEncodingLayers = numPosEncodingLayers  # The number of operator layers during positional encoding.
        self.numPosLiftedChannels = numPosLiftedChannels  # The number of channels to lift to during positional encoding.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence length.

        # Positional encoding parameters.
        self.numConvolutionalLayers = 2  # The number of convolutional layers to learn the encoding.
        self.numEncodingStamps = 10  # The number of binary bits in the encoding (010 = 2 signals; 3 encodings).

        # Neural operator parameters.
        self.numDecompositions = 2     # Number of decompositions for the wavelet transform.
        self.wavelet = 'bior3.7'       # Wavelet type for the wavelet transform: bior3.7, db3, dmey
        self.mode = 'zero'             # Mode for the wavelet transform.

        # Initialize the neural operator layer.
        self.learnNeuralOperatorLayers = nn.ModuleList([])
        self.unlearnNeuralOperatorLayers = nn.ModuleList([])
        # Initialize the neural operator layer.
        self.learnPostProcessingLayers = nn.ModuleList([])
        self.unlearnPostProcessingLayers = nn.ModuleList([])

        # For each encoder model.
        for modelInd in range(self.numPosEncodingLayers):
            # Create the spectral convolution layers.
            self.learnNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=self.numPosLiftedChannels + 1, numOutputSignals=self.numPosLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, numLayers=1, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', skipConnectionProtocol='singleCNN'))
            self.unlearnNeuralOperatorLayers.append(waveletNeuralOperatorLayer(numInputSignals=self.numPosLiftedChannels + 1, numOutputSignals=self.numPosLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, numLayers=1, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', skipConnectionProtocol='singleCNN'))

            # Create the post-processing layers.
            self.learnPostProcessingLayers.append(self.signalPostProcessing_forPosEnc(inChannel=self.numPosLiftedChannels))
            self.unlearnPostProcessingLayers.append(self.signalPostProcessing_forPosEnc(inChannel=self.numPosLiftedChannels))
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
        self.learnStampEncodingFNN = self.learnEncodingStampFNN(numFeatures=self.lowFrequencyShape)
        self.unlearnStampEncodingFNN = self.learnEncodingStampFNN(numFeatures=self.lowFrequencyShape)

        # Initialize the lifting operators.
        self.learnedLiftingModel = self.liftingOperator_forPosEnc(outChannels=self.numPosLiftedChannels)
        self.unlearnedLiftingModel = self.liftingOperator_forPosEnc(outChannels=self.numPosLiftedChannels)

        # Initialize the projection operators.
        self.learnedProjectionModel = self.projectionOperator_forPosEnc(inChannels=self.numPosLiftedChannels)
        self.unlearnedProjectionModel = self.projectionOperator_forPosEnc(inChannels=self.numPosLiftedChannels)

        # Smoothing kernels.
        self.gausKernel_forPosStamp = self.smoothingKernel(kernelSize=3)

    # ---------------------------------------------------------------------- #
    # -------------------- Learned Positional Encoding --------------------- #

    def addPositionalEncoding(self, inputData):
        return self.positionalEncoding(inputData, self.encodingStamp, self.learnStampEncodingFNN, self.learnedLiftingModel, self.learnNeuralOperatorLayers, self.learnPostProcessingLayers, self.learnedProjectionModel)

    def removePositionalEncoding(self, inputData):
        return self.positionalEncoding(inputData, self.decodingStamp, self.unlearnStampEncodingFNN, self.unlearnedLiftingModel, self.unlearnNeuralOperatorLayers, self.unlearnPostProcessingLayers, self.unlearnedProjectionModel)

    def positionalEncoding(self, inputData, encodingStamp, learnStampEncodingFNN, learnedLiftingModel, learnNeuralOperatorLayers, learnPostProcessingLayers, projectionModel):
        # Initialize and learn an encoded stamp for each signal index.
        finalStamp = self.compileStampEncoding(inputData, encodingStamp, learnStampEncodingFNN)
        positionEncodedData = self.applyNeuralOperator(inputData, finalStamp, learnedLiftingModel, learnNeuralOperatorLayers, learnPostProcessingLayers, projectionModel)
        
        return positionEncodedData

    def applyNeuralOperator(self, inputData, finalStamp, learnedLiftingModel, learnNeuralOperatorLayers, learnPostProcessingLayers, projectionModel):
        # Set up the variables for signal encoding.
        batchSize, numSignals, signalDimension = inputData.size()

        # Reshape the data and add stamp encoding to process each signal separately.
        inputData = inputData.view(batchSize * numSignals, 1, signalDimension)
        finalStamp = finalStamp.view(batchSize * numSignals, 1, self.lowFrequencyShape)
        # positionEncodedData dimension: batchSize*numSignals, 1, signalDimension
        # finalStamp dimension: batchSize*numSignals, 1, lowFrequencyShape

        # Lifting operators to expand signal information.
        positionEncodedData = learnedLiftingModel(inputData)
        # positionEncodedData dimension: batchSize*numSignals, numPosLiftedChannels, signalDimension

        # For each neural operator layer.
        for modelInd in range(self.numPosEncodingLayers):
            # Keep attention to the initial signal.
            positionEncodedData = torch.cat(tensors=(inputData, positionEncodedData), dim=1)
            # processedData dimension: batchSize*numSignals, numPosLiftedChannels + 1, signalDimension
            
            # Apply the neural operator and the skip connection.
            positionEncodedData = learnNeuralOperatorLayers[modelInd](positionEncodedData, lowFrequencyTerms=finalStamp, highFrequencyTerms=None)
            # positionEncodedData dimension: batchSize*numSignals, numPosLiftedChannels, signalDimension

            # Apply non-linearity to the processed data.
            positionEncodedData = learnPostProcessingLayers[modelInd](positionEncodedData)
            # positionEncodedData dimension: batchSize*numSignals, numPosLiftedChannels, signalDimension

        # Projection operators to compress signal information.
        positionEncodedData = projectionModel(positionEncodedData)
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
            # Smooth the encoding stamp.
            currentStamp = self.applySmoothing(encodingStamp[stampInd].unsqueeze(0).unsqueeze(0), kernelWeights=self.gausKernel_forPosStamp).squeeze(0).squeeze(0)

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
