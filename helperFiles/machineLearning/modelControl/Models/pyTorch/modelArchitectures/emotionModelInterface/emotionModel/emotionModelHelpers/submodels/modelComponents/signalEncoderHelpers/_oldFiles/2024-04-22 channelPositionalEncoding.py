# PyTorch
import torch
import torch.nn as nn

# Import machine learning files
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from .signalEncoderModules import signalEncoderModules


class channelPositionalEncoding(signalEncoderModules):
    def __init__(self, sequenceBounds=(90, 300)):
        super(channelPositionalEncoding, self).__init__()
        # General parameters.
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence length.

        # Positional encoding parameters.
        self.nFreqModes = self.sequenceBounds[1] // 2 + 1  # Number of Fourier modes (frequencies) to use.
        self.numConvolutionalLayers = 2  # The number of convolutional layers to learn the encoding.
        self.numStampEncodingLayers = 2  # The number of layers to learn the encoding.
        self.numEncodingStamps = 10  # The number of binary bits in the encoding (010 = 2 signals; 3 encodings).

        # A list of parameters to encode each signal.
        self.encodingStamp = nn.ParameterList()  # A list of learnable parameters for learnable signal positions.
        self.decodingStamp = nn.ParameterList()  # A list of learnable parameters for learnable signal positions.
        # Initialize the encoding parameters.
        init_std = (2 / sequenceBounds[1]) ** 0.5

        # For each encoding bit.
        for stampInd in range(self.numEncodingStamps):
            # Assign a learnable parameter to the signal.
            self.encodingStamp.append(torch.nn.Parameter(torch.randn(self.nFreqModes) * init_std))
            self.decodingStamp.append(torch.nn.Parameter(torch.randn(self.nFreqModes) * init_std))

        # Initialize the encoding parameters.
        self.learnStampEncodingFNN = nn.ParameterList()
        self.unlearnStampEncodingFNN = nn.ParameterList()

        # For each encoding matrix.
        for stampInd in range(self.numStampEncodingLayers):
            # Learn the encoding stamp for each signal.
            self.learnStampEncodingFNN.append(self.neuralWeightParameters(inChannel=2, outChannel=1, finalDimension=self.nFreqModes))
            self.unlearnStampEncodingFNN.append(self.neuralWeightParameters(inChannel=2, outChannel=1, finalDimension=self.nFreqModes))
        # Learn how to integrate the learned fourier information into the signal.
        self.learnedUpdateFourierStateWeights = torch.nn.Parameter(torch.randn(self.numStampEncodingLayers) * (2 / self.numStampEncodingLayers) ** 0.5)
        self.unlearnedUpdateFourierStateWeights = torch.nn.Parameter(torch.randn(self.numStampEncodingLayers) * (2 / self.numStampEncodingLayers) ** 0.5)

        # Initialize the encoding parameters.
        self.learnStampEncodingCNN = nn.ModuleList()
        self.unlearnStampEncodingCNN = nn.ModuleList()

        # For each encoding matrix.
        for stampInd in range(self.numConvolutionalLayers):
            # Learn the encoding stamp for each signal.
            self.learnStampEncodingCNN.append(self.learnEncodingStampCNN())
            self.unlearnStampEncodingCNN.append(self.learnEncodingStampCNN())

        # Initialize helper classes.
        self.dataInterface = emotionDataInterface

    # ---------------------------------------------------------------------- #
    # -------------------- Learned Positional Encoding --------------------- #

    def addPositionalEncoding(self, inputData):
        return self.positionalEncoding(inputData, self.encodingStamp, self.learnStampEncodingFNN, self.learnStampEncodingCNN, self.learnedUpdateFourierStateWeights)

    def removePositionalEncoding(self, inputData):
        return self.positionalEncoding(inputData, self.decodingStamp, self.unlearnStampEncodingFNN, self.unlearnStampEncodingCNN, self.unlearnedUpdateFourierStateWeights)

    def positionalEncoding(self, inputData, encodingStamp, learnStampEncodingFNN, learnStampEncodingCNN, learnedUpdateFourierStateWeights):
        # Apply a small network to learn the encoding.
        positionEncodedData = self.encodingInterface(inputData, learnStampEncodingCNN[0], useCheckpoint=False) + inputData

        # Initialize and learn an encoded stamp for each signal index.
        finalStamp = self.compileStampEncoding(inputData, encodingStamp)
        positionEncodedData = self.addStampEncoding(positionEncodedData, finalStamp, learnStampEncodingFNN, learnedUpdateFourierStateWeights)

        # Apply a small network to learn the encoding.
        positionEncodedData = self.encodingInterface(positionEncodedData, learnStampEncodingCNN[1], useCheckpoint=False) + inputData

        return positionEncodedData

    def compileStampEncoding(self, inputData, encodingStamp):
        # Set up the variables for signal encoding.
        batchSize, numSignals, signalDimension = inputData.size()
        finalStamp = torch.zeros((batchSize, numSignals, self.nFreqModes), device=inputData.device)

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
            # encodingVector dim: numSignals, nFreqModes

            # Add the stamp encoding to all the signals in all the batches.
            finalStamp = finalStamp + (stampInd % 2 == 0) * encodingVector.unsqueeze(0)
            # finalStamp dim: batchSize, numSignals, nFreqModes. Note, the unused signals are essentially zero-padded.

        return finalStamp

    def addStampEncoding(self, inputData, finalStamp, learnStampEncodingFNN, learnedUpdateFourierStateWeights):
        # Extract the input data dimensions.
        batchSize, numInputSignals, sequenceLength = inputData.size()

        # Pad the data to the maximum sequence length.
        inputData = torch.nn.functional.pad(inputData, (self.sequenceBounds[1] - sequenceLength, 0), mode='constant', value=0)
        # inputData dimension: batchSize, numLiftedChannels, maxSequenceLength

        # Perform wavelet decomposition.
        lowFrequency, bandFrequencies = self.dwt(inputData)  # Note: each channel is treated independently.
        # bandFrequencies[decompositionLayer] dimension: batchSize, numLiftedChannels, bandFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numLiftedChannels, lowFrequencyShape

        # Create a new tensor with information about the initial sequence.
        sequenceInformation = torch.ones((batchSize, numInputSignals, 1), device=inputData.device) * sequenceLength / self.sequenceBounds[1]
        # Inject information about the sequence length.
        for bandFrequencyInd in range(len(bandFrequencies)):
            bandFrequencies[bandFrequencyInd] = torch.cat(tensors=(bandFrequencies[bandFrequencyInd], sequenceInformation.clone()), dim=-1)
        lowFrequency = torch.cat(tensors=(lowFrequency, sequenceInformation.clone()), dim=-1)



        # Reshape the fourier space and add the stamp encoding.
        fourierData = fourierData.view(batchSize * numInputSignals, 1, self.nFreqModes)
        fourierTransformData = finalStamp.view(batchSize * numInputSignals, 1, self.nFreqModes)
        # fourierData and finalStamp dimension: batchSize*numSignals, 1, nFreqModes

        for layerInd in range(self.numStampEncodingLayers):
            # Inject information about the sequence.
            fourierTransformData = torch.cat(tensors=(fourierTransformData, fourierData), dim=1)
            # fourierTransformData dimension: batchSize*numSignals, 2, nFreqModes

            # Multiply relevant Fourier modes (Sampling low-frequency spectrum).
            fourierTransformData = torch.einsum('oin,bin->bon', learnStampEncodingFNN[layerInd], fourierTransformData)
            # fourierTransformData dimension: batchSize*numSignals, 1, nFreqModes

            # Add the stamp encoding to the Fourier space.
            learningRate = learnedUpdateFourierStateWeights[layerInd]
            fourierTransformData = fourierData + learningRate*fourierTransformData

        # Reshape the fourier space.
        fourierTransformData = fourierTransformData.view(batchSize, numSignals, self.nFreqModes)
        # fourierTransformData dimension: batchSize, numSignals, nFreqModes

        # Return to physical space
        outputData = torch.fft.irfft(fourierTransformData, n=self.sequenceBounds[1], dim=-1, norm='ortho')[:, :, 0:sequenceLength]
        # outputData dimension: batchSize, numSignals, signalDimension

        return outputData







    def waveletNeuralOperator(self, inputData):
        # Extract the input data dimensions.
        batchSize, numInputSignals, sequenceLength = inputData.size()

        # Pad the data to the maximum sequence length.
        inputData = torch.nn.functional.pad(inputData, (self.sequenceBounds[1] - sequenceLength, 0), mode='constant', value=0)
        # inputData dimension: batchSize, numLiftedChannels, maxSequenceLength

        # Perform wavelet decomposition.
        lowFrequency, bandFrequencies = self.dwt(inputData)  # Note: each channel is treated independently.
        # bandFrequencies[decompositionLayer] dimension: batchSize, numLiftedChannels, bandFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numLiftedChannels, lowFrequencyShape

        # Create a new tensor with information about the initial sequence.
        sequenceInformation = torch.ones((batchSize, numInputSignals, 1), device=inputData.device) * sequenceLength / self.sequenceBounds[1]
        # Inject information about the sequence length.
        for bandFrequencyInd in range(len(bandFrequencies)):
            bandFrequencies[bandFrequencyInd] = torch.cat(tensors=(bandFrequencies[bandFrequencyInd], sequenceInformation.clone()), dim=-1)
        lowFrequency = torch.cat(tensors=(lowFrequency, sequenceInformation.clone()), dim=-1)

        # Learn a new set of wavelet coefficients to transform the data.
        for bandFrequencyInd in range(len(bandFrequencies)):
            bandFrequencies[bandFrequencyInd] = torch.einsum('oin,bin->bon', self.bandFrequenciesWeights[bandFrequencyInd], bandFrequencies[bandFrequencyInd])[:, :, 0:-1]
        lowFrequency = torch.einsum('oin,bin->bon', self.lowFrequencyWeights, lowFrequency)[:, :, 0:-1]
        # 'oin,bin->bon' = weights.size(), frequencies.size() -> finalFrequencies.size()
        # b = batchSize, i = numLiftedChannels, o = numOutputSignals, n = nFreqModes

        # Perform wavelet reconstruction.
        reconstructedData = self.idwt((lowFrequency, bandFrequencies))
        # reconstructedData dimension: batchSize, numSignals, sequenceLength

        # Remove the padding.
        reconstructedData = reconstructedData[:, :, -sequenceLength:]

        # Add the bias terms.
        reconstructedData = reconstructedData + self.operatorBiases
        # outputData dimension: batchSize, numOutputSignals, signalDimension

        return reconstructedData














