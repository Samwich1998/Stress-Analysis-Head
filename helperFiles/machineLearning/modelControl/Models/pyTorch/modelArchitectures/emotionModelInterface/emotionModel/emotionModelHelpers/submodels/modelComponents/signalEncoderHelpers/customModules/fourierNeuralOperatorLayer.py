import torch
from torch import nn

# Import machine learning files
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionModelHelpers.generalMethods.complexHelperMethods import complexHelperMethods
from ..signalEncoderModules import signalEncoderModules


class fourierNeuralOperatorLayer(signalEncoderModules):

    def __init__(self, numInputChannels, numOutputChannels, maxSequenceLength, maxFreqNodes=None, applyFourierConvolution=True):
        super(fourierNeuralOperatorLayer, self).__init__()
        # Fourier neural operator parameters.
        self.applyFourierConvolution = applyFourierConvolution  # Whether to apply an extra convolution and FNN in the fourier space.
        self.maxSequenceLength = maxSequenceLength  # The maximum signal length.
        self.numOutputChannels = numOutputChannels  # Number of channels we want in the end.
        self.numInputChannels = numInputChannels  # Number of channels in the input data.
        self.maxFreqNodes = maxFreqNodes  # Maximum number of Fourier nodes to use.

        # Adjust the number of Fourier modes.
        self.nFreqModes = self.maxSequenceLength // 2 + 1  # Number of Fourier modes (frequencies) to use.
        if maxFreqNodes is not None: self.nFreqModes = min(self.nFreqModes, maxFreqNodes)

        # Initialize Fourier neural operator parameters.
        self.fourierWeights = self.fourierWeightParameters(inChannel=numInputChannels, outChannel=numOutputChannels, nFreqModes=self.nFreqModes + 1)
        self.skipConnectionModel = self.skipConnectionEncoding(inChannel=numInputChannels, outChannel=numOutputChannels)
        self.fourierBiases = self.fourierBiasParameters(numChannels=numOutputChannels)

        if applyFourierConvolution:
            # Initialize the convolutional parameters.
            self.convolveFourierSpace = self.fourierConvolution(inChannel=numOutputChannels)

        # Initialize activation method.
        self.activationFunction = nn.SELU()  # Activation function for the Fourier neural operator.

        # Import helper classes.
        self.complexHelperMethods = complexHelperMethods

    def forward(self, inputData):
        # Apply the Fourier neural operator and the skip connection.
        fourierOperatorOutput = self.fourierNeuralOperator(inputData)
        fourierLayerData = fourierOperatorOutput + self.skipConnectionModel(inputData)

        # Apply the activation function.
        fourierLayerData = self.activationFunction(fourierLayerData)
        # fourierLayerData dimension: batchSize, numOutputChannels, signalDimension

        return fourierLayerData

    def fourierNeuralOperator(self, inputData):
        # Extract the input data dimensions.
        batchSize, numSignals, sequenceLength = inputData.size()

        # Project the data into the Fourier domain.
        fourierData = torch.fft.rfft(inputData, n=self.maxSequenceLength, dim=-1, norm='ortho')
        fourierData = fourierData[:, :, 0:self.nFreqModes]  # Extract the relevant Fourier modes.
        # fourierData dimension: batchSize, numInputChannels, nFreqModes

        # Inject information about the sequence length.
        sequenceInformation = torch.ones((batchSize, self.numInputChannels, 1), dtype=torch.cfloat, device=inputData.device) * sequenceLength / self.maxSequenceLength
        fourierData = torch.cat(tensors=(fourierData, sequenceInformation), dim=-1)
        # fourierData dimension: batchSize, numInputChannels, nFreqModes + 1

        # Multiply relevant Fourier modes (Sampling low-frequency spectrum).
        fourierTransformData = torch.einsum('oin,bin->bon', self.fourierWeights, fourierData)[:, :, 0:self.nFreqModes]  # Self-attention to all signals
        fourierTransformData = self.complexHelperMethods.applyComplexTransformation(fourierTransformData, self.activationFunction)  # Apply an activation function.
        # 'oin,bin->bon' = fourierWeights.size(), fourierData.size() -> fourierTransformData.size()
        # b = batchSize, i = numInputChannels, o = numInputChannels, n = nFreqModes
        # fourierTransformData dimension: batchSize, numOutputChannels, nFreqModes

        if self.applyFourierConvolution:
            # Convolve the fourier space to ensure spatial significance of the frequency bins.
            fourierTransformData = self.complexHelperMethods.applyComplexTransformation(fourierTransformData, self.convolveFourierSpace)  # Treat the fourier space as a time series wave.
            # fourierTransformData dimension: batchSize, numOutputChannels, nFreqModes

        # Return to physical space
        outputData = torch.fft.irfft(fourierTransformData, n=self.maxSequenceLength, dim=-1, norm='ortho')[:, :, 0:sequenceLength]
        # outputData dimension: batchSize, numOutputChannels, signalDimension

        # Add the bias terms.
        outputData = outputData + self.fourierBiases
        # outputData dimension: batchSize, numOutputChannels, signalDimension

        return outputData
