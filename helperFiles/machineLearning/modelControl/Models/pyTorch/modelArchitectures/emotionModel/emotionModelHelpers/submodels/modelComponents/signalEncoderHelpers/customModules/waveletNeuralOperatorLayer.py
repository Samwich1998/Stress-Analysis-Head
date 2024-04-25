# PyTorch
import math
import torch
from torch import nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# Import machine learning files
from ..signalEncoderModules import signalEncoderModules


class waveletNeuralOperatorLayer(signalEncoderModules):

    def __init__(self, numInputSignals, numOutputSignals, sequenceBounds, numDecompositions=2, wavelet='db3', mode='zero', numLayers=1, encodeLowFrequency=True, encodeHighFrequencies=True):
        super(waveletNeuralOperatorLayer, self).__init__()
        # Fourier neural operator parameters.
        self.encodeHighFrequencies = encodeHighFrequencies  # Whether to encode the high frequencies.
        self.encodeLowFrequency = encodeLowFrequency  # Whether to encode the low frequency signal.
        self.numDecompositions = numDecompositions  # Maximum number of decompositions to apply.
        self.numOutputSignals = numOutputSignals  # Number of output signals.
        self.numInputSignals = numInputSignals  # Number of input signals.
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence length.
        self.numLayers = numLayers  # The number of layers to learn the encoding. I think this should be 1.
        self.wavelet = wavelet  # The wavelet to use for the decomposition.
        self.mode = mode  # The padding mode to use for the decomposition. Options: 'zero', 'symmetric', 'reflect' or 'periodization'.

        # Verify that the number of decomposition layers is appropriate.
        maximumNumDecompositions = math.floor(math.log(self.sequenceBounds[0]) / math.log(2))  # The sequence length can be up to 2**numDecompositions.
        assert self.numDecompositions < maximumNumDecompositions, f'The number of decompositions must be less than {maximumNumDecompositions}.'
        # Assert that the number of layers is appropriate.
        assert self.numLayers == 1, 'The number of layers should be 1. If you think otherwise, you should add an activation in the wavelet domain ... which could be fine.'

        # Initialize the wavelet decomposition and reconstruction layers.
        self.dwt = DWT1DForward(J=self.numDecompositions, wave=self.wavelet, mode=self.mode)
        self.idwt = DWT1DInverse(wave=self.wavelet, mode=self.mode)

        # Get the expected output shapes (hard to calculate by hand).
        lowFrequency, highFrequencies = self.dwt(torch.randn(1, 1, sequenceBounds[1]))
        self.highFrequenciesShapes = [highFrequency.size(-1) for highFrequency in highFrequencies]  # Optimally: maxSequenceLength / decompositionLayer**2
        self.lowFrequencyShape = lowFrequency.size(-1)  # Optimally: maxSequenceLength / numDecompositions**2

        # Initialize wavelet neural operator parameters.
        self.skipConnectionModel = self.skipConnectionEncoding(inChannel=numInputSignals, outChannel=numOutputSignals)
        self.operatorBiases = self.neuralBiasParameters(numChannels=numOutputSignals)

        if self.encodeLowFrequency:
            self.lowFrequencyWeights = nn.ParameterList()

            # Initialize the low frequency weights to learn how to change the channels.
            self.lowFrequencyWeights.append(self.neuralWeightParameters(inChannel=numInputSignals, outChannel=numOutputSignals, finalDimension=self.lowFrequencyShape))

            # For each subsequent layer.
            for layerInd in range(self.numLayers-1):
                # Learn a new set of wavelet coefficients to transform the data.
                self.lowFrequencyWeights.append(self.neuralWeightParameters(inChannel=numOutputSignals, outChannel=numOutputSignals, finalDimension=self.lowFrequencyShape))

        if self.encodeHighFrequencies:
            self.highFrequenciesWeights = nn.ParameterList()
            for highFrequenciesInd in range(len(self.highFrequenciesShapes)):
                self.highFrequenciesWeights.append(nn.ParameterList())

                # Initialize the high frequency weights to learn how to change the channels.
                self.highFrequenciesWeights[highFrequenciesInd].append(self.neuralWeightParameters(inChannel=numInputSignals, outChannel=numOutputSignals, finalDimension=self.highFrequenciesShapes[highFrequenciesInd]))

                # For each subsequent layer.
                for layerInd in range(self.numLayers-1):
                    # Learn a new set of wavelet coefficients to transform the data.
                    self.highFrequenciesWeights[highFrequenciesInd].append(self.neuralWeightParameters(inChannel=numOutputSignals, outChannel=numOutputSignals, finalDimension=self.highFrequenciesShapes[highFrequenciesInd]))

        # Initialize activation method.
        self.activationFunction = nn.SELU()  # Activation function for the Fourier neural operator.

    def forward(self, inputData, lowFrequencyTerms=None, highFrequencyTerms=None):
        # Apply the wavelet neural operator and the skip connection.
        neuralOperatorOutput = self.waveletNeuralOperator(inputData, lowFrequencyTerms, highFrequencyTerms)
        neuralOperatorOutput = neuralOperatorOutput + self.skipConnectionModel(inputData)
        # neuralOperatorOutput dimension: batchSize, numOutputSignals, sequenceLength

        # Apply the activation function.
        neuralOperatorOutput = self.activationFunction(neuralOperatorOutput)
        # neuralOperatorOutput dimension: batchSize, numOutputSignals, sequenceLength

        return neuralOperatorOutput

    def waveletNeuralOperator(self, inputData, lowFrequencyTerms=None, highFrequencyTerms=None):
        # Extract the input data dimensions.
        batchSize, numInputSignals, sequenceLength = inputData.size()

        # Pad the data to the maximum sequence length.
        inputData = torch.nn.functional.pad(inputData, (self.sequenceBounds[1] - sequenceLength, 0), mode='constant', value=0)
        # inputData dimension: batchSize, numInputSignals, maxSequenceLength

        # Perform wavelet decomposition.
        lowFrequency, highFrequencies = self.dwt(inputData)  # Note: each channel is treated independently here.
        # highFrequencies[decompositionLayer] dimension: batchSize, numInputSignals, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numInputSignals, lowFrequencyShape

        if self.encodeHighFrequencies:
            # Learn a new set of wavelet coefficients to transform the data.
            for highFrequencyInd in range(len(highFrequencies)):
                highFrequencies[highFrequencyInd] = self.applyEncoding(highFrequencies[highFrequencyInd], self.highFrequenciesWeights[highFrequencyInd], highFrequencyTerms)
                # frequencies dimension: batchSize, numOutputSignals, highFrequenciesShapes[decompositionLayer]

        if self.encodeLowFrequency:
            # Learn a new set of wavelet coefficients to transform the data.
            lowFrequency = self.applyEncoding(lowFrequency, self.lowFrequencyWeights, lowFrequencyTerms)
            # frequencies dimension: batchSize, numOutputSignals, lowFrequencyShape

        # Perform wavelet reconstruction.
        reconstructedData = self.idwt((lowFrequency, highFrequencies))
        # reconstructedData dimension: batchSize, numOutputSignals, sequenceLength

        # Remove the padding.
        reconstructedData = reconstructedData[:, :, -sequenceLength:]
        # reconstructedData dimension: batchSize, numOutputSignals, sequenceLength

        # Add the bias terms.
        reconstructedData = reconstructedData + self.operatorBiases
        # outputData dimension: batchSize, numOutputSignals, sequenceLength

        return reconstructedData

    def applyEncoding(self, frequencies, weights, frequencyTerms=None):
        if frequencyTerms is not None:
            # Apply the learned wavelet coefficients.
            frequencies = frequencies + frequencyTerms
            # frequencies dimension: batchSize, numInputSignals, frequencyDimension

        for layer in range(self.numLayers):
            # Learn a new set of wavelet coefficients to transform the data.
            frequencies = torch.einsum('oin,bin->bon', weights[layer], frequencies)
            # b = batchSize, i = numInputSignals, o = numOutputSignals, n = signalDimension
            # 'oin,bin->bon' = weights.size(), frequencies.size() -> frequencies.size()
            # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

        return frequencies
