# PyTorch
import math
import torch
from torch import nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# Import machine learning files
from ..signalEncoderModules import signalEncoderModules

# Notes:
# - The wavelet neural operator layer is a custom module that applies a wavelet decomposition and reconstruction to the input data.
# - The wavelet neural operator layer is used to learn the encoding of the input data.
# Wavelet options:
#   Biorthogonal Wavelets ('bior'):
#       bior1.1, bior1.3, bior1.5
#       bior2.2, bior2.4, bior2.6, bior2.8
#       bior3.1, bior3.3, bior3.5, bior3.7, bior3.9
#       bior4.4
#       bior5.5
#       bior6.8
#   Complex Gaussian Wavelets ('cgau'):
#       cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8
#       cmor
#   Coiflet Wavelets ('coif'):
#       coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8, coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17
#   Daubechies Wavelets ('db'):
#       db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23, db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35, db36, db37, db38
#   Miscellaneous Wavelets and Other Families:
#       dmey, fbsp
#       Gaussian Wavelets: gaus1, gaus2, gaus3, gaus4, gaus5, gaus6, gaus7, gaus8
#       haar, mexh, morl, shan
#       Reverse Biorthogonal Wavelets: rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6, rbio2.8, rbio3.1, rbio3.3, rbio3.5, rbio3.7, rbio3.9, rbio4.4, rbio5.5, rbio6.8
#       Symlet Wavelets: sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20


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
        self.wavelet = wavelet  # The wavelet to use for the decomposition. Options: 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'
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
