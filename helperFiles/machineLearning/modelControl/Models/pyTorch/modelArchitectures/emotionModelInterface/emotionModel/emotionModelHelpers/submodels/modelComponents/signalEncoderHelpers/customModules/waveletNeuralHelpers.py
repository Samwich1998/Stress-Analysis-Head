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


class waveletNeuralHelpers(signalEncoderModules):

    def __init__(self, numInputSignals, numOutputSignals, sequenceBounds, numDecompositions=2, wavelet='db3', mode='zero', addBiasTerm=False, activationMethod="none",
                 encodeLowFrequencyProtocol=0, encodeHighFrequencyProtocol=0, independentChannels=False, skipConnectionProtocol='CNN'):
        super(waveletNeuralHelpers, self).__init__()
        # Fourier neural operator parameters.
        self.skipConnectionProtocol = skipConnectionProtocol  # The skip connection protocol to use.
        self.independentChannels = independentChannels  # Whether to treat each channel independently.
        self.numDecompositions = numDecompositions  # Maximum number of decompositions to apply.
        self.numOutputSignals = numOutputSignals  # Number of output signals.
        self.numInputSignals = numInputSignals  # Number of input signals.
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence length.
        self.addBiasTerm = addBiasTerm  # Whether to add bias terms to the output.
        self.wavelet = wavelet  # The wavelet to use for the decomposition. Options: 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'
        self.mode = mode  # The padding mode to use for the decomposition. Options: 'zero', 'symmetric', 'reflect' or 'periodization'.

        # Assert that the protocol is valid.
        assert encodeHighFrequencyProtocol in ['highFreq', 'allFreqs', 'none'], "The high-frequency encoding protocol must be 'highFreq', 'allFreqs', 'none'."
        assert encodeLowFrequencyProtocol in ['lowFreq', 'allFreqs', 'none'], "The low-frequency encoding protocol must be 'lowFreq', 'allFreqs', 'none'."
        # Decide on the frequency encoding protocol.
        self.encodeHighFrequencies = encodeHighFrequencyProtocol in ['highFreq', 'allFreqs']  # Whether to encode the high frequencies.
        self.encodeLowFrequency = encodeLowFrequencyProtocol in ['lowFreq', 'allFreqs']  # Whether to encode the low-frequency signal.
        self.encodeLowFrequencyFull = encodeHighFrequencyProtocol == 'allFreqs'  # Whether to encode the high frequencies into the low-frequency signal.
        self.encodeHighFrequencyFull = encodeLowFrequencyProtocol == 'allFreqs'  # Whether to encode the low-frequency signal into the high-frequency signal.

        # Verify that the number of decomposition layers is appropriate.
        maximumNumDecompositions = math.floor(math.log(sequenceBounds[0]) / math.log(2))  # The sequence length can be up to 2**numDecompositions.
        assert self.numDecompositions < maximumNumDecompositions, f'The number of decompositions must be less than {maximumNumDecompositions}.'

        # Assert the validity of the parameters under independent channels.
        if self.independentChannels:
            assert self.skipConnectionProtocol in ["none", "identity", "independentCNN"], "You cannot have a skip connection dependant on channel info if channels are independent!"
            assert self.numOutputSignals == 1, "The number of output channel is irrelevant. Please use 1."
            assert self.numInputSignals == 1, "The number of input channel is irrelevant. Please use 1."

        # Initialize the wavelet decomposition and reconstruction layers.
        self.dwt = DWT1DForward(J=self.numDecompositions, wave=self.wavelet, mode=self.mode)
        self.idwt = DWT1DInverse(wave=self.wavelet, mode=self.mode)

        # Get the expected output shapes (hard to calculate by hand).
        lowFrequency, highFrequencies = self.dwt(torch.randn(1, 1, sequenceBounds[1]))
        self.highFrequenciesShapes = [highFrequency.size(-1) for highFrequency in highFrequencies]  # Optimally: maxSequenceLength / decompositionLayer**2
        self.lowFrequencyShape = lowFrequency.size(-1)  # Optimally: maxSequenceLength / numDecompositions**2

        # Initialize wavelet neural operator parameters.
        self.highFrequenciesWeights, self.fullHighFrequencyWeights = self.getHighFrequencyWeights()  # Learnable parameters for the high-frequency signal.
        self.lowFrequencyWeights, self.fullLowFrequencyWeights = self.getLowFrequencyWeights()  # Learnable parameters for the low-frequency signal.
        self.skipConnectionModel = self.getSkipConnectionProtocol(skipConnectionProtocol)  # Skip connection model for the Fourier neural operator.
        if self.addBiasTerm: self.operatorBiases = self.neuralBiasParameters(numChannels=numOutputSignals)  # Bias terms for the Fourier neural operator.
        self.activationFunction = self.getActivationMethod(activationType=activationMethod, useSwitchActivation=True)  # Activation function for the Fourier neural operator.

    def getSkipConnectionProtocol(self, skipConnectionProtocol):
        # Decide on the skip connection protocol.
        if skipConnectionProtocol == 'none':
            skipConnectionModel = self.zero
        elif skipConnectionProtocol == 'identity':
            skipConnectionModel = nn.Identity()
        elif skipConnectionProtocol == 'singleCNN':
            skipConnectionModel = self.skipConnectionEncoding(inChannel=self.numInputSignals, outChannel=self.numOutputSignals)
        elif skipConnectionProtocol == 'independentCNN':
            skipConnectionModel = self.independentSkipConnectionEncoding(inChannel=self.numInputSignals, outChannel=self.numOutputSignals)
        else:
            raise ValueError("The skip connection protocol must be in ['none', 'identity', 'CNN'].")

        return skipConnectionModel

    def getHighFrequencyWeights(self):
        if self.encodeHighFrequencies:
            highFrequenciesWeights = nn.ParameterList()
            for highFrequenciesInd in range(len(self.highFrequenciesShapes)):
                if self.independentChannels:
                    # Initialize the high-frequency weights to learn how to change.
                    highFrequenciesWeights.append(self.neuralWeightIndependentModel(numInputFeatures=self.highFrequenciesShapes[highFrequenciesInd], numOutputFeatures=self.highFrequenciesShapes[highFrequenciesInd]))
                else:
                    # Initialize the high-frequency weights to learn how to change the channels.
                    highFrequenciesWeights.append(self.neuralWeightParameters(inChannel=self.numInputSignals, outChannel=self.numOutputSignals, finalFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd]))
        else:
            highFrequenciesWeights = None

        if self.encodeHighFrequencyFull:
            fullHighFrequencyWeights = nn.ParameterList()
            for highFrequenciesInd in range(len(self.highFrequenciesShapes)):
                if self.independentChannels:
                    # Initialize the high-frequency weights to learn how to change.
                    highFrequenciesWeights.append(self.neuralWeightIndependentModel(numInputFeatures=self.lowFrequencyShape + self.highFrequenciesShapes[highFrequenciesInd], numOutputFeatures=self.highFrequenciesShapes[highFrequenciesInd]))
                else:
                    # Initialize the frequency weights to learn how to change the channels.
                    fullHighFrequencyWeights.append(self.neuralCombinationWeightParameters(inChannel=self.numOutputSignals, initialFrequencyDim=self.lowFrequencyShape + self.highFrequenciesShapes[highFrequenciesInd], finalFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd]))
        else:
            fullHighFrequencyWeights = None

        return highFrequenciesWeights, fullHighFrequencyWeights

    def getLowFrequencyWeights(self):
        if self.encodeLowFrequency:
            if self.independentChannels:
                # Initialize the low-frequency weights to learn how to change.
                lowFrequencyWeights = self.neuralWeightIndependentModel(numInputFeatures=self.lowFrequencyShape, numOutputFeatures=self.lowFrequencyShape)
            else:
                # Initialize the low-frequency weights to learn how to change the channels.
                lowFrequencyWeights = self.neuralWeightParameters(inChannel=self.numInputSignals, outChannel=self.numOutputSignals, finalFrequencyDim=self.lowFrequencyShape)
        else:
            lowFrequencyWeights = None

        if self.encodeLowFrequencyFull:
            if self.independentChannels:
                # Initialize the low-frequency weights to learn how to change.
                fullLowFrequencyWeights = self.neuralWeightIndependentModel(numInputFeatures=self.lowFrequencyShape + sum(self.highFrequenciesShapes), numOutputFeatures=self.lowFrequencyShape)
            else:
                # Initialize the frequency weights to learn how to change the channels.
                fullLowFrequencyWeights = self.neuralCombinationWeightParameters(inChannel=self.numOutputSignals, initialFrequencyDim=self.lowFrequencyShape + sum(self.highFrequenciesShapes), finalFrequencyDim=self.lowFrequencyShape)
        else:
            fullLowFrequencyWeights = None

        return lowFrequencyWeights, fullLowFrequencyWeights

    @staticmethod
    def zero(x):
        return 0
