# General
from torch import nn

# Import machine learning files
from .waveletNeuralHelpers import waveletNeuralHelpers


class waveletNeuralOperatorWeights(waveletNeuralHelpers):

    def __init__(self, numInputSignals, numOutputSignals, sequenceBounds, numDecompositions=2, waveletType='db3', mode='zero', addBiasTerm=False, smoothingKernelSize=0, activationMethod="none",
                 encodeLowFrequencyProtocol=0, encodeHighFrequencyProtocol=0, useConvolutionFlag=True, independentChannels=False, skipConnectionProtocol='CNN'):
        super(waveletNeuralOperatorWeights, self).__init__(numInputSignals, numOutputSignals, sequenceBounds, numDecompositions, waveletType, mode, addBiasTerm, smoothingKernelSize, activationMethod,
                                                           encodeLowFrequencyProtocol, encodeHighFrequencyProtocol, useConvolutionFlag, independentChannels, skipConnectionProtocol)
        # Initialize wavelet neural operator parameters.
        if self.smoothingKernelSize: self.smoothingKernel = self.getSmoothingKernel(kernelSize=self.smoothingKernelSize)  # Smoothing kernel for the Fourier neural operator.
        if self.addBiasTerm: self.operatorBiases = self.neuralBiasParameters(numChannels=numOutputSignals)  # Bias terms for the Fourier neural operator.
        self.highFrequenciesWeights, self.fullHighFrequencyWeights = self.getHighFrequencyWeights()  # Learnable parameters for the high-frequency signal.
        self.lowFrequencyWeights, self.fullLowFrequencyWeights = self.getLowFrequencyWeights()  # Learnable parameters for the low-frequency signal.
        self.activationFunction = self.getActivationMethod(activationType=activationMethod)  # Activation function for the Fourier neural operator.
        self.skipConnectionModel = self.getSkipConnectionProtocol(skipConnectionProtocol)  # Skip connection model for the Fourier neural operator.

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
        # Initialize the high-frequency weights.
        fullHighFrequencyWeights = None
        highFrequenciesWeights = None

        if self.encodeHighFrequencies:
            highFrequenciesWeights = nn.ParameterList()
            for highFrequenciesInd in range(len(self.highFrequenciesShapes)):
                highFrequencyParam = self.zero
                if not self.removeHighFrequencies:
                    highFrequencyParam = self.getNeuralWeightParameters(inChannel=self.numInputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd],
                                                                        finalFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd], lowFreqSignal=False)
                # Store the high-frequency weights.
                highFrequenciesWeights.append(highFrequencyParam)

        if self.encodeHighFrequencyFull:
            fullHighFrequencyWeights = nn.ParameterList()
            for highFrequenciesInd in range(len(self.highFrequenciesShapes)):
                fullHighFrequencyWeights.append(self.getNeuralWeightParameters(inChannel=self.numOutputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.lowFrequencyShape + self.highFrequenciesShapes[highFrequenciesInd],
                                                                               finalFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd], lowFreqSignal=False))

        return highFrequenciesWeights, fullHighFrequencyWeights

    def getLowFrequencyWeights(self):
        # Initialize the low-frequency weights.
        fullLowFrequencyWeights = None
        lowFrequencyWeights = None

        if self.encodeLowFrequency:
            lowFrequencyWeights = self.zero
            if not self.removeLowFrequency:
                lowFrequencyWeights = self.getNeuralWeightParameters(inChannel=self.numInputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.lowFrequencyShape, finalFrequencyDim=self.lowFrequencyShape, lowFreqSignal=True)

        if self.encodeLowFrequencyFull:
            fullLowFrequencyWeights = self.getNeuralWeightParameters(inChannel=self.numOutputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.lowFrequencyShape + sum(self.highFrequenciesShapes),
                                                                     finalFrequencyDim=self.lowFrequencyShape, lowFreqSignal=True)

        return lowFrequencyWeights, fullLowFrequencyWeights

    def getNeuralWeightParameters(self, inChannel, outChannel, initialFrequencyDim, finalFrequencyDim, lowFreqSignal=False):
        if self.useConvolutionFlag:
            if self.independentChannels:
                # Initialize the frequency weights to learn how to change.
                assert inChannel == outChannel, "The number of input and output signals must be equal."

                # Initialize the low-frequency weights to learn how to change.
                return self.independentNeuralWeightCNN(inChannel=inChannel, outChannel=outChannel)

            if lowFreqSignal:
                # Initialize the low-frequency weights to learn how to change.
                return self.neuralWeightLowCNN(inChannel=inChannel, outChannel=outChannel)
            else:
                # Initialize the high-frequency weights to learn how to change.
                return self.neuralWeightHighCNN(inChannel=inChannel, outChannel=outChannel)

        else:
            if self.independentChannels:
                # Initialize the high-frequency weights to learn how to change.
                assert inChannel == outChannel, "The number of input and output signals must be equal."
                return self.neuralWeightIndependentModel(numInputFeatures=initialFrequencyDim, numOutputFeatures=finalFrequencyDim)

            if initialFrequencyDim == finalFrequencyDim:
                # Initialize the high-frequency weights to learn how to change the channels.
                return self.neuralWeightParameters(inChannel=inChannel, outChannel=outChannel, finalFrequencyDim=finalFrequencyDim)
            else:
                # Initialize the high-frequency weights to learn how to change the channels.
                assert inChannel == outChannel, "The number of input and output signals must be equal."
                return self.neuralCombinationWeightParameters(inChannel=outChannel, initialFrequencyDim=initialFrequencyDim, finalFrequencyDim=finalFrequencyDim)
