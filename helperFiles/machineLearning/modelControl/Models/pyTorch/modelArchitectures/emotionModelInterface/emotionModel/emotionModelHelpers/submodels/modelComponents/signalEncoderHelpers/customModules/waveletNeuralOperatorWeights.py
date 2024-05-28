# General
from torch import nn

# Import machine learning files
from .waveletNeuralHelpers import waveletNeuralHelpers


class waveletNeuralOperatorWeights(waveletNeuralHelpers):

    def __init__(self, numInputSignals, numOutputSignals, sequenceBounds, numDecompositions=2, waveletType='db3', mode='zero', addBiasTerm=False, activationMethod="none",
                 encodeLowFrequencyProtocol=0, encodeHighFrequencyProtocol=0, useLowFreqCNN=True, independentChannels=False, skipConnectionProtocol='CNN'):
        super(waveletNeuralOperatorWeights, self).__init__(numInputSignals, numOutputSignals, sequenceBounds, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                                                           encodeLowFrequencyProtocol, encodeHighFrequencyProtocol, useLowFreqCNN, independentChannels, skipConnectionProtocol)

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
                highFrequenciesWeights.append(self.getNeuralWeightParameters(inChannel=self.numInputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd],
                                                                             finalFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd], lowFrequencyFlag=False))

        if self.encodeHighFrequencyFull:
            fullHighFrequencyWeights = nn.ParameterList()
            for highFrequenciesInd in range(len(self.highFrequenciesShapes)):
                fullHighFrequencyWeights.append(self.getNeuralWeightParameters(inChannel=self.numOutputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.lowFrequencyShape + self.highFrequenciesShapes[highFrequenciesInd],
                                                                               finalFrequencyDim=self.highFrequenciesShapes[highFrequenciesInd], lowFrequencyFlag=False))

        return highFrequenciesWeights, fullHighFrequencyWeights

    def getLowFrequencyWeights(self):
        # Initialize the low-frequency weights.
        fullLowFrequencyWeights = None
        lowFrequencyWeights = None

        if self.encodeLowFrequency:
            lowFrequencyWeights = self.getNeuralWeightParameters(inChannel=self.numInputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.lowFrequencyShape, finalFrequencyDim=self.lowFrequencyShape, lowFrequencyFlag=True)

        if self.encodeLowFrequencyFull:
            fullLowFrequencyWeights = self.getNeuralWeightParameters(inChannel=self.numOutputSignals, outChannel=self.numOutputSignals, initialFrequencyDim=self.lowFrequencyShape + sum(self.highFrequenciesShapes),
                                                                     finalFrequencyDim=self.lowFrequencyShape, lowFrequencyFlag=True)

        return lowFrequencyWeights, fullLowFrequencyWeights

    def getNeuralWeightParameters(self, inChannel, outChannel, initialFrequencyDim, finalFrequencyDim, lowFrequencyFlag=False):
        if self.useLowFreqCNN: # and lowFrequencyFlag:
            if self.independentChannels:
                # Initialize the frequency weights to learn how to change.
                assert inChannel == outChannel, "The number of input and output signals must be equal."
                return self.independentNeuralWeightCNN(inChannel=inChannel, outChannel=outChannel)
            return self.neuralWeightCNN(inChannel=inChannel, outChannel=outChannel)

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
