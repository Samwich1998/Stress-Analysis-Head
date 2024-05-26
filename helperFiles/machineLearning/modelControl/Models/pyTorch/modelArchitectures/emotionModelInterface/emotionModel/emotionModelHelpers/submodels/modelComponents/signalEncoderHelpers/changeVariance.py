# ---------------------------- Imported Modules ---------------------------- #
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class changeVariance(signalEncoderModules):

    def __init__(self, waveletType, sequenceBounds, debuggingResults=False):
        super(changeVariance, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.sequenceBounds = sequenceBounds

        # Neural operator parameters.
        self.activationMethod = self.getActivationMethod_var()
        self.numDecompositions = 2     # Number of decompositions for the waveletType transform.
        self.waveletType = waveletType  # wavelet type for the waveletType transform: bior, db3, dmey
        self.mode = 'zero'             # Mode for the waveletType transform.

        # Create the spectral convolution layers.
        self.forwardNeuralOperatorVar = waveletNeuralOperatorLayer(numInputSignals=1, numOutputSignals=1, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, waveletType=self.waveletType, mode=self.mode, addBiasTerm=False, activationMethod=self.activationMethod, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', independentChannels=True, skipConnectionProtocol='identity')
        self.reverseNeuralOperatorVar = waveletNeuralOperatorLayer(numInputSignals=1, numOutputSignals=1, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, waveletType=self.waveletType, mode=self.mode, addBiasTerm=False, activationMethod=self.activationMethod, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', independentChannels=True, skipConnectionProtocol='identity')

    def adjustSignalVariance(self, inputData):
        # Apply the neural operator and the skip connection.
        return self.forwardNeuralOperatorVar(inputData, lowFrequencyTerms=None, highFrequencyTerms=None)

    def unAdjustSignalVariance(self, inputData):
        # Apply the neural operator and the skip connection.
        return self.reverseNeuralOperatorVar(inputData, lowFrequencyTerms=None, highFrequencyTerms=None)
