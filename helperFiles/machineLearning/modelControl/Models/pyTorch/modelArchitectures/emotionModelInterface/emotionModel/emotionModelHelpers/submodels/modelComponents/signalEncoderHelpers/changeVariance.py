# ---------------------------- Imported Modules ---------------------------- #
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class changeVariance(signalEncoderModules):

    def __init__(self, sequenceBounds, debuggingResults=False):
        super(changeVariance, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.sequenceBounds = sequenceBounds

        # Neural operator parameters.
        self.activationMethod = self.getActivationMethod_var()
        self.numDecompositions = 2     # Number of decompositions for the wavelet transform.
        self.wavelet = 'bior3.7'       # Wavelet type for the wavelet transform: bior3.7, db3, dmey
        self.mode = 'zero'             # Mode for the wavelet transform.

        # Create the spectral convolution layers.
        self.forwardNeuralOperatorVar = waveletNeuralOperatorLayer(numInputSignals=self.numPosLiftedChannels, numOutputSignals=self.numPosLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, activationMethod=self.activationMethod, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', independentChannels=True, skipConnectionProtocol='none')
        self.reverseNeuralOperatorVar = waveletNeuralOperatorLayer(numInputSignals=self.numPosLiftedChannels, numOutputSignals=self.numPosLiftedChannels, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, activationMethod=self.activationMethod, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', independentChannels=True, skipConnectionProtocol='none')

    def adjustSignalVariance(self, inputData):
        # Apply the neural operator and the skip connection.
        return self.forwardNeuralOperatorVar(inputData, lowFrequencyTerms=None, highFrequencyTerms=None)

    def unAdjustSignalVariance(self, inputData):
        # Apply the neural operator and the skip connection.
        return self.reverseNeuralOperatorVar(inputData, lowFrequencyTerms=None, highFrequencyTerms=None)
