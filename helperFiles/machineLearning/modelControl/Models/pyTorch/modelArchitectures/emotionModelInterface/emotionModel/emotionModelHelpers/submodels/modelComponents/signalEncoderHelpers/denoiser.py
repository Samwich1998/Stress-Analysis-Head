# Import machine learning files
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from .signalEncoderModules import signalEncoderModules


class denoiser(signalEncoderModules):

    def __init__(self, sequenceBounds, debuggingResults=False):
        super(denoiser, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.sequenceBounds = sequenceBounds

        # Neural operator parameters.
        self.activationMethod = self.getActivationMethod_denoiser()
        self.numDecompositions = 2  # Number of decompositions for the wavelet transform.
        self.wavelet = 'bior3.7'  # Wavelet type for the wavelet transform: bior3.7, db3, dmey
        self.mode = 'zero'  # Mode for the wavelet transform.

        # Create the spectral convolution layers.
        self.denoiserNeuralOperator = waveletNeuralOperatorLayer(numInputSignals=1, numOutputSignals=1, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, wavelet=self.wavelet, mode=self.mode, addBiasTerm=False,
                                                                 activationMethod=self.activationMethod, encodeLowFrequencyProtocol='none', encodeHighFrequencyProtocol='highFreq', independentChannels=True, skipConnectionProtocol='identity')

        # Allow the final signals to denoise at the end.
        self.gausKernel_forPosPreds = self.smoothingKernel(kernelSize=9)
        self.gausKernel_forSigEnc = self.smoothingKernel(kernelSize=5)
        self.gausKernel_forVar = self.smoothingKernel(kernelSize=5)

    def applySmoothing_forPosPreds(self, inputData):
        return self.applySmoothing(inputData, self.gausKernel_forSigEnc)

    def applySmoothing_forSigEnc(self, inputData):
        return self.applySmoothing(inputData, self.gausKernel_forSigEnc)

    def applySmoothing_forVar(self, inputData):
        return self.applySmoothing(inputData, self.gausKernel_forVar)

    def applyDenoiser(self, inputData):
        # Apply the neural operator and the skip connection.
        return self.denoiserNeuralOperator(inputData, lowFrequencyTerms=None, highFrequencyTerms=None)
