# Import machine learning files
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from .signalEncoderModules import signalEncoderModules


class denoiser(signalEncoderModules):

    def __init__(self, waveletType, sequenceBounds):
        super(denoiser, self).__init__()
        # General parameters.
        self.sequenceBounds = sequenceBounds

        # Neural operator parameters.
        self.activationMethod = self.getActivationMethod_denoiser()
        self.numDecompositions = 1  # Number of decompositions for the waveletType transform.
        self.waveletType = waveletType  # wavelet type for the waveletType transform: bior, db3, dmey
        self.mode = 'zero'  # Mode for the waveletType transform.

        # Create the spectral convolution layers.
        self.denoiserNeuralOperator = waveletNeuralOperatorLayer(numInputSignals=1, numOutputSignals=1, sequenceBounds=sequenceBounds, numDecompositions=self.numDecompositions, waveletType=self.waveletType, mode=self.mode, addBiasTerm=False,
                                                                 activationMethod=self.activationMethod, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', useCNN=False, independentChannels=True, skipConnectionProtocol='identity')

        # Allow the final signals to denoise at the end.
        self.gausKernel_forPosPreds = self.smoothingKernel(kernelSize=5)
        self.gausKernel_forSigEnc = self.smoothingKernel(kernelSize=5)

    def applySmoothing_forPosPreds(self, inputData):
        return self.applySmoothing(inputData, self.gausKernel_forPosPreds)

    def applySmoothing_forSigEnc(self, inputData):
        return self.applySmoothing(inputData, self.gausKernel_forSigEnc)

    def applyDenoiser(self, inputData):
        # Apply the neural operator and the skip connection.
        return self.denoiserNeuralOperator(inputData, lowFrequencyTerms=None, highFrequencyTerms=None)
