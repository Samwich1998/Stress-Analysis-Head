# Import machine learning files
from .customModules.waveletNeuralOperatorLayer import waveletNeuralOperatorLayer
from .signalEncoderModules import signalEncoderModules


class denoiser(signalEncoderModules):

    def __init__(self, waveletType, sequenceBounds):
        super(denoiser, self).__init__()
        # General parameters.
        self.sequenceBounds = sequenceBounds

        # Create the convolution layers.
        self.denoiseSignals = self.denoiserModel()

        # Allow the final signals to denoise at the end.
        self.gausKernel_forPosPreds = self.smoothingKernel(kernelSize=3)
        self.gausKernel_forSigEnc = self.smoothingKernel(kernelSize=3)

    def applySmoothing_forPosPreds(self, inputData):
        return self.applySmoothing(inputData, self.gausKernel_forPosPreds)

    def applySmoothing_forSigEnc(self, inputData):
        return self.applySmoothing(inputData, self.gausKernel_forSigEnc)

    def applyDenoiser(self, inputData):
        return self.denoiseSignals(inputData)
