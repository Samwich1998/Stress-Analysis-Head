# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class denoiser(signalEncoderModules):

    def __init__(self, debuggingResults=False):
        super(denoiser, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool

        # Allow the final signals to denoise at the end.
        self.gausKernel_forPosEnc = self.smoothingKernel(kernelSize=3)
        self.gausKernel_forSigEnc = self.smoothingKernel(kernelSize=5)
        self.gausKernel_forVar = self.smoothingKernel(kernelSize=7)
        self.denoiseSignals = self.denoiserModel(inChannel=1)

    def applySmoothing_forSigEnc(self, inputData):
        return self.applySmoothing(inputData, self.gausKernel_forSigEnc)

    def applySmoothing_forPosEnc(self, inputData):
        return self.applySmoothing(inputData, self.gausKernel_forPosEnc)

    def applySmoothing_forVar(self, inputData):
        return self.applySmoothing(inputData, self.gausKernel_forVar)

    def applyDenoiser(self, inputData):
        return self.encodingInterface_reshapeMethod(inputData, self.denoiseSignals)
