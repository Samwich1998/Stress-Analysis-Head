
# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class denoiser(signalEncoderModules):

    def __init__(self, debuggingResults=False):
        super(denoiser, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool

        # Allow the final signals to denoise at the end.
        self.denoiseSignals = self.denoiserModel(inChannel=1)

    def applyDenoiser(self, inputData):
        return self.encodingInterface(inputData, self.denoiseSignals)
