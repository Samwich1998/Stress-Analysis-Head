import torch.nn.functional as F

# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class denoiser(signalEncoderModules):

    def __init__(self, debuggingResults=False):
        super(denoiser, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool

        # Allow the final signals to denoise at the end.
        self.denoiseSignals = self.denoiserModel(inChannel=1)
        self.gausKernel = self.encodingDenoiserModel(inChannel=1)

    def smoothingFunc(self, inputData):
        return F.conv1d(inputData, self.gausKernel, bias=None, stride=1, padding=1, dilation=1, groups=1)

    def applySmoothing(self, inputData):
        return self.encodingInterface(inputData, self.smoothingFunc)

    def applyDenoiser(self, inputData):
        return self.encodingInterface(inputData, self.denoiseSignals)
