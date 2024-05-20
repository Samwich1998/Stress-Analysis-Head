import torch.nn.functional as F

# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class denoiser(signalEncoderModules):

    def __init__(self, debuggingResults=False):
        super(denoiser, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool

        # Allow the final signals to denoise at the end.
        self.gausKernel_forPosEnc = self.averageDenoiserModel(inChannel=1)
        self.gausKernel_forVar = self.averageDenoiserModel(inChannel=1)
        self.denoiseSignals = self.denoiserModel(inChannel=1)

    def applySmoothing_forPosEnc(self, inputData):
        kernelWeights = self.gausKernel_forPosEnc.expand(inputData.size(1), 1, 3)  # Note: Output channels are set to 1 for sharing

        return F.conv1d(inputData, kernelWeights, bias=None, stride=1, padding=1, dilation=1, groups=inputData.size(1))

    def applySmoothing_forVar(self, inputData):
        kernelWeights = self.gausKernel_forPosEnc.expand(inputData.size(1), 1, 3)  # Note: Output channels are set to 1 for sharing

        return F.conv1d(inputData, kernelWeights, bias=None, stride=1, padding=1, dilation=1, groups=inputData.size(1))

    def applyDenoiser(self, inputData):
        return self.encodingInterface(inputData, self.denoiseSignals)
