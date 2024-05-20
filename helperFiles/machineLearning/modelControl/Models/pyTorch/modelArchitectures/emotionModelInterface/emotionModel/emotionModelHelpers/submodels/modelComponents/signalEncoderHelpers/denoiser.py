import torch.nn.functional as F

# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class denoiser(signalEncoderModules):

    def __init__(self, debuggingResults=False):
        super(denoiser, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool

        # Allow the final signals to denoise at the end.
        self.gausKernel_forPosEnc = self.averageDenoiserModel(inChannel=1, kernelSize=3)
        self.gausKernel_forVar = self.averageDenoiserModel(inChannel=1, kernelSize=5)
        self.denoiseSignals = self.denoiserModel(inChannel=1)

    def applySmoothing_forPosEnc(self, inputData):
        # Specify the inputs.
        kernelSize = self.gausKernel_forPosEnc.size(-1)
        numSignals = inputData.size(1)

        kernelWeights = self.gausKernel_forPosEnc.expand(numSignals, 1, kernelSize)  # Note: Output channels are set to 1 for sharing

        return F.conv1d(inputData, kernelWeights, bias=None, stride=1, padding=1, dilation=1, groups=numSignals)

    def applySmoothing_forVar(self, inputData):
        # Specify the inputs.
        kernelSize = self.gausKernel_forVar.size(-1)
        numSignals = inputData.size(1)

        kernelWeights = self.gausKernel_forVar.expand(numSignals, 1,  kernelSize)  # Note: Output channels are set to 1 for sharing

        return F.conv1d(inputData, kernelWeights, bias=None, stride=1, padding=1, dilation=1, groups=numSignals)

    def applyDenoiser(self, inputData):
        return self.encodingInterface(inputData, self.denoiseSignals)
