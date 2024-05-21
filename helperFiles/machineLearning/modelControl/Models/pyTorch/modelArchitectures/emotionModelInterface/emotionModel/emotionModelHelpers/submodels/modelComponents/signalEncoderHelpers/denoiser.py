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
        self.gausKernel_forVar5 = self.averageDenoiserModel(inChannel=1, kernelSize=5)
        self.gausKernel_forVar3 = self.averageDenoiserModel(inChannel=1, kernelSize=3)
        self.denoiseSignals = self.denoiserModel(inChannel=1)

    def applySmoothing_forPosEnc(self, inputData):
        # Specify the inputs.
        kernelSize = self.gausKernel_forPosEnc.size(-1)
        numSignals = inputData.size(1)

        kernelWeights = self.gausKernel_forPosEnc.expand(numSignals, 1, kernelSize)  # Note: Output channels are set to 1 for sharing

        return F.conv1d(inputData, kernelWeights, bias=None, stride=1, padding=1, dilation=1, groups=numSignals)

    def applySmoothing_forVar(self, inputData):
        # Specify the inputs.
        kernelSize5 = self.gausKernel_forVar5.size(-1)
        kernelSize3 = self.gausKernel_forVar3.size(-1)
        numSignals = inputData.size(1)

        # Reshape the weights for smoothing.
        kernelWeights5 = self.gausKernel_forVar.expand(numSignals, 1,  kernelSize5)  # Note: Output channels are set to 1 for sharing
        kernelWeights3 = self.gausKernel_forVar.expand(numSignals, 1,  kernelSize3)  # Note: Output channels are set to 1 for sharing

        # Smooth the data.
        inputData = F.conv1d(inputData, kernelWeights5, bias=None, stride=1, padding=2, dilation=1, groups=numSignals)
        inputData = F.conv1d(inputData, kernelWeights3, bias=None, stride=1, padding=1, dilation=1, groups=numSignals)

        return inputData

    def applyDenoiser(self, inputData):
        return self.encodingInterface_reshapeMethod(inputData, self.denoiseSignals)
