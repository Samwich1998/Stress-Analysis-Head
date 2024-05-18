
# PyTorch
from torch import nn

# Import files for machine learning
from ..modelHelpers.convolutionalHelpers import convolutionalHelpers, ResNet


class signalEncoderModules(convolutionalHelpers):

    def __init__(self):
        super(signalEncoderModules, self).__init__()

    # ------------------- Positional Encoding Architectures ------------------- #

    def learnModulePositionalEncoding(self, inChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[inChannel, 4], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[4, 4, 4, 4, 4], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[4, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
        )

    # ------------------- Signal Encoding Architectures ------------------- #

    def preProcessChannels(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering. Maximum 32 channels.
                self.convolutionalOneFilters(numChannels=[inChannel, 2*inChannel], kernel_size=3, dilation=1, group=1),
                self.convolutionalThreeFiltersBlock(numChannels=[2*inChannel, 2*inChannel, 2*inChannel, 2*inChannel], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[2*inChannel, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
            self.convolutionalOneFilters(numChannels=[inChannel, outChannel], kernel_size=3, dilation=1, group=1),
        )

    def encodeChannels(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: feature engineering. Keep at 32 channels minimum
            self.convolutionalOneFilters(numChannels=[inChannel, 32], kernel_size=3, dilation=1, group=1),
            self.convolutionalFourFiltersBlock(numChannels=[32, 32, 32, 32, 32], kernel_sizes=3, dilations=1, groups=1),
            self.convolutionalOneFilters(numChannels=[32, outChannel], kernel_size=3, dilation=1, group=1),
        )

    def learnModuleEncoding(self, inChannel=1):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering. Keep at 32 channels minimum
                self.convolutionalOneFilters(numChannels=[inChannel, 32], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[32, 32, 32, 32, 32], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[32, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
        )

    def postProcessChannels(self, inChannel=1):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering. Maximum 32 channels.
                self.convolutionalOneFilters(numChannels=[inChannel, 2*inChannel], kernel_size=3, dilation=1, group=1),
                self.convolutionalThreeFiltersBlock(numChannels=[2*inChannel, 2*inChannel, 2*inChannel, 2*inChannel], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[2*inChannel, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
        )

    # ------------------- Final Statistics Architectures ------------------- #

    def varianceTransformation(self, inChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                self.convolutionalOneFilters(numChannels=[inChannel, 4], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[4, 4, 4, 4, 4], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[4, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
        )

    # ----------------------- Denoiser Architectures ----------------------- #

    def denoiserModel(self, inChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                self.convolutionalOneFilters(numChannels=[inChannel, 4], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[4, 4, 4, 4, 4], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[4, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
        )
