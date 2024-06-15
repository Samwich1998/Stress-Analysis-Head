# PyTorch
from torch import nn

# Import files for machine learning
from ..modelHelpers.convolutionalHelpers import convolutionalHelpers, ResNet


class featureExtractionModules(convolutionalHelpers):

    def __init__(self):
        super(featureExtractionModules, self).__init__()

    # ------------------- Common Feature Architectures ------------------- #

    def commonSignalFeatureExtraction(self, inChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[inChannel, 16], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[16, 16, 16, 16, 16], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[16, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[inChannel, 8], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[8, 8, 8, 8, 8], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[8, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[inChannel, 4], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[4, 4, 4, 4, 4], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[4, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
        )

    def commonChannelFeatureExtraction(self, inChannel=32, outChannel=8):
        assert outChannel == 8, outChannel
        assert inChannel == 32, inChannel

        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[32, 64], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[64, 64, 64, 64, 64], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[64, 32], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),

            # Channel reduction.
            self.convolutionalOneFilters(numChannels=[32, 16], kernel_size=3, dilation=1, group=1),

            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[16, 32], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[32, 32, 32, 32, 32], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[32, 16], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),

            # Channel reduction.
            self.convolutionalOneFilters(numChannels=[16, 8], kernel_size=3, dilation=1, group=1),

            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[8, 16], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[16, 16, 16, 16, 16], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[16, 8], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
        )

    # ------------------- Common Activity Architectures ------------------- #

    def commonActivityFeatureExtraction(self, inChannel=8):
        assert inChannel == 8, inChannel

        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[8, 32], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[32, 32, 32, 32, 32], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[32, 8], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[8, 16], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[16, 16, 16, 16, 16], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[16, 8], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
        )

    # -------------------- Common Emotion Architectures ------------------- #

    def commonEmotionFeatureExtraction(self, inChannel=8):
        assert inChannel == 8, inChannel

        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[8, 32], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[32, 32, 32, 32, 32], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[32, 8], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 8 channels maximum
                self.convolutionalOneFilters(numChannels=[8, 16], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[16, 16, 16, 16, 16], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[16, 8], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
        )
