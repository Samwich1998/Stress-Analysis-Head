
# PyTorch
from torch import nn

# Import files for machine learning
from ..modelHelpers.convolutionalHelpers import ResNet, convolutionalHelpers


class signalInfoMappingModules(convolutionalHelpers):

    def __init__(self):
        super(signalInfoMappingModules, self).__init__()

    def learnSignalInfoModule(self, inChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 16 channels maximum
                self.convolutionalOneFilters(numChannels=[inChannel, 8], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[8, 8, 8, 8, 8], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[8, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering: Use 16 channels maximum
                self.convolutionalOneFilters(numChannels=[inChannel, 4], kernel_size=3, dilation=1, group=1),
                self.convolutionalFourFiltersBlock(numChannels=[4, 4, 4, 4, 4], kernel_sizes=3, dilations=1, groups=1),
                self.convolutionalOneFilters(numChannels=[4, inChannel], kernel_size=3, dilation=1, group=1),
            ), numCycles=1),
        )
