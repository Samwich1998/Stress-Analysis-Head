# PyTorch
import torch
from torch import nn

# Import files for machine learning
from ..modelHelpers.convolutionalHelpers import convolutionalHelpers, ResNet


class signalEncoderModules(convolutionalHelpers):

    def __init__(self):
        super(signalEncoderModules, self).__init__()

    # ------------------- Positional Encoding Architectures ------------------- #'

    def learnEncodingStampCNN(self):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 8], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            self.convolutionalFiltersBlocks(numBlocks=6, numChannels=[8, 8], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[8, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
        )

    def learnEncodingStampFNN(self, numFeatures=1):
        return nn.Sequential(
            self.modelHelpers.initialize_weights_lecun(nn.Linear(numFeatures, numFeatures)),
            nn.SELU(),
        )

    # ------------------- Signal Encoding Architectures ------------------- #

    def liftingOperator(self, inChannel=1, outChannel=2):
        assert 2*inChannel <= outChannel, "The output channel must be twice the input channel."

        return nn.Sequential(
            # Convolution architecture: lifting operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, 2*inChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[2*inChannel, 2*inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[2*inChannel, 2*inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[2*inChannel, 2*inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            # Convolution architecture: lifting operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[2*inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
        )

    @staticmethod
    def neuralWeightParameters(inChannel=1, outChannel=2, secondDimension=46):
        # Calculate standard deviation based on LeCun Normal initialization suited for SELU
        init_std = (1 / inChannel * secondDimension) ** 0.5  # Adjusted fan_in for SELU according to LeCun's recommendation

        return nn.Parameter(torch.randn((outChannel, inChannel, secondDimension)) * init_std)

    @staticmethod
    def neuralCombinationWeightParameters(inChannel=1, initialFrequencyDim=2, finalFrequencyDim=1):
        # Calculate standard deviation based on LeCun Normal initialization suited for SELU
        init_std = (1 / inChannel * initialFrequencyDim) ** 0.5  # Adjusted fan_in for SELU according to LeCun's recommendation

        return nn.Parameter(torch.randn((inChannel, initialFrequencyDim, finalFrequencyDim)) * init_std)

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        return nn.Parameter(torch.zeros((1, numChannels, 1)))

    def skipConnectionEncoding(self, inChannel=2, outChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[outChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
        )

    def signalPostProcessing(self, inChannel=2):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=3, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),
        )

    def projectionOperator(self, inChannel=2, outChannel=1):
        assert 2*outChannel <= inChannel, "The input channel must be more than twice the output channel."

        return nn.Sequential(
            # Convolution architecture: lifting operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, 2*outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[2*outChannel, 2*outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[2*outChannel, 2*outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[2*outChannel, 2*outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            # Convolution architecture: lifting operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[2*outChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
        )

    # ------------------- Final Statistics Architectures ------------------- #

    def varianceTransformation(self, inChannel=1):
        assert inChannel == 1, "The input channel must be 1."

        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 8], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[8, 8], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[8, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),
        )

    # ----------------------- Denoiser Architectures ----------------------- #

    def denoiserModel(self, inChannel=1):
        assert inChannel == 1, "The input channel must be 1."

        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 8], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=6, numChannels=[8, 8], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[8, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),
        )
