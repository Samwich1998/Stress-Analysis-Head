# PyTorch
import torch
from torch import nn

# Import files for machine learning
from ..modelHelpers.convolutionalHelpers import convolutionalHelpers, ResNet


class signalEncoderModules(convolutionalHelpers):

    def __init__(self):
        super(signalEncoderModules, self).__init__()

    # ------------------- Positional Encoding Architectures ------------------- #

    def learnEncodingStampCNN(self):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[4, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[4, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
        )

    def positionalEncodingStamp(self, stampLength=1):
        # Initialize the weights with a uniform distribution.
        parameter = nn.Parameter(torch.randn(stampLength))
        parameter = self.weightInitialization.heNormalInit(parameter, stampLength)

        return parameter

    def learnEncodingStampFNN(self, numFeatures=1):
        return nn.Sequential(
            self.weightInitialization.initialize_weights(nn.Linear(numFeatures, numFeatures), activationMethod='selu', layerType='fc'),
            nn.SELU(),

            self.weightInitialization.initialize_weights(nn.Linear(numFeatures, numFeatures), activationMethod='selu', layerType='fc'),
            nn.SELU(),
        )

    # ------------------- Signal Encoding Architectures ------------------- #

    def liftingOperator(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            # Convolution architecture: lifting operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
        )

    @staticmethod
    def neuralWeightParameters(inChannel=1, outChannel=2, secondDimension=46):
        # Initialize the weights with a normal distribution.
        parameter = torch.ones((outChannel, inChannel, secondDimension)) / inChannel
        minChannel = min(inChannel, outChannel)

        # For each reference dimension.
        for secondDimInd in range(secondDimension):
            # Case 1: Use a slice of the identity matrix
            parameter[:, :, secondDimInd][0:minChannel, 0:minChannel] = torch.eye(minChannel) - parameter[:, :, secondDimInd][0:minChannel, 0:minChannel]

        return nn.Parameter(parameter)

    @staticmethod
    def neuralCombinationWeightParameters(inChannel=1, initialFrequencyDim=2, finalFrequencyDim=1):
        # Initialize the weights with a normal distribution.
        parameter = torch.ones((inChannel, initialFrequencyDim, finalFrequencyDim)) / initialFrequencyDim
        minFrequencyDim = min(initialFrequencyDim, finalFrequencyDim)

        # For each reference dimension.
        for firstDimIndex in range(inChannel):
            # Case 1: Use a slice of the identity matrix
            parameter[firstDimIndex, :, :][0:minFrequencyDim, 0:minFrequencyDim] = torch.eye(minFrequencyDim) - parameter[firstDimIndex, :, :][0:minFrequencyDim, 0:minFrequencyDim]

        return nn.Parameter(parameter)

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        return nn.Parameter(torch.zeros((1, numChannels, 1)))

    def linearSkipConnectionEncoding(self, inChannel=2, outChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
        )

    def skipConnectionEncoding(self, inChannel=2, outChannel=1):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            self.convolutionalFiltersBlocks(numBlocks=2, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
        )

    def signalPostProcessing(self, inChannel=2):
        return nn.Sequential(
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),

            # ResNet(module=nn.Sequential(
            #     # Convolution architecture: feature engineering.
            #     self.convolutionalFiltersBlocks(numBlocks=3, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            # ), numCycles=1),
        )

    def projectionOperator(self, inChannel=2, outChannel=1):
        return nn.Sequential(
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[outChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[outChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering.
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[outChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),
        )

    # ------------------- Final Statistics Architectures ------------------- #

    def varianceTransformation(self, inChannel=1):
        assert inChannel == 1, "The input channel must be 1."

        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[4, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[4, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),
        )

    # ----------------------- Denoiser Architectures ----------------------- #

    def denoiserModel(self, inChannel=1):
        assert inChannel == 1, "The input channel must be 1."

        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 8], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[8, 8], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[8, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),
        )
