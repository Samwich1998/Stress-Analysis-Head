import torch.nn.functional as F
from torch import nn
import torch

# Import files for machine learning
from ....optimizerMethods.activationFunctions import switchActivation, boundedExp
from ..modelHelpers.convolutionalHelpers import convolutionalHelpers, ResNet


class signalEncoderModules(convolutionalHelpers):

    def __init__(self):
        super(signalEncoderModules, self).__init__()

    # ------------------- Positional Encoding Architectures ------------------- #

    def liftingOperator_forPosEnc(self, outChannels=4):
        return nn.Sequential(
            # Convolution architecture: lifting operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, outChannels], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
        )

    def projectionOperator_forPosEnc(self, inChannels=1):
        return nn.Sequential(
            # Convolution architecture: lifting operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannels, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
        )

    def learnEncodingStampCNN(self):
        return nn.Sequential(
            # Convolution architecture: lifting operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
            self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[4, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[4, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
        )

    def positionalEncodingStamp(self, stampLength=1):
        # Initialize the weights with a uniform distribution.
        parameter = nn.Parameter(torch.randn(stampLength))
        parameter = self.weightInitialization.heNormalInit(parameter, fan_in=stampLength)

        return parameter

    def learnEncodingStampFNN(self, numFeatures=1):
        return nn.Sequential(
            self.weightInitialization.initialize_weights(nn.Linear(numFeatures, numFeatures), activationMethod='boundedExp', layerType='fc'),
            boundedExp(),
        )

    # ------------------- Signal Encoding Architectures ------------------- #

    def liftingOperator(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: residual connection, feature engineering
            self.convolutionalFilters_resNetBlocks(numResNets=4, numBlocks=4, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),

            # Convolution architecture: lifting operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
        )

    def neuralWeightParameters(self, inChannel=1, outChannel=2, secondDimension=46):
        # Initialize the weights with a normal distribution.
        parameter = nn.Parameter(torch.randn((outChannel, inChannel, secondDimension)))
        parameter = self.weightInitialization.xavierNormalInit(parameter, fan_in=inChannel*secondDimension, fan_out=outChannel*secondDimension)

        return parameter

    def neuralCombinationWeightParameters(self, inChannel=1, initialFrequencyDim=2, finalFrequencyDim=1):
        # Initialize the weights with a normal distribution.
        parameter = nn.Parameter(torch.randn((finalFrequencyDim, initialFrequencyDim, inChannel)))
        parameter = self.weightInitialization.xavierNormalInit(parameter, fan_in=inChannel*initialFrequencyDim, fan_out=inChannel*finalFrequencyDim)

        return parameter

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        parameter = nn.Parameter(torch.zeros((1, numChannels, 1)))

        return parameter

    @staticmethod
    def neuralOperatorActivation(useSwitchActivation):
        activationFunction = boundedExp()

        if useSwitchActivation:
            activationFunction = switchActivation(activationFunction, switchState=True)

        return activationFunction

    def skipConnectionEncoding(self, inChannel=2, outChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
        )

    def signalPostProcessing(self, inChannel=2):
        return nn.Sequential(
            self.convolutionalFilters_resNetBlocks(numResNets=1, numBlocks=4, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
        )

    def projectionOperator(self, inChannel=2, outChannel=1):
        return nn.Sequential(
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),

            # Convolution architecture: residual connection, feature engineering
            self.convolutionalFilters_resNetBlocks(numResNets=4, numBlocks=4, numChannels=[outChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
        )

    # ------------------- Final Statistics Architectures ------------------- #

    def varianceTransformation(self, inChannel=1):
        assert inChannel == 1, "The input channel must be 1."

        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[4, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[4, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
            ), numCycles=1),
        )

    # ----------------------- Denoiser Architectures ----------------------- #

    @staticmethod
    def smoothingKernel(kernelSize=3):
        # Initialize kernel weights.
        averageWeights = torch.ones([kernelSize], dtype=torch.float32) / kernelSize  # Uniform weights/average.

        # Set the parameter weights
        averageKernel = nn.Parameter(
            averageWeights.view(1, 1, kernelSize),
            requires_grad=False,  # Do not learn/change these weights.
        )

        return averageKernel

    @staticmethod
    def applySmoothing(inputData, kernelWeights):
        # Specify the inputs.
        kernelSize = kernelWeights.size(-1)
        numSignals = inputData.size(1)

        # Expand the kernel weights to match the channels.
        kernelWeights = kernelWeights.expand(numSignals, 1, kernelSize)  # Note: Output channels are set to 1 for sharing

        return F.conv1d(inputData, kernelWeights, bias=None, stride=1, padding=1 * (kernelSize - 1) // 2, dilation=1, groups=numSignals)

    def denoiserModel(self, inChannel=1):
        assert inChannel == 1, "The input channel must be 1."

        return nn.Sequential(
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[4, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[4, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp', numLayers=None, useSwitchActivation=True),
            ), numCycles=1),
        )
