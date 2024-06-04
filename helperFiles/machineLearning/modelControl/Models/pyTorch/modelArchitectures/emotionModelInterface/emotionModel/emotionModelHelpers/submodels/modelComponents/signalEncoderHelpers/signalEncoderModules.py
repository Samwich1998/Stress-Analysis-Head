import math

import torch.nn.functional as F
from torch import nn
import torch

# Import files for machine learning
from ..modelHelpers.convolutionalHelpers import convolutionalHelpers, independentModelCNN, ResNet


class signalEncoderModules(convolutionalHelpers):

    def __init__(self):
        super(signalEncoderModules, self).__init__()

    def linearModel(self, numInputFeatures=1, numOutputFeatures=1, activationMethod='none'):
        return nn.Sequential(
            self.weightInitialization.initialize_weights(nn.Linear(numInputFeatures, numOutputFeatures), activationMethod='none', layerType='fc'),
            self.getActivationMethod(activationMethod),
        )

    # ------------------- Wavelet Neural Operator Architectures ------------------- #

    def neuralWeightIndependentModel(self, numInputFeatures=1, numOutputFeatures=1):
        return nn.Sequential(
            self.linearModel(numInputFeatures=numInputFeatures, numOutputFeatures=numOutputFeatures, activationMethod='none'),
        )

    def neuralWeightParameters(self, inChannel=1, outChannel=2, finalFrequencyDim=46):
        # Initialize the weights with a normal distribution.
        parameter = nn.Parameter(torch.randn((outChannel, inChannel, finalFrequencyDim)))
        parameter = self.weightInitialization.xavierNormalInit(parameter, fan_in=inChannel * finalFrequencyDim, fan_out=outChannel * finalFrequencyDim)

        return parameter

    def neuralCombinationWeightParameters(self, inChannel=1, initialFrequencyDim=2, finalFrequencyDim=1):
        # Initialize the weights with a normal distribution.
        parameter = nn.Parameter(torch.randn((finalFrequencyDim, initialFrequencyDim, inChannel)))
        parameter = self.weightInitialization.xavierNormalInit(parameter, fan_in=inChannel * initialFrequencyDim, fan_out=inChannel * finalFrequencyDim)

        return parameter

    def neuralWeightHighCNN(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: feature engineering. Detailed coefficients tend to look like delta spikes ... I think kernel_size of 1 is optimal.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='pointwise', activationType='none', numLayers=None, addBias=False),
        )

    def neuralWeightLowCNN(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: feature engineering. Detailed coefficients tend to look like low-frequency waves.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='pointwise', activationType='none', numLayers=None, addBias=False),
        )

    def independentNeuralWeightCNN(self, inChannel=2, outChannel=1):
        assert inChannel == outChannel, "The number of input and output signals must be equal."

        return independentModelCNN(
            module=self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='none', numLayers=None, addBias=False),
            useCheckpoint=False,
        )

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        parameter = nn.Parameter(torch.zeros((1, numChannels, 1)))

        return parameter

    def skipConnectionEncoding(self, inChannel=2, outChannel=1):
        assert inChannel != 1, "The number of input signals must be greater than 1 or use a 'independentCNN' as kernel_size is 1."

        return nn.Sequential(
            # Convolution architecture: feature engineering
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='none', numLayers=None, addBias=False),
        )

    def independentSkipConnectionEncoding(self, inChannel=2, outChannel=1):
        assert inChannel == outChannel, "The number of input and output signals must be equal."

        return independentModelCNN(
            module=self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='none', numLayers=None, addBias=False),
            useCheckpoint=False,
        )

    # ------------------- Positional Encoding Architectures ------------------- #

    @staticmethod
    def getActivationMethod_posEncoder():
        return "none"

    @staticmethod
    def positionalEncodingStamp(stampLength=1, stampInd=0, signalMinMaxScale=1):
        # Create an array of values from 0 to stampLength - 1
        x = torch.arange(stampLength, dtype=torch.float32)
        amplitude = signalMinMaxScale/4
        frequency = stampInd

        # Generate the sine wave
        sine_wave = amplitude * torch.sin(2 * math.pi * frequency * x / stampLength)

        return sine_wave

    def predictedPosEncodingIndex(self, numFeatures=2, numClasses=1):
        return nn.Sequential(
            independentModelCNN(
                useCheckpoint=False,
                module=nn.Sequential(
                    # Convolution architecture: feature engineering
                    self.convolutionalFilters_resNetBlocks(numResNets=1, numBlocks=1, numChannels=[1, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='boundedExp_0_2', numLayers=None, addBias=True),
                    self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[4, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='boundedExp_0_2', numLayers=None, addBias=False),
                    self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[4, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='boundedExp_0_2', numLayers=None, addBias=False),
                ),
            ),

            # Neural architecture: self attention.
            self.linearModel(numInputFeatures=numFeatures, numOutputFeatures=numClasses, activationMethod='boundedExp_0_2'),
            self.linearModel(numInputFeatures=numClasses, numOutputFeatures=numClasses, activationMethod='boundedExp_0_2'),
        )

    # ------------------- Signal Encoding Architectures ------------------- #

    def liftingOperator(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: lifting operator. Keep kernel_sizes as 1 for an interpretable encoding space and faster (?) convergence.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='pointwise', activationType='boundedExp_0_2', numLayers=None, addBias=False),
        )

    @staticmethod
    def getActivationMethod_channelEncoder():
        return 'boundedExp_0_2'

    def signalPostProcessing(self, inChannel=2, bottleneckChannel=2):
        return nn.Sequential(
            # Convolution architecture: feature engineering. Keep kernel_sizes as 1 for faster (?) convergence. THe purpose of kernel_sizes as 3 is to prevent gibbs phenomenon.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, bottleneckChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='boundedExp_0_2', numLayers=None, addBias=False),
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[bottleneckChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='boundedExp_0_2', numLayers=None, addBias=False),
        )

    def projectionOperator(self, inChannel=2, outChannel=1):
        return nn.Sequential(
            # Convolution architecture: projection operator. Keep kernel_sizes as 1 for an interpretable encoding space and faster (?) convergence.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='pointwise', activationType='boundedExp_0_2', numLayers=None, addBias=False),
        )

    def heuristicEncoding(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: heuristic operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='none', numLayers=None, addBias=False),
        )

    # ----------------------- Denoiser Architectures ----------------------- #

    def denoiserModel(self):
        return independentModelCNN(
            ResNet(
                module=nn.Sequential(
                    self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 4], kernel_sizes=5, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='boundedExp_0_2', numLayers=None, addBias=False),
                    self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[4, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='boundedExp_0_2', numLayers=None, addBias=False),
                    self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[4, 1], kernel_sizes=5, dilations=1, groups=1, strides=1, convType='conv1D_gausInit', activationType='boundedExp_0_2', numLayers=None, addBias=False),
                )
            ), useCheckpoint=False,
        )

    @staticmethod
    def getSmoothingKernel(kernelSize=3, averageWeights=None):
        if averageWeights is not None:
            assert len(averageWeights) == kernelSize, "The kernel size and the average weights must be the same size."
            averageWeights = torch.tensor(averageWeights, dtype=torch.float32)
        else:
            averageWeights = torch.ones([kernelSize], dtype=torch.float32)
        # Initialize kernel weights.
        averageWeights = averageWeights / averageWeights.sum()

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
