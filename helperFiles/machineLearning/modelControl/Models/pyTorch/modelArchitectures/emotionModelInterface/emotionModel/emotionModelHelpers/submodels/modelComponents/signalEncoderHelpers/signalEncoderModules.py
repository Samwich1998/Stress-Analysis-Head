import torch.nn.functional as F
from torch import nn
import torch

# Import files for machine learning
from ....optimizerMethods.activationFunctions import boundedExp
from ..modelHelpers.convolutionalHelpers import convolutionalHelpers, independentModelCNN, addModules, ResNet


class signalEncoderModules(convolutionalHelpers):

    def __init__(self):
        super(signalEncoderModules, self).__init__()

    # ------------------- Wavelet Neural Operator Architectures ------------------- #

    def neuralWeightIndependentModel(self, numInputFeatures=1, numOutputFeatures=1):
        return nn.Sequential(
            self.weightInitialization.initialize_weights(nn.Linear(numInputFeatures, numOutputFeatures), activationMethod='none', layerType='fc'),
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

    def neuralWeightCNN(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp_0_2', numLayers=None, addBias=False, useSwitchActivation=True),
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='none', numLayers=None, addBias=False, useSwitchActivation=True),
        )

    def independentNeuralWeightCNN(self, inChannel=2, outChannel=1):
        assert inChannel == outChannel, "The number of input and output signals must be equal."

        return independentModelCNN(
            useCheckpoint=False,
            module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 4], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp_0_2', numLayers=None, addBias=False, useSwitchActivation=True),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[4, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='none', numLayers=None, addBias=False, useSwitchActivation=True),
            ),
        )

    @staticmethod
    def neuralBiasParameters(numChannels=2):
        parameter = nn.Parameter(torch.zeros((1, numChannels, 1)))

        return parameter

    def skipConnectionEncoding(self, inChannel=2, outChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='none', numLayers=None, addBias=False, useSwitchActivation=True),
        )

    def independentSkipConnectionEncoding(self, inChannel=2, outChannel=1):
        assert inChannel == outChannel == 1, "The number of input and output signals must be 1."

        return independentModelCNN(
            module=self.skipConnectionEncoding(inChannel=1, outChannel=1),
            useCheckpoint=False,
        )

    # ------------------- Positional Encoding Architectures ------------------- #

    @staticmethod
    def getActivationMethod_posEncoder():
        return "none"

    def positionalEncodingStamp(self, stampLength=1):
        # Initialize the weights with a uniform distribution.
        parameter = nn.Parameter(torch.randn(stampLength))
        parameter = self.weightInitialization.heNormalInit(parameter, fan_in=stampLength)

        return parameter

    def predictedPosEncodingIndex(self, numFeatures=2, numClasses=1):
        firstModule = nn.Sequential(
            self.weightInitialization.initialize_weights(nn.Linear(numFeatures, numFeatures), activationMethod='boundedExp_0_2', layerType='fc'),
            boundedExp(topExponent=0, nonLinearityRegion=2),

            self.weightInitialization.initialize_weights(nn.Linear(numFeatures, numFeatures), activationMethod='boundedExp_0_2', layerType='fc'),
            boundedExp(topExponent=0, nonLinearityRegion=2),

            self.weightInitialization.initialize_weights(nn.Linear(numFeatures, numClasses), activationMethod='boundedExp_0_2', layerType='fc'),
            boundedExp(topExponent=0, nonLinearityRegion=2),
        )

        secondModule = nn.Sequential(
            self.weightInitialization.initialize_weights(nn.Linear(numFeatures, numClasses), activationMethod='boundedExp_0_2', layerType='fc'),
        )

        return nn.Sequential(
            addModules(
                firstModule=firstModule,
                secondModule=secondModule,
                secondModuleScale=-1,
                scalingFactor=1,
            ),
        )

    # ------------------- Signal Encoding Architectures ------------------- #

    def heuristicEncoding(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: heuristic operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='none', numLayers=None, addBias=False, useSwitchActivation=True),
        )

    def liftingOperator(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: lifting operator. Keep kernel_sizes as 1 for an interpretable encoding space and faster (?) convergence.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp_0_2', numLayers=None, addBias=False, useSwitchActivation=True),
        )

    @staticmethod
    def getActivationMethod_channelEncoder():
        return 'boundedExp_0_2'

    def signalPostProcessing(self, inChannel=2, bottleneckChannel=2):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, bottleneckChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp_0_2', numLayers=None, addBias=False, useSwitchActivation=True),
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[bottleneckChannel, inChannel], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp_0_2', numLayers=None, addBias=False, useSwitchActivation=True),
        )

    def heuristicEncodingLayer(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: heuristic operator.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='none', numLayers=None, addBias=False, useSwitchActivation=True),
        )

    def projectionOperator(self, inChannel=2, outChannel=1):
        return nn.Sequential(
            # Convolution architecture: projection operator. Keep kernel_sizes as 1 for an interpretable encoding space and faster (?) convergence.
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[inChannel, outChannel], kernel_sizes=1, dilations=1, groups=1, strides=1, convType='conv1D', activationType='boundedExp_0_2', numLayers=None, addBias=False, useSwitchActivation=True),
        )

    # ----------------------- Denoiser Architectures ----------------------- #

    @staticmethod
    def getActivationMethod_denoiser():
        return "none"

    @staticmethod
    def smoothingKernel(kernelSize=3, averageWeights=None):
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
