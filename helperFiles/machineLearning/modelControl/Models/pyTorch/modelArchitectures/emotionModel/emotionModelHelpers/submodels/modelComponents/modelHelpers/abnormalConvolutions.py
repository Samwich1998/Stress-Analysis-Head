# PyTorch
import torch
import torch.nn as nn

# Import helper classes.
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionModelHelpers.generalMethods.weightInitialization import weightInitialization


class abnormalConvolutions(nn.Module):

    def __init__(self):
        super(abnormalConvolutions, self).__init__()
        # Initialize helper classes.
        self.weightInitialization = weightInitialization()

    # --------------------------- Pooling Layers --------------------------- #

    @staticmethod
    def splitPooling(inputData, poolingLayers):
        """
        Applies different pooling layers to different channel splits of the input data.
        
        Parameters:
        inputData (torch.Tensor): The input data tensor of shape (batchSize, numChannels, sequenceLength).
        poolingLayers (list of nn.Module): A list of pooling layer objects to be applied to the input data.
        
        Returns:
        torch.Tensor: Pooled output with the same number of channels as inputData.
        """
        numChannels = inputData.size(1)
        if numChannels % len(poolingLayers) != 0:
            raise ValueError("Number of channels in inputData must be divisible by the number of pooling layers.")

        poolingSplit = numChannels // len(poolingLayers)
        channelSplits = torch.split(inputData, poolingSplit, dim=1)

        # Apply pooling layers in parallel and concatenate the results
        pooledOutputs = [poolingLayer(split) for split, poolingLayer in zip(channelSplits, poolingLayers)]
        pooledOutput = torch.cat(pooledOutputs, dim=1)

        return pooledOutput

    @staticmethod
    def minPooling(maxPooling):
        return lambda x: -maxPooling(-x)

    def getPoolingLayers(self, poolingTypes, compressionDim):
        # Initialize the pooling layers.
        poolingLayers = []

        if "max" in poolingTypes:
            # Add the maximum pooling layer
            poolingLayers.append(nn.AdaptiveMaxPool1d(compressionDim))

        elif "avg" in poolingTypes:
            # Add the average pooling layer
            poolingLayers.append(nn.Upsample(size=compressionDim, mode='linear', align_corners=True))

        elif "min" in poolingTypes:
            # Add the minimum pooling layer
            poolingLayers.append(self.minPooling(nn.AdaptiveMaxPool1d(compressionDim)))

        return poolingLayers


# -------------------------------------------------------------------------- #

class splitPoolingHead(nn.Module):
    def __init__(self, module, poolingLayers):
        super().__init__()
        # General helpers.
        self.poolingLayers = poolingLayers
        self.module = module

    def forward(self, inputs):
        return self.module(inputs, self.poolingLayers)
