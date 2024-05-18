
# General
import gc
import math

# Pytorch
import torch
from torch import nn

# Import files for machine learning
from ..modelHelpers.convolutionalHelpers import convolutionalHelpers, ResNet
from ....emotionDataInterface import emotionDataInterface


class autoencoderModules(convolutionalHelpers):
    def __init__(self, accelerator, compressionFactor, expansionFactor):
        super(autoencoderModules, self).__init__()
        # General parameters
        self.compressionFactor = compressionFactor   # The compression factor of the signals.
        self.expansionFactor = expansionFactor       # The expansion factor of the signals.
        self.accelerator = accelerator               # Hugging face model optimizations.
        
        # Initialize helper classes
        self.dataInterface = emotionDataInterface

    # ---------------------- Encoder-Specific Modules ---------------------- #

    def signalEncodingModule(self):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 16], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[16, 16], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[16, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                # Convolution architecture: feature engineering
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 8], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[8, 8], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[8, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),
        )

    def ladderModules(self):
        return nn.Sequential(
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 16], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[16, 16], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[16, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
        )

    def varianceTransformation(self):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 2], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[2, 2], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[2, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),
        )

    def denoiserModel(self):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[1, 2], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=4, numChannels=[2, 2], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
                self.convolutionalFiltersBlocks(numBlocks=1, numChannels=[2, 1], kernel_sizes=3, dilations=1, groups=1, strides=1, convType='conv1D', activationType='selu', numLayers=None),
            ), numCycles=1),
        )

    # ---------------------- Data Structure Interface ---------------------- #
  
    def simulateEncoding(self, currentSequenceLength, targetSequenceLength):
        encodingPath = [currentSequenceLength]
        
        # While we haven't converged to the target length.
        while currentSequenceLength != targetSequenceLength:
            # Simulate how the sequence will change during the next iteration.
            currentSequenceLength = self.getNextSequenceLength(currentSequenceLength, targetSequenceLength)
            
            # Store the path the sequence takes.
            encodingPath.append(currentSequenceLength)
        
        return encodingPath
                    
    def getNextSequenceLength(self, currentSequenceLength, targetSequenceLength):
        # If we are a factor of 2 away from the target length
        if currentSequenceLength*self.expansionFactor <= targetSequenceLength:
            return math.floor(currentSequenceLength*self.expansionFactor)
        
        # If we are less than halfway to the target length
        elif currentSequenceLength <= targetSequenceLength:
            return targetSequenceLength
        
        # If we are a factor of 2 away from the target length
        elif targetSequenceLength <= currentSequenceLength/self.compressionFactor:
            return math.ceil(currentSequenceLength/self.compressionFactor)
        
        # If we are less than halfway to the target length
        elif targetSequenceLength <= currentSequenceLength:
            return targetSequenceLength
        
# -------------------------------------------------------------------------- #
