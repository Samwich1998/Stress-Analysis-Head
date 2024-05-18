# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import time

# PyTorch
import torch.nn as nn
from torchsummary import summary

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import _convolutionalHelpers

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class signalEncoderBase(_convolutionalHelpers.convolutionalHelpers):
    def __init__(self, signalDimension, numExpandedSignals, numCompressedSignals):
        super(signalEncoderBase, self).__init__()
        self.signalDimension = signalDimension
        self.numExpandedSignals = numExpandedSignals
        self.numCompressedSignals = numCompressedSignals
        
        # Create model for learning local information.
        # self.expandSignals = self.signalExpansionModule(inChannel = numCompressedSignals)
        # self.compressSignals = self.signalCompressionModule(inChannel = numExpandedSignals)
        
        # Delta learning modules to predict the residuals.
        self.simpleExpansion = self.changeChannels(numChannels = [64, 512], kernel_sizes = [3], dilations = [1], groups = [64])
        self.simpleCompression = self.changeChannels(numChannels = [512, 64], kernel_sizes = [3], dilations = [1], groups = [64])
        
    # ---------------------------------------------------------------------- #
    # ------------------- Machine Learning Architectures ------------------- #

    def signalCompressionModule(self, inChannel = 2, outChannel = 1):
        return nn.Sequential( 
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 2*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [inChannel, inChannel]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 2*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [inChannel, inChannel]),
                
                # Convolution architecture: channel expansion
                self.changeChannels(numChannels = [inChannel, 256], kernel_sizes = [3], dilations = [1], groups = [256]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [256, 2*256, 256], kernel_sizes = [3, 3], dilations = [1, 1], groups = [256, 256]),
                self.deconvolutionalFilter_resNet(numChannels = [256, 2*256, 256], kernel_sizes = [3, 3], dilations = [1, 1], groups = [256, 256]),
                
                # Convolution architecture: channel expansion
                self.changeChannels(numChannels = [256, 128], kernel_sizes = [3], dilations = [1], groups = [128]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [128, 2*128, 128], kernel_sizes = [3, 3], dilations = [1, 1], groups = [128, 128]),
                self.deconvolutionalFilter_resNet(numChannels = [128, 2*128, 128], kernel_sizes = [3, 3], dilations = [1, 1], groups = [128, 128]),
                
                # Convolution architecture: channel expansion
                self.changeChannels(numChannels = [128, 64], kernel_sizes = [3], dilations = [1], groups = [64]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [64, 2*64, 64], kernel_sizes = [3, 3], dilations = [1, 1], groups = [64, 64]),
                self.deconvolutionalFilter_resNet(numChannels = [64, 2*64, 64], kernel_sizes = [3, 3], dilations = [1, 1], groups = [64, 64]),
        )
    
    def signalExpansionModule(self, inChannel = 2, outChannel = 1):
        return nn.Sequential( 
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 2*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [inChannel, inChannel]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 2*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [inChannel, inChannel]),
                
                # Convolution architecture: channel expansion
                self.changeChannels(numChannels = [inChannel, 128], kernel_sizes = [3], dilations = [1], groups = [64]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [128, 2*128, 128], kernel_sizes = [3, 3], dilations = [1, 1], groups = [128, 128]),
                self.deconvolutionalFilter_resNet(numChannels = [128, 2*128, 128], kernel_sizes = [3, 3], dilations = [1, 1], groups = [128, 128]),
                
                # Convolution architecture: channel expansion
                self.changeChannels(numChannels = [128, 256], kernel_sizes = [3], dilations = [1], groups = [128]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [256, 2*256, 256], kernel_sizes = [3, 3], dilations = [1, 1], groups = [256, 256]),
                self.deconvolutionalFilter_resNet(numChannels = [256, 2*256, 256], kernel_sizes = [3, 3], dilations = [1, 1], groups = [256, 256]),
                
                # Convolution architecture: channel expansion
                self.changeChannels(numChannels = [256, 512], kernel_sizes = [3], dilations = [1], groups = [256]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [512, 2*512, 512], kernel_sizes = [3, 3], dilations = [1, 1], groups = [512, 512]),
                self.deconvolutionalFilter_resNet(numChannels = [512, 2*512, 512], kernel_sizes = [3, 3], dilations = [1, 1], groups = [512, 512]),
        )
    
    def deconvolutionalFilter_resNet(self, numChannels = [6, 6, 6], kernel_sizes = [3, 3], dilations = [1, 2, 1], groups = [1, 1]):
        return _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter(numChannels = numChannels, kernel_sizes = kernel_sizes, dilations = dilations, groups = groups),
                ), numCycles = 1)
        
    def deconvolutionalFilter(self, numChannels = [6, 6, 6], kernel_sizes = [3, 3], dilations = [1, 2], groups = [1, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
                # Convolution architecture: feature engineering
                nn.Conv1d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=kernel_sizes[0], stride=1, 
                          dilation = dilations[0], padding=paddings[0], padding_mode='reflect', groups=groups[0], bias=True),
                nn.SELU(),
                
                # Convolution architecture: feature engineering
                nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=kernel_sizes[1], stride=1, 
                          dilation = dilations[1], padding=paddings[1], padding_mode='reflect', groups=groups[1], bias=True),
                nn.SELU(),
        )
    
    def changeChannels(self, numChannels = [6, 6], kernel_sizes = [3], dilations = [1], groups = [1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
                # Convolution architecture: feature engineering
                nn.Conv1d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=kernel_sizes[0], stride=1, 
                          dilation = dilations[0], padding=paddings[0], padding_mode='reflect', groups=groups[0], bias=True),
                nn.SELU(),
        )
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numExpandedSignals = 512, numCompressedSignals = 64):
        super(signalEncoding, self).__init__(signalDimension, numExpandedSignals, numCompressedSignals)        
                        
    def forward(self, signalData):
        """ The shape of signalData: (batchSize, numSignals, sequenceLength) """
        # Specify the current input shape of the data.
        batchSize, numExpandedSignals, signalDimension = signalData.size()
        assert self.numExpandedSignals == numExpandedSignals
        assert self.signalDimension == signalDimension
                
        # ------------------------ CNN Architecture ------------------------ # 

        # Apply CNN architecture to compress the data.
        # encodedSignals = self.compressSignals(signalData) + self.simpleCompression(signalData)
        encodedSignals = self.simpleCompression(signalData)
        # signalData dimension: batchSize, numCompressedSignals, signalDimension
        
        # ------------------------------------------------------------------ # 
        
        return encodedSignals
        
    def printParams(self):
        # signalEncoding(signalDimension = 64, numExpandedSignals = 512, numCompressedSignals = 64).to('cpu').printParams()
        t1 = time.time()
        summary(self, (self.numExpandedSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
class signalDecoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numExpandedSignals = 512, numCompressedSignals = 64):
        super(signalDecoding, self).__init__(signalDimension, numExpandedSignals, numCompressedSignals)        
                        
    def forward(self, signalData):
        """ The shape of signalData: (batchSize, numSignals, sequenceLength) """
        # Specify the current input shape of the data.
        batchSize, numCompressedSignals, signalDimension = signalData.size()
        assert self.numCompressedSignals == numCompressedSignals
        assert self.signalDimension == signalDimension
                
        # ------------------------ CNN Architecture ------------------------ # 

        # Apply CNN architecture to compress the data.
        # encodedSignals = self.expandSignals(signalData) + self.simpleExpansion(signalData)
        encodedSignals = self.simpleExpansion(signalData)
        # signalData dimension: batchSize, numCompressedSignals, signalDimension
        
        # ------------------------------------------------------------------ # 
        
        return encodedSignals
        
    def printParams(self):
        # signalDecoding(signalDimension = 64, numExpandedSignals = 512, numCompressedSignals = 64).to('cpu').printParams()
        t1 = time.time()
        summary(self, (self.numCompressedSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
