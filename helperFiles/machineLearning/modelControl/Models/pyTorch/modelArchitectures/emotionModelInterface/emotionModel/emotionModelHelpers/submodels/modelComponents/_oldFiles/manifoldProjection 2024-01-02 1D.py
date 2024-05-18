# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys

# PyTorch
import torch.nn as nn
from torchsummary import summary

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import _convolutionalHelpers

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class manifoldProjectionHead(nn.Module):
    def __init__(self, signalDimension = 32, manifoldLength = 32, numEncodedSignals = 50):
        super(manifoldProjectionHead, self).__init__()
        # General shape parameters.
        self.signalDimension = signalDimension
        self.manifoldLength = manifoldLength
        self.numEncodedSignals = numEncodedSignals
        
        # Assert that we reduce the dimensionality by exactly 2.
        assert manifoldLength == signalDimension/2, f"We expect {manifoldLength} == {signalDimension}/2"
        
    def changeChannels(self, numChannels = [6, 6, 6], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]):
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
    
    # conv = nn.Conv2d(in_channels=numChannels, out_channels=numChannels, kernel_size=(1, width), groups=height, padding=0)
    
    def convolutionalFilter(self, numChannels = [6, 6, 6], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return _convolutionalHelpers.ResNet(
                module = nn.Sequential(
                    # Convolution architecture: feature engineering
                    nn.Conv1d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=kernel_sizes[0], stride=1, 
                              dilation = dilations[0], padding=paddings[0], padding_mode='reflect', groups=groups[0], bias=True),
                    nn.SELU(),
                    
                    # Add a residual connecton.
                    _convolutionalHelpers.ResNet(
                            module = nn.Sequential(
                            # Convolution architecture: feature engineering
                            nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[1], kernel_size=kernel_sizes[1], stride=1, 
                                      dilation = dilations[1], padding=paddings[1], padding_mode='reflect', groups=groups[1], bias=True),
                            nn.SELU(),
                    ), numCycles = 1),
                    
                    # Convolution architecture: feature engineering
                    nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=kernel_sizes[2], stride=1, 
                              dilation = dilations[2], padding=paddings[2], padding_mode='reflect', groups=groups[2], bias=True),
                    nn.SELU(),
                ), numCycles = 1)
    
    def deconvolutionalFilter(self, numChannels = [6, 6, 6], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return _convolutionalHelpers.ResNet(
                module = nn.Sequential(
                    # Convolution architecture: feature engineering
                    nn.Conv1d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=kernel_sizes[0], stride=1, 
                              dilation = dilations[0], padding=paddings[0], padding_mode='reflect', groups=groups[0], bias=True),
                    nn.SELU(),
                    
                    # Convolution architecture: feature engineering
                    nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[1], kernel_size=kernel_sizes[1], stride=1, 
                              dilation = dilations[1], padding=paddings[1], padding_mode='reflect', groups=groups[1], bias=True),
                    nn.SELU(),
                    
                    # Convolution architecture: feature engineering
                    nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=kernel_sizes[2], stride=1, 
                              dilation = dilations[2], padding=paddings[2], padding_mode='reflect', groups=groups[2], bias=True),
                    nn.SELU(),
                ), numCycles = 1)

# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalProjection(manifoldProjectionHead):
    def __init__(self, signalDimension = 64, manifoldLength = 32, numEncodedSignals = 50):
        super(signalProjection, self).__init__(signalDimension, manifoldLength, numEncodedSignals)
        # Define a trainable weight that is signal-specific.
        # self.signalWeights = nn.Parameter(torch.randn((1, self.numEncodedSignals, 1)))
        # signalWeights: How does each signal contribute to the biological profile (latent manifold).

        # Encode spatial features.
        self.projectSignals = nn.Sequential(
            
            # ---------- Dimension: batchSize, 1, signalDimension ----------- # 

            # Convolution architecture: channel expansion
            self.changeChannels(numChannels = [1, 2, 4], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
            
            # ---------- Dimension: batchSize, 6, signalDimension ----------- # 

            # Convolution architecture: feature engineering
            self.convolutionalFilter(numChannels = [4, 8, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 3], groups = [2, 2, 2]),
            self.convolutionalFilter(numChannels = [4, 8, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [2, 2, 2]),
            
            # Apply a pooling layer to reduce the signal's dimension.
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=2, dilation=1, padding=1, padding_mode='reflect', groups=1, bias=True),
            
            # ---------- Dimension: batchSize, 6, manifoldLength --------- # 

            # Convolution architecture: feature engineering
            self.convolutionalFilter(numChannels = [4, 8, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [2, 2, 2]),
            self.convolutionalFilter(numChannels = [4, 8, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [2, 2, 2]),
            
            # ---------- Dimension: batchSize, 6, manifoldLength --------- # 

            # Convolution architecture: channel compression
            self.changeChannels(numChannels = [4, 2, 1], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
            
            # Convolution architecture: feature engineering
            self.convolutionalFilter(numChannels = [1, 1, 1], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalFilter(numChannels = [1, 1, 1], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            
            # ---------- Dimension: batchSize, 1, manifoldLength --------- # 
        )
        
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numEncodedSignals, compressedLength) """  
        # Extract the incoming data's dimension.
        batchSize, numEncodedSignals, compressedLength = inputData.size()
        # Assert that we have the expected data format.
        assert numEncodedSignals == self.numEncodedSignals, \
            f"You initialized {self.numEncodedSignals} signals but only provided {numEncodedSignals} signals."
        assert compressedLength == self.signalDimension, \
            f"You provided a signal of length {compressedLength}, but we expected {self.signalDimension}."
            
        # ------------------------ CNN Architecture ------------------------ # 
        
        # Reshape the data to the expected input into the CNN architecture.
        signalData = inputData.view(batchSize * numEncodedSignals, 1, compressedLength) # Seperate out indivisual signals.
        # signalData dimension: batchSize*numEncodedSignals, 1, sequenceLength
        
        # Apply CNN architecture to compress the data.
        projectedSignals = self.projectSignals(signalData)
        # projectedSignals dimension: batchSize*numEncodedSignals, 1, manifoldLength
        
        # Seperate put each signal into its respective batch.
        manifoldData = projectedSignals.view(batchSize, numEncodedSignals, self.manifoldLength) 
        # projectedSignals dimension: batchSize, numEncodedSignals, manifoldLength
        
        # ------------------------------------------------------------------ # 

        return manifoldData

    def printParams(self):
        # signalProjection(signalDimension = 64, manifoldLength = 32, numEncodedSignals = 100).to('cpu').printParams()
        summary(self, (self.numEncodedSignals, self.signalDimension))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

# -------------------------------------------------------------------------- #
# -------------------------- Decoder Architecture -------------------------- #   

class manifoldReconstruction(manifoldProjectionHead):
    def __init__(self, signalDimension=64, manifoldLength=32, numEncodedSignals=50):
        super(manifoldReconstruction, self).__init__(signalDimension, manifoldLength, numEncodedSignals)
        
        # Encode spatial features.
        self.unprojectSignals = nn.Sequential(
            
            # ---------- Dimension: batchSize, 1, manifoldLength ----------- # 

            # Convolution architecture: channel expansion
            self.changeChannels(numChannels = [1, 2, 4], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
            
            # ---------- Dimension: batchSize, 6, manifoldLength ----------- # 

            # Convolution architecture: feature engineering
            self.convolutionalFilter(numChannels = [4, 8, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 3], groups = [2, 2, 2]),
            self.convolutionalFilter(numChannels = [4, 8, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [2, 2, 2]),
            
            # Apply a pooling layer to reduce the signal's dimension.
            nn.ConvTranspose1d(in_channels=4, out_channels=4, kernel_size=3, stride=2, dilation=1, padding=1, output_padding=1, groups=1, bias=True, padding_mode='zeros'),
            
            # ---------- Dimension: batchSize, 6, signalDimension --------- # 

            # Convolution architecture: feature engineering
            self.convolutionalFilter(numChannels = [4, 8, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [2, 2, 2]),
            self.convolutionalFilter(numChannels = [4, 8, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [2, 2, 2]),
            
            # ---------- Dimension: batchSize, 6, signalDimension --------- # 

            # Convolution architecture: channel compression
            self.changeChannels(numChannels = [4, 2, 1], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
            
            # Convolution architecture: feature engineering
            self.convolutionalFilter(numChannels = [1, 1, 1], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalFilter(numChannels = [1, 1, 1], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            
            # ---------- Dimension: batchSize, 1, signalDimension --------- # 
        )

    def forward(self, manifoldData):
        """ The shape of manifoldData: (batchSize, numEncodedSignals, manifoldLength) """
        batchSize, numEncodedSignals, manifoldLength = manifoldData.size()
        
        # Assert we have the expected data format.
        assert manifoldLength == self.manifoldLength, f"Expected manifold dimension of {self.manifoldLength}, but got {manifoldLength}."
        assert numEncodedSignals == self.numEncodedSignals, f"Expected {self.numEncodedSignals} signals, but got {numEncodedSignals} signals."
            
        # ------------------------ CNN Architecture ------------------------ # 

        # Reshape the signals.
        projectedSignals = manifoldData.view(batchSize*numEncodedSignals, 1, manifoldLength) 
        # compressedSignals dimension: batchSize*numEncodedSignals, 1, manifoldLength

        # Apply CNN architecture to decompress the data.
        reconstructedSignals = self.unprojectSignals(projectedSignals)
        # reconstructedSignals dimension: batchSize*numEncodedSignals, 1, signalDimension
                
        # Organize the signals into the original batches.
        reconstructedData = reconstructedSignals.view(batchSize, numEncodedSignals, self.signalDimension)
        # compressedSignals dimension: batchSize, numEncodedSignals, signalDimension
        
        # ------------------------------------------------------------------ # 

        return reconstructedData

    def printParams(self):
        # manifoldReconstruction(signalDimension = 64, manifoldLength = 32, numEncodedSignals = 100).printParams()
        summary(self, (self.numEncodedSignals, self.manifoldLength))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        