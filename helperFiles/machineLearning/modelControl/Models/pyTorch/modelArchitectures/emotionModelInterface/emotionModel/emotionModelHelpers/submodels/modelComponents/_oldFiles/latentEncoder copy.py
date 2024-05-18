# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# Import files
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import positionEncodings

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class latentTransformationHead(nn.Module):
    def __init__(self, signalDimension = 32, latentDimension = 32, numSignals = 50):
        super(latentTransformationHead, self).__init__()
        # General shape parameters.
        self.signalDimension = signalDimension
        self.latentDimension = latentDimension
        self.numSignals = numSignals

# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(latentTransformationHead):
    def __init__(self, signalDimension = 32, latentDimension = 32, numSignals = 50):
        super(signalEncoding, self).__init__(signalDimension, latentDimension, numSignals)
        # Positional encoding.
        self.positionEncoding = positionEncodings.positionalEncoding.T[0]
        
        # Define a trainable weight that is signal-specific.
        self.signalWeights = nn.Parameter(torch.randn((1, self.numSignals, 1)))
        # signalWeights: How does each signal contribute to the biological profile (latent manifold).
        
        # Encode spatial features.
        self.signalCombination = nn.Sequential(
            # Convolution architecture: Layer 1, Conv 1
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), stride=1,  dilation = 1, padding=(2,1), padding_mode='reflect', groups=1, bias=True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), stride=1,  dilation = 1, padding=(2,1), padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
            
            # Convolution architecture: Layer 1, Conv 1
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(numSignals//2 + numSignals%2, 3), stride=1,  dilation = 1, padding=(0, 1), padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
            
            # Convolution architecture: Layer 2, Conv 1
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(5, 3), stride=1,  dilation = 1, padding=(2,1), padding_mode='reflect', groups=1, bias=True),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(6, 3), stride=1,  dilation = 1, padding=(2,1), padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
            
            # Convolution architecture: Layer 1, Conv 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(numSignals//2, 3), stride=1,  dilation = 1, padding=(0, 1), padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
            
            # Convolution architecture: Layer 2, Conv 1
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(1, 3), stride=1,  dilation = 1, padding=(0,1), padding_mode='reflect', groups=1, bias=True),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1, 3), stride=1,  dilation = 1, padding=(0,1), padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        assert numSignals >= 4, "Based on my signal reduction algorith, I expect at least 4 signals."
        
        self.latentEncoding = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(signalDimension, latentDimension, bias = True),
            nn.SELU(),
        )
        
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """  
        # Extract the incoming data's dimension.
        batchSize, numSignals, compressedLength = inputData.size()
        # Assert that we have the expected data format.
        assert numSignals == self.numSignals, \
            f"You initialized {self.numSignals} signals but only provided {numSignals} signals."
        assert compressedLength == self.signalDimension, \
            f"You provided a signal of length {compressedLength}, but we expected {self.signalDimension}."
        
        # ----------------------- Signal Combination ----------------------- #   
        
        # Data preparation.
        inputData = inputData.unsqueeze(1)
        # inputData dimension: batchSize, 1, numSignals, compressedLength
        
        # Capture signal information and condense into a single vector.
        combinedSignals = self.signalCombination(inputData)
        # combinedSignals dimension: batchSize, 1, 1, compressedLength
        assert combinedSignals.shape[2] == 1, "Assert I removed all the signals"
                
        # Data preparation.
        combinedSignals = combinedSignals.squeeze(1).squeeze(1)
        # inputData dimension: batchSize, 1, numSignals, compressedLength

        # ------------------------- Latent Encoding ------------------------ #   
        
        # Add positional encoding.
        combinedSignals = self.positionEncoding + combinedSignals
        # combinedSignals dimension: batchSize, compressedLength
        
        # Encoder the signals into the latent dimension.
        latentData = self.latentEncoding(combinedSignals)
        # latentData dimension: batchSize, latentDimension
                
        # ------------------------------------------------------------------ #

        return latentData

    def printParams(self):
        #signalEncoding(signalDimension = 32, latentDimension = 32, numSignals = 100).printParams()
        summary(self, (self.numSignals, self.signalDimension))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

# -------------------------------------------------------------------------- #
# -------------------------- Decoder Architecture -------------------------- #   

class signalDecoding(latentTransformationHead):
    def __init__(self, signalDimension=32, latentDimension=32, numSignals=50):
        super(signalDecoding, self).__init__(signalDimension, latentDimension, numSignals)
        # A list of modules to decode each signal.
        self.signalDecodingModules_1 = nn.ModuleList()
        self.signalDecodingModules_2 = nn.ModuleList()
        
        self.latentDecoding = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(latentDimension, signalDimension, bias = True),
            nn.SELU(),
        )
        
        # Encode spatial features.
        self.signalDifferentiation = nn.Sequential(
            # Convolution architecture: Layer 2, Conv 1
            nn.ConvTranspose2d(in_channels=1, out_channels=4, kernel_size=(1, 3), stride=1, dilation = 1, padding=(0,1), groups=1, bias=True),
            nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=(1, 3), stride=1, dilation = 1, padding=(0,1), groups=1, bias=True),
            nn.SELU(),

            # Convolution architecture: Layer 2, Conv 1
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(numSignals//2, 3), stride=1, dilation = 1, padding=(0,1), groups=1, bias=True),
            nn.SELU(),
            
            # Convolution architecture: Layer 2, Conv 1
            nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=(6, 3), stride=1, dilation = 1, padding=(2,1), groups=1, bias=True),
            nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(5, 3), stride=1, dilation = 1, padding=(2,1), groups=1, bias=True),
            nn.SELU(),
            
            # Convolution architecture: Layer 2, Conv 1
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=(numSignals//2 + numSignals%2, 3), stride=1, padding=(0,1), groups=1, bias=True),
            nn.SELU(),
            
            # Convolution architecture: Layer 2, Conv 1
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(5, 3), stride=1, padding=(2,1), groups=1, bias=True),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(5, 3), stride=1, padding=(2,1), groups=1, bias=True),
            nn.SELU(),
        )

    def forward(self, latentData):
        """ The shape of latentData: (batchSize, latentDimension) """
        batchSize, latentDimension = latentData.size()
        
        # Assert we have the expected data format.
        assert latentDimension == self.latentDimension, \
            f"Expected latent dimension of {self.latentDimension} but got {latentDimension}."
            
        # ------------------------- Latent Decoding ------------------------ #   
        
        # Decode the signals from the latent dimension.
        combinedSignals = self.latentDecoding(latentData)
        # Calculate dimension: batchSize, compressedLength
                
        # --------------------- Signal Differentiation --------------------- #   
        
        # Add a channel for the signals.
        combinedSignals = combinedSignals.unsqueeze(1).unsqueeze(1)
        # latentData dimension: batchSize, 1, 1, compressedLength
        
        # Reconstruct the signals.
        reconstructedData = self.signalDifferentiation(combinedSignals)
        # latentData dimension: batchSize, 1, self.numSignals, compressedLength
        assert reconstructedData.shape[2] == self.numSignals, "Assert I added all the signals"
        
        # Remove a channel.
        reconstructedData = reconstructedData.squeeze(1)
        # latentData dimension: batchSize, compressedLength
        
        # ------------------------------------------------------------------ #

        return reconstructedData

    def printParams(self):
        #signalDecoding(signalDimension = 32, latentDimension = 32, numSignals = 100).printParams()
        summary(self, (self.latentDimension,))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        