# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import time

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import _convolutionalHelpers
import positionEncodings

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class signalEncoderBase(_convolutionalHelpers.convolutionalHelpers):
    def __init__(self, signalDimension = 64, numEncodedSignals = 64, numGroupPoints = 8):
        super(signalEncoderBase, self).__init__()
        # General shape parameters.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Specify the CPU or GPU capabilities.
        self.numEncodedSignals = numEncodedSignals
        self.signalDimension = signalDimension
        self.numGroupPoints = numGroupPoints
        self.numGroupPoints_upSampling = int(numGroupPoints/2)
        
        # Positional encoding.
        self.positionEncoding = positionEncodings.positionalEncoding.T[5].to(self.device)
        
        # Specify the normalizaion layers.
        self.layerNorm = nn.LayerNorm(self.signalDimension, eps=1E-10)
        
        # Initialize the weights
        self.initializeWeights()
        
    def initializeWeights(self):
                
        # ------------------------ ANN Architecture ------------------------ # 
                
        # Compress the signals by half.
        self.signalCompression = nn.Sequential(
            # Residual connection
            _convolutionalHelpers.ResNet(
                module = nn.Sequential( 
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                    # Neural architecture: Layer 1.
                    nn.Linear(self.numGroupPoints, self.numGroupPoints*2, bias = True),
                    nn.SELU(),
                    
                    # Neural architecture: Layer 1.
                    nn.Linear(self.numGroupPoints*2, self.numGroupPoints, bias = True),
                    nn.SELU(),
                ), numCycles = 1),
                
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                    # Neural architecture: Layer 1.
                    nn.Linear(self.numGroupPoints, self.numGroupPoints*2, bias = True),
                    nn.SELU(),
                    
                    # Neural architecture: Layer 1.
                    nn.Linear(self.numGroupPoints*2, self.numGroupPoints, bias = True),
                    nn.SELU(),
                ), numCycles = 1),
            ), numCycles = 1),
                
            # Neural architecture: Layer 2.
            nn.Linear(self.numGroupPoints, int(self.numGroupPoints/2), bias = True),
            nn.SELU(),
        )
        
        # Expand the signals by 2.
        self.signalExpansion = nn.Sequential(
            # Residual connection
            _convolutionalHelpers.ResNet(
                module = nn.Sequential( 
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                    # Neural architecture: Layer 1.
                    nn.Linear(self.numGroupPoints_upSampling, self.numGroupPoints_upSampling*2, bias = True),
                    nn.SELU(),
                    
                    # Neural architecture: Layer 1.
                    nn.Linear(self.numGroupPoints_upSampling*2, self.numGroupPoints_upSampling, bias = True),
                    nn.SELU(),
                ), numCycles = 1),
                
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                    # Neural architecture: Layer 1.
                    nn.Linear(self.numGroupPoints_upSampling, self.numGroupPoints_upSampling*2, bias = True),
                    nn.SELU(),
                    
                    # Neural architecture: Layer 1.
                    nn.Linear(self.numGroupPoints_upSampling*2, self.numGroupPoints_upSampling, bias = True),
                    nn.SELU(),
                ), numCycles = 1),
            ), numCycles = 1),
                
            # Neural architecture: Layer 2.
            nn.Linear(self.numGroupPoints_upSampling, self.numGroupPoints_upSampling*2, bias = True),
            nn.SELU(),
        )
        
        # ------------------------ CNN Architecture ------------------------ # 
        
        # Put more information into the signals' dimension.
        self.signalEncodingCNN = self.signalArchitecture_2D(numChannels = [1, 4, 8], groups = [1, 1, 1])
        self.localEncodingCNN = self.signalArchitecture_2D(numChannels = [1, 2, 4], groups = [1, 1, 1])
        
        # ------------------------------------------------------------------ # 
        
    def signalArchitecture_2D(self, numChannels = [1, 2, 4], groups = [1, 1, 1]):  
        numChannels = [int(numChannel) for numChannel in numChannels]
        groups = [int(group) for group in groups]

        return nn.Sequential(            
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                # Convolution architecture: feature engineering
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 2), padding=(0, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 3), padding=(0, 3), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                # Convolution architecture: feature engineering
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
            ), numCycles = 1))
        
    def pairSignals_downSampling(self, inputData, targetNumSignals):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.shape
        numRepeatedSignals = 0
        
        # Check if numSignals is odd
        if numSignals % 2 == 1:  
            # Repeat the last signal and concatenate it to the original tensor
            last_signal = inputData[:, -1, :].unsqueeze(1)
            inputData = torch.cat([inputData, last_signal], dim=1)
            
            # Reassign the number of signals.
            numSignals = inputData.size(1)
            numRepeatedSignals += last_signal.size(1)
            
        # If we are only slightly above the final number of signals.
        if targetNumSignals < numSignals < targetNumSignals*2:
            # Add a buffer of extra signals.
            last_signals = inputData[:, -(targetNumSignals*2 - numSignals):, :]
            inputData = torch.cat([inputData, last_signals], dim=1)
            
            # Reassign the number of signals.
            numSignals = inputData.size(1)
            numRepeatedSignals += last_signals.size(1)
                    
        # Pair up the signals.
        pairedData = inputData.view(batchSize, int(numSignals/2), 2, signalDimension)
        pairedData = pairedData.transpose(2, 3).contiguous().view(batchSize, int(numSignals/2), int(2*signalDimension))
        # pairedData dimension: batchSize, int(numSignals/2), int(2*signalDimension))
                        
        # Break apart the signals.
        brokenData = pairedData.view(batchSize, int(numSignals/2), -1, self.numGroupPoints) 
        # brokenData dimension: batchSize, int(numSignals/2), int(2*signalDimension/self.numGroupPoints), self.numGroupPoints
                    
        return brokenData, numRepeatedSignals
    
    def compileSignals_downSampling(self, inputData, numRepeatedSignals):
        # Extract the incoming data's dimension.
        batchSize, numSignals, pairedSignalDimension, finalNumGroupPoints = inputData.shape
                            
        # Pair up the signals.
        recombinedSignals = inputData.view(batchSize, numSignals, -1, 2)
        recombinedSignals = recombinedSignals.transpose(2, 3).contiguous().view(batchSize, -1, self.signalDimension)
        # recombinedSignals dimension: batchSize, numSignals*finalNumGroupPoints, self.signalDimension
                                
        return recombinedSignals[:, 0: 2*numSignals - numRepeatedSignals]
        
    def pairSignals_upSampling(self, inputData):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.shape

        # Break apart the signals.
        brokenData = inputData.view(batchSize, numSignals, -1, self.numGroupPoints_upSampling) 
        # brokenData dimension: batchSize, numSignals, int(signalDimension/self.numGroupPoints_upSampling), self.numGroupPoints_upSampling
                    
        return brokenData
    
    def compileSignals_upSampling(self, inputData):
        # Extract the incoming data's dimension.
        batchSize, numSignals, pairedSignalDimension, finalNumGroupPoints = inputData.shape
                            
        # Pair up the signals.
        recombinedSignals = inputData.view(batchSize, -1, self.signalDimension)
        # recombinedSignals dimension: batchSize, numSignals*2, self.signalDimension
                        
        return recombinedSignals
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numEncodedSignals = 64, numGroupPoints = 8):
        super(signalEncoding, self).__init__(signalDimension, numEncodedSignals, numGroupPoints)

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.size()
        
        # Assert that we have the expected data format.
        assert 2 <= numSignals, f"We need at least 3 signals for the encoding. You only provided {numSignals}."
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."
        
        # ----------------------- Data Preprocessing ----------------------- #
                            
        # Add positional encoding to the input.
        encodedData = inputData + self.positionEncoding
        # encodedData dimension: batchSize, numSignals, signalDimension
        
        # ------------------- Signal Expansion Algorithm ------------------- # 
            
        # While we have too few signals to process.
        while encodedData.size(1) <= self.numEncodedSignals:
            # print("Up", 0, encodedData.shape)
            # Pair up the signals into groups.
            pairedData = self.pairSignals_upSampling(encodedData)
            # pairedData dimension: batchSize, unknownSignals, signalDimension/self.numGroupPoints_upSampling, self.numGroupPoints_upSampling
            # print("Up", 1, pairedData.shape)
            
            # Learn how to upsample the data.
            encodedPairedData = self.signalExpansion(pairedData)
            # pairedData dimension: batchSize, unknownSignals, signalDimension/self.numGroupPoints_upSampling, self.numGroupPoints_upSampling*2
            # print("Up", 2, encodedPairedData.shape)

            # Recompile the signals into their final form.
            encodedData = self.compileSignals_upSampling(encodedPairedData)
            # encodedData dimension: batchSize, unknownSignals*2, signalDimension
            # print("Up", 3, encodedData.shape)
            
            # Create a channel for the signal data.
            encodedData = encodedData.unsqueeze(1)
            # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
            
            # Encode data with signal information.
            encodedData = self.localEncodingCNN(encodedData)
            # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
            
            # Remove the extra channel.
            encodedData = encodedData.squeeze(1)
            # encodedData dimension: batchSize, numEncodedSignals, signalDimension
            
            # Apply layer norm
            # encodedData = self.layerNorm(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension

        # ------------------ Signal Compression Algorithm ------------------ # 
        
        # While we have too many signals to process.
        while self.numEncodedSignals < encodedData.size(1):
            # print("Down", 0, encodedData.shape)
            
            # Pair up the signals into groups.
            pairedData, numRepeatedSignals = self.pairSignals_downSampling(encodedData, self.numEncodedSignals)
            # brokenData dimension: batchSize, int(unknownSignals/2), int(2*signalDimension/self.numGroupPoints), self.numGroupPoints
            # print("Down", 1, pairedData.shape)

            # Learn how to downsample the data.
            encodedPairedData = self.signalCompression(pairedData)
            # brokenData dimension: batchSize, int(unknownSignals/2), int(2*signalDimension/self.numGroupPoints), self.numGroupPoints/2
            # print("Down", 2, encodedPairedData.shape)
            
            # Recompile the signals into their final form.
            encodedData = self.compileSignals_downSampling(encodedPairedData, numRepeatedSignals)
            # encodedData dimension: batchSize, int(unknownSignals/2), signalDimension
            # print("Down", 3, encodedData.shape)
            
            # Create a channel for the signal data.
            encodedData = encodedData.unsqueeze(1)
            # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
            
            # Encode data with signal information.
            encodedData = self.localEncodingCNN(encodedData)
            # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
            
            # Remove the extra channel.
            encodedData = encodedData.squeeze(1)
            # encodedData dimension: batchSize, numEncodedSignals, signalDimension
            
            # Apply layer norm
            # encodedData = self.layerNorm(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension

        # ------------------------------------------------------------------ # 
        
        # ---------------------- Signal Reconstruction --------------------- # 
        
        # Create a channel for the signal data.
        encodedData = encodedData.unsqueeze(1)
        # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
        
        # Encode data with signal information.
        encodedData = self.signalEncodingCNN(encodedData)
        # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
        
        # Remove the extra channel.
        encodedData = encodedData.squeeze(1)
        # encodedData dimension: batchSize, numEncodedSignals, signalDimension

        # ------------------------------------------------------------------ # 
                
        return encodedData

    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64, numEncodedSignals = 64, numGroupPoints = 8).to('cpu').printParams(numSignals = 4)
        t1 = time.time()
        summary(self, (numSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

# -------------------------------------------------------------------------- #
# -------------------------- Decoder Architecture -------------------------- #   

class signalDecoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numEncodedSignals = 64, numGroupPoints = 8):
        super(signalDecoding, self).__init__(signalDimension, numEncodedSignals, numGroupPoints)        
                        
    def forward(self, inputData, targetNumSignals = 499):
        """ The shape of inputData: (batchSize, numEncodedSignals, signalDimension) """
        # Extract the incoming data's dimension.
        batchSize, numEncodedSignals, signalDimension = inputData.size()
        
        # Assert that we have the expected data format.
        assert 2 <= targetNumSignals, f"We need at least 3 signals for the encoding. You only provided {targetNumSignals}."
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."

        # ----------------------- Data Preprocessing ----------------------- #
                            
        # Add positional encoding to the input.
        encodedData = inputData + self.positionEncoding
        # encodedData dimension: batchSize, numEncodedSignals, signalDimension
        
        # ------------------- Signal Expansion Algorithm ------------------- # 
            
        # While we have too few signals to process.
        while encodedData.size(1) <= targetNumSignals:
            # print("Up", 0, encodedData.shape)
            # Pair up the signals into groups.
            pairedData = self.pairSignals_upSampling(encodedData)
            # pairedData dimension: batchSize, unknownSignals, signalDimension/self.numGroupPoints_upSampling, self.numGroupPoints_upSampling
            # print("Up", 1, pairedData.shape)
            
            # Learn how to upsample the data.
            encodedPairedData = self.signalExpansion(pairedData)
            # pairedData dimension: batchSize, unknownSignals, signalDimension/self.numGroupPoints_upSampling, self.numGroupPoints_upSampling*2
            # print("Up", 2, encodedPairedData.shape)

            # Recompile the signals into their final form.
            encodedData = self.compileSignals_upSampling(encodedPairedData)
            # encodedData dimension: batchSize, unknownSignals*2, signalDimension
            
            # Create a channel for the signal data.
            encodedData = encodedData.unsqueeze(1)
            # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
            
            # Encode data with signal information.
            encodedData = self.localEncodingCNN(encodedData)
            # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
            
            # Remove the extra channel.
            encodedData = encodedData.squeeze(1)
            # encodedData dimension: batchSize, numEncodedSignals, signalDimension
            
            # Apply layer norm
            # encodedData = self.layerNorm(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            # print("Up", 3, encodedData.shape)

        # ------------------ Signal Compression Algorithm ------------------ # 
        
        # While we have too many signals to process.
        while targetNumSignals < encodedData.size(1):
            # print("Down", 0, encodedData.shape)
            
            # Pair up the signals into groups.
            pairedData, numRepeatedSignals = self.pairSignals_downSampling(encodedData, targetNumSignals)
            # brokenData dimension: batchSize, int(unknownSignals/2), int(2*signalDimension/self.numGroupPoints), self.numGroupPoints
            # print("Down", 1, pairedData.shape)

            # Learn how to downsample the data.
            encodedPairedData = self.signalCompression(pairedData)
            # brokenData dimension: batchSize, int(unknownSignals/2), int(2*signalDimension/self.numGroupPoints), self.numGroupPoints/2
            # print("Down", 2, encodedPairedData.shape)
            
            # Recompile the signals into their final form.
            encodedData = self.compileSignals_downSampling(encodedPairedData, numRepeatedSignals)
            # encodedData dimension: batchSize, int(unknownSignals/2), signalDimension
            # print("Down", 3, encodedData.shape)
            
            # Create a channel for the signal data.
            encodedData = encodedData.unsqueeze(1)
            # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
            
            # Encode data with signal information.
            encodedData = self.localEncodingCNN(encodedData)
            # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
            
            # Remove the extra channel.
            encodedData = encodedData.squeeze(1)
            # encodedData dimension: batchSize, numEncodedSignals, signalDimension
            
            # Apply layer norm
            # encodedData = self.layerNorm(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
        
        # ---------------------- Signal Reconstruction --------------------- # 
        
        # Create a channel for the signal data.
        encodedData = encodedData.unsqueeze(1)
        # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
        
        # Encode data with signal information.
        encodedData = self.signalEncodingCNN(encodedData)
        # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
        
        # Remove the extra channel.
        encodedData = encodedData.squeeze(1)
        # encodedData dimension: batchSize, numEncodedSignals, signalDimension

        # ------------------------------------------------------------------ # 
                
        return encodedData

    def printParams(self):
        # signalDecoding(signalDimension = 64, numEncodedSignals = 64, numGroupPoints = 8).to('cpu').printParams()
        t1 = time.time()
        summary(self, (self.numEncodedSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        