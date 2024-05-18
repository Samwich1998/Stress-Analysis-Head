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
    def __init__(self, signalDimension = 64, numEncodedSignals = 64, numGroupPoints = 16):
        super(signalEncoderBase, self).__init__()
        # General shape parameters.
        self.numEncodedSignals = numEncodedSignals
        self.signalDimension = signalDimension
        self.numGroupPoints = numGroupPoints
        
        # Positional encoding.
        self.positionEncoding = positionEncodings.positionalEncoding.T[5]
        
        # Initialize the weights
        self.initializeWeights()
        
    def initializeWeights(self):
                
        # ------------------------ ANN Architecture ------------------------ # 
                
        # Compress the signals by half.
        self.signalCompression = nn.Sequential(
                # Neural architecture: Layer 1.
                nn.Linear(self.numGroupPoints, self.numGroupPoints*2, bias = True),
                nn.SELU(),
                
                # Neural architecture: Layer 1.
                nn.Linear(self.numGroupPoints*2, self.numGroupPoints, bias = True),
                nn.SELU(),
                
                # Neural architecture: Layer 2.
                nn.Linear(self.numGroupPoints, int(self.numGroupPoints/2), bias = True),
                nn.SELU(),
        )
        
        # Expand the signals by 2.
        self.signalExpansion = nn.Sequential(
                # Neural architecture: Layer 1.
                nn.Linear(self.numGroupPoints, int(self.numGroupPoints*2), bias = True),
                nn.SELU(),
                
                # Neural architecture: Layer 2.
                nn.Linear(int(self.numGroupPoints*2), int(self.numGroupPoints*2), bias = True),
                nn.SELU(),
        )
        
        # ------------------------------------------------------------------ # 
        
    def pairSignals_downSampling(self, inputData):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.shape
        
        # Check if numSignals is odd
        if numSignals % 2 == 1:  
            # Repeat the last signal and concatenate it to the original tensor
            last_signal = inputData[:, -1, :].unsqueeze(1)
            inputData = torch.cat([inputData, last_signal], dim=1)
            
            # Reassign the number of signals.
            numSignals = inputData.size(1)
            
        numRepeatedSignals = 0
        # If we are only slightly above the final number of signals.
        if self.numEncodedSignals < numSignals < self.numEncodedSignals*2:
            # Add a buffer of extra signals.
            last_signals = inputData[:, -(self.numEncodedSignals*2 - numSignals):, :]
            inputData = torch.cat([inputData, last_signals], dim=1)
            
            # Reassign the number of signals.
            numSignals = inputData.size(1)
            numRepeatedSignals = last_signals.size(1)
                    
        # Pair up the signals.
        pairedData = inputData.view(batchSize, int(numSignals/2), 2, signalDimension)
        pairedData = pairedData.transpose(2, 3).contiguous().view(batchSize, int(numSignals/2), int(2*signalDimension))
        # pairedData dimension: batchSize, int(numSignals/2), int(2*signalDimension))
                        
        # Break apart the signals.
        brokenData = pairedData.view(batchSize, int(numSignals/2), -1, self.numGroupPoints) 
        # brokenData dimension: batchSize, int(numSignals/2), int(2*signalDimension/self.numGroupPoints), self.numGroupPoints
                    
        return brokenData, numRepeatedSignals
    
    def compileSignals_downSampling(self, inputData):
        # Extract the incoming data's dimension.
        batchSize, numSignals, pairedSignalDimension, finalNumGroupPoints = inputData.shape
        
        # Check if numSignals is odd
        if numSignals % 2 == 1:  
            # Repeat the last signal and concatenate it to the original tensor
            last_signal = inputData[:, -1, :].unsqueeze(1)
            inputData = torch.cat([inputData, last_signal], dim=1)
            
            # Reassign the number of signals.
            numSignals = inputData.size(1)
                            
        # Pair up the signals.
        recombinedSignals = inputData.view(batchSize, -1, self.signalDimension, 2)
        recombinedSignals = recombinedSignals.transpose(2, 3).contiguous().view(batchSize, -1, self.signalDimension)
        # recombinedSignals dimension: batchSize, numSignals*finalNumGroupPoints, self.signalDimension
                        
        return recombinedSignals
        
    def pairSignals_upSampling(self, inputData):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.shape

        # Break apart the signals.
        brokenData = inputData.view(batchSize, numSignals, -1, self.numGroupPoints) 
        # brokenData dimension: batchSize, numSignals, int(signalDimension/self.numGroupPoints), self.numGroupPoints
                    
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
    def __init__(self, signalDimension = 64, numEncodedSignals = 64, numGroupPoints = 16):
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
        numRepeatedSignals = 0

        # ------------------ Signal Compression Algorithm ------------------ # 
        
        # While we have too many signals to process.
        while self.numEncodedSignals < encodedData.size(1):
            # Pair up the signals into groups.
            pairedData, numRepeatedSignals = self.pairSignals_downSampling(encodedData)
            # brokenData dimension: batchSize, int(numSignals/2), int(2*signalDimension/self.numGroupPoints), self.numGroupPoints

            # Learn how to downsample the data.
            encodedPairedData = self.signalCompression(pairedData)
            # brokenData dimension: batchSize, int(numSignals/2), int(2*signalDimension/self.numGroupPoints), self.numGroupPoints/2
            
            # Recompile the signals into their final form.
            encodedData = self.compileSignals_downSampling(encodedPairedData)
            # encodedData dimension: batchSize, int(numSignals/2), signalDimension
                        
        # ------------------- Signal Expansion Algorithm ------------------- # 
            
        # While we have too few signals to process.
        while encodedData.size(1) <= self.numEncodedSignals / 2:
            # Pair up the signals into groups.
            pairedData = self.pairSignals_upSampling(encodedData)
            # pairedData dimension: batchSize, numSignals, signalDimension/self.numGroupPoints, self.numGroupPoints
            
            # Learn how to upsample the data.
            encodedPairedData = self.signalExpansion(pairedData)
            # pairedData dimension: batchSize, numSignals, signalDimension/self.numGroupPoints, self.numGroupPoints*2

            # Recompile the signals into their final form.
            encodedData = self.compileSignals_upSampling(encodedPairedData)
            # encodedData dimension: batchSize, numSignals*2, signalDimension

        if encodedData.size(1) < self.numEncodedSignals:
            # Add a buffer of extra signals.
            last_signals = encodedData[:, -(self.numEncodedSignals - encodedData.size(1)):, :]
            encodedData = torch.cat([encodedData, last_signals], dim=1)
            # Count the number of repeated signals.
            numRepeatedSignals = last_signals.size(1)
                
        # ------------------------------------------------------------------ #   
        print(encodedData.shape, self.numEncodedSignals, numRepeatedSignals)
                
        return encodedData, numRepeatedSignals

    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64, numEncodedSignals = 64, numGroupPoints = 16).to('cpu').printParams(numSignals = 4)
        t1 = time.time()
        summary(self, (numSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

# -------------------------------------------------------------------------- #
# -------------------------- Decoder Architecture -------------------------- #   

class signalDecoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numEncodedSignals = 64, numGroupPoints = 16):
        super(signalDecoding, self).__init__(signalDimension, numEncodedSignals, numGroupPoints)        
                        
    def forward(self, encodedData, numSignals = 10):
        """ The shape of encodedData: (batchSize, numEncodedSignals, signalDimension) """
        # Extract the incoming data's dimension.
        batchSize, numEncodedSignals, signalDimension = encodedData.size()
        
        # Assert that we have the expected data format.
        assert 2 <= numSignals, f"We need at least 3 signals for the encoding. You only provided {numSignals}."
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."

        # ----------------------- Data Preprocessing ----------------------- #
                            
        # Add positional encoding to the input.
        encodedData = encodedData + self.positionEncoding
        # encodedData dimension: batchSize, numEncodedSignals, signalDimension

        # ------------------ Signal Compression Algorithm ------------------ # 
        
        # While we have too many signals to process.
        while numSignals < encodedData.size(1):
            # Pair up the signals into groups.
            pairedData, _ = self.pairSignals_downSampling(encodedData)
            # pairedData dimension: batchSize, int(numEncodedSignals/2), self.numGroupPoints, int(2*signalDimension/self.numGroupPoints))

            # Learn how to downsample the data.
            encodedPairedData = self.signalCompression(pairedData)
            
            # Recompile the signals into their final form.
            encodedData = self.compileSignals(encodedPairedData)
            # encodedData dimension: batchSize, int(numEncodedSignals/2), signalDimension
                        
        # ------------------- Signal Expansion Algorithm ------------------- # 
            
        # While we have too few signals to process.
        while encodedData.size(1) <= numSignals / 2:
            # Pair up the signals into groups.
            pairedData, _ = self.pairSignals_downSampling(encodedData)
            # pairedData dimension: batchSize, int(numEncodedSignals/2), self.numGroupPoints, int(2*signalDimension/self.numGroupPoints))

            # Learn how to upsample the data.
            encodedPairedData = self.signalExpansion(pairedData)
            # pairedData dimension: batchSize, int(numEncodedSignals/2), self.numGroupPoints, int(signalDimension/self.numGroupPoints))

            # Recompile the signals into their final form.
            encodedData = self.compileSignals(encodedPairedData)
            # encodedData dimension: batchSize, int(numEncodedSignals/2), signalDimension
                    
        if encodedData.size(1) < numSignals:
            # Add a buffer of extra signals.
            last_signals = encodedData[:, -(self.numEncodedSignals - encodedData.size(1)):, :]
            encodedData = torch.cat([encodedData, last_signals], dim=1)
                
        # ------------------------------------------------------------------ #   
        print(encodedData.shape, numSignals)
        
        return encodedData

    def printParams(self):
        # signalDecoding(signalDimension = 64, numEncodedSignals = 64, numGroupPoints = 16).to('cpu').printParams()
        t1 = time.time()
        summary(self, (self.numEncodedSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        