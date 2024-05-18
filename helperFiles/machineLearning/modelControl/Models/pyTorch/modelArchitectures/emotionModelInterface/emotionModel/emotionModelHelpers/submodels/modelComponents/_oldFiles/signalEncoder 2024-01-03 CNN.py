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

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class signalEncoderBase(_convolutionalHelpers.convolutionalHelpers):
    def __init__(self, signalDimension = 16, numEncodedSignals = 16):
        super(signalEncoderBase, self).__init__()
        # General shape parameters.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Specify the CPU or GPU capabilities.
        self.numEncodedSignals = numEncodedSignals
        self.signalDimension = signalDimension
        
    def initializeWeights(self):
        
        # ------------------------- Pooling Layers ------------------------- # 
        
        numPoolingChannels = 8
        # Compile the learnable pooling layers.
        signalExpansionLayer = nn.ConvTranspose2d(in_channels=numPoolingChannels, out_channels=numPoolingChannels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0), groups=1, bias=True, dilation=1, padding_mode='zeros', device=self.device)
        signalCompressionLayer = nn.Conv2d(in_channels=numPoolingChannels, out_channels=numPoolingChannels, kernel_size=(3, 3), stride=(2, 1), dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=1, bias=True, device=self.device)
        # Compile the learnable pooling layers with channel expansion.
        self.signalExpansion = self.poolingArchitecture(self.changeChannels([1, numPoolingChannels]), signalExpansionLayer, self.changeChannels([numPoolingChannels, 1]))
        self.signalCompression = self.poolingArchitecture(self.changeChannels([1, numPoolingChannels]), signalCompressionLayer, self.changeChannels([numPoolingChannels, 1]))

        numPoolingChannels = 6
        # Create channel expansion and compression modules.
        self.channelExpansion = self.changeChannels([1, numPoolingChannels])
        self.channelCompression = self.changeChannels([numPoolingChannels, 1])
        
        # Specify the normalizaion layers.
        self.layerNorm = nn.LayerNorm(self.signalDimension, eps=1E-10)
                
        # ------------------------ CNN Architecture ------------------------ # 
                
        # Create model for learning local information.
        self.finalSignalEncoding = self.projectionArchitecture_2D(numChannels = [1, 4, 6], groups = [1, 1, 1])
        self.localSignalEncoding = self.projectionArchitecture_2D(numChannels = [1, 4, 6], groups = [1, 1, 1])
        self.signalReconstruction = self.signalArchitecture_2D(numChannels = [1, 2, 4], groups = [1, 1, 1])
        self.adjustSignals = self.minorSignalArchitecture_2D(numChannels = [1, 2, 4], groups = [1, 1])
        
        # ------------------------------------------------------------------ # 
        
    def setupSignalPooling(self, numSignals):
        # Specify specific pooling strategies.
        finalLayerMaxPooling = nn.AdaptiveMaxPool2d((numSignals, self.signalDimension))
        finalLayerAvgPooling = nn.AdaptiveAvgPool2d((numSignals, self.signalDimension))
        finalLayerMinPooling = self.minPooling(finalLayerMaxPooling)
        
        # Compile the pooling layers
        finalPoolingLayers = [finalLayerMaxPooling, finalLayerAvgPooling, finalLayerMinPooling]
        
        # Create the final pooling modules.
        self.finalSignalPooling = nn.Sequential(
            # Convolution architecture: channel expansion
            _convolutionalHelpers.ResNet(
                module = self.channelExpansion,
                numCycles = 1
            ),
            
            # Apply a pooling layer to reduce the signal's dimension.
            _convolutionalHelpers.splitPoolingHead(module = self.splitPooling, poolingLayers = finalPoolingLayers),
            
            # Convolution architecture: channel compression
            self.channelCompression,
        )
        
    def poolingArchitecture(self, channelExpansion, poolingMethod, channelCompression):
        return nn.Sequential(
            # Convolution architecture: channel expansion
            _convolutionalHelpers.ResNet(
                module = channelExpansion,
                numCycles = 1),
            # Apply a pooling layer.
            poolingMethod,
            # Convolution architecture: channel compression
            channelCompression,
        )
            
    def changeChannels(self, numChannels):
        return nn.Sequential(
            # Convolution architecture: channel expansion
            nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(3, 1), stride=1, dilation = 1, padding=(1, 0), padding_mode='reflect', groups=1, bias=True),
            # nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation = 1, padding=(0, 1), padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )

    def projectionArchitecture_2D(self, numChannels = [1, 4], groups = [1, 1]):  
        numChannels = [int(numChannel) for numChannel in numChannels]
        groups = [int(group) for group in groups]

        return nn.Sequential(            
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                        # Convolution architecture: channel expansion
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(3, 1), stride=1, dilation=(1, 1), padding=(1, 0), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(3, 1), stride=1, dilation=(1, 1), padding=(1, 0), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(2, 2), padding=(2, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(2, 2), padding=(2, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
               
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Convolution architecture: channel compression
                        nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[0], kernel_size=(3, 1), stride=1, dilation=(1, 1), padding=(1, 0), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(3, 1), stride=1, dilation=(1, 1), padding=(1, 0), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
            ), numCycles = 1))
    
    def signalArchitecture_2D(self, numChannels = [1, 4, 8], groups = [1, 1, 1]):  
        numChannels = [int(numChannel) for numChannel in numChannels]
        groups = [int(group) for group in groups]

        return nn.Sequential(            
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
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
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 2), padding=(0, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
            ), numCycles = 1))
    
    def minorSignalArchitecture_2D(self, numChannels = [1, 4, 8], groups = [1, 1]):  
        numChannels = [int(numChannel) for numChannel in numChannels]
        groups = [int(group) for group in groups]

        return nn.Sequential(            
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                # Convolution architecture: feature engineering
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 2), padding=(0, 2), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                # Convolution architecture: feature engineering
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 3), padding=(0, 3), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
            ), numCycles = 1))
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numEncodedSignals = 32):
        super(signalEncoding, self).__init__(signalDimension, numEncodedSignals)
        # Initialize transformer parameters.
        self.numEncodedSignals = numEncodedSignals  # The number of times the latent channel is bigger than the value.
        
        # Initialize the model weights.
        self.initializeWeights()
        self.setupSignalPooling(numEncodedSignals)
                        
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.size()
        # Assert that we have the expected data format.
        assert 3 <= numSignals, f"We need at least 3 signals for the encoding. You only provided {numSignals}."
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."
        
        # ----------------------- Data Preprocessing ----------------------- # 
                    
        # Create a channel for the signal data.
        encodedData = inputData.unsqueeze(1)
        # encodedData dimension: batchSize, 1, numSignals, signalDimension
        
        # ------------------ Signal Compression Algorithm ------------------ # 
        
        # While we have too many signals to process.
        while self.numEncodedSignals < encodedData.size(2) / 2:
            # Encode nearby signals with information from each other.
            encodedData = self.localSignalEncoding(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
                        
            # Pool half the signals together.
            encodedData = self.signalCompression(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Adjust the signals.
            encodedData = self.adjustSignals(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Apply layer norm
            # encodedData = self.layerNorm(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
        # ------------------- Signal Expansion Algorithm ------------------- # 
            
        # While we have too many signals to process.
        while encodedData.size(2) < self.numEncodedSignals / 2:
            # Encode nearby signals with information from each other.
            encodedData = self.localSignalEncoding(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Upsample to twice as many signals.
            encodedData = self.signalExpansion(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Adjust the signals.
            encodedData = self.adjustSignals(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Apply layer norm
            # encodedData = self.layerNorm(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension

        # ----------------- Finalize the Number of Signals ----------------- # 

        # Encode nearby signals with information from each other.
        encodedData = self.finalSignalEncoding(encodedData)
        # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
        
        # Condense the signals down to a common number.
        encodedData = self.finalSignalPooling(encodedData)
        # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
        
        # Encode nearby signals with information from each other.
        encodedData = self.signalReconstruction(encodedData)
        # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
        
        # Remove the extra channel.
        encodedData = encodedData.squeeze(1)
        # encodedData dimension: batchSize, numEncodedSignals, signalDimension
                
        # ------------------------------------------------------------------ #   
        
        return encodedData

    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64, numEncodedSignals = 64).to('cpu').printParams(numSignals = 4)
        t1 = time.time()
        summary(self, (numSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

# -------------------------------------------------------------------------- #
# -------------------------- Decoder Architecture -------------------------- #   

class signalDecoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numEncodedSignals = 64):
        super(signalDecoding, self).__init__(signalDimension, numEncodedSignals)        
        # Initialize the model weights.
        self.initializeWeights()
                        
    def forward(self, encodedData, numSignals = 10):
        """ The shape of encodedData: (batchSize, numEncodedSignals, signalDimension) """
        batchSize, numEncodedSignals, signalDimension = encodedData.size()
        self.setupSignalPooling(numSignals)
        
        # Assert that we have the expected data format.
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."
        assert numEncodedSignals == self.numEncodedSignals, f"You provided {numEncodedSignals} condensed signals, but we expected {self.numEncodedSignals}."
                        
        # ----------------------- Data Preprocessing ----------------------- #
                            
        # Create a channel for the signal data.
        encodedData = encodedData.unsqueeze(1)
        # encodedData dimension: batchSize, 1, numEncodedSignals, signalDimension
           
        # ------------------ Signal Compression Algorithm ------------------ # 
        
        # While we have too many signals to process.
        while numSignals < encodedData.size(2) / 2:
            # Encode nearby signals with information from each other.
            encodedData = self.localSignalEncoding(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
                        
            # Pool half the signals together.
            encodedData = self.signalCompression(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Adjust the signals.
            encodedData = self.adjustSignals(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Apply layer norm
            encodedData = self.layerNorm(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
                                                
        # ------------------- Signal Expansion Algorithm ------------------- # 
            
        # While we have too many signals to process.
        while encodedData.size(2) < numSignals / 2:
            # Encode nearby signals with information from each other.
            encodedData = self.localSignalEncoding(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Upsample to twice as many signals.
            encodedData = self.signalExpansion(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Adjust the signals.
            encodedData = self.adjustSignals(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Apply layer norm
            encodedData = self.layerNorm(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
        # ----------------- Finalize the Number of Signals ----------------- # 
                
        # Encode nearby signals with information from each other.
        encodedData = self.finalSignalEncoding(encodedData)
        # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
                        
        # Condense the signals down to a common number.
        decodedData = self.finalSignalPooling(encodedData)
        # decodedData dimension: batchSize, 1, numSignals, signalDimension
                                
        # Encode nearby signals with information from each other.
        decodedData = self.signalReconstruction(decodedData)
        # decodedData dimension: batchSize, 1, numSignals, signalDimension
                
        # Remove the extra channel.
        decodedData = decodedData.squeeze(1)
        # decodedData dimension: batchSize, numSignals, signalDimension

        # ------------------------------------------------------------------ #   
        
        return decodedData

    def printParams(self):
        # signalDecoding(signalDimension = 64, numEncodedSignals = 32).to('cpu').printParams()
        t1 = time.time()
        summary(self, (self.numEncodedSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        