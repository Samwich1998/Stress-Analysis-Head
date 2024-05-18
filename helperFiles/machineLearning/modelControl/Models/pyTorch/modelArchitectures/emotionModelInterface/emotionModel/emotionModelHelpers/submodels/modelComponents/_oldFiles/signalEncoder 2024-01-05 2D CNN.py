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
    def __init__(self, signalDimension = 64, numEncodedSignals = 64):
        super(signalEncoderBase, self).__init__()
        # General shape parameters.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Specify the CPU or GPU capabilities.
        self.numEncodedSignals = numEncodedSignals
        self.signalDimension = signalDimension
        
    def initializeWeights(self):
        
        # ------------------------- Pooling Layers ------------------------- # 
        
        numPoolingChannels = 6
        # Compile the learnable pooling layers.
        signalExpansionLayer = nn.ConvTranspose2d(in_channels=numPoolingChannels, out_channels=numPoolingChannels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(1, 0), groups=1, bias=True, dilation=1, padding_mode='zeros', device=self.device)
        signalCompressionLayer = nn.Conv2d(in_channels=numPoolingChannels, out_channels=numPoolingChannels, kernel_size=(3, 3), stride=(2, 1), dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=1, bias=True, device=self.device)
        # Compile average pooling layers.
        averageSignalCompressionLayer = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), ceil_mode=False, count_include_pad=True, divisor_override=None)
        averageSignalExpansionLayer = nn.UpsamplingBilinear2d(scale_factor=(2,1))
        # Compile the learnable pooling layers with channel expansion.
        self.signalExpansion = self.poolingArchitecture(self.changeChannels([1, numPoolingChannels]), _convolutionalHelpers.addModules(signalExpansionLayer, averageSignalExpansionLayer), self.changeChannels([numPoolingChannels, 1]))
        self.signalCompression = self.poolingArchitecture(self.changeChannels([1, numPoolingChannels]), _convolutionalHelpers.addModules(signalCompressionLayer, averageSignalCompressionLayer), self.changeChannels([numPoolingChannels, 1]))

        numPoolingChannels = 9
        # Create channel expansion and compression modules.
        self.channelExpansion = self.changeChannels([1, numPoolingChannels])
        self.channelCompression = self.changeChannels([numPoolingChannels, 1])
                        
        # ------------------------ CNN Architecture ------------------------ # 
                
        # Create model for learning local information.
        self.localSignalEncoding = self.projectionArchitecture_2D(numChannels = [1, 4, 6], groups = [1, 1, 1])
        self.signalReconstruction = self.signalArchitecture_2D(numChannels = [1, 2, 4], groups = [1, 1, 1])
        self.adjustSignals = self.minorSignalArchitecture_2D(numChannels = [1, 2, 4], groups = [1, 1, 1])
        
        # ------------------------------------------------------------------ # 
        
    def setupSignalPooling(self, targetNumSignals):
        # Specify specific pooling strategies.
        finalLayerMaxPooling = nn.AdaptiveMaxPool2d((targetNumSignals, self.signalDimension))
        finalLayerAvgPooling = nn.AdaptiveAvgPool2d((targetNumSignals, self.signalDimension))
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
            nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(3, 3), stride=1, dilation = 1, padding=(1, 1), padding_mode='reflect', groups=1, bias=True),
            nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation = 1, padding=(1, 1), padding_mode='reflect', groups=1, bias=True),
            nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation = 1, padding=(1, 1), padding_mode='reflect', groups=1, bias=True),
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
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 2), padding=(0, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 3), padding=(0, 3), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 2), padding=(1, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 2), padding=(0, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 2), padding=(1, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 2), padding=(0, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                                                
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 2), padding=(1, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Convolution architecture: channel compression
                        nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[0], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
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
                        nn.SELU(),
                ), numCycles = 1),
    
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
                ), numCycles = 1),
    
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
                ), numCycles = 1),
    
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
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
                               nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                               nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 2), padding=(0, 2), padding_mode='reflect', groups=groups[2], bias=True),
                               nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 3), padding=(0, 3), padding_mode='reflect', groups=groups[1], bias=True),
                               nn.SELU(),
                       ), numCycles = 1),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
            ), numCycles = 1))
    
    def calculateStandardizationLoss(self, inputData, expectedMean = 0, expectedStandardDeviation = 1, dim=-1):
        # Calculate the data statistics on the last dimension.
        standardDeviationData = inputData.std(dim=dim, keepdim=False)
        meanData = inputData.mean(dim=dim, keepdim=False)

        # Calculate the squared deviation from mean = 0; std = 1.
        standardDeviationError = ((standardDeviationData - expectedStandardDeviation) ** 2).mean(dim=(1, 2))
        meanError = ((meanData - expectedMean) ** 2).mean(dim=(1, 2))
        
        return 0.5*meanError + 0.5*standardDeviationError
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numEncodedSignals = 32):
        super(signalEncoding, self).__init__(signalDimension, numEncodedSignals)        
        # Initialize the model weights.
        self.initializeWeights()
                        
    def forward(self, inputData, targetNumSignals = 499, signalEncodingLoss = 0, numIterations = 1e-15, trainingFlag = False):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.size()
        self.setupSignalPooling(targetNumSignals)

        # Assert that we have the expected data format.
        assert 3 <= numSignals, f"We need at least 3 signals for the encoding. You only provided {numSignals}."
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."

        # ----------------------- Data Preprocessing ----------------------- # 
                    
        # Create a channel for the signal data.
        encodedData = inputData.unsqueeze(1)
        # encodedData dimension: batchSize, 1, numSignals, signalDimension
        
        # ------------------ Signal Compression Algorithm ------------------ # 
        
        # While we have too many signals to process.
        while targetNumSignals < encodedData.size(2) / 2:
            # Encode nearby signals with information from each other.
            encodedData = self.localSignalEncoding(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
                        
            # Pool half the signals together.
            encodedData = self.signalCompression(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            if trainingFlag:
                # Keep tracking on the normalization loss through each loop.
                signalEncodingLoss = signalEncodingLoss + self.calculateStandardizationLoss(encodedData, expectedMean = 0, expectedStandardDeviation = 1, dim=-1)
                numIterations = numIterations + 1
            
        # ------------------- Signal Expansion Algorithm ------------------- # 
            
        # While we have too many signals to process.
        while encodedData.size(2) < targetNumSignals / 2:
            # Encode the sequence along its dimension.
            encodedData = self.adjustSignals(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            # Upsample to twice as many signals.
            encodedData = self.signalExpansion(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
            if trainingFlag:
                # Keep tracking on the normalization loss through each loop.
                signalEncodingLoss = signalEncodingLoss + self.calculateStandardizationLoss(encodedData, expectedMean = 0, expectedStandardDeviation = 1, dim=-1)
                numIterations = numIterations + 1

        # ----------------- Finalize the Number of Signals ----------------- # 
        
        # If we are compressing the data.
        if targetNumSignals < encodedData.size(2):
            # Encode nearby signals with information from each other.
            encodedData = self.localSignalEncoding(encodedData)
            # encodedData dimension: batchSize, 1, unknownSignals, signalDimension
            
        # If we are expanding the data.
        elif targetNumSignals < encodedData.size(2):
            # Adjust the signals.
            encodedData = self.adjustSignals(encodedData)
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
        
        return encodedData, signalEncodingLoss/numIterations
        
    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64, numEncodedSignals = 64).to('cpu').printParams(numSignals = 4)
        t1 = time.time()
        summary(self, (numSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        