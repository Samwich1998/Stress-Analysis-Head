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
# --------------------------- Model Architecture --------------------------- #

class featureExtraction(nn.Module):
    def __init__(self, numCommonFeatures = 32, numEncodedSignals = 64, manifoldLength = 32):
        super(featureExtraction, self).__init__()
        # General parameters.
        self.numEncodedSignals = numEncodedSignals      # The dimensionality of the encoded data.
        self.numCommonFeatures = numCommonFeatures      # The number of features from considering all the signals.
        self.manifoldLength = manifoldLength            # The final length of each signal after projection.
        self.numSignalFeatures = 4
        # Tunable parameters.
        self.finalNumSignals = int(numEncodedSignals/4)
                
        # Reduce the size of the features.
        self.cnnFeatureReduction = nn.Sequential(
                # Convolution architecture: channel expansion
                self.changeChannels(numChannels = [1, 2, 4]),
                
                # Encode nearby signals with information from each other.
                self.projectionArchitecture_2D(numChannels = [4, 4, 4], groups = [1, 1, 1]),
                # Convolution architecture: learnable feature reduction.
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=(2, 1), dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=1, bias=True),
                
                # Encode nearby signals with information from each other.
                self.projectionArchitecture_2D(numChannels = [4, 4, 4], groups = [1, 1, 1]),
                # Convolution architecture: learnable feature reduction.
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=(2, 1), dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=1, bias=True),
                
                # Convolution architecture: channel expansion
                self.changeChannels(numChannels = [4, 2, 1]),
                # Encode nearby signals with information from each other.
                self.projectionArchitecture_2D(numChannels = [1, 2, 4], groups = [1, 1, 1]),
                
                # Convolution architecture: channel compression

        )
        
        # # Learn signal features.
        # self.extractSignalFeatures = nn.Sequential(
        #         # Neural architecture: Layer 1.
        #         nn.Linear(self.manifoldLength, 16, bias = True),
        #         nn.SELU(),
                
        #         # Neural architecture: Layer 2.
        #         nn.Linear(16, self.numSignalFeatures, bias = True),
        #         nn.SELU(),
        # )
    
        # # Learn final features.
        # self.extractFeatures = nn.Sequential(
        #         # Neural architecture: Layer 1.
        #         nn.Linear(self.numSignalFeatures*self.finalNumSignals, 32, bias = True),
        #         nn.BatchNorm1d(32, affine = True, momentum = 0.1, track_running_stats=True),
        #         nn.SELU(),
                
        #         # Neural architecture: Layer 2.
        #         nn.Linear(32, self.numCommonFeatures, bias = True),
        #         nn.BatchNorm1d(self.numCommonFeatures, affine = True, momentum = 0.1, track_running_stats=True),
        #         nn.SELU(),
        # )
    
    def changeChannels(self, numChannels = [1, 2, 4]):
        return nn.Sequential(
            # Convolution architecture: channel expansion
            nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation = 1, padding=(1, 1), padding_mode='reflect', groups=1, bias=True),
            nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation = 1, padding=(1, 1), padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
    def projectionArchitecture_2D(self, numChannels = [1, 4, 8], groups = [1, 1, 1]):  
        numChannels = [int(numChannel) for numChannel in numChannels]
        groups = [int(group) for group in groups]

        return nn.Sequential(            
                # Residual connection
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                        # Convolution architecture: channel expansion
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(2, 1), padding=(2, 1), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation=(3, 1), padding=(3, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Residual connection
                        _convolutionalHelpers.ResNet(
                             module = nn.Sequential(
                                nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[2], kernel_size=(3, 3), stride=1, dilation=(1, 2), padding=(1, 2), padding_mode='reflect', groups=groups[2], bias=True),
                                nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[1], kernel_size=(3, 3), stride=1, dilation=(1, 3), padding=(1, 3), padding_mode='reflect', groups=groups[1], bias=True),
                                nn.SELU(),
                        ), numCycles = 1),
                        
                        # Convolution architecture: channel compression
                        nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[0], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[0], kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=groups[0], bias=True),
                        nn.SELU(),
            ), numCycles = 1))
            
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numEncodedSignals, manifoldLength) """  
        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, numEncodedSignals, manifoldLength = inputData.size()
        
        # Assert we have the expected data format.
        assert manifoldLength == self.manifoldLength, f"Expected manifold dimension of {self.manifoldLength}, but got {manifoldLength}."
        assert numEncodedSignals == self.numEncodedSignals, f"Expected {self.numEncodedSignals} signals, but got {numEncodedSignals} signals."
        
        # ---------------------- CNN Feature Reduction --------------------- # 

        # Create a channel for the signal data.
        inputData = inputData.unsqueeze(1)
        # inputData dimension: batchSize, 1, numEncodedSignals, manifoldLength
        
        # Reduce the information by half.
        reducedFeatures = self.cnnFeatureReduction(inputData)
        # reducedFeatures dimension: batchSize, 1, finalNumSignals, manifoldLength
        
        # Remove the extra channel.
        reducedFeatures = reducedFeatures.squeeze(1)
        # encodedData dimension: batchSize, finalNumSignals, signalDimension
        
        # -------------------- Final Feature Extraction -------------------- # 
        
        # -------------------- Signal Feature Extraction ------------------- # 

        # # Apply CNN architecture to decompress the data.
        # signalFeatureData = self.extractSignalFeatures(reducedFeatures)
        # # signalFeatureData dimension: batchSize, finalNumSignals, numSignalFeatures
        
        # # -------------------- Final Feature Extraction -------------------- # 
        
        # # Flatten the data to synthesize all the features together.
        # featureData = signalFeatureData.view(batchSize, self.finalNumSignals*self.numSignalFeatures)
        # # featureData dimension: batchSize, finalNumSignals*numSignalFeatures
        
        # # Extract the final features from the signals.
        # finalFeatures = self.extractFeatures(featureData)
        # # featureData dimension: batchSize, numCommonFeatures

        # # ------------------------------------------------------------------ # 

        # return finalFeatures
    
    def printParams(self):
        # featureExtraction(numCommonFeatures = 32, numEncodedSignals = 64, manifoldLength = 32).to('cpu').printParams()
        summary(self, (self.numEncodedSignals, self.manifoldLength,))
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

    
    