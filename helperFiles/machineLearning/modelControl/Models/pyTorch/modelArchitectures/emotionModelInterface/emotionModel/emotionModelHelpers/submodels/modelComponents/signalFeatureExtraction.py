# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import math

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class signalFeatureExtraction(nn.Module):
    def __init__(self, signalDimension = 16, numSignalFeatures = 2, numSignals = 50):
        super(signalFeatureExtraction, self).__init__()
        # General parameters.
        self.numSignalFeatures = numSignalFeatures
        self.signalDimension = signalDimension
        self.numSignals = numSignals
        
        # Pooling layer.
        self.avPooling_Stride2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        
        self.reduceSignalsCNN_1 = nn.Sequential(
                # Convolution architecture: Layer 1, Conv 1-2
                nn.Conv1d(in_channels=1, out_channels=4, kernel_size=7, stride=1, dilation = 1, padding=3, padding_mode='reflect', groups=1, bias=True),
                nn.Conv1d(in_channels=4, out_channels=1, kernel_size=7, stride=1, dilation = 1, padding=3, padding_mode='reflect', groups=1, bias=True),
                nn.LayerNorm(self.signalDimension, eps = 1E-10),
                nn.SELU(),
        )
        
        self.reduceSignalsFC = nn.Sequential(
                # Neural architecture: Layer 1.
                nn.Linear(math.ceil(self.signalDimension), self.numSignalFeatures, bias = True),
                # nn.BatchNorm1d(signalDimension*2, affine = True, momentum = 0.1, track_running_stats=True),
                nn.SELU(),
        )
            
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, signalDimension) """  
        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, numSignals, signalDimension = inputData.size()
        assert numSignals == self.numSignals, f"{numSignals} {self.numSignals}"
        
        # Process all the signals at once.
        signalData = inputData.view(batchSize*numSignals, 1, signalDimension)

        # Learn new features from the signals using CNN.
        signalData = (signalData + self.reduceSignalsCNN_1(signalData))/2
        signalData = signalData.squeeze(1)
        # Learn new features from the signals using FC.
        signalData = self.reduceSignalsFC(signalData)
        
        # Reorganize the data by the indivisual signal.
        compressedData = signalData.view(batchSize, numSignals, self.numSignalFeatures)
        # Dimension: batchSize, numSignals, numSignalFeatures
        
        return compressedData
    
    def printParams(self):
        #signalFeatureExtraction(signalDimension = 16, numSignalFeatures = 2, numSignals = 50).printParams()
        summary(self, (self.numSignals, self.signalDimension,))
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

    
    