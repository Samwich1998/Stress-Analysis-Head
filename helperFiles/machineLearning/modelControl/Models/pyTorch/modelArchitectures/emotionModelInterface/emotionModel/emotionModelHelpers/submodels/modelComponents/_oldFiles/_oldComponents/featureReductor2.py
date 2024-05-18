# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class featureReductor(nn.Module):
    def __init__(self, inputDimension = 16, outputDimension = 8):
        super(featureReductor, self).__init__()

        # Learn final features
        self.featureReductionLayers = nn.Sequential(            
            # Neural architecture: Layer 3.
            nn.Linear(inputDimension, 64, bias = True),
            nn.BatchNorm1d(64, track_running_stats = True),
            nn.GELU(),
            nn.Dropout(0.5),
            
            # Neural architecture: Layer 3.
            nn.Linear(64, 32, bias = True),
            nn.BatchNorm1d(32, track_running_stats = True),
            nn.GELU(),
            nn.Dropout(0.5),
            
            # Neural architecture: Layer 3.
            nn.Linear(32, 16, bias = True),
            nn.BatchNorm1d(16, track_running_stats = True),
            nn.GELU(),
            nn.Dropout(0.5),
            
            # Neural architecture: Layer 4.
            nn.Linear(16, outputDimension, bias = True),
            nn.BatchNorm1d(outputDimension, track_running_stats = True),
            nn.GELU(),
        )

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, compressedDim, signalEncodedDim) """
        # Apply non-linearity to learn new features.
        inputData = self.featureReductionLayers(inputData)
        # Dimension: batchSize, numChannels*imageHeight*imageWidth
                
        return inputData
    
    def printParams(self, inputDimension = 16, outputDimension = 30):
        #featureReductor().printParams()
        summary(self, (inputDimension,))

    
    