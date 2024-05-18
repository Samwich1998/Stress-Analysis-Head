# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class finalEmotionPrediction(nn.Module):
    def __init__(self, inputDimension = 900, numEmotions = 6, emotionLength = 100):
        super(finalEmotionPrediction, self).__init__()
        # General parameters.
        self.inputDimension = inputDimension
        self.emotionLength = emotionLength
        self.numEmotions = numEmotions
    
        # A list of modules to encode each signal.
        self.emotionModules = nn.ModuleList()  # Use ModuleList to store child modules.
        # signalEncodingModules dimension: self.numSignals

        # For each signal.
        for emotionInd in range(self.numEmotions):
            # Learn final features
            self.emotionModules.append(
                nn.Sequential(
                    # Neural architecture: Layer 1.
                    nn.Linear(inputDimension, 64, bias = True),
                    nn.BatchNorm1d(64, affine = True, momentum = 0.1, track_running_stats=True),
                    nn.GELU(),
                    
                    # Neural architecture: Layer 1.
                    nn.Linear(64, 32, bias = True),
                    nn.BatchNorm1d(32, affine = True, momentum = 0.1, track_running_stats=True),
                    nn.GELU(),
                    
                    # Neural architecture: Layer 1.
                    nn.Linear(32, emotionLength, bias = True),
                )
            )
            
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals*signalLength) """  
        batchSize, inputDimension = inputData.size()
        assert inputDimension == self.inputDimension, print(inputData.shape[1], self.inputDimension)
        
        finalEmotionDistributions = torch.zeros((self.numEmotions, batchSize, self.emotionLength))
        
        for emotionInd in range(self.numEmotions):
            # Reduce the dimension of the signal data
            finalEmotionDistributions[emotionInd, :, :] = self.emotionModules[emotionInd](inputData)
            # Dimension: batchSize, emotionLength

        return finalEmotionDistributions
    
    def printParams(self, inputDimension = 16):
        #signalEncoding(inputDimension = 16, outputDimension = 16, numSignals = 50).printParams()
        summary(self, (inputDimension,))

    
    