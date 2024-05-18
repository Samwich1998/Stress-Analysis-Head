# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class encoderCNN(nn.Module):
    def __init__(self, numIndicesShift = 3):
        super(encoderCNN, self).__init__()
        
        # Encode spatial features.
        self.compressSignals = nn.Sequential(
            # Convolution architecture: Layer 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(numIndicesShift, 3), stride=(2, 1), padding=(0,1)),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1)),
            nn.SiLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),
        
            # Convolution architecture: Layer 2
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1)),
            nn.SiLU(),
            nn.BatchNorm2d(4),
        
            # # Convolution architecture: Layer 3
            # nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=(0, 1)),
            # nn.MaxPool2d(kernel_size=2, stride=(1, 1)),
            # nn.SiLU(),
            # nn.BatchNorm2d(4),
        
            # Convolution architecture: Layer 4
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=(1,1), padding=(0, 0)),
            nn.SiLU(),
        )

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, sequenceLength, numSignals) """
        # Reshape the data to contain one channel
        inputData = inputData.unsqueeze(1)
        
        # Apply CNN architecture.
        inputData = self.compressSignals(inputData)
        # The new dimension: batchSize, numChannels, imageHeight, imageWidth
                
        return inputData
    
    def printParams(self, maxSeqLength = 150, signalEmbeddingDim = 8):
        #encoderCNN(4, 120, 3).printParams()
        summary(self, (maxSeqLength, signalEmbeddingDim))
        
    
class decodingLayer(nn.Module):
    def __init__(self, inputShape = (1, 1, 11, 1), maxSeqLength = 120):
        super(decodingLayer, self).__init__()
        self.inputShape = inputShape

        # Decode spatial features.
        self.expandSignals = nn.Sequential(
            # Deconvolution architecture: Layer 1
            nn.ConvTranspose2d(in_channels=inputShape[1], out_channels=2, kernel_size=(2, 2), stride=(2,2), padding=0),
            nn.SiLU(),
            nn.BatchNorm2d(2),
            nn.Dropout(0.1),

            # Deconvolution architecture: Layer 2
            nn.ConvTranspose2d(in_channels=2, out_channels=4, kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.SiLU(),
            nn.BatchNorm2d(4),

            # Deconvolution architecture: Layer 4
            nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=(1,1), padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(4),
        )
        
        # Find the final dimension of the image.
        finalDeconvShape = self.expandSignals(
                    torch.ones(self.inputShape)).shape
        finalDeconvDimension = torch.tensor(finalDeconvShape[1:]).prod().item()

        # Fully Connected layers for final signal reconstruction
        self.fc_layers = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(finalDeconvDimension, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5),

            # Neural architecture: Layer 2.
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.SiLU(),

            # Neural architecture: Layer 3.
            nn.Linear(256, maxSeqLength, bias=True),
        )
                
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numChannels, imageHeight, imageWidth) """
        # Apply deconvolution CNN architecture.
        inputData = self.expandSignals(inputData)
        # The new dimension: batchSize, numChannels, finalHeight, finalWidth
        
        # Apply ANN architecture.
        inputData = inputData.view(inputData.size(0), -1) # Flatten the image to contain a list of features
        inputData = self.fc_layers(inputData)
        # The new dimension: batchSize, maxSeqLength

        return inputData
    
    def printParams(self):
        #decodingLayer().printParams()
        summary(self, self.inputShape[1:])
    
    
    
    
    
    