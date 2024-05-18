# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class featureReductor(nn.Module):
    def __init__(self, inputShape = (1, 3), numFeatures = 5):
        super(featureReductor, self).__init__()
        
        # Encode spatial features.
        self.leanSpatialPatterns = nn.Sequential(
            # Convolution architecture: Layer 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 2), stride=(1, 1), dilation = 1, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=3, stride=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(8, track_running_stats = True),
            nn.Dropout(0.1),
        
            # Convolution architecture: Layer 2
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(2, 2), stride=(1, 1), dilation = 1, padding=(1, 1)),
            nn.GELU(),
        )
        
        # Pass fake data through the CNN to calculate the final shape
        finalEncodedImageShape = self.leanSpatialPatterns(torch.ones((1, 1, inputShape[0], inputShape[1]))).shape
        compressedDim = torch.tensor(finalEncodedImageShape[1:]).prod().item()
        assert compressedDim > 25, f"Just a though, compressedDim is small: {compressedDim}"
        
        # Learn final features
        self.featureExtractionLayers = nn.Sequential(            
            # # Neural architecture: Layer 2.
            # nn.Linear(21, 32, bias = True),
            # nn.BatchNorm1d(32, track_running_stats = True),
            # nn.GELU(),
            # nn.Dropout(0.5),
            
            # Neural architecture: Layer 3.
            nn.Linear(compressedDim, 32, bias = True),
            nn.BatchNorm1d(32, track_running_stats = True),
            nn.GELU(),
            nn.Dropout(0.5),
            
            # Neural architecture: Layer 3.
            nn.Linear(32, 16, bias = True),
            nn.BatchNorm1d(16, track_running_stats = True),
            nn.GELU(),
            nn.Dropout(0.5),
            
            # Neural architecture: Layer 4.
            nn.Linear(16, numFeatures, bias = True),
            nn.BatchNorm1d(numFeatures, track_running_stats = True),
            nn.GELU(),
        )

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, compressedDim, signalEncodedDim) """
        # Reshape the data to contain one channel
        inputData = inputData.unsqueeze(1)
        
        # Learn spatial features within the data.
        inputData = self.leanSpatialPatterns(inputData)
        # Dimension: batchSize, numChannels, imageHeight, imageWidth
        
        # Flatten the encoded data into a list of features.
        inputData = inputData.view(inputData.size(0), -1) 
        # Dimension: batchSize, numChannels*imageHeight*imageWidth
        
        # Apply non-linearity to learn new features.
        inputData = self.featureExtractionLayers(inputData)
        # Dimension: batchSize, numChannels*imageHeight*imageWidth

                
        return inputData
    
    def printParams(self, compressedDim = 18, signalEncodedDim = 10):
        #featureReductor(numFeatures = 4).printParams()
        summary(self, (compressedDim, signalEncodedDim))

    
    