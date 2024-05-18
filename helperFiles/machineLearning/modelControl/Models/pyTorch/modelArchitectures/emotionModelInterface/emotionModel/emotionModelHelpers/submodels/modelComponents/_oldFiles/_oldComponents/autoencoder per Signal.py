# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class encodingLayer(nn.Module):
    def __init__(self, maxSeqLength, compressedLength):
        super(encodingLayer, self).__init__()
        
        # Encode spatial features.
        self.compressSignals = nn.Sequential(
            # Convolution architecture: Layer 1
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(8, track_running_stats = True),
            nn.Dropout(0.1),
        
            # Convolution architecture: Layer 2
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1, dilation = 1, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(4, track_running_stats = True),
        
            # Convolution architecture: Layer 3
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=1, dilation = 1, padding=1),
            nn.GELU(),
        )
        
        # Pass fake data through the CNN to calculate the final shape
        finalEncodedImageShape = self.compressSignals(torch.ones((1, 1, maxSeqLength))).shape
        compressedDim = torch.tensor(finalEncodedImageShape[1:]).prod().item()
        
        # Learn final features
        self.applyNonLinearity = nn.Sequential(                        
            # Neural architecture: Layer 1
            nn.Linear(compressedDim, 32, bias = True),
            nn.BatchNorm1d(32, track_running_stats = True),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # Neural architecture: Layer 3
            nn.Linear(32, compressedLength, bias = True),
            nn.BatchNorm1d(compressedLength, track_running_stats = True),
            nn.GELU(),
        )

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, sequenceLength) """
        
        # Apply CNN architecture to reduce spatial dimension.
        inputData = inputData.unsqueeze(1) # Add one channel to the signal.
        compressedData = self.compressSignals(inputData)
        # The new dimension: batchSize, numChannels = 1, imageLength
        
        # Flatten out the CNN output.
        compressedDataFlattened = compressedData.view(compressedData.size(0), -1)
        # compressedData dimension: batchSize, numChannels*imageLength -> batchSize, compressedDim
        
        # Apply the ANN architecture to add non-linearity.
        compressedData = self.applyNonLinearity(compressedDataFlattened)
        # The new dimension: batchSize, compressedLength
                
        return compressedData
    
    def printParams(self, maxSeqLength = 150):
        #encodingLayer(maxSeqLength = 150, compressedLength = 16).printParams(maxSeqLength = 150)
        summary(self, (maxSeqLength,)) # summary(model, inputShape)
        
    
class decodingLayer(nn.Module):
    def __init__(self, compressedLength = 16, maxSeqLength = 150):
        super(decodingLayer, self).__init__()
        
        finalLinearDim = 64
        # Learn final features
        self.applyNonLinearity = nn.Sequential(                        
            # Neural architecture: Layer 1
            nn.Linear(compressedLength, 32, bias = True),
            nn.BatchNorm1d(32, track_running_stats = True),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # Neural architecture: Layer 2
            nn.Linear(32, finalLinearDim, bias = True),
            nn.BatchNorm1d(finalLinearDim, track_running_stats = True),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Decode spatial features.
        self.expandSignals = nn.Sequential(
            # Deconvolution architecture: Layer 1
            nn.ConvTranspose1d(in_channels=1, out_channels=4, kernel_size=2, stride=2, dilation = 1, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(4, track_running_stats = True),
            nn.Dropout(0.1),

            # Deconvolution architecture: Layer 2
            nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size=2, stride=2, dilation = 1, padding=0),
            nn.GELU(),
        )
        
        # Find the final dimension of the decompressed image.
        finalDeconvShape = self.expandSignals(torch.ones((1, 1, finalLinearDim))).shape
        finalDeconvDimension = torch.tensor(finalDeconvShape[1:]).prod().item()

        # Fully Connected layers for final signal reconstruction
        self.flattenToSequence = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(finalDeconvDimension, 256, bias = True),
            nn.BatchNorm1d(256, track_running_stats = True),
            nn.GELU(),
            nn.Dropout(0.2),
            
            # Neural architecture: Layer 2
            nn.Linear(256, maxSeqLength, bias=True),
        )
                
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, compressedLength) """
        
        # Reconstruct a compressed signal.
        inputData = self.applyNonLinearity(inputData)
        # The new dimension: batchSize, finalLinearDim

        # Apply a CNN deconvolution for a smoothed upscale.
        inputData = inputData.unsqueeze(1) # Reshape the data to contain one channel
        inputData = self.expandSignals(inputData)
        # The new dimension: batchSize, numChannels, imageLength
        
        # Match the original sequence length.
        inputData = inputData.view(inputData.size(0), -1) # Flatten the image
        inputData = self.flattenToSequence(inputData)
        # The new dimension: batchSize, maxSeqLength

        return inputData
    
    def printParams(self, compressedLength = 16, maxSeqLength = 150):
        #decodingLayer(compressedLength = 16, maxSeqLength = 150).printParams()
        summary(self, (compressedLength,))
    
    
    
    
    
    