# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class encodingLayer(nn.Module):
    def __init__(self, sequenceLength, compressedLength):
        super(encodingLayer, self).__init__()
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength
        
        # Encode spatial features.
        self.compressSignals = nn.Sequential(
            # Convolution architecture: Layer 1
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, dilation = 1, padding=1),
            nn.BatchNorm1d(4, affine = True, momentum = 0.1, track_running_stats = True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        
            # Convolution architecture: Layer 2
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, stride=1, dilation = 1, padding=1),
            nn.BatchNorm1d(2, affine = True, momentum = 0.1, track_running_stats = True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        
            # Convolution architecture: Layer 3
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, dilation = 1, padding=1),
            nn.GELU(),
        )
        
        # Pass fake data through the CNN to calculate the final shape
        finalEncodedImageShape = self.compressSignals(torch.ones((1, 1, self.sequenceLength))).shape
        compressedDim = torch.tensor(finalEncodedImageShape[1:]).prod().item()
        
        # Learn final features
        self.applyNonLinearity = nn.Sequential(                        
            # Neural architecture: Layer 1
            nn.Linear(compressedDim, 64, bias = True),
            nn.BatchNorm1d(64, affine = True, momentum = 0.1, track_running_stats = True),
            nn.GELU(),
            
            # Neural architecture: Layer 3
            nn.Linear(64, self.compressedLength, bias = True),
            nn.BatchNorm1d(self.compressedLength, affine = True, momentum = 0.1, track_running_stats = True),
        )

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert self.sequenceLength == sequenceLength
        
        # Reshape the data to the expected input into the CNN architecture.
        signalData = inputData.view(batchSize * numSignals, sequenceLength) # Seperate out indivisual signals.
        signalData = signalData.unsqueeze(1) # Add one channel to the signal.
        # signalData dimension: batchSize*numSignals, 1, sequenceLength
        
        # Apply CNN architecture to reduce spatial dimension.
        compressedSignals = self.compressSignals(signalData) # The new dimension: batchSize*numSignals, numChannels = 1, imageLength
        compressedSignalsFlattened = compressedSignals.view(batchSize * numSignals, -1) # Flatten out the CNN output.
        # compressedSignals dimension: batchSize*numSignals, numChannels*imageLength -> batchSize*numSignals, compressedDim
        
        # Apply the ANN architecture to add non-linearity.
        compressedSignals = self.applyNonLinearity(compressedSignalsFlattened)  # The new dimension: batchSize*numSignals, compressedLength
        compressedData = compressedSignals.view(batchSize, numSignals, self.compressedLength) # Seperate put each signal into its respective batch.
        # compressedData dimension: batchSize, numSignals, self.compressedLength
                
        return compressedData
    
    def printParams(self, numSignals = 50):
        #encodingLayer(sequenceLength = 240, compressedLength = 32).printParams(numSignals = 75)
        summary(self, (numSignals, self.sequenceLength,)) # summary(model, inputShape)
        
    
class decodingLayer(nn.Module):
    def __init__(self, compressedLength = 32, sequenceLength = 240):
        super(decodingLayer, self).__init__()
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength
                
        finalLinearDim = 64
        # Learn final features
        self.applyNonLinearity = nn.Sequential(                        
            # Neural architecture: Layer 1
            nn.Linear(compressedLength, 32, bias = True),
            nn.BatchNorm1d(32, affine = True, momentum = 0.1, track_running_stats = True),
            nn.GELU(),
            
            # Neural architecture: Layer 2
            nn.Linear(32, finalLinearDim, bias = True),
            # nn.BatchNorm1d(finalLinearDim, affine = True, momentum = 0.1, track_running_stats = True),
            nn.GELU(),
        )
        
        # Decode spatial features.
        self.expandSignals = nn.Sequential(
            # Deconvolution architecture: Layer 1
            nn.ConvTranspose1d(in_channels=1, out_channels=2, kernel_size=2, stride=2, dilation = 1, padding=0),
            nn.BatchNorm1d(2, affine = True, momentum = 0.1, track_running_stats = True),
            nn.GELU(),

            # Deconvolution architecture: Layer 2
            nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=2, stride=2, dilation = 1, padding=0),
            nn.GELU(),
        )
        
        # Find the final dimension of the decompressed image.
        finalDeconvShape = self.expandSignals(torch.ones((1, 1, finalLinearDim))).shape
        finalDeconvDimension = torch.tensor(finalDeconvShape[1:]).prod().item()

        # Fully Connected layers for final signal reconstruction
        self.flattenToSequence = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(finalDeconvDimension, sequenceLength, bias = True),
            # nn.BatchNorm1d(256, affine = True, momentum = 0.1, track_running_stats = True),
            # nn.GELU(),
            
            # # Neural architecture: Layer 2
            # nn.Linear(256, sequenceLength, bias=True),
        )
                
    def forward(self, compressedData):
        """ The shape of compressedData: (batchSize, numSignals, compressedLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, compressedLength = compressedData.size()
        assert self.compressedLength == compressedLength
        
        # Reconstruct a compressed signal.
        compressedSignals = compressedData.view(batchSize*numSignals, self.compressedLength) # Seperate put each signal into its respective batch.
        compressedSignals = self.applyNonLinearity(compressedSignals)
        # compressedSignals dimension: batchSize*numSignals, finalLinearDim

        # Apply a CNN deconvolution for a smoothed upscale.
        compressedSignals = compressedSignals.unsqueeze(1) # compressedSignals dimension: batchSize*numSignals, 1, finalLinearDim
        decompressedSignals = self.expandSignals(compressedSignals)
        # decompressedSignals dimension: batchSize*numSignals, numChannels = 1, imageLength
        
        # Match the original sequence length.
        decompressedSignals = decompressedSignals.view(batchSize*numSignals, -1) # decompressedSignals dimension: batchSize*numSignals, numChannels*imageLength
        decompressedSignals = self.flattenToSequence(decompressedSignals)
        # decompressedSignals dimension: batchSize*numSignals, sequenceLength
        
        # Reconstruct the original signals.
        reconstructedData = decompressedSignals.view(batchSize, numSignals, self.sequenceLength)   # Organize the signals into the original batches.
        # reconstructedData dimension: batchSize, numSignals, sequenceLength

        return reconstructedData
    
    def printParams(self, numSignals = 75, compressedLength = 32, sequenceLength = 240):
        #decodingLayer(compressedLength = 32, sequenceLength = 240).printParams()
        summary(self, (numSignals, compressedLength,))
    
    
    
    
    
    