# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class autoencoderParameters(nn.Module):
    def __init__(self, sequenceLength, compressedLength):
        super(autoencoderParameters, self).__init__()
        # General shape parameters.
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength
        
        # Shape parameters.
        self.compressionLengthCNN = int(sequenceLength/12)
        self.numOutputChannels = 8
        
class encodingLayer(autoencoderParameters):
    def __init__(self, sequenceLength, compressedLength):
        super(encodingLayer, self).__init__(sequenceLength, compressedLength)     
        
        # Pooling layer.
        self.avPooling_Stride2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.avPooling_Stride3 = nn.AvgPool1d(kernel_size=3, stride=3, padding=1)
        
        # Encode spatial features.
        self.compressSignalsCNN_1 = nn.Sequential(
            # Convolution architecture: Layer 1, Conv 1
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 1, Conv 2
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 1, Conv 2
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1),
            # nn.GELU(),
        )
        
        self.compressSignalsCNN_2 = nn.Sequential(           
            # Convolution architecture: Layer 2, Conv 1
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 2, Conv 2
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 2, Conv 2
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1),
            # nn.GELU(),
        )
        
        self.compressSignalsCNN_3 = nn.Sequential(           
            # Convolution architecture: Layer 3, Conv 1
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 3, Conv 1
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 3, Conv 2
            nn.Conv1d(in_channels=8, out_channels=self.numOutputChannels, kernel_size=3, stride=1, dilation = 1, padding=1),
            # nn.GELU(),
        )
        
        # Pass fake data through the CNN to calculate the final shape
        fakeData = torch.ones((1, 1, self.sequenceLength))
        fakeData = self.compressSignalsCNN_1(fakeData)
        fakeData = self.avPooling_Stride3(fakeData)
        fakeData = self.compressSignalsCNN_2(fakeData)
        fakeData = self.avPooling_Stride2(fakeData)
        fakeData = self.compressSignalsCNN_3(fakeData)
        fakeData = self.avPooling_Stride2(fakeData)
        # Check the final dimensions of the data
        compressedDim = torch.tensor(fakeData.shape[1:]).prod().item()
        assert self.compressionLengthCNN*self.numOutputChannels == compressedDim, f"compressedDim: {compressedDim}, self.compressionLengthCNN: {self.compressionLengthCNN}"
        
        # Learn final features
        self.compressSignalsFC = nn.Sequential(                        
            # Neural architecture: Layer 1
            nn.Linear(compressedDim, self.compressedLength, bias = True),
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
        compressedSignals = signalData + self.compressSignalsCNN_1(signalData)
        compressedSignals = self.avPooling_Stride3(compressedSignals)
        compressedSignals = compressedSignals + self.compressSignalsCNN_2(compressedSignals)
        compressedSignals = self.avPooling_Stride2(compressedSignals)
        compressedSignals = compressedSignals + self.compressSignalsCNN_3(compressedSignals)
        compressedSignals = self.avPooling_Stride2(compressedSignals)
        # compressedSignals dimension: batchSize*numSignals, numChannels, imageLength
        
        # Apply the ANN architecture to add non-linearity.
        compressedSignalsFlattened = compressedSignals.view(batchSize * numSignals, -1) # Flatten out the CNN output.
        compressedSignals = self.compressSignalsFC(compressedSignalsFlattened)  # The new dimension: batchSize*numSignals, compressedLength
        compressedData = compressedSignals.view(batchSize, numSignals, self.compressedLength) # Seperate put each signal into its respective batch.
        # compressedData dimension: batchSize, numSignals, self.compressedLength
                
        return compressedData
    
    def printParams(self, numSignals = 75):
        #encodingLayer(sequenceLength = 300, compressedLength = 64).printParams(numSignals = 75)
        summary(self, (numSignals, self.sequenceLength,)) # summary(model, inputShape)
        
    
class decodingLayer(autoencoderParameters):
    def __init__(self, compressedLength, sequenceLength):
        super(decodingLayer, self).__init__(sequenceLength, compressedLength)

        # Decode spatial features.
        self.expandSignalsFC = nn.Sequential(                        
            # Neural architecture: Layer 1
            nn.Linear(self.compressedLength, self.compressionLengthCNN*self.numOutputChannels, bias = True),
            # nn.BatchNorm1d(self.compressionLengthCNN*self.numOutputChannels, affine = True, momentum = 0.1, track_running_stats = True),
            # nn.GELU(),
        )
        
        # Pooling layer.
        self.upPooling_Stride2 = nn.Upsample(scale_factor=2, mode='linear')
        self.upPooling_Stride3 = nn.Upsample(scale_factor=3, mode='linear')
        
        # Decode spatial features.
        self.expandSignalsCNN_1 = nn.Sequential(
            # Convolution architecture: Layer 1, Conv 1 (reverse of encoder's Layer 3, Conv 2)
            nn.ConvTranspose1d(in_channels=self.numOutputChannels, out_channels=8, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 1, Conv 1 (reverse of encoder's Layer 3, Conv 2)
            nn.ConvTranspose1d(in_channels=self.numOutputChannels, out_channels=8, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 1, Conv 2 (reverse of encoder's Layer 3, Conv 1)
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.GELU(),
        )
        
        # Decode spatial features.
        self.expandSignalsCNN_2 = nn.Sequential(           
            # Convolution architecture: Layer 2, Conv 1 (reverse of encoder's Layer 2, Conv 2)
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 2, Conv 1 (reverse of encoder's Layer 2, Conv 2)
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 2, Conv 2 (reverse of encoder's Layer 2, Conv 1)
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.GELU(),
        )
        
        # Decode spatial features.
        self.expandSignalsCNN_3 = nn.Sequential(           
            # Convolution architecture: Layer 3, Conv 1 (reverse of encoder's Layer 1, Conv 2)
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 3, Conv 1 (reverse of encoder's Layer 1, Conv 2)
            nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.GELU(),
            # Convolution architecture: Layer 3, Conv 2 (reverse of encoder's Layer 1, Conv 1)
            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=3, stride=1, dilation=1, padding=1),
            # nn.GELU()
        )
                
        # Find the final dimension of the decompressed image.
        # finalDeconvShape = self.expandSignalsCNN(torch.ones((1, self.numOutputChannels, self.compressionLengthCNN))).shape
        # finalDeconvDimension = torch.tensor(finalDeconvShape[1:]).prod().item()
        # # Assert the correct shape of the expanded image.
        # assert self.sequenceLength == finalDeconvDimension, f"sequenceLength: {sequenceLength}; finalDeconvDimension: {finalDeconvDimension}; self.numOutputChannels: {self.numOutputChannels}"
                
    def forward(self, compressedData):
        """ The shape of compressedData: (batchSize, numSignals, compressedLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, compressedLength = compressedData.size()
        assert self.compressedLength == compressedLength
        
        # Expand the compressed signals through an FC network.
        compressedSignals = compressedData.view(batchSize*numSignals, self.compressedLength) # Seperate put each signal into its respective batch.
        compressedSignals = self.expandSignalsFC(compressedSignals)
        # compressedSignals dimension: batchSize*numSignals, compressionLengthCNN
        
        # Reshape for CNN.
        decompressedSignals = compressedSignals.view(batchSize*numSignals, self.numOutputChannels, self.compressionLengthCNN)

        # Apply a CNN deconvolution for a smoothed upscale.
        decompressedSignals = self.upPooling_Stride2(decompressedSignals)
        decompressedSignals = decompressedSignals + self.expandSignalsCNN_1(decompressedSignals)
        decompressedSignals = self.upPooling_Stride2(decompressedSignals)
        decompressedSignals = decompressedSignals + self.expandSignalsCNN_2(decompressedSignals)
        decompressedSignals = self.upPooling_Stride3(decompressedSignals)
        decompressedSignals = self.expandSignalsCNN_3(decompressedSignals)

        # decompressedSignals dimension: batchSize*numSignals, numChannels = 1, self.sequenceLength

        # Reconstruct the original signals.
        reconstructedData = decompressedSignals.view(batchSize, numSignals, self.sequenceLength)   # Organize the signals into the original batches.
        # reconstructedData dimension: batchSize, numSignals, sequenceLength

        return reconstructedData
    
    def printParams(self, numSignals = 75):
        #decodingLayer(compressedLength = 64, sequenceLength = 300).printParams()
        summary(self, (numSignals, self.compressedLength,))
    
    
    
    
    
    