# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
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
        
        # Layern dimensions.
        self.firstCompressionDim = 200
        self.secondCompressionDim = 100
        
        # Normalization layers
        self.initialStandardization = nn.LayerNorm(self.sequenceLength, eps = 1E-10, elementwise_affine=False)
        self.firstLayerStandardization = nn.LayerNorm(self.firstCompressionDim, eps = 1E-10, elementwise_affine=False)
        self.secondLayerStandardization = nn.LayerNorm(self.secondCompressionDim, eps = 1E-10, elementwise_affine=False)
        self.finalStandardization = nn.LayerNorm(self.compressedLength, eps = 1E-10, elementwise_affine=False)
        
class encodingLayer(autoencoderParameters):
    def __init__(self, sequenceLength, compressedLength):
        super(encodingLayer, self).__init__(sequenceLength, compressedLength)   
        # Autoencoder notes:
        #   padding: the number of added values around the image borders. padding = dilation * (kernel_size - 1) // 2
        #   dilation: the number of indices skipped between kernel points.
        #   kernel_size: the number of indices within the sliding filter.
        #   stride: the number of indices skipped when sliding the filter.
        
        # Pooling layers.
        self.initialPooling = nn.AdaptiveAvgPool1d(self.sequenceLength)
        self.firstLayerPooling = nn.AdaptiveAvgPool1d(self.firstCompressionDim)
        self.secondLayerPooling = nn.AdaptiveAvgPool1d(self.secondCompressionDim)
        self.finalPooling = nn.AdaptiveAvgPool1d(self.compressedLength)
        
        # Expand the number of data channels.
        self.channelExpansion = nn.Sequential(           
            # Convolution architecture: Layer 3, Conv 1
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
        )

        # Encode spatial features.
        self.compressSignalsCNN_1 = nn.Sequential(
            # Convolution architecture: Layer 1, Conv 1-3
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 3, padding=3, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        self.compressSignalsCNN_2 = nn.Sequential(  
            # Convolution architecture: Layer 1, Conv 1
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 3, padding=3, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        self.compressSignalsCNN_3 = nn.Sequential(   
            # Convolution architecture: Layer 1, Conv 1-3
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 3, padding=3, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Compress the number of data channels.
        self.channelCompression = nn.Sequential(           
            # Convolution architecture: Layer 3, Conv 1
            nn.Conv1d(in_channels=6, out_channels=1, kernel_size=3, stride=1, dilation = 1, padding=1, bias=True),
        )

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert self.sequenceLength == sequenceLength
        
        # Reshape the data to the expected input into the CNN architecture.
        signalData = inputData.view(batchSize * numSignals, 1, sequenceLength) # Seperate out indivisual signals.
        # signalData dimension: batchSize*numSignals, 1, sequenceLength
        
        # ------------------------ CNN Architecture ------------------------ # 
        
        # Increase the number of channels in the encocder.
        compressedSignals_0 = self.channelExpansion(signalData)
        # Add a residual connection to prevent loss of information.
        compressedSignals_0 = compressedSignals_0 + signalData
        # compressedSignals_0 dimension: batchSize*numSignals, 8, sequenceLength

        # Apply the first CNN block to reduce spatial dimension.
        compressedSignals_1 = self.compressSignalsCNN_1(compressedSignals_0)
        compressedSignals_1 = self.compressSignalsCNN_1(compressedSignals_1)
        compressedSignals_1 = self.compressSignalsCNN_1(compressedSignals_1)
        # Add a residual connection to prevent loss of information.
        compressedSignals_1 = compressedSignals_1 + signalData
        # Apply a pooling layer to reduce the signal's dimension.
        compressedSignals_1 = self.firstLayerPooling(compressedSignals_1)
        # compressedSignals_1 dimension: batchSize*numSignals, 8, self.firstCompressionDim
        
        # Apply the second CNN block to reduce spatial dimension.
        compressedSignals_2 = self.compressSignalsCNN_2(compressedSignals_1)
        compressedSignals_2 = self.compressSignalsCNN_2(compressedSignals_2)
        compressedSignals_2 = self.compressSignalsCNN_2(compressedSignals_2)
        # Add a residual connection to prevent loss of information.
        compressedSignals_2 = compressedSignals_2 + compressedSignals_1
        # Apply a pooling layer to reduce the signal's dimension.
        compressedSignals_2 = self.secondLayerPooling(compressedSignals_2)
        # Standardize the data to prevent a covariant shift.
        # compressedSignals_2 = self.secondLayerStandardization(compressedSignals_2)
        # compressedSignals_2 dimension: batchSize*numSignals, 8, self.secondCompressionDim

        # Apply the third CNN block to reduce spatial dimension.
        compressedSignals_3 = self.compressSignalsCNN_3(compressedSignals_2)
        compressedSignals_3 = self.compressSignalsCNN_3(compressedSignals_3)
        compressedSignals_3 = self.compressSignalsCNN_3(compressedSignals_3)
        # Add a residual connection to prevent loss of information.
        compressedSignals_3 = compressedSignals_3 + compressedSignals_2
        # Apply a pooling layer to reduce the signal's dimension.
        compressedSignals_3 = self.finalPooling(compressedSignals_3)
        # compressedSignals_3 dimension: batchSize*numSignals, 8, compressedLength
        
        # Decrease the number of channels in the encocder.
        compressedSignals_4 = self.channelCompression(compressedSignals_3)
        # Standardize the data to prevent a covariant shift.
        compressedSignals = self.finalStandardization(compressedSignals_4)
        # compressedSignals dimension: batchSize*numSignals, 1, compressedLength
        
        # ------------------------------------------------------------------ # 
        
        # Seperate put each signal into its respective batch.
        compressedData = compressedSignals.view(batchSize, numSignals, self.compressedLength) 
        # compressedData dimension: batchSize, numSignals, self.compressedLength

        return compressedData
    
    def printParams(self, numSignals = 50):
        #encodingLayer(sequenceLength = 300, compressedLength = 64).printParams(numSignals = 50)
        summary(self, (numSignals, self.sequenceLength,)) # summary(model, inputShape)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
    
class decodingLayer(autoencoderParameters):
    def __init__(self, compressedLength, sequenceLength):
        super(decodingLayer, self).__init__(sequenceLength, compressedLength)
        
        nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
        
        # Upsampling layers.
        self.initialPooling = nn.Upsample(size=self.sequenceLength, mode='linear', align_corners=True)        
        self.firstLayerPooling = nn.Upsample(size=self.firstCompressionDim, mode='linear', align_corners=True)
        self.secondLayerPooling = nn.Upsample(size=self.secondCompressionDim, mode='linear', align_corners=True)
        self.finalPooling = nn.Upsample(size=self.compressedLength, mode='linear', align_corners=True)
        
        # Expand the number of data channels.
        self.channelExpansion = nn.Sequential(           
            # Convolution architecture: Layer 3, Conv 1
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
        )
        
        # Decode spatial features.
        self.expandSignalsCNN_1 = nn.Sequential(
            # Convolution architecture: Layer 1, Conv 1
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 3, padding=3, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Decode spatial features.
        self.expandSignalsCNN_2 = nn.Sequential(           
            # Convolution architecture: Layer 1, Conv 1
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 3, padding=3, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Decode spatial features.
        self.expandSignalsCNN_3 = nn.Sequential(           
            # Convolution architecture: Layer 1, Conv 1
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 3, padding=3, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Compress the number of data channels.
        self.channelCompression = nn.Sequential(   
            # Convolution architecture: Layer 3, Conv 1
            nn.Conv1d(in_channels=6, out_channels=1, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
        )
                
    def forward(self, compressedData):
        """ The shape of compressedData: (batchSize, numSignals, compressedLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, compressedLength = compressedData.size()
        assert self.compressedLength == compressedLength
        
        # Expand the compressed signals through an FC network.
        compressedSignals = compressedData.view(batchSize*numSignals, 1, self.compressedLength) # Seperate put each signal into its respective batch.
        # compressedSignals dimension: batchSize*numSignals, compressedLength
        
        # ------------------------ CNN Architecture ------------------------ # 
        
        # Increase the number of channels in the encocder.
        compressedSignals_0 = self.channelExpansion(compressedSignals)
        # Add a residual connection to prevent loss of information.
        compressedSignals_0 = compressedSignals_0 + compressedSignals
        # compressedSignals_0 dimension: batchSize*numSignals, 8, compressedLength

        # Apply the first CNN block to increase the spatial dimension.
        compressedSignals_1 = self.expandSignalsCNN_1(compressedSignals_0)
        compressedSignals_1 = self.expandSignalsCNN_1(compressedSignals_1)
        compressedSignals_1 = self.expandSignalsCNN_1(compressedSignals_1)
        # Add a residual connection to prevent loss of information.
        compressedSignals_1 = compressedSignals_1 + compressedSignals_0
        # Apply a pooling layer to increase the signal's dimension.
        compressedSignals_1 = self.secondLayerPooling(compressedSignals_1)
        # compressedSignals_1 dimension: batchSize*numSignals, 8, self.secondCompressionDim
        
        # Apply the second CNN block to increase the spatial dimension.
        compressedSignals_2 = self.expandSignalsCNN_2(compressedSignals_1)
        compressedSignals_2 = self.expandSignalsCNN_2(compressedSignals_2)
        compressedSignals_2 = self.expandSignalsCNN_2(compressedSignals_2)
        # Add a residual connection to prevent loss of information.
        compressedSignals_2 = compressedSignals_2 + compressedSignals_1
        # Apply a pooling layer to increase the signal's dimension.
        compressedSignals_2 = self.firstLayerPooling(compressedSignals_2)
        # Standardize the data to prevent a covariant shift.
        # compressedSignals_2 = self.firstLayerStandardization(compressedSignals_2)
        # compressedSignals_2 dimension: batchSize*numSignals, 8, self.firstCompressionDim

        # Apply the third CNN block to increase the spatial dimension.
        compressedSignals_3 = self.expandSignalsCNN_3(compressedSignals_2)
        compressedSignals_3 = self.expandSignalsCNN_3(compressedSignals_3)
        compressedSignals_3 = self.expandSignalsCNN_3(compressedSignals_3)
        # Add a residual connection to prevent loss of information.
        compressedSignals_3 = compressedSignals_3 + compressedSignals_2
        # Apply a pooling layer to increase the signal's dimension.
        compressedSignals_3 = self.initialPooling(compressedSignals_3)
        # compressedSignals_3 dimension: batchSize*numSignals, 8, sequenceLength

        # Decrease the number of channels in the encocder.
        decompressedSignals = self.channelCompression(compressedSignals_3)
        # decompressedSignals dimension: batchSize*numSignals, 1, sequenceLength
        
        # ------------------------------------------------------------------ # 

        # Reconstruct the original signals.
        reconstructedData = decompressedSignals.view(batchSize, numSignals, self.sequenceLength)   # Organize the signals into the original batches.
        # reconstructedData dimension: batchSize, numSignals, sequenceLength

        return reconstructedData
    
    def printParams(self, numSignals = 50):
        #decodingLayer(compressedLength = 64, sequenceLength = 300).printParams()
        summary(self, (numSignals, self.compressedLength,))
    
    
    
    
    
    