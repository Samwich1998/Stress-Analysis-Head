# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import time

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# Import helper models
import _convolutionalLinearLayer

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class autoencoderParameters(nn.Module):
    def __init__(self, sequenceLength = 240, compressedLength = 64):
        super(autoencoderParameters, self).__init__()
        # General shape parameters.
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength
        
        # Layern dimensions.
        # self.firstCompressionDim = sequenceLength
        # self.secondCompressionDim = int(sequenceLength/2) # 120
        # self.thirdCompressionDim = int(sequenceLength/4) # 60
        
        self.firstCompressionDim = 180
        self.secondCompressionDim = 140
        self.thirdCompressionDim = 100
    
    def splitPooling(self, inputData, poolingLayers):
        """
        Applies different pooling layers to different channel splits of the input data.
        
        Parameters:
        inputData (torch.Tensor): The input data tensor of shape (batchSize, numChannels, sequenceLength).
        poolingLayers (list of nn.Module): A list of pooling layer objects to be applied to the input data.
        
        Returns:
        torch.Tensor: Pooled output with the same number of channels as inputData.
        """
        # Specify the input shape and pooling splits.
        batchSize, numChannels, sequenceLength = inputData.size()
        poolingSplit = numChannels // len(poolingLayers)
        
        startChannel = 0
        compiledPoolingResults = []
        # For each provided pooling method.
        for poolingInd, poolingLayer in enumerate(poolingLayers):
            # Adjust the channel range for pooling
            endChannel = startChannel + poolingSplit
            # Pool all the channels in the end.
            if poolingInd == len(poolingLayers) - 1:
                endChannel = numChannels
            
            # Apply the pooling layer
            poolingChannels = inputData[:, startChannel:endChannel, :]
            pooledChannels = poolingLayer(poolingChannels)
            # Store the pooling results.
            compiledPoolingResults.append(pooledChannels)
            
            # Move onto the next channels.
            startChannel = endChannel
        
        # Concatenate the pooled outputs along the channel dimension
        pooledOutput = torch.cat(compiledPoolingResults, dim=1)
        return pooledOutput
    
    def minPooling(self, maxPooling):
        return lambda x: -maxPooling(-x)
        
class encodingLayer(autoencoderParameters):
    def __init__(self, sequenceLength, compressedLength):
        super(encodingLayer, self).__init__(sequenceLength, compressedLength)   
        # Autoencoder notes:
        #   padding: the number of added values around the image borders. padding = dilation * (kernel_size - 1) // 2
        #   dilation: the number of indices skipped between kernel points.
        #   kernel_size: the number of indices within the sliding filter.
        #   stride: the number of indices skipped when sliding the filter.
        
        # ------------------------- Pooling Layers ------------------------- # 
        
        # Max pooling layers.
        self.firstLayerMaxPooling = nn.AdaptiveMaxPool1d(self.firstCompressionDim)
        self.secondLayerMaxPooling = nn.AdaptiveMaxPool1d(self.secondCompressionDim)
        self.thirdLayerMaxPooling = nn.AdaptiveMaxPool1d(self.thirdCompressionDim)
        self.finalMaxPooling = nn.AdaptiveMaxPool1d(self.compressedLength)
        # Average pooling layers.
        self.firstLayerAvgPooling = nn.AdaptiveAvgPool1d(self.firstCompressionDim)
        self.secondLayerAvgPooling = nn.AdaptiveAvgPool1d(self.secondCompressionDim)
        self.thirdLayerAvgPooling = nn.AdaptiveAvgPool1d(self.thirdCompressionDim)
        self.finalAvgPooling = nn.AdaptiveAvgPool1d(self.compressedLength)
        # Min pooling layers.
        self.firstLayerMinPooling = self.minPooling(nn.AdaptiveMaxPool1d(self.firstCompressionDim))
        self.secondLayerMinPooling = self.minPooling(nn.AdaptiveMaxPool1d(self.secondCompressionDim))
        self.thirdLayerMinPooling = self.minPooling(nn.AdaptiveMaxPool1d(self.thirdCompressionDim))
        self.finalMinPooling = self.minPooling(nn.AdaptiveMaxPool1d(self.compressedLength))
        
        # Compile the pooling layers
        self.firstPoolingLayers = [self.firstLayerMaxPooling, self.firstLayerMinPooling, self.firstLayerAvgPooling]
        self.secondPoolingLayers = [self.secondLayerMaxPooling, self.secondLayerMinPooling, self.secondLayerAvgPooling]
        self.thirdPoolingLayers = [self.thirdLayerMaxPooling, self.thirdLayerMinPooling, self.thirdLayerAvgPooling]
        self.finalPoolingLayers = [self.finalMaxPooling, self.finalMinPooling, self.finalAvgPooling]
        
        # ------------------------ CNN Architecture ------------------------ # 

        # Expand the number of data channels.
        self.channelExpansion = nn.Sequential(           
            # Convolution architecture: Layer 1, Conv 1
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
        )

        # Compress and encode spatial features.
        self.compressSignalsCNN_1 = nn.Sequential(
            # Convolution architecture: Layer 2, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Compress and encode spatial features.
        self.compressSignalsCNN_compression_1 = nn.Sequential(
            # Convolution architecture: Layer 2, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=2, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Compress and encode spatial features.
        self.compressSignalsCNN_2 = nn.Sequential(  
            # Convolution architecture: Layer 3, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Compress and encode spatial features.
        self.compressSignalsCNN_compression_2 = nn.Sequential(
            # Convolution architecture: Layer 2, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=2, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Compress and encode spatial features.
        self.compressSignalsCNN_3 = nn.Sequential(   
            # Convolution architecture: Layer 4, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Compress and encode spatial features.
        self.compressSignalsCNN_compression_3 = nn.Sequential(
            # Convolution architecture: Layer 2, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=2, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        self.compressSignalsCNN_4 = nn.Sequential(   
            # Convolution architecture: Layer 5, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Compress and encode spatial features.
        self.compressSignalsCNN_compression_4 = nn.Sequential(
            # Convolution architecture: Layer 2, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=2, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        
        # Compress the number of data channels.
        self.channelPreCompression = nn.Sequential(           
            # Convolution architecture: Layer 5, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Compress the number of data channels.
        self.channelCompression = nn.Sequential(           
            # Convolution architecture: Layer 6, Conv 1
            nn.Conv1d(in_channels=6, out_channels=1, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
        )
        
        # ------------------------------------------------------------------ # 

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
        # compressedSignals_0 dimension: batchSize*numSignals, 6, sequenceLength

        # Apply the first CNN block to reduce spatial dimension.
        compressedSignals_1 = self.compressSignalsCNN_1(compressedSignals_0)
        compressedSignals_1 = self.compressSignalsCNN_1(compressedSignals_1)
        # Add a residual connection to prevent loss of information.
        compressedSignals_1 = compressedSignals_1 + compressedSignals_0
        # Apply a pooling layer to reduce the signal's dimension.
        compressedSignals_1 = self.splitPooling(compressedSignals_1, self.firstPoolingLayers)
        # compressedSignals_1 = self.compressSignalsCNN_compression_1(compressedSignals_1)
        # compressedSignals_1 dimension: batchSize*numSignals, 6, self.firstCompressionDim

        # Apply the second CNN block to reduce spatial dimension.
        compressedSignals_2 = self.compressSignalsCNN_2(compressedSignals_1)
        compressedSignals_2 = self.compressSignalsCNN_2(compressedSignals_2)
        # Add a residual connection to prevent loss of information.
        compressedSignals_2 = compressedSignals_2 + compressedSignals_1
        # Apply a pooling layer to reduce the signal's dimension.
        compressedSignals_2 = self.splitPooling(compressedSignals_2, self.secondPoolingLayers)
        # compressedSignals_2 = self.compressSignalsCNN_compression_2(compressedSignals_2)
        # compressedSignals_2 dimension: batchSize*numSignals, 6, self.secondCompressionDim
        
        # Apply the third CNN block to reduce spatial dimension.
        compressedSignals_3 = self.compressSignalsCNN_3(compressedSignals_2)
        compressedSignals_3 = self.compressSignalsCNN_3(compressedSignals_3)
        # Add a residual connection to prevent loss of information.
        compressedSignals_3 = compressedSignals_3 + compressedSignals_2
        # Apply a pooling layer to reduce the signal's dimension.
        compressedSignals_3 = self.splitPooling(compressedSignals_3, self.thirdPoolingLayers)
        # compressedSignals_3 = self.compressSignalsCNN_compression_3(compressedSignals_3)
        # compressedSignals_3 dimension: batchSize*numSignals, 6, thirdCompressionDim
        
        # Apply the fourth CNN block to reduce spatial dimension.
        compressedSignals_4 = self.compressSignalsCNN_4(compressedSignals_3)
        compressedSignals_4 = self.compressSignalsCNN_4(compressedSignals_4)
        # Add a residual connection to prevent loss of information.
        compressedSignals_4 = compressedSignals_4 + compressedSignals_3
        # Apply a pooling layer to reduce the signal's dimension.
        compressedSignals_4 = self.splitPooling(compressedSignals_4, self.finalPoolingLayers)
        # compressedSignals_4 = self.compressSignalsCNN_compression_4(compressedSignals_4)
        # compressedSignals_4 dimension: batchSize*numSignals, 6, compressedLength

        # Apply the fifth CNN block to reduce spatial dimension.
        compressedSignals_5 = self.channelPreCompression(compressedSignals_4)
        compressedSignals_5 = self.channelPreCompression(compressedSignals_5)
        # Add a residual connection to prevent loss of information.
        compressedSignals_5 = compressedSignals_5 + compressedSignals_4
        # Decrease the number of channels in the encocder.
        compressedSignals = self.channelCompression(compressedSignals_5)
        # compressedSignals dimension: batchSize*numSignals, 1, compressedLength
        
        # Seperate put each signal into its respective batch.
        compressedData = compressedSignals.view(batchSize, numSignals, self.compressedLength) 
        # compressedData dimension: batchSize, numSignals, self.compressedLength
        
        # ------------------------------------------------------------------ # 
        
        return compressedData
    
    def printParams(self, numSignals = 2):
        #encodingLayer(sequenceLength = 240, compressedLength = 64).printParams(numSignals = 2)
        t1 = time.time()
        summary(self, (numSignals, self.sequenceLength,)) # summary(model, inputShape)
        t2 = time.time()
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        print("Total time:", t2 - t1)
    
class decodingLayer(autoencoderParameters):
    def __init__(self, compressedLength, sequenceLength):
        super(decodingLayer, self).__init__(sequenceLength, compressedLength)
        
        # ------------------------- Pooling Layers ------------------------- # 
                
        # Upsampling layers.
        self.initialUpsample = nn.Upsample(size=self.sequenceLength, mode='linear', align_corners=True)        
        self.firstLayerUpsample = nn.Upsample(size=self.firstCompressionDim, mode='linear', align_corners=True)
        self.secondLayerUpsample = nn.Upsample(size=self.secondCompressionDim, mode='linear', align_corners=True)
        self.thirdLayerUpsample = nn.Upsample(size=self.thirdCompressionDim, mode='linear', align_corners=True)
        
        # ------------------------ CNN Architecture ------------------------ # 
        
        # Expand the number of data channels.
        self.channelExpansion = nn.Sequential(           
            # Convolution architecture: Layer 1, Conv 1
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
        )
        
        # Decode spatial features.
        self.expandSignalsCNN_1 = nn.Sequential(
            # Convolution architecture: Layer 2, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Upsampling features.
        self.expandSignalsCNN_expansion_1 = nn.Sequential(           
            # Convolution architecture: Layer 3, Conv 1-2
            nn.ConvTranspose1d(in_channels=6, out_channels=6, kernel_size=3, stride=2, padding=1, output_padding=1, groups=1, bias=True, dilation=1, padding_mode='zeros'),
            nn.SELU(),
        )
        
        # Decode spatial features.
        self.expandSignalsCNN_2 = nn.Sequential(           
            # Convolution architecture: Layer 3, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Upsampling features.
        self.expandSignalsCNN_expansion_2 = nn.Sequential(           
            # Convolution architecture: Layer 3, Conv 1-2
            nn.ConvTranspose1d(in_channels=6, out_channels=6, kernel_size=3, stride=2, padding=1, output_padding=1, groups=1, bias=True, dilation=1, padding_mode='zeros'),
            nn.SELU(),
        )
                
        # Decode spatial features.
        self.expandSignalsCNN_3 = nn.Sequential(           
            # Convolution architecture: Layer 4, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Upsampling features.
        self.expandSignalsCNN_expansion_3 = nn.Sequential(           
            # Convolution architecture: Layer 3, Conv 1-2
            nn.ConvTranspose1d(in_channels=6, out_channels=6, kernel_size=3, stride=2, padding=1, output_padding=1, groups=1, bias=True, dilation=1, padding_mode='zeros'),
            nn.SELU(),
        )
        
        # Decode spatial features.
        self.expandSignalsCNN_4 = nn.Sequential(           
            # Convolution architecture: Layer 5, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Expand the number of data channels.
        self.channelPreCompression = nn.Sequential(           
            # Convolution architecture: Layer 2, Conv 1-2
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Compress the number of data channels.
        self.channelCompression = nn.Sequential(   
            # Convolution architecture: Layer 6, Conv 1
            nn.Conv1d(in_channels=6, out_channels=1, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
        )
        
        # ------------------------ ANN Architecture ------------------------ # 
        
        # # Compress the number of data points.
        # self.decompressSignalsANN_1 = nn.Sequential( 
        #     # Neural architecture: Layer 1.
        #     nn.Linear(self.compressedLength, self.compressedLength, bias = True),
        #     nn.SELU(),
        # )
        
        # self.decompressSignalsANN_LinearConv = _convolutionalLinearLayer.convolutionalLinearLayer(
        #         initialDimension = self.sequenceLength,
        #         compressedDim = self.sequenceLength,
        #         compressedEmbeddedDim = 8,
        #         embeddingStride = 1,
        #         embeddedDim = 8,
        # )
        
        # # Compress the number of data points.
        # self.convolutionalLinearLayer = nn.Sequential( 
        #     # Neural architecture: Layer 1.
        #     nn.Linear(self.decompressSignalsANN_LinearConv.embeddedDim, self.decompressSignalsANN_LinearConv.compressedEmbeddedDim, bias = True),
        #     nn.SELU(),
        # )
        
        # ------------------------------------------------------------------ # 
                
    def forward(self, compressedData):
        """ The shape of compressedData: (batchSize, numSignals, compressedLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, compressedLength = compressedData.size()
        assert self.compressedLength == compressedLength
        
        # ------------------------ ANN Architecture ------------------------ # 
        
        # compressedData = compressedData.view(batchSize*numSignals, self.compressedLength) 
        # compressedData = self.decompressSignalsANN_LinearConv(compressedData, self.convolutionalLinearLayer)
        # compressedSignals dimension: batchSize*numSignals, self.compressedLength


        # Synthesize the information in the embedded space.
        # compressedData = self.decompressSignalsANN_1(compressedData)
        # compressedSignals dimension: batchSize, numSignals, self.compressedLength
        
        # Reshape the signals.
        compressedSignals = compressedData.view(batchSize*numSignals, 1, self.compressedLength) 
        # compressedSignals dimension: batchSize*numSignals, 1, self.compressedLength

        # ------------------------ CNN Architecture ------------------------ # 

        # Increase the number of channels in the encocder.
        decompressedSignals_0 = self.channelExpansion(compressedSignals)
        # Add a residual connection to prevent loss of information.
        decompressedSignals_0 = decompressedSignals_0 + compressedSignals
        # compressedSignals_0 dimension: batchSize*numSignals, 6, self.compressedLength
        
        # Apply a upsample layer to increase the signal's dimension.
        decompressedSignals_10 = self.thirdLayerUpsample(decompressedSignals_0)
        # decompressedSignals_10 = self.expandSignalsCNN_expansion_1(decompressedSignals_0)
        # Apply the first CNN block to increase the spatial dimension.
        decompressedSignals_1 = self.expandSignalsCNN_1(decompressedSignals_10)
        decompressedSignals_1 = self.expandSignalsCNN_1(decompressedSignals_1)
        # Add a residual connection to prevent loss of information.
        decompressedSignals_1 = decompressedSignals_1 + decompressedSignals_10
        # decompressedSignals_1 dimension: batchSize*numSignals, 6, self.thirdCompressionDim

        # Apply a upsample layer to increase the signal's dimension.
        decompressedSignals_20 = self.secondLayerUpsample(decompressedSignals_1)
        # decompressedSignals_20 = self.expandSignalsCNN_expansion_2(decompressedSignals_1)
        # decompressedSignals_20 = decompressedSignals_1
        # Apply the second CNN block to increase the spatial dimension.
        decompressedSignals_2 = self.expandSignalsCNN_2(decompressedSignals_20)
        decompressedSignals_2 = self.expandSignalsCNN_2(decompressedSignals_2)
        # Add a residual connection to prevent loss of information.
        decompressedSignals_2 = decompressedSignals_2 + decompressedSignals_20
        # decompressedSignals_2 dimension: batchSize*numSignals, 6, self.secondCompressionDim
        
        # Apply a upsample layer to increase the signal's dimension.
        decompressedSignals_30 = self.firstLayerUpsample(decompressedSignals_2)
        # decompressedSignals_30 = self.expandSignalsCNN_expansion_3(decompressedSignals_2)
        # Apply the third CNN block to increase the spatial dimension.
        decompressedSignals_3 = self.expandSignalsCNN_3(decompressedSignals_30)
        decompressedSignals_3 = self.expandSignalsCNN_3(decompressedSignals_3)
        # Add a residual connection to prevent loss of information.
        decompressedSignals_3 = decompressedSignals_3 + decompressedSignals_30
        # decompressedSignals_3 dimension: batchSize*numSignals, 6, self.firstCompressionDim

        # Apply a upsample layer to increase the signal's dimension.
        decompressedSignals_40 = self.initialUpsample(decompressedSignals_3)
        # decompressedSignals_40 = decompressedSignals_3
        # Apply the fourth CNN block to increase the spatial dimension.
        decompressedSignals_4 = self.expandSignalsCNN_4(decompressedSignals_40)
        decompressedSignals_4 = self.expandSignalsCNN_4(decompressedSignals_4)
        # Add a residual connection to prevent loss of information.
        decompressedSignals_4 = decompressedSignals_4 + decompressedSignals_40
        # decompressedSignals_4 dimension: batchSize*numSignals, 6, sequenceLength

        # Apply the pre=expansion CNN block to increase the spatial dimension.
        decompressedSignals_5 = self.channelPreCompression(decompressedSignals_4)
        decompressedSignals_5 = self.channelPreCompression(decompressedSignals_5)
        # Decrease the number of channels in the encocder.
        decompressedSignals = self.channelCompression(decompressedSignals_5)
        # decompressedSignals dimension: batchSize*numSignals, 1, sequenceLength
        
        # ------------------------ ANN Architecture ------------------------ # 

       # Organize the signals into the original batches.
        # reconstructedData = decompressedSignals.view(batchSize, numSignals, self.sequenceLength)
        # reconstructedData dimension: batchSize, numSignals, sequenceLength
        
        
        # decompressedSignals = decompressedSignals.view(batchSize*numSignals, self.sequenceLength) 
        # decompressedSignals = decompressedSignals + self.decompressSignalsANN_LinearConv(decompressedSignals, self.convolutionalLinearLayer)
        # # compressedSignals dimension: batchSize*numSignals, self.sequenceLength
        
        # Organize the signals into the original batches.
        reconstructedData = decompressedSignals.view(batchSize, numSignals, self.sequenceLength)
        # compressedSignals dimension: batchSize, numSignals, self.sequenceLength
        
        # ------------------------------------------------------------------ # 

        return reconstructedData
    
    def printParams(self, numSignals = 2):
        #decodingLayer(compressedLength = 64, sequenceLength = 240).printParams(numSignals = 2)
        summary(self, (numSignals, self.compressedLength,))
    
    
    
    
    
    