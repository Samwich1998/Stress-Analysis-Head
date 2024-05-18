# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import time
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import _convolutionalHelpers

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class encodingHead(nn.Module):
    def __init__(self, signalDimension = 16, numCondensedSignals = 16):
        super(encodingHead, self).__init__()
        # General shape parameters.
        self.numCondensedSignals = numCondensedSignals
        self.signalDimension = signalDimension
    
    def convolutionalFilter(self, numChannels = 9, kernel_sizes = [3, 3], dilations = [1, 2], groups = 1):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
            # Convolution architecture
            nn.Conv1d(in_channels=numChannels*groups, out_channels=numChannels*groups, kernel_size=kernel_sizes[0], stride=1, 
                      dilation = dilations[0], padding=paddings[0], padding_mode='reflect', groups=groups, bias=True),
            nn.Conv1d(in_channels=numChannels*groups, out_channels=numChannels*groups, kernel_size=kernel_sizes[1], stride=1, 
                      dilation = dilations[1], padding=paddings[1], padding_mode='reflect', groups=groups, bias=True),
            nn.SELU(),
        )
        
    def projectionArchitecture_2D(self, numChannels = [16, 8, 4, 1], kernel_sizes = [3, 3, 3, 3], dilations = [1, 2, 2, 1], groups = [1, 1, 1, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
            # Convolution architecture
            nn.Conv2d(in_channels=1, out_channels=numChannels[0], kernel_size=(1, kernel_sizes[0]), stride=1, 
                      dilation = (1, dilations[0]), padding=(0, paddings[0]), padding_mode='reflect', groups=groups[0], bias=True),
            nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(1, kernel_sizes[1]), stride=1, 
                      dilation = (1, dilations[1]), padding=(0, paddings[1]), padding_mode='reflect', groups=groups[1], bias=True),
            nn.SELU(),

            nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, kernel_sizes[2]), stride=1, 
                      dilation = (1, dilations[2]), padding=(0, paddings[2]), padding_mode='reflect', groups=groups[2], bias=True),
            nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[3], kernel_size=(1, kernel_sizes[3]), stride=1, 
                      dilation = (1, dilations[3]), padding=(0, paddings[3]), padding_mode='reflect', groups=groups[3], bias=True),
            nn.SELU(),
        )
        
    def projectionArchitecture_1D(self, numChannels = [8, 4, 1], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
            # Convolution architecture
            nn.Conv1d(in_channels=1, out_channels=numChannels[0], kernel_size=kernel_sizes[0], stride=1, 
                      dilation=dilations[0], padding=paddings[0], padding_mode='reflect', groups=groups[0], bias=True),
            nn.Conv1d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=kernel_sizes[1], stride=1, 
                      dilation=dilations[1], padding=paddings[1], padding_mode='reflect', groups=groups[1], bias=True),
            nn.SELU(),

            nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=kernel_sizes[2], stride=1, 
                      dilation=dilations[2], padding=paddings[2], padding_mode='reflect', groups=groups[2], bias=True),
            nn.Conv1d(in_channels=numChannels[2], out_channels=numChannels[3], kernel_size=kernel_sizes[3], stride=1, 
                      dilation=dilations[3], padding=paddings[3], padding_mode='reflect', groups=groups[3], bias=True),
            nn.SELU(),
        )  

    def projectionArchitecture_1D(self, numChannels = [8, 4, 1], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
            # Residual connection
            _convolutionalHelpers.ResNet(
                 module = nn.Sequential(
                    # Convolution architecture: channel expansion
                    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=1, bias=True),
                    nn.SELU(),
                    
                    _convolutionalHelpers.ResNet(
                         module = nn.Sequential(
                            # Convolution architecture: feature engineering
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=4, bias=True),
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=1, dilation=(1, 2), padding=(0, 2), padding_mode='reflect', groups=4, bias=True),
                            nn.SELU(),  
                            
                            # Convolution architecture: feature engineering
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=4, bias=True),
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=(2, 2), padding=(2, 2), padding_mode='reflect', groups=4, bias=True),
                            nn.SELU(),  
                    ), numCycles = 1),
                    
                    # Convolution architecture: channel compression
                    nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=1, bias=True),
                    nn.SELU(),
            ), numCycles = 1),
        )

# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(encodingHead):
    def __init__(self, signalDimension = 64, numCondensedSignals = 32, numDecoderLayers = 1, numHeads = 1, dropout_rate = 0.3):
        super(signalEncoding, self).__init__(signalDimension, numCondensedSignals)
        # Initialize transformer parameters.
        self.decoderHeadDim = signalDimension // numHeads    # The dimension of each decoder head (subset of signalDimension).
        self.numCondensedSignals = numCondensedSignals  # The number of times the latent channel is bigger than the value.
        self.numDecoderLayers = numDecoderLayers  # The number of times the data passes through the transformer block.
        self.dropout_rate = dropout_rate # Dropout rate for the feed forward layers.
        self.numHeads = numHeads    # The number of parallel heads for segmenting each embeddeding.
        
        # Assert proper transformer parameters.
        assert self.decoderHeadDim * numHeads == self.signalDimension, f"signalDimension must be divisible by numHeads: {self.decoderHeadDim}, {numHeads}, {self.signalDimension}"
        assert 0 < self.decoderHeadDim, f"The dimension of the transformer head cannot be less than or equal to zero: {self.decoderHeadDim}"
        
        # Define a trainable weight that is subject-specific.
        self.learnableStart = nn.Parameter(torch.randn(self.numCondensedSignals, self.signalDimension)) # The base state of the latent space.

        # Create model for transforming decoder parameters.
        self.decoderValueProjection = self.projectionArchitecture_2D(numChannels = [4, 8, 16, self.numCondensedSignals], kernel_sizes = [3, 3, 3, 3], dilations = [1, 2, 2, 1], groups = [1, 1, 2, 4])
        
        self.latentProjection = nn.Sequential(
            # Residual connection
            _convolutionalHelpers.ResNet(
                 module = nn.Sequential(
                    # Convolution architecture: channel expansion
                    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=1, bias=True),
                    nn.SELU(),
                    
                    _convolutionalHelpers.ResNet(
                         module = nn.Sequential(
                            # Convolution architecture: feature engineering
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=4, bias=True),
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=1, dilation=(1, 2), padding=(0, 2), padding_mode='reflect', groups=4, bias=True),
                            nn.SELU(),  
                            
                            # Convolution architecture: feature engineering
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=4, bias=True),
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=(2, 2), padding=(2, 2), padding_mode='reflect', groups=4, bias=True),
                            nn.SELU(),  
                    ), numCycles = 1),
                    
                    # Convolution architecture: channel compression
                    nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=1, bias=True),
                    nn.SELU(),
            ), numCycles = 1),
        )
        
        # Add non-linearity to attention.
        self.feedForwardDecoderLayers = nn.Sequential(
            # Residual connection
            _convolutionalHelpers.ResNet(
                 module = nn.Sequential(
                    # Convolution architecture: channel expansion
                    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=1, bias=True),
                    nn.SELU(),
                    
                    _convolutionalHelpers.ResNet(
                         module = nn.Sequential(
                            # Convolution architecture: feature engineering
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=4, bias=True),
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=1, dilation=(1, 2), padding=(0, 2), padding_mode='reflect', groups=4, bias=True),
                            nn.Dropout(self.dropout_rate),
                            nn.SELU(),  
                            
                            # Convolution architecture: feature engineering
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=(1, 1), padding=(1, 1), padding_mode='reflect', groups=4, bias=True),
                            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, dilation=(2, 2), padding=(2, 2), padding_mode='reflect', groups=4, bias=True),
                            nn.SELU(),  
                    ), numCycles = 1),
                    
                    # Convolution architecture: channel compression
                    nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=1, dilation=(1, 1), padding=(0, 1), padding_mode='reflect', groups=1, bias=True),
                    nn.SELU(),
            ), numCycles = 1),
        )

        # Initialize the layer normalization.
        self.layerNorm_CA = nn.ModuleList([nn.LayerNorm(self.signalDimension, eps=1E-10) for _ in range(self.numDecoderLayers)])
        self.layerNorm_LFF = nn.ModuleList([nn.LayerNorm(self.signalDimension, eps=1E-10) for _ in range(self.numDecoderLayers)])
                
        # Initialize holder parameters.
        self.allAttentionWeights = []  # Dimension: numEpochs, numLayers, attentionWeightDim, where attentionWeightDim = batch_size, numHeads, seq_length, seq_length

    def forward(self, inputData, decoderModel = None, allTrainingData = False, calculateLoss = False):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """
        # if allTrainingData: self.allAttentionWeights.append([])

        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.size()
        # Assert that we have the expected data format.
        assert signalDimension == self.signalDimension, \
            f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."
            
        if decoderModel == None:
            print("This should be a fake pass through signal combination.")
            decoderModel = signalDecoding(self.signalDimension, self.numCondensedSignals, numSignals)
        
        # ----------------------- Data Preprocessing ----------------------- # 

        # Initialize the transformer tensors.
        valueData = self.getValueData(inputData)
        latentData = self.learnableStart.expand(batchSize, -1, -1)
        # latentData dimension: batchSize, numCondensedSignals, signalDimension
        # valueData dimension: batchSize, numHeads, numSignals, numCondensedSignals, decoderHeadDim
        
        # ------------------------ Transformer Loop ------------------------ # 
        
        # For each decoding layer.
        for layerInd in range(self.numDecoderLayers):            
            # Combine the values from the cross attentuation block.
            latentData = self.crossAttentionBlock(valueData, latentData, inputData, decoderModel, layerInd, allTrainingData)
            # latentData dimension: batchSize, numCondensedSignals, signalDimension
            
            # Apply a non-linearity to the transformer.
            latentData = self.feedForwardDecoderBlock(latentData, layerInd) # Apply feed-forward block.
            # latentData dimension: batchSize, numCondensedSignals, signalDimension
                        
        # # Final latent projection.
        # latentData = latentData + self.finalProjection(latentData.unsqueeze(1)).squeeze(1)
        # # latentData dimension: batchSize, numCondensedSignals, signalDimension
                
        # ------------------------------------------------------------------ #   
        
        return latentData
    
    # ------------------------- Transformer Blocks ------------------------- #  

    def crossAttentionBlock(self, valueData, latentData, inputData, decoderModel, layerInd, allTrainingData):
        # Apply a normalization to each section of the latent data.
        normLatentData = self.layerNorm_CA[layerInd](latentData)
        # normLatentData dimension: batchSize, numCondensedSignals, signalDimension

        # Predict how well the data can be reconstructed.
        with torch.no_grad():
            predictedSignal = decoderModel(normLatentData)
        predictionError = (predictedSignal - inputData) ** 2
        # predictedSignal dimension: batchSize, numSignals, signalDimension
        # predictionError dimension: batchSize, numSignals, signalDimension
        
        # Apply cross attention block to the signals.
        residual, attentionWeights = self.applyDecoderAttention(valueData, predictionError)
        # residual dimension: batchSize, numCondensedSignals, signalDimension
        # attentionWeights dimension: batchSize, numHeads, numSignals

        # if allTrainingData: self.allAttentionWeights[-1].append(attentionWeights[0:1])  #  Only take the first batch due to memory constraints.
            
        # Apply residual connection.
        attentionOutput = residual + normLatentData
        # attentionOutput dimension: batchSize, numCondensedSignals, signalDimension
                
        return attentionOutput
    
    def feedForwardDecoderBlock(self, latentData, layerInd): 
        # Apply a normalization to each section of the latent data.
        normLatentData = self.layerNorm_LFF[layerInd](latentData)
        # normLatentData dimension: batchSize, numCondensedSignals, signalDimension
        
        # Apply feed forward block with residual connection to the signals.
        finalLatentData = self.feedForwardDecoderLayers(normLatentData.unsqueeze(1)).squeeze(1)
        # finalLatentData dimension: batchSize, numCondensedSignals, signalDimension
        
        return finalLatentData  
    
    # ------------------------ Attention Mechanism ------------------------- # 
    
    def getValueData(self, inputData):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.size()
        
        # Prepare the data
        inputData_CNN = inputData.unsqueeze(1)
        
        # Project the inputs into their respective spaces.
        value = self.decoderValueProjection(inputData_CNN) + inputData_CNN    # The value of each token we are adding.
        # Value dimension: batchSize, numCondensedSignals, numSignals, signalDimension
        
        # Segment the tokens into their respective heads/subgroups.
        value = value.view(batchSize, self.numCondensedSignals, numSignals, self.numHeads, self.decoderHeadDim).permute(0, 3, 2, 1, 4).contiguous()
        # Value dimension: batchSize, numHeads, numSignals, numCondensedSignals, decoderHeadDim
        
        return value
    
    def applyDecoderAttention(self, value, predictionError):
        # Extract the incoming data's dimension.
        batchSize, numHeads, numSignals, numCondensedSignals, decoderHeadDim  = value.size()
        
        # Segment the tokens into their respective heads/subgroups.
        predictionError = predictionError.view(batchSize, numSignals, self.numHeads, -1).transpose(1, 2).contiguous()
        # predictionError dimension: batchSize, numHeads, numSignals, decoderHeadDim

        # Score each signal influence the latent data.
        meanSquaredError = predictionError.mean(dim=-1)
        attentionWeights = F.softmax(meanSquaredError, dim=-1)
        # attentionWeights dimension: batchSize, numHeads, numSignals
        
        # Apply a weights to each head.
        attentionOutput = attentionWeights.view(batchSize, self.numHeads, numSignals, 1, 1) * value
        # finalOutput dimension: batchSize, numHeads, numSignals, numCondensedSignals, decoderHeadDim
        
        # Recombine the attention weights from all the different heads.
        attentionOutput = attentionOutput.permute(0, 2, 3, 1, 4).contiguous() # attentionOutput dimension: batchSize, numSignals, numCondensedSignals, numHeads, decoderHeadDim
        attentionOutput = attentionOutput.view(batchSize, numSignals, self.numCondensedSignals, self.signalDimension)
        # attentionOutput dimension: batchSize, numSignals, numCondensedSignals, signalDimension
        
        # Calculate the residual error.
        residual = attentionOutput.sum(dim=1)
        # residual dimension: batchSize, numCondensedSignals, signalDimension

        # Consolidate into reference-space with residual connection.
        residual = self.latentProjection(residual.unsqueeze(1)).squeeze(1)
        # residual dimension: batchSize, numCondensedSignals, signalDimension

        return residual, attentionWeights  

    # ----------------------- General Helper Methods ----------------------- # 
    
    def unfoldLatentData(self, latentData):
        # Extract the incoming data's dimension.
        batchSize, numCondensedSignals, signalDimension = latentData.size()
        # Assert the integrity of the inputs.
        assert self.signalDimension == signalDimension
        self.numCondensedSignals == numCondensedSignals
        
        # Reshape the array after nomalization
        unfoldedLatentData = latentData.transpose(1, 2).contiguous().view(batchSize, numCondensedSignals*signalDimension).contiguous()
        # normLatentData dimension: batchSize, latentDimension
        
        return unfoldedLatentData
    
    def refoldLatentData(self, latentData):
        # Extract the incoming data's dimension.
        batchSize, latentDimension = latentData.size()
        # Assert the integrity of the inputs.
        assert self.signalDimension * self.numCondensedSignals == latentDimension
    
        # Reshape the array for nomalization
        refoldedLatentData = latentData.view(batchSize, self.signalDimension, self.numCondensedSignals).transpose(1, 2).contiguous()
        # normLatentData dimension: batchSize, numCondensedSignals, signalDimension
        
        return refoldedLatentData
        

    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64, numCondensedSignals = 32, numDecoderLayers = 8, numHeads = 2).printParams(numSignals = 4)
        t1 = time.time()
        summary(self, (numSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

# -------------------------------------------------------------------------- #
# -------------------------- Decoder Architecture -------------------------- #   

class signalDecoding(encodingHead):
    def __init__(self, signalDimension = 64, numCondensedSignals = 64, numSignals = 50):
        super(signalDecoding, self).__init__(signalDimension, numCondensedSignals)
        # General parameters.
        self.numSignals = numSignals
                        
        # Layer normalizations.
        # self.initialLayerNorm = nn.LayerNorm(self.signalDimension, eps=1E-10)
        
        # For each signal.
        self.groupConv = nn.Sequential(
                # Convolution architecture: channel compression
                nn.Conv1d(in_channels=self.numCondensedSignals*numSignals, out_channels=16*numSignals, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=numSignals*4, bias=True),
                nn.Conv1d(in_channels=16*numSignals, out_channels=8*numSignals, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=numSignals*2, bias=True),
                nn.BatchNorm1d(8*numSignals, affine = True, momentum = 0.1, track_running_stats=True),
                nn.SELU(),
                    
                # Residual connection
                _convolutionalHelpers.ResNet(
                     module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.convolutionalFilter(numChannels = 8, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                        self.convolutionalFilter(numChannels = 8, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                ), numCycles = 1),
                
                # Residual connection
                _convolutionalHelpers.ResNet(
                     module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.convolutionalFilter(numChannels = 8, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                        self.convolutionalFilter(numChannels = 8, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                ), numCycles = 1),
                
                # Residual connection
                _convolutionalHelpers.ResNet(
                     module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.convolutionalFilter(numChannels = 8, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                        self.convolutionalFilter(numChannels = 8, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                ), numCycles = 1),
                
                # Convolution architecture: channel compression
                nn.Conv1d(in_channels=8*numSignals, out_channels=4*numSignals, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=numSignals, bias=True),
                nn.BatchNorm1d(4*numSignals, affine = True, momentum = 0.1, track_running_stats=True),
                nn.SELU(),

                # Residual connection
                _convolutionalHelpers.ResNet(
                     module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.convolutionalFilter(numChannels = 4, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                        self.convolutionalFilter(numChannels = 4, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                ), numCycles = 1),

                # Residual connection
                _convolutionalHelpers.ResNet(
                     module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.convolutionalFilter(numChannels = 4, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                        self.convolutionalFilter(numChannels = 4, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                ), numCycles = 1),
                
                # Residual connection
                _convolutionalHelpers.ResNet(
                     module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.convolutionalFilter(numChannels = 4, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                        self.convolutionalFilter(numChannels = 4, kernel_sizes = [3, 3], dilations = [1, 2], groups = numSignals),
                ), numCycles = 1),
                
                # Convolution architecture: channel compression
                nn.Conv1d(in_channels=4*numSignals, out_channels=2*numSignals, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=numSignals, bias=True),
                nn.Conv1d(in_channels=2*numSignals, out_channels=numSignals, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=numSignals, bias=True),
                nn.SELU(),
        )

    def forward(self, latentData):
        """ The shape of latentData: (batchSize, numCondensedSignals, signalDimension) """
        batchSize, numCondensedSignals, signalDimension = latentData.size()
        
        # Assert that we have the expected data format.
        assert signalDimension == self.signalDimension, \
            f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."
        assert numCondensedSignals == self.numCondensedSignals, \
            f"You provided {numCondensedSignals} condensed signals, but we expected {self.numCondensedSignals}."
                
        # --------------------- Signal Differentiation --------------------- #  
        
        # Reshape to concatenate all signals along the channel dimension.
        reconstructedSignals = latentData.repeat(1, self.numSignals, 1)
        # reconstructedData dimension: batchSize, self.numSignals*self.numCondensedSignals, self.signalDimension

        # Reconstruct the original signal using grouped convolutions.
        reconstructedData = self.groupConv(reconstructedSignals)
        # reconstructedData dimension: batchSize, self.numSignals, self.signalDimension
        
        # ------------------------------------------------------------------ #

        return reconstructedData

    def printParams(self):
        #signalDecoding(signalDimension = 64, numCondensedSignals = 32, numSignals = 2).printParams()
        t1 = time.time()
        summary(self, (self.numCondensedSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        