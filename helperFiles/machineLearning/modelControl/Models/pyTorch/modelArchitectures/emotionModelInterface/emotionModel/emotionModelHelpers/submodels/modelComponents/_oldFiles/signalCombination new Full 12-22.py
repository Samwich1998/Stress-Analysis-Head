# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import _convolutionalLinearLayer

import matplotlib.pyplot as plt

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class encodingHead(nn.Module):
    def __init__(self, signalDimension = 16, latentDimension = 16):
        super(encodingHead, self).__init__()
        # General shape parameters.
        self.signalDimension = signalDimension
        self.latentDimension = latentDimension
        
    def projectionArchitecture_2D(self, numChannels = [8, 4, 1], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
            # Convolution architecture
            nn.Conv2d(in_channels=1, out_channels=numChannels[0], kernel_size=(1, kernel_sizes[0]), stride=1, 
                      dilation = (1, dilations[0]), padding=(0, paddings[0]), padding_mode='reflect', groups=1, bias=True),
            nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(1, kernel_sizes[1]), stride=1, 
                      dilation = (1, dilations[1]), padding=(0, paddings[1]), padding_mode='reflect', groups=1, bias=True),
            nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, kernel_sizes[2]), stride=1, 
                      dilation = (1, dilations[2]), padding=(0, paddings[2]), padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
    def projectionArchitecture_1D(self, numChannels = [8, 4, 1], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
            # Convolution architecture
            nn.Conv1d(in_channels=1, out_channels=numChannels[0], kernel_size=kernel_sizes[0], stride=1, 
                      dilation=dilations[0], padding=paddings[0], padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=kernel_sizes[1], stride=1, 
                      dilation=dilations[1], padding=paddings[1], padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=kernel_sizes[2], stride=1, 
                      dilation=dilations[2], padding=paddings[2], padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )        

# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(encodingHead):
    def __init__(self, signalDimension = 64, latentDimension = 64, numDecoderLayers = 1, numHeads = 1, dropout_rate = 0.3):
        super(signalEncoding, self).__init__(signalDimension, latentDimension)
        # Initialize transformer parameters.
        self.scaledChannels = latentDimension // signalDimension  # The number of times the latent channel is bigger than the value.
        self.decoderHeadDim = latentDimension // numHeads    # The dimension of each decoder head (subset of latentDimension).
        self.numDecoderLayers = numDecoderLayers  # The number of times the data passes through the transformer block.
        self.dropout_rate = dropout_rate # Dropout rate for the feed forward layers.
        self.numHeads = numHeads    # The number of parallel heads for segmenting each embeddeding.
        
        # Assert proper transformer parameters.
        assert self.scaledChannels == latentDimension / signalDimension, f"latentDimension ({latentDimension}) should be an exact multiple of signalDimension ({signalDimension})"
        assert self.decoderHeadDim * numHeads == self.latentDimension, f"latentDimension must be divisible by numHeads: {self.decoderHeadDim}, {numHeads}, {self.latentDimension}"
        assert 0 < self.decoderHeadDim, f"The dimension of the transformer head cannot be less than or equal to zero: {self.decoderHeadDim}"
        
        # Define a trainable weight that is subject-specific.
        self.learnableStart = nn.Parameter(torch.randn(1, self.latentDimension)) # The base state of the latent space.

        # Create model for transforming decoder parameters.
        self.decoderValueProjection = self.projectionArchitecture_2D(numChannels = [16, 16, self.scaledChannels], kernel_sizes = [5, 5, 5], dilations = [1, 2, 1])
        self.latentProjection = self.projectionArchitecture_1D(numChannels = [16, 8, 1], kernel_sizes = [5, 5, 5], dilations = [1, 2, 1])

        # Final projection.
        self.finalProjection = self.projectionArchitecture_1D(numChannels = [8, 4, 1], kernel_sizes = [5, 5, 5], dilations = [1, 2, 1])
        
        # Add non-linearity to attention.
        self.feedForwardDecoderLayers = nn.Sequential(
            # Convolution architecture
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.Dropout(self.dropout_rate),
            nn.SELU(),
            
            # Convolution architecture
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Initialize the layer normalization.
        self.layerNorm_CA = nn.ModuleList([nn.LayerNorm(self.latentDimension, eps=1E-10) for _ in range(self.numDecoderLayers)])
        self.layerNorm_LFF = nn.ModuleList([nn.LayerNorm(self.latentDimension, eps=1E-10) for _ in range(self.numDecoderLayers)])
                
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
            decoderModel = signalDecoding(self.signalDimension, self.latentDimension, numSignals)
        
        # plt.plot(inputData[0][0].detach().cpu())
        # plt.plot(inputData[0][1].detach().cpu())
        # plt.plot(inputData[-1][0].detach().cpu())
        # plt.plot(inputData[-1][1].detach().cpu())
        # plt.title('inputData')
        # plt.show()

        # Initialize the final feature vector.
        # latentData = self.initializeLatentStart(transformedData[:, torch.randint(low=0, high=numSignals, size = ())])
        latentData = self.learnableStart.expand(batchSize, -1)
        # latentData dimension: batchSize, latentDimension
        
        # plt.plot(latentData[0].detach().cpu())
        # plt.plot(latentData[-1].detach().cpu())
        # plt.title('latentData1')
        # plt.show()

        # For each decoding layer.
        for layerInd in range(self.numDecoderLayers):
            # Normalize the latent data.
            latentData = self.layerNorm_CA[layerInd](latentData)        

            # Predict how well the data can be reconstructed.
            predictedSignal = decoderModel(latentData)
            predictionError = (predictedSignal - inputData) ** 2
            # predictionError dimension: batchSize, numSignals, signalDimension
            # predictionError dimension: batchSize, numSignals, signalDimension
            
            # Combine the values from the cross attentuation block.
            latentData = self.crossAttentionBlock(inputData, latentData, predictionError, layerInd, allTrainingData)
            # latentData dimension: batchSize, latentDimension

            # Apply a non-linearity to the transformer.
            latentData = self.feedForwardDecoderBlock(latentData, layerInd) # Apply feed-forward block.
            # latentData dimension: batchSize, latentDimension
                        
        # Final latent projection.
        latentData = self.finalProjection(latentData.unsqueeze(1)).squeeze(1)
        # latentData dimension: batchSize, latentDimension
        
        # plt.plot(latentData[0].detach().cpu())
        # plt.plot(latentData[-1].detach().cpu())
        # # plt.xlim((50, 100))
        # plt.title('End')
        # plt.show()
                
        # ------------------------------------------------------------------ #   
                    
        return latentData
    
    # ------------------------- Transformer Blocks ------------------------- #   
    
    def crossAttentionBlock(self, signalData, latentData, predictionError, layerInd, allTrainingData):
        # Apply cross attention block to the signals.
        residual, attentionWeights = self.applyDecoderAttention(signalData, predictionError)
        # if allTrainingData: self.allAttentionWeights[-1].append(attentionWeights[0:1])  #  Only take the first batch due to memory constraints.
            
        # Apply residual connection.
        attentionOutput = residual + latentData
        # attentionOutput dimension: batchSize, latentDimension
                
        return attentionOutput
    
    def feedForwardDecoderBlock(self, latentData, layerInd):   
        # Apply feed forward block to the signals.
        normLatentData = self.layerNorm_LFF[layerInd](latentData)        
        finalLatentData = self.feedForwardDecoderLayers(normLatentData.unsqueeze(1)).squeeze(1)
        # finalLatentData dimension: batchSize, latentDimension
        
        # Apply residual connection.
        # finalLatentData = finalLatentData + normLatentData
        # finalLatentData dimension: batchSize, latentDimension
        
        return finalLatentData  
    
    # ------------------------ Attention Mechanism ------------------------- #   
    
    def applyDecoderAttention(self, value, predictionError):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = value.size()
        # Assert the integrity of the inputs.
        assert value.size() == predictionError.size()
                                   
        # Project the inputs into their respective spaces.
        value = self.decoderValueProjection(value.unsqueeze(1))    # The value of each token we are adding.
        # Value dimension: batchSize, numChannels=self.scaledChannels, numSignals, signalDimension
                        
        # Reorganize the key/value dimension to fit the reference.
        value = value.permute(0, 2, 1, 3).reshape(batchSize, 1, numSignals, self.scaledChannels, signalDimension).contiguous()
        value = value.transpose(3, 4).reshape(batchSize, 1, numSignals, self.scaledChannels*signalDimension).contiguous()
        # Value dimension: batchSize, numChannels=1, numSignals, latentDimension  
        # Note, all self.scaledChannels values appear before the next signalDimension index.
        
        # Segment the tokens into their respective heads/subgroups.
        value = value.view(batchSize, numSignals, self.numHeads, self.decoderHeadDim).transpose(1, 2).contiguous()
        predictionError = predictionError.view(batchSize, numSignals, self.numHeads, -1).transpose(1, 2).contiguous()
        # predictionError dimension: batchSize, numHeads, numSignals, encoderDimension
        # Value dimension: batchSize, numHeads, numSignals, decoderHeadDim

        # Score each signal influence the latent data.
        meanSquaredError = predictionError.mean(dim=-1)
        attentionWeights = F.softmax(meanSquaredError, dim=-1).unsqueeze(-1)
        # attentionWeights dimension: batchSize, numHeads, numSignals, 1
        
        # Apply a weights to each head.
        attentionOutput = attentionWeights * value
        # finalOutput dimension: batchSize, numHeads, numSignals, decoderHeadDim

        # Recombine the attention weights from all the different heads.
        attentionOutput = attentionOutput.transpose(1, 2).contiguous().view(batchSize, numSignals, self.latentDimension)
        # attentionOutput dimension: batchSize, numSignals, latentDimension
                
        # Consolidate into reference-space
        residual = attentionOutput.mean(dim=1).unsqueeze(1)
        residual = self.latentProjection(residual).squeeze(1)
        # residual dimension: batchSize, latentDimension

        return residual, attentionWeights  

    # ----------------------- General Helper Methods ----------------------- # 

    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64, latentDimension = 1024, numDecoderLayers = 8, numHeads = 2).printParams(numSignals = 4)
        summary(self, (numSignals, self.signalDimension))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

# -------------------------------------------------------------------------- #
# -------------------------- Decoder Architecture -------------------------- #   

class signalDecoding(encodingHead):
    def __init__(self, signalDimension = 64, latentDimension = 64, numSignals = 50):
        super(signalDecoding, self).__init__(signalDimension, latentDimension)
        # General parameters.
        self.numSignals = numSignals
        self.scaleFactor = latentDimension // signalDimension
        
        # Assert proper decoder parameters.
        assert self.scaleFactor == latentDimension / signalDimension, f"latentDimension ({latentDimension}) should be an exact multiple of signalDimension ({signalDimension})"
        
        # A list of modules to decode each signal.
        self.signalDecodingModules_LinearConv = nn.ModuleList()
        self.signalDecodingModules_ANN = nn.ModuleList()
        self.channelReduction_CNN = nn.ModuleList()
        
        # Layer normalizations.
        self.initialLayerNorm = nn.LayerNorm(self.signalDimension, eps=1E-10)
        
        # For each signal.
        for signalInd in range(self.numSignals):   
            self.signalDecodingModules_ANN.append(
                _convolutionalLinearLayer.convolutionalLinearLayer(
                        initialDimension = self.signalDimension,
                        compressedDim = self.signalDimension,
                        compressedEmbeddedDim = 16,
                        embeddingStride = 1,
                        embeddedDim = 16,
                )
            )
            
            self.signalDecodingModules_LinearConv.append(
                nn.Sequential(
                    # Neural architecture: Layer 1.
                    nn.Linear(self.signalDecodingModules_ANN[signalInd].embeddedDim, self.signalDecodingModules_ANN[signalInd].compressedEmbeddedDim, bias = True),
                    nn.SELU(),
                )
            )

             
            self.channelReduction_CNN.append(
                nn.Sequential(
                    # Convolution architecture
                    nn.Conv1d(in_channels=self.scaleFactor, out_channels=8, kernel_size=7, stride=1, dilation = 1, padding=3, padding_mode='reflect', groups=1, bias=True),
                    nn.Conv1d(in_channels=8, out_channels=4, kernel_size=5, stride=1, dilation = 1, padding=2, padding_mode='reflect', groups=1, bias=True),
                    nn.SELU(),

                    nn.Conv1d(in_channels=4, out_channels=2, kernel_size=5, stride=1, dilation = 1, padding=2, padding_mode='reflect', groups=1, bias=True),
                    nn.Conv1d(in_channels=2, out_channels=1, kernel_size=5, stride=1, dilation = 2, padding=4, padding_mode='reflect', groups=1, bias=True),
                    # nn.LayerNorm(self.signalDimension, eps=1E-10),
                    nn.SELU(),
                )
            )

        self.decompressSignalsANN_LinearConv = _convolutionalLinearLayer.convolutionalLinearLayer(
                initialDimension = self.latentDimension,
                compressedDim = self.latentDimension,
                compressedEmbeddedDim = 32,
                embeddingStride = 1,
                embeddedDim = 32,
        )
        
        # Shared weights.
        self.latentDecoding_ANN = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.decompressSignalsANN_LinearConv.embeddedDim, self.decompressSignalsANN_LinearConv.compressedEmbeddedDim, bias = True),
            nn.Dropout(0.01),
            nn.SELU(),
        )
        
        self.sharedCNN = nn.Sequential(
            # Convolution architecture
            nn.Conv1d(in_channels=self.scaleFactor, out_channels=self.scaleFactor*2, kernel_size=5, stride=1, dilation = 1, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),

            nn.Conv1d(in_channels=self.scaleFactor*2, out_channels=self.scaleFactor, kernel_size=5, stride=1, dilation = 2, padding=4, padding_mode='reflect', groups=1, bias=True),
        )

    def forward(self, latentData):
        """ The shape of latentData: (batchSize, latentDimension) """
        batchSize, latentDimension = latentData.size()
        
        # Assert proper transformer parameters.
        assert latentDimension == self.latentDimension, \
            f"Expected latent dimension of {self.latentDimension} but got {latentDimension}."
    
        # Create a new tensor to hold the reconstructed signals.
        reconstructedData = torch.zeros((batchSize, self.numSignals, self.signalDimension), device=latentData.device)
            
        # ------------------------- Latent Decoding ------------------------ # 
        
        # import time
        # t1 = time.time()

        # Decode the signals from the latent dimension.
        # decodedLatentData = self.decompressSignalsANN_LinearConv(latentData, self.latentDecoding_ANN)
        # decodedLatentData dimension: batchSize, self.signalDimension
        
        # t2 = time.time(); print(0, t2-t1)
        # t1 = time.time()
                
        # Reshape the array
        latentData = latentData.view(batchSize, self.signalDimension, self.scaleFactor).transpose(1, 2)
        # decodedLatentData dimension: batchSize, self.scaleFactor, self.signalDimension
        
        # Learn complex features from the latent data.
        decodedLatentData = self.sharedCNN(latentData)
        decodedLatentData = self.sharedCNN(decodedLatentData)
        # Add a residual connection to prevent loss of information.
        decodedLatentData = decodedLatentData + latentData
        # decodedLatentData dimension: batchSize, self.scaleFactor, self.signalDimension
                
        # Apply layer normalization.
        decodedLatentData = self.initialLayerNorm(decodedLatentData)
        # decodedLatentData dimension: batchSize, self.scaleFactor, self.signalDimension
        # t2 = time.time(); print(1, t2-t1)

        # import matplotlib.pyplot as plt
        # plt.plot(decodedLatentData[0].flatten().detach().cpu())
                
        # --------------------- Signal Differentiation --------------------- #  
        
        # For each signal.
        for signalInd in range(self.numSignals): 
            
            # t1 = time.time()
            # Reduce the number of channels.
            signalData = self.channelReduction_CNN[signalInd](decodedLatentData).squeeze(1)
            # signalData dimension: batchSize, self.signalDimension
                        
            # Reconstruct the signalplt.shpw() from the signal-space.
            # signalData = self.signalDecodingModules_CNN[signalInd](signalData).squeeze(1)          
            # signalData dimension: batchSize, self.signalDimension 
            
            # if signalInd == 0:  plt.plot(signalData[0].detach().cpu())
            # t2 = time.time(); print(2, t2-t1)
            # t1 = time.time()
            
            # Transform the latent data into its respective signal-space
            # linearConvLayer = self.signalDecodingModules_LinearConv[signalInd]
            # signalData = self.signalDecodingModules_ANN[signalInd](signalData, linearConvLayer)
            # # signalData dimension: batchSize, self.signalDimension
            # t2 = time.time(); print(3, t2-t1)
            
            # if signalInd == 0:  plt.plot(signalData[0].detach().cpu())

            
            # Store the final results.
            reconstructedData[:, signalInd, :] = signalData
            # reconstructedData dimension: batchSize, self.numSignals, self.signalDimension

        # ------------------------------------------------------------------ #
        # plt.show()

        return reconstructedData

    def printParams(self):
        #signalDecoding(signalDimension = 64, latentDimension = 1024, numSignals = 2).printParams()
        import time
        t1 = time.time()
        summary(self, (self.latentDimension,))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        