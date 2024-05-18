# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import time

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import matplotlib.pyplot as plt

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class encodingHead(nn.Module):
    def __init__(self, signalDimension = 16, latentDimension = 16):
        super(encodingHead, self).__init__()
        # General shape parameters.
        self.signalDimension = signalDimension
        self.latentDimension = latentDimension
        
    def projectionArchitecture_2D(self, numChannels = [16, 8, 4, 1], kernel_sizes = [3, 3, 3, 3], dilations = [1, 2, 2, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
            # Convolution architecture
            nn.Conv2d(in_channels=1, out_channels=numChannels[0], kernel_size=(1, kernel_sizes[0]), stride=1, 
                      dilation = (1, dilations[0]), padding=(0, paddings[0]), padding_mode='reflect', groups=1, bias=True),
            nn.Conv2d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=(1, kernel_sizes[1]), stride=1, 
                      dilation = (1, dilations[1]), padding=(0, paddings[1]), padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),

            nn.Conv2d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=(1, kernel_sizes[2]), stride=1, 
                      dilation = (1, dilations[2]), padding=(0, paddings[2]), padding_mode='reflect', groups=1, bias=True),
            nn.Conv2d(in_channels=numChannels[2], out_channels=numChannels[3], kernel_size=(1, kernel_sizes[3]), stride=1, 
                      dilation = (1, dilations[3]), padding=(0, paddings[3]), padding_mode='reflect', groups=1, bias=True),
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
            nn.SELU(),

            nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=kernel_sizes[2], stride=1, 
                      dilation=dilations[2], padding=paddings[2], padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=numChannels[2], out_channels=numChannels[3], kernel_size=kernel_sizes[3], stride=1, 
                      dilation=dilations[3], padding=paddings[3], padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )        

# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(encodingHead):
    def __init__(self, signalDimension = 64, latentDimension = 64, numDecoderLayers = 1, numHeads = 1, dropout_rate = 0.3):
        super(signalEncoding, self).__init__(signalDimension, latentDimension)
        # Initialize transformer parameters.
        self.scaleFactor = latentDimension // signalDimension  # The number of times the latent channel is bigger than the value.
        self.decoderHeadDim = latentDimension // numHeads    # The dimension of each decoder head (subset of latentDimension).
        self.numDecoderLayers = numDecoderLayers  # The number of times the data passes through the transformer block.
        self.dropout_rate = dropout_rate # Dropout rate for the feed forward layers.
        self.numHeads = numHeads    # The number of parallel heads for segmenting each embeddeding.
        
        # Assert proper transformer parameters.
        assert self.scaleFactor == latentDimension / signalDimension, f"latentDimension ({latentDimension}) should be an exact multiple of signalDimension ({signalDimension})"
        assert self.decoderHeadDim * numHeads == self.latentDimension, f"latentDimension must be divisible by numHeads: {self.decoderHeadDim}, {numHeads}, {self.latentDimension}"
        assert 0 < self.decoderHeadDim, f"The dimension of the transformer head cannot be less than or equal to zero: {self.decoderHeadDim}"
        
        # Define a trainable weight that is subject-specific.
        self.learnableStart = nn.Parameter(torch.randn(1, self.latentDimension)) # The base state of the latent space.

        # Create model for transforming decoder parameters.
        self.decoderValueProjection = self.projectionArchitecture_2D(numChannels = [4, 8, 16, self.scaleFactor], kernel_sizes = [3, 5, 5, 3], dilations = [1, 1, 2, 1])
        self.latentProjection = self.projectionArchitecture_1D(numChannels = [4, 8, 4, 1], kernel_sizes = [3, 5, 5, 3], dilations = [1, 1, 1, 1])

        # Final projection.
        self.finalProjection = self.projectionArchitecture_1D(numChannels = [4, 8, 4, 1], kernel_sizes = [3, 5, 5, 3], dilations = [1, 1, 1, 1])
        
        # Add non-linearity to attention.
        self.feedForwardDecoderLayers = nn.Sequential(
            # Convolution architecture
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=1, dilation = 1, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=1, dilation = 2, padding=4, padding_mode='reflect', groups=1, bias=True),
            nn.Dropout(self.dropout_rate),
            nn.SELU(),
            
            # Convolution architecture
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
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
        
        # For each decoding layer.
        for layerInd in range(self.numDecoderLayers):
            # Apply a non-linearity to the transformer.
            latentData = self.feedForwardDecoderBlock(latentData, layerInd) # Apply feed-forward block.
            # latentData dimension: batchSize, latentDimension
            
            # Combine the values from the cross attentuation block.
            latentData = self.crossAttentionBlock(inputData, latentData, decoderModel, inputData, layerInd, allTrainingData)
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
    
    def normalizeLatentData(self, latentData, normalizationMethod):
        # Extract the incoming data's dimension.
        batchSize, latentDimension = latentData.size()
    
        # Reshape the array for nomalization
        latentData = latentData.view(batchSize, self.signalDimension, self.scaleFactor).transpose(1, 2).contiguous()
        normLatentData = (latentData)
        # normLatentData dimension: batchSize, scaleFactor, signalDimension
        
        # Reshape the array after nomalization
        normLatentData = normLatentData.transpose(1, 2).contiguous().view(batchSize, latentDimension).contiguous()
        # normLatentData dimension: batchSize, latentDimension
        
        return normLatentData
        
    def crossAttentionBlock(self, signalData, latentData, decoderModel, inputData, layerInd, allTrainingData):
        # Apply a normalization to each section of the latent data.
        normLatentData = self.normalizeLatentData(latentData, self.layerNorm_CA[layerInd])
        # normLatentData dimension: batchSize, latentDimension

        # Predict how well the data can be reconstructed.
        with torch.no_grad():
            predictedSignal = decoderModel(normLatentData)
        predictionError = (predictedSignal - inputData) ** 2
        # predictedSignal dimension: batchSize, numSignals, signalDimension
        # predictionError dimension: batchSize, numSignals, signalDimension
        
        # Apply cross attention block to the signals.
        residual, attentionWeights = self.applyDecoderAttention(signalData, predictionError)
        # if allTrainingData: self.allAttentionWeights[-1].append(attentionWeights[0:1])  #  Only take the first batch due to memory constraints.
            
        # Apply residual connection.
        attentionOutput = residual + normLatentData
        # attentionOutput dimension: batchSize, latentDimension
                
        return attentionOutput
    
    def feedForwardDecoderBlock(self, latentData, layerInd): 
        # Apply a normalization to each section of the latent data.
        normLatentData = self.normalizeLatentData(latentData, self.layerNorm_LFF[layerInd])
        # normLatentData dimension: batchSize, latentDimension
        
        # Apply feed forward block to the signals.
        finalLatentData = self.feedForwardDecoderLayers(normLatentData.unsqueeze(1)).squeeze(1)
        # finalLatentData dimension: batchSize, latentDimension
        
        # Apply residual connection.
        finalLatentData = finalLatentData + normLatentData
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
        # Value dimension: batchSize, numChannels=self.scaleFactor, numSignals, signalDimension
                        
        # Reorganize the key/value dimension to fit the reference.
        value = value.permute(0, 2, 1, 3).reshape(batchSize, 1, numSignals, self.scaleFactor, signalDimension).contiguous()
        value = value.transpose(3, 4).reshape(batchSize, 1, numSignals, self.scaleFactor*signalDimension).contiguous()
        # Value dimension: batchSize, numChannels=1, numSignals, latentDimension  
        # Note, all self.scaleFactor values appear before the next signalDimension index.
        
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
        t1 = time.time()
        summary(self, (numSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
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
                
        # Layer normalizations.
        self.initialLayerNorm = nn.LayerNorm(self.signalDimension, eps=1E-10)
        
        # For each signal.
        self.groupConv = nn.Sequential(
                # Convolution architecture
                nn.Conv1d(in_channels=self.scaleFactor*numSignals, out_channels=8*numSignals, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=numSignals, bias=True),
                nn.Conv1d(in_channels=8*numSignals, out_channels=8*numSignals, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=numSignals, bias=True),
                nn.SELU(),

                nn.Conv1d(in_channels=8*numSignals, out_channels=8*numSignals, kernel_size=7, stride=1, dilation = 1, padding=3, padding_mode='reflect', groups=numSignals, bias=True),
                nn.Conv1d(in_channels=8*numSignals, out_channels=8*numSignals, kernel_size=7, stride=1, dilation = 2, padding=6, padding_mode='reflect', groups=numSignals, bias=True),
                nn.SELU(),
                
                nn.Conv1d(in_channels=8*numSignals, out_channels=8*numSignals, kernel_size=5, stride=1, dilation = 1, padding=2, padding_mode='reflect', groups=numSignals, bias=True),
                nn.Conv1d(in_channels=8*numSignals, out_channels=8*numSignals, kernel_size=5, stride=1, dilation = 2, padding=4, padding_mode='reflect', groups=numSignals, bias=True),
                nn.SELU(),
                
                nn.Conv1d(in_channels=8*numSignals, out_channels=8*numSignals, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=numSignals, bias=True),
                nn.Conv1d(in_channels=8*numSignals, out_channels=8*numSignals, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=numSignals, bias=True),
                nn.SELU(),
                
                nn.Conv1d(in_channels=8*numSignals, out_channels=4*numSignals, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=numSignals, bias=True),
                nn.Conv1d(in_channels=4*numSignals, out_channels=4*numSignals, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=numSignals, bias=True),
                nn.SELU(),
                                
                nn.Conv1d(in_channels=4*numSignals, out_channels=2*numSignals, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=numSignals, bias=True),
                nn.Conv1d(in_channels=2*numSignals, out_channels=numSignals, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=numSignals, bias=True),
                nn.SELU(),
        )
        
        self.sharedCNN = nn.Sequential(
            # Convolution architecture
            nn.Conv1d(in_channels=self.scaleFactor, out_channels=self.scaleFactor, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
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
                
        # --------------------- Signal Differentiation --------------------- #  
        
        # Reshape to concatenate all signals along the channel dimension.
        reconstructedSignals = decodedLatentData.repeat(1, self.numSignals, 1)
        # reconstructedData dimension: batchSize, self.numSignals*self.scaleFactor, self.signalDimension

        # Reconstruct the original signal using grouped convolutions.
        reconstructedData = self.groupConv(reconstructedSignals)
        # reconstructedData dimension: batchSize, self.numSignals, self.signalDimension
        
        # ------------------------------------------------------------------ #

        return reconstructedData

    def printParams(self):
        #signalDecoding(signalDimension = 64, latentDimension = 1024, numSignals = 2).printParams()
        t1 = time.time()
        summary(self, (self.latentDimension,))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        