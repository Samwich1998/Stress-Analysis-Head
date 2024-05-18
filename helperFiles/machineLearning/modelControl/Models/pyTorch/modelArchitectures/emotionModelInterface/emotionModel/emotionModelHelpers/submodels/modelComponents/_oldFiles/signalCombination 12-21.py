# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import math

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import matplotlib.pyplot as plt


# Import helper models
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import _convolutionalLinearLayer
import positionEncodings
import signalEmbedding

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class encodingHead(nn.Module):
    def __init__(self, signalDimension = 16, latentDimension = 16):
        super(encodingHead, self).__init__()
        # General shape parameters.
        self.signalDimension = signalDimension
        self.latentDimension = latentDimension

# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(encodingHead):
    def __init__(self, signalDimension = 64, embeddedDimension = 16, latentDimension = 64, 
                       numEncoderLayers = 1, numDecoderLayers = 1, numHeads = 1, dropout_rate = 0.3):
        super(signalEncoding, self).__init__(signalDimension, latentDimension)
        # Positional encoding.
        self.positionEncoding = positionEncodings
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Specify the CPU or GPU capabilities.
        
        # Initialize transformer parameters.
        self.encoderHeadDim = embeddedDimension // numHeads  # The dimension of each encoder head (subset of embeddedDim).
        self.decoderHeadDim = latentDimension // numHeads    # The dimension of each decoder head (subset of latentDimension).
        self.embeddedDim = embeddedDimension      # The dimension of each token in the transformer.
        self.numDecoderLayers = numDecoderLayers  # The number of times the data passes through the transformer block.
        self.numEncoderLayers = numEncoderLayers  # The number of times the data passes through the transformer block.
        self.dropout_rate = dropout_rate # Dropout rate for the feed forward layers.
        self.numHeads = numHeads    # The number of parallel heads for segmenting each embeddeding.
        
        # Assert proper transformer parameters.
        assert self.decoderHeadDim * numHeads == self.latentDimension, f"latentDimension must be divisible by numHeads: {self.decoderHeadDim}, {numHeads}, {self.latentDimension}"
        assert self.encoderHeadDim * numHeads == self.embeddedDim, f"embeddedDim must be divisible by numHeads: {self.encoderHeadDim}, {numHeads}, {self.embeddedDim}"
        assert 0 < self.decoderHeadDim, f"The dimension of the transformer head cannot be less than or equal to zero: {self.decoderHeadDim}"
        assert 0 < self.encoderHeadDim, f"The dimension of the transformer head cannot be less than or equal to zero: {self.encoderHeadDim}"
        
        # Initial projection.
        self.signalEmbedding = signalEmbedding.embeddingLayer(signalDimension, embeddedDimension).to(self.device)
        self.initializeLatentStart = nn.Linear(embeddedDimension, latentDimension, bias = True)
        
        # Create model for transforming signals into query, key, value and embedded dimensions.
        self.embeddingProjection = nn.Linear(embeddedDimension, embeddedDimension, bias = True)
        self.valueProjection = nn.Linear(embeddedDimension, embeddedDimension, bias = True)
        self.queryProjection = nn.Linear(embeddedDimension, embeddedDimension, bias = True)
        self.keyProjection = nn.Linear(embeddedDimension, embeddedDimension, bias = True)
                
        # Create model for transforming decoder parameters.
        self.decoderQueryProjection = nn.Linear(latentDimension, latentDimension, bias = True)
        self.decoderValueProjection = nn.Linear(embeddedDimension, latentDimension, bias = True)
        self.decoderKeyProjection = nn.Linear(embeddedDimension, embeddedDimension, bias = True)
        self.latentProjection = nn.Linear(latentDimension, latentDimension, bias = True)
        
        # Bilinear attention blocks.
        self.encoderBilinearBlock = nn.Linear(self.encoderHeadDim, self.encoderHeadDim, bias = True)
        self.decoderBilinearBlock = nn.Linear(self.encoderHeadDim + self.decoderHeadDim, self.decoderHeadDim, bias = True)
        
        # Final projection.
        self.finalProjection = nn.Linear(latentDimension, latentDimension, bias = True)
        
        # Add non-linearity to attention.
        self.feedForwardEncoderLayers = nn.Sequential(
            # It is common to expand the self.embeddedDim
                # By a factor of 2-4 (small), 
                # By a factor of 4-8 (medium), 
                # By a factor of 8-16 (large).
                
            # Neural architecture: Layer 1.
            nn.Linear(self.embeddedDim, self.embeddedDim*2, bias = True),
            nn.Dropout(self.dropout_rate),
            nn.SELU(),
            
            
            # # Neural architecture: Layer 3.
            nn.Linear(self.embeddedDim*2, self.embeddedDim, bias = True),
        )
        
        # Add non-linearity to attention.
        self.feedForwardDecoderLayers = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.latentDimension, int(self.latentDimension/2), bias = True),
            nn.Dropout(self.dropout_rate),
            nn.SELU(),
            
            # # Neural architecture: Layer 3.
            nn.Linear(int(self.latentDimension/2), self.latentDimension, bias = True),
        )
        
        # Initialize the layer normalization.
        self.layerNorm_CA = nn.ModuleList([nn.LayerNorm(self.latentDimension, eps=1E-10) for _ in range(self.numDecoderLayers)])
        self.layerNorm_SA = nn.ModuleList([nn.LayerNorm(self.embeddedDim, eps=1E-10) for _ in range(self.numEncoderLayers)])
        self.layerNorm_FF = nn.ModuleList([nn.LayerNorm(self.embeddedDim, eps=1E-10) for _ in range(self.numEncoderLayers)])
        self.layerNorm_LFF = nn.ModuleList([nn.LayerNorm(self.latentDimension, eps=1E-10) for _ in range(self.numDecoderLayers)])

        
        # Define a trainable weight that is subject-specific.
        self.learnableStart = nn.Parameter(torch.randn(1, self.latentDimension))
        # self.learnableStart: The base state of the latent space.
                
        # Initialize holder parameters.
        self.allAttentionWeights = []  # Dimension: numEpochs, numLayers, attentionWeightDim, where attentionWeightDim = batch_size, numHeads, seq_length, seq_length

    def forward(self, inputData, allTrainingData = False, calculateLoss = False):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """
        # if allTrainingData: self.allAttentionWeights.append([]);

        # Extract the incoming data's dimension.
        batchSize, numSignals, compressedLength = inputData.size()
        # Assert that we have the expected data format.
        assert compressedLength == self.signalDimension, \
            f"You provided a signal of length {compressedLength}, but we expected {self.signalDimension}."
        
        # plt.plot(inputData[0][0].detach().cpu())
        # plt.plot(inputData[0][1].detach().cpu())
        # plt.plot(inputData[-1][0].detach().cpu())
        # plt.plot(inputData[-1][1].detach().cpu())
        # plt.title('inputData')
        # plt.show()

        # ------------------------ Apply Transformer ----------------------- #  

        # Add positional encoding.
        positionEncodedData = self.positionEncoding.positionalEncoding[-1, :] + inputData
        # positionEncodedData dimension: batchSize, numSignals, compressedLength
        
        # Compress the information from the signals.
        embeddedData = self.signalEmbedding(positionEncodedData)
        # embeddedData dimension: batchSize, numSignals, embeddedDimension
        
        # Initialize the transformed data.
        transformedData = embeddedData.clone()
        # transformedData dimension: batchSize, numSignals, embeddedDimension

        if calculateLoss:
            # Enforce the signal embedding to be standardized.
            signalEmbeddingMeanLoss, signalEmbeddingStandardDeviationLoss = self.calculateStandardizationLoss(transformedData, expectedMean = 0, expectedStandardDeviation = 1, dim=-1)
        
        # plt.plot(embeddedData[0][0].detach().cpu())
        # plt.plot(embeddedData[0][1].detach().cpu())
        # plt.plot(embeddedData[-1][0].detach().cpu())
        # plt.plot(embeddedData[-1][1].detach().cpu())
        # plt.title('embeddedData')
        # plt.show()
        
        # For each encoding layer.
        for layerInd in range(self.numEncoderLayers):            
            # Embed the signal information into each signal using self-attention.
            transformedData = self.selfAttentionBlock(transformedData, allTrainingData, layerInd)
            # transformedData dimension: batchSize, numSignals, embeddedDimension

            # Apply residual connection.
            transformedData = transformedData + embeddedData
            
            # Apply a non-linearity to the transformer.
            transformedData = self.feedForwardEncoderBlock(transformedData, layerInd) # Apply feed-forward block.
            # transformedData dimension: batchSize, numSignals, embeddedDimension

        # plt.plot(transformedData[0][0].detach().cpu())
        # plt.plot(transformedData[0][1].detach().cpu())
        # plt.plot(transformedData[-1][0].detach().cpu())
        # plt.plot(transformedData[-1][1].detach().cpu())
        # plt.title('transformedData')
        # plt.show()

        # Initialize the final feature vector.
        # latentData = self.initializeLatentStart(transformedData[:, torch.randint(low=0, high=numSignals, size = ())])
        initialLatentData = self.learnableStart.expand(batchSize, -1)
        # latentData dimension: batchSize, latentDimension
        
        # plt.plot(initialLatentData[0].detach().cpu())
        # plt.plot(initialLatentData[-1].detach().cpu())
        # plt.title('latentData1')
        # plt.show()

        # For each decoding layer.
        for layerInd in range(self.numDecoderLayers):
            # Combine the values from the cross attentuation block.
            latentData = self.crossAttentionBlock(transformedData, initialLatentData, layerInd, allTrainingData)
            # latentData dimension: batchSize, latentDimension

            # Apply a non-linearity to the transformer.
            latentData = self.feedForwardDecoderBlock(latentData, layerInd) # Apply feed-forward block.
            # latentData dimension: batchSize, latentDimension
            
            # initialLatentData = initialLatentData + latentData
            
        # Final latent projection.
        latentData = self.finalProjection(latentData)
        
        # plt.plot(latentData[0].detach().cpu())
        # plt.plot(latentData[-1].detach().cpu())
        # # plt.xlim((50, 100))
        # plt.title('End')
        # plt.show()
        # sys.exit()
                
        # ------------------------------------------------------------------ #   
                    
        return latentData
    
    # ------------------------- Transformer Blocks ------------------------- #   
    
    def selfAttentionBlock(self, transformedData, allTrainingData, layerInd):
        # Apply self attention block to the signals.
        normTransformedData = self.layerNorm_SA[layerInd](transformedData)        
        attentionOutput, attentionWeights = self.applyEncoderAttention(normTransformedData, normTransformedData, normTransformedData)
        # if allTrainingData: self.allAttentionWeights[-1].append(attentionWeights[0:1])  #  Only take the first batch due to memory constraints.
        
        # Apply residual connection.
        attentionOutput = attentionOutput + normTransformedData
        # attentionOutput dimension: batchSize, numSignals, embeddedDimension
                
        return attentionOutput
    
    def feedForwardEncoderBlock(self, transformedData, layerInd):
        # Apply feed forward block to the signals.
        normTransformedData = self.layerNorm_FF[layerInd](transformedData)
        finalTransformedData = self.feedForwardEncoderLayers(normTransformedData)
        # finalTransformedData dimension: batchSize, numSignals, compressedLength
        
        # Apply residual connection.
        finalTransformedData = finalTransformedData + normTransformedData
        # finalTransformedData dimension: batchSize, numSignals, embeddedDimension
        
        return finalTransformedData
    
    def crossAttentionBlock(self, transformedData, latentData, layerInd, allTrainingData):
        # Apply cross attention block to the signals.
        normLatentData = self.layerNorm_CA[layerInd](latentData)        
        attentionOutput, attentionWeights = self.applyDecoderAttention(normLatentData, transformedData, transformedData)
        # if allTrainingData: self.allAttentionWeights[-1].append(attentionWeights[0:1])  #  Only take the first batch due to memory constraints.
        
        # Apply residual connection.
        # attentionOutput = attentionOutput + normLatentData
        # attentionOutput dimension: batchSize, latentDimension
                
        return attentionOutput
    
    def feedForwardDecoderBlock(self, latentData, layerInd):       
        # Apply feed forward block to the signals.
        normLatentData = self.layerNorm_LFF[layerInd](latentData)        
        finalLatentData = self.feedForwardDecoderLayers(normLatentData)
        # finalLatentData dimension: batchSize, latentDimension
        
        # Apply residual connection.
        # finalLatentData = finalLatentData + normLatentData
        # finalLatentData dimension: batchSize, latentDimension
        
        return finalLatentData  
    
    # ------------------------ Attention Mechanism ------------------------- #   
    
    def _calculateAttention(self, query, key, value, bilinearBlock):
        # Extract the incoming data's dimension.
        batchSize, numHeads, numTokensKey, headDimKey = key.size()
                            
        # Calculate the similarity weight between the query and key tokens.
        bilinearKey = bilinearBlock(key).transpose(-2, -1)
        attentionWeights = torch.matmul(query, bilinearKey) / math.sqrt(headDimKey)
        attentionWeights = F.softmax(attentionWeights, dim=-1)
        # attentionWeights dimension: batchSize, numHeads, numTokensQuery, numTokensKey
      
        # Apply a weighted average of each value for this query.
        finalOutput = torch.matmul(attentionWeights, value)
        # finalOutput dimension: batchSize, numHeads, numTokensQuery, headDimValue
                        
        return finalOutput, attentionWeights
    
    def applyEncoderAttention(self, query, key, value):
        # Extract the incoming data's dimension.
        batchSize, numTokensQuery, embeddedDimQuery = query.size()
        batchSize, numTokensKey, embeddedDimKey = key.size()
                        
        # Project the inputs into their respective spaces.
        value = self.valueProjection(value)     # The weight of the key's token.
        query = self.queryProjection(query)     # The reference token we are analyzing.
        key = self.keyProjection(key)           # The token we are comparing to the reference. 
        
        # Segment the tokens into their respective heads/subgroups.
        query = query.view(batchSize, -1, self.numHeads, self.encoderHeadDim).transpose(1, 2)
        key = key.view(batchSize, -1, self.numHeads, self.encoderHeadDim).transpose(1, 2)
        value = value.view(batchSize, -1, self.numHeads, self.encoderHeadDim).transpose(1, 2)
        # Query, key and value dimensions: batchSize, numHeads, numTokens, encoderHeadDim
        
        # Calculate the attention weight for each token.
        attentionOutput, attentionWeights = self._calculateAttention(query, key, value, self.encoderBilinearBlock)
        # attentionOutput dimension: batchSize, numHeads, numTokens, encoderHeadDim
        
        # Recombine the attention weights from all the different heads.
        attentionOutput = attentionOutput.transpose(1, 2).contiguous().view(batchSize, numTokensQuery, self.embeddedDim)
        # attentionOutput dimension: batchSize, numTokens, embeddedDim
        
        # Transform the concatenated heads back into the embedding space.
        attentionOutput = self.embeddingProjection(attentionOutput)
        # attentionOutput dimension: batchSize, numTokens, embeddedDim
        
        return attentionOutput, attentionWeights  
    
    def applyDecoderAttention(self, query, key, value):
        # Extract the incoming data's dimension.
        batchSize, numTokensValue, embeddedDimValue = value.size()
        batchSize, numTokensKey, embeddedDimKey = key.size()
        batchSize, embeddedDimQuery = query.size()
        # Assert the integrity of the inputs.
        assert embeddedDimValue == embeddedDimKey, f"Key-value dimensions must match: {embeddedDimKey} {embeddedDimValue}"
        assert numTokensValue == numTokensKey, f"Key-value tokens must match: {numTokensKey} {numTokensValue}"
                        
        # Project the inputs into their respective spaces.
        query = self.decoderQueryProjection(query)   # The reference token (feature vector) we are compiling.
        value = self.decoderValueProjection(value)   # The weight of the key's token.
        key = self.decoderKeyProjection(key)                # The token we are comparing to the reference. 
        # Query dimension: batchSize, latentDimension
        # Value dimension:batchSize, numSignals, latentDimension
        # Key dimension: batchSize, numSignals, embeddedDimension

        # Segment the tokens into their respective heads/subgroups.
        query = query.view(batchSize, -1, self.numHeads, self.decoderHeadDim).transpose(1, 2)
        value = value.view(batchSize, -1, self.numHeads, self.decoderHeadDim).transpose(1, 2)
        key = key.view(batchSize, -1, self.numHeads, self.encoderHeadDim).transpose(1, 2)
        # Query dimension: batchSize, numHeads, 1, decoderHeadDim
        # Value dimension: batchSize, numHeads, numSignals, decoderHeadDim
        # Key dimension: batchSize, numHeads, numSignals, encoderHeadDim

        # Add the query information to the keys.
        queryForConcat = query.expand(batchSize, self.numHeads, numTokensKey, self.decoderHeadDim)
        key = torch.cat((key, queryForConcat), dim=-1)
        
        # Calculate the attention weight for each token.
        attentionOutput, attentionWeights = self._calculateAttention(query, key, value, self.decoderBilinearBlock)
        # attentionOutput dimension: batchSize, numHeads, numTokensQuery, decoderHeadDim

        # Recombine the attention weights from all the different heads.
        attentionOutput = attentionOutput.transpose(1, 2).contiguous().view(batchSize, self.latentDimension)
        # attentionOutput dimension: batchSize, latentDimension
        
        # Transform the concatenated heads back into the embedding space.
        attentionOutput = self.latentProjection(attentionOutput)
        # attentionOutput dimension: batchSize, latentDimension
        
        return attentionOutput, attentionWeights  

    # ----------------------- General Helper Methods ----------------------- # 
    
    def calculateStandardizationLoss(self, inputData, expectedMean = 0, expectedStandardDeviation = 1, dim=-1):
        # Calculate the data statistics on the last dimension.
        standardDeviationData = inputData.std(dim=dim)
        meanData = inputData.mean(dim=dim)

        # Calculate the squared deviation from mean = 0; std = 1.
        standardDeviationError = ((standardDeviationData - expectedStandardDeviation) ** 2).mean()
        meanError = ((meanData - expectedMean) ** 2).mean()
        
        return meanError, standardDeviationError

    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64, embeddedDimension = 16, latentDimension = 64, numEncoderLayers = 1, numDecoderLayers = 1, numHeads = 1).printParams(numSignals = 2)
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
        
        # A list of modules to decode each signal.
        self.signalDecodingModules_ANN = nn.ModuleList()
        self.signalDecodingModules_CNN = nn.ModuleList()
        self.signalDecodingModules_BatchNorm = nn.ModuleList()
        
        # For each signal.
        for signalInd in range(self.numSignals):
            # Reconstruct the signal from the latent dimension
            self.signalDecodingModules_ANN.append(
                nn.Sequential( 
                    # Neural architecture: Layer 1.
                    nn.Linear(self.signalDimension, int(self.signalDimension/2), bias = True),
                    # nn.BatchNorm1d(int(self.signalDimension/2), affine = True, momentum = 0.1, track_running_stats=True),
                    nn.Dropout(0.1),
                    nn.SELU(),
                    
                    # Neural architecture: Layer 2.
                    nn.Linear(int(self.signalDimension/2), self.signalDimension, bias = True),
                    nn.SELU(),
                )
            )
            
            self.signalDecodingModules_BatchNorm.append(
                    nn.BatchNorm1d(self.signalDimension, affine = True, momentum = 0.1, track_running_stats=True)
            )

                
            self.signalDecodingModules_CNN.append(
                nn.Sequential(
                    # Convolution architecture
                    nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
                    nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
                    nn.LayerNorm(self.signalDimension, eps=1E-10),
                    nn.SELU(),

                    # Convolution architecture
                    nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
                    nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
                    nn.SELU(),
                )
            )
        
        self.latentDecoding_ANN = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.latentDimension, self.signalDimension, bias = True),
            nn.Dropout(0.1),
            nn.SELU(),
            
            # Neural architecture: Layer 2.
            nn.Linear(self.signalDimension, self.signalDimension, bias = True),
            nn.SELU(),
        )

    def forward(self, latentData):
        """ The shape of latentData: (batchSize, latentDimension) """
        batchSize, latentDimension = latentData.size()
        
        # Assert we have the expected data format.
        assert latentDimension == self.latentDimension, \
            f"Expected latent dimension of {self.latentDimension} but got {latentDimension}."
    
        # Create a new tensor to hold the reconstructed signals.
        reconstructedData = torch.zeros((batchSize, self.numSignals, self.signalDimension), device=latentData.device)
            
        # ------------------------- Latent Decoding ------------------------ # 

        # Decode the signals from the latent dimension.
        decodedLatentData = self.latentDecoding_ANN(latentData)
        # decodedLatentData dimension: batchSize, latentDimension
        
        # import matplotlib.pyplot as plt
        # plt.plot(decodedLatentData[0].detach().cpu())
        # import time
                
        # --------------------- Signal Differentiation --------------------- #  
        
        # For each signal.
        for signalInd in range(self.numSignals):
            # t1 = time.time()
            
            # Transform the latent data into its respective signal-space
            signalData = self.signalDecodingModules_ANN[signalInd](decodedLatentData)
            # signalData dimension: batchSize, self.signalDimension
            
            # t2 = time.time(); print(1, t2-t1)
            # if signalInd == 0:  plt.plot(signalData[0].detach().cpu())
            # t1 = time.time()
            
            
            # Apply batch norm.
            # signalData = self.signalDecodingModules_BatchNorm[signalInd](signalData)
            # # signalData dimension: batchSize, self.signalDimension
            
            # signalData = decodedLatentData.clone()
            
            # # t2 = time.time(); print(2, t2-t1)
            # if signalInd == 0:  plt.plot(signalData[0].detach().cpu())
            # # t1 = time.time()

            # Reshape the data for a CNN.
            # signalData = signalData.unsqueeze(1)
            # signalData dimension: batchSize, 1, self.signalDimension
                        
            # Reconstruct the signalplt.shpw() from the signal-space.
            # signalData = self.signalDecodingModules_CNN[signalInd](signalData).squeeze(1)          
            # signalData dimension: batchSize, self.signalDimension 
            
            # t2 = time.time(); print(3, t2-t1)
            # if signalInd == 0:  plt.plot(signalData[0].detach().cpu())
            # t1 = time.time()
            


            reconstructedData[:, signalInd, :] = signalData

        # ------------------------------------------------------------------ #
        # plt.show()

        return reconstructedData

    def printParams(self):
        #signalDecoding(signalDimension = 64, latentDimension = 64, numSignals = 2).printParams()
        summary(self, (self.latentDimension,))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        