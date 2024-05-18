# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

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
        self.decoderReferenceProjection = self.projectionArchitecture_1D(numChannels = [8, 4, 1], kernel_sizes = [5, 5, 5], dilations = [1, 2, 1])
        self.decoderValueProjection = self.projectionArchitecture_2D(numChannels = [8, 4, self.scaledChannels], kernel_sizes = [5, 5, 5], dilations = [1, 2, 1])
        self.decoderKeyProjection = self.projectionArchitecture_2D(numChannels = [8, 4, self.scaledChannels], kernel_sizes = [5, 5, 5], dilations = [1, 2, 1])
        self.latentProjection = self.projectionArchitecture_1D(numChannels = [8, 4, 1], kernel_sizes = [5, 5, 5], dilations = [1, 2, 1])

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
        batchSize, numSignals, compressedLength = inputData.size()
        # Assert that we have the expected data format.
        assert compressedLength == self.signalDimension, \
            f"You provided a signal of length {compressedLength}, but we expected {self.signalDimension}."
            
        if decoderModel == None:
            print("This should be a fake pass through signal combination.")
            decoderModel = signalDecoding(self.signalDimension, self.latentDimension, numSignals)
        
        plt.plot(inputData[0][0].detach().cpu())
        plt.plot(inputData[0][1].detach().cpu())
        plt.plot(inputData[-1][0].detach().cpu())
        plt.plot(inputData[-1][1].detach().cpu())
        plt.title('inputData')
        plt.show()

        # Initialize the final feature vector.
        # latentData = self.initializeLatentStart(transformedData[:, torch.randint(low=0, high=numSignals, size = ())])
        initialLatentData = self.learnableStart.expand(batchSize, -1)
        # latentData dimension: batchSize, latentDimension
        
        plt.plot(initialLatentData[0].detach().cpu())
        plt.plot(initialLatentData[-1].detach().cpu())
        plt.title('latentData1')
        plt.show()

        # For each decoding layer.
        for layerInd in range(self.numDecoderLayers):
            # Combine the values from the cross attentuation block.
            latentData = self.crossAttentionBlock(inputData, initialLatentData, decoderModel, layerInd, allTrainingData)
            # latentData dimension: batchSize, latentDimension

            # Apply a non-linearity to the transformer.
            latentData = self.feedForwardDecoderBlock(latentData, layerInd) # Apply feed-forward block.
            # latentData dimension: batchSize, latentDimension
                        
        # Final latent projection.
        latentData = self.finalProjection(latentData.unsqueeze(1)).squeeze(1)
        # latentData dimension: batchSize, latentDimension
        
        plt.plot(latentData[0].detach().cpu())
        plt.plot(latentData[-1].detach().cpu())
        # plt.xlim((50, 100))
        plt.title('End')
        plt.show()
                
        # ------------------------------------------------------------------ #   
                    
        return latentData
    
    # ------------------------- Transformer Blocks ------------------------- #   
    
    def crossAttentionBlock(self, signalData, latentData, decoderModel, layerInd, allTrainingData):
        # Apply cross attention block to the signals.
        normLatentData = self.layerNorm_CA[layerInd](latentData)        
        residual, attentionWeights = self.applyDecoderAttention(normLatentData, signalData, signalData, decoderModel)
        # if allTrainingData: self.allAttentionWeights[-1].append(attentionWeights[0:1])  #  Only take the first batch due to memory constraints.

        # Apply residual connection.
        attentionOutput = residual + normLatentData
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
    
    def _calculateAttention(self, reference, key, value, decoderModel):
        # Extract the incoming data's dimension.
        batchSize, numHeads, numTokensRef, headDimRef = reference.size()
        batchSize, numHeads, numTokensKey, headDimKey = key.size()
        
        # Check to see if the decoder can reconstruct the original signal.
        predictedSignal = decoderModel(reference)
        
        # We will treat the 'numHeads' dimension as a batch dimension for convolution purposes
        reference = reference.view(batchSize*numHeads, numTokensRef, headDimRef) # Merge batchSize and numHeads for groups
        key = key.view(batchSize*numHeads, numTokensKey, headDimKey).transpose(0, 1)  # Merge batchSize and numHeads for batch
        # Key dimension: numSignals, batchSize*numHeads, decoderHeadDim
        # Reference dimension: batchSize*numHeads, 1, decoderHeadDim

        # Apply convolution, treating each head as a separate batch.
        attentionWeights = F.conv1d(key, reference, stride=1, padding=0, groups=batchSize * numHeads)
        # attentionWeights dimension: numSignals, batchSize*numHeads, 1 
        
        # Organize the scores in the final layer.
        attentionWeights = attentionWeights.transpose(0, 1).view(batchSize, numHeads, numTokensKey)
        attentionWeights = F.softmax(attentionWeights, dim=-1).unsqueeze(-1)
        # attentionWeights dimension: batchSize, numHeads, decoderHeadDim, 1

        # Apply a weighted average of each value for this query.
        finalOutput = attentionWeights * value
        # finalOutput dimension: batchSize, numHeads, numTokensKey, decoderHeadDim
  
        return finalOutput, attentionWeights
    
    def applyDecoderAttention(self, reference, key, value, decoderModel):
        # Extract the incoming data's dimension.
        batchSize, numTokensValue, embeddedDimValue = value.size()
        batchSize, numTokensKey, embeddedDimKey = key.size()
        batchSize, embeddedDimReference = reference.size()
        # Assert the integrity of the inputs.
        assert embeddedDimValue == embeddedDimKey, f"Key-value dimensions must match: {embeddedDimKey} {embeddedDimValue}"
        assert numTokensValue == numTokensKey, f"Key-value tokens must match: {numTokensKey} {numTokensValue}"
    
        # Project the inputs into their respective spaces.
        reference = self.decoderReferenceProjection(reference.unsqueeze(1))  # The reference token we are embedding.
        value = self.decoderValueProjection(value.unsqueeze(1))    # The value of each token we are adding.
        key = self.decoderKeyProjection(key.unsqueeze(1))        # The value of each token we are adding.
        # Value dimension: batchSize, numChannels=self.scaledChannels, numSignals, signalDimension
        # Key dimension: batchSize, numChannels=self.scaledChannels, numSignals, signalDimension
        # Reference dimension: batchSize, numChannels=1, latentDimension
                
        # Reorganize the key/value dimension to fit the reference.
        value = value.permute(0, 2, 1, 3).reshape(batchSize, 1, numTokensValue, embeddedDimReference).contiguous()
        key = key.permute(0, 2, 1, 3).reshape(batchSize, 1, numTokensKey, embeddedDimReference).contiguous()
        # Value dimension: batchSize, numChannels=1, numSignals, latentDimension
        # Key dimension: batchSize, numChannels=1, numSignals, latentDimension

        # Segment the tokens into their respective heads/subgroups.
        reference = reference.view(batchSize, -1, self.numHeads, self.decoderHeadDim).transpose(1, 2).contiguous()
        value = value.view(batchSize, -1, self.numHeads, self.decoderHeadDim).transpose(1, 2).contiguous()
        key = key.view(batchSize, -1, self.numHeads, self.decoderHeadDim).transpose(1, 2).contiguous()
        # Value dimension: batchSize, numHeads, numSignals, decoderHeadDim
        # Key dimension: batchSize, numHeads, numSignals, decoderHeadDim
        # Reference dimension: batchSize, numHeads, 1, decoderHeadDim
        
        # Calculate the attention weight for each token.
        attentionOutput, attentionWeights = self._calculateAttention(reference, key, value, decoderModel)
        # attentionOutput dimension: batchSize, numHeads, numTokensKey, decoderHeadDim
                
        # Recombine the attention weights from all the different heads.
        attentionOutput = attentionOutput.transpose(1, 2).contiguous().view(batchSize, numTokensKey, self.latentDimension)
        # attentionOutput dimension: batchSize, numSignals, latentDimension
                
        # Consolidate into reference-space
        residual = attentionOutput.mean(dim=1).unsqueeze(1)
        residual = self.latentProjection(residual).squeeze(1)
        # residual dimension: batchSize, latentDimension

        return residual, attentionWeights  

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
        # signalEncoding(signalDimension = 64, latentDimension = 256, numDecoderLayers = 1, numHeads = 2).printParams(numSignals = 4)
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
        
        
        