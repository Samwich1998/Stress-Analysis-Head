# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# Import files
import positionEncodings


# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class latentTransformationHead(nn.Module):
    def __init__(self, signalDimension = 64, latentDimension = 32):
        super(latentTransformationHead, self).__init__()
        # General shape parameters.
        self.signalDimension = signalDimension
        self.latentDimension = latentDimension
        
        # Signal embedding parameters
        self.signalEmbeddingStride = 2
        self.latentEmbeddingDim = 7
        self.embeddedDim = 14  # The sliding window for each signal.
        
        # Signal embedding parameters.
        self.signalEmbedding_indexSlices = torch.arange(0, signalDimension - self.embeddedDim + 1, self.signalEmbeddingStride).long()
        self.latentReconstruction_indexSlices =  torch.linspace(0, latentDimension - self.latentEmbeddingDim, len(self.signalEmbedding_indexSlices)).long()
        assert len(self.signalEmbedding_indexSlices) == len(self.latentReconstruction_indexSlices), "Must have the same number of segments to maintain 1:1 relationship."

        # Assert the integrity of signal embedding parameters.
        self.assertValidEmbedding(signalDimension, self.embeddedDim, self.signalEmbedding_indexSlices)
        self.assertValidEmbedding(latentDimension, self.latentEmbeddingDim, self.latentReconstruction_indexSlices)
    
    def assertValidEmbedding(self, signalDimension, embeddedDim, indexSlices):
        # Assert the integrity of start indices.
        assert indexSlices.diff()[0] <= embeddedDim, f"If you use these parameters you will not be using all the datapoints. See for yourself: {indexSlices} {embeddedDim}"
        assert indexSlices[-1] == signalDimension - embeddedDim, f"If you use these parameters you will miss the final points. See for yourself: {indexSlices}"
        assert len(indexSlices.diff().unique()) == 1, f"If you use these parameters you will not be evenly spaced. See for yourself: {indexSlices}"
        assert indexSlices[0] == 0, f"If you use these parameters you will miss the initial points. See for yourself: {indexSlices}"

    # --------------------------- Helper Methods --------------------------- #
    
    def embedSignals(self, inputData, embeddedDim, sliceIndices):        
        # Utilize tensor slicing and indexing to avoid explicit loops
        segmentedData = torch.stack([inputData[:, startInd:startInd + embeddedDim] for startInd in sliceIndices], dim=1)
        
        return segmentedData 

    def signalDefragmentation(self, segmentedData, signalDimension, embeddedDim, sliceIndices):        
        # Extract the incoming shape of the fragmented tensor.
        batchSize, numSegments, latentDimension = segmentedData.size()

        # Initialize the tensor to store holders.
        reconstructedData = torch.zeros(batchSize, signalDimension, requires_grad=False)
        contributions = torch.zeros_like(reconstructedData)
                    
        # Loop over each segment 
        for i, startInd in enumerate(sliceIndices):
            endInd = startInd + embeddedDim
                        
            # place it in the reconstructed data tensor
            reconstructedData[:, startInd:endInd] = reconstructedData[:, startInd:endInd].clone() + segmentedData[:, i, :]
            contributions[:, startInd:endInd] = contributions[:, startInd:endInd].clone() + 1
    
        # Average the contributions from overlapping segments
        reconstructedData = reconstructedData / contributions
    
        return reconstructedData
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(latentTransformationHead):
    def __init__(self, signalDimension = 64, latentDimension = 32, numSignals = 50):
        super(signalEncoding, self).__init__(signalDimension, latentDimension)
        # General parameters.
        self.numSignals = numSignals            # The number of independant signals in the model.

        # A list of modules to encode each signal.
        self.signalEncodingModules = nn.ModuleList()  # Use ModuleList to store child modules.
        # signalEncodingModules dimension: self.numSignals
        
        # For each signal.
        for signalInd in range(self.numSignals):                        
            # Find the mean
            self.signalEncodingModules.append(
                nn.Sequential(
                    # Neural architecture: Layer 1.
                    nn.Linear(self.embeddedDim, self.latentEmbeddingDim, bias = True),
                    nn.SELU(),
                )
            )

        self.latentTransformation = nn.Sequential(
            # Convolution architecture: Layer 1, Conv 1-3
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, dilation = 2, padding=4, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=1, dilation = 3, padding=3, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )

        # Positional encoding.
        self.positionEncoding = positionEncodings.positionalEncoding.T[0]
        
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """  
        # Extract the incoming data's dimension.
        batchSize, numSignals, compressedLength = inputData.size()
        # Assert that we have the expected data format.
        assert numSignals == self.numSignals, \
            f"You initialized {self.numSignals} signals but only provided {numSignals} signals."
        assert compressedLength == self.signalDimension, \
            f"You provided a signal of length {compressedLength}, but we expected {self.signalDimension}."

        # Create a new tensor to hold updated values.
        latentSignals = torch.zeros((batchSize, numSignals, self.latentDimension), requires_grad=False)

        # For each signal.
        for signalInd in range(self.numSignals):
            signalData = inputData[:, signalInd, :]
            # Embed the signal data by the local timepoints.
            embeddedSignalData = self.embedSignals(signalData, self.embeddedDim, self.signalEmbedding_indexSlices)
            # embeddedSignalData dimension: batchSize, numSegments, embeddedDim
               
            # Apply encoder to find the mean and log-variance mapping.
            latentData = self.signalEncodingModules[signalInd](embeddedSignalData)
            # latentData dimension: batchSize, numSegments, self.latentEmbeddingDim
            
            # Reshape the reconstructed data to get the final signal.
            reconstructedLatentData = self.signalDefragmentation(latentData, self.latentDimension, self.latentEmbeddingDim, self.latentReconstruction_indexSlices)
            # latentData dimension: batchSize, self.latentDimension

            # Store the encoded signals.
            latentSignals[:, signalInd, :] = reconstructedLatentData
            # Dimension: batchSize, numSignals, self.latentDimension
            
        # Calculate the mean latent representation for the batch
        averageLatentSignals = latentSignals.mean(dim=1, keepdim=True)  # Shape: (batchSize, 1, latentDimension)
            
        # # Process each signal through the common network to encourage a common latent space
        # commonLatentSignals = self.latentTransformation(averageLatentSignals)
        # # latentMeans and latentLogVars dimension: batchSize, 1, self.latentDimension
            
        # Combine the individual and common latent representations
        latentSignals = latentSignals + averageLatentSignals
        
        # TODO: backprop on mean = 0 and std = 1. reconstruct all signals from each latent

        return latentSignals

    def printParams(self):
        #signalEncoding(signalDimension = 64, latentDimension = 32, numSignals = 50).printParams()
        summary(self, (self.numSignals, self.signalDimension))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

# -------------------------------------------------------------------------- #
# -------------------------- Decoder Architecture -------------------------- #   

class signalDecoding(latentTransformationHead):
    def __init__(self, signalDimension=64, latentDimension=32, numSignals=50):
        super(signalDecoding, self).__init__(signalDimension, latentDimension)
        self.numSignals = numSignals

        # A list of modules to decode each signal.
        self.signalDecodingModules = nn.ModuleList()
        
        # For each signal.
        for signalInd in range(self.numSignals):
            # Reconstruct the signal from the latent dimension
            self.signalDecodingModules.append(
                nn.Sequential(
                    nn.Linear(self.latentEmbeddingDim, self.embeddedDim, bias=True),
                    nn.SELU(),
                )
            )
            
        self.latentTransformation = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.latentDimension, self.latentDimension, bias = True),
            nn.SELU(),
        )

    def forward(self, latentSignalData):
        """ The shape of latentSignalData: (batchSize, numSignals, latentDimension) """
        batchSize, numSignals, latentDimension = latentSignalData.size()
        
        # Assert we have the expected data format.
        assert numSignals == self.numSignals, \
            f"Expected {self.numSignals} signals but got {numSignals}."
        assert latentDimension == self.latentDimension, \
            f"Expected latent dimension of {self.latentDimension} but got {latentDimension}."

        # Create a new tensor to hold the reconstructed signals.
        reconstructedSignalData = torch.zeros((batchSize, numSignals, self.signalDimension), requires_grad=False)
        
        # Remap the latent space.
        # latentSignalData = self.latentTransformation(latentSignalData)
        
        # For each signal.
        for signalInd in range(self.numSignals):
            latentData = latentSignalData[:, signalInd, :]
                        
            # Embed the signal data by the local timepoints.
            embeddedLatentData = self.embedSignals(latentData, self.latentEmbeddingDim, self.latentReconstruction_indexSlices)
            # embeddedLatentData dimension: batchSize, numSegments, self.latentEmbeddingDim
                        
            # Apply decoder to reconstruct the signal from the latent space.
            semiReconstructedData = self.signalDecodingModules[signalInd](embeddedLatentData)
            # semiReconstructedData dimension: batchSize, numSegments, self.embeddedDim

            # Reshape the reconstructed data to get the final signal.
            reconstructedData = self.signalDefragmentation(semiReconstructedData, self.signalDimension, self.embeddedDim, self.signalEmbedding_indexSlices)
            # reconstructedData dimension: batchSize, self.signalDimension
            
            # Store the reconstructed signals.
            reconstructedSignalData[:, signalInd, :] = reconstructedData
            # reconstructedData dimension: batchSize, self.numSignals, self.signalDimension

        return reconstructedSignalData

    def printParams(self):
        #signalDecoding(signalDimension = 64, latentDimension = 32, numSignals = 50).printParams()
        summary(self, (self.numSignals, self.latentDimension))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        