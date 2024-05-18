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

class variationalAutoencoderParameters(nn.Module):
    def __init__(self, signalDimension = 75, latentDimension = 16):
        super(variationalAutoencoderParameters, self).__init__()
        # General shape parameters.
        self.signalDimension = signalDimension
        self.latentDimension = latentDimension
        
        # Signal embedding parameters
        self.latentEmbeddingDim = 5
        self.numSegments = 12
        self.embeddedDim = 9  # The sliding window for each signal.
        
        # Signal embedding parameters.
        self.signalEmbedding_indexSlices = torch.linspace(0, signalDimension - self.embeddedDim, self.numSegments).long()
        self.latentReconstruction_indexSlices = torch.linspace(0, latentDimension - self.latentEmbeddingDim, self.numSegments).long()
        
        # Assert the integrity of signal embedding parameters.
        assert self.embeddedDim % 2 == 1, f"Embedded dimension must be odd so that we dont have any bias: {self.embeddedDim}"
        assert signalDimension <= self.embeddedDim * self.numSegments, f"We need to use all the data points: {signalDimension}, {self.embeddedDim}, {self.numSegments}"
        assert (signalDimension - self.embeddedDim) % (self.numSegments - 1) == 0, f"If you use these parameters you will not be evenly spaced. See for yourself: {self.signalEmbedding_indexSlices}"
        assert len(self.signalEmbedding_indexSlices.diff().unique()) == 1, f"If you use these parameters you will not be evenly spaced. See for yourself: {self.signalEmbedding_indexSlices}"
        # Assert the integrity of latent space reconstruction.
        assert self.latentEmbeddingDim % 2 == 1, f"The latent dimension must be odd so that we dont have any bias when recombining: {self.latentEmbeddingDim}"
        assert latentDimension <= self.latentEmbeddingDim * self.numSegments, f"We need to use all the data points: {latentDimension}, {self.latentEmbeddingDim}, {self.numSegments}"
        assert (latentDimension - self.latentEmbeddingDim) % (self.numSegments - 1) == 0, f"If you use these parameters you will not be evenly spaced. See for yourself: {self.latentReconstruction_indexSlices}"
        assert len(self.latentReconstruction_indexSlices.diff().unique()) == 1, f"If you use these parameters you will not be evenly spaced. See for yourself: {self.latentReconstruction_indexSlices}"

    # --------------------------- Helper Methods --------------------------- #
    
    def embedSignals(self, inputData, embeddedDim, sliceIndices):        
        # Utilize tensor slicing and indexing to avoid explicit loops
        segmentedData = torch.stack([inputData[:, startInd:startInd + embeddedDim] for startInd in sliceIndices], dim=1)
        
        return segmentedData 

    def signalDefragmentation(self, segmentedData, signalDimension, embeddedDim, sliceIndices):
        batchSize, numSignals, latentDimension = segmentedData.size()

        # Initialize the tensor to store the defragmented data
        reconstructedData = torch.zeros(batchSize, signalDimension)
    
        # Count the number of contributions to each data point for later averaging
        contributions = torch.zeros_like(reconstructedData)
    
        # Loop over each segment 
        for i, startInd in enumerate(sliceIndices):
            # place it in the reconstructed data tensor
            reconstructedData[:, startInd:startInd + embeddedDim] += segmentedData[:, i, :]
            contributions[:, startInd:startInd + embeddedDim] += 1
    
        # Average the contributions from overlapping segments
        reconstructedData /= contributions
    
        return reconstructedData
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(variationalAutoencoderParameters):
    def __init__(self, signalDimension = 75, latentDimension = 16, numSignals = 50):
        super(signalEncoding, self).__init__(signalDimension, latentDimension)
        # General parameters.
        self.numSignals = numSignals            # The number of independant signals in the model.

        # A list of modules to encode each signal.
        self.signalEncodingModules_Mean = nn.ModuleList()  # Use ModuleList to store child modules.
        self.signalEncodingModules_LogVar = nn.ModuleList()  # Use ModuleList to store child modules.
        # signalEncodingModules dimension: self.numSignals
        
        # For each signal.
        for signalInd in range(self.numSignals):                        
            # Find the mean
            self.signalEncodingModules_Mean.append(
                nn.Sequential(
                    # Neural architecture: Layer 1.
                    nn.Linear(self.embeddedDim, self.latentEmbeddingDim, bias = True),
                    nn.SELU(),
                )
            )
            
            # Find the mean
            self.signalEncodingModules_LogVar.append(
                nn.Sequential(
                    # Neural architecture: Layer 1.
                    nn.Linear(self.embeddedDim, self.latentEmbeddingDim, bias = True),
                    nn.SELU(),
                )
            )
            
        self.latentMeanTransformation = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.latentDimension, self.latentDimension, bias = True),
            nn.SELU(),
        )
        
        self.latentLogVarTransformation = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.latentDimension, self.latentDimension, bias = True),
            nn.SELU(),
        )
        
        # Positional encoding.
        self.positionEncoding = positionEncodings.positionalEncoding.T[0]
  
    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()    # Dim: batchSize, latentDimension
        eps = torch.randn_like(std)   # Dim: batchSize, latentDimension
        return mu + eps * std  

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
        latentSignalMeans = torch.zeros((batchSize, numSignals, self.latentDimension), requires_grad=False)
        latentSignalLogVars = torch.zeros((batchSize, numSignals, self.latentDimension), requires_grad=False)
        
        # Add positional encoding.
        inputData = inputData + self.positionEncoding
        
        # For each signal.
        for signalInd in range(self.numSignals):
            signalData = inputData[:, signalInd, :]
            # Embed the signal data by the local timepoints.
            embeddedSignalData = self.embedSignals(signalData, self.embeddedDim, self.signalEmbedding_indexSlices)
            # embeddedSignalData dimension: batchSize, numSegments, embeddedDim
               
            # Apply encoder to find the mean and log-variance mapping.
            latentMeans = self.signalEncodingModules_Mean[signalInd](embeddedSignalData)
            latentLogVars = self.signalEncodingModules_LogVar[signalInd](embeddedSignalData)
            # latentMeans and latentLogVars dimension: batchSize, numSegments, self.latentEmbeddingDim
            
            # Reshape the reconstructed data to get the final signal.
            reconstructedLatentMeans = self.signalDefragmentation(latentMeans, self.latentDimension, self.latentEmbeddingDim, self.latentReconstruction_indexSlices)
            reconstructedLatentLogVars = self.signalDefragmentation(latentLogVars, self.latentDimension, self.latentEmbeddingDim, self.latentReconstruction_indexSlices)
            # latentMeans and latentLogVars dimension: batchSize, self.latentDimension

            # Store the encoded signals.
            latentSignalMeans[:, signalInd, :] = reconstructedLatentMeans
            latentSignalLogVars[:, signalInd, :] = reconstructedLatentLogVars
            # Dimension: batchSize, numSignals, self.latentDimension
            
        # Calculate the mean latent representation for the batch
        averageLatentMeans = latentSignalMeans.mean(dim=1, keepdim=True)  # Shape: (batchSize, 1, latentDimension)
        averageLatentLogVars = latentSignalLogVars.mean(dim=1, keepdim=True)  # Shape: (batchSize, 1, latentDimension)
    
        # Process each signal through the common network to encourage a common latent space
        commonLatentMeans = self.latentMeanTransformation(averageLatentMeans)
        commonLatentLogVars = self.latentLogVarTransformation(averageLatentLogVars)  
        # latentMeans and latentLogVars dimension: batchSize, 1, self.latentDimension
    
        # Combine the individual and common latent representations
        latentSignalMeans = (latentSignalMeans + commonLatentMeans) / 2
        latentSignalLogVars = (latentSignalLogVars + commonLatentLogVars) / 2
        
        # Reparametrize to the latent space.
        latentData = self.reparameterize(latentSignalMeans, latentSignalLogVars)

        return latentData, latentSignalMeans, latentSignalLogVars

    def printParams(self):
        #signalEncoding(signalDimension = 75, latentDimension = 16, numSignals = 50).printParams()
        summary(self, (self.numSignals, self.signalDimension))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

# -------------------------------------------------------------------------- #
# -------------------------- Decoder Architecture -------------------------- #   

class signalDecoding(variationalAutoencoderParameters):
    def __init__(self, signalDimension=75, latentDimension=16, numSignals=50):
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
        
        # For each signal.
        for signalInd in range(self.numSignals):
            latentData = latentSignalData[:, signalInd, :]
            
            # Embed the signal data by the local timepoints.
            embeddedLatentData = self.embedSignals(latentData, self.latentEmbeddingDim, self.latentReconstruction_indexSlices)
            # embeddedLatentData dimension: batchSize, self.numSegments, self.latentEmbeddingDim
            
            # Apply decoder to reconstruct the signal from the latent space.
            semiReconstructedData = self.signalDecodingModules[signalInd](embeddedLatentData)
            # semiReconstructedData dimension: batchSize, self.numSegments, self.embeddedDim

            # Reshape the reconstructed data to get the final signal.
            reconstructedData = self.signalDefragmentation(semiReconstructedData, self.signalDimension, self.embeddedDim, self.signalEmbedding_indexSlices)
            # reconstructedData dimension: batchSize, self.signalDimension
            
            # Store the reconstructed signals.
            reconstructedSignalData[:, signalInd, :] = reconstructedData
            # reconstructedData dimension: batchSize, self.numSignals, self.signalDimension

        return reconstructedSignalData

    def printParams(self):
        #signalDecoding(signalDimension = 75, latentDimension = 16, numSignals = 50).printParams()
        summary(self, (self.numSignals, self.latentDimension))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        