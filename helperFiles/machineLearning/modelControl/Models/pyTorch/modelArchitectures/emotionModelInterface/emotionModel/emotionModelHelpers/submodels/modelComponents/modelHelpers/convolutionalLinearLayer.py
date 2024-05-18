# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class convolutionalLinearLayer(nn.Module):
    def __init__(self, initialDimension, compressedDim, embeddedDim, compressedEmbeddedDim, embeddingStride):
        super(convolutionalLinearLayer, self).__init__()
        # General shape parameters.
        self.initialDimension = initialDimension    # The initial length of the sequence (int).
        self.compressedDim = compressedDim          # The compressed length of the sequence (int).
        
        # Signal embedding parameters
        self.compressedEmbeddedDim = compressedEmbeddedDim # The embedded dimension of the compressed sequence.
        self.embeddingStride = embeddingStride  # The number of points to skip between embeddings.
        self.embeddedDim = embeddedDim          # The embedded dimension of the initial sequence.
        
        # Signal embedding parameters.
        self.signalEmbedding_indexSlices = torch.arange(0, initialDimension - self.embeddedDim + 1, self.embeddingStride).long()
        self.compressedEmbedding_indexSlices =  torch.linspace(0, compressedDim - self.compressedEmbeddedDim, len(self.signalEmbedding_indexSlices)).long()
        assert len(self.signalEmbedding_indexSlices) == len(self.compressedEmbedding_indexSlices), "Must have the same number of segments to maintain 1:1 relationship."
        assert len(self.signalEmbedding_indexSlices) - 1 <= compressedDim - self.compressedEmbeddedDim, f"This will not work: {len(self.signalEmbedding_indexSlices)-1} !<= {compressedDim} - {self.compressedEmbeddedDim}"
        
        # Assert the integrity of signal embedding parameters.
        self.assertValidEmbedding(initialDimension, self.embeddedDim, self.signalEmbedding_indexSlices)
        self.assertValidEmbedding(compressedDim, self.compressedEmbeddedDim, self.compressedEmbedding_indexSlices)

    def assertValidEmbedding(self, initialDimension, embeddedDim, indexSlices):
        # Assert the integrity of start indices.
        assert indexSlices.diff()[0] <= embeddedDim, f"If you use these parameters you will not be using all the datapoints. See for yourself: {indexSlices} {embeddedDim}"
        assert indexSlices[-1] == initialDimension - embeddedDim, f"If you use these parameters you will miss the final points. See for yourself: {indexSlices}"
        assert len(indexSlices.diff().unique()) == 1, f"If you use these parameters you will not be evenly spaced. See for yourself: {indexSlices}"
        assert indexSlices[0] == 0, f"If you use these parameters you will miss the initial points. See for yourself: {indexSlices}"

    # --------------------------- Helper Methods --------------------------- #
    
    def embedSignals(self, inputData, embeddedDim, sliceIndices):        
        # Utilize tensor slicing and indexing to avoid explicit loops
        segmentedData = torch.stack([inputData[:, startInd:startInd + embeddedDim] for startInd in sliceIndices], dim=1)
        
        return segmentedData 

    def signalDefragmentation(self, segmentedData, initialDimension, embeddedDim, sliceIndices):        
        # Extract the incoming shape of the fragmented tensor.
        batchSize, numSegments, compressedDim = segmentedData.size()

        # Initialize the tensor to store holders.
        reconstructedData = torch.zeros(batchSize, initialDimension, device=segmentedData.device)
        contributions = torch.zeros_like(reconstructedData, device=segmentedData.device)
                    
        # Loop over each segment 
        for i, startInd in enumerate(sliceIndices):
            endInd = startInd + embeddedDim
                                    
            # place it in the reconstructed data tensor
            reconstructedData[:, startInd:endInd] = reconstructedData[:, startInd:endInd] + segmentedData[:, i, :]
            contributions[:, startInd:endInd] = contributions[:, startInd:endInd] + 1
    
        # Average the contributions from overlapping segments
        reconstructedData = reconstructedData / contributions
    
        return reconstructedData
    
    def forward(self, inputData, signalEncodingModule):
        """ The shape of inputData: (batchSize, initialDimension) """  
        # Assert that we have the expected data format.
        assert len(inputData.size()) == 2, inputData.size()
        assert inputData.size(-1) == self.initialDimension, \
            f"You provided a signal of dimensions {inputData.size()}, but we expected a last dimension of {self.initialDimension}."

        # Embed the signal data by the local timepoints.
        embeddedSignalData = self.embedSignals(inputData, self.embeddedDim, self.signalEmbedding_indexSlices)
        # embeddedSignalData dimension: batchSize, numSegments, embeddedDim
        
        # Apply encoder to compress the embedded dimension.
        latentSignalData = signalEncodingModule(embeddedSignalData)
        # latentSignalData dimension: batchSize, numSegments, compressedEmbeddedDim
            
        # Reshape the data to reconstruct the compressed signal.
        reconstructedLatentData = self.signalDefragmentation(latentSignalData, self.compressedDim, self.compressedEmbeddedDim, self.compressedEmbedding_indexSlices)
        # reconstructedLatentData dimension: batchSize, compressedDim
        
        # Assert the correct final output.
        assert reconstructedLatentData.size()[1] == self.compressedDim
        
        return reconstructedLatentData   

        