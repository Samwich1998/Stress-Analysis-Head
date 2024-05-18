# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn

# -------------------------------------------------------------------------- #
# --------------------------- Position Encoding ---------------------------- #

class signalEmbedding(nn.Module):
    def __init__(self, embeddingDim, numIndicesShift, clampBoundaries = False):
        super(signalEmbedding, self).__init__()
        # General model parameters.
        self.clampBoundaries = clampBoundaries  # Pad the shifted signal with the first/last value. If False, wrap the signal around.
        self.numIndicesShift = numIndicesShift  # The number of signal indices to shift. Note, preference is given for future values.
        self.embeddingDim = embeddingDim        # The number of shifted columns to output. If even, preference is given towards future values.
        
    def forward(self, inputData):
        """ The shape of inputData = (batchSize, sequenceLength)"""
        # Preprocess the data.
        inputData = inputData.unsqueeze(2)
        batchSize, sequenceLength, numSignals = inputData.size()
        assert numSignals == 1

        # Create an array to represent the indices for shifting the data.
        shift_indices = torch.arange(-self.numIndicesShift*(self.embeddingDim // 2), 
                                      self.numIndicesShift*(self.embeddingDim // 2) + 1, 
                                      self.numIndicesShift)[-self.embeddingDim:]
        # NOTE: For even embeddingDim, we will give preference to later timepoints.

        # Add shift_indices to the original indices to get the desired indices for embedding
        unboundedEmbedIndices = (shift_indices.view(1, 1, -1) + torch.arange(sequenceLength).view(1, -1, 1)) 
        
        if self.clampBoundaries:
            embed_indices = torch.clamp(unboundedEmbedIndices, min=0, max=sequenceLength - 1)
        else:
            embed_indices = unboundedEmbedIndices % sequenceLength

        # Reshape the tensor to match the size of the old implementation
        embeddedData = inputData[:, embed_indices, :] # Use advanced indexing to gather the values at the embedding indices
        embeddedData = embeddedData.view(batchSize, sequenceLength, -1)

        return embeddedData

    def testSpeed(self):
        # Define general parameters.
        clampBoundaries = False
        numIndicesShift = 1
        embeddingDim = 15
        batchSize = 64
        
        # Create the input data with the signals.
        inputData = torch.tensor([list(range(0, 15, 1))*batchSize]).view(batchSize, -1, 1)
        # Initialize the signal embedding class.
        signalEmbeddingClass = signalEmbedding(embeddingDim, numIndicesShift, clampBoundaries)
        
        import time
        t1 = time.time()
        embeddedData = signalEmbeddingClass.forward(inputData)
        t2 = time.time()
        print(t2-t1)
        print(embeddedData.size())
        print(embeddedData[0][0:25])
        
        signalEmbeddingClass.plotSignalEmbedding(embeddedData)
    
    # ---------------------------------------------------------------------- #
    # --------------------------- General Methods -------------------------- #
    
    def plotSignalEmbedding(self, embeddedData):
        a = 0
        for signal in embeddedData:
            plt.figure(figsize=(8,5))
            plt.pcolormesh(signal.detach().numpy().T, cmap='RdBu_r')
            plt.xlabel('Sequence Index')
            plt.ylabel('Embedding Dimension')
            plt.title('Signal Embedding')
            plt.colorbar()
            plt.show()
            
            plt.plot(signal[:, 0]);
            plt.show()
            
            a+=1
            if a == 1:
                break