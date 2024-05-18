# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import math
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn

# -------------------------------------------------------------------------- #
# --------------------------- Position Encoding ---------------------------- #

class PositionalEncoding(nn.Module):
    def __init__(self, embeddingDim = 32, numTokens = 120, n = 10.0, maxDeviation = 0.1):
        """
        Parameters
        ----------
        embeddingDim : The number of bits used to describe the position.
        numTokens : The length of the input sequence data.
        n : User-defined scalar, set to 10,000 by the authors of Attention Is All You Need.
        """
        super(PositionalEncoding, self).__init__()
        # Assert the integrity of the input parameters.
        assert embeddingDim % 2 == 0 or embeddingDim == 1, "embeddingDim must be an even number"
        
        # General model parameters.
        self.numTokens = numTokens
        self.embeddingDim = embeddingDim
        self.maxDeviation = maxDeviation
        
        # General position encoding parameters.
        sinusoidalIndices = torch.arange(0, self.embeddingDim, 2).float() # Mapping column indices to sin (even) or cosine (odd))
        
        # Develop the encoding scheme.
        sinusoidalDecay = self.exponentialPositionalEncoding(sinusoidalIndices, n)
        positionIndices = torch.arange(0, self.numTokens, dtype=torch.float).unsqueeze(1)
        # Encoded the indices of each signal: self.numTokens x self.embeddingDim
        positionalEncoding = torch.zeros(self.numTokens, self.embeddingDim)
        positionalEncoding.data[:, 0::2] = torch.sin(positionIndices * sinusoidalDecay) * self.maxDeviation
        positionalEncoding.data[:, 1::2] = torch.cos(positionIndices * sinusoidalDecay) * self.maxDeviation
        # Register the position encoding as a non-trainable parameter.
        self.register_buffer('positionalEncoding', positionalEncoding)
                
        # Assert the uniqueness of each position encoding map.
        self.assertUniquePositions()

    def forward(self, inputData):
        """ The shape of inputData = (numBatches, numTimePoints, embeddingDim)"""        
        # Assert the integrity of the input data.
        batchSize, sequenceLength, embeddingDim = inputData.size()
        assert sequenceLength <= self.numTokens, "You have too many timepoints in your signal."
        assert embeddingDim == self.embeddingDim, "Your signals do not match the expected embedding dimension."
        
        # Add positional encoding based on the sequence length of inputData
        positionalEncodings = self.positionalEncoding[0:sequenceLength, :] # Get the positional encodings
        inputData = inputData + positionalEncodings # Apply the positional encoding to the data.
        return inputData
    
    # ---------------------------------------------------------------------- #
    # -------------------- Types of Positional Encoding -------------------- #
    
    def absolutePositionalEncoding(self, sinusoidalIndices, n):
        """ Original Encoding in the 'Attentional is All You Need' Paper """
        return torch.pow(n, -2*sinusoidalIndices/self.embeddingDim)
    
    def exponentialPositionalEncoding(self, sinusoidalIndices, n):
        return torch.exp(sinusoidalIndices * (-math.log(n) / self.embeddingDim))
    
    def rotaryPositionalEncoding(self):
        """ Encoding in 'Roformer: enhanced transformer with rotary position embedding' Paper """
        pass
    
    # ---------------------------------------------------------------------- #
    # --------------------------- General Methods -------------------------- #
    
    def plotpositionalEncoding(self):
        plt.figure(figsize=(8, 5))
        plt.pcolormesh(PositionalEncoding(1, 18, 5).positionalEncoding.detach().numpy().T, cmap='RdBu_r')
        plt.xlabel('Sequence Index')
        plt.ylabel('Encoding Dimension')
        plt.title('Positional Encoding')
        plt.colorbar()
        plt.show()
        
    def assertUniquePositions(self):
        # Check if columns are unique
        unique_columns = torch.unique(self.positionalEncoding, dim=0)
        assert unique_columns.size(0) == self.positionalEncoding.size(0), \
            f"You have not properly encoded the sequence. Found {unique_columns.size(0)} unique \
            sequence positions when there are {self.positionalEncoding.size(0)} points."
            
            
