# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class encodingLayer(nn.Module):
    def __init__(self, sequenceLength, compressedLength):
        super(encodingLayer, self).__init__()
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength
        
        # Learn final features
        self.compressSignalsFC = nn.Sequential(                        
            # Neural architecture: Layer 1
            nn.Linear(self.sequenceLength, self.compressedLength, bias = True),
            nn.BatchNorm1d(64, affine = True, momentum = 0.1, track_running_stats = True),
        )

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert self.sequenceLength == sequenceLength
        
        # Reshape the data to the expected input into the CNN architecture.
        signalData = inputData.view(batchSize * numSignals, sequenceLength) # Seperate out indivisual signals.
        # signalData dimension: batchSize*numSignals, sequenceLength
        
        # Apply FC architecture to reduce spatial dimension.
        compressedSignals = self.compressSignalsFC(signalData) # The new dimension: batchSize*numSignals, self.compressedLength
        compressedData = compressedSignals.view(batchSize, numSignals, self.compressedLength) # Seperate put each signal into its respective batch.
        # compressedData dimension: batchSize, numSignals, self.compressedLength
                
        return compressedData
    
    def printParams(self, numSignals = 50, sequenceLength = 300):
        #encodingLayer(sequenceLength = 300, compressedLength = 32).printParams(numSignals = 75, sequenceLength = 300)
        summary(self, (numSignals, sequenceLength,)) # summary(model, inputShape)
        
    
class decodingLayer(nn.Module):
    def __init__(self, compressedLength = 32, sequenceLength = 300):
        super(decodingLayer, self).__init__()
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength
                
        # Learn final features
        self.expandSignals = nn.Sequential(                        
            # Neural architecture: Layer 1
            nn.Linear(self.compressedLength, self.sequenceLength, bias = True),
        )
                
    def forward(self, compressedData):
        """ The shape of compressedData: (batchSize, numSignals, compressedLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, compressedLength = compressedData.size()
        assert self.compressedLength == compressedLength
        
        # Reconstruct a compressed signal.
        compressedSignals = compressedData.view(batchSize*numSignals, self.compressedLength) # Seperate put each signal into its respective batch.
        # compressedSignals dimension: batchSize*numSignals, self.compressedLength

        # Apply a FC architecture to reconstruct the signals.
        decompressedSignals = self.expandSignals(compressedSignals) # The new dimension: batchSize*numSignals, self.sequenceLength
        reconstructedData = decompressedSignals.view(batchSize, numSignals, self.sequenceLength)   # Organize the signals into the original batches.
        # reconstructedData dimension: batchSize, numSignals, sequenceLength

        return reconstructedData
    
    def printParams(self, numSignals = 75, compressedLength = 32, sequenceLength = 300):
        #decodingLayer(compressedLength = 32, sequenceLength = 300).printParams()
        summary(self, (numSignals, compressedLength,))
    

    
    
    