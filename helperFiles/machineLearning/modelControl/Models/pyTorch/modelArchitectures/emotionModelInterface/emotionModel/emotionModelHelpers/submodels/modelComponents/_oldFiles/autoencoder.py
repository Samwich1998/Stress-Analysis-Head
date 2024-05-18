# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import time

# PyTorch
import torch.nn as nn
from torchsummary import summary

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/autoencoderHelpers/")
import autoencoder200s
import autoencoder100s
import autoencoder90s

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #
        
class encodingLayer(nn.Module):
    def __init__(self, sequenceLength, compressedLength):
        super(encodingLayer, self).__init__()  
        # General shape parameters.
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength

        if 200 <= sequenceLength:
            self.encoderModel = autoencoder200s.encodingLayer(sequenceLength, compressedLength)
        elif 100 <= sequenceLength:
            self.encoderModel = autoencoder100s.encodingLayer(sequenceLength, compressedLength)
        elif 64 <= sequenceLength:
            self.encoderModel = autoencoder90s.encodingLayer(sequenceLength, compressedLength)
        else:
            raise Exception(f"No Model Prepared for sequenceLength of {sequenceLength} and compressedLength of {compressedLength}")
                
        # ------------------------------------------------------------------ # 

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert self.sequenceLength == sequenceLength
                
        # ------------------------ CNN Architecture ------------------------ # 
        
        # Reshape the data to the expected input into the CNN architecture.
        signalData = inputData.view(batchSize * numSignals, 1, sequenceLength) # Seperate out indivisual signals.
        # signalData dimension: batchSize*numSignals, 1, sequenceLength

        # Apply CNN architecture to compress the data.
        compressedSignals = self.encoderModel(signalData)
        # signalData dimension: batchSize*numSignals, 1, compressedLength
        
        # Seperate put each signal into its respective batch.
        compressedData = compressedSignals.view(batchSize, numSignals, self.compressedLength) 
        # compressedData dimension: batchSize, numSignals, compressedLength
        
        # ------------------------------------------------------------------ # 
        
        return compressedData
    
    def printParams(self, numSignals = 2):
        # encodingLayer(sequenceLength = 240, compressedLength = 64).to('cpu').printParams(numSignals = 2)
        t1 = time.time()
        summary(self, (numSignals, self.sequenceLength,)) # summary(model, inputShape)
        t2 = time.time()
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        print("Total time:", t2 - t1)
    
class decodingLayer(nn.Module):
    def __init__(self, sequenceLength, compressedLength):
        super(decodingLayer, self).__init__()     
        # General shape parameters.
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength

        if 200 <= sequenceLength:
            self.decoderModel = autoencoder200s.decodingLayer(sequenceLength, compressedLength)
        elif 100 <= sequenceLength:
            self.decoderModel = autoencoder100s.decodingLayer(sequenceLength, compressedLength)
        elif 64 <= sequenceLength:
            self.decoderModel = autoencoder90s.decodingLayer(sequenceLength, compressedLength)
        else:
            raise Exception(f"No Model Prepared for sequenceLength of {sequenceLength} and compressedLength of {compressedLength}")
                
        # ------------------------------------------------------------------ # 
                
    def forward(self, compressedData):
        """ The shape of compressedData: (batchSize, numSignals, compressedLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, compressedLength = compressedData.size()
        assert self.compressedLength == compressedLength
        
        # ------------------------ CNN Architecture ------------------------ # 

        # Reshape the signals.
        compressedSignals = compressedData.view(batchSize*numSignals, 1, self.compressedLength) 
        # compressedSignals dimension: batchSize*numSignals, 1, self.compressedLength

        # Apply CNN architecture to decompress the data.
        decompressedSignals = self.decoderModel(compressedSignals)
        
        # Organize the signals into the original batches.
        reconstructedData = decompressedSignals.view(batchSize, numSignals, self.sequenceLength)
        # compressedSignals dimension: batchSize, numSignals, self.sequenceLength
        
        # ------------------------------------------------------------------ # 

        return reconstructedData
    
    def printParams(self, numSignals = 2):
        # decodingLayer(compressedLength = 64, sequenceLength = 240).to('cpu').printParams(numSignals = 2)
        summary(self, (numSignals, self.compressedLength,))
    

