# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import gc
import sys
import math

# PyTorch
import torch

# Import helper files.
sys.path.append(os.path.dirname(__file__) + "/modelComponents/")
import signalEncoder   # Framwork for encoding/decoding of all signals.

# Import global model
sys.path.append(os.path.dirname(__file__) + "/../../../")
import _globalPytorchModel

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class signalDecoderModel(_globalPytorchModel.globalModel):
    def __init__(self, compressedLength, numEncodedSignals, featureNames):
        super(signalDecoderModel, self).__init__()
        # General model parameters.
        self.numEncodedSignals = numEncodedSignals  # The final number of signals to accept, encoding all signal information.
        self.compressedLength = compressedLength   # The initial length of each incoming signal.
        self.numSignals = len(featureNames)        # The number of signals from the specific dataset.
        self.featureNames = featureNames           # The names of each feature/signal in the model. Dim: numSignals 

        # ---------------------- Signal Reconstruction --------------------- #  
        
        # Method to reconstruct the original signal.
        self.reconstructSignals = signalEncoder.signalDecoding(
                    numEncodedSignals = self.numEncodedSignals, 
                    signalDimension = self.compressedLength, 
                    numSignals = self.numSignals,
        )
        
        # ------------------------------------------------------------------ #
        
        # Reset the model
        self.resetModel()
    
    def resetModel(self):        
        # Autoencoder signal reconstructed loss holders.
        self.trainingLosses_signalReconstruction = []    # List of signal reconstruction (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_signalReconstruction = []     # List of signal reconstruction (autoencoder) testing losses. Dim: numEpochs    

    def forward(self, encodedData, sortingIndices, maxBatchSignals):
        """ The shape of inputData: (batchSize, numEncodedSignals, compressedLength) """
        
        # ----------------------- Data Preprocessing ----------------------- #  

        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, numEncodedSignals, compressedLength = encodedData.size()
        # encodedData dimension: batchSize, numEncodedSignals, compressedLength
        
        # Assert the integrity of the incoming data.
        assert compressedLength == self.compressedLength, f"The signals have length {compressedLength}, but the model expected {self.compressedLength} points."
        assert numEncodedSignals == self.numEncodedSignals, f"The have {numEncodedSignals} condensed signals, but the model expected {self.numEncodedSignals}."
          
        # Initialize holders for the output of each batch
        reconstructedData = torch.zeros((batchSize, self.numSignals, self.compressedLength), device=encodedData.device)
        
        # Calculate the size of each sub-batch
        subBatchSize = maxBatchSignals // self.numSignals
        numSubBatches = math.ceil(batchSize / subBatchSize)
        
        # For each sub-batch of data.
        for subBatchIdx in range(numSubBatches):
            # Calculate start and end indices for the current sub-batch
            startIdx = subBatchIdx * subBatchSize
            endIdx = min((subBatchIdx + 1) * subBatchSize, batchSize)
            
            # -------------------- Signal Reconstruction ------------------- #  
        
            # Reconstruct the initial signals.
            reconstructedData[startIdx:endIdx] = self.reconstructSignals(encodedData[startIdx:endIdx], sortingIndices[startIdx:endIdx])
            # reconstructedData dimension: batchSize, numSignals, compressedLength
            
            # Clear the cache memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # -------------------------------------------------------------- #  

        return reconstructedData
            
        # ------------------------------------------------------------------ #  
        
    # DEPRECATED
    def shapInterface(self, encodedData):
        # Ensure proper data format.
        batchSize, compressedLength = encodedData.shape
        encodedDataTensor = torch.tensor(encodedData.tolist())
                
        # Reshape the inputs to integrate into the model's expected format.
        encodedDataTensor = encodedDataTensor.unsqueeze(1)
        
        # predict the activities.
        compressedData, reconstructedData = self.forward(encodedDataTensor, reconstructData = True)

        return reconstructedData.detach().cpu().numpy()
    
    
    