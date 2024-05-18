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
import autoencoder   # Framwork for compression/decompression of time-series signals.

# Import global model
sys.path.append(os.path.dirname(__file__) + "/../../../")
import _globalPytorchModel

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class autoencoderModel(_globalPytorchModel.globalModel):
    def __init__(self, sequenceLength, compressedLength):
        super(autoencoderModel, self).__init__()
        # General model parameters.
        self.compressedLength = compressedLength  # The final length of the compressed signal after the autoencoder. MUST BE CHANGED IN AUTOENCODER.py
        self.sequenceLength = sequenceLength      # The length of each incoming signal: features used in the model.

        # ----------------------- Signal Compression ----------------------- #  
        
        # Method to remove unnecessary timepoints.
        self.compressSignals = autoencoder.encodingLayer(
                    compressedLength = self.compressedLength,
                    sequenceLength = self.sequenceLength,
        )
        
        # ---------------------- Signal Reconstruction --------------------- #  
        
        # Method to reconstruct the original signal.
        self.reconstructSignals = autoencoder.decodingLayer(
                    compressedLength = self.compressedLength,
                    sequenceLength = self.sequenceLength,
        )
        
        # ------------------------------------------------------------------ #
        
        # Reset the model
        self.resetModel()
    
    def resetModel(self):        
        # Autoencoder signal reconstructed loss holders.
        self.trainingLosses_signalReconstruction = []    # List of signal reconstruction (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_signalReconstruction = []     # List of signal reconstruction (autoencoder) testing losses. Dim: numEpochs
        
        # Autoencoder compressed mean loss holders.
        self.trainingLosses_compressedMean = []    # List of compressed mean (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_compressedMean = []     # List of compressed mean (autoencoder) testing losses. Dim: numEpochs
        # Autoencoder compressed standard deviation loss holders.
        self.trainingLosses_compressedSTD = []    # List of compressed standard deviation (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_compressedSTD = []     # List of compressed standard deviation (autoencoder) testing losses. Dim: numEpochs      

    def forward(self, signalData, reconstructSignals = False, trainingFlag = False, maxBatchSignals = 5000):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        
        # ----------------------- Data Preprocessing ----------------------- #  

        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, numSignals, sequenceLength = signalData.size()
        signalData = signalData.to(torch.float32) # Floats are required for gradient tracking.
        # signalData dimension: batchSize, numSignals, sequenceLength
        
        # Assert the integrity of the incoming data.
        assert sequenceLength == self.sequenceLength, f"The signals have length {sequenceLength}, but the model expected {self.sequenceLength} points."
              
        # Initialize holders for the output of each batch
        compressedData = torch.zeros((batchSize, numSignals, self.compressedLength), device=signalData.device)
        reconstructedData = torch.zeros((batchSize, numSignals, self.sequenceLength), device=signalData.device) if reconstructSignals else None
        
        # Calculate the size of each sub-batch
        subBatchSize = maxBatchSignals // numSignals
        numSubBatches = math.ceil(batchSize / subBatchSize)
        
        # For each sub-batch of data.
        for subBatchIdx in range(numSubBatches):
            # Calculate start and end indices for the current sub-batch
            startIdx = subBatchIdx * subBatchSize
            endIdx = min((subBatchIdx + 1) * subBatchSize, batchSize)

            # --------------------- Signal Compression --------------------- # 
                
            # Data reduction: remove unnecessary timepoints from the signals.
            compressedData[startIdx:endIdx] = self.compressSignals(signalData[startIdx:endIdx])  # Apply CNN for feature compression.
            # compressedData dimension: batchSize, numSignals, self.compressedLength
            
            # Clear the cache memory
            gc.collect(); torch.cuda.empty_cache();
            
            # -------------------- Signal Reconstruction ------------------- #  
            
            # If we are reconstructing the data.
            if reconstructSignals:    
                # Try and reconstruct the compressed data from the shallow (outer) CNN network.
                noisyCompressedData = compressedData[startIdx:endIdx].clone() + torch.rand(compressedData[startIdx:endIdx].size(), device=compressedData.device).uniform_(-0.05, 0.05) if trainingFlag else compressedData[startIdx:endIdx].clone()
                
                # Reconstruct the initial signals.
                reconstructedData[startIdx:endIdx] = self.reconstructSignals(noisyCompressedData)
                # reconstructedData dimension: batchSize, numSignals, sequenceLength
                
                # Clear the cache memory
                gc.collect(); torch.cuda.empty_cache();

        return compressedData, reconstructedData
            
        # ------------------------------------------------------------------ #  
    
    def shapInterface(self, signalData):
        # Ensure proper data format.
        batchSize, sequenceLength = signalData.shape
        signalDataTensor = torch.tensor(signalData.tolist())
                
        # Reshape the inputs to integrate into the model's expected format.
        signalDataTensor = signalDataTensor.unsqueeze(1)
        
        # predict the activities.
        compressedData, reconstructedData = self.forward(signalDataTensor, reconstructSignals = True)

        return reconstructedData.detach().cpu().numpy()
    
    
    