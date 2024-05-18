# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys

# PyTorch
import torch

# Import helper files.
sys.path.append(os.path.dirname(__file__) + "/modelComponents/")
import latentEncoder     # Map each signal into a common latent space.

# Import global model
sys.path.append(os.path.dirname(__file__) + "/../../../")
import _globalPytorchModel

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class latentEncoderModel(_globalPytorchModel.globalModel):
    def __init__(self, combinedSignalLength, latentLength, featureNames):
        super(latentEncoderModel, self).__init__()
        # General model parameters.
        self.combinedSignalLength = combinedSignalLength  # The final length of the compressed signal after the autoencoder. MUST BE CHANGED IN AUTOENCODER.py
        self.numSignals = len(featureNames)       # The number of signals going into the model.
        self.featureNames = featureNames          # The names of each feature/signal in the model. Dim: numSignals 
        self.latentLength = latentLength          # The final length of the compressed signal after the autoencoder.

        # ------------------------ Signal Encoding ------------------------- #  
        
        # Method to find a common signal-agnostic space.
        self.encodeSignals = latentEncoder.signalEncoding(
                    signalDimension = self.combinedSignalLength,
                    latentDimension = self.latentLength,
                    numSignals = self.numSignals,
        )
        
        # ------------------------ Signal Decoding ------------------------- #  
        
        # Method to reconstruct the signal-specific information.
        self.decodeSignals = latentEncoder.signalDecoding(
                    signalDimension = self.combinedSignalLength,
                    latentDimension = self.latentLength,
                    numSignals = self.numSignals,
        )
        
        # ------------------------------------------------------------------ #  
        
        # Reset the model
        self.resetModel()
    
    def resetModel(self):        
        # Compressed signal reconstruction loss holders.
        self.trainingLosses_latentReconstruction = []  # List of latent reconstruction (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_latentReconstruction = []   # List of latent reconstruction (autoencoder) testing losses. Dim: numEpochs
        # Latent sparsity loss holders.
        self.trainingLosses_latentSparsity = []      # List of latent sparsity training losses. Dim: numEpochs
        self.testingLosses_latentSparsity = []       # List of latent sparsity testing losses. Dim: numEpochs
        
    def forward(self, compressedData, reconstructSignals = False):
        """ The shape of inputData: (batchSize, combinedSignalLength) """
        
        # ----------------------- Data Preprocessing ----------------------- #  

        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, combinedSignalLength = compressedData.size()
        # signalData dimension: batchSize, combinedSignalLength
        
        # Assert the integrity of the incoming data.
        assert self.numSignals == numSignals, f"The model was expecting {self.numSignals} signals, but recieved {numSignals}"
        assert combinedSignalLength == self.combinedSignalLength, f"The signals have length {combinedSignalLength}, but the model expected {self.combinedSignalLength} points."
        
        # ------------------------ Signal Encoding ------------------------- # 
            
        # Map each signal into a common shared latent space.
        latentData = self.encodeSignals(compressedData)
        # encodedData dimension: batchSize, latentLength
        
        # ---------------------- Signal Reconstruction --------------------- #  
        
        reconstructedCompressedData = None
        # If we are reconstructing the signals.
        if reconstructSignals:    
            # Reconstruct the compressed signals.
            reconstructedCompressedData = self.decodeSignals(latentData)
            # reconstructedData dimension: batchSize, numSignals, combinedSignalLength

        return latentData, reconstructedCompressedData
            
        # ------------------------------------------------------------------ #  
    
    # DEPRECATED
    def shapInterface(self, reshapedSignalFeatures):
        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, numFeatures = reshapedSignalFeatures.shape
        reshapedSignalFeatures = torch.tensor(reshapedSignalFeatures.tolist())
        assert numFeatures == self.numSignals*self.numSignalFeatures, f"{numFeatures} {self.numSignals} {self.numSignalFeatures}"
                
        # Reshape the inputs to integrate into the model's expected format.
        signalFeatures = reshapedSignalFeatures.view((batchSize, self.numSignals, self.numSignalFeatures))
        
        # predict the activities.
        activityDistribution = self.forward(signalFeatures, predictActivity = True, allSignalFeatures = True)

        return activityDistribution.detach().numpy()
    
    
    