# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys

# PyTorch
import torch
import torch.nn as nn

# Import files for machine learning
from ...._globalPytorchModel import globalModel

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class sharedManifoldModel(globalModel):
    def __init__(self, manifoldLength, finalNumSignals):
        super(sharedManifoldModel, self).__init__()
        # General model parameters.
        self.finalNumSignals = finalNumSignals     # The final number of signals to input into the shared model.
        self.manifoldLength = manifoldLength       # The final length of each signal after projection.

        # ------------------------ Signal Encoding ------------------------- #  
        
        self.linearTransformation = nn.Parameter(torch.randn(self.manifoldLength, self.manifoldLength))
        
        # ------------------------------------------------------------------ #  
        
    def forward(self, manifoldData, remapSignals = False):
        """ The shape of inputData: (batchSize, finalNumSignals, manifoldLength) """
        
        # ----------------------- Data Preprocessing ----------------------- #  

        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, finalNumSignals, manifoldLength = manifoldData.size()
        # encodedData dimension: batchSize, numEncodedSignals, compressedLength
        
        # Assert the integrity of the incoming data.
        assert finalNumSignals == self.finalNumSignals, f"The model was expecting {self.finalNumSignals} signals, but recieved {finalNumSignals}"
        assert manifoldLength == self.manifoldLength, f"The signals have length {manifoldLength}, but the model expected {self.manifoldLength} points."
        
        # ------------------------ Signal Encoding ------------------------- # 
            
        # Create a linear transformation of the manifold data.
        transformedManifoldData = self.linearTransformation @ manifoldData.transpose(1, 2)
        # transformedManifoldData dimension: batchSize, manifoldLength, finalNumSignals
            
        # ------------------------------------------------------------------ #  
        
        return transformedManifoldData
    
    