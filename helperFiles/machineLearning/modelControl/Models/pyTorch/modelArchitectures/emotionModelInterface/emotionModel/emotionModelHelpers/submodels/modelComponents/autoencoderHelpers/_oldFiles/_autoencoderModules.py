# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/../modelHelpers/")
import _convolutionalHelpers

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class autoencoderModules(_convolutionalHelpers.convolutionalHelpers):
    def __init__(self, sequenceLength = 240, compressedLength = 64):
        super(autoencoderModules, self).__init__()
        # General shape parameters.
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength
        
        # Specify the amount of compression.
        self.compressionAmount = sequenceLength - compressedLength    
        
    # ----------------- Autoencoder-Specific Architectures ----------------- #
  
        