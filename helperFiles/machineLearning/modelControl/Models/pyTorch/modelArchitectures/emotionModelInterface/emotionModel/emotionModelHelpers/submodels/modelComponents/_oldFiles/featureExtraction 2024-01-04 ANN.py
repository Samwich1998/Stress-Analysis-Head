# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class featureExtraction(nn.Module):
    def __init__(self, numCommonFeatures = 32, numEncodedSignals = 64, manifoldLength = 32):
        super(featureExtraction, self).__init__()
        # General parameters.
        self.numEncodedSignals = numEncodedSignals      # The dimensionality of the encoded data.
        self.numCommonFeatures = numCommonFeatures      # The number of features from considering all the signals.
        self.manifoldLength = manifoldLength            # The final length of each signal after projection.
        self.numSignalFeatures = 4
        # Tunable parameters.
        self.finalNumSignals = int(numEncodedSignals/4)
                
        # Reduce the size of the features.
        self.signalReduction = nn.Sequential(
                # Neural architecture: Layer 1.
                nn.Linear(self.numEncodedSignals, int(self.numEncodedSignals/2), bias = True),
                nn.SELU(),
                
                # Neural architecture: Layer 2.
                nn.Linear(int(self.numEncodedSignals/2), self.finalNumSignals, bias = True),
                nn.SELU(),
        )
        
        # Learn signal features.
        self.extractSignalFeatures = nn.Sequential(
                # Neural architecture: Layer 1.
                nn.Linear(self.manifoldLength, 8, bias = True),
                nn.SELU(),
                
                # Neural architecture: Layer 2.
                nn.Linear(8, self.numSignalFeatures, bias = True),
                nn.SELU(),
        )
    
        # Learn final features.
        self.extractFeatures = nn.Sequential(
                # Neural architecture: Layer 1.
                nn.Linear(self.numSignalFeatures*self.finalNumSignals, 32, bias = True),
                nn.BatchNorm1d(32, affine = True, momentum = 0.1, track_running_stats=True),
                nn.SELU(),
                
                # Neural architecture: Layer 2.
                nn.Linear(32, self.numCommonFeatures, bias = True),
                nn.BatchNorm1d(self.numCommonFeatures, affine = True, momentum = 0.1, track_running_stats=True),
                nn.SELU(),
        )
            
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numEncodedSignals, manifoldLength) """  
        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, numEncodedSignals, manifoldLength = inputData.size()
        
        # Assert we have the expected data format.
        assert manifoldLength == self.manifoldLength, f"Expected manifold dimension of {self.manifoldLength}, but got {manifoldLength}."
        assert numEncodedSignals == self.numEncodedSignals, f"Expected {self.numEncodedSignals} signals, but got {numEncodedSignals} signals."
        
        # ---------------------- CNN Feature Reduction --------------------- # 

        # Create a channel for the signal data.
        inputData = inputData.transpose(1, 2)
        # inputData dimension: batchSize, manifoldLength, numEncodedSignals
        
        # Reduce the information by half.
        reducedSignals = self.signalReduction(inputData)
        # reducedSignals dimension: batchSize, manifoldLength, finalNumSignals
        
        # Remove the extra channel.
        reducedSignals = reducedSignals.transpose(2, 1)
        # inputData dimension: batchSize, finalNumSignals, manifoldLength
                
        # -------------------- Signal Feature Extraction ------------------- # 

        # Apply CNN architecture to decompress the data.
        signalFeatureData = self.extractSignalFeatures(reducedSignals)
        # signalFeatureData dimension: batchSize, finalNumSignals, numSignalFeatures
        
        # -------------------- Final Feature Extraction -------------------- # 
        
        # Flatten the data to synthesize all the features together.
        featureData = signalFeatureData.view(batchSize, self.finalNumSignals*self.numSignalFeatures)
        # featureData dimension: batchSize, finalNumSignals*numSignalFeatures
        
        # Extract the final features from the signals.
        finalFeatures = self.extractFeatures(featureData)
        # featureData dimension: batchSize, numCommonFeatures

        # ------------------------------------------------------------------ # 

        return finalFeatures
    
    def printParams(self):
        # featureExtraction(numCommonFeatures = 32, numEncodedSignals = 64, manifoldLength = 32).to('cpu').printParams()
        summary(self, (self.numEncodedSignals, self.manifoldLength,))
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

    
    