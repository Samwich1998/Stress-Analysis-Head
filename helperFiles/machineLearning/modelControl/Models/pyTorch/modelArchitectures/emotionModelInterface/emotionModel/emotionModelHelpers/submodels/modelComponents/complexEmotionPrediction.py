# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class complexEmotionPrediction(nn.Module):
    def __init__(self, numEmotions, numBasicEmotions, numCommonFeatures):
        super(complexEmotionPrediction, self).__init__()
        # General parameters.
        self.numBasicEmotions = numBasicEmotions
        self.numEmotions = numEmotions
        self.numCommonFeatures = numCommonFeatures
        
        # Define a trainable weight that is subject-specific.
        self.allBasicEmotionWeights = nn.Parameter(torch.randn(self.numEmotions, 1, self.numBasicEmotions, 1))
        # self.allBasicEmotionWeights: For each emotion we are predicting, how should we recombine the basic emotions (basis states)
    
        # A list of modules to encode each signal.
        self.emotionModules = nn.ModuleList()  # Use ModuleList to store child modules.
        # signalEncodingModules dimension: self.numEmotions

        # Find the weights of each basic emotion.
        for emotionInd in range(self.numEmotions):
            # Model the complex emotions.
            self.emotionModules.append(nn.Sequential(
                    # Neural architecture: Layer 1.
                    nn.Linear(self.numCommonFeatures, 32, bias = True),
                    nn.SELU(),
                    
                    # Neural architecture: Layer 2.
                    nn.Linear(32, self.numBasicEmotions, bias = True),
                )
            )
            
    def forward(self, basicEmotionDistributions, encodedFeatures):
        """ 
        The shape of encodedFeatures: (batchSize, numCommonFeatures) 
        The shape of basicEmotionDistributions: (batchSize, numBasicEmotions, emotionLength) 
        """  
        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, numBasicEmotions, emotionLength = basicEmotionDistributions.size()
        assert encodedFeatures.shape[1] == self.numCommonFeatures
        assert numBasicEmotions == self.numBasicEmotions
                
        # Setup variables for predicting the final emotions.
        allBasicEmotionWeights = F.normalize(self.allBasicEmotionWeights.exp(), dim=2, p=1) # Dim: self.numEmotions, 1, self.numBasicEmotions, 1
        finalEmotionDistributions = torch.zeros((self.numEmotions, batchSize, emotionLength), device=basicEmotionDistributions.device) # Initialize a holder for all the final emotion predictions
        
        # For each emotion in the prediction model.
        for emotionInd in range(self.numEmotions):
            # Update how the basic emotions's recombine based on the activity.
            updatedBasicEmotionWeights = self.emotionModules[emotionInd](encodedFeatures).unsqueeze(2) # Dim: batchSize, self.numBasicEmotions, 1
            updatedBasicEmotionWeights = allBasicEmotionWeights[emotionInd] + updatedBasicEmotionWeights
            # updatedBasicEmotionWeights dimension: batchSize, self.numBasicEmotions, 1
            
            # Normalize the weight distribution.
            distributionSign = updatedBasicEmotionWeights.sign() # Keep track of the sign.
            normalizedBasicEmotionWeights = distributionSign*F.normalize(updatedBasicEmotionWeights.abs(), dim=1, p=1)
            # normalizedBasicEmotionWeights dimension: batchSize, self.numBasicEmotions, 1
            
            # Use the basic emotion basis states to predict the final emotion distribution.
            finalEmotionDistribution = ((basicEmotionDistributions * normalizedBasicEmotionWeights).sum(dim = 1))
            # finalEmotionDistributions[emotionInd, :, :] = F.softmax(finalEmotionDistribution, dim=1, p=1) 
            finalEmotionDistributions[emotionInd, :, :] = F.softmax(finalEmotionDistribution, dim=1) 
            # finalEmotionDistributions dimension: self.numEmotions, batchSize, emotionLength

        return finalEmotionDistributions
    
    def printParams(self, emotionLength = 25):
        # complexEmotionPrediction(numEmotions = 30, numBasicEmotions = 6, numCommonFeatures = 64).printParams()
        summary(self, (self.numCommonFeatures,), (self.numBasicEmotions, emotionLength,))

    
    