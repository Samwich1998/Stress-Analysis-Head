# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class subjectInterpretation(nn.Module):
    def __init__(self, numActivities, numSubjects, numInterpreterHeads, numBasicEmotions):
        super(subjectInterpretation, self).__init__()
        # General parameters.
        self.numSubjects = numSubjects
        self.numActivities = numActivities
        self.numBasicEmotions = numBasicEmotions
        self.numInterpreterHeads = numInterpreterHeads
        
        # Define a trainable weight that is subject-specific.
        self.allSubjectWeights = nn.Parameter(torch.randn(self.numSubjects, self.numInterpreterHeads, self.numBasicEmotions, 1))
        # self.allSubjectWeights: How does each subject utilize each interpretation for every basic emotion.
    
        # Find the weights of each interpretation mindset.
        self.interpretActivity = nn.Sequential(
                # Neural architecture: Layer 1.
                nn.Linear(self.numActivities, 16, bias = True),
                nn.SELU(),
                
                # Neural architecture: Layer 2.
                nn.Linear(16, self.numInterpreterHeads, bias = True),
        )
            
    def forward(self, allBasicEmotionDistributions, activityDistribution, subjectInds):
        """ 
        The shape of activityDistribution: (batchSize, numActivities) 
        The shape of allBasicEmotionDistributions: (batchSize, numInterpreterHeads, numBasicEmotions, emotionLength) 
        """  
        # Assert proper incoming data format.
        assert activityDistribution.shape[1] == self.numActivities
        assert allBasicEmotionDistributions.shape[1] == self.numInterpreterHeads
        assert allBasicEmotionDistributions.shape[2] == self.numBasicEmotions
        
        # Normalize the weights of the distributions.
        allSubjectWeights = F.normalize(self.allSubjectWeights.exp()[subjectInds, :, :, :], dim = 1, p=1)
        # allSubjectWeights dimension: batchSize, self.numInterpreterHeads, self.numBasicEmotions, 1
        
        # Update the mood/mindset's weights based on the activity.
        updatedSubjectWeights = self.interpretActivity(activityDistribution).unsqueeze(2).unsqueeze(3) # allSubjectWeights dimension: batchSize, self.numInterpreterHeads, 1, 1
        updatedSubjectWeights = allSubjectWeights + updatedSubjectWeights
        # updatedSubjectWeights dimension: batchSize, self.numInterpreterHeads, self.numBasicEmotions, 1
        
        # Normalize the weight distribution.
        distributionSign = updatedSubjectWeights.sign() # Keep track of the sign.
        normalizedSubjectWeights = distributionSign*F.normalize(updatedSubjectWeights.abs(), dim = 1, p=1)
        # normalizedSubjectWeights dimension: batchSize, self.numInterpreterHeads, self.numBasicEmotions, 1
                
        # Average out each mindet's emotional distributions.
        basicEmotionDistributions = (allBasicEmotionDistributions * normalizedSubjectWeights).sum(dim = 1)
        # basicEmotionDistributions dimension: batchSize, self.numBasicEmotions, self.emotionLength

        return basicEmotionDistributions
    
    
    