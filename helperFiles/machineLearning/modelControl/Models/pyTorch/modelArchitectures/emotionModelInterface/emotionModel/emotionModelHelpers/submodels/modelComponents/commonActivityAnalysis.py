
# PyTorch
import torch.nn as nn
from torchsummary import summary


class commonActivityAnalysis(nn.Module):
    def __init__(self, numCommonFeatures, numActivityFeatures):
        super(commonActivityAnalysis, self).__init__()
        # General parameters.
        self.numCommonFeatures = numCommonFeatures
        self.numActivityFeatures = numActivityFeatures
        
        self.predictActivity = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(numCommonFeatures, 32, bias = True),
            nn.SELU(),
            
            # Neural architecture: Layer 2.
            nn.Linear(32, self.numActivityFeatures, bias = True),
            nn.SELU(),
            
            # Neural architecture: Layer 3.
            nn.Linear(self.numActivityFeatures, self.numActivityFeatures, bias = True),
            # nn.BatchNorm1d(self.numActivityFeatures, track_running_stats=True),
            nn.SELU(),
        )
            
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numFeatures) """ 
        # Specify the current input shape of the data.
        batchSize, numFeatures = inputData.size()
        assert numFeatures == self.numCommonFeatures, print(numFeatures, self.numCommonFeatures)
        
        # Classify the probability of each activity.
        activityDistribution = self.predictActivity(inputData)
        # activityDistribution dimension: batchSize, self.numActivityFeatures

        return activityDistribution
    
    def printParams(self):
        # commonActivityAnalysis(numCommonFeatures = 64, numActivityFeatures = 32).to('cpu').printParams()
        summary(self, (self.numCommonFeatures,))

    
    