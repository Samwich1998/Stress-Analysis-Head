
# PyTorch
import torch.nn as nn
from torchsummary import summary


class activityRecognition(nn.Module):
    def __init__(self, numActivityFeatures, numActivities):
        super(activityRecognition, self).__init__()
        # General parameters.
        self.numActivityFeatures = numActivityFeatures
        self.numActivities = numActivities

        self.predictActivity = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(numActivityFeatures, numActivityFeatures, bias=True),
            nn.SELU(),

            # Neural architecture: Layer 1.
            nn.Linear(numActivityFeatures, self.numActivities, bias=True),
        )

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numFeatures) """
        # Specify the current input shape of the data.
        batchSize, numFeatures = inputData.size()
        assert numFeatures == self.numActivityFeatures, f"{numFeatures}, {self.numActivityFeatures}"

        # Classify the probability of each activity.
        activityDistribution = self.predictActivity(inputData)
        # activityDistribution dimension: batchSize, self.numActivities

        return activityDistribution

    def printParams(self):
        # activityRecognition(numActivityFeatures = 16, numActivities = 20).to('cpu').printParams()
        summary(self, (self.numActivityFeatures,))
