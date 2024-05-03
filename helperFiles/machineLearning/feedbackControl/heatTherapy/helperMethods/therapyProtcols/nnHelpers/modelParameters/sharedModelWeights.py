# General
from torch import nn


class sharedModelWeights(nn.Module):
    def __init__(self, numInputTempFeatures, numSharedTempFeatures, numInputLossFeatures, numSharedLossFeatures):
        # General model parameters.
        super().__init__()
        self.numSharedTempFeatures = numSharedTempFeatures  # The number of shared temperature features.
        self.numSharedLossFeatures = numSharedLossFeatures  # The number of shared loss features.
        self.numInputTempFeatures = numInputTempFeatures  # The number of initial temperature features.
        self.numInputLossFeatures = numInputLossFeatures  # The number of initial loss features.

        # Shared model parameters.
        self.sharedTempFeatureExtraction = nn.Sequential(
            # Neural architecture
            nn.Linear(in_features=self.numInputTempFeatures, out_features=2*self.numInputTempFeatures, bias=True),
            nn.SELU(),

            # Neural architecture
            nn.Linear(in_features=2*self.numInputTempFeatures, out_features=self.numSharedTempFeatures, bias=True),
            nn.SELU(),
        )

        # Shared model parameters.
        self.sharedLossFeatureExtraction = nn.Sequential(
            # Neural architecture
            nn.Linear(in_features=self.numInputLossFeatures, out_features=self.numInputLossFeatures, bias=True),
            nn.SELU(),

            # Neural architecture
            nn.Linear(in_features=self.numInputLossFeatures, out_features=self.numSharedLossFeatures, bias=True),
            nn.SELU(),
        )

    def calculateSharedTempFeatures(self, inputData):
        return self.sharedTempFeatureExtraction(inputData)

    def calculateSharedLossFeatures(self, inputData):
        return self.sharedLossFeatureExtraction(inputData)

