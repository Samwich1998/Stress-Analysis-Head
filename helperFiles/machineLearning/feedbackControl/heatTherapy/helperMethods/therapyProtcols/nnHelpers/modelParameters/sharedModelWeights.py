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

        self.numParameters = 4
        # Shared model parameters.
        self.sharedTempFeatureExtraction = nn.Sequential(
            # Neural architecture
            nn.Linear(in_features=self.numInputTempFeatures, out_features=2*self.numInputTempFeatures, bias=True), # numInputTempFeatures = 4
            nn.SELU(),
            
            # Neural architecture
            nn.Linear(in_features=2*self.numInputTempFeatures, out_features=2*self.numInputTempFeatures, bias=True), # numInputTempFeatures = 4
            nn.SELU(),

            # Neural architecture
            nn.Linear(in_features=2*self.numInputTempFeatures, out_features=self.numSharedTempFeatures, bias=True), # numSharedTempFeatures = 22
            nn.SELU(),

            #Hyperparameter tunning
            nn.Linear(in_features=self.numSharedTempFeatures, out_features=self.numSharedTempFeatures, bias=True),  # numSharedTempFeatures = 22 -> 4
            nn.SELU(),

        )

        # Shared model parameters.
        self.sharedLossFeatureExtraction = nn.Sequential(
            # Neural architecture
            nn.Linear(in_features=self.numInputLossFeatures, out_features=2*self.numInputLossFeatures, bias=True), # numInputLossFeatures = 15
            nn.SELU(),
            
            # Neural architecture
            nn.Linear(in_features=2*self.numInputLossFeatures, out_features=2*self.numInputLossFeatures, bias=True), # numInputLossFeatures = 15
            nn.SELU(),

            # Neural architecture
            nn.Linear(in_features=2*self.numInputLossFeatures, out_features=self.numSharedLossFeatures, bias=True), # numsharedLossFeaturer = 6 (15->6)
            nn.SELU(),

            #Hyperparameter tunning
            nn.Linear(in_features=self.numSharedLossFeatures, out_features=self.numSharedLossFeatures, bias=True),  # numSharedTempFeatures = 6 -> 15
            nn.SELU(),
        )

        # Shared model parameters.
        self.sharedFeatureExtractionTotal = nn.Sequential(
            # Neural architecture
            nn.Linear(in_features=self.numParameters, out_features=2*self.numInputLossFeatures, bias=True), # 4 to 15; numInputLossFeatures = 15
            nn.SELU(),

            # Neural architecture
            nn.Linear(in_features=2*self.numInputLossFeatures, out_features=2*self.numInputLossFeatures, bias=True), # numInputLossFeatures = 15 (15 to 30)
            nn.SELU(),

            # Neural architecture
            nn.Linear(in_features=2*self.numInputLossFeatures, out_features=2*self.numInputLossFeatures, bias=True), # numsharedLossFeaturer = 6 (30 to 15)
            nn.SELU(),

            nn.Linear(in_features=2*self.numInputLossFeatures, out_features=self.numSharedLossFeatures, bias=True),  # 15 to 6
            nn.SELU(),
        )


    def calculateSharedTempFeatures(self, inputData):
        return self.sharedTempFeatureExtraction(inputData)

    def calculateSharedLossFeatures(self, inputData):
        return self.sharedLossFeatureExtraction(inputData)

