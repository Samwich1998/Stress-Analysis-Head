# General
import torch
from torch import nn


class specificModelWeights(nn.Module):
    def __init__(self, numSharedTempFeatures, numSharedLossFeatures, numTempBins=9, numLossBins=11, numLosses=3, numTemperatures=1):
        # General model parameters.
        super().__init__()
        # General model parameters.
        self.numSharedTempFeatures = numSharedTempFeatures    # The number of shared temperature features.
        self.numSharedLossFeatures = numSharedLossFeatures    # The number of shared loss features.
        self.numTemperatures = numTemperatures     # The number of predicted temperatures.
        self.numTempBins = numTempBins      # The number of temperature bins.
        self.numLossBins = numLossBins      # The number of loss bins.
        self.numLosses = numLosses          # The number of losses.

        # Initialize the module holders.
        self.lossModules = nn.ModuleList()  # Loss modules for each loss [PA, NA, SA].
        self.tempModules = nn.ModuleList()  # Temperature modules for each temperatures [T1, T2, T3].

        # For each loss module.
        for tempModuleInd in range(self.numTemperatures):
            self.tempModules.append(nn.Sequential(
                # Neural architecture
                nn.Linear(self.numSharedTempFeatures, self.numTempBins, bias=True),
                nn.Softmax(dim=-1),  # Softmax activation along the feature dimension
            ))

        # For each loss module.
        for lossModuleInd in range(self.numLosses):
            self.lossModules.append(nn.Sequential(
                # Neural architecture
                nn.Linear(self.numSharedLossFeatures, self.numLossBins, bias=True),
                nn.Softmax(dim=-1),  # Softmax activation along the feature dimension
            ))

    def predictNextTemperature(self, inputData):
        # Extract the input dimensions.
        batchSize, numInputFeatures = inputData.size()

        # Initialize a holder for the loss predictions.
        finalTempPredictions = torch.zeros(self.numTemperatures, batchSize, self.numTempBins)

        # For each loss module.
        for tempModuleInd in range(self.numTemperatures):
            finalTempPredictions[tempModuleInd] = self.tempModules[tempModuleInd](inputData)

        return finalTempPredictions

    def predictNextLoss(self, inputData):
        # Extract the input dimensions.
        batchSize, numInputFeatures = inputData.size()

        # Initialize a holder for the loss predictions.
        finalLossPredictions = torch.zeros(self.numLosses, batchSize, self.numLossBins)
        # For each loss module.
        for lossModuleInd in range(self.numLosses):
            finalLossPredictions[lossModuleInd] = self.lossModules[lossModuleInd](inputData)

        return finalLossPredictions
