# General
import torch
from torch import nn


class specificModelWeights(nn.Module):
    def __init__(self, numSharedTempFeatures, numSharedLossFeatures, numTempBins=9, numLossBins=11, numLosses=3, numTemperatures=1):
        # General model parameters.
        super().__init__()
        # General model parameters.
        self.numSharedTempFeatures = numSharedTempFeatures  # The number of shared temperature features.
        self.numSharedLossFeatures = numSharedLossFeatures  # The number of shared loss features.
        self.numTemperatures = numTemperatures  # The number of predicted temperatures.
        self.numTempBins = numTempBins  # The number of temperature bins.
        self.numLossBins = numLossBins  # The number of loss bins.
        self.numLosses = numLosses  # The number of losses.

        # added parameters
        self.numInputTempFeatures = self.numTemperatures + self.numLosses  # The number of input temperature features.
        self.numInputLossFeatures = self.numTemperatures * (self.numTempBins + 1) + self.numLosses  # The number of input loss features.
        # Calculate the number of output shared features.
        # arbitrary
        self.numSharedTempFeatures = self.numTempBins * 2  # The number of shared temperature features. Output dimension of the shared model.
        self.numSharedLossFeatures = self.numLosses * 2  # The number of shared loss features. Output dimension of the shared model.

        self.numParameters = 4
        # ------------------------------

        # Initialize the module holders.
        self.lossModules = nn.ModuleList()  # Loss modules for each loss [PA, NA, SA].
        self.tempModules = nn.ModuleList()  # Temperature modules for each temperatures [T1, T2, T3].
        self.stateModules = nn.ModuleList()  # Coefficient modules for delta temperature [a, b, c, d].

        # For each loss module.
        for tempModuleInd in range(self.numTemperatures):
            self.tempModules.append(nn.Sequential(
                # Neural architecture
                nn.Linear(self.numSharedTempFeatures, self.numSharedTempFeatures, bias=True),
                nn.SELU(),

                nn.Linear(self.numSharedTempFeatures, self.numTempBins, bias=True),
                nn.Softmax(dim=-1),  # Softmax activation along the feature dimension

            ))

        # For each loss module.
        for lossModuleInd in range(self.numLosses):
            self.lossModules.append(nn.Sequential(
                nn.Linear(self.numSharedLossFeatures, 2 * self.numSharedLossFeatures, bias=True),
                nn.SELU(),
                nn.Linear(2 * self.numSharedLossFeatures, 2 * self.numSharedLossFeatures, bias=True),
                nn.SELU(),
                nn.Linear(2 * self.numSharedLossFeatures, self.numSharedLossFeatures, bias=True),
                nn.SELU(),
                nn.Linear(self.numSharedLossFeatures, self.numLossBins, bias=True),
            ))

        # TODO: double check
        # For each coefficient module.
        for stateModuleInd in range(self.numParameters):
            self.stateModules.append(nn.Sequential(
                nn.Linear(self.numSharedLossFeatures, 2 * self.numSharedLossFeatures, bias=True),  # 6 to 22
                nn.SELU(),

                nn.Linear(2 * self.numSharedLossFeatures, 2 * self.numSharedLossFeatures, bias=True),  # 22 to 22
                nn.SELU(),

                nn.Linear(2 * self.numSharedLossFeatures, 2*self.numSharedLossFeatures, bias=True),  # 22 to 11
                nn.SELU(),

                nn.Linear(2*self.numSharedLossFeatures, out_features=1, bias=True),  # 11 to 1
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

    def predictingStates(self, inputData):
        # Extract the input dimensions.
        batchSize, numInputFeatures = inputData.size()

        # Initialize a holder for the loss predictions.
        deltafinalStatePredictions = torch.zeros(self.numParameters, batchSize, 1)
        # For each loss module.
        for stateModuleInd in range(self.numParameters):
            deltafinalStatePredictions[stateModuleInd] = self.stateModules[stateModuleInd](inputData)

        return deltafinalStatePredictions
