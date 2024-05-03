# General
import torch
from torch import nn

from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.generalProtocol import generalProtocol


class lossCalculations:
    def __init__(self, loss_bins, numTemperatures, numLosses):
        # Specify specific model parameters.
        self.numTemperatures = numTemperatures  # The number of temperatures to predict.
        self.numLosses = numLosses  # The number of losses to predict. (PA, NA, SA)
        self.loss_bins = loss_bins  # The loss bins for the model.

        # Model parameters.
        self.classificationLoss = nn.CrossEntropyLoss(weight=None, reduction='none', label_smoothing=0.0)

        # Initialize the optimal final loss.
        self.optimalLoss = [1, 0, 0]  # The optimal final loss bin index. [PA, NA, SA].
        self.optimalFinalLossBinIndex = list(map(lambda loss: generalProtocol.getBinIndex(self.loss_bins, loss), self.optimalLoss))
        self.optimalFinalLossBinIndex = torch.tensor(data=self.optimalFinalLossBinIndex, dtype=torch.long)
        # self.optimalFinalLoss_bin dimensions: [numLosses].

    def scoreModel(self, therapyState, trueLossValues):
        """ Score the model based on the final loss. """
        # Unpack the final loss predictions.
        finalTemperaturePredictions, finalLossPredictions = therapyState
        # finalTemperaturePrediction dimensions: [numTemperatures, batchSize, numTempBins].
        # finalLossPrediction dimensions: [numLosses, batchSize, numLossBins].
        # trueLossValues dimensions: [numLosses].

        # Prepare the final loss predictions.
        batchSize = finalTemperaturePredictions.size(1)  # Extract the batch size.
        trueLossValues = list(map(lambda loss: generalProtocol.getBinIndex(self.loss_bins, loss), trueLossValues))
        trueLossValues = torch.tensor(data=trueLossValues, dtype=torch.long)  # Convert the true loss values to a tensor.
        trueLossValues = trueLossValues.unsqueeze(-1).expand(-1, batchSize)  # trueLossValues dimensions: [numLosses, batchSize].

        lossPredictionLoss = 0
        # Bias the model to predict the next loss.
        for lossInd in range(self.numLosses):
            lossPredictionLoss = lossPredictionLoss + self.classificationLoss(finalLossPredictions[lossInd], trueLossValues[lossInd]).mean()

        minimizeLossBias = 0
        # Bias the model to minimize the loss.
        for lossInd in range(self.numLosses):
            expectedLoss = self.optimalFinalLossBinIndex[lossInd].expand(batchSize)  # expectedLoss dimensions: [batchSize].
            minimizeLossBias = minimizeLossBias + self.classificationLoss(finalLossPredictions[lossInd], expectedLoss).mean()

        return lossPredictionLoss, minimizeLossBias
