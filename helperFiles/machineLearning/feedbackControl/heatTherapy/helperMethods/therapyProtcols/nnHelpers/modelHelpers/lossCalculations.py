# General
import torch
from torch import nn
import torch.nn.functional as F
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.generalProtocol import generalProtocol


class lossCalculations:
    def __init__(self, loss_bins, numTemperatures, numLosses):
        # Specify specific model parameters.
        self.numTemperatures = numTemperatures  # The number of temperatures to predict.
        self.numLosses = numLosses  # The number of losses to predict. (PA, NA, SA)
        self.loss_bins = loss_bins  # The loss bins for the model.

        # Model parameters.
        self.classificationLoss = nn.CrossEntropyLoss(weight=None, reduction='none', label_smoothing=0.0)
        self.divergenceLoss = nn.KLDivLoss(reduction='batchmean', log_target=False)
        self.MSELoss = nn.MSELoss(reduction='none')
        self.predictionLossType = "MSE"  # The type of loss to use for the model.
        self.optimalLossType = "MSE"  # The type of loss to use for the model.

        # Initialize the optimal final loss.
        self.optimalLoss = [1, 0, 0]  # The optimal final loss bin index. [PA, NA, SA].
        self.optimalFinalLossBinIndex = list(map(lambda loss: generalProtocol.getBinIndex(self.loss_bins, loss), self.optimalLoss))
        self.optimalFinalLossBinIndex = torch.tensor(data=self.optimalFinalLossBinIndex, dtype=torch.long)
        # self.optimalFinalLoss_bin dimensions: [numLosses].

    def scoreModel(self, therapyState, trueLossValues):
        """ Score the model based on the final loss. """
        # Unpack the final loss predictions.
        finalTemperaturePredictions, finalLossPredictions = therapyState
        numLosses, batchSize, numLossBins = finalLossPredictions.size()
        # trueLossValues dimensions: [numLosses] if self.onlineTraining else [numLosses, numLossBins]
        # finalTemperaturePrediction dimensions: [numTemperatures, batchSize, numTempBins].
        # finalLossPrediction dimensions: [numLosses, batchSize, numLossBins].

        # Assert the validity of the parameters.
        assert numLosses == self.numLosses, "The number of losses must match the expected number of losses."
        assert len(trueLossValues) == self.numLosses, "The number of true loss values must match the expected number of losses."

        # Preset the loss values.
        lossPredictionLoss = 0
        minimizeLossBias = 0

        # For each mental state score.
        for lossInd in range(self.numLosses):
            # Bias the model to predict the next loss.
            self.lossCalculation(lossPredictionLoss, finalLossPredictions, trueLossValues, lossInd, lossType=self.predictionLossType)

            # Bias the model to minimize the loss.
            expectedLoss = self.optimalFinalLossBinIndex.expand(self.numLosses, batchSize)  # expectedLoss dimensions: [self.numLosses, batchSize].
            self.lossCalculation(minimizeLossBias, finalLossPredictions, expectedLoss, lossInd, lossType=self.optimalLossType)

        return lossPredictionLoss, minimizeLossBias

    def lossCalculation(self, lossPredictionLoss, finalLossPredictions, trueLossValues, lossInd, lossType):
        # KL divergence loss
        if lossType == "KL":
            # Add the loss value.
            lossPredictionLoss = lossPredictionLoss + self.divergenceLoss(F.log_softmax(finalLossPredictions[lossInd], dim=-1), trueLossValues[lossInd])

        # Cross-entropy loss
        elif lossType == "CE":
            # Prepare the final loss predictions for classification.
            trueLossValues = list(map(lambda loss: generalProtocol.getBinIndex(self.loss_bins, loss), trueLossValues))
            trueLossValues = torch.tensor(data=trueLossValues, dtype=torch.long)  # Convert the true loss values to a tensor.
            trueLossValues = trueLossValues.unsqueeze(-1).expand(-1, finalLossPredictions.size(1))  # trueLossValues dimensions: [numLosses, batchSize].

            # Add the loss value.
            lossPredictionLoss = lossPredictionLoss + self.classificationLoss(finalLossPredictions[lossInd], trueLossValues[lossInd].argmax(dim=-1)).mean()

        # MSE loss
        elif lossType == "MSE":
            # Prepare the final loss predictions for MSE.
            trueLossValues = trueLossValues.unsqueeze(-1).expand(-1, finalLossPredictions.size(1))  # trueLossValues dimensions: [numLosses, batchSize].

            # Add the loss value.
            lossPredictionLoss = lossPredictionLoss + self.MSELoss(finalLossPredictions[lossInd], trueLossValues[lossInd]).mean()
        else:
            raise ValueError("The loss type is not recognized.")

        return lossPredictionLoss