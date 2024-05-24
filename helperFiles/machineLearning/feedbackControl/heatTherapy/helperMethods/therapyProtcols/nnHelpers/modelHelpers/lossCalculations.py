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
        self.MSELoss = nn.MSELoss(reduction='mean')

        # Initialize the optimal final loss.
        self.optimalLoss = [1, 0, 0]  # The optimal final loss bin index. [PA, NA, SA].
        self.optimalFinalLossBinIndex = list(map(lambda loss: generalProtocol.getBinIndex(self.loss_bins, loss), self.optimalLoss))
        self.optimalFinalLossBinIndex = torch.tensor(data=self.optimalFinalLossBinIndex, dtype=torch.long)
        # self.optimalFinalLoss_bin dimensions: [numLosses].

    def scoreModel(self, therapyState, trueLossValues):
        """ Score the model based on the final loss. """
        # Unpack the final loss predictions.
        finalTemperaturePredictions, finalLossPredictions = therapyState
        # trueLossValues dimensions: [numLosses] if self.onlineTraining else [numLosses, numLossBins]
        # finalTemperaturePrediction dimensions: [numTemperatures, batchSize, numTempBins].
        # finalLossPrediction dimensions: [numLosses, batchSize, numLossBins].

        # Prepare the final loss predictions.
        batchSize = finalTemperaturePredictions.size(1)  # Extract the batch size.
        trueLossValues = list(map(lambda loss: generalProtocol.getBinIndex(self.loss_bins, loss), trueLossValues))
        trueLossValues = torch.tensor(data=trueLossValues, dtype=torch.long)  # Convert the true loss values to a tensor.
        trueLossValues = trueLossValues.unsqueeze(-1).expand(-1, batchSize)  # trueLossValues dimensions: [numLosses, batchSize].

        lossPredictionLoss = 0
        # Bias the model to predict the next loss.
        for lossInd in range(self.numLosses):
            lossPredictionLoss = lossPredictionLoss + self.classificationLoss(finalLossPredictions[lossInd], trueLossValues).mean()

        minimizeLossBias = 0
        # Bias the model to minimize the loss.
        for lossInd in range(self.numLosses):
            expectedLoss = self.optimalFinalLossBinIndex[lossInd].expand(batchSize)  # expectedLoss dimensions: [batchSize].
            minimizeLossBias = minimizeLossBias + self.classificationLoss(finalLossPredictions[lossInd], expectedLoss).mean()

        return lossPredictionLoss, minimizeLossBias

    def scoreModel_offline(self, therapyState, deltaLossValues):
        """ Score the model based on the final loss. """
        # Unpack the final loss predictions.
        finalTemperaturePredictions = therapyState[0]
        finalLossPredictions = therapyState[1:]
        # trueLossValues dimensions: [numLosses] if self.onlineTraining else [numLosses, numLossBins]
        # finalTemperaturePrediction dimensions: [numTemperatures, batchSize, numTempBins].
        # finalLossPrediction dimensions: [numLosses, batchSize, numLossBins].

        # Unpack the loss predictions.
        numLosses, batchSize, numLossBins = finalLossPredictions.size()
        finalLossPredictions = finalLossPredictions.squeeze(dim=2)
        assert numLosses == self.numLosses, "The number of losses must match the expected number of losses."
        assert len(deltaLossValues) == self.numLosses, "The number of true loss values must match the expected number of losses."
        trueLossValues = deltaLossValues.unsqueeze(1)

        # Prepare the final loss predictions.
        lossPredictionLoss = 0

        # Bias the model to predict the next loss.
        for lossInd in range(self.numLosses):
            # KL divergence loss
            #lossPredictionLoss = lossPredictionLoss + self.divergenceLoss(F.log_softmax(finalLossPredictions[lossInd], dim=-1), trueLossValues[lossInd])
            # cross entropy loss
            #lossPredictionLoss = lossPredictionLoss + self.classificationLoss(finalLossPredictions[lossInd], trueLossValues[lossInd].argmax(dim=-1)).mean()
            # MSE loss
            lossPredictionLoss = lossPredictionLoss + self.MSELoss(finalLossPredictions[lossInd], trueLossValues[lossInd])
        minimizeLossBias = 0
        # Bias the model to minimize the loss.
        for lossInd in range(self.numLosses):
            expectedLoss = self.optimalFinalLossBinIndex[lossInd].expand(batchSize)  # expectedLoss dimensions: [batchSize].
            #minimizeLossBias = minimizeLossBias + self.classificationLoss(finalLossPredictions[lossInd], expectedLoss).mean()
            minimizeLossBias = minimizeLossBias + self.MSELoss(finalLossPredictions[lossInd], expectedLoss)
        return lossPredictionLoss, minimizeLossBias
