# General
import torch
from torch import nn, optim

# Import files.
from .generalProtocol import generalProtocol
from .nnHelpers.heatTherapyModel import heatTherapyModel


class nnProtocol(generalProtocol):
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, modelName, onlineTraining=False):
        super().__init__(temperatureBounds, tempBinWidth, simulationParameters)
        # General model parameters.
        self.onlineTraining = onlineTraining  # Whether to train the model live.
        self.modelName = modelName  # The model's unique identifier.
        self.optimizer = None       # The optimizer for the model.

        # self.lossFunction = nn.MSELoss()
        # self.lossFunction = nn.CrossEntropyLoss(weight=None, reduction='none', label_smoothing=0.0)
        self.optimalLoss = [1,0,0] # The optimal final loss bin index. [PA, NA, SA].
        self.optimalFinalLoss_bin = [self.getBinIndex(self.loss_bins, self.optimalLoss[i]) for i in range(len(self.optimalLoss))]
        self.optimalFinalLoss_bin = torch.tensor(self.optimalFinalLoss_bin, dtype=torch.long) # target: dtype = long
        self.predictedLosses = []
        # Model parameters.
        #TODO not hard coding
        self.model = heatTherapyModel(numTemperatures=1, numLosses=3, numTempBins=11, numLossBins=51)  # The model for the therapy.
        self.setupModelHelpers()

    def updateTherapyState(self):
        # Unpack the current user state.
        currentUserState = self.userFullStatePath[-1]
        currentUserTemp, PA, NA, SA = currentUserState # loss is 1 single value calculated from the 3 losses

        # Update the temperatures visited.
        tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        currentUserState = torch.tensor([currentUserTemp, PA, NA, SA], dtype=torch.float32)

        finalTemperaturePredictions, finalLossPredictions = self.model(currentUserState)
        # finalTemperaturePrediction dimensions: [numTemperatures, batchSize, numTempBins]. # numTemperatures = 1, batchSize = 1, numTempBins = 11
        # finalLossPrediction dimensions: [numLosses, batchSize, numLossBins]. # numLosses = 3, batchSize = 1, numLossBins = 51
        #print('finalTemperaturePredictions: ', finalTemperaturePredictions)
        print('finalLossPredictions: ', finalLossPredictions)

        
        newUserTemp = finalTemperaturePredictions.argmax(dim=2)  #TODO: Not differentiable
        print('newUserTemp: ', newUserTemp)
        # newUserTemp dimensions: [numTemperatures, batchSize].
        updated_Temp_ind = newUserTemp.squeeze(0).item()
        updated_Temp = self.temp_bins[updated_Temp_ind] + 0.5
        self.predictedLosses.append(finalLossPredictions)
        print('updated_Temp: ', updated_Temp)
        return updated_Temp

    def updateWeights(self, actualLoss): # interface with users
        """
        :param actualLoss: [numLosses, batchSize, numLossBins]
        self.predictedLosses: [numEpochs, numLosses, batchSize, numLossBins]
        """
        # Calculate the loss.
        error1, error2 = self.compileLosses(finalLossPredictions=self.predictedLosses[-1], targetLossInds=actualLoss)
        total_error = error1 + error2
        print('total_error: ', total_error)
        # Backpropagation.
        total_error.backward()      # Calculate the gradients.
        self.optimizer.step()       # Update the weights.
        self.optimizer.zero_grad()  # Zero the gradients.
        print("Loss: ", total_error.item())

    # ------------------------ Machine Learning ------------------------ #

    # implement loss loss (with calcualted loss from PA, NA, SA)
    def compileLosses(self, finalLossPredictions, targetLossInds):
        """
        :param finalLossPredictions: [numLosses, batchSize, numLossBins] probabilities instead of actual loss
        :param targetLossInds: [numLosses, batchSize] probabilities instead of actual loss
        """
        # Assert the dimensions are correct.
        assert finalLossPredictions.size(2) == self.model.numLossBins, f"The number of loss bins have to match: {finalLossPredictions.size(0)} != {self.model.numLossBins}"
        assert finalLossPredictions.size(0) == targetLossInds.size(0), f"The number of losses have to match: {finalLossPredictions.size(0)} != {targetLossInds.size(0)}"
        assert finalLossPredictions.size(0) == self.model.numLosses, f"The number of losses have to match: {finalLossPredictions.size(0)} != {self.model.numLosses}"
        assert finalLossPredictions.size(1) == targetLossInds.size(1), f"Batch sizes have to match: {finalLossPredictions.size(1)} != {targetLossInds.size(1)}"

        lossPredictionLoss = 0
        # Bias the model to predict the next loss.
        for lossInd in range(self.model.numLosses):
            print(f"targetLossInds ({lossInd}): ", targetLossInds[lossInd])
            lossPredictionLoss = lossPredictionLoss + self.lossFunction(finalLossPredictions[lossInd], targetLossInds[lossInd]).mean()

        print('self.optimalFinalLoss_bin: ', self.optimalFinalLoss_bin)
        minimizeLossBias = 0
        # Bias the model to minimize the loss.
        for lossInd in range(self.model.numLosses):
            expectedLoss = self.optimalFinalLoss_bin[lossInd].expand(targetLossInds.size(1))
            minimizeLossBias = minimizeLossBias + self.lossFunction(finalLossPredictions[lossInd], expectedLoss).mean()

        return lossPredictionLoss, minimizeLossBias


    # def setupModelHelpers(self):
    #     # Define the optimizer.
    #     self.optimizer = optim.AdamW(params=self.model.parameters(), lr=1e-3)

    def setupModelHelpers(self):
        # Define the optimizer.
        self.optimizer = optim.AdamW([
            # Specify the model parameters for the signal mapping.
            {'params': self.model.sharedModelWeights.parameters(), 'weight_decay': 1E-2, 'lr': 1E-1},

            # Specify the model parameters for the feature extraction.
            {'params': self.model.specificModelWeights.parameters(), 'weight_decay': 1E-2, 'lr': 1E-1},
        ])

        # TODO: this will porobably be MSE
        self.lossFunction = nn.CrossEntropyLoss(weight=None, reduction='none', label_smoothing=0.0)
        #self.lossFunction = nn.MSELoss()

