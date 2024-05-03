# General
import torch
from torch import nn, optim

# Import files.
from .generalProtocol import generalProtocol
from .nnHelpers.heatTherapyModel import heatTherapyModel
from .nnHelpers.modelHelpers.lossCalculations import lossCalculations


class nnProtocol(generalProtocol):
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, modelName, onlineTraining=False):
        super().__init__(temperatureBounds, tempBinWidth, simulationParameters)
        # General model parameters.
        self.onlineTraining = onlineTraining  # Whether to train the model live.
        self.modelName = modelName  # The model's unique identifier.

        # Specify specific model parameters.
        self.numTemperatures = 1  # The number of temperatures to predict.
        self.numLosses = 3  # The number of losses to predict. (PA, NA, SA)

        # Model parameters.
        self.optimizer = None  # The optimizer for the model.

        # Model parameters.
        self.model = heatTherapyModel(numTemperatures=self.numTemperatures, numLosses=self.numLosses, numTempBins=self.numTempBins, numLossBins=self.numLossBins)  # The model for the therapy.
        self.setupModelHelpers()

        # Initialize helper classes.
        self.lossCalculations = lossCalculations(loss_bins=self.loss_bins, numTemperatures=self.numTemperatures, numLosses=self.numLosses)

    # ------------------------ Setup nnProtocol ------------------------ #

    def setupModelHelpers(self):
        # Define the optimizer.
        self.optimizer = optim.AdamW([
            # Specify the model parameters for the signal mapping.
            {'params': self.model.sharedModelWeights.parameters(), 'weight_decay': 1E-2, 'lr': 1E-5},

            # Specify the model parameters for the feature extraction.
            {'params': self.model.specificModelWeights.parameters(), 'weight_decay': 1E-2, 'lr': 1E-5},
        ])

    # ------------------------ nnProtocol ------------------------ #

    def updateTherapyState(self):
        # Unpack the current user state.
        currentUserState = torch.tensor(self.userFullStatePath[-1], dtype=torch.float32)
        # currentUserState dimensions: [numInputFeatures=4]. Values: [temperature, PA, NA, SA]

        # Prepare the model for training.
        self.optimizer.zero_grad()  # Zero the gradients.

        # Forward pass through the model.
        finalTemperaturePredictions, finalLossPredictions = self.model(currentUserState)
        # finalTemperaturePrediction dimensions: [numTemperatures=1, batchSize=1, numTempBins=11].
        # finalLossPrediction dimensions: [numLosses=3, batchSize=1, numLossBins=51].

        return (finalTemperaturePredictions, finalLossPredictions), None

    def getNextState(self, therapyState):
        """ Overwrite the general getNextState method to include the neural network. """
        # Unpack the final temperature predictions.
        finalTemperaturePredictions = therapyState[0]

        # Get the new temperature to be compatible with the general protocol method.
        assert finalTemperaturePredictions.size() == (self.numTemperatures, 1, self.numTempBins), f"Expected 1 temperature and batch for training, but got {finalTemperaturePredictions.size()}"
        newUserTemp_bin = finalTemperaturePredictions.argmax(dim=2)[0][0].item()  # Assumption on input dimension
        # newUserTemp dimensions: single value (probability)

        newUserTemp = self.temp_bins[newUserTemp_bin] + 0.5 # mid temp
        super().getNextState(newUserTemp)

    # ------------------------ Machine Learning ------------------------ #

    def updateWeights(self, lossPredictionLoss, minimizeLossBias):
        # Calculate the total error.
        total_error = lossPredictionLoss + 0.5*minimizeLossBias
        print('total_error: ', total_error)

        # Backpropagation.
        total_error.backward()  # Calculate the gradients.
        self.optimizer.step()   # Update the weights.
        self.optimizer.zero_grad()  # Zero the gradients.
