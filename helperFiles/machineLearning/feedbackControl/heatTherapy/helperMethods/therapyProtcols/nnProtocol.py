# General
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR
import random

# Import files.
from .generalProtocol import generalProtocol
from .nnHelpers.heatTherapyModel import heatTherapyModel
from .nnHelpers.modelHelpers.lossCalculations import lossCalculations
from .nnHelpers.heatTherapyModelUpdate import heatTherapyModelUpdate


class nnProtocol(generalProtocol):
    def __init__(self, temperatureBounds, simulationParameters, modelName, onlineTraining=False):
        super().__init__(temperatureBounds, simulationParameters)
        # General model parameters.
        self.onlineTraining = onlineTraining  # Whether to train the model live.
        self.modelName = modelName  # The model's unique identifier.

        # Model parameters.
        self.optimizer = None  # The optimizer for the model.
        # The scheduler for the optimizer.
        self.scheduler = None

        # Model parameters.
        self.model = heatTherapyModelUpdate(numTemperatures=self.numTemperatures, numLosses=self.numLosses, numTempBins=self.numTempBins, numLossBins=self.numLossBins)  # The model for the therapy.
        self.setupModelHelpers()
        self.setupModelScheduler()

        # Initialize helper classes.
        self.lossCalculations = lossCalculations(loss_bins=self.loss_bins, numTemperatures=self.numTemperatures, numLosses=self.numLosses)

        # keeping track of state alterations
        self.sampled_temperatures = set()  # The set of sampled temperatures.

        # epsilon based exploration
        self.epsilon = 0.1

    # ------------------------ Setup nnProtocol ------------------------ #

    def setupModelHelpers(self):
        # LR: [1E-2, 1E-6] -> 1E-3, 1E-4 is typical
        # LR: [1E-3, 1E-8]     

        # Define the optimizer.
        self.optimizer = optim.AdamW([
            # Specify the model parameters for the signal mapping.
            {'params': self.model.sharedModelWeights.parameters(), 'weight_decay': 1E-10, 'lr': 1E-3},

            # Specify the model parameters for the feature extraction.
            {'params': self.model.specificModelWeights.parameters(), 'weight_decay': 1E-10, 'lr': 1E-3},
        ])

    def setupModelScheduler(self):
        # The scheduler for the optimizer.
        #self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)

    # ------------------------ nnProtocol ------------------------ #

    def updateTherapyState(self):
        """ currentUserState dimensions: [temperature, PA, NA, SA] """
        if not self.onlineTraining:
            # Extract the most recent state and standardize the temperature
            temperature, PA, NA, SA = self.userFullStatePath[-1]
            temperature = self.standardizeTemperature(temperature)
            # Update currentUserState and prepare for input to the model
            currentUserState = torch.tensor([temperature, PA, NA, SA], dtype=torch.float32)  # dim: tensor[T, PA, NA, SA]
        else:
            currentUserState = torch.tensor(self.userFullStatePath[-1], dtype=torch.float32)
        # Restart gradient tracking.
        self.optimizer.zero_grad()  # Zero the gradients.
        # Forward pass through the model.
        finalStatePredictions = self.model(currentUserState)
        # finalTemperaturePrediction dimensions: [numTemperatures=1, batchSize=1, numTempBins=11].
        # finalLossPrediction dimensions: [numLosses=3, batchSize=1, numLossBins=11].
        return finalStatePredictions, None

    # ------------------------ exploration for nnProtocol simulation ------------------------ #

    def explore_temperature(self, predicted_temp_change, epsilon):
        if random.uniform(0, 1) < epsilon:
            print('------- exploring -------')
            predicted_temp_change = random.uniform(0, 5)
            return predicted_temp_change
        else:
            return predicted_temp_change

    def large_temperature_exploration(self, predicted_delta_temp, threshold):
        if predicted_delta_temp > threshold:
            return threshold
        else:
            return predicted_delta_temp

    def getNextState(self, therapyState):
        """ Overwrite the general getNextState method to include the neural network. """
        # Unpack the final temperature predictions.
        finalTemperaturePredictions = therapyState[0]  # dim: torch.size([1, 1, 11])
        finalTemperaturePredictions = finalTemperaturePredictions.unsqueeze(0).expand(self.numTemperatures, 1, self.numTempBins)
        # Get the new temperature to be compatible with the general protocol method.
        assert finalTemperaturePredictions.size() == (self.numTemperatures, 1, self.numTempBins), f"Expected 1 temperature and batch for training, but got {finalTemperaturePredictions.size()}"

        # For online training only (not much difference between bins, so take the middle point of each bin as the next temperature adjustments)
        if self.onlineTraining:
            newUserTemp = self.unstandardizeTemperature(finalTemperaturePredictions[0][0][0].item())
        else:
            newUserTemp = self.unstandardizeTemperature(finalTemperaturePredictions[0][0][0].item())
            # if random.uniform(0,1) < self.epsilon:
            #     newUserTemp = self.sample_temperature(newUserTemp)
        print('newUserTemp: ', newUserTemp)
        newUserLoss_simulated, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated = super().getNextState(newUserTemp)
        self.userFullStatePathDistribution.append([newUserTemp, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated])
        self.userStatePath_simulated.append([newUserTemp, newUserLoss_simulated])

    def getNextState_explore(self, therapyState):
        finalTemperaturePredictions = therapyState[0]  # dim: torch.size([1, 1, 11])

        # Get the new temperature to be compatible with the general protocol method.
        assert finalTemperaturePredictions.size() == (self.numTemperatures, 1, self.numTempBins), f"Expected 1 temperature and batch for training, but got {finalTemperaturePredictions.size()}"
        newUserTemp_bin = finalTemperaturePredictions.argmax(dim=2)[0][0].item()  # Assumption on input dimension

        newUserTemp = self.sample_temperature(self.temp_bins[newUserTemp_bin])
        newUserLoss_simulated, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated = super().getNextState(newUserTemp)

        self.userFullStatePathDistribution.append([newUserTemp, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated])
        self.userStatePath_simulated.append([newUserTemp, newUserLoss_simulated])

    def sample_temperature(self, previous_temperature):
        while True:
            # Sample a new temperature uniformly between 0 and (upper_bound - lower_bound)
            random_temp = random.uniform(0, self.temperatureBounds[1] - self.temperatureBounds[0])
            newUserTemp = previous_temperature + random_temp

            # Ensure the temperature is within the bounds 30 to 50
            if self.temperatureBounds[0] <= newUserTemp <= self.temperatureBounds[1] and abs(random_temp) >= 2:
                # Check if the temperature has been sampled before
                if newUserTemp not in self.sampled_temperatures:
                    # Add the new temperature to the set of sampled temperatures
                    self.sampled_temperatures.add(newUserTemp)
                    return newUserTemp  # Return the valid temperature

    # ------------------------ Machine Learning ------------------------ #

    def updateWeights(self, lossPredictionLoss, minimizeLossBias):
        # Calculate the total error.

        #TODO: check
        total_error = lossPredictionLoss
        print('total_error: ', total_error)
        # Backpropagation.
        total_error.backward()  # Calculate the gradients.
        self.optimizer.step()  # Update the weights.
        self.optimizer.zero_grad()  # Zero the gradients.
