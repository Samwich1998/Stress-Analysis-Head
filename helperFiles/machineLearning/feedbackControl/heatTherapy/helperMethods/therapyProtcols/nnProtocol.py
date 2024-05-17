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
        # The scheduler for the optimizer.
        self.scheduler = None


        # Model parameters.
        #self.model = heatTherapyModel(numTemperatures=self.numTemperatures, numLosses=self.numLosses, numTempBins=self.numTempBins, numLossBins=self.numLossBins)  # The model for the therapy.
        self.model = heatTherapyModelUpdate(numTemperatures=self.numTemperatures, numLosses=self.numLosses, numTempBins=self.numTempBins, numLossBins=self.numLossBins)  # The model for the therapy.
        self.setupModelHelpers()
        self.setupModelScheduler()



        # Initialize helper classes.
        self.lossCalculations = lossCalculations(loss_bins=self.loss_bins, numTemperatures=self.numTemperatures, numLosses=self.numLosses)

        # keeping track of state alterations
        self.sampled_temperatures = set()  # The set of sampled temperatures.
    # ------------------------ Setup nnProtocol ------------------------ #

    def setupModelHelpers(self):
        # LR: [1E-2, 1E-6] -> 1E-3, 1E-4 is typical
        # LR: [1E-3, 1E-8]     
        
        # Define the optimizer.
        self.optimizer = optim.AdamW([
            # Specify the model parameters for the signal mapping.
            {'params': self.model.sharedModelWeights.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4},

            # Specify the model parameters for the feature extraction.
            {'params': self.model.specificModelWeights.parameters(), 'weight_decay': 1E-10, 'lr': 1E-4},
        ])

    def setupModelScheduler(self):
        # The scheduler for the optimizer.
        #self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
    # ------------------------ nnProtocol ------------------------ #

    def updateTherapyState(self):
        # currentUserState dimensions: [numInputFeatures=4]. Values: [temperature, PA, NA, SA]
        if not self.onlineTraining:
            # Unpack the current user state path distribution.
            # temperature, PA_distribution_array, NA_distribution_array, SA_distribution_array = self.userFullStatePathDistribution[-1]
            #
            # #TODO: torch.multinomial
            # import numpy as np
            # # Randomly select one element from each distribution based on its probability
            # PA_distribution = np.random.choice(PA_distribution_array)
            # NA_distribution = np.random.choice(NA_distribution_array)
            # SA_distribution = np.random.choice(SA_distribution_array)
            temperature, PA, NA, SA = self.userFullStatePath[-1]

            # Update currentUserState with randomly chosen values
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

    def explore_temperature(self, predicted_temp_prob, epsilon):
        if random.uniform(0,1) < epsilon:
            print('------- exploring -------')
            return random.choice(range(len(predicted_temp_prob[0][0])))
        else:
            return predicted_temp_prob.argmax(dim=2)[0][0].item()

    def large_temperature_exploration(self, predicted_delta_temp, threshold):
        if predicted_delta_temp > threshold:
            return threshold
        else:
            return predicted_delta_temp

    def getNextState(self, therapyState):
        """ Overwrite the general getNextState method to include the neural network. """
        # Unpack the final temperature predictions.
        finalTemperaturePredictions = therapyState[0] # dim: torch.size([1, 1, 11])
        finalTemperaturePredictions = finalTemperaturePredictions.unsqueeze(0).expand(self.numTemperatures, 1, self.numTempBins)
        # print("2",finalTemperaturePredictions.size())
        # print('3',finalTemperaturePredictions.argmax(dim=2))
        # print('4', finalTemperaturePredictions.argmax(dim=2)[0])
        # print('5', finalTemperaturePredictions.argmax(dim=2)[0][0])
        # print('6', finalTemperaturePredictions.argmax(dim=2)[0][0].item())
        # print('7', range(len(finalTemperaturePredictions[0][0])))
        # print('8', finalTemperaturePredictions[0][0][0].item())
        # exit()

        # Get the new temperature to be compatible with the general protocol method.
        assert finalTemperaturePredictions.size() == (self.numTemperatures, 1, self.numTempBins), f"Expected 1 temperature and batch for training, but got {finalTemperaturePredictions.size()}"

        # epsilon-greedy exploration
        #newUserTemp_bin = self.explore_temperature(finalTemperaturePredictions, 0.1)
        #newUserTemp_bin = finalTemperaturePredictions.argmax(dim=2)[0][0].item()  # Assumption on input dimension
        # newUserTemp dimensions: single value (probability)

        # For online training only (not much difference between bins, so take the middle point of each bin as the next temperature adjustments)
        if self.onlineTraining:
            newUserTemp = self.userStatePath[-1][0] + self.large_temperature_exploration(finalTemperaturePredictions[0][0][0].item(), 3)
        else:
            newUserTemp = self.userStatePath[-1][0] + self.large_temperature_exploration(finalTemperaturePredictions[0][0][0].item(), 3)
            print('finalTemperaturePredictions: ', finalTemperaturePredictions[0][0][0].item())
            print('newUserTemp: ', newUserTemp)
            print('#$%#$%#$%#')


        print('pass 3')
        newUserLoss_simulated, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated = super().getNextState(newUserTemp)
        print('pass 4')
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
        total_error = lossPredictionLoss #+ 0.5*minimizeLossBias
        print('total_error: ', total_error)

        # Backpropagation.
        print(f"total_error dtype: {total_error.dtype}")
        total_error.backward()  # Calculate the gradients.
        self.optimizer.step()   # Update the weights.
        self.optimizer.zero_grad()  # Zero the gradients.
        #self.scheduler.step()
        # print the learning rate changes after scheduler
        #print('learning rate: ', self.optimizer.param_groups[0]['lr'])
