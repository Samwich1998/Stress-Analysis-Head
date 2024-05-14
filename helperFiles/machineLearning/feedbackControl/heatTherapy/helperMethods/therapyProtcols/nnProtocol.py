# General
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR


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
        # The scheduler for the optimizer.
        self.scheduler = None


        # Model parameters.
        self.model = heatTherapyModel(numTemperatures=self.numTemperatures, numLosses=self.numLosses, numTempBins=self.numTempBins, numLossBins=self.numLossBins)  # The model for the therapy.
        self.setupModelHelpers()
        self.setupModelScheduler()



        # Initialize helper classes.
        self.lossCalculations = lossCalculations(loss_bins=self.loss_bins, numTemperatures=self.numTemperatures, numLosses=self.numLosses)

    # ------------------------ Setup nnProtocol ------------------------ #

    def setupModelHelpers(self):
        # LR: [1E-2, 1E-6] -> 1E-3, 1E-4 is typical
        # LR: [1E-3, 1E-8]     
        
        # Define the optimizer.
        self.optimizer = optim.AdamW([
            # Specify the model parameters for the signal mapping.
            {'params': self.model.sharedModelWeights.parameters(), 'weight_decay': 1E-10, 'lr': 5E-4},

            # Specify the model parameters for the feature extraction.
            {'params': self.model.specificModelWeights.parameters(), 'weight_decay': 1E-10, 'lr': 5E-4},
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
            temperature, PA_distribution_array, NA_distribution_array, SA_distribution_array = self.userFullStatePathDistribution[-1]
            import numpy as np
            # Randomly select one element from each distribution based on its probability
            PA_distribution = np.random.choice(PA_distribution_array)
            NA_distribution = np.random.choice(NA_distribution_array)
            SA_distribution = np.random.choice(SA_distribution_array)

            # Update currentUserState with randomly chosen values
            currentUserState = torch.tensor([temperature, PA_distribution, NA_distribution, SA_distribution], dtype=torch.float32)  # dim: tensor[T, PA, NA, SA]
        else:
            currentUserState = torch.tensor(self.userFullStatePath[-1], dtype=torch.float32)

        # Restart gradient tracking.
        self.optimizer.zero_grad()  # Zero the gradients.

        # Forward pass through the model.
        finalTemperaturePredictions, finalLossPredictions = self.model(currentUserState)

        # finalTemperaturePrediction dimensions: [numTemperatures=1, batchSize=1, numTempBins=11].
        # finalLossPrediction dimensions: [numLosses=3, batchSize=1, numLossBins=11].

        # print('final Temperature prediction (value should be changing for every epoch)', finalTemperaturePredictions[0][0])
        # print('final Loss prediction PA', finalLossPredictions[0][0])
        # print('final Loss prediction NA', finalLossPredictions[1][0])
        # print('final Loss prediction SA', finalLossPredictions[2][0])

        # ------------------------ debug plotting ------------------------ #
        import matplotlib.pyplot as plt
        #plt.plot(self.temp_bins, finalTemperaturePredictions[0][0].detach().cpu())



        # plt.ylabel(f'Temp prob {finalTemperaturePredictions.size()}')
        # plt.xlabel('temp bins')
        # plt.plot(self.loss_bins, finalLossPredictions[0][0].detach().cpu())
        # plt.plot(self.loss_bins, finalLossPredictions[1][0].detach().cpu())
        # plt.plot(self.loss_bins, finalLossPredictions[2][0].detach().cpu())
        # plt.show()

        # ---------------------------------------------------------------- #
        return (finalTemperaturePredictions, finalLossPredictions), None

    def getNextState(self, therapyState):
        """ Overwrite the general getNextState method to include the neural network. """
        # Unpack the final temperature predictions.
        finalTemperaturePredictions = therapyState[0] # dim: torch.size([1, 1, 11])

        # Get the new temperature to be compatible with the general protocol method.
        assert finalTemperaturePredictions.size() == (self.numTemperatures, 1, self.numTempBins), f"Expected 1 temperature and batch for training, but got {finalTemperaturePredictions.size()}"
        newUserTemp_bin = finalTemperaturePredictions.argmax(dim=2)[0][0].item()  # Assumption on input dimension
        # newUserTemp dimensions: single value (probability)

        # For online training only (not much difference between bins, so take the middle point of each bin as the next temperature adjustments)
        if self.onlineTraining:
            newUserTemp = self.temp_bins[newUserTemp_bin] + 0.5  # mid temp
        else:
            newUserTemp = torch.FloatTensor(1,).uniform_(self.temperatureBounds[0], self.temperatureBounds[1])[0]
            #newUserTemp = self.temp_bins[newUserTemp_bin] + 0.5  # mid temp
            newUserTemp = 30
        newUserLoss_simulated, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated = super().getNextState_offline(30)

        print('*****PA_dist_simulated: ', PA_dist_simulated)
        print('*****NA_dist_simulated: ', NA_dist_simulated)
        print('*****SA_dist_simulated: ', SA_dist_simulated)

        self.userFullStatePathDistribution.append([newUserTemp, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated])
        self.userStatePath_simulated.append([newUserTemp, newUserLoss_simulated])

    # def getNextState_offline(self, therapyState):
    #     """ Overwrite the general getNextState method to include the neural network. """
    #     # Unpack the final temperature predictions.
    #     finalTemperaturePredictions = therapyState[0]  # dim: torch.size([1, 1, 11])
    #
    #     # Get the new temperature to be compatible with the general protocol method.
    #     assert finalTemperaturePredictions.size() == (self.numTemperatures, 1, self.numTempBins), f"Expected 1 temperature and batch for training, but got {finalTemperaturePredictions.size()}"
    #     newUserTemp_bin = finalTemperaturePredictions.argmax(dim=2)[0][0].item()  # Assumption on input dimension
    #     # newUserTemp dimensions: single value (probability)
    #
    #     # For online training only (not much difference between bins, so take the middle point of each bin as the next temperature adjustments)
    #     if self.onlineTraining:
    #         newUserTemp = self.temp_bins[newUserTemp_bin] + 0.5  # mid temp
    #     else:
    #         newUserTemp = torch.FloatTensor(1, ).uniform_(self.temperatureBounds[0], self.temperatureBounds[1])[0]
    #
    #     newUserLoss_simulated, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated = super().getNextState_offline(newUserTemp)
    #
    #     self.userFullStatePathDistribution.append([newUserTemp, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated])
    #     self.userStatePath_simulated.append([newUserTemp, newUserLoss_simulated])

    # ------------------------ Machine Learning ------------------------ #

    def updateWeights(self, lossPredictionLoss, minimizeLossBias):
        # Calculate the total error.

        #TODO: check
        total_error = lossPredictionLoss # + 0.5*minimizeLossBias
        #total_error = lossPredictionLoss
        print('total_error: ', total_error)

        # Backpropagation.
        total_error.backward()  # Calculate the gradients.
        self.optimizer.step()   # Update the weights.
        self.optimizer.zero_grad()  # Zero the gradients.

        #self.scheduler.step()
        # print the learning rate changes after scheduler
        #print('learning rate: ', self.optimizer.param_groups[0]['lr'])



