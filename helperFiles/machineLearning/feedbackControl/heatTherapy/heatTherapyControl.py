
# General.
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import sys
sys.path.append("./../../../../")
import helperFiles

# Import the necessary libraries.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.aStarProtocol import aStarProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.basicProtocol import basicProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.nnProtocol import nnProtocol


class heatTherapyControl:
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, therapyMethod="aStarProtocol", plotResults=False):
        # General parameters.
        self.simulationParameters = simulationParameters
        self.temperatureBounds = temperatureBounds
        self.tempBinWidth = tempBinWidth
        self.plotResults = plotResults

        # Therapy parameters.
        self.therapyProtocol = None
        self.therapyMethod = None

        # Set up the therapy protocols.
        self.setupTherapyProtocols(therapyMethod)

    def setupTherapyProtocols(self, therapyMethod):
        # Change the therapy method.
        self.therapyMethod = therapyMethod

        if self.therapyMethod == "aStarProtocol":
            self.therapyProtocol = aStarProtocol(self.temperatureBounds, self.tempBinWidth, self.simulationParameters, learningRate=2)
        elif self.therapyMethod == "basicProtocol":
            self.therapyProtocol = basicProtocol(self.temperatureBounds, self.tempBinWidth, self.simulationParameters)
        elif self.therapyMethod == "nnProtocol":
            self.therapyProtocol = nnProtocol(self.temperatureBounds, self.tempBinWidth, self.simulationParameters, modelName="2024-04-12 heatTherapyModel", onlineTraining=False)
        else:
            raise ValueError("Invalid therapy method provided.")

    def runTherapyProtocol(self, maxIterations=None):
        # Initialize holder parameters.
        self.therapyProtocol.initializeUserState()

        # --------------------------------for plotting purposes-----------------------------------
        iteration = 0
        loss_prediction_loss = []
        loss_bias = []
        current_user_loss = []
        epoch_list = []

        pa_heatmap = np.zeros((11,11))
        na_heatmap = np.zeros((11,11))
        sa_heatmap = np.zeros((11,11))
        pa_heatmap_predicted = np.zeros((11,11))
        na_heatmap_predicted = np.zeros((11,11))
        sa_heatmap_predicted = np.zeros((11,11))
        temperature_bins = np.arange(30, 52, 2)
        loss_bins = np.arange(0, 1.1, 0.1)
        # ----------------------------------------------------------------------------------------

        # Until the therapy converges.
        while not self.therapyProtocol.finishedTherapy:
            # Get the next states for the therapy.
            therapyState, allMaps = self.therapyProtocol.updateTherapyState()
            if self.therapyMethod == "nnProtocol":
                if self.therapyProtocol.onlineTraining:
                    print('------Online training started-------')
                    # Calculate the final loss.
                    trueLossValues = self.therapyProtocol.userFullStatePath[-1][1:] # from simulation data
                    lossPredictionLoss, minimizeLossBias = self.therapyProtocol.lossCalculations.scoreModel(therapyState, trueLossValues) # losspredictionloss is from the model
                    self.therapyProtocol.getNextState(therapyState)
                    self.therapyProtocol.updateWeights(lossPredictionLoss, minimizeLossBias)
                    currentUserLoss = self.therapyProtocol.userStatePath[-1][1]
                    loss_prediction_loss.append(lossPredictionLoss.item())
                    loss_bias.append(minimizeLossBias.item())
                    current_user_loss.append(currentUserLoss)
                    iteration += 1
                    epoch_list.append(iteration)
                    if self.plotResults:
                        self.therapyProtocol.plotTherapyResults_nn(epoch_list, loss_prediction_loss, loss_bias, current_user_loss)
                elif not self.therapyProtocol.onlineTraining:
                    print('------simulation (offline) training started-------')
                    trueLossValues = self.therapyProtocol.userFullStatePathDistribution[-1][1:]  # from simulation data
                    lossPredictionLoss, minimizeLossBias = self.therapyProtocol.lossCalculations.scoreModel_offline(therapyState, trueLossValues) #losspredictionloss is from the model
                    self.therapyProtocol.getNextState(therapyState)
                    self.therapyProtocol.updateWeights(lossPredictionLoss, minimizeLossBias)
                    # self.therapyProtocol.getNextState(therapyState)
                    currentUserLoss = self.therapyProtocol.userStatePath_simulated[-1][1]


                    # predicted map
                    # predicted loss distributions
                    currentPA_pred = torch.softmax(therapyState[1][0].squeeze(), dim=-1).detach().numpy()
                    currentNA_pred = torch.softmax(therapyState[1][1].squeeze(), dim=-1).detach().numpy()
                    currentSA_pred = torch.softmax(therapyState[1][2].squeeze(), dim=-1).detach().numpy()


                    # Extract the temperature and three loss values from the userFullStatePathDistribution
                    currentTemp = self.therapyProtocol.userFullStatePathDistribution[-1][0]
                    currentPA = self.therapyProtocol.userFullStatePathDistribution[-1][1]
                    currentNA = self.therapyProtocol.userFullStatePathDistribution[-1][2]
                    currentSA = self.therapyProtocol.userFullStatePathDistribution[-1][3]

                    print('#%#$%#currentPA: ', currentPA)
                    print('#$%#$%#$5currentNA: ', currentNA)
                    print('#$%#$%#currentSA: ', currentSA)

                    # Determine the index
                    temperature_index = np.digitize(currentTemp, temperature_bins) - 1

                    # Fill the heatmaps
                    for i in range(11):
                        pa_heatmap[i, temperature_index] = currentPA[i]
                        na_heatmap[i, temperature_index] = currentNA[i]
                        sa_heatmap[i, temperature_index] = currentSA[i]
                        pa_heatmap_predicted[i, temperature_index] = currentPA_pred[i]
                        na_heatmap_predicted[i, temperature_index] = currentNA_pred[i]
                        sa_heatmap_predicted[i, temperature_index] = currentSA_pred[i]

    
                    # Plotting the heat maps after updating them in each iteration
                    fig, axes = plt.subplots(2, 3, figsize=(20, 6))

                    # PA heat map
                    sns.heatmap(pa_heatmap, ax=axes[0,0], cmap='coolwarm', xticklabels=temperature_bins, yticklabels=np.round(loss_bins, 2), annot=False)
                    axes[0,0].set_title('PA Distribution')
                    axes[0,0].set_xlabel('Temperature')
                    axes[0,0].set_ylabel('Loss')

                    # NA heat map
                    sns.heatmap(na_heatmap, ax=axes[0,1], cmap='coolwarm', xticklabels=temperature_bins, yticklabels=np.round(loss_bins, 2), annot=False)
                    axes[0,1].set_title('NA Distribution')
                    axes[0,1].set_xlabel('Temperature')
                    axes[0,1].set_ylabel('Loss')

                    # SA heat map
                    sns.heatmap(sa_heatmap, ax=axes[0,2], cmap='coolwarm', xticklabels=temperature_bins, yticklabels=np.round(loss_bins, 2), annot=False)
                    axes[0,2].set_title('SA Distribution')
                    axes[0,2].set_xlabel('Temperature')
                    axes[0,2].set_ylabel('Loss')

                    # PA predicted heat map
                    sns.heatmap(pa_heatmap_predicted, ax=axes[1,0], cmap='coolwarm', xticklabels=temperature_bins, yticklabels=np.round(loss_bins, 2), annot=False)
                    axes[1,0].set_title('PA Distribution Predicted')
                    axes[1,0].set_xlabel('Temperature')
                    axes[1,0].set_ylabel('Loss')

                    # NA predicted heat map
                    sns.heatmap(na_heatmap_predicted, ax=axes[1,1], cmap='coolwarm', xticklabels=temperature_bins, yticklabels=np.round(loss_bins, 2), annot=False)
                    axes[1,1].set_title('NA Distribution Predicted')
                    axes[1,1].set_xlabel('Temperature')
                    axes[1,1].set_ylabel('Loss')

                    # SA predicted heat map
                    sns.heatmap(sa_heatmap_predicted, ax=axes[1,2], cmap='coolwarm', xticklabels=temperature_bins, yticklabels=np.round(loss_bins, 2), annot=False)
                    axes[1,2].set_title('SA Distribution Predicted')
                    axes[1,2].set_xlabel('Temperature')
                    axes[1,2].set_ylabel('Loss')

                    plt.tight_layout()
                    plt.show()

                    colors = ['k', 'b', 'g']
                    labels = ['PA', 'NA', 'SA']

                    plt.figure(figsize=(10, 6))

                    for i in range(3):
                        # Plot true loss values
                        plt.plot(trueLossValues[i], color=colors[i], label=f'True {labels[i]}')
                        # Plot softmax predictions
                        softmax_values = torch.softmax(therapyState[1][i].squeeze(), dim=-1).detach().numpy()
                        plt.plot(softmax_values, color=colors[i], linestyle='--', label=f'Predicted {labels[i]}')

                    plt.xlabel('Loss bins')
                    plt.ylabel('Probability')
                    plt.title('Comparison of True and Predicted Loss Values')
                    plt.legend()  # Add a legend to clarify the plot
                    plt.grid(True)  # Optional: Add grid for better readability
                    plt.show()

                    loss_prediction_loss.append(lossPredictionLoss.item())
                    loss_bias.append(minimizeLossBias.item())
                    current_user_loss.append(currentUserLoss)
                    iteration += 1
                    epoch_list.append(iteration)

                    currentTemp = self.therapyProtocol.userFullStatePathDistribution[-1][0]
                    currentPA = self.therapyProtocol.userFullStatePathDistribution[-1][1]
                    currentNA = self.therapyProtocol.userFullStatePathDistribution[-1][2]
                    currentSA = self.therapyProtocol.userFullStatePathDistribution[-1][3]

                    print(f"Current temperature: {currentTemp}")
                    print(f"Current PA: {currentPA}")
                    print(f"Current NA: {currentNA}")
                    print(f"Current SA: {currentSA}")
                    if self.plotResults:
                        self.therapyProtocol.plotTherapyResults_nn(epoch_list, loss_prediction_loss, loss_bias, current_user_loss)
            if self.plotResults:
                if self.therapyMethod == "aStarProtocol":
                    self.therapyProtocol.plotTherapyResults(allMaps)
                    print(f"Alpha after iteration: {self.therapyProtocol.percentHeuristic}\n")
                elif self.therapyMethod == "basicProtocol":
                    self.therapyProtocol.plotTherapyResults_basic(allMaps) # For basic protocol, allMaps is the simulated map (only 1)
                    time.sleep(0.1)
            # Check if the therapy has converged.
            self.therapyProtocol.checkConvergence(maxIterations)


if __name__ == "__main__":
    # User parameters.
    userTherapyMethod = "nnProtocol"  # The therapy algorithm to run. Options: "aStarProtocol", "basicProtocol"
    userTemperatureBounds = (30, 50)  # The temperature bounds for the therapy.
    plotTherapyResults = True  # Whether to plot the results.
    userTempBinWidth = 2  # The temperature bin width for the therapy.

    # Simulation parameters.
    currentSimulationParameters = {
        'heuristicMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'simulatedMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'numSimulationHeuristicSamples': 10,  # The number of simulation samples to generate.
        'numSimulationTrueSamples': 50,  # The number of simulation samples to generate.
        'simulateTherapy': True,  # Whether to simulate the therapy.
    }

    # Initialize the therapy protocol
    therapyProtocol = heatTherapyControl(userTemperatureBounds, userTempBinWidth, currentSimulationParameters, therapyMethod=userTherapyMethod, plotResults=plotTherapyResults)

    # Run the therapy protocol.
    therapyProtocol.runTherapyProtocol(maxIterations=500)

    # TODO: lost per epoch  (optimal, predicted, actual loss) (lossPredictionLoss, minimizeLossBias, currentUserLoss) (checked)
    # TODO: change the architecture
    # TODO: save the good copy
    # TODO: figure out the crossentropyerror high issue?
    # TODO: other types of losses for classification (look into them)
    # TODO: add a learning rate scheduler   (checked)
    # TODO: online training
    # TODO: saving and loading models
    # TODO: look into gradient clipping and spectral normalization 

    # loss vs epoch
