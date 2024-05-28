# General
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch


class plottingProtocolsMain:
    def __init__(self, modelParameterBounds, allParameterBins, parameterBinWidths, predictionBounds, allNumPredictionBins, predictionBinWidths):
        # General parameters.
        self.modelParameterBounds = modelParameterBounds  # The bounds for the parameter.
        self.parameterBinWidths = parameterBinWidths  # The width of the parameter bins.
        self.predictionBinWidths = predictionBinWidths  # The width of the loss bins.
        self.allNumPredictionBins = allNumPredictionBins  # The number of loss bins.
        self.allParameterBins = allParameterBins  # The number of parameter bins.
        self.predictionBounds = predictionBounds  # The bounds for the loss.

        # Initialize heatmaps for plotting
        heatmap_size = (self.allParameterBins, self.allNumPredictionBins)
        self.pa_heatmap = np.zeros(heatmap_size)
        self.na_heatmap = np.zeros(heatmap_size)
        self.sa_heatmap = np.zeros(heatmap_size)
        self.pa_heatmap_predicted = np.zeros(heatmap_size)
        self.na_heatmap_predicted = np.zeros(heatmap_size)
        self.sa_heatmap_predicted = np.zeros(heatmap_size)

    @staticmethod
    def plotSimulatedMap(simulatedMap):
        # Initialize the user maps.
        plt.imshow(simulatedMap)
        plt.show()

    def plotTherapyResults(self, userStatePath, final_map_pack):
        # Unpack the user states.
        benefitFunction, heuristic_map, personalized_map, simulated_map = final_map_pack
        num_steps = len(userStatePath)

        # Titles and maps to be plotted.
        titles = ['Projected Benefit Map', 'Heuristic Map', 'Personalized Map', 'Simulated Map']
        maps = [benefitFunction, heuristic_map, personalized_map, simulated_map]
        plottingImage = None

        # Setup subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Color and alpha gradient for the path
        alphas = np.linspace(0.1, 1, num_steps - 1)  # Adjust alpha for segments

        # Plotting each map with the corresponding user state path
        for i, (map_to_plot, title) in enumerate(zip(maps, titles)):
            ax = axs[i % 2, i // 2]
            plottingImage = ax.imshow(map_to_plot.T, cmap='coolwarm', extent=[self.modelParameterBounds[0], self.modelParameterBounds[1], self.predictionBounds[0], self.predictionBounds[1]], aspect='auto', origin='lower')

            # Plot past user states with fading red line
            for j in range(num_steps - 1):
                ax.plot([userStatePath[j][0], userStatePath[j + 1][0]],
                        [userStatePath[j][1], userStatePath[j + 1][1]],
                        color=(0, 0, 0, alphas[j]), linewidth=2)

            ax.scatter(userStatePath[-1][0], userStatePath[-1][1], color='tab:red', label='Current State', edgecolor='black', s=75, zorder=10)
            ax.set_title(f'{title} (After Iteration {num_steps})')
            ax.set_xlabel('parameter')
            ax.set_ylabel('Loss')
            ax.legend()

        # Adjust layout to prevent overlap and ensure titles and labels are visible
        fig.tight_layout()
        fig.colorbar(plottingImage, ax=axs.ravel().tolist(), label='Probability')
        plt.show()

        print(f"New current state after iteration {num_steps + 1}: parameter = {userStatePath[-1][0]}, Loss = {userStatePath[-1][1]}")

    # ------------------------ Basic Protocol plotting ------------------------ #

    def plotTherapyResults_basic(self, userStatePath, simulated_map):
        # Unpack the user states.
        num_steps = len(userStatePath)

        # Set up figure
        fig, ax = plt.subplots(figsize=(6, 6))

        # Color and alpha gradient for the path
        alphas = np.linspace(0.1, 1, num_steps - 1)  # Adjust alpha for segments

        # Plotting the simulated map with the corresponding user state path
        im = ax.imshow(simulated_map.T, cmap='coolwarm', extent=[self.modelParameterBounds[0], self.modelParameterBounds[1], self.predictionBounds[0], self.predictionBounds[1]], aspect='auto', origin='lower')

        # Plot past user states with fading color
        for j in range(num_steps - 1):
            ax.plot([userStatePath[j][0], userStatePath[j + 1][0]],
                    [userStatePath[j][1], userStatePath[j + 1][1]],
                    color=(1, 0, 0, alphas[j]), linewidth=2)  # using red color for the path

        # Highlight the current state
        ax.scatter(userStatePath[-1][0], userStatePath[-1][1], color='tab:red', label='Current State', edgecolor='black', s=75, zorder=10)

        # Set titles and labels
        ax.set_title(f'Simulated Map (After Iteration {num_steps})')
        ax.set_xlabel('parameter')
        ax.set_ylabel('Loss')
        ax.legend()

        # Add colorbar for the map
        fig.colorbar(im, ax=ax, label='Probability')
        plt.show()

        # Print current state information
        print(f"New current state after iteration {num_steps + 1}: parameter = {userStatePath[-1][0]}, Loss = {userStatePath[-1][1]}")

    # ------------------------ NN Plotting ------------------------ #

    @staticmethod
    def plotTherapyResults_nn(epoch_list, loss_prediction_loss):
        plt.style.use('seaborn-v0_8-pastel')
        plt.figure(figsize=(10, 6))

        # Plot the loss prediction loss
        plt.plot(epoch_list, loss_prediction_loss, label='Loss Prediction Loss', linestyle='-', color='darkblue', linewidth=2, markersize=8)

        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Therapy Results per Epoch', fontsize=16)
        plt.legend(frameon=True, loc='best', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

    @staticmethod
    def plot_loss_comparison(trueLossValues, therapyState):
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
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_heatmaps(self, currentPA, currentNA, currentSA, currentPA_pred, currentNA_pred, currentSA_pred, currentTemp):
        # Define bins
        parameter_bins = np.arange(self.modelParameterBounds[0], self.modelParameterBounds[1], self.parameterBinWidths)
        loss_bins = np.arange(self.predictionBounds[0], self.predictionBounds[1], self.predictionBinWidths)

        # Determine the index
        parameter_index = np.digitize(currentTemp, parameter_bins) - 1

        # Fill the heatmaps
        for i in range(self.allNumPredictionBins):
            self.pa_heatmap[i, parameter_index] = currentPA[i]
            self.na_heatmap[i, parameter_index] = currentNA[i]
            self.sa_heatmap[i, parameter_index] = currentSA[i]
            self.pa_heatmap_predicted[i, parameter_index] = currentPA_pred[i]
            self.na_heatmap_predicted[i, parameter_index] = currentNA_pred[i]
            self.sa_heatmap_predicted[i, parameter_index] = currentSA_pred[i]

        # Plotting the heat maps
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        heatmaps = [
            (self.pa_heatmap, 'PA Distribution'),
            (self.na_heatmap, 'NA Distribution'),
            (self.sa_heatmap, 'SA Distribution'),
            (self.pa_heatmap_predicted, 'PA Distribution Predicted'),
            (self.na_heatmap_predicted, 'NA Distribution Predicted'),
            (self.sa_heatmap_predicted, 'SA Distribution Predicted')
        ]

        for idx, (heatmap, title) in enumerate(heatmaps):
            sns.heatmap(heatmap, ax=axes[idx // 3, idx % 3], cmap='coolwarm', xticklabels=parameter_bins, yticklabels=np.round(loss_bins, 2), annot=False)
            axes[idx // 3, idx % 3].set_title(title)
            axes[idx // 3, idx % 3].set_xlabel('parameter')
            axes[idx // 3, idx % 3].set_ylabel('Loss')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_delta_loss_comparison(epoch_list, deltaListPA, deltaLossNA, deltaLossSA, predictedLossPA, predictedLossNA, predictedLossSA):
        # Setting up the plot style
        plt.style.use('seaborn-v0_8-pastel')
        fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

        # Plotting delta PA and predicted PA
        axs[0].plot(epoch_list, deltaListPA, label='delta PA', linestyle='-', color='darkblue', linewidth=2, markersize=8)
        axs[0].plot(epoch_list, predictedLossPA, label='predicted PA', linestyle='--', color='pink', linewidth=2, markersize=8)
        axs[0].set_ylabel('Loss changes (PA)', fontsize=14)
        axs[0].set_title('Delta vs Predicted PA', fontsize=16)
        axs[0].legend(frameon=True, loc='best', fontsize=12)
        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plotting delta NA and predicted NA
        axs[1].plot(epoch_list, deltaLossNA, label='delta NA', linestyle='-', color='firebrick', linewidth=2, markersize=8)
        axs[1].plot(epoch_list, predictedLossNA, label='predicted NA', linestyle='--', color='brown', linewidth=2, markersize=8)
        axs[1].set_ylabel('Loss changes (NA)', fontsize=14)
        axs[1].set_title('Delta vs Predicted NA', fontsize=16)
        axs[1].legend(frameon=True, loc='best', fontsize=12)
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plotting delta SA and predicted SA
        axs[2].plot(epoch_list, deltaLossSA, label='delta SA', linestyle='-', color='forestgreen', linewidth=2, markersize=8)
        axs[2].plot(epoch_list, predictedLossSA, label='predicted SA', linestyle='--', color='olive', linewidth=2, markersize=8)
        axs[2].set_xlabel('Epoch', fontsize=14)
        axs[2].set_ylabel('Loss changes (SA)', fontsize=14)
        axs[2].set_title('Delta vs Predicted SA', fontsize=16)
        axs[2].legend(frameon=True, loc='best', fontsize=12)
        axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adjusting layout
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_temp(epoch_list, temp_list):
        plt.style.use('seaborn-v0_8-pastel')
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_list, temp_list, label='delta PA', marker='o', linestyle='-', color='darkblue', linewidth=2, markersize=8)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('temp', fontsize=14)
        plt.title('Therapy Results per Epoch', fontsize=16)
        plt.legend(frameon=True, loc='best', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
