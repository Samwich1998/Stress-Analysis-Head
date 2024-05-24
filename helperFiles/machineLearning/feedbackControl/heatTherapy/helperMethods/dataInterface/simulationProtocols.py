# General
import numpy as np
import torch

class simulationProtocols:
    def __init__(self, temp_bins, loss_bins, lossBinWidth, temperatureBounds, lossBounds, simulationParameters):
        # General parameters.
        self.startingPoint = [47,  0.5966159,   0.69935307,  0.91997683]
        self.lossBinWidth = lossBinWidth
        self.lossBounds = lossBounds
        self.temp_bins = temp_bins
        self.loss_bins = loss_bins
        self.timeDelay = 10

        # Simulation parameters
        self.numSimulationHeuristicSamples = simulationParameters['numSimulationHeuristicSamples']
        self.numSimulationTrueSamples = simulationParameters['numSimulationTrueSamples']
        self.heuristicMapType = simulationParameters['heuristicMapType']
        self.simulatedMapType = simulationParameters['simulatedMapType']
        self.temperatureBounds = temperatureBounds
        self.simulatedMap = None

        self.PA_map_simulated = None
        self.NA_map_simulated = None
        self.SA_map_simulated = None

        # Gumbel-Softmax parameters
        self.initial_temperature = 5
        self.temperature_decay = 0.99

    # ------------------------ Simulation Interface ------------------------ #

    def getSimulatedTime(self, lastTimePoint=None):
        # Simulate a new time.
        return lastTimePoint + self.timeDelay if lastTimePoint is not None else 0



    def gumbel_softmax_sample(self, logits):
        gumbels = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel(0, 1)
        gumbels = (logits + gumbels) / self.initial_temperature  # Add gumbels and divide by temperature
        return torch.nn.functional.softmax(gumbels, dim=-1)

    def update_temperature(self):
        self.initial_temperature *= self.temperature_decay

    def sampleNewLoss(self, currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, bufferZone=0):
        simulatedMap = torch.tensor(self.simulatedMap, dtype=torch.float32)
        PA_map_simulated = torch.tensor(self.PA_map_simulated, dtype=torch.float32)
        NA_map_simulated = torch.tensor(self.NA_map_simulated, dtype=torch.float32)
        SA_map_simulated = torch.tensor(self.SA_map_simulated, dtype=torch.float32)
        loss_bins = torch.tensor(self.loss_bins, dtype=torch.float32)

        # Standard deviation for the Gaussian boost
        std = 0.1

        # Calculate new loss probabilities and Gaussian boost
        newLossProbabilities = simulatedMap[newTempBinIndex] / torch.sum(simulatedMap[newTempBinIndex])
        gaussian_boost = torch.exp(-0.5 * ((torch.arange(len(newLossProbabilities), dtype=torch.float32) - currentLossIndex) / std) ** 2)
        gaussian_boost = gaussian_boost / torch.sum(gaussian_boost)

        # Combine the two distributions and normalize
        newLossProbabilities = newLossProbabilities + gaussian_boost
        newLossProbabilities = newLossProbabilities / torch.sum(newLossProbabilities)

        # Use Gumbel-Softmax for differentiable sampling
        soft_sample = self.gumbel_softmax_sample(newLossProbabilities)

        # Sample distribution of loss at a certain temperature for PA, NA, SA
        newLossProbabilities_PA = PA_map_simulated[newTempBinIndex] / torch.sum(PA_map_simulated[newTempBinIndex])
        gaussian_boost_PA = torch.exp(-0.5 * ((torch.arange(len(newLossProbabilities_PA), dtype=torch.float32) - currentLossIndex) / std) ** 2)
        gaussian_boost_PA = gaussian_boost_PA / torch.sum(gaussian_boost_PA)

        newLossProbabilities_PA = newLossProbabilities_PA / torch.sum(newLossProbabilities_PA)
        newLossProbabilities_PA = newLossProbabilities_PA.clone().detach().requires_grad_(True)
        soft_sample_PA = self.gumbel_softmax_sample(newLossProbabilities_PA)
        newUserLoss_PA = torch.sum(soft_sample_PA * loss_bins)
        print('newUserLoss_PA: ', newUserLoss_PA)
        print('soft_sample_PA: ', soft_sample_PA)
        print(f"Gradient tracking enabled: {soft_sample_PA.requires_grad}")
        loss_distribution_perTemp_PA = newLossProbabilities_PA.clone().detach().requires_grad_(True)

        newLossProbabilities_NA = NA_map_simulated[newTempBinIndex] / torch.sum(NA_map_simulated[newTempBinIndex])
        gaussian_boost_NA = torch.exp(-0.5 * ((torch.arange(len(newLossProbabilities_NA), dtype=torch.float32) - currentLossIndex) / std) ** 2)
        gaussian_boost_NA = gaussian_boost_NA / torch.sum(gaussian_boost_NA)

        newLossProbabilities_NA = newLossProbabilities_NA / torch.sum(newLossProbabilities_NA)
        newLossProbabilities_NA = newLossProbabilities_NA.clone().detach().requires_grad_(True)
        soft_sample_NA = self.gumbel_softmax_sample(newLossProbabilities_NA)
        newUserLoss_NA = torch.sum(soft_sample_NA * loss_bins)
        print('newUserLoss_NA: ', newUserLoss_NA)
        print('soft_sample_NA: ', soft_sample_NA)
        print(f"Gradient tracking enabled: {soft_sample_NA.requires_grad}")
        loss_distribution_perTemp_NA = newLossProbabilities_NA.clone().detach().requires_grad_(True)

        newLossProbabilities_SA = SA_map_simulated[newTempBinIndex] / torch.sum(SA_map_simulated[newTempBinIndex])
        gaussian_boost_SA = torch.exp(-0.5 * ((torch.arange(len(newLossProbabilities_SA), dtype=torch.float32) - currentLossIndex) / std) ** 2)
        gaussian_boost_SA = gaussian_boost_SA / torch.sum(gaussian_boost_SA)

        newLossProbabilities_SA = newLossProbabilities_SA / torch.sum(newLossProbabilities_SA)
        newLossProbabilities_SA = newLossProbabilities_SA.clone().detach().requires_grad_(True)
        soft_sample_SA = self.gumbel_softmax_sample(newLossProbabilities_SA)
        newUserLoss_SA = torch.sum(soft_sample_SA * loss_bins)
        print('newUserLoss_SA: ', newUserLoss_SA)
        print('soft_sample_SA: ', soft_sample_SA)
        print(f"Gradient tracking enabled: {soft_sample_SA.requires_grad}")
        loss_distribution_perTemp_SA = newLossProbabilities_SA.clone().detach().requires_grad_(True)


        # Update the temperature for annealing
        self.update_temperature()
        print('initial_temperature: ', self.initial_temperature)

        return newUserLoss_PA, newUserLoss_NA, newUserLoss_SA, loss_distribution_perTemp_PA, loss_distribution_perTemp_NA, loss_distribution_perTemp_SA

    def sampleNewLoss_HMM(self, currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, bufferZone=0.01):
        # if we changed the temperature.
        if newTempBinIndex != currentTempBinIndex or np.random.rand() < 0.1:
            # Sample a new loss from the distribution.
            newLossProbabilities = self.simulatedMap[newTempBinIndex] / np.sum(self.simulatedMap[newTempBinIndex])
            gaussian_boost = np.exp(-0.5 * ((np.arange(len(newLossProbabilities)) - currentLossIndex) / 0.1) ** 2)
            gaussian_boost = gaussian_boost / np.sum(gaussian_boost)

            # Combine the two distributions.
            newLossProbabilities = newLossProbabilities + gaussian_boost
            newLossProbabilities = newLossProbabilities / np.sum(newLossProbabilities)

            # Sample a new loss from the distribution.
            newLossBinIndex = np.random.choice(a=len(newLossProbabilities), p=newLossProbabilities)
            newUserLoss = self.loss_bins[newLossBinIndex]
        else:
            newUserLoss = currentUserLoss + np.random.normal(loc=0, scale=0.01)

        return max(self.loss_bins[0] + bufferZone, min(self.loss_bins[-1] - bufferZone, newUserLoss))

    def getFirstPoint(self):
        return self.startingPoint  # (T, PA, NA, SA)

    # ------------------------ Sampling Methods ------------------------ #

    def generateSimulatedMap(self, numSimulationSamples, simulatedMapType=None):
        simulatedMapType = simulatedMapType if simulatedMapType is not None else self.simulatedMapType
        """ Final dimension: numSimulationSamples, (T, PA, NA, SA); 2D array """
        if simulatedMapType == "uniformSampling":
            return self.uniformSampling(numSimulationSamples)
        elif simulatedMapType == "linearSampling":
            return self.linearSampling(numSimulationSamples)
        elif simulatedMapType == "parabolicSampling":
            return self.parabolicSampling(numSimulationSamples)
        else:
            raise Exception()

    def uniformSampling(self, numSimulationSamples):
        # Randomly generate (uniform sampling) the temperature, PA, NA, SA for each data point.
        simulatePoints = np.random.rand(numSimulationSamples, 4)

        # Adjust the temperature to fit within the bounds.
        temperatureRange = self.temperatureBounds[1] - self.temperatureBounds[0]
        simulatePoints[:, 0] = self.temperatureBounds[0] + temperatureRange * simulatePoints[:, 0]

        return simulatePoints

    def linearSampling(self, numSimulationSamples):
        simulatePoints = np.zeros((numSimulationSamples, 4))

        linear_temps = np.linspace(self.temperatureBounds[0], self.temperatureBounds[1], numSimulationSamples)
        simulatePoints[:, 0] = linear_temps

        linear_losses = np.linspace(self.lossBounds[0], self.lossBounds[1], numSimulationSamples)
        simulatePoints[:, 1:] = np.random.rand(numSimulationSamples, 3) * linear_losses[:, np.newaxis]

        return simulatePoints

    def parabolicSampling(self, numSimulationSamples):
        simulatePoints = np.zeros((numSimulationSamples, 4))

        # Generate parabolic temperature distribution
        t = np.linspace(0, 1, numSimulationSamples)  # Normalized linear space
        parabolic_temps = self.temperatureBounds[0] + (self.temperatureBounds[1] - self.temperatureBounds[0]) * t ** 2
        simulatePoints[:, 0] = parabolic_temps + np.random.rand(numSimulationSamples)

        # Generate parabolic loss distribution
        t = np.linspace(0, 1, numSimulationSamples)  # Normalized linear space
        parabolic_losses = self.lossBounds[0] + (self.lossBounds[1] - self.lossBounds[0]) * t ** 1.2
        simulatePoints[:, 1:] = np.random.rand(numSimulationSamples, 3) * parabolic_losses[:, np.newaxis]
        simulatePoints[:, 1:] = simulatePoints[:, 1:] * np.random.rand(numSimulationSamples, 3)

        return simulatePoints
