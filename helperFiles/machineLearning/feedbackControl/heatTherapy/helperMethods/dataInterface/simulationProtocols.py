# General
import numpy as np
import random
import torch

# Import helper files.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.dataInterface.dataInterface import dataInterface
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.helperTherapyMethods.generalMethods import generalMethods


class simulationProtocols:
    def __init__(self, temp_bins, loss_bins, lossBinWidth, temperatureBounds, lossBounds, numLosses, lossWeights, optimalState, simulationParameters):
        # General parameters.
        self.startingPoint = self.randomlySamplePoint()
        self.lossBinWidth = lossBinWidth
        self.lossBounds = lossBounds
        self.temp_bins = temp_bins
        self.loss_bins = loss_bins
        self.numLosses = numLosses

        # Hardcoded parameters.
        self.timeDelay = 10

        # Simulation parameters.
        self.numSimulationHeuristicSamples = simulationParameters['numSimulationHeuristicSamples']
        self.numSimulationTrueSamples = simulationParameters['numSimulationTrueSamples']
        self.heuristicMapType = simulationParameters['heuristicMapType']
        self.simulatedMapType = simulationParameters['simulatedMapType']
        self.temperatureBounds = temperatureBounds

        # Uninitialized parameters.
        self.PA_map_simulated = None
        self.NA_map_simulated = None
        self.SA_map_simulated = None
        self.simulatedMap = None

        # Gumbel-Softmax parameters
        self.gumbelTemperatureDecay = 0.99
        self.gumbelTemperature = 5

        # Initialize helper classes.
        self.dataInterface = dataInterface(lossWeights, optimalState, temperatureBounds)
        self.generalMethods = generalMethods()

    # ------------------------ Getter Methods ------------------------ #

    def getSimulatedTime(self, lastTimePoint=None):
        return lastTimePoint + self.timeDelay if lastTimePoint is not None else 0

    def getFirstPoint(self):
        return self.startingPoint  # (T, PA, NA, SA)

    def initializeSimulatedMaps(self, lossWeights, gausSTD, applyGaussianFilter):
        # Get the simulated data points.
        initialSimulatedStates = self.generateSimulatedMap()
        # initialSimulatedStates dimension: numSimulationTrueSamples, (T, PA, NA, SA); 2D array

        # Get the simulated matrix from the simulated points.
        initialSimulatedData = self.dataInterface.compileStates(initialSimulatedStates)  # initialSimulatedData dimension: numSimulationTrueSamples, (T, L).
        self.NA_map_simulated = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.temp_bins, self.loss_bins, gausSTD, noise=0.05, applyGaussianFilter=applyGaussianFilter)
        self.SA_map_simulated = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.temp_bins, self.loss_bins, gausSTD, noise=0.1, applyGaussianFilter=applyGaussianFilter)
        self.PA_map_simulated = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.temp_bins, self.loss_bins, gausSTD, noise=0.0, applyGaussianFilter=applyGaussianFilter)

        # say that state anxiety has a slightly higher weight
        self.simulatedMap = (lossWeights[0]*self.PA_map_simulated + lossWeights[1]*self.NA_map_simulated + lossWeights[2]*self.SA_map_simulated) / np.sum(lossWeights)
        print('####simulatedMap: ', self.simulatedMap)

    # ------------------------ Simulation Interface ------------------------ #

    def randomlySamplePoint(self):
        # generate a random temperature within the bounds.
        randomTemperature = np.random.uniform(self.temperatureBounds[0], self.temperatureBounds[1])

        randomLosses = []
        # For each loss value.
        for lossInd in range(self.numLosses):
            # generate a random loss within the bounds.
            randomLosses.append(np.random.uniform(self.lossBounds[0], self.lossBounds[1]))

        # Combine all the loss values.
        return [randomTemperature] + randomLosses

    def gumbel_softmax_sample(self, logits):
        gumbels = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel(0, 1)
        gumbels = (logits + gumbels) / self.gumbelTemperature  # Add gumbels and divide by temperature
        return torch.nn.functional.softmax(gumbels, dim=-1)

    def update_temperature(self):
        self.gumbelTemperature *= self.gumbelTemperatureDecay

    def getSimulatedLoss(self, currentUserState, newUserTemp=None):
        # Unpack the current user state.
        currentUserTemp, currentUserLoss = currentUserState
        newUserTemp = currentUserTemp if newUserTemp is None else newUserTemp

        # Calculate the bin indices for the current and new user states.
        currentTempBinIndex = self.dataInterface.getBinIndex(self.temp_bins, currentUserTemp)
        currentLossIndex = self.dataInterface.getBinIndex(self.loss_bins, currentUserLoss)
        newTempBinIndex = self.dataInterface.getBinIndex(self.temp_bins, newUserTemp)

        # Simulate a new user loss.
        PA, NA, SA, PA_dist, NA_dist, SA_dist = self.sampleNewLoss(currentLossIndex, newTempBinIndex)
        PA_np = PA.detach().numpy()
        NA_np = NA.detach().numpy()
        SA_np = SA.detach().numpy()

        newUserLoss = self.dataInterface.calculateLoss(np.asarray([[PA_np, NA_np, SA_np]]))[0]
        return newUserLoss, PA, NA, SA, PA_dist, NA_dist, SA_dist

    def sampleNewLoss(self, currentLossIndex, newTempBinIndex, gausSTD=0.1):
        PA_map_simulated = torch.tensor(self.PA_map_simulated, dtype=torch.float32)
        NA_map_simulated = torch.tensor(self.NA_map_simulated, dtype=torch.float32)
        SA_map_simulated = torch.tensor(self.SA_map_simulated, dtype=torch.float32)
        simulatedMap = torch.tensor(self.simulatedMap, dtype=torch.float32)
        loss_bins = torch.tensor(self.loss_bins, dtype=torch.float32)

        # Calculate new loss probabilities and Gaussian boost
        newLossProbabilities = simulatedMap[newTempBinIndex] / torch.sum(simulatedMap[newTempBinIndex])
        gaussian_boost = self.generalMethods.createGaussianArray(inputData=newLossProbabilities, gausMean=currentLossIndex, gausSTD=gausSTD, torchFlag=True)

        # Combine the two distributions and normalize
        newLossProbabilities = newLossProbabilities + gaussian_boost
        newLossProbabilities = newLossProbabilities / torch.sum(newLossProbabilities)

        # Sample distribution of loss at a certain temperature for PA, NA, SA
        newLossProbabilities_PA = PA_map_simulated[newTempBinIndex] / torch.sum(PA_map_simulated[newTempBinIndex])
        gaussian_boost = self.generalMethods.createGaussianArray(inputData=newLossProbabilities_PA, gausMean=currentLossIndex, gausSTD=gausSTD, torchFlag=True)


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
        print('initial_temperature: ', self.gumbelTemperature)

        return newUserLoss_PA, newUserLoss_NA, newUserLoss_SA, loss_distribution_perTemp_PA, loss_distribution_perTemp_NA, loss_distribution_perTemp_SA

    def sampleSingleLoss(self, newLossProbabilities_SA, loss_bins):
        # Normalizing the loss distribution.
        newLossProbabilities_SA = newLossProbabilities_SA / torch.sum(newLossProbabilities_SA)
        newLossProbabilities_SA = newLossProbabilities_SA.clone().requires_grad_(True)

        soft_sample_SA = self.gumbel_softmax_sample(newLossProbabilities_SA)
        newUserLoss_SA = torch.sum(soft_sample_SA * loss_bins)
        print('newUserLoss_SA: ', newUserLoss_SA)
        print('soft_sample_SA: ', soft_sample_SA)
        print(f"Gradient tracking enabled: {soft_sample_SA.requires_grad}")
        loss_distribution_perTemp_SA = newLossProbabilities_SA.clone().detach().requires_grad_(True)

        return newUserLoss_SA, loss_distribution_perTemp_SA

    # ------------------------ Sampling Methods ------------------------ #

    def generateSimulatedMap(self):
        """ Final dimension: numSimulationSamples, (T, PA, NA, SA); 2D array """
        if self.simulatedMapType == "uniformSampling":
            return self.uniformSampling(self.numSimulationTrueSamples)
        elif self.simulatedMapType == "linearSampling":
            return self.linearSampling(self.numSimulationTrueSamples)
        elif self.simulatedMapType == "parabolicSampling":
            return self.parabolicSampling(self.numSimulationTrueSamples)
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
