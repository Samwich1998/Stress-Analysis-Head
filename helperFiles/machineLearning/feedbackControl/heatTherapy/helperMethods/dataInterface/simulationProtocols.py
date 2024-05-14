# General
import numpy as np


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

    # ------------------------ Simulation Interface ------------------------ #

    def getSimulatedTime(self, lastTimePoint=None):
        # Simulate a new time.
        return lastTimePoint + self.timeDelay if lastTimePoint is not None else 0



    def sampleNewLoss(self, currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, bufferZone=0):
        # if we changed the temperature.
        loss_distribution_perTemp_PA = []
        loss_distribution_perTemp_NA = []
        loss_distribution_perTemp_SA = []
        if newTempBinIndex != currentTempBinIndex or np.random.rand() < 0.1:
            # Sample a new loss from the distribution.
            std = 0.1
            newLossProbabilities = self.simulatedMap[newTempBinIndex] / np.sum(self.simulatedMap[newTempBinIndex])
            gaussian_boost = np.exp(-0.5 * ((np.arange(len(newLossProbabilities)) - currentLossIndex) / std) ** 2)
            gaussian_boost = gaussian_boost / np.sum(gaussian_boost)

            # Sample a new loss from the distribution.
            newLossBinIndex_PA = np.random.choice(a=len(newLossProbabilities))
            print('#######newLossBinIndex_PA: ', newLossBinIndex_PA)
            newUserLoss_PA = self.loss_bins[newLossBinIndex_PA]
            newLossBinIndex_NA = np.random.choice(a=len(newLossProbabilities))
            newUserLoss_NA = self.loss_bins[newLossBinIndex_NA]
            newLossBinIndex_SA = np.random.choice(a=len(newLossProbabilities))
            newUserLoss_SA = self.loss_bins[newLossBinIndex_SA]

            # sample distribution of loss at a certain temperature for PA, NA, SA (nn simulation purposes) random distribution based on the map
            newLossProbabilities_PA = self.PA_map_simulated[newTempBinIndex] / np.sum(self.PA_map_simulated[newTempBinIndex])
            gaussian_boost_PA = np.exp(-0.5 * ((np.arange(len(newLossProbabilities_PA)) - newLossBinIndex_PA) / std) ** 2)
            gaussian_boost_PA = gaussian_boost_PA / np.sum(gaussian_boost_PA)

            newLossProbabilities_NA = self.NA_map_simulated[newTempBinIndex] / np.sum(self.NA_map_simulated[newTempBinIndex])
            gaussian_boost_NA = np.exp(-0.5 * ((np.arange(len(newLossProbabilities_NA)) - newLossBinIndex_NA) / std) ** 2)
            gaussian_boost_NA = gaussian_boost_NA / np.sum(gaussian_boost_NA)

            newLossProbabilities_SA = self.SA_map_simulated[newTempBinIndex] / np.sum(self.SA_map_simulated[newTempBinIndex])
            gaussian_boost_SA = np.exp(-0.5 * ((np.arange(len(newLossProbabilities_SA)) - newLossBinIndex_SA) / std) ** 2)
            gaussian_boost_SA = gaussian_boost_SA / np.sum(gaussian_boost_SA)

            # Combine the two distributions.
            newLossProbabilities = newLossProbabilities + gaussian_boost
            newLossProbabilities = newLossProbabilities / np.sum(newLossProbabilities)

            # get the distribution of loss at a certain temperature for PA, NA, SA (nn simulation purposes) different gaussian boost
            newLossProbabilities_PA = newLossProbabilities_PA / np.sum(newLossProbabilities_PA)
            loss_distribution_perTemp_PA = newLossProbabilities_PA.copy()

            newLossProbabilities_NA = newLossProbabilities_NA / np.sum(newLossProbabilities_NA)
            loss_distribution_perTemp_NA = newLossProbabilities_NA.copy()

            newLossProbabilities_SA = newLossProbabilities_SA / np.sum(newLossProbabilities_SA)
            loss_distribution_perTemp_SA = newLossProbabilities_SA.copy()
        else:
            print('passed checkmark %%%%%%%')
            newUserLoss_PA = currentUserLoss + np.random.normal(loc=0, scale=0.01)
            newUserLoss_NA = currentUserLoss + np.random.normal(loc=0, scale=0.01)
            newUserLoss_SA = currentUserLoss + np.random.normal(loc=0, scale=0.01)
            newLossProbabilities = self.simulatedMap[newTempBinIndex] / np.sum(self.simulatedMap[newTempBinIndex])
            noise = np.random.normal(loc=0, scale=0.01, size=newLossProbabilities.shape)
            newLossProbabilities = newLossProbabilities + noise
            # Ensure no negative probabilities
            newLossProbabilities = np.clip(newLossProbabilities, 0, None)
            newLossProbabilities = newLossProbabilities / np.sum(newLossProbabilities)
            newLossProbabilities_PA = self.PA_map_simulated[newTempBinIndex] / np.sum(self.PA_map_simulated[newTempBinIndex])
            noise_PA = np.random.normal(loc=0, scale=0.01, size=newLossProbabilities_PA.shape)
            newLossProbabilities_PA = newLossProbabilities_PA + noise_PA
            # Ensure no negative probabilities
            newLossProbabilities_PA = np.clip(newLossProbabilities_PA, 0, None)
            newLossProbabilities_PA = newLossProbabilities_PA / np.sum(newLossProbabilities_PA)
            newLossProbabilities_NA = self.NA_map_simulated[newTempBinIndex] / np.sum(self.NA_map_simulated[newTempBinIndex])
            noise_NA = np.random.normal(loc=0, scale=0.01, size=newLossProbabilities_NA.shape)
            newLossProbabilities_NA = newLossProbabilities_NA + noise_NA
            # Ensure no negative probabilities
            newLossProbabilities_NA = np.clip(newLossProbabilities_NA, 0, None)
            newLossProbabilities_NA = newLossProbabilities_NA / np.sum(newLossProbabilities_NA)
            newLossProbabilities_SA = self.SA_map_simulated[newTempBinIndex] / np.sum(self.SA_map_simulated[newTempBinIndex])
            noise_SA = np.random.normal(loc=0, scale=0.01, size=newLossProbabilities_SA.shape)
            newLossProbabilities_SA = newLossProbabilities_SA + noise_SA
            # Ensure no negative probabilities
            newLossProbabilities_SA = np.clip(newLossProbabilities_SA, 0, None)
            newLossProbabilities_SA = newLossProbabilities_SA / np.sum(newLossProbabilities_SA)
            loss_distribution_perTemp_PA = newLossProbabilities_PA.copy()
            loss_distribution_perTemp_NA = newLossProbabilities_NA.copy()
            loss_distribution_perTemp_SA = newLossProbabilities_SA.copy()

        return max(self.loss_bins[0] + bufferZone, min(self.loss_bins[-1] - bufferZone, newUserLoss_PA)), max(self.loss_bins[0] + bufferZone, min(self.loss_bins[-1] - bufferZone, newUserLoss_NA)), max(self.loss_bins[0] + bufferZone, min(self.loss_bins[-1] - bufferZone, newUserLoss_SA)), loss_distribution_perTemp_PA, loss_distribution_perTemp_NA, loss_distribution_perTemp_SA


    def sampleNewLoss_offline(self, currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, bufferZone=0):
        # if we changed the temperature.
        loss_distribution_perTemp_PA = []
        loss_distribution_perTemp_NA = []
        loss_distribution_perTemp_SA = []

        # Sample a new loss from the distribution.
        std = 0.1
        newLossProbabilities = self.simulatedMap[newTempBinIndex] / np.sum(self.simulatedMap[newTempBinIndex])
        gaussian_boost = np.exp(-0.5 * ((np.arange(len(newLossProbabilities)) - currentLossIndex) / std) ** 2)
        gaussian_boost = gaussian_boost / np.sum(gaussian_boost)

        # Sample a new loss from the distribution.
        newLossBinIndex_PA = np.random.choice(a=len(newLossProbabilities))
        newUserLoss_PA = self.loss_bins[newLossBinIndex_PA]
        newLossBinIndex_NA = np.random.choice(a=len(newLossProbabilities))
        newUserLoss_NA = self.loss_bins[newLossBinIndex_NA]
        newLossBinIndex_SA = np.random.choice(a=len(newLossProbabilities))
        newUserLoss_SA = self.loss_bins[newLossBinIndex_SA]

        # sample distribution of loss at a certain temperature for PA, NA, SA (nn simulation purposes) random distribution based on the map
        newLossProbabilities_PA = self.PA_map_simulated[newTempBinIndex] / np.sum(self.PA_map_simulated[newTempBinIndex])

        newLossProbabilities_NA = self.NA_map_simulated[newTempBinIndex] / np.sum(self.NA_map_simulated[newTempBinIndex])

        newLossProbabilities_SA = self.SA_map_simulated[newTempBinIndex] / np.sum(self.SA_map_simulated[newTempBinIndex])

        # Combine the two distributions.
        newLossProbabilities = newLossProbabilities + gaussian_boost
        newLossProbabilities = newLossProbabilities / np.sum(newLossProbabilities)

        # get the distribution of loss at a certain temperature for PA, NA, SA (nn simulation purposes) different gaussian boost
        newLossProbabilities_PA = newLossProbabilities_PA / np.sum(newLossProbabilities_PA)
        loss_distribution_perTemp_PA = newLossProbabilities_PA.copy()

        newLossProbabilities_NA = newLossProbabilities_NA / np.sum(newLossProbabilities_NA)
        loss_distribution_perTemp_NA = newLossProbabilities_NA.copy()

        newLossProbabilities_SA = newLossProbabilities_SA / np.sum(newLossProbabilities_SA)
        loss_distribution_perTemp_SA = newLossProbabilities_SA.copy()
        return max(self.loss_bins[0] + bufferZone, min(self.loss_bins[-1] - bufferZone, newUserLoss_PA)), max(self.loss_bins[0] + bufferZone, min(self.loss_bins[-1] - bufferZone, newUserLoss_NA)), max(self.loss_bins[0] + bufferZone,
                                                                                                                                                                                                         min(self.loss_bins[-1] - bufferZone,
                                                                                                                                                                                                             newUserLoss_SA)), loss_distribution_perTemp_PA, loss_distribution_perTemp_NA, loss_distribution_perTemp_SA

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
