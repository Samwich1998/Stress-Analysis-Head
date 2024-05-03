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

    # ------------------------ Simulation Interface ------------------------ #

    def getSimulatedTime(self, lastTimePoint=None):
        # Simulate a new time.
        return lastTimePoint + self.timeDelay if lastTimePoint is not None else 0

    def sampleNewLoss(self, currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, bufferZone=0.01):
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
            newLossBinIndex_PA = np.random.choice(a=len(newLossProbabilities), p=newLossProbabilities)
            newUserLoss_PA = self.loss_bins[newLossBinIndex_PA]
            newLossBinIndex_NA = np.random.choice(a=len(newLossProbabilities), p=newLossProbabilities)
            newUserLoss_NA = self.loss_bins[newLossBinIndex_NA]
            newLossBinIndex_SA = np.random.choice(a=len(newLossProbabilities), p=newLossProbabilities)
            newUserLoss_SA = self.loss_bins[newLossBinIndex_SA]
        else:
            newUserLoss_PA = currentUserLoss + np.random.normal(loc=0, scale=0.01)
            newUserLoss_NA = currentUserLoss + np.random.normal(loc=0, scale=0.01)
            newUserLoss_SA = currentUserLoss + np.random.normal(loc=0, scale=0.01)

        return max(self.loss_bins[0] + bufferZone, min(self.loss_bins[-1] - bufferZone, newUserLoss_PA)), max(self.loss_bins[0] + bufferZone, min(self.loss_bins[-1] - bufferZone, newUserLoss_NA)), max(self.loss_bins[0] + bufferZone, min(self.loss_bins[-1] - bufferZone, newUserLoss_SA))

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
