# General
import numpy as np


class dataInterface:

    def __init__(self, lossWeights, optimalState, temperatureBounds):
        # General parameters
        self.temperatureBounds = temperatureBounds
        self.lossWeights = lossWeights
        self.optimalState = optimalState

    def compileStates(self, initialStates):
        # Ensure the input is a numpy array.
        initialStates = np.asarray(initialStates)

        # Interface for 1D data.
        if initialStates.ndim == 1:
            initialStates = np.expand_dims(initialStates, axis=0)

        # Extract the input dimensions.
        numSimulationSamples, numStateTerms = initialStates.shape

        # Initialize data holder
        initialData = np.zeros((numSimulationSamples, 2))  # initialData dimension: numSimulationSamples, (T, Loss); 2D array

        # Convert the simulated points to a 2D matrix.
        initialData[:, 1] = self.calculateLoss(initialStates[:, 1:])
        initialData[:, 0] = initialStates[:, 0]

        return initialData

    def calculateLoss(self, initialMentalStates):
        """ initialStates: numPoints, (PA, NA, SA); 2D array"""
        # Calculate the compiled loss.
        compiledLoss = self.lossWeights * np.power(initialMentalStates - self.optimalState, 2)
        compiledLoss = np.sum(compiledLoss, axis=1)
        # compiledLoss dimension: numPoints

        # Normalize the loss values.
        compiledLoss = compiledLoss / self.lossWeights.sum()
        # compiledLoss dimension: numPoints

        return compiledLoss

    def boundNewTemperature(self, newTemp):
        return max(self.temperatureBounds[0], min(self.temperatureBounds[1], newTemp))

    def standardizeTemperature(self, temperature):
        return (temperature - self.temperatureBounds[0]) / (self.temperatureBounds[1] - self.temperatureBounds[0])

    def unstandardizeTemperature(self, temperature):
        return temperature * (self.temperatureBounds[1] - self.temperatureBounds[0]) + self.temperatureBounds[0]

    # ------------------------ Static Methods ------------------------ #

    @staticmethod
    def getBinIndex(allBins, binValue):
        return min(np.searchsorted(allBins, binValue, side='right'), len(allBins) - 1)

    @staticmethod
    def initializeBins(bounds, binWidth):
        # Generate the temperature bins
        finalBins = np.arange(bounds[0], bounds[1], step=binWidth)

        # Edge case: the last bin must be the upper bound
        if finalBins[-1] != bounds[1]: finalBins = np.append(finalBins, bounds[1])

        return finalBins
