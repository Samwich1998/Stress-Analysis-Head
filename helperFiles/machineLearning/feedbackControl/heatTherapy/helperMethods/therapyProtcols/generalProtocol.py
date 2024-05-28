# General
import numpy as np
import abc

from .helperTherapyMethods.generalMethods import generalMethods
from .plottingProtocols.plottingProtocolsMain import plottingProtocolsMain
from ..dataInterface.dataInterface import dataInterface
# Import helper files.
from ..dataInterface.empatchProtocols import empatchProtocols
from ..dataInterface.simulationProtocols import simulationProtocols


class generalProtocol(abc.ABC):
    def __init__(self, temperatureBounds, simulationParameters):
        # General parameters.
        self.predictionOrder = ["Positive Affect", "Negative Affect", "State Anxiety"]  # The order of the mental health predictions.
        self.simulateTherapy = simulationParameters['simulateTherapy']  # Whether to simulate the therapy.
        self.temperatureBounds = temperatureBounds  # The temperature bounds for the therapy.
        self.userFullStatePathDistribution = []  # The path of the user state with loss distribution. Tailored for simulation nnProtocol (T, PA_distribution, NA_distribution, SA_distribution)
        self.applyGaussianFilter = True     # Whether to apply a Gaussian filter on the discrete maps.
        self.temperatureTimepoints = []     # Time delays for each discrete temperature-loss pair. (time, T)
        self.simulateTherapy = False        # Whether to simulate the therapy.
        self.finishedTherapy = False        # Whether the therapy has finished.
        self.userFullStatePath = []         # The path of the user state. Order: (T, PA, NA, SA)
        self.userStatePath = []             # The path of the user state. Order: (T, Loss)

        # Hardcoded information.
        self.lossWeights = np.array([0.1, 0.1, 0.8])    # The weights for the loss function. [PA, NA, SA]
        self.gausSTD = np.array([2.5, 1.5])      # The standard deviation for the Gaussian distribution: [T, L]
        self.optimalState = [1, 0, 0]            # The final goal for the therapy. [PA, NA, SA]
        self.numTemperatures = 1            # The number of temperatures to predict.
        self.lossBounds = (0, 1)            # The bounds for the loss function.
        self.lossBinWidth = 0.1             # The bin width for the loss function.
        self.tempBinWidth = 2               # The bin width for the user temperature.
        # Specify hardcoded-related parameters.
        self.optimalLoss = [self.lossBounds[1], self.lossBounds[0], self.lossBounds[0]]  # The optimal loss for the therapy: PA, NA, SA
        self.numLosses = len(self.optimalLoss)  # The number of losses to predict.

        # Initialize the loss and temperature bins.
        self.temp_bins = dataInterface.initializeBins(self.temperatureBounds, self.tempBinWidth)
        self.loss_bins = dataInterface.initializeBins(self.lossBounds, self.lossBinWidth)
        # Initialize the number of bins for the temperature and loss.
        self.numTempBins = len(self.temp_bins)
        self.numLossBins = len(self.loss_bins)

        # If we are simulating.
        if self.simulateTherapy:
            # Define a helper class for simulation parameters.
            self.simulationProtocols = simulationProtocols(self.temp_bins, self.loss_bins, self.lossBinWidth, self.temperatureBounds, self.lossBounds, self.numLosses, self.lossWeights, self.optimalState, simulationParameters)
        else:
            # Define a helper class for experimental parameters.
            self.empatchProtocols = empatchProtocols(self.predictionOrder)
        # initialize helper classes.
        self.plottingProtocolsMain = plottingProtocolsMain(self.temperatureBounds, self.numTempBins, self.tempBinWidth, self.lossBounds, self.numLossBins, self.lossBinWidth)
        self.dataInterface = dataInterface(self.lossWeights, self.optimalState, self.temperatureBounds)
        self.generalMethods = generalMethods()

        # Initialize protocol parameters.
        self.initializeMaps()

    # ------------------------ Track User States ------------------------ #

    def initializeMaps(self):
        if self.simulateTherapy:
            self.simulationProtocols.initializeSimulatedMaps(self.lossWeights, self.gausSTD, self.applyGaussianFilter)
        else:
            # real data points
            initialSimulatedStates = self.empatchProtocols.getTherapyData()
            # initialSimulatedStates = self.simulationProtocols.generateSimulatedMap(self.simulationProtocols.numSimulationTrueSamples, simulatedMapType=self.simulationProtocols.simulatedMapType)
            initialSimulatedData = self.dataInterface.compileStates(initialSimulatedStates)  # initialSimulatedData dimension: numSimulationTrueSamples, (T, L).
            self.simulationProtocols.NA_map_simulated = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.temp_bins, self.loss_bins, self.gausSTD, noise=0.05, applyGaussianFilter=self.applyGaussianFilter)
            self.simulationProtocols.SA_map_simulated = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.temp_bins, self.loss_bins, self.gausSTD, noise=0.1, applyGaussianFilter=self.applyGaussianFilter)
            self.simulationProtocols.PA_map_simulated = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.temp_bins, self.loss_bins, self.gausSTD, noise=0.0, applyGaussianFilter=self.applyGaussianFilter)

            # say that state anxiety has a slightly higher weight
            self.simulationProtocols.simulatedMap = 0.3 * self.simulationProtocols.PA_map_simulated + 0.3 * self.simulationProtocols.NA_map_simulated + 0.4 * self.simulationProtocols.SA_map_simulated

    def initializeUserState(self, userName):
        # Get the user information.
        timePoint, userState = self.getCurrentState()  # userState: (T, PA, NA, SA)
        tempIndex = self.dataInterface.getBinIndex(self.temp_bins, userState[0])

        # Track the user state and time delay.
        self.temperatureTimepoints.append((timePoint, tempIndex))  # temperatureTimepoints: (numEpochs, 2=(time, tempIndex))
        self.userFullStatePath.append(userState)  # userFullStatePath: (numEpochs, 4=(T, PA, NA, SA))

        # Calculate the initial user loss.
        initialUserState = self.dataInterface.compileStates(userState)[-1]

        # Initialize to the current state
        newUserLoss_simulated, PA, NA, SA, PA_dist, NA_dist, SA_dist = self.simulationProtocols.getSimulatedLoss(initialUserState, newUserTemp=None)  # userState[0] # _dist shape = (11.) probability distributions
        self.userFullStatePathDistribution.append([userState[0], PA_dist, NA_dist, SA_dist])

    def getCurrentState(self):
        if self.simulateTherapy:
            # Simulate a new time point by adding a constant delay factor.
            lastTimePoint = self.temperatureTimepoints[-1][0] if len(self.temperatureTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)
            getFirstPoint = self.simulationProtocols.getFirstPoint()
        else:
            # TODO: Implement a method to get the current user state.
            # Simulate a new time.
            lastTimePoint = self.temperatureTimepoints[-1][0] if len(self.temperatureTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)
            getFirstPoint = self.simulationProtocols.getFirstPoint()

        # Returning timePoint, (T, PA, NA, SA)
        return timePoint, getFirstPoint

    def getNextState(self, newUserTemp):
        if self.simulateTherapy:
            # Simulate a new time.
            lastTimePoint = self.temperatureTimepoints[-1][0] if len(self.temperatureTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)

            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA, _, _, _ = self.simulationProtocols.getSimulatedLoss(self.userStatePath[-1], newUserTemp)
            newUserLoss_simulated, newUserLoss_PA_simulated, newUserLoss_NA_simulated, newUserLoss_SA_simulated, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated = self.simulationProtocols.getSimulatedLoss(self.userStatePath[-1], newUserTemp)
            # User state update
            tempIndex = self.dataInterface.getBinIndex(self.temp_bins, newUserTemp)
            self.userStatePath.append([newUserTemp, newUserLoss_simulated])
            self.temperatureTimepoints.append((timePoint, tempIndex))
            self.userFullStatePath.append([newUserTemp, PA, NA, SA])
            self.userStatePath.append([newUserTemp, newUserLoss])
            return newUserLoss_simulated, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated
        else:
            # TODO: Implement a method to get the next user state.
            # Simulate a new time.
            lastTimePoint = self.temperatureTimepoints[-1][0] if len(self.temperatureTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)
            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA, PA_dist, NA_dist, SA_dist = self.simulationProtocols.getSimulatedLoss(self.userStatePath[-1], newUserTemp)
        # Get the bin index for the new temperature.
        tempIndex = self.dataInterface.getBinIndex(self.temp_bins, newUserTemp)
        # Update the user state.
        self.temperatureTimepoints.append((timePoint, tempIndex))
        self.userFullStatePath.append([newUserTemp, PA, NA, SA])
        self.userStatePath.append([newUserTemp, newUserLoss])
        return newUserLoss, PA_dist, NA_dist, SA_dist
        # return newUserLoss, PA, NA, SA

    def checkConvergence(self, maxIterations):
        # Check if the therapy has converged.
        if maxIterations is not None:
            if len(self.userStatePath) >= maxIterations:
                self.finishedTherapy = True
        else:
            # TODO: Implement a convergence check. Maybe based on stagnant loss.
            pass

    # ------------------------ Child Class Contract ------------------------ #

    @abc.abstractmethod
    def updateTherapyState(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")
