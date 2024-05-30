# General
import torch
import abc

# Import helper files.
from .....modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from .plottingProtocols.plottingProtocolsMain import plottingProtocolsMain
from ..dataInterface.simulationProtocols import simulationProtocols
from ..dataInterface.empatchProtocols import empatchProtocols
from .helperTherapyMethods.generalMethods import generalMethods
from ..dataInterface.dataInterface import dataInterface


class generalTherapyProtocol(abc.ABC):
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters):
        # General parameters.
        self.unNormalizedParameterBinWidths = unNormalizedParameterBinWidths  # The parameter bounds for the therapy.
        self.simulateTherapy = simulationParameters['simulateTherapy']  # Whether to simulate the therapy.
        self.initialParameterBounds = initialParameterBounds  # The parameter bounds for the therapy.
        self.modelParameterBounds = [0, 1]  # The model parameter bounds for the therapy.
        self.applyGaussianFilter = True  # Whether to apply a Gaussian filter on the discrete maps.
        self.finishedTherapy = False  # Whether the therapy has finished.

        # Initialize the hard-coded survey information.
        self.compileModelInfoClass = compileModelInfo()  # The class for compiling model information.
        # Get information from the hard-coded survey information.
        self.predictionBinWidths = self.compileModelInfoClass.standardErrorMeasurements  # using the SEM as the bin width for the losses (PA, NA, SA)
        self.optimalPredictions = self.compileModelInfoClass.optimalPredictions  # The bounds for the mental health predictions.
        self.predictionWeights = self.compileModelInfoClass.predictionWeights  # The weights for the loss function. [PA, NA, SA]
        self.predictionBounds = self.compileModelInfoClass.predictionBounds  # The bounds for the mental health predictions.
        self.predictionOrder = self.compileModelInfoClass.predictionOrder  # The order of the mental health predictions.

        # Convert to torch tensors.
        self.unNormalizedParameterBinWidths = torch.tensor(self.unNormalizedParameterBinWidths)  # Dimensions: numParameters
        self.initialParameterBounds = torch.tensor(self.initialParameterBounds)  # Dimensions: numParameters, 2 #i.e.(lower and upper bounds): tensor([35, 50])
        self.modelParameterBounds = torch.tensor(self.modelParameterBounds)  # Dimensions: 2 # tensor([0, 1]) normalized already
        self.predictionBinWidths = torch.tensor(self.predictionBinWidths)  # Dimensions: numPredictions
        self.optimalPredictions = torch.tensor(self.optimalPredictions)  # Dimensions: numPredictions
        self.predictionBounds = torch.tensor(self.predictionBounds)  # Dimensions: numPredictions, 2
        # Get the parameters in the correct data format.
        if self.initialParameterBounds.ndim == 1: self.initialParameterBounds = torch.unsqueeze(self.initialParameterBounds, dim=0) # tensor([[35, 50]])
        if self.predictionBounds.ndim == 1: self.predictionBounds = torch.unsqueeze(self.predictionBounds, dim=0) # ([[5, 25], [5, 25], [20, 80]])
        self.numParameters = len(self.initialParameterBounds)  # The number of parameters. # 1 for now

        # Calculated parameters.
        self.parameterBinWidths = dataInterface.normalizeParameters(currentParamBounds=self.initialParameterBounds - self.initialParameterBounds[:, 0:1], normalizedParamBounds=self.modelParameterBounds, currentParamValues=self.unNormalizedParameterBinWidths)
        self.predictionBinWidths = dataInterface.normalizeParameters(currentParamBounds=self.predictionBounds - self.predictionBounds[:, 0:1], normalizedParamBounds=self.modelParameterBounds, currentParamValues=self.predictionBinWidths)
        self.optimalNormalizedState = dataInterface.normalizeParameters(currentParamBounds=self.predictionBounds, normalizedParamBounds=self.modelParameterBounds, currentParamValues=self.optimalPredictions)
        self.gausParameterSTDs = self.parameterBinWidths.clone()  # The standard deviation for the Gaussian distribution for parameters.
        self.numPredictions = len(self.optimalNormalizedState)  # The number of losses to predict.
        self.gausLossSTDs = self.predictionBinWidths.clone()  # The standard deviation for the Gaussian distribution for losses.

        # Initialize the loss and parameter bins.
        self.allParameterBins = dataInterface.initializeAllBins(self.modelParameterBounds, self.parameterBinWidths)    # Note this is an UNEVEN 2D list. [[parameter]] bin list
        self.allPredictionBins = dataInterface.initializeAllBins(self.modelParameterBounds, self.predictionBinWidths)  # Note this is an UNEVEN 2D list. [[PA], [NA], [SA]] bin list

        # Initialize the number of bins for the parameter and loss.
        self.allNumParameterBins = [len(self.allParameterBins[parameterInd]) for parameterInd in range(self.numParameters)] # Parameter number of Bins in the list
        self.allNumPredictionBins = [len(self.allPredictionBins[lossInd]) for lossInd in range(self.numPredictions)] #PA, NA, SA number of bins in the list

        # Define a helper class for experimental parameters.
        self.simulationProtocols = simulationProtocols(self.allParameterBins, self.allPredictionBins, self.predictionBinWidths, self.modelParameterBounds, self.numPredictions, self.numParameters, self.predictionWeights, self.optimalNormalizedState, simulationParameters)
        self.plottingProtocolsMain = plottingProtocolsMain(self.modelParameterBounds, self.allNumParameterBins, self.parameterBinWidths, self.predictionBounds, self.allNumPredictionBins, self.predictionBinWidths)
        self.dataInterface = dataInterface(self.predictionWeights, self.optimalNormalizedState)
        self.empatchProtocols = empatchProtocols(self.predictionOrder, self.predictionBounds, self.modelParameterBounds)
        self.generalMethods = generalMethods()

        # Reset the therapy parameters.
        self.userMentalStatePath = None
        self.paramStatePath = None
        self.timePoints = None
        self.resetTherapy()

    def resetTherapy(self):
        # Reset the therapy parameters.
        self.userMentalStatePath = []  # The path of the user's mental state: PA, NA, SA
        self.finishedTherapy = False  # Whether the therapy has finished.
        self.paramStatePath = []  # The path of the therapy parameters: numParameters
        self.timePoints = []  # The time points for the therapy.

        # Reset the therapy maps.
        self.initializeMaps()
        print('passed here')

    # ------------------------ Track User States ------------------------ #

    def initializeMaps(self):
        if self.simulateTherapy:
            self.simulationProtocols.initializeSimulatedMaps(self.predictionWeights, self.gausLossSTDs, self.applyGaussianFilter)
        else:
            # initialSimulatedStates = self.simulationProtocols.generateSimulatedMap(self.simulationProtocols.numSimulationTrueSamples, simulatedMapType=self.simulationProtocols.simulatedMapType)
            # real data points
            temperature, pa, na, sa = self.empatchProtocols.getTherapyData()
            # sort the temperature, pa, na, sa into correct format passed to generate initialSimulatedData
            initialSimulatedStates = torch.stack([temperature, pa, na, sa], dim=1)
            initialSimulatedData = self.dataInterface.calculateCompiledLoss(initialSimulatedStates)  # initialSimulatedData dimension: numSimulationTrueSamples, (T, L).
            self.simulationProtocols.NA_map_simulated = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.allParameterBins, self.allPredictionBins, self.gausLossSTDs, noise=0.05, applyGaussianFilter=self.applyGaussianFilter)
            self.simulationProtocols.SA_map_simulated = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.allParameterBins, self.allPredictionBins, self.gausLossSTDs, noise=0.1, applyGaussianFilter=self.applyGaussianFilter)
            self.simulationProtocols.PA_map_simulated = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.allParameterBins, self.allPredictionBins, self.gausLossSTDs, noise=0.0, applyGaussianFilter=self.applyGaussianFilter)

            # say that state anxiety has a slightly higher weight
            self.simulationProtocols.simulatedMap = 0.3 * self.simulationProtocols.PA_map_simulated + 0.3 * self.simulationProtocols.NA_map_simulated + 0.4 * self.simulationProtocols.SA_map_simulated

    def initializeUserState(self, userName):
        # Get the user information.
        timePoint, userState = self.getCurrentState()  # userState: (T, PA, NA, SA)

        # Track the user state and time delay.
        self.userFullStatePath.append(userState)  # userFullStatePath: (numEpochs, 4=(timePoint, T, PA, NA, SA))

        # Calculate the initial user loss.
        initialUserState = self.dataInterface.compileStates(userState)[-1]

        # Initialize to the current state
        newUserLoss_simulated, PA, NA, SA, PA_dist, NA_dist, SA_dist = self.simulationProtocols.getSimulatedLoss(initialUserState, newUserTemp=None)  # userState[0] # _dist shape = (11.) probability distributions
        self.userFullStatePathDistribution.append([userState[0], PA_dist, NA_dist, SA_dist])

    def getCurrentState(self):
        if self.simulateTherapy:
            # Simulate a new time point by adding a constant delay factor.
            lastTimePoint = self.parameterTimepoints[-1][0] if len(self.parameterTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)
            getFirstPoint = self.simulationProtocols.getFirstPoint()
        else:
            # TODO: Implement a method to get the current user state.
            # Simulate a new time.
            lastTimePoint = self.parameterTimepoints[-1][0] if len(self.parameterTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)
            getFirstPoint = self.simulationProtocols.getFirstPoint()

        # Returning timePoint, (T, PA, NA, SA)
        return timePoint, getFirstPoint

    def getNextState(self, newUserTemp):
        if self.simulateTherapy:
            # Simulate a new time.
            lastTimePoint = self.parameterTimepoints[-1][0] if len(self.parameterTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)

            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA, _, _, _ = self.simulationProtocols.getSimulatedLoss(self.userStatePath[-1], newUserTemp)
            newUserLoss_simulated, newUserLoss_PA_simulated, newUserLoss_NA_simulated, newUserLoss_SA_simulated, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated = self.simulationProtocols.getSimulatedLoss(self.userStatePath[-1],
                                                                                                                                                                                                                     newUserTemp)
            # User state update
            tempIndex = self.dataInterface.getBinIndex(self.allParameterBins, newUserTemp)
            self.userStatePath.append([newUserTemp, newUserLoss_simulated])
            self.parameterTimepoints.append((timePoint, tempIndex))
            self.userFullStatePath.append([newUserTemp, PA, NA, SA])
            self.userStatePath.append([newUserTemp, newUserLoss])
            return newUserLoss_simulated, PA_dist_simulated, NA_dist_simulated, SA_dist_simulated
        else:
            # TODO: Implement a method to get the next user state.
            # Simulate a new time.
            lastTimePoint = self.parameterTimepoints[-1][0] if len(self.parameterTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)
            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA, PA_dist, NA_dist, SA_dist = self.simulationProtocols.getSimulatedLoss(self.userStatePath[-1], newUserTemp)
        # Get the bin index for the new parameter.
        tempIndex = self.dataInterface.getBinIndex(self.allParameterBins, newUserTemp)
        # Update the user state.
        self.parameterTimepoints.append((timePoint, tempIndex))
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
