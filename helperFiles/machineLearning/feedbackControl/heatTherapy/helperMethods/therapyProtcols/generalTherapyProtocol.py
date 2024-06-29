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
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod):
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
        self.predictionBinWidths = self.compileModelInfoClass.standardErrorMeasurements  # using the SEM as bin width for the losses (PA, NA, SA)
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
        print('allParameterBins: ', self.allParameterBins)
        print('allPredictionBins: ', self.allPredictionBins)

        # Initialize the number of bins for the parameter and loss.
        self.allNumParameterBins = [len(self.allParameterBins[parameterInd]) for parameterInd in range(self.numParameters)]  # Parameter number of Bins in the list
        self.allNumPredictionBins = [len(self.allPredictionBins[lossInd]) for lossInd in range(self.numPredictions)]  #PA, NA, SA number of bins in the list

        # Define a helper class for experimental parameters.
        self.simulationProtocols = simulationProtocols(self.allParameterBins, self.allPredictionBins, self.predictionBinWidths, self.modelParameterBounds, self.numPredictions, self.numParameters, self.predictionWeights, self.optimalNormalizedState, simulationParameters)
        self.plottingProtocolsMain = plottingProtocolsMain(self.modelParameterBounds, self.allNumParameterBins, self.parameterBinWidths, self.predictionBounds, self.allNumPredictionBins, self.predictionBinWidths)
        #self.empatchProtocols = empatchProtocols(self.predictionOrder, self.predictionBounds, self.modelParameterBounds, therapyMethod=therapyMethod)
        self.dataInterface = dataInterface(self.predictionWeights, self.optimalNormalizedState)
        self.generalMethods = generalMethods()

        # Reset the therapy parameters.
        self.userMentalStatePath = None
        self.paramStatePath = None
        self.timePoints = None
        self.userMentalStateCompiledLoss = None
        self.resetTherapy()

    def resetTherapy(self):
        # Reset the therapy parameters.
        self.userMentalStatePath = []  # The path of the user's mental state: PA, NA, SA
        self.finishedTherapy = False  # Whether the therapy has finished.
        self.paramStatePath = []  # The path of the therapy parameters: numParameters
        self.timePoints = [] # The time points for the therapy.
        self.userMentalStateCompiledLoss = []  # The compiled loss for the user's mental state.

        # Reset the therapy maps.
        self.initializeMaps()
        print('passed here')

    # ------------------------ Track User States ------------------------ #
    def initializeMaps(self):
        if self.simulateTherapy:
            self.simulationProtocols.initializeSimulatedMaps(self.predictionWeights, self.gausLossSTDs, self.applyGaussianFilter)
        else:
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
        timePoints, parameters, emptionStates = self.getCurrentState()  # TODO: (double check) dim: numPoints, timePoint: t; emotionStates: (PA, NA, SA); prediction: predict the next state
        # Track the user state and time delay.
        self.timePoints.append(timePoints) # TODO: check dimension
        self.paramStatePath.append(parameters) # TODO: check dimension
        self.userMentalStatePath.append(emptionStates) # TODO: check dimension

        # Calculate the initial user loss.
        compiledLoss = self.dataInterface.calculateCompiledLoss(emptionStates[-1])  # compile the loss state for the current emotion state
        self.userMentalStateCompiledLoss.append(compiledLoss) # TODO: check dimension


    def getCurrentState(self):
        if self.simulateTherapy:
            # Simulate a new time point by adding a constant delay factor.
            currentTime, currentParam, currentPredictions = self.simulationProtocols.getInitialState() #TODO: check starting point, what are them
            print('passed here')
            return currentTime, currentParam, currentPredictions
        else:
            # TODO: Implement a method to get the current user state.
            # Simulate a new time.
            pass

        # Returning timePoint, (T, PA, NA, SA)

    def getNextState(self, newParamValues):
        if self.simulateTherapy:
            # Simulate a new time.
            lastTimePoint = self.timePoints[-1][0] if len(self.timePoints) != 0 else 0
            newTimePoint = self.simulationProtocols.getSimulatedTimes(self.simulationProtocols.initialPoints, lastTimePoint)

            # get the current user state
            currentParam = self.paramStatePath[-1]
            currentEmotionStates = self.userMentalStatePath[-1]

            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA = self.simulationProtocols.getSimulatedCompiledLoss(currentParam, currentEmotionStates, newParamValues)

            # User state update
            self.timePoints.append(newTimePoint)  # TODO: check dimension
            self.paramStatePath.append(newParamValues)  # TODO: check dimension
            self.userMentalStatePath.append((PA, NA, SA))  # TODO: check dimension
            self.userMentalStateCompiledLoss.append(newUserLoss)  # TODO: check dimension
        else:
            pass

    def checkConvergence(self, maxIterations):
        # Check if the therapy has converged.
        if maxIterations is not None:
            if len(self.userMentalStatePath) >= maxIterations:
                self.finishedTherapy = True
        else:
            # TODO: Implement a convergence check. Maybe based on stagnant loss.
            pass

    # ------------------------ Child Class Contract ------------------------ #

    @abc.abstractmethod
    def updateTherapyState(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")
