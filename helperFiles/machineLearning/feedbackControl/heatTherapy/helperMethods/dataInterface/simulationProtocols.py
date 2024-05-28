# General
import torch

# Import helper files.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.dataInterface.dataInterface import dataInterface
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.helperTherapyMethods.generalMethods import generalMethods


class simulationProtocols:
    def __init__(self, allParameterBins, allPredictionBins, predictionBinWidths, modelParameterBounds, numPredictions, numParameters, predictionWeights, optimalNormalizedState, simulationParameters):
        # General parameters.
        self.optimalNormalizedState = optimalNormalizedState
        self.modelParameterBounds = modelParameterBounds
        self.simulationParameters = simulationParameters
        self.predictionBinWidths = predictionBinWidths
        self.allPredictionBins = allPredictionBins
        self.predictionWeights = predictionWeights
        self.allParameterBins = allParameterBins
        self.numPredictions = numPredictions
        self.numParameters = numParameters
        # Hardcoded parameters.
        self.initialTimePoint = 0
        self.initialPoints = 1
        self.timeDelay = 10

        # Given simulation parameters.
        self.uniformParamSampler = torch.distributions.uniform.Uniform(modelParameterBounds[0], modelParameterBounds[1])
        self.numSimulationHeuristicSamples = simulationParameters['numSimulationHeuristicSamples']
        self.numSimulationTrueSamples = simulationParameters['numSimulationTrueSamples']
        self.heuristicMapType = simulationParameters['heuristicMapType']
        self.simulatedMapType = simulationParameters['simulatedMapType']

        # Simulated parameters.
        self.startingTimes, self.startingParams, self.startingPredictions = self.randomlySamplePoints(numPoints=self.initialPoints, lastTimePoint=self.initialTimePoint - self.timeDelay)

        # Uninitialized parameters.
        self.simulatedMapPA = None
        self.simulatedMapNA = None
        self.simulatedMapSA = None
        self.simulatedMapCompiledLoss = None

        # Initialize helper classes.
        self.dataInterface = dataInterface(predictionWeights, optimalNormalizedState)
        self.generalMethods = generalMethods()

        # ------------------------ Simulate Individual Points ------------------------ #

    def getSimulatedTimes(self, numPoints, lastTimePoint=None):
        # If no time is given, start over.
        lastTimePoint = lastTimePoint or -self.timeDelay

        # Simulate the time points.
        currentTimePoint = lastTimePoint + self.timeDelay
        simulatedTimes = torch.arange(currentTimePoint + self.timeDelay, currentTimePoint + numPoints*self.timeDelay, self.timeDelay)
        # simulatedTimes dimension: numPoints

        return simulatedTimes

    def getInitialState(self):
        return self.startingTimes, self.startingParams, self.startingPredictions  # (initialPoints), (initialPoints, numParams), (initialPoints, numPredictions)

    def randomlySamplePoints(self, numPoints=1, lastTimePoint=None):
        # generate a random temperature within the bounds.
        sampledPredictions = self.uniformParamSampler.sample(torch.Size([numPoints, self.numPredictions]))
        sampledParameters = self.uniformParamSampler.sample(torch.Size([numPoints, self.numParameters]))
        simulatedTimes = self.getSimulatedTimes(numPoints, lastTimePoint)
        # sampledPredictions dimension: numPoints, numPredictions
        # sampledParameters dimension: numPoints, numParameters
        # simulatedTimes dimension: numPoints

        return simulatedTimes, sampledParameters, sampledPredictions

    # ------------------------ Simulation Interface ------------------------ #

    def initializeSimulatedMaps(self, lossWeights, gausSTDs, applyGaussianFilter):
        # Get the simulated data points.
        simulatedTimes, sampledParameters, sampledPredictions = self.generateSimulatedMap()
        # sampledPredictions dimension: numSimulationTrueSamples, numPredictions
        # sampledParameters dimension: numSimulationTrueSamples, numParameters
        # simulatedTimes dimension: numSimulationTrueSamples

        # Get the simulated matrix from the simulated points.
        simulatedCompiledLoss = self.dataInterface.calculateCompiledLoss(sampledPredictions)
        self.simulatedMapPA = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.allParameterBins, self.allPredictionBins, gausSTDs[0], noise=0.1, applyGaussianFilter=applyGaussianFilter)
        self.simulatedMapNA = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.allParameterBins, self.allPredictionBins, gausSTDs[1], noise=0.1, applyGaussianFilter=applyGaussianFilter)
        self.simulatedMapSA = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.allParameterBins, self.allPredictionBins, gausSTDs[2], noise=0.1, applyGaussianFilter=applyGaussianFilter)

        # say that state anxiety has a slightly higher weight
        self.simulatedMapCompiledLoss = (lossWeights[0]*self.simulatedMapPA + lossWeights[1]*self.simulatedMapNA + lossWeights[2]*self.simulatedMapSA) / torch.sum(lossWeights)

    def getSimulatedLoss(self, currentUserState, newUserTemp=None):
        # Unpack the current user state.
        currentUserTemp, currentUserLoss = currentUserState
        newUserTemp = currentUserTemp if newUserTemp is None else newUserTemp

        # Calculate the bin indices for the current and new user states.
        currentLossIndex = self.dataInterface.getBinIndex(self.allPredictionBins, currentUserLoss)
        newTempBinIndex = self.dataInterface.getBinIndex(self.allParameterBins, newUserTemp)

        # Simulate a new user loss.
        PA, NA, SA, PA_dist, NA_dist, SA_dist = self.sampleNewLoss(currentLossIndex, newTempBinIndex)
        PA_np = PA.detach().numpy()
        NA_np = NA.detach().numpy()
        SA_np = SA.detach().numpy()

        newUserLoss = self.dataInterface.calculateCompiledLoss(torch.asarray([[PA_np, NA_np, SA_np]]))[0]
        return newUserLoss, PA, NA, SA, PA_dist, NA_dist, SA_dist

    def sampleNewLoss(self, currentLossIndex, newTempBinIndex, gausSTD=0.1):
        simulatedMapPA = torch.tensor(self.simulatedMapPA, dtype=torch.float32)
        simulatedMapNA = torch.tensor(self.simulatedMapNA, dtype=torch.float32)
        simulatedMapSA = torch.tensor(self.simulatedMapSA, dtype=torch.float32)
        simulatedMapCompiledLoss = torch.tensor(self.simulatedMap, dtype=torch.float32)
        allPredictionBins = torch.tensor(self.allPredictionBins, dtype=torch.float32)

        # Calculate new loss probabilities and Gaussian boost
        newLossProbabilities = simulatedMap[newTempBinIndex] / torch.sum(simulatedMap[newTempBinIndex])
        gaussian_boost = self.generalMethods.createGaussianArray(inputData=newLossProbabilities, gausMean=currentLossIndex, gausSTD=gausSTD, torchFlag=True)

        # Combine the two distributions and normalize
        newLossProbabilities = newLossProbabilities + gaussian_boost
        newLossProbabilities = newLossProbabilities / torch.sum(newLossProbabilities)

        # Sample distribution of loss at a certain temperature for PA, NA, SA
        newLossProbabilities_PA = simulatedMapPA[newTempBinIndex] / torch.sum(simulatedMapPA[newTempBinIndex])
        gaussian_boost = self.generalMethods.createGaussianArray(inputData=newLossProbabilities_PA, gausMean=currentLossIndex, gausSTD=gausSTD, torchFlag=True)

        # Update the temperature for annealing
        self.update_temperature()
        print('initial_temperature: ', self.gumbelTemperature)

        return newUserLoss_PA, newUserLoss_NA, newUserLoss_SA, loss_distribution_perTemp_PA, loss_distribution_perTemp_NA, loss_distribution_perTemp_SA

    def sampleSingleLoss(self, newLossProbabilities, allPredictionBins):
        # Normalizing the loss distribution.
        newLossProbabilities = newLossProbabilities / torch.sum(newLossProbabilities)

        newUserLoss_SA = torch.sum(newLossProbabilities * allPredictionBins)
        print('newUserLoss_SA: ', newUserLoss_SA)
        print('soft_sample_SA: ', newLossProbabilities)

        return newUserLoss_SA, newLossProbabilities

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

    def uniformSampling(self, numSimulationSamples, lastTimePoint=None):
        # Randomly generate (uniform sampling) the times, temperature, PA, NA, SA for each data point.
        simulatedTimes, sampledParameters, sampledPredictions = self.randomlySamplePoints(numPoints=numSimulationSamples, lastTimePoint=lastTimePoint)
        # sampledPredictions dimension: numSimulationSamples, numPredictions
        # sampledParameters dimension: numSimulationSamples, numParameters
        # simulatedTimes dimension: numSimulationSamples

        return simulatedTimes, sampledParameters, sampledPredictions

    # TODO: Implement the following methods.
    def linearSampling(self, numSimulationSamples, lastTimePoint=None):
        # Randomly generate (uniform sampling) the times, temperature, PA, NA, SA for each data point.
        simulatedTimes, sampledParameters, sampledPredictions = self.uniformSampling(numSimulationSamples=numSimulationSamples, lastTimePoint=lastTimePoint)
        # sampledPredictions dimension: numSimulationSamples, numPredictions
        # sampledParameters dimension: numSimulationSamples, numParameters
        # simulatedTimes dimension: numSimulationSamples

        # Add a bias towards higher values.
        sampledPredictions = sampledPredictions.pow(2)
        sampledParameters = sampledParameters.pow(2)
        simulatedTimes = simulatedTimes.pow(2)

        return simulatedTimes, sampledParameters, sampledPredictions

    # TODO: Implement the following methods.
    def parabolicSampling(self, numSimulationSamples, lastTimePoint=None):
        # Randomly generate (uniform sampling) the times, temperature, PA, NA, SA for each data point.
        simulatedTimes, sampledParameters, sampledPredictions = self.uniformSampling(numSimulationSamples=numSimulationSamples, lastTimePoint=lastTimePoint)
        # sampledPredictions dimension: numSimulationSamples, numPredictions
        # sampledParameters dimension: numSimulationSamples, numParameters
        # simulatedTimes dimension: numSimulationSamples

        # Add a bias towards higher values.
        sampledPredictions = sampledPredictions.pow(2)
        sampledParameters = sampledParameters.pow(2)
        simulatedTimes = simulatedTimes.pow(2)

        return simulatedTimes, sampledParameters, sampledPredictions
