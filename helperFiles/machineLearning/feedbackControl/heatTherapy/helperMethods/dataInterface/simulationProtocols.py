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
        self.startingPoints = [47,  0.5966159,   0.69935307,  0.91997683] # Can use a random start point, for now just this

        # Given simulation parameters.
        self.uniformParamSampler = torch.distributions.uniform.Uniform(torch.FloatTensor([modelParameterBounds[0]]), torch.FloatTensor([modelParameterBounds[1]]))
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
        # TODO: prediction is for loss, parameter is the whatever we can control/change
        return self.startingTimes, self.startingParams, self.startingPredictions  # (initialPoints), (initialPoints, numParams), (initialPoints, predictions = emotion states)

    def randomlySamplePoints(self, numPoints=1, lastTimePoint=None):
        # generate a random temperature within the bounds.
        sampledPredictions = self.uniformParamSampler.sample(torch.Size([numPoints, self.numPredictions])).unsqueeze(-1) # torch.size ([numPoints, numPredictions, 1, 1]) ([1,3,1,1])
        sampledParameters = self.uniformParamSampler.sample(torch.Size([numPoints, self.numParameters])).unsqueeze(-1) # torch.size ([numPoints, numParameters, 1, 1]) ([1,1,1,1]) for heat therapy we just have temperature so second dim is 1
        simulatedTimes = self.getSimulatedTimes(numPoints, lastTimePoint) # torch.size([0]) at the beginning. nothing is in here. tensor([], dtype=torch.int64)
        # sampledPredictions dimension: numPoints, numPredictions
        # sampledParameters dimension: numPoints, numParameters
        # simulatedTimes dimension: numPoints

        return simulatedTimes, sampledParameters, sampledPredictions

    # ------------------------ Simulation Interface ------------------------ #

    def initializeSimulatedMaps(self, lossWeights, gausSTDs, applyGaussianFilter):
        # Get the simulated data points.
        simulatedTimes, sampledParameters, sampledPredictions = self.generateSimulatedMap()
        simulatedTimes = torch.cat((torch.tensor([0]), simulatedTimes))
        # sampledPredictions dimension: numSimulationTrueSamples, numPoints, numPredictions, 1, 1; torch.Size([30, 3, 1, 1])
        # sampledParameters dimension: numSimulationTrueSamples, numPoints, numParameters, 1, 1; torch.Size([30, 1, 1, 1])
        # simulatedTimes dimension: numSimulationTrueSamples with concatination of 0 at the beginning torch([30]) otherwise numSimulationTrueSamples - 1

        # Get the simulated matrix from the simulated points.
        simulatedCompiledLoss = self.dataInterface.calculateCompiledLoss(sampledPredictions) # sampledPredictions: numPoints, (PA, NA, SA); 2D array

        # ensure data dimension matches
        simulatedCompiledLoss = simulatedCompiledLoss.unsqueeze(2)  # shape: [30, 1, 1, 1] #TODO: check what each dimension means

        # Compiling the simulated Data for generating probability matrix
        initialSimulatedData = torch.cat((sampledParameters, sampledPredictions, simulatedCompiledLoss), dim=1) # numPoints, (Parameter, emotionPrediction, compiledLoss), 1, 1; torch.Size([30, 5, 1, 1])
        # initialSimulatedData_PA = initialSimulatedData[:, [0, 1], :, :]  # Shape: [30, 2, 1, 1]
        #
        # # Extract for NA: Parameter and second emotion prediction
        # initialSimulatedData_NA = initialSimulatedData[:, [0, 2], :, :]  # Shape: [30, 2, 1, 1]
        #
        # # Extract for SA: Parameter and third emotion prediction
        # initialSimulatedData_SA = initialSimulatedData[:, [0, 3], :, :]  # Shape: [30, 2, 1, 1]
        print('initialSimulatedData: ', initialSimulatedData)
        #print('initialSimulatedData_PA: ', initialSimulatedData_PA)
        self.simulatedMapPA = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.allParameterBins, self.allPredictionBins, gausSTDs[0], noise=0.1, applyGaussianFilter=applyGaussianFilter)
        self.simulatedMapNA = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.allParameterBins, self.allPredictionBins, gausSTDs[1], noise=0.1, applyGaussianFilter=applyGaussianFilter)
        self.simulatedMapSA = self.generalMethods.getProbabilityMatrix(initialSimulatedData, self.allParameterBins, self.allPredictionBins, gausSTDs[2], noise=0.1, applyGaussianFilter=applyGaussianFilter)

        # say that state anxiety has a slightly higher weight and normalize
        self.simulatedMapCompiledLoss = (lossWeights[0]*self.simulatedMapPA + lossWeights[1]*self.simulatedMapNA + lossWeights[2]*self.simulatedMapSA) / torch.sum(lossWeights)

    def getSimulatedCompiledLoss(self, currentParam, currentUserState, newUserTemp=None):
        # Unpack the current user state.
        currentUserTemp = currentParam
        currentUserLoss = self.dataInterface.calculateCompiledLoss(currentUserState)

        newUserTemp = currentUserTemp if newUserTemp is None else newUserTemp

        # Calculate the bin indices for the current and new user states.
        currentLossIndex = self.dataInterface.getBinIndex(self.allPredictionBins, currentUserLoss)
        newTempBinIndex = self.dataInterface.getBinIndex(self.allParameterBins, newUserTemp)

        # Simulate a new user loss.
        sampledLoss_compiled, PA_compiled, NA_compiled, SA_compiled = self.sampleNewLoss(currentLossIndex, newTempBinIndex)
        newUserLoss = self.dataInterface.calculateCompiledLoss(torch.asarray([[PA_compiled, NA_compiled, SA_compiled]]))[0]
        return newUserLoss, PA_compiled, NA_compiled, SA_compiled #TODO: check is sampledLoss_compiled == newUserLoss

    def sampleNewLoss(self, currentLossIndex, newParamIndex, gausSTD=0.1):
        simulatedMapPA = torch.tensor(self.simulatedMapPA, dtype=torch.float32)
        simulatedMapNA = torch.tensor(self.simulatedMapNA, dtype=torch.float32)
        simulatedMapSA = torch.tensor(self.simulatedMapSA, dtype=torch.float32)
        simulatedMapCompiledLoss = torch.tensor(self.simulatedMapCompiledLoss, dtype=torch.float32)
        allPredictionBins = torch.tensor(self.allPredictionBins, dtype=torch.float32)

        # Calculate new loss probabilities and Gaussian boost
        newLossProbabilities = simulatedMapCompiledLoss[newParamIndex] / torch.sum(simulatedMapCompiledLoss[newParamIndex])
        gaussian_boost = self.generalMethods.createGaussianArray(inputData=newLossProbabilities, gausMean=currentLossIndex, gausSTD=gausSTD, torchFlag=True)

        # Combine the two distributions and normalize
        newLossProbabilities = newLossProbabilities + gaussian_boost
        newLossProbabilities = newLossProbabilities / torch.sum(newLossProbabilities)

        # Sample distribution of loss at a certain temperature for PA, NA, SA
        simulatedSpecificMaps = {'PA': simulatedMapPA, 'NA': simulatedMapNA, 'SA': simulatedMapSA}
        specificLossProbabilities = {}

        for key in simulatedSpecificMaps.keys():
            specificLossProbabilities[key] = simulatedSpecificMaps[key][newParamIndex] / torch.sum(simulatedSpecificMaps[key][newParamIndex])
            gaussian_boost = self.generalMethods.createGaussianArray(inputData=specificLossProbabilities[key], gausMean=currentLossIndex, gausSTD=gausSTD, torchFlag=True)
            specificLossProbabilities[key] = specificLossProbabilities[key] + gaussian_boost
            specificLossProbabilities[key] = specificLossProbabilities[key] / torch.sum(specificLossProbabilities[key])

        # specificLossProbabilities contains the normalized loss probabilities for PA, NA, and SA

        return newLossProbabilities, specificLossProbabilities['PA'], specificLossProbabilities['NA'], specificLossProbabilities['SA']

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
        # TODO: (check) Randomly generate (uniform sampling) the times, temperature, PA, NA, SA for each data point.
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
