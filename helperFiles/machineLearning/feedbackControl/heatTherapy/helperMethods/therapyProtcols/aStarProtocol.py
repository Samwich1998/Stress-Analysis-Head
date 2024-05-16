# General
import numpy as np

from .generalProtocol import generalProtocol


class aStarProtocol(generalProtocol):
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, learningRate=5):
        super().__init__(temperatureBounds, tempBinWidth, simulationParameters)
        # Define update parameters.
        self.gausSTD = np.array([0.05, 2.5])  # The standard deviation for the Gaussian distribution.
        self.learningRate = learningRate  # The learning rate for the therapy.
        self.discretePersonalizedMap = []  # The discrete personalized map.

        # Bias terms.
        self.percentHeuristic = 1  # The percentage of the heuristic map to use.
        self.explorationBias = 1  # The bias for exploration.
        self.uncertaintyBias = 1  # The bias for uncertainty.

        # Specific A Star Protocol parameters.
        self.tempBinsVisited = np.full(self.numTempBins, False)
        self.decayConstant = 1 / (2 * 3600)  # The decay constant for the personalized map.

        # Initialize the heuristic and personalized maps.
        self.heuristicMap = self.initializeHeuristicMaps()  # Estimate on what temperatures you like. Based on population average.
        self.initializeFirstPersonalizedMap()  # list of probability maps.

    def updateTherapyState(self):
        # Unpack the current user state.
        currentUserState = self.userStatePath[-1]  # Order: (T, Loss) (get the latest user state) each (T, Loss) stored in the userStatePath list
        currentUserTemp, currentUserLoss = currentUserState

        # Update the temperatures visited.
        tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        self.tempBinsVisited[tempBinIndex] = True  # documents the temperature visited by indexing the temperature bin and set the value to true

        # Update the personalized user map.
        self.trackCurrentState(currentUserState)  # Keep track of each discrete temperature-loss pair.
        personalizedMap = self.getUpdatedPersonalizedMap()  # Convert the discrete map to a probability distribution.

        # Get the current time point.
        timePoint, userState = self.getCurrentState()
        self.temperatureTimepoints.append((timePoint, tempBinIndex))

        # Combine the heuristic and personalized maps and update the weighting.
        probabilityMap = self.percentHeuristic * self.heuristicMap + (1 - self.percentHeuristic) * personalizedMap
        self.updateAlpha()

        # Find the best temperature in the gradient direction.
        newUserTemp, benefitFunction = self.findOptimalDirection(probabilityMap, currentUserState)
        newUserTemp = newUserTemp + self.uncertaintyBias * np.random.normal(loc=0, scale=0.5)  # Add noise to the gradient.
        # deltaTemp = self.findNewTemperature(currentUserState, gradientDirection)
        # deltaTemp = deltaTemp + self.uncertaintyBias * np.random.normal(loc=0, scale=0.5)  # Add noise to the gradient.

        # Calculate the new temperature.
        newUserTemp = self.boundNewTemperature(newUserTemp, bufferZone=1)

        return newUserTemp, (benefitFunction, self.heuristicMap, personalizedMap, self.simulationProtocols.simulatedMap)

    def findNewTemperature(self, currentUserState, gradientDirection):
        # Unpack the current user state.
        currentUserTemp, currentUserLoss = currentUserState

        # Determine a direction based on temperature gradient at current loss
        tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        tempGradient = gradientDirection[tempBinIndex]

        return tempGradient

    # ------------------------ Update Parameters ------------------------ #

    def findOptimalDirection(self, probabilityMap, currentUserState):
        # Calculate benefits/loss of exploring/moving.
        potentialLossBenefit = self.loss_bins
        probabilityMap = probabilityMap / probabilityMap.sum(axis=1)[:, np.newaxis]  # Normalize the probability map.

        # Calculate the expected rewards.
        potentialRewards = potentialLossBenefit[np.newaxis, :]
        expectedRewards = probabilityMap * potentialRewards

        # Find the best temperature bin index in the rewards.
        expectedRewardAtTemp = expectedRewards.sum(axis=1)  # Normalize across temperature bins.
        bestTempBinIndex = np.argmin(expectedRewardAtTemp)

        return self.temp_bins[bestTempBinIndex] + self.tempBinWidth / 2, expectedRewards

        # # Compute the gradient.
        # potentialTemperatureRewards = np.gradient(potentialTemperatureRewards)  # Dimension: 2, numTempBins, numLossBins

    def updateAlpha(self):
        # Calculate the percentage of the temperature bins visited.
        percentConfidence = self.tempBinsVisited.sum() / len(self.tempBinsVisited)

        # Update the confidence flags.
        self.percentHeuristic = min(self.percentHeuristic, 1 - percentConfidence) - 0.001
        self.percentHeuristic = min(1.0, max(0.0, self.percentHeuristic))

        # Update the bias terms.
        self.explorationBias = self.percentHeuristic  # TODO
        self.uncertaintyBias = self.percentHeuristic  # TODO

    # ------------------------ Personalization Interface ------------------------ #

    def trackCurrentState(self, currentUserState):
        # Smoothen out the discrete map into a probability distribution.
        probabilityMatrix = self.getProbabilityMatrix([currentUserState])
        self.discretePersonalizedMap.append(probabilityMatrix)  # the discretePersonalizedMap list will store the probability matrix

    @staticmethod
    def personalizedMapWeightingFunc(timeDelays, decay_constant):
        # Ebbinghaus forgetting curve.
        return np.exp(-decay_constant * np.asarray(timeDelays))

    def getUpdatedPersonalizedMap(self):
        # Assert the integrity of the state tracking.
        print(f"Length of temperatureTimepoints: {len(self.temperatureTimepoints)}")
        print(f"Content of temperatureTimepoints: {self.temperatureTimepoints}")
        print(f"Length of discretePersonalizedMap: {len(self.discretePersonalizedMap)}")
        print(f"Content of discretePersonalizedMap: {self.discretePersonalizedMap}")
        assert len(self.temperatureTimepoints) == len(self.discretePersonalizedMap), \
            f"The time delays and discrete maps are not the same length. {len(self.temperatureTimepoints)} {len(self.discretePersonalizedMap)}"
        # Unpack the temperature-timepoints relation.
        tempTimepoints = np.asarray(self.temperatureTimepoints)
        associatedTempInds = tempTimepoints[:, 1]
        timePoints = tempTimepoints[:, 0]

        # Get the weighting for each discrete temperature-loss pair.
        currentTimeDelays = np.abs(timePoints - timePoints[-1])
        personalizedMapWeights = self.personalizedMapWeightingFunc(currentTimeDelays, self.decayConstant)

        # For each temperature bin.
        for tempIndex in range(self.numTempBins):
            # If the temperature bin has been visited.
            if tempIndex in associatedTempInds:
                tempIndMask = associatedTempInds == tempIndex

                # Normalize the weights per this bin.
                personalizedMapWeights[tempIndMask] = personalizedMapWeights[tempIndMask] / personalizedMapWeights[tempIndMask].sum()

        # Perform a weighted average of all the personalized maps.
        personalizedMap = np.sum(self.discretePersonalizedMap * personalizedMapWeights[:, np.newaxis, np.newaxis], axis=0)

        if self.applyGaussianFilter:
            # Smoothen the personalized map.
            personalizedMap = self.smoothenArray(personalizedMap, sigma=self.gausSTD[::-1])

        # Normalize the personalized map.
        personalizedMap = personalizedMap / personalizedMap.sum()  # Normalize along the temperature axis.

        return personalizedMap

    def initializeFirstPersonalizedMap(self):
        # Initialize a uniform personalized map. No bias.
        uniformMap = np.ones((self.numTempBins, self.numLossBins))
        uniformMap = uniformMap / uniformMap.sum()

        # Store the initial personalized map estimate.
        # self.discretePersonalizedMap.append(uniformMap)
        # self.timePoints.append((0, ))

    # ------------------------ Heuristic Interface ------------------------ #

    def initializeHeuristicMaps(self):
        if self.simulateTherapy:
            # Get the simulated data points.
            initialHeuristicStates = self.simulationProtocols.generateSimulatedMap(self.simulationProtocols.numSimulationHeuristicSamples, simulatedMapType=self.simulationProtocols.heuristicMapType)
            # initialHeuristicStates dimension: numSimulationHeuristicSamples, (T, PA, NA, SA); 2D array
        else:
            # Get the real data points.
            initialHeuristicStates = self.empatchProtocols.getTherapyData()

        # Get the heuristic matrix from the simulated points.
        initialHeuristicData = self.compileLossStates(initialHeuristicStates)  # InitialData dim: numPoints, (T, L)
        heuristicMap = self.getProbabilityMatrix(initialHeuristicData)  # Adding Gaussian distributions and normalizing the probability

        return heuristicMap
