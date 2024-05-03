# General
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
import numpy as np
import random
import abc

# Import helper files.
from ..dataInterface.empatchProtocols import empatchProtocols
from ..dataInterface.simulationProtocols import simulationProtocols


class generalProtocol(abc.ABC):
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters):
        # General parameters.
        self.predictionOrder = ["Positive Affect", "Negative Affect", "State Anxiety"]  # The order of the mental health predictions.
        self.temperatureBounds = temperatureBounds  # The temperature bounds for the therapy.
        self.tempBinWidth = tempBinWidth    # The temperature bin width for the therapy.
        self.applyGaussianFilter = False    # Whether to apply a Gaussian filter on the discrete maps.
        self.temperatureTimepoints = []     # Time delays for each discrete temperature-loss pair. (time, T)
        self.simulateTherapy = False        # Whether to simulate the therapy.
        self.finishedTherapy = False        # Whether the therapy has finished.
        self.userFullStatePath = []         # The path of the user state. Order: (T, PA, NA, SA)
        self.userStatePath = []             # The path of the user state. Order: (T, Loss)

        # Define loss information
        self.lossWeights = np.array([0.1, 0.1, 0.8])    # The weights for the loss function. [PA, NA, SA]
        self.gausSTD = np.array([0.05, 2.5])  # The standard deviation for the Gaussian distribution
        self.finalGoal = np.array([1, 0, 0])     # The final goal for the therapy. [PA, NA, SA]
        self.lossBounds = (0, 1)         # The bounds for the loss function.
        self.lossBinWidth = 0.02         # The bin width for the loss function.

        # Initialize the loss and temperature bins.
        self.temp_bins = self.initializeBins(self.temperatureBounds, self.tempBinWidth, bufferZone=0)
        self.loss_bins = self.initializeBins(self.lossBounds, self.lossBinWidth, bufferZone=0)
        # Initialize the number of bins for the temperature and loss.
        self.numTempBins = len(self.temp_bins)
        self.numLossBins = len(self.loss_bins)

        # If we are simulating.
        if simulationParameters['simulateTherapy']:
            # Define a helper class for simulation parameters.
            self.simulationProtocols = simulationProtocols(self.temp_bins, self.loss_bins, self.lossBinWidth, self.temperatureBounds, self.lossBounds, simulationParameters)
            self.simulateTherapy = True
        else:
            self.empatchProtocols = empatchProtocols(self.predictionOrder)

            self.simulateTherapy = False

            # TODO: Delete this line
            self.simulationProtocols = simulationProtocols(self.temp_bins, self.loss_bins, self.lossBinWidth, self.temperatureBounds, self.lossBounds, simulationParameters)

        # Initialize the user maps.
        self.initializeSimulatedMaps()

    # ------------------------ Track User States ------------------------ #

    def initializeSimulatedMaps(self):
        if self.simulateTherapy:
            # Get the simulated data points.
            initialSimulatedStates = self.simulationProtocols.generateSimulatedMap(self.simulationProtocols.numSimulationTrueSamples, simulatedMapType=self.simulationProtocols.simulatedMapType)
            # initialHeuristicStates dimension: numSimulationHeuristicSamples, (T, PA, NA, SA); 2D array
            # initialSimulatedStates dimension: numSimulationTrueSamples, (T, PA, NA, SA); 2D array

            # Get the simulated matrix from the simulated points.
            initialSimulatedData = self.compileLossStates(initialSimulatedStates)  # initialSimulatedData dimension: numSimulationTrueSamples, (T, L).
            self.simulationProtocols.simulatedMap = self.getProbabilityMatrix(initialSimulatedData)  # Spreading delta function probability.
        else:
            initialSimulatedStates = self.simulationProtocols.generateSimulatedMap(self.simulationProtocols.numSimulationTrueSamples, simulatedMapType=self.simulationProtocols.simulatedMapType)
            initialSimulatedData = self.compileLossStates(initialSimulatedStates)  # initialSimulatedData dimension: numSimulationTrueSamples, (T, L).
            self.simulationProtocols.simulatedMap = self.getProbabilityMatrix(initialSimulatedData)  # Spreading delta function probability.

    @staticmethod
    def smoothenArray(deltaFunctionMatrix, sigma):
        return gaussian_filter(deltaFunctionMatrix, sigma=sigma)

    def createGaussianMap(self, gausMean, gausSTD):
        # Generate a grid for Gaussian distribution calculations
        x, y = np.meshgrid(self.loss_bins, self.temp_bins)

        # Calculate Gaussian distribution values across the grid
        gaussMatrix = np.exp(-0.5 * ((x - gausMean[0]) ** 2 / gausSTD[0] ** 2 + (y - gausMean[1]) ** 2 / gausSTD[1] ** 2))
        gaussMatrix = gaussMatrix / gaussMatrix.sum()  # Normalize the Gaussian matrix

        return gaussMatrix

    def getProbabilityMatrix(self, initialData):
        """ initialData: numPoints, (T, L); 2D array"""
        # Initialize probability matrix holder.
        probabilityMatrix = np.zeros((self.numTempBins, self.numLossBins))

        # Calculate the probability matrix.
        for initialDataPoints in initialData:
            currentUserTemp, currentUserLoss = initialDataPoints

            if self.applyGaussianFilter:
                # Generate a delta function probability.
                tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
                lossBinIndex = self.getBinIndex(self.loss_bins, currentUserLoss)
                probabilityMatrix[tempBinIndex, lossBinIndex] += 1  # map out bins and fill out with discrete values
            else:
                # Generate 2D gaussian matrix.
                gaussianMatrix = self.createGaussianMap(gausMean=(currentUserLoss, currentUserTemp), gausSTD=self.gausSTD)
                probabilityMatrix += gaussianMatrix  # Add the gaussian map to the matrix

        if self.applyGaussianFilter:
            # Smoothen the probability matrix.
            probabilityMatrix = self.smoothenArray(probabilityMatrix, sigma=self.gausSTD[::-1])

        # Normalize the probability matrix.
        probabilityMatrix = probabilityMatrix / probabilityMatrix.sum()

        return probabilityMatrix

    def initializeUserState(self):
        # Get the user information.
        timePoint, userState = self.getCurrentState()  # userState: (T, PA, NA, SA)
        tempIndex = self.getBinIndex(self.temp_bins, userState[0])

        # Track the user state and time delay.
        self.userStatePath.append(self.compileLossStates(np.asarray([userState]))[0])  # userStatePath: (numEpochs, 2=(T, Loss))
        self.temperatureTimepoints.append((timePoint, tempIndex))  # temperatureTimepoints: (numEpochs, 2=(time, tempIndex))
        self.userFullStatePath.append(userState)  # userFullStatePath: (numEpochs, 4=(T, PA, NA, SA))

    def getCurrentState(self):
        if self.simulateTherapy:
            # Simulate a new time.
            lastTimePoint = self.temperatureTimepoints[-1][0] if len(self.temperatureTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)

            return timePoint, self.simulationProtocols.getFirstPoint()  # Returning timePoint, (T, PA, NA, SA)
        else:
            # TODO: Implement a method to get the current user state.
            # Simulate a new time.
            lastTimePoint = self.temperatureTimepoints[-1][0] if len(self.temperatureTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)

            return timePoint, self.simulationProtocols.getFirstPoint()

    def compileLossStates(self, initialStates):
        # Initialize data holder
        initialData = np.zeros((len(initialStates), 2))  # initialData dimension: numSimulationSamples, (T, Loss); 2D array

        # Convert the simulated points to a 2D matrix.
        initialData[:, 1] = self.calculateLoss(initialStates[:, 1:])
        initialData[:, 0] = initialStates[:, 0]

        return initialData

    def getNextState(self, newUserTemp):
        if self.simulateTherapy:
            # Simulate a new time.
            lastTimePoint = self.temperatureTimepoints[-1][0] if len(self.temperatureTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)

            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA = self.getSimulatedLoss(self.userStatePath[-1], newUserTemp)
        else:
            # TODO: Implement a method to get the next user state.
            # Simulate a new time.
            lastTimePoint = self.temperatureTimepoints[-1][0] if len(self.temperatureTimepoints) != 0 else 0
            timePoint = self.simulationProtocols.getSimulatedTime(lastTimePoint)

            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA = self.getSimulatedLoss(self.userStatePath[-1], newUserTemp)
            PA, NA, SA = None, None, None

        # Get the bin index for the new temperature.
        tempIndex = self.getBinIndex(self.temp_bins, newUserTemp)

        # Update the user state.
        self.temperatureTimepoints.append((timePoint, tempIndex))
        self.userFullStatePath.append([newUserTemp, PA, NA, SA])
        self.userStatePath.append([newUserTemp, newUserLoss])

    def get_emotion_state_distinct(self, currentTemp, simulatedEmoMap):
        bin_index = self.getBinIndex(self.temp_bins, currentTemp)

        # Extract the first column which corresponds to temperatures
        temperatures = [row[0] for row in simulatedEmoMap]

        # Get bin indexes for all temperatures
        bin_index_map = [self.getBinIndex(self.temp_bins, temp) for temp in temperatures]
        matching_entries = [simulatedEmoMap[i] for i in range(len(bin_index_map)) if bin_index_map[i] == bin_index]
        if matching_entries:
            # Randomly select one of the matching entries
            selected_entry = random.choice(matching_entries)
            pa, na, sa = selected_entry[1] + np.random.normal(loc=0, scale=0.01), selected_entry[2] + np.random.normal(loc=0, scale=0.01), selected_entry[3] + np.random.normal(loc=0, scale=0.01)
            print('Randomly selected PA, NA, SA: ', (pa, na, sa))
        else:
            print('No matching temperature bins found, initializing PA, NA, SA to a constant value....')
            pa = na = sa = 1.0 / self.numLossBins

        return pa, na, sa

    def checkConvergence(self, maxIterations):
        # Check if the therapy has converged.
        if maxIterations is not None:
            if len(self.userStatePath) >= maxIterations:
                self.finishedTherapy = True
        else:
            # TODO: Implement a convergence check. Maybe based on stagnant loss.
            pass

    # ------------------------ Loss Interface ------------------------ #

    def calculateLoss(self, initialMentalStates):
        """ initialStates: numPoints, (PA, NA, SA); 2D array"""
        compiledLoss = np.sum(self.lossWeights * np.power(initialMentalStates - self.finalGoal, 2), axis=1) / self.lossWeights.sum()

        return compiledLoss

    def getSimulatedLoss(self, currentUserState, newUserTemp):
        # Unpack the current user state.
        currentUserTemp, currentUserLoss = currentUserState
        print('currentUserState:', currentUserState)
        # Calculate the bin indices for the current and new user states.
        currentTempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        # z = self.calculateLoss(currentUserLoss)
        print('currentUserLoss:', currentUserLoss)
        currentLossIndex = self.getBinIndex(self.loss_bins, currentUserLoss)
        newTempBinIndex = self.getBinIndex(self.temp_bins, newUserTemp)

        # Simulate a new user loss.
        PA, NA, SA = self.simulationProtocols.sampleNewLoss(currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, bufferZone=0.01)
        newUserLoss = self.calculateLoss(np.asarray([[PA, NA, SA]]))[0]
        return newUserLoss, PA, NA, SA

    # ------------------------ initial Heuristic map for NN adaptive training ------------------------ #
    def createGaussianMap_nn(self, gausMean, gausSTD):
        # Generate a grid for Gaussian distribution calculations
        x, y = np.meshgrid(self.loss_bins, self.temp_bins)

        # Calculate Gaussian distribution values across the grid
        gaussMatrix = np.exp(-0.5 * ((x - gausMean[0]) ** 2 / gausSTD[0] ** 2 + (y - gausMean[1]) ** 2 / gausSTD[1] ** 2))
        gaussMatrix = gaussMatrix / gaussMatrix.sum()  # Normalize the Gaussian matrix

        return gaussMatrix

    def getProbabilityMatrix_nn(self, initialData):
        """ initialData: numPoints, (T, L); 2D array"""
        # Initialize probability matrix holder.
        probabilityMatrix = np.zeros((self.numTempBins, self.numLossBins))

        # Calculate the probability matrix.
        for initialDataPoints in initialData:
            currentUserTemp, currentUserLoss = initialDataPoints

            # Generate 2D gaussian matrix.
            gaussianMatrix = self.createGaussianMap_nn(gausMean=(currentUserLoss, currentUserTemp), gausSTD=self.gaussSTD)
            probabilityMatrix += gaussianMatrix  # Add the gaussian map to the matrix


        # Normalize the probability matrix.
        probabilityMatrix = probabilityMatrix / probabilityMatrix.sum()

        return probabilityMatrix

    # ------------------------ Data Structure Interface ------------------------ #

    @staticmethod
    def getBinIndex(allBins, binValue):
        return min(np.searchsorted(allBins, binValue, side='right'), len(allBins) - 1)

    @staticmethod
    def initializeBins(bounds, binWidth, bufferZone):
        # Calculate the required number of bins
        num_bins = int(np.ceil((bounds[1] - bounds[0] + 2*bufferZone) / binWidth)) + 1

        # Generate the temperature bins
        finalBins = np.linspace(bounds[0]-bufferZone, bounds[1]+bufferZone, num=num_bins)

        return finalBins

    def boundNewTemperature(self, newTemp, bufferZone=0.5):
        return max(self.temperatureBounds[0] + bufferZone, min(self.temperatureBounds[1] - bufferZone, newTemp))

    # ------------------------ Debugging Interface ------------------------ #

    def plotTherapyResults(self, final_map_pack):
        # Unpack the user states.
        benefitFunction, heuristic_map, personalized_map, simulated_map = final_map_pack
        num_steps = len(self.userStatePath)

        # Titles and maps to be plotted.
        titles = ['Projected Benefit Map', 'Heuristic Map', 'Personalized Map', 'Simulated Map']
        maps = [benefitFunction, heuristic_map, personalized_map, simulated_map]

        # Setup subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Color and alpha gradient for the path
        alphas = np.linspace(0.1, 1, num_steps - 1)  # Adjust alpha for segments

        # Plotting each map with the corresponding user state path
        for i, (map_to_plot, title) in enumerate(zip(maps, titles)):
            ax = axs[i % 2, i // 2]
            im = ax.imshow(map_to_plot.T, cmap='coolwarm', extent=[self.temperatureBounds[0], self.temperatureBounds[1], self.lossBounds[0], self.lossBounds[1]], aspect='auto', origin='lower')

            # Plot past user states with fading red line
            for j in range(num_steps - 1):
                ax.plot([self.userStatePath[j][0], self.userStatePath[j + 1][0]],
                        [self.userStatePath[j][1], self.userStatePath[j + 1][1]],
                        color=(0, 0, 0, alphas[j]), linewidth=2)

            ax.scatter(self.userStatePath[-1][0], self.userStatePath[-1][1], color='tab:red', label='Current State', edgecolor='black', s=75, zorder=10)
            ax.set_title(f'{title} (After Iteration {num_steps})')
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Loss')
            ax.legend()

        # Adjust layout to prevent overlap and ensure titles and labels are visible
        fig.tight_layout()
        fig.colorbar(im, ax=axs.ravel().tolist(), label='Probability')
        plt.show()

        print(f"New current state after iteration {num_steps + 1}: Temperature = {self.userStatePath[-1][0]}, Loss = {self.userStatePath[-1][1]}")

    # ------------------------ Basic Protocol plotting ------------------------ #

    import matplotlib.pyplot as plt
    import numpy as np

    def plotTherapyResults_basic(self, simulated_map):
        # Unpack the user states.
        num_steps = len(self.userStatePath)

        # Set up figure
        fig, ax = plt.subplots(figsize=(6, 6))

        # Color and alpha gradient for the path
        alphas = np.linspace(0.1, 1, num_steps - 1)  # Adjust alpha for segments

        # Plotting the simulated map with the corresponding user state path
        im = ax.imshow(simulated_map.T, cmap='coolwarm', extent=[self.temperatureBounds[0], self.temperatureBounds[1], self.lossBounds[0], self.lossBounds[1]], aspect='auto', origin='lower')

        # Plot past user states with fading color
        for j in range(num_steps - 1):
            ax.plot([self.userStatePath[j][0], self.userStatePath[j + 1][0]],
                    [self.userStatePath[j][1], self.userStatePath[j + 1][1]],
                    color=(1, 0, 0, alphas[j]), linewidth=2)  # using red color for the path

        # Highlight the current state
        ax.scatter(self.userStatePath[-1][0], self.userStatePath[-1][1], color='tab:red', label='Current State', edgecolor='black', s=75, zorder=10)

        # Set titles and labels
        ax.set_title(f'Simulated Map (After Iteration {num_steps})')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Loss')
        ax.legend()

        # Add colorbar for the map
        fig.colorbar(im, ax=ax, label='Probability')

        # Show plot
        plt.show()

        # Print current state information
        print(f"New current state after iteration {num_steps + 1}: Temperature = {self.userStatePath[-1][0]}, Loss = {self.userStatePath[-1][1]}")

    # ------------------------ Child Class Contract ------------------------ #

    @abc.abstractmethod
    def updateTherapyState(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")
