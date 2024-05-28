from hmmlearn import hmm
import numpy as np

# Import files.
from .generalTherapyProtocol import generalTherapyProtocol


class HMMTherapyProtocol(generalTherapyProtocol):
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters):
        super().__init__(initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters)
        self.numStates = self.allNumParameterBins # number of states in the HMM

        # HMM intialization:
        # transition matrix initialization: probability of transitioning from one state to another
        self.A = np.random.rand(self.numStates, self.numStates) # dimension: [numStates, numStates]
        self.A /= self.A.sum(axis=1, keepdims=True) # normalize the transition matrix

        # emission matrix: 
        self.B = self.simulationProtocols.simulatedMap.copy() # dimension: [numStates, numPredictionBins] // TODO: rethink about normalization, simulated map is normalzied across the whole
        #self.B = np.random.rand(self.numStates, self.numStates)
        self.B /= self.B.sum(axis=1, keepdims=True) # normalize the emission matrix

        # initial state probabilities
        self.pi = np.random.rand(self.numStates)
        self.pi /= self.pi.sum()

        # simulation training HMM
        self.sequence_len = 50
        self.sequence_num = 100


    # sequence generation for HMM training purpose
    def generate_sequence(self):
        length = self.sequence_len
        observations = np.zeros(length, dtype=int)
        states = np.zeros(length, dtype=int) # tracking the hidden states

        # Initial state
        states[0] = np.random.choice(self.numStates, p=self.pi)
        observations[0] = np.random.choice(self.numPredictionBins, p=self.B[states[0], :])

        for t in range(1, length):
            states[t] = np.random.choice(self.numStates, p=self.A[states[t-1], :])
            observations[t] = np.random.choice(self.numPredictionBins, p=self.B[states[t], :])

        return observations, states

    def generate_multiple_sequences(self):
        num_sequence = self.sequence_num
        all_observations = []
        all_states = []
        for _ in range(num_sequence):
            observations, states = self.generate_sequence()
            all_observations.append(observations)
            all_states.append(states)
        return all_observations, all_states

    # forward backward algorithm
    def forward(self, observations, normalize=False):
        T = len(observations) # observation length index by time T
        alpha = np.zeros((T, self.numStates)) # dim: [T, numStates]
        alpha[0, :] = self.pi * self.B[:, observations[0]] # dim [numStates]

        if normalize:
            alpha[0, :] /= np.sum(alpha[0, :])

        # recursive: from t = 1 to observation length (T)
        for t in range(1, T):
            for j in range(self.numStates):
                alpha[t, j] = self.B[j, observations[t]] * np.sum(alpha[t-1, :] * self.A[:, j])

        return alpha

    def backward(self, observations, normalize=False):
        T = len(observations)
        beta = np.zeros((T, self.numStates))

        # initialize the base cases (t = T-1)
        beta[T-1, :] = 1

        if normalize:
            beta[T-1, :] /= np.sum(beta[T-1, :])

        # recursive: from t = T-2 to 0
        for t in range(T-2, -1, -1):
            for i in range(self.numStates):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1, :])

        return beta

    # parameter update with baum welch algorithm
    def baum_welch(self, observations, max_iterations=10, tolerance=1e-6):
        prev_log_likelihood = -np.inf # lower limit of log prob
        T = len(observations)

        for iteration in range(max_iterations):
            alpha = self.forward(observations, normalize=True)
            beta = self.backward(observations, normalize=True)

            # Expectation step (E-step)
            gamma = np.zeros((T, self.numStates))
            xi = np.zeros((T - 1, self.numStates, self.numStates)) # T-1 = # of transitions. joint probabilty of transition from each state i to state j at time t+1

            for t in range(T):
                gamma[t, :] = alpha[t, :] * beta[t, :] / np.sum(alpha[t, :] * beta[t, :])

                if t < T - 1:
                    for i in range(self.numStates):
                        for j in range(self.numStates):
                            xi[t, i, j] = (alpha[t, i] * self.A[i, j] * self.B[j, observations[t + 1]] * beta[t + 1, j])
                    xi[t, :, :] /= np.sum(xi[t, :, :])

            # Maximizationn step (M-step)
            self.pi = gamma[0, :]
            for i in range(self.numStates):
                for j in range(self.numStates):
                    self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

                for k in range(self.numPredictionBins):
                    mask = (observations == k) # array of boolean statement of observations == k
                    self.B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

            # Check for convergence
            log_likelihood = np.sum(np.log(np.sum(alpha, axis=1)))
            if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < tolerance:
                break
            prev_log_likelihood = log_likelihood
            #print out the iteration, transition matrix, emission matrix, initial states and log likelihood
            print(f"Iteration {iteration}:")
            print(f"Transition matrix: {self.A}")
            print(f"Emission matrix: {self.B}")
            print(f"Initial states: {self.pi}")
            print(f"Log likelihood: {log_likelihood}")

    def train(self, max_iterations=100, tolerance=1e-6):
        all_observations, all_states = self.generate_multiple_sequences()
        print('observations:', all_observations)
        print('states:', all_states)
        for observations in all_observations:
            self.baum_welch(observations, max_iterations, tolerance)


    def predict_next_temperature(self, currentState):
        current_temperature, current_loss = currentState
        current_observation = np.array([current_loss])

        # Estimate current state probabilities
        alpha = self.forward(current_observation)
        current_state_probs = alpha[-1, :]

        # Predict future state probabilities
        next_state_probs = np.dot(current_state_probs, self.A)

        # expected loss for each possible next temperature
        expected_losses = np.zeros(self.allNumParameterBins)
        for temp in range(self.allNumParameterBins):
            for state in range(self.numStates):
                expected_losses[temp] += next_state_probs[state] * self.B[state, temp]

        # get the temperature with the lowest expected loss
        optimal_temperature = np.argmin(expected_losses)

        return optimal_temperature, min(expected_losses)

    def updateTherapyState(self):
        optimal_temperature, loss = self.predict_next_temperature(self.userStatePath[-1])
        self.userStatePath.append((optimal_temperature, loss))


    def initializeUserState_HMM(self):
        # Get the user information.
        timePoint, userState = self.getCurrentState()  # userState: (T, PA, NA, SA)
        tempIndex = self.getBinIndex(self.temp_bins, userState[0])
        # Track the user state and time delay.
        self.userStatePath.append(self.compileLossStates(np.asarray([userState]))[0])  # userStatePath: (numEpochs, 2=(T, Loss))
        self.temperatureTimepoints.append((timePoint, tempIndex))  # temperatureTimepoints: (numEpochs, 2=(time, tempIndex))
        self.userFullStatePath.append(userState)  # userFullStatePath: (numEpochs, 4=(T, PA, NA, SA))

    def initializeStartingProbabilityMatrix_HMM(self):
        initial_probs = np.full(self.allNumParameterBins, 1 / self.allNumParameterBins)
        return initial_probs

    def initializeTransitionMatrix_HMM(self):
        transition_matrix = np.full((self.allNumParameterBins, self.allNumParameterBins), 1 / self.allNumParameterBins)
        return transition_matrix

    def initializeMeans_HMM(self):
        # initialize means based on each temperature bin associated loss values from simulated map (get the mean of the loss values in this temperature bin)
        means = np.zeros(self.allNumParameterBins)
        for i in range(self.allNumParameterBins):
            means[i] = np.mean(self.simulationProtocols.simulatedMap[i])
        return means

    def initializeVariances_HMM(self):
        # initialize variance based on each tempearture bin associated loss values from simulated map (get the variance of the loss values in this temperature bin)
        variances = np.zeros(self.allNumParameterBins)
        for i in range(self.allNumParameterBins):
            variances[i] = np.var(self.simulationProtocols.simulatedMap[i])
        return variances

    def randomSampleTempLossPairs(self, numSamples):
        # randomly sample temperature-loss pairs from the simulated map
        tempLossPairs = []
        loss = []
        for i in range(numSamples):
            tempIndex = np.random.choice(self.allNumParameterBins)
            lossIndex = np.random.choice(self.numPredictionBins)
            tempLossPairs.append([self.temp_bins[tempIndex], self.loss_bins[lossIndex]])
            loss.append(self.loss_bins[lossIndex])
        return tempLossPairs, loss


    def getSimulatedLoss_HMM(self, currentUserState, newUserTemp):
        # Unpack the current user state.
        currentUserTemp, currentUserLoss = currentUserState

        # Calculate the bin indices for the current and new user states.
        currentTempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        currentLossIndex = self.getBinIndex(self.loss_bins, currentUserLoss)
        newTempBinIndex = self.getBinIndex(self.temp_bins, newUserTemp)

        # Simulate a new user loss.
        newUserLoss = self.simulationProtocols.sampleNewLoss_HMM(currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex)

        return newUserLoss

    def sampleNewLoss_HMM(self, currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, bufferZone=0.01):
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
            newLossBinIndex = np.random.choice(a=len(newLossProbabilities), p=newLossProbabilities)
            newUserLoss = self.loss_bins[newLossBinIndex]
        else:
            newUserLoss = currentUserLoss + np.random.normal(loc=0, scale=0.01)

        return max(self.loss_bins[0] + bufferZone, min(self.loss_bins[-1] - bufferZone, newUserLoss))