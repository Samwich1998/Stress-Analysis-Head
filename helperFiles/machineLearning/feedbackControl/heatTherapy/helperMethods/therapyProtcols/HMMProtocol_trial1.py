from hmmlearn import hmm
import numpy as np
from .generalTherapyProtocol import generalTherapyProtocol
class HMMTherapyProtocol_trial1(generalTherapyProtocol):
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, start_probabilities, transition_matrix, means, variances):
        super().__init__(temperatureBounds, tempBinWidth, simulationParameters)
        self.numStates = self.allNumParameterBins
        self.model = hmm.GaussianHMM(n_components=self.numStates, covariance_type="diag", init_params="")
        self.model.startprob_ = start_probabilities
        self.model.transmat_ = transition_matrix
        self.model.means_ = np.array(means).reshape(-1, 1)
        self.model.covars_ = np.array(variances).reshape(-1, 1)



    def updateTherapyState(self):
        pass

    def update_and_predict(self, observations):
        # Update model with new observation
        self.model.fit(np.array(observations).reshape(-1, 1))
        # Predict the next state
        next_state = self.model.predict(np.array(observations).reshape(-1, 1))[-1]
        print('next state:', next_state)
        return next_state

    def getNextState(self, newUserTemp):
        # get the next state loss based on the new temperature and simulated map
        newUserLoss = self.getSimulatedLoss_HMM(self.userStatePath[-1], newUserTemp)
        self.userStatePath.append((newUserTemp, newUserLoss))
        print('userStatePath:', self.userStatePath)

    def convergence_check(self, recent_predictions, threshold):
        # Check if the last few predictions are the same
        if len(set(recent_predictions[-threshold:])) == 1:
            return True
        return False
