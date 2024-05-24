# General
import torch
import time
import sys

# Spyder interface
sys.path.append("./../../../../")
import helperFiles

# Import the necessary libraries.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.basicProtocol import basicProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.aStarProtocol import aStarProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.HMMProtocol import HMMProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.nnProtocol import nnProtocol


class heatTherapyControl:
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, therapyMethod="aStarProtocol", plotResults=False):
        # General parameters.
        self.simulationParameters = simulationParameters
        self.temperatureBounds = temperatureBounds
        self.tempBinWidth = tempBinWidth
        self.plotResults = plotResults

        # Therapy parameters.
        self.therapyProtocol = None
        self.therapyMethod = None

        # Set up the therapy protocols.
        self.setupTherapyProtocols(therapyMethod)

    def setupTherapyProtocols(self, therapyMethod):
        # Change the therapy method.
        self.therapyMethod = therapyMethod
        if self.therapyMethod == "aStarProtocol":
            self.therapyProtocol = aStarProtocol(self.temperatureBounds, self.tempBinWidth, self.simulationParameters, learningRate=2)
        elif self.therapyMethod == "basicProtocol":
            self.therapyProtocol = basicProtocol(self.temperatureBounds, self.tempBinWidth, self.simulationParameters)
        elif self.therapyMethod == "nnProtocol":
            self.therapyProtocol = nnProtocol(self.temperatureBounds, self.tempBinWidth, self.simulationParameters, modelName="2024-04-12 heatTherapyModel", onlineTraining=False)
        elif self.therapyMethod == "HMMProtocol":
            self.therapyProtocol = HMMProtocol(self.temperatureBounds, self.tempBinWidth, self.simulationParameters)
        else:
            raise ValueError("Invalid therapy method provided.")

    def runHMMProtocol(self):
        # train HMM model
        self.therapyProtocol.train()
        # predict the optimal state
        # update the user state path
        self.therapyProtocol.updateTherapyState()
        # print out the optimal temperature and loss
        print('Optimal temperature and loss:', self.therapyProtocol.userStatePath[-1])

    def runTherapyProtocol(self, maxIterations=None):
        # Initialize holder parameters.
        self.therapyProtocol.initializeUserState()

        # --------------------------------for plotting purposes-----------------------------------

        # Delta loss predictions.
        predictedDeltaLossList_PA = []
        predictedDeltaLossList_NA = []
        predictedDeltaLossList_SA = []
        # True delta loss values.
        deltaLossList_PA = []
        deltaLossList_NA = []
        deltaLossList_SA = []

        loss_prediction_loss = []  # Model predicted output loss (backpropagation loss).
        current_user_loss = []  # True output loss (single compiled loss).
        epoch_list = []  # The epoch path.
        loss_bias = []  # Optimal state loss.
        tempList = []  # The temperature path.

        # ----------------------------------------------------------------------------------------

        trainingIteration = 0
        # Until the therapy converges.
        while not self.therapyProtocol.finishedTherapy:
            # Get the next state to move to for the therapy.
            therapyState, allMaps = self.therapyProtocol.updateTherapyState()
            trainingIteration += 1

            # If we are in the nnProtocol.
            if self.therapyMethod == "nnProtocol":
                # Update the mental state (sample or real) based on the new temperature.
                self.therapyProtocol.getNextState(therapyState)

                # Calculate the delta loss for the updated state.
                deltaLossValues = self.therapyProtocol.userFullStatePath[-1][1:] - self.therapyProtocol.userFullStatePath[-2][1:]
                lossPredictionLoss, minimizeLossBias = self.therapyProtocol.lossCalculations.scoreModel(therapyState, deltaLossValues)

                # Backpropogate and update the weights based on the loss.
                self.therapyProtocol.updateWeights(lossPredictionLoss, minimizeLossBias)

                # Prepare for plotting.
                currentUserLoss = self.therapyProtocol.userStatePath_simulated[-1][1]
                flattened_therapyState = therapyState[1:].view(-1).detach().cpu().numpy()
                deltaPA, deltaNA, deltaSA = deltaLossValues
                PA, NA, SA = flattened_therapyState

                # Storing the results for plotting/printing purposes.
                tempList.append(self.therapyProtocol.userFullStatePath[-1][0])
                loss_prediction_loss.append(lossPredictionLoss.item())
                loss_bias.append(minimizeLossBias.item())
                current_user_loss.append(currentUserLoss)
                epoch_list.append(trainingIteration)
                predictedDeltaLossList_PA.append(PA)
                predictedDeltaLossList_NA.append(NA)
                predictedDeltaLossList_SA.append(SA)
                deltaLossList_PA.append(deltaPA)
                deltaLossList_NA.append(deltaNA)
                deltaLossList_SA.append(deltaSA)

                # Print the results for debugging.
                print('trueLossValues: ', self.therapyProtocol.userFullStatePath[-1][1:])
                print('^^^^^^ NA loss', self.therapyProtocol.optimalLoss[1] + therapyState[2])
                print('****** latestLoss: ', lossPredictionLoss.item())
                print('predictedLossValues: ', flattened_therapyState)
                print('temperature', flattened_therapyState[0])
                print('delta Loss Values: ', deltaLossValues)
                print('therapy State: ', therapyState)

            # ----------------------------------------------------------------------------------------------------------------

            if self.plotResults:
                if self.therapyMethod == "nnProtocol":
                    self.therapyProtocol.plotTherapyResults_nn(epoch_list, loss_prediction_loss, loss_bias, current_user_loss)
                    # self.therapyProtocol.plot_loss_comparison(deltaLossValues, therapyState)
                    # self.therapyProtocol.plot_heatmaps(currentPA, currentNA, currentSA, currentPA_pred, currentNA_pred, currentSA_pred, currentTemp)
                    self.therapyProtocol.plot_delta_loss_comparison(epoch_list, deltaLossList_PA, deltaLossList_NA, deltaLossList_SA, predictedDeltaLossList_PA, predictedDeltaLossList_NA, predictedDeltaLossList_SA)
                    self.therapyProtocol.plot_temp(epoch_list, tempList)

                elif self.therapyMethod == "aStarProtocol":
                    self.therapyProtocol.plotTherapyResults(allMaps)
                    print(f"Alpha after iteration: {self.therapyProtocol.percentHeuristic}\n")

                elif self.therapyMethod == "basicProtocol":
                    self.therapyProtocol.plotTherapyResults_basic(allMaps)  # For basic protocol, allMaps is the simulated map (only 1)
                    time.sleep(0.1)

            # Check if the therapy has converged.
            self.therapyProtocol.checkConvergence(maxIterations)


if __name__ == "__main__":
    # User parameters.
    userTherapyMethod = "nnProtocol"  # The therapy algorithm to run. Options: "aStarProtocol", "basicProtocol"
    userTemperatureBounds = (30, 50)  # The temperature bounds for the therapy.
    plotTherapyResults = True  # Whether to plot the results.
    userTempBinWidth = 2  # The temperature bin width for the therapy.

    # Simulation parameters.
    currentSimulationParameters = {
        'heuristicMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'simulatedMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'numSimulationHeuristicSamples': 100,  # The number of simulation samples to generate.
        'numSimulationTrueSamples': 50,  # The number of simulation samples to generate.
        'simulateTherapy': True,  # Whether to simulate the therapy.
    }

    # Initialize the therapy protocol
    therapyProtocol = heatTherapyControl(userTemperatureBounds, userTempBinWidth, currentSimulationParameters, therapyMethod=userTherapyMethod, plotResults=plotTherapyResults)

    # Run the therapy protocol.
    #therapyProtocol.runTherapyProtocol(maxIterations=500)
    therapyProtocol.runHMMProtocol()

    # TODO: lost per epoch  (optimal, predicted, actual loss) (lossPredictionLoss, minimizeLossBias, currentUserLoss) (checked)
    # TODO: change the architecture
    # TODO: save the good copy
    # TODO: figure out the crossentropyerror high issue?
    # TODO: other types of losses for classification (look into them)
    # TODO: add a learning rate scheduler   (checked)
    # TODO: online training
    # TODO: saving and loading models
    # TODO: look into gradient clipping and spectral normalization 

    # loss vs epoch
