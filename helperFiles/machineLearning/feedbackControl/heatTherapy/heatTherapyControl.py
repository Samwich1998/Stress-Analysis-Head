# General.
import torch
import sys

# Spyder interface.
sys.path.append("./../../../../")
import helperFiles

# Import the necessary libraries.
from helperFiles.machineLearning.feedbackControl.heatTherapy.heatTherapyHelpers import heatTherapyHelpers


class heatTherapyControl(heatTherapyHelpers):
    def __init__(self, userName, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod, plotResults=False):
        super().__init__(userName=userName, initialParameterBounds=initialParameterBounds, unNormalizedParameterBinWidths=unNormalizedParameterBinWidths,
                         simulationParameters=simulationParameters, therapyMethod=therapyMethod, plotResults=plotResults)

    def runTherapyProtocol(self, maxIterations=None):
        # Initialize holder parameters such as the user maps.
        self.therapyProtocol.initializeUserState(userName=self.userName)

        # Until the therapy converges.
        while not self.therapyProtocol.finishedTherapy:
            # Get the next states for the therapy.
            therapyState, allMaps = self.therapyProtocol.updateTherapyState()

            print(f"Therapy state: {therapyState}\n")
            if self.therapyMethod == "nnTherapyProtocol":
                if self.therapyProtocol.onlineTraining:
                    print('------Online training started-------')
                    # Calculate the final loss.
                    self.therapyProtocol.getNextState(therapyState)
                    trueLossValues = self.therapyProtocol.userFullStatePath[-1][1:]
                    if iteration > 2:
                        deltaLossValues = [true - prev for true, prev in zip(trueLossValues, self.therapyProtocol.userFullStatePath[-2][1:])]  # self.therapyProtocol.optimalNormalizedState
                    else:
                        deltaLossValues = [true - optimal for true, optimal in zip(trueLossValues, self.therapyProtocol.optimalNormalizedState)]
                    lossPredictionLoss, minimizeLossBias = self.therapyProtocol.lossCalculations.scoreModel(therapyState, deltaLossValues)  # losspredictionloss is from the model
                    self.latestLoss = lossPredictionLoss.item()
                    self.therapyProtocol.updateWeights(lossPredictionLoss, minimizeLossBias)
                    currentUserLoss = self.therapyProtocol.userStatePath[-1][1]
                    loss_prediction_loss.append(lossPredictionLoss.item())
                    loss_bias.append(minimizeLossBias.item())
                    current_user_loss.append(currentUserLoss)
                    iteration += 1
                    epoch_list.append(iteration)
                    if self.plotResults:
                        self.therapyProtocol.plotTherapyResults_nn(epoch_list, loss_prediction_loss)

                elif not self.therapyProtocol.onlineTraining:
                    print('------simulation (offline) training started-------')
                    trueLossValues = self.therapyProtocol.userFullStatePath[-1][1:]
                    if iteration > 2:
                        deltaLossValues = [true - prev for true, prev in zip(trueLossValues, self.therapyProtocol.userFullStatePath[-2][1:])]  # self.therapyProtocol.optimalNormalizedState
                    else:
                        deltaLossValues = [true - optimal for true, optimal in zip(trueLossValues, self.therapyProtocol.optimalNormalizedState)]
                    deltaLossValues = torch.tensor(deltaLossValues, dtype=torch.float32)
                    lossPredictionLoss, minimizeLossBias = self.therapyProtocol.lossCalculations.scoreModel(therapyState, deltaLossValues)  #losspredictionloss is from the model
                    # print the datatype of lossPredictionLoss
                    self.therapyProtocol.getNextState(therapyState)
                    self.latestLoss = lossPredictionLoss.item()
                    print('****** latestLoss: ', self.latestLoss)
                    print('therapy State: ', therapyState)
                    print('^^^^^^ NA loss', self.therapyProtocol.optimalNormalizedState[1] + therapyState[2])
                    print('iteration: ', iteration)

                    self.therapyProtocol.updateWeights(lossPredictionLoss, minimizeLossBias)
                    currentUserLoss = self.therapyProtocol.userStatePath_simulated[-1][1]

                    # -------------------------------------- data arrangement for plotting purposes-----------------------------------
                    # predicted map
                    # predicted loss distributions
                    # currentPA_pred = torch.softmax(therapyState[1][0].squeeze(), dim=-1).detach().numpy()
                    # currentNA_pred = torch.softmax(therapyState[1][1].squeeze(), dim=-1).detach().numpy()
                    # currentSA_pred = torch.softmax(therapyState[1][2].squeeze(), dim=-1).detach().numpy()

                    # Extract the temperature and three loss values from the userFullStatePathDistribution
                    currentTemp = self.therapyProtocol.userFullStatePath[-1][0]
                    # currentPA = self.therapyProtocol.userFullStatePath[-1][1]
                    # currentNA = self.therapyProtocol.userFullStatePath[-1][2]
                    # currentSA = self.therapyProtocol.userFullStatePath[-1][3]

                    loss_prediction_loss.append(lossPredictionLoss.item())
                    loss_bias.append(minimizeLossBias.item())
                    current_user_loss.append(currentUserLoss)
                    iteration += 1
                    epoch_list.append(iteration)

                    flattened_therapyState = therapyState[1:].view(-1).tolist()
                    deltaPA, deltaNA, deltaSA = deltaLossValues
                    print('temperature', flattened_therapyState[0])
                    print('delta Loss Values: ', deltaLossValues)
                    print('trueLossValues: ', trueLossValues)
                    print('predictedLossValues: ', flattened_therapyState)
                    PA, NA, SA = flattened_therapyState
                    deltaLossList_PA.append(deltaPA)
                    deltaLossList_NA.append(deltaNA)
                    deltaLossList_SA.append(deltaSA)
                    predictedDeltaLossList_PA.append(PA)
                    predictedDeltaLossList_NA.append(NA)
                    predictedDeltaLossList_SA.append(SA)
                    tempList.append(currentTemp)

                    # ----------------------------------------------------------------------------------------------------------------

                    if self.plotResults:
                        self.therapyProtocol.plotTherapyResults_nn(epoch_list, loss_prediction_loss)
                        # self.therapyProtocol.plot_loss_comparison(deltaLossValues, therapyState)
                        # self.therapyProtocol.plot_heatmaps(currentPA, currentNA, currentSA, currentPA_pred, currentNA_pred, currentSA_pred, currentTemp)
                        self.therapyProtocol.plot_delta_loss_comparison(epoch_list, deltaLossList_PA, deltaLossList_NA, deltaLossList_SA, predictedDeltaLossList_PA, predictedDeltaLossList_NA, predictedDeltaLossList_SA)
                        self.therapyProtocol.plot_temp(epoch_list, tempList)
            if self.plotResults:
                if self.therapyMethod == "aStarTherapyProtocol":
                    self.therapyProtocol.plotTherapyResults(allMaps)
                    print(f"Alpha after iteration: {self.therapyProtocol.percentHeuristic}\n")
                elif self.therapyMethod == "basicTherapyProtocol":
                    self.therapyProtocol.plotTherapyResults_basic(allMaps)  # For basic protocol, allMaps is the simulated map (only 1)
            # Check if the therapy has converged.
            self.therapyProtocol.checkConvergence(maxIterations)


if __name__ == "__main__":
    # User parameters.
    userTherapyMethod = "hmmTherapyProtocol"  # The therapy algorithm to run. Options: "aStarTherapyProtocol", "basicTherapyProtocol", "nnTherapyProtocol", "hmmTherapyProtocol"
    testingUserName = "Squirtle"  # The username for the therapy.
    temperatureBounds = (35, 50)  # The temperature bounds for the therapy.
    temperatureBinWidth = 2.5  # The temperature bounds for the therapy.
    plotTherapyResults = True  # Whether to plot the results.

    # Simulation parameters.
    currentSimulationParameters = {
        'heuristicMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'simulatedMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'numSimulationHeuristicSamples': 100,  # The number of simulation samples to generate.
        'numSimulationTrueSamples': 50,  # The number of simulation samples to generate.
        'simulateTherapy': True,  # Whether to simulate the therapy.
    }

    # Initialize the therapy protocol
    therapyProtocol = heatTherapyControl(testingUserName, temperatureBounds, temperatureBinWidth, currentSimulationParameters, therapyMethod=userTherapyMethod, plotResults=plotTherapyResults)

    # Run the therapy protocol.
    therapyProtocol.runTherapyProtocol(maxIterations=2000)
