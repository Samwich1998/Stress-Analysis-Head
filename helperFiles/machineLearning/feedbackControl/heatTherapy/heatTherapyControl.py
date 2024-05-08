
# General.
import time

# Import the necessary libraries.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.aStarProtocol import aStarProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.basicProtocol import basicProtocol
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
        else:
            raise ValueError("Invalid therapy method provided.")

    def runTherapyProtocol(self, maxIterations=None):
        # Initialize holder parameters.
        self.therapyProtocol.initializeUserState()
        iteration = 0
        loss_prediction_loss = []
        loss_bias = []
        current_user_loss = []
        epoch_list = []
        # Until the therapy converges.
        while not self.therapyProtocol.finishedTherapy:
            # Get the next states for the therapy.
            therapyState, allMaps = self.therapyProtocol.updateTherapyState()
            self.therapyProtocol.getNextState(therapyState)

            if self.therapyMethod == "nnProtocol":
                # Calculate the final loss.
                trueLossValues = self.therapyProtocol.userFullStatePath[-1][1:] # from simulation data
                lossPredictionLoss, minimizeLossBias = self.therapyProtocol.lossCalculations.scoreModel(therapyState, trueLossValues) #losspredictionloss is from the model
                self.therapyProtocol.updateWeights(lossPredictionLoss, minimizeLossBias)
                currentUserLoss = self.therapyProtocol.userStatePath[-1][1]
                print('lossPredictionLoss: ', lossPredictionLoss)
                print('minimizeLossBias: ', minimizeLossBias)
                print('currentUserLoss: ', currentUserLoss)
                loss_prediction_loss.append(lossPredictionLoss.item())
                loss_bias.append(minimizeLossBias.item())
                current_user_loss.append(currentUserLoss)
                iteration += 1
                epoch_list.append(iteration)
                if self.plotResults:
                    self.therapyProtocol.plotTherapyResults_nn(epoch_list, loss_prediction_loss, loss_bias, current_user_loss)

            if self.plotResults:
                if self.therapyMethod == "aStarProtocol":
                    self.therapyProtocol.plotTherapyResults(allMaps)
                    print(f"Alpha after iteration: {self.therapyProtocol.percentHeuristic}\n")
                elif self.therapyMethod == "basicProtocol":
                    self.therapyProtocol.plotTherapyResults_basic(allMaps) # For basic protocol, allMaps is the simulated map (only 1)
                    time.sleep(0.1)
            # Check if the therapy has converged.
            self.therapyProtocol.checkConvergence(maxIterations)


if __name__ == "__main__":
    # User parameters.
    userTherapyMethod = "basicProtocol"  # The therapy algorithm to run. Options: "aStarProtocol", "basicProtocol"
    userTemperatureBounds = (30, 50)  # The temperature bounds for the therapy.
    plotTherapyResults = True  # Whether to plot the results.
    userTempBinWidth = 2  # The temperature bin width for the therapy.

    # Simulation parameters.
    currentSimulationParameters = {
        'heuristicMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'simulatedMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'numSimulationHeuristicSamples': 10,  # The number of simulation samples to generate.
        'numSimulationTrueSamples': 50,  # The number of simulation samples to generate.
        'simulateTherapy': False,  # Whether to simulate the therapy.
    }

    # Initialize the therapy protocol
    therapyProtocol = heatTherapyControl(userTemperatureBounds, userTempBinWidth, currentSimulationParameters, therapyMethod=userTherapyMethod, plotResults=plotTherapyResults)

    # Run the therapy protocol.
    therapyProtocol.runTherapyProtocol(maxIterations=100)

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
