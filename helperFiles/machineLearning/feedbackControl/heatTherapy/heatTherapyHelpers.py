# Import the necessary libraries.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.aStarProtocol import aStarProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.basicProtocol import basicProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.HMMProtocol import HMMProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.nnProtocol import nnProtocol


class heatTherapyHelpers:
    def __init__(self, userName, temperatureBounds, simulationParameters, therapyMethod="aStarProtocol", plotResults=False):
        # General parameters.
        self.simulationParameters = simulationParameters  # The simulation parameters for the therapy.
        self.temperatureBounds = temperatureBounds  # The temperature bounds for the therapy.
        self.plotResults = plotResults  # Whether to plot the results.
        self.userName = userName  # The username for the therapy.

        # Therapy parameters.
        self.therapyProtocol = None
        self.therapyMethod = None

        # Set up the therapy protocols.
        self.setupTherapyProtocols(therapyMethod)

    def setUserName(self, userName):
        self.userName = userName

    def setupTherapyProtocols(self, therapyMethod):
        # Change the therapy method.
        self.therapyMethod = therapyMethod
        if self.therapyMethod == "aStarProtocol":
            self.therapyProtocol = aStarProtocol(self.temperatureBounds, self.simulationParameters, learningRate=2)
        elif self.therapyMethod == "basicProtocol":
            self.therapyProtocol = basicProtocol(self.temperatureBounds, self.simulationParameters)
        elif self.therapyMethod == "nnProtocol":
            self.therapyProtocol = nnProtocol(self.temperatureBounds, self.simulationParameters, modelName="2024-04-12 heatTherapyModel", onlineTraining=False)
        elif self.therapyMethod == "HMMProtocol":
            self.therapyProtocol = HMMProtocol(self.temperatureBounds, self.simulationParameters)
        else:
            raise ValueError("Invalid therapy method provided.")
