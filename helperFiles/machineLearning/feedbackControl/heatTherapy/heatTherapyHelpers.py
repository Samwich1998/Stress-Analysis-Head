# Import the necessary libraries.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.aStarProtocol import aStarTherapyProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.basicProtocol import basicTherapyProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.hmmProtocol import hmmTherapyProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.nnProtocol import nnTherapyProtocol


class heatTherapyHelpers:
    def __init__(self, userName, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod="aStarTherapyProtocol", plotResults=False):
        # General parameters.
        self.unNormalizedParameterBinWidths = unNormalizedParameterBinWidths  # The bin widths for the parameter bounds.
        self.initialParameterBounds = initialParameterBounds  # The parameter bounds for the therapy.
        self.simulationParameters = simulationParameters  # The simulation parameters for the therapy.
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
        if self.therapyMethod == "aStarTherapyProtocol":
            self.therapyProtocol = aStarTherapyProtocol(self.initialParameterBounds, self.simulationParameters, learningRate=2)
        elif self.therapyMethod == "basicTherapyProtocol":
            self.therapyProtocol = basicTherapyProtocol(self.initialParameterBounds, self.simulationParameters)
        elif self.therapyMethod == "nnTherapyProtocol":
            self.therapyProtocol = nnTherapyProtocol(self.initialParameterBounds, self.simulationParameters, modelName="2024-04-12 heatTherapyModel", onlineTraining=False)
        elif self.therapyMethod == "hmmTherapyProtocol":
            self.therapyProtocol = hmmTherapyProtocol(self.initialParameterBounds, self.unNormalizedParameterBinWidths, self.simulationParameters, therapyMethod="HeatingPad")
        else:
            raise ValueError("Invalid therapy method provided.")
