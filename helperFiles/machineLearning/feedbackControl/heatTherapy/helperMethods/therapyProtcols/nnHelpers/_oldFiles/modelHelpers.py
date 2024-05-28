# General
from torch import nn, optim

# Import files.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.nnHelpers.heatTherapyModel import heatTherapyModel


class modelHelpers:
    def __init__(self, modelName, numTempBins, numLossBins, onlineTraining=False):
        # General parameters.
        self.onlineTraining = onlineTraining  # Whether to train the model live.
        self.numTempBins = numTempBins  # The number of temperature bins.
        self.numLossBins = numLossBins  # The number of loss bins.
        self.modelName = modelName      # The model's unique identifier.
        self.numTemperatures = 1        # The number of temperatures to predict. # will be 2D for music therapy
        self.numLosses = 3              # The number of losses to predict. (PA, NA, SA)

        # Model parameters.
        self.lossFunction = None    # The loss function for the model.
        self.optimizer = None       # The optimizer for the model.

    # ------------------------ Machine Learning Setup ------------------------ #

    def setupModel(self):
        # Define the optimizer.
        self.optimizer = optim.AdamW([
            # Specify the model parameters for the signal mapping.
            {'params': self.model.sharedModelWeights.parameters(), 'weight_decay': 1E-5, 'lr': 1E-3},

            # Specify the model parameters for the feature extraction.
            {'params': self.model.specificModelWeights.parameters(), 'weight_decay': 1E-5, 'lr': 1E-3},
        ])

        # TODO: this will porobably be MSE
        self.lossFunction = nn.CrossEntropyLoss(weight=None, reduction='none', label_smoothing=0.0)
        #self.lossFunction = nn.MSELoss()

    # ------------------------ Machine Learning ------------------------ #

    def compileLosses(self, finalTemperaturePredictions, finalLossPredictions, targetTempInds=None, targetLossInds=None):
        temperaturePredictionLoss = 0
        if targetTempInds is not None:
            # Add a bias for the model to predict the next temperature.
            for tempInd in range(finalTemperaturePredictions.size(0)):
                temperaturePredictionLoss = temperaturePredictionLoss + self.lossFunction(finalTemperaturePredictions[tempInd], targetTempInds[:, tempInd]).mean()

        lossPredictionLoss = 0
        if targetLossInds is not None:
            # Add a bias for the model to predict the next loss.
            for lossInd in range(finalLossPredictions.size(0)):
                lossPredictionLoss = lossPredictionLoss + self.lossFunction(finalLossPredictions[lossInd], targetLossInds[:, lossInd]).mean()

        minimizeLossBias = 0
        # Add a bias for the model to minimize the loss at the next temperature.
        for lossInd in range(finalLossPredictions.size(0)):
            minimizeLossBias = minimizeLossBias + self.lossFunction(finalLossPredictions[lossInd], self.optimalFinalLoss[lossInd]).mean()

        return temperaturePredictionLoss, lossPredictionLoss, minimizeLossBias

    def trainOneEpoch(self, initialPatientStates, targetTempInds):
        """
        inputData: The input data for the model. Dimensions: [batchSize, numInputFeatures=4].
        targetTempInds: The index of the output data. Dimensions: [batchSize, numParameters].
        """
        # Set up the model for training.
        self.optimizer.zero_grad()
        self.setupTraining()

        # Forward pass through the model.
        finalTemperaturePredictions, finalLossPredictions = self.model(initialPatientStates)     # Predict the probability of which temperature bin we should aim for.
        # finalTemperaturePrediction dimensions: [numParameters, batchSize, allNumParameterBins].
        # finalLossPrediction dimensions: [numPredictions, batchSize, numPredictionBins].

        # Loss calculation.
        temperaturePredictionLoss, lossPredictionLoss, minimizeLossBias = self.compileLosses(finalTemperaturePredictions, finalLossPredictions, targetTempInds)
        finalLoss = temperaturePredictionLoss + lossPredictionLoss + minimizeLossBias
        # loss dimensions: [batchSize].

        # Backward pass.
        finalLoss.backward()    # Calculate the gradients.
        self.optimizer.step()   # Update the weights.

    def modelPrediction(self, initialPatientStates):
        # Set the model to evaluation mode.
        self.setupTrainingFlags(self.model, trainingFlag=False)

        # Forward pass.
        finalTemperaturePredictions, finalLossPredictions = self.model(initialPatientStates)

        return finalTemperaturePredictions, finalLossPredictions

    # ------------------------ Training/Testing Switching ------------------------ #

    def setupTraining(self):
        # Label the model we are training.
        if self.onlineTraining:
            self.setupTrainingFlags(self.model.specificModelWeights, trainingFlag=True)
            self.setupTrainingFlags(self.model.sharedModelWeights, trainingFlag=False)
        else:
            self.setupTrainingFlags(self.model.specificModelWeights, trainingFlag=True)
            self.setupTrainingFlags(self.model.sharedModelWeights, trainingFlag=True)

    def setupTrainingFlags(self, model, trainingFlag):
        # Change the training/testing modes
        self.changeTrainingModes(model, trainingFlag)

        # For each model parameter.
        for param in model.parameters():
            # Change the gradient tracking status
            param.requires_grad = trainingFlag

    @staticmethod
    def changeTrainingModes(model, trainingFlag):
        # Set the model to training mode.
        if trainingFlag:
            model.train()
            # Or evaluation mode.
        else:
            model.eval()
