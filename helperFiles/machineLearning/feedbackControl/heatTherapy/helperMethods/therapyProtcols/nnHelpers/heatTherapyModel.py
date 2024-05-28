# General
import torch
from torch import nn
# Import files.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.nnHelpers.modelParameters.sharedModelWeights import sharedModelWeights
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.therapyProtcols.nnHelpers.modelParameters.specificModelWeights import specificModelWeights


class heatTherapyModel(nn.Module):
    def __init__(self, numTemperatures=1, numLosses=3, numTempBins=11, numLossBins=101):
        # General model parameters.
        super().__init__()
        # General model parameters.
        self.numTemperatures = numTemperatures  # The number of predicted temperatures. 1 temperature
        self.numTempBins = numTempBins  # The number of temperature bins.
        self.numLossBins = numLossBins  # The number of loss bins.
        self.numLosses = numLosses  # The number of losses. 3 losses (PA, NA, SA)

        # Calculate the number of input shared features.
        self.numInputTempFeatures = self.numTemperatures + self.numLosses  # The number of input temperature features.
        self.numInputLossFeatures = self.numTemperatures * (self.numTempBins + 1) + self.numLosses  # The number of input loss features.
        # Calculate the number of output shared features.
        # arbitrary
        self.numSharedTempFeatures = self.numTempBins * 2  # The number of shared temperature features. Output dimension of the shared model.
        self.numSharedLossFeatures = self.numLosses * 2  # The number of shared loss features. Output dimension of the shared model.

        # Shared model weights based on the population data.
        self.sharedModelWeights = sharedModelWeights(
            numSharedTempFeatures=self.numSharedTempFeatures,  # The number of shared temperature features.
            numSharedLossFeatures=self.numSharedLossFeatures,  # The number of shared loss features.
            numInputTempFeatures=self.numInputTempFeatures,  # The number of initial temperature features.
            numInputLossFeatures=self.numInputLossFeatures,  # The number of initial loss features.
        )

        # Specific model weights based on the individual data.
        self.specificModelWeights = specificModelWeights(
            numSharedTempFeatures=self.numSharedTempFeatures,  # The number of shared temperature features.
            numSharedLossFeatures=self.numSharedLossFeatures,  # The number of shared loss features.
            numTemperatures=self.numTemperatures,  # The number of predicted temperatures.
            numTempBins=self.numTempBins,  # The number of temperature bins.
            numLossBins=self.numLossBins,  # The number of loss bins.
            numLosses=self.numLosses,  # The number of losses.
        )

    def forward(self, initialPatientStates):
        print('initialPatientStates: ', initialPatientStates)
        print('number of temperature bins: ', self.numTempBins)
        print('number of loss bins: ', self.numLossBins)

        # Add a batch dimension
        if initialPatientStates.dim() == 1:
            initialPatientStates = initialPatientStates.unsqueeze(0)  # Add a batch dimension [[T, PA, NA, SA]].

        # Assert the validity of the input tensor.
        assert len(initialPatientStates.size()) == 2, "Input tensor must have 2 dimensions."
        assert initialPatientStates.dtype == torch.float32, "Tensor must be of type float32"
        assert initialPatientStates.size(1) == 4, "Input tensor must have 4 features."
        # Extract the dimensions of the input data.
        batchSize, numInitialFeatures = initialPatientStates.size()
        # initialPatientStates dimensions: [batchSize, numInputFeatures=4].

        # Predict the next temperatures for the patient.
        finalTemperaturePredictions = self.predictNextTemperature(initialPatientStates)
        # finalTemperaturePrediction dimensions: [numParameters, batchSize, allNumParameterBins].

        # Update the patient's state. RESNET
        compiledTemperaturePredictions = finalTemperaturePredictions.transpose(0, 1).contiguous().view(batchSize, self.numTemperatures * self.numTempBins)
        nextPatientStates = torch.cat(tensors=(compiledTemperaturePredictions, initialPatientStates), dim=1)
        # nextPatientStates dimensions: [batchSize, numInputLossFeatures].
        print('nextPatientStates: ', nextPatientStates)

        # Predict the expected loss of the patient.
        finalLossPredictions = self.predictNextState(nextPatientStates)
        # finalLossPrediction dimensions: [numPredictions, batchSize, numPredictionBins].

        # Assert the dimensions are correct.
        assert finalTemperaturePredictions.size() == (self.numTemperatures, initialPatientStates.size(0), self.numTempBins), f"Incorrect dimensions: {finalTemperaturePredictions.size()}"
        assert finalLossPredictions.size() == (self.numLosses, initialPatientStates.size(0), self.numLossBins), f"Incorrect dimensions: {finalLossPredictions.size()}"

        return finalTemperaturePredictions, finalLossPredictions

    def predictNextTemperature(self, patientStates):
        """ patientStates: The patient's current state. Dimensions: [batchSize, numInputFeatures=4]. """
        # Predict the next temperature of the patient.
        sharedTempFeatures = self.sharedModelWeights.sharedTempFeatureExtraction(patientStates)  # Extract the shared model features.
        finalTemperaturePredictions = self.specificModelWeights.predictNextTemperature(sharedTempFeatures)
        # finalTemperaturePrediction dimensions: [numParameters, batchSize, allNumParameterBins].

        return finalTemperaturePredictions

    def predictNextState(self, patientStates):
        # Predict the expected loss of the patient.
        sharedLossFeatures = self.sharedModelWeights.sharedLossFeatureExtraction(patientStates)  # Extract the shared model features.
        finalLossPredictions = self.specificModelWeights.predictNextLoss(sharedLossFeatures)
        # finalLossPrediction dimensions: [numPredictions, batchSize, numPredictionBins].

        return finalLossPredictions
