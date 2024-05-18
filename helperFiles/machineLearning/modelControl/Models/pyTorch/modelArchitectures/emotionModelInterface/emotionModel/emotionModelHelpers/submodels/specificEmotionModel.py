# PyTorch
import torch

# Import files for machine learning
from .modelComponents.complexEmotionPrediction import complexEmotionPrediction
from .modelComponents.subjectInterpretation import subjectInterpretation
from .modelComponents.activityRecognition import activityRecognition
from ....._globalPytorchModel import globalModel


class specificEmotionModel(globalModel):
    def __init__(self, numCommonSignals, numActivityFeatures, activityNames, numBasicEmotions, numInterpreterHeads, numSubjects, emotionNames, featureNames):
        super(specificEmotionModel, self).__init__()
        # General model parameters.
        self.numActivityFeatures = numActivityFeatures  # The number of common activity features to extract.
        self.numInterpreterHeads = numInterpreterHeads  # The number of ways to interpret a set of physiological signals.
        self.numCommonSignals = numCommonSignals  # The number of features from considering all the signals.
        self.numBasicEmotions = numBasicEmotions  # The number of basic emotions (basis states of emotions).
        self.numActivities = len(activityNames)  # The number of activities to predict.
        self.numEmotions = len(emotionNames)  # The number of emotions to predict.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.

        # ------------------- Human Activity Recognition ------------------- #  

        # Predict the activity (background context) the subject is experiencing.
        self.classifyHumanActivity = activityRecognition(
            numActivityFeatures=self.numActivityFeatures,
            numActivities=self.numActivities,
        )

        # ------------------ Basic Emotion Classification ------------------ #  

        # Predict which type of thinker the user is.
        self.predictUserEmotions = subjectInterpretation(
            numInterpreterHeads=self.numInterpreterHeads,
            numBasicEmotions=self.numBasicEmotions,
            numActivities=self.numActivities,
            numSubjects=self.numSubjects,
        )

        # --------------------- Emotion Classification --------------------- #

        # Predict which type of thinker the user is.
        self.predictComplexEmotions = complexEmotionPrediction(
            numCommonFeatures=self.numCommonSignals,
            numBasicEmotions=self.numBasicEmotions,
            numEmotions=self.numEmotions,
        )

        # ------------------------------------------------------------------ # 

        # Reset the model
        self.resetModel()

        # Initialize loss holders.
        self.trainingLosses_signalReconstruction = None
        self.testingLosses_signalReconstruction = None
        self.trainingLosses_timeAnalysis = None
        self.testingLosses_timeAnalysis = None
        self.trainingLosses_mappedMean = None
        self.testingLosses_mappedMean = None
        self.trainingLosses_mappedSTD = None
        self.testingLosses_mappedSTD = None

    def forward(self):
        return None

    def resetModel(self):
        # Autoencoder manifold reconstructed loss holders.
        self.trainingLosses_signalReconstruction = []  # List of manifold reconstruction (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_signalReconstruction = []  # List of manifold reconstruction (autoencoder) testing losses. Dim: numEpochs

        # Autoencoder manifold mean loss holders.
        self.trainingLosses_manifoldMean = []  # List of manifold mean (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_manifoldMean = []  # List of manifold mean (autoencoder) testing losses. Dim: numEpochs
        # Autoencoder manifold standard deviation loss holders.
        self.trainingLosses_manifoldSTD = []  # List of manifold standard deviation (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_manifoldSTD = []  # List of manifold standard deviation (autoencoder) testing losses. Dim: numEpochs

        # ------------------------------------------------------------------ #  

    # DEPRECATED
    def shapInterface(self, reshapedSignalFeatures):
        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, numFeatures = reshapedSignalFeatures.shape
        reshapedSignalFeatures = torch.tensor(reshapedSignalFeatures.tolist())
        assert numFeatures == self.numSignals * self.numSignalFeatures, f"{numFeatures} {self.numSignals} {self.numSignalFeatures}"

        # Reshape the inputs to integrate into the model's expected format.
        signalFeatures = reshapedSignalFeatures.view((batchSize, self.numSignals, self.numSignalFeatures))

        # predict the activities.
        activityDistribution = self.forward(signalFeatures, predictActivity=True, allSignalFeatures=True)

        return activityDistribution.detach().numpy()
