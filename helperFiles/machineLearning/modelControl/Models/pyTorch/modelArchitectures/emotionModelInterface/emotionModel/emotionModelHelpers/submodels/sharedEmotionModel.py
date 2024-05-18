# PyTorch
import torch

# Import files for machine learning
from .modelComponents.basicEmotionPredictions import basicEmotionPredictions
from .modelComponents.commonActivityAnalysis import commonActivityAnalysis
from .modelComponents.featureExtraction import featureExtraction
from ....._globalPytorchModel import globalModel


class sharedEmotionModel(globalModel):
    def __init__(self, compressedLength, numEncodedSignals, numCommonSignals, numActivityFeatures, numInterpreterHeads, numBasicEmotions, emotionLength):
        super(sharedEmotionModel, self).__init__()
        # General model parameters.
        self.numInterpreterHeads = numInterpreterHeads  # The number of ways to interpret a set of physiological signals.
        self.numActivityFeatures = numActivityFeatures  # The number of common activity features to extract.
        self.numEncodedSignals = numEncodedSignals  # The number of signals in each batch. Each signal is a combination of multiple.
        self.numCommonSignals = numCommonSignals  # The number of features from considering all the signals.
        self.numBasicEmotions = numBasicEmotions  # The number of basic emotions (basis states of emotions).
        self.compressedLength = compressedLength  # The final length of each signal after projection.

        # Last layer activation.
        self.lastActivityLayer = None  # A string representing the last layer for activity prediction. Option: 'softmax', 'logsoftmax', or None.
        self.lastEmotionLayer = None  # A string representing the last layer for emotion prediction. Option: 'softmax', 'logsoftmax', or None.

        self.extractCommonFeatures = featureExtraction(
            numEncodedSignals=self.numEncodedSignals,
            compressedLength=self.compressedLength,
        )

        # Synthesize the common features for activity recognition.
        self.extractActivityFeatures = commonActivityAnalysis(
            numCommonFeatures=self.numCommonSignals,
            numActivityFeatures=self.numActivityFeatures,
        )

        # Initialize basic emotion predictions.
        self.predictBasicEmotions = basicEmotionPredictions(
            numInterpreterHeads=self.numInterpreterHeads,
            numCommonFeatures=self.numCommonSignals,
            numBasicEmotions=self.numBasicEmotions,
            emotionLength=self.compressedLength,
        )

    def forward(self, mappedSignalData, subjectIdentifiers, specificEmotionModel, trainingFlag=False):
        """ The shape of manifoldData: (batchSize, numEncodedSignals, compressedLength) """

        # ----------------------- Data Preprocessing ----------------------- #  

        # Extract the incoming data's dimension.
        batchSize, numEncodedSignals, compressedLength = mappedSignalData.size()
        subjectInds = subjectIdentifiers[:, 0]  # The first subject identifier is the subject index. subjectInds dimension: batchSize

        # Assert the integrity of the incoming data.
        assert numEncodedSignals == self.numEncodedSignals, f"The model was expecting {self.numEncodedSignals} signals, but received {numEncodedSignals}"
        assert compressedLength == self.compressedLength, f"The signals have length {compressedLength}, but the model expected {self.compressedLength} points."

        # ----------------------- Feature Extraction ----------------------- #  

        # Extract features synthesizing all the signal information.
        featureData = self.extractCommonFeatures(mappedSignalData)
        # featureData dimension: batchSize, self.numCommonSignals

        # ------------------- Human Activity Recognition ------------------- #  

        # Predict which activity the subject is experiencing.
        activityFeatures = self.extractActivityFeatures(featureData)
        activityDistribution = specificEmotionModel.classifyHumanActivity(activityFeatures)
        activityDistribution = self.applyFinalActivation(activityDistribution, self.lastActivityLayer)  # Normalize the distributions for the expected loss function.
        # activityDistribution dimension: batchSize, self.numActivities

        # ------------------ Basic Emotion Classification ------------------ #  

        # For each possible interpretation, predict a set of basic emotional states.
        eachBasicEmotionDistribution = self.predictBasicEmotions(featureData)
        # eachBasicEmotionDistribution dimension: batchSize, self.numInterpreterHeads, self.numBasicEmotions, self.emotionLength

        # Decide which set of interpretations the user is following. 
        basicEmotionDistributions = specificEmotionModel.predictUserEmotions(eachBasicEmotionDistribution, activityDistribution, subjectInds)
        # basicEmotionDistributions dimension: batchSize, self.numBasicEmotions, self.emotionLength

        # ----------------- Complex Emotion Classification ----------------- #  

        # Recombine the basic emotions into one complex emotional state.
        finalEmotionDistributions = specificEmotionModel.predictComplexEmotions(basicEmotionDistributions, featureData)
        # finalEmotionDistributions = self.applyFinalActivation(finalEmotionDistributions, self.lastEmotionLayer) # Normalize the distributions for the expected loss function.
        # finalEmotionDistributions dimension: self.numEmotions, batchSize, self.emotionLength

        # # import matplotlib.pyplot as plt
        # # plt.plot(torch.arange(0, 10, 10/self.emotionLength).detach().cpu().numpy() - 0.5, finalEmotionDistributions[emotionInd][0].detach().cpu().numpy()); plt.show()

        return featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions

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
