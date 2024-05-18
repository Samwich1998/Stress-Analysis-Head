# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import gc
import sys
import time
import math
# PyTorch
import torch

# Import helper files.
sys.path.append(os.path.dirname(__file__) + "/Emotion Model Helpers/submodels/modelComponents/")
import complexEmotionPrediction  # Predict the final probability distributions for each emotion.
import basicEmotionPredictions  # Feed-Forward architecture for classifying emotions.
import commonActivityAnalysis  # Synthesize the information into activity features.
import subjectInterpretation
import activityRecognition  # Predict the final probability distributions for each activity.
import featureExtraction  # Compress signal information into a 1D vector of features.

# Import submodels
sys.path.append(os.path.dirname(__file__) + "/Emotion Model Helpers/submodels/")
import manifoldEncoderModel
import signalEncoderModel  # A signal encoder pipeline to make a universal feature vector.
import signalDecoderModel
import autoencoderModel  # An autoencoder pipeline for compressing indivisual signals.

# Import global model
sys.path.append(os.path.dirname(__file__) + "/../")
import _globalPytorchModel


# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class modelHead(_globalPytorchModel.globalModel):
    def __init__(self, emotionLength, sequenceLength, numSubjectIdentifiers, demographicLength,
                 emotionNames, activityNames, featureNames, numSubjects, metaTraining=False):
        super(modelHead, self).__init__()
        # General model parameters.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Specify the CPU or GPU capabilities.
        self.demographicLength = demographicLength  # The number of demographic information (age, weight, etc). Subject index not included.
        self.numActivities = len(activityNames)  # The number of activities to predict.
        self.numEmotions = len(emotionNames)  # The number of emotions to predict.
        self.sequenceLength = sequenceLength  # The length of each incoming signal: features used in the model.
        self.numSignals = len(featureNames)  # The number of signals going into the model.
        self.emotionLength = emotionLength  # The number of inxdices (ratings) in every final emotion distribution.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.metaTraining = metaTraining  # A flag representing if this is metatraining data.
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.
        # Last layer activation.
        self.lastActivityLayer = None  # A string representing the last layer for activity prediction. Option: 'softmax', 'logsoftmax', or None.
        self.lastEmotionLayer = None  # A string representing the last layer for emotion prediction. Option: 'softmax', 'logsoftmax', or None.

        # Tunable signal parameters.
        self.compressedLength = 64  # The final length of the compressed signal after the autoencoder.
        self.numEncodedSignals = 64  # The final number of signals to accept, encoding all signal information.
        self.manifoldLength = 32  # The final signal length after manifold projection.

        # Tunable feature parameters.
        self.numCommonFeatures = 32  # The number of features from considering all the signals.
        self.numActivityFeatures = 16  # The number of common activity features to extract.
        # Tunable emotion parameters.
        self.numInterpreterHeads = 8  # The number of ways to interpret a set of physiological signals.
        self.numBasicEmotions = 12  # The number of basic emotions (basis states of emotions).

        # ------------------------ Data Compression ------------------------ # 

        # The autoencoder model reduces the incoming signal's dimension.
        self.autoencoderModel = autoencoderModel.autoencoderModel(
            compressedLength=self.compressedLength,
            sequenceLength=self.sequenceLength,
        ).to(self.device)

        # The signal encoder model to find a common feature vector across all signals.
        self.signalEncoderModel = signalEncoderModel.signalEncoderModel(
            numEncodedSignals=self.numEncodedSignals,
            compressedLength=self.compressedLength,
            numDecoderLayers=8,
            dropout_rate=0.1,
            numHeads=4,
        ).to(self.device)

        # The signal decoder model to reconstruct all signals from the common feature vector.
        self.signalDecoderModel = signalDecoderModel.signalDecoderModel(
            numEncodedSignals=self.numEncodedSignals,
            compressedLength=self.compressedLength,
            featureNames=self.featureNames,
        ).to(self.device)

        # ---------------------- Manifold Projection ----------------------- # 

        # The manifold projection model maps each signal to a common dimension.
        self.manifoldEncoderModel = manifoldEncoderModel.manifoldEncoderModel(
            numEncodedSignals=self.numEncodedSignals,
            compressedLength=self.compressedLength,
            manifoldLength=self.manifoldLength,
            featureNames=self.featureNames,
        ).to(self.device)

        # -------------------- Common Feature Extraction ------------------- #  

        self.extractFeatures = featureExtraction.featureExtraction(
            numSignalFeatures=self.manifoldLength,
            outputDimension=self.numCommonFeatures,
            numSignals=int(self.manifoldLength / self.manifoldLength),
            # numSignals = self.numSignals,
        ).to(self.device)

        # ------------------- Human Activity Recognition ------------------- #  

        # Synthesize the common features for activity recognition.
        self.extractActivityFeatures = commonActivityAnalysis.commonActivityAnalysis(
            inputDimension=self.numCommonFeatures,
            outputDimension=self.numActivityFeatures,
        ).to(self.device)

        # Predict the activity (background context) the subject is experiencing.
        self.classifyHumanActivity = activityRecognition.activityRecognition(
            inputDimension=self.numActivityFeatures,
            numActivities=self.numActivities,
        ).to(self.device)

        # ------------------ Basic Emotion Classification ------------------ #  

        # Initialize basic emotion predictions.
        self.predictBasicEmotions = basicEmotionPredictions.basicEmotionPredictions(
            numInterpreterHeads=self.numInterpreterHeads,
            numBasicEmotions=self.numBasicEmotions,
            inputDimension=self.numCommonFeatures,
            emotionLength=self.emotionLength,
        ).to(self.device)

        # Predict which type of thinker the user is.
        self.predictUserEmotions = subjectInterpretation.subjectInterpretation(
            numInterpreterHeads=self.numInterpreterHeads,
            numBasicEmotions=self.numBasicEmotions,
            numActivities=self.numActivities,
            numSubjects=self.numSubjects,
        ).to(self.device)

        # --------------------- Emotion Classification --------------------- #

        # Predict which type of thinker the user is.
        self.predictComplexEmotions = complexEmotionPrediction.complexEmotionPrediction(
            numBasicEmotions=self.numBasicEmotions,
            numFeatures=self.numCommonFeatures,
            numEmotions=self.numEmotions,
        ).to(self.device)

        # ------------------------------------------------------------------ # 

    # ---------------------------------------------------------------------- #  
    # -------------------------- Model Components -------------------------- #  

    def compressData(self, signalData, reconstructData=True, maxBatchSignals=30000):
        t1 = time.time()
        # Forward pass through the autoencoder for data compression.
        compressedData, reconstructedData = self.autoencoderModel(signalData, reconstructData, maxBatchSignals)
        # compressedData dimension: batchSize, numSignals, self.compressedLength
        # reconstructedData dimension: batchSize, numSignals, sequenceLengthh
        # signalData dimension: batchSize, numSignals, sequenceLength
        t2 = time.time();
        print("\tAutoencoder:", t2 - t1)

        return compressedData, reconstructedData

    # --------------------------- Latent Encoding -------------------------- #  

    def signalEncoding(self, signalData, reconstructSignals=True, compileVariables=False, maxBatchSignals=30000):
        # Compile the variables for signal encoding.
        with torch.no_grad():
            compressedData, reconstructedData = self.compressData(signalData, compileVariables)

        t1 = time.time()
        # Forward pass through the signal encoder to find common signal space.
        encodedData, sortingIndices = self.signalEncoderModel(compressedData, maxBatchSignals)
        # encodedData dimension: batchSize, numEncodedSignals, compressedLength
        t2 = time.time();
        print("\tSignal Encoder:", t2 - t1)

        t1 = time.time()
        # Forward pass through the signal decoder to reconstruct all the signals.
        reconstructedCompressedData = self.signalDecoderModel(encodedData, sortingIndices, maxBatchSignals) if reconstructSignals else None

        # encodedData = torch.randn((len(signalData), self.numEncodedSignals, self.compressedLength))
        # reconstructedCompressedData = torch.randn((len(signalData), len(compressedData[0]), self.compressedLength))

        # reconstructedCompressedData dimension: batchSize, numSignals, compressedLength
        t2 = time.time();
        print("\tSignal Decoder:", t2 - t1)

        return compressedData, reconstructedData, encodedData, reconstructedCompressedData

    def manifoldProjection(self, signalData, reconstructSignals=True, compileVariables=False):
        # Compile the variables for latent encoding.
        with torch.no_grad():
            compressedData, reconstructedData, encodedData, reconstructedCompressedData = self.signalEncoding(signalData, compileVariables, compileVariables)

        t1 = time.time()
        # Forward pass through the latent encoder to find common latent space.
        manifoldData, reconstructedEncodedData = self.manifoldEncoderModel(encodedData, reconstructSignals)
        # reconstructedEncodedData dimension: batchSize, numEncodedSignals, compressedLength
        # manifoldData dimension: batchSize, numEncodedSignals, manifoldLength
        t2 = time.time();
        print("\tManifold Projection:", t2 - t1)

        return compressedData, reconstructedData, encodedData, reconstructedCompressedData, manifoldData, reconstructedEncodedData

    # ------------------------- Feature Extraction ------------------------- #  

    def getFeatures(self, signalData, compileVariables=False):
        # Compile the features.
        with torch.no_grad():
            compressedData, reconstructedData, encodedData, reconstructedCompressedData, manifoldData, reconstructedEncodedData \
                = self.manifoldProjection(signalData, compileVariables, compileVariables)

            # Extract a list of features synthesizing all the signal information.
        # featureData = self.extractFeatures(manifoldData)
        featureData = torch.randn((manifoldData.size(0), self.numCommonFeatures), device=signalData.device)
        # featureData dimension: batchSize, self.numCommonFeatures

        return compressedData, reconstructedData, encodedData, reconstructedCompressedData, manifoldData, reconstructedEncodedData, featureData

    # --------------------- Human Activity Recognition --------------------- #  

    def predictActivity(self, signalData, compileVariables=False):
        # Extract a set of features from the data.
        compressedData, reconstructedData, encodedData, reconstructedCompressedData, manifoldData, reconstructedEncodedData, featureData, \
            = self.getFeatures(signalData, compileVariables)

        # Predict which activity the subject is experiencing.
        activityFeatures = self.extractActivityFeatures(featureData)
        activityDistribution = self.classifyHumanActivity(activityFeatures)
        activityDistribution = self.applyFinalActivation(activityDistribution, self.lastActivityLayer)  # Normalize the distributions for the expected loss function.
        # activityDistribution dimension: batchSize, self.numActivities

        return compressedData, reconstructedData, encodedData, reconstructedCompressedData, \
            manifoldData, reconstructedEncodedData, featureData, activityDistribution

    # ----------------------- Emotion Classification ----------------------- #  

    def emotionPrediction(self, signalData, subjectInds, compileVariables=False):
        # Extract a set of features from the data.
        compressedData, reconstructedData, encodedData, reconstructedCompressedData, manifoldData, reconstructedEncodedData, \
            featureData, activityDistribution = self.predictActivity(signalData, compileVariables)

        # ------------------ Basic Emotion Classification ------------------ #  

        # For each possible interpretation, predict a set of basic emotional states.
        eachBasicEmotionDistribution = self.predictBasicEmotions(featureData)
        # eachBasicEmotionDistribution dimension: batchSize, self.numInterpreterHeads, self.numBasicEmotions, self.emotionLength

        # Decide which set of interpretations the user is following. 
        basicEmotionDistributions = self.predictUserEmotions(eachBasicEmotionDistribution, activityDistribution, subjectInds)
        # basicEmotionDistributions dimension: batchSize, self.numBasicEmotions, self.emotionLength

        # ----------------- Complex Emotion Classification ----------------- #  

        # Recombine the basic emotions into one complex emotional state.
        finalEmotionDistributions = self.predictComplexEmotions(basicEmotionDistributions, featureData)
        # finalEmotionDistributions = self.applyFinalActivation(finalEmotionDistributions, self.lastEmotionLayer) # Normalize the distributions for the expected loss function.
        # finalEmotionDistributions dimension: self.numEmotions, batchSize, self.emotionLength

        # # import matplotlib.pyplot as plt
        # # plt.plot(torch.arange(0, 10, 10/self.emotionLength).detach().cpu().numpy() - 0.5, finalEmotionDistributions[emotionInd][0].detach().cpu().numpy()); plt.show()

        return compressedData, reconstructedData, encodedData, reconstructedCompressedData, manifoldData, reconstructedEncodedData, \
            featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions

    # ------------------------- Full Forward Calls ------------------------- #  

    def forward(self, signalData, demographicData, subjectInds, compileVariables=False):
        """ The shape of inputData: (batchSize, numSignals, signalInfoLength) """
        compressedData, reconstructedData, encodedData, reconstructedCompressedData, manifoldData, reconstructedEncodedData, \
            featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions \
            = self.emotionPrediction(signalData, subjectInds, compileVariables)

        return compressedData, reconstructedData, encodedData, reconstructedCompressedData, manifoldData, reconstructedEncodedData, \
            featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions \
 \
    def fullDataPass(self, signalData, demographicData, subjectInds, maxBatchSize=256, compileVariables=False):
        # Specify the current input shape of the data.
        batch_size, num_signals, sequence_length = signalData.shape
        device = signalData.device

        # Calculate total number of batches
        numBatches = int(math.ceil(signalData.shape[0] / maxBatchSize))

        # Preallocate tensors for the autoencoder.
        compressedData = torch.empty((batch_size, num_signals, self.compressedLength), device=device)
        reconstructedData = torch.empty((batch_size, num_signals, sequence_length), device=device)
        # Preallocate tensors for the signal encoder.
        encodedData = torch.empty((batch_size, self.numEncodedSignals, self.compressedLength), device=device)
        reconstructedCompressedData = torch.empty((batch_size, num_signals, self.compressedLength), device=device)
        # Preallocate tensors for the manifold projection.
        manifoldData = torch.empty((batch_size, self.numEncodedSignals, self.manifoldLength), device=device)
        reconstructedEncodedData = torch.empty((batch_size, self.numEncodedSignals, self.compressedLength), device=device)
        # Preallocate tensors for emotion and activity prediction.
        featureData = torch.empty((batch_size, self.numCommonFeatures), device=device)
        activityDistribution = torch.empty((batch_size, self.numActivities), device=device)
        eachBasicEmotionDistribution = torch.empty((batch_size, self.numInterpreterHeads, self.numBasicEmotions, self.emotionLength), device=device)
        finalEmotionDistributions = torch.empty((self.numEmotions, batch_size, self.emotionLength), device=device)

        # For each batch of the data
        for batchIdx in range(numBatches):
            # Calculate start and end indices for current batch
            startIdx = batchIdx * maxBatchSize
            endIdx = (batchIdx + 1) * maxBatchSize

            # Forward pass for the current batch
            compressedBatchData, reconstructedBatchData, encodedBatchData, reconstructedCompressedBatchData, manifoldBatchData, \
                reconstructedEncodedBatchData, featureBatchData, batchActivityDistribution, batchBasicEmotionDistribution, batchEmotionDistributions \
                = self.emotionPrediction(signalData[startIdx:endIdx], subjectInds[startIdx:endIdx], compileVariables)

            # Autoencoder: assign the results to the preallocated tensors
            compressedData[startIdx:endIdx] = compressedBatchData
            reconstructedData[startIdx:endIdx] = reconstructedBatchData
            # Signal encoding: assign the results to the preallocated tensors
            encodedData[startIdx:endIdx] = encodedBatchData
            reconstructedCompressedData[startIdx:endIdx] = reconstructedCompressedBatchData
            # Manifold projection: assign the results to the preallocated tensors
            manifoldData[startIdx:endIdx] = manifoldBatchData
            reconstructedEncodedData[startIdx:endIdx] = reconstructedEncodedBatchData
            # Emotion/activity model: assign the results to the preallocated tensors
            featureData[startIdx:endIdx] = featureBatchData
            activityDistribution[startIdx:endIdx] = batchActivityDistribution
            eachBasicEmotionDistribution[startIdx:endIdx] = batchBasicEmotionDistribution
            finalEmotionDistributions[:, startIdx:endIdx, :] = batchEmotionDistributions

            # Clear the cache memory
            torch.cuda.empty_cache()
            gc.collect()

        # Return preallocated tensors
        return compressedData, reconstructedData, encodedData, reconstructedCompressedData, manifoldData, reconstructedEncodedData, \
            featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions \
 \
    # --------------------------- SHAP Analysis ---------------------------- #  

    def shapInterface(self, reshapedSignalFeatures):
        # Extract the incoming data's dimension and ensure proper data format.
        batchSize, numFeatures = reshapedSignalFeatures.shape
        reshapedSignalFeatures = torch.tensor(reshapedSignalFeatures.tolist(), device=reshapedSignalFeatures.device)
        assert numFeatures == self.numSignals * self.manifoldLength, f"{numFeatures} {self.numSignals} {self.manifoldLength}"

        # Reshape the inputs to integrate into the model's expected format.
        signalFeatures = reshapedSignalFeatures.view((batchSize, self.numSignals, self.manifoldLength))

        # predict the activities.
        activityDistribution = self.forward(signalFeatures, predictActivity=True, allSignalFeatures=True)

        return activityDistribution.detach().cpu().numpy()

    # ---------------------------------------------------------------------- #
