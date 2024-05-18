# General
import torch
import random

# Helper classes
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.classWeightHelpers import classWeightHelpers
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods


class emotionDataInterface:

    @staticmethod
    def addNoise(data, trainingFlag, noiseSTD=0.01):
        # If we are training, add noise to the final state to ensure continuity of the latent space.
        return data + torch.randn_like(data, device=data.device) * noiseSTD if trainingFlag else data

    @staticmethod
    def shuffleDimension(signalDatas):
        # Ensure that all tensors have the same shape
        assert all(signalDatas[0].shape == signalDatas[i].shape for i in range(1, len(signalDatas)))

        # Get the shape of the tensors
        batchSize, numSignals, sequenceLength = signalDatas[0].shape

        # Generate random permutation indices for shuffling
        shuffle_indices = torch.randperm(numSignals)

        # Shuffle each tensor in the batch along the numSignals dimension
        augmentedSignalDatas = (signalData[:, shuffle_indices, :] for signalData in signalDatas)

        return augmentedSignalDatas

    def changeSignalLength(self, minimumSignalLength, signalDatas):
        # Assuming signalData is your tensor with dimensions [batchSize, numSignals, sequenceLength]
        assert all(signalDatas[0].shape == signalDatas[i].shape for i in range(len(signalDatas)))
        batchSize, numSignals, sequenceLength = signalDatas[0].shape

        # Find a random place to cut the data.
        randomSignalEnd = torch.tensor(generalMethods.biased_high_sample(minimumSignalLength, sequenceLength, randomValue=random.uniform(0, 1)), dtype=torch.int32).item()

        # Slice all the data at the same index
        augmentedSignalDatas = (self.getRecentSignalPoints(signalData, randomSignalEnd) for signalData in signalDatas)

        return augmentedSignalDatas

    def changeNumSignals(self, signalDatas, minNumSignals, maxNumSignals, alteredDim=1):
        # Assuming signalData is your tensor with dimensions [batchSize, numSignals, sequenceLength]
        assert all(signalDatas[0].shape == signalDatas[i].shape for i in range(len(signalDatas)))
        numSignals = signalDatas[0].size(alteredDim)
        minValue = max(minNumSignals+1, int(numSignals/3))

        # Expand the number of signals.
        repeat_times = (maxNumSignals + numSignals - 1) // numSignals  # Calculate the number of times we need to repeat the tensor
        signalDatas = [signalData.repeat_interleave(repeat_times, dim=alteredDim)[:, 0:maxNumSignals, :] for signalData in signalDatas]  # Repeat the tensor along the second dimension

        # Shuffle the signals to ensure that we are not always removing the same signals.
        signalDatas = self.shuffleDimension(signalDatas)

        # Find a random place to cut the data.
        randomEnd = torch.tensor(generalMethods.biased_high_sample(minValue, maxNumSignals, randomValue=random.uniform(0, 1)), dtype=torch.int32).item()

        finalDatas = []
        for signalData in signalDatas:
            # Slice all the data at the same index
            finalDatas.append(self.getRecentSignals(signalData, randomEnd))

        return finalDatas

    @staticmethod
    def getRecentSignalPoints(signalData, finalLength):
        return signalData[:, :, -finalLength:].contiguous()

    @staticmethod
    def getRecentSignals(signalData, finalLength):
        return signalData[:, 0:finalLength, :].contiguous()

    # ---------------------------------------------------------------------- #
    # ---------------------- Data Structure Interface ---------------------- #  

    @staticmethod
    def separateData(inputData, sequenceLength, numSubjectIdentifiers, demographicLength):
        # Extract the incoming data's dimension and ensure a proper data format.
        batchSize, numSignals, signalInfoLength = inputData.size()
        # inputData = inputData.to(torch.float32)  # Floats are required for gradient tracking.

        # Assert the validity of the input data
        assert signalInfoLength == sequenceLength + numSubjectIdentifiers + demographicLength, \
            f"{signalInfoLength} != {sequenceLength} {numSubjectIdentifiers} {demographicLength}"

        # Separate the sequence and demographic information.
        subjectIdentifiers = inputData[:, 0, sequenceLength:sequenceLength + numSubjectIdentifiers].to(torch.int)
        demographicData = inputData[:, :, sequenceLength + numSubjectIdentifiers:]  # .to(torch.float32)
        signalData = inputData[:, :, 0:sequenceLength]  # .to(torch.float32)
        # demographicData dimension: batchSize, numSignals, demographicLength
        # signalData dimension: batchSize, numSignals, sequenceLength
        # subjectInds dimension: batchSize, numSubjectIdentifiers

        return signalData, demographicData, subjectIdentifiers

    def getReconstructionIndex(self, allTrainingMasks):
        # Find the first label index with training points.
        reconstructionIndices = self.getLabelInds_withPoints(allTrainingMasks)
        assert len(reconstructionIndices) != 0, f"We should have some training data: {reconstructionIndices}"

        return reconstructionIndices[0]

    @staticmethod
    def getActivityColumn(allLabels, activityLabelInd):
        return allLabels[:, activityLabelInd]

    @staticmethod
    def getEmotionMasks(allLabelsMask, numEmotions):
        return allLabelsMask[:, 0:numEmotions]

    @staticmethod
    def getEmotionColumn(allLabels, emotionInd):
        return allLabels[:, emotionInd]

    def getActivityLabels(self, allLabels, allLabelsMask, activityLabelInd):
        activityMask = self.getActivityColumn(allLabelsMask, activityLabelInd)
        allActivityLabels = self.getActivityColumn(allLabels, activityLabelInd)

        return allActivityLabels[activityMask]

    def getEmotionLabels(self, emotionInd, allLabels, allLabelsMask):
        emotionDataMask = self.getEmotionColumn(allLabelsMask, emotionInd)
        emotionLabels = self.getEmotionColumn(allLabels, emotionInd)

        return emotionLabels[emotionDataMask]

    def getEmotionDistributions(self, emotionInd, predictedEmotionLabels, allLabels, allLabelsMask, allEmotionClasses, emotionLength):
        # Organize the emotion's training information.
        emotionLabels = self.getEmotionLabels(emotionInd, allLabels, allLabelsMask)
        emotionDataMask = self.getEmotionColumn(allLabelsMask, emotionInd)

        # Get the predicted and true emotion distributions.
        predictedTrainingEmotions = predictedEmotionLabels[emotionInd][emotionDataMask]
        trueTrainingEmotions = classWeightHelpers.gausEncoding(emotionLabels, allEmotionClasses[emotionInd], emotionLength)

        return predictedTrainingEmotions, trueTrainingEmotions

    @staticmethod
    def getLabelInds_withPoints(allTrainingMasks):
        # Find the label indices with training points.
        numTrainingPoints = allTrainingMasks.sum(dim=0)  # Find the number of training points per label.
        goodExperimentalInds = numTrainingPoints.nonzero()
        # If there are no good indices, return an empty list.
        if len(goodExperimentalInds) == 0: return []

        return goodExperimentalInds[:, 0]
