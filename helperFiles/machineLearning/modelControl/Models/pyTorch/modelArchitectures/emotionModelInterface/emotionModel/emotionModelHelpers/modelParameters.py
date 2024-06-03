import random

# Import helper files.
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods


class modelParameters:
    def __init__(self, userInputParams, accelerator=None):
        # General parameters
        self.gpuFlag = accelerator.device.type == 'cuda'
        self.userInputParams = userInputParams
        self.accelerator = accelerator

        # Helper classes.
        self.generalMethods = generalMethods()

    @staticmethod
    def getModelInfo(submodel, specificInfo=None):
        # Base case: information hard-coded.
        if specificInfo is not None:
            return specificInfo

        # No model information to load.
        loadSubmodelEpochs = None
        loadSubmodelDate = None
        loadSubmodel = None

        if submodel == "autoencoder":
            # Model loading information.
            loadSubmodelDate = f"2024-04-06 Final signalEncoder on cuda at numExpandedSignals 4 at numSigEncodingLayers 4"  # The date the model was trained.
            loadSubmodel = "signalEncoder"  # The model's component we are loading.
            loadSubmodelEpochs = -1  # The number of epochs the loading model was trained.

        elif submodel == "emotionPrediction":
            # Model loading information.
            loadSubmodelDate = f"2024-01-10 Final signalEncoder"  # The date the model was trained.
            loadSubmodel = "autoencoder"  # The model's component we are loading.
            loadSubmodelEpochs = -1  # The number of epochs the loading model was trained.

        return loadSubmodelDate, loadSubmodelEpochs, loadSubmodel

    def getAugmentationDeviation(self, submodel):
        # Get the submodels to save
        if submodel == "signalEncoder":
            addingNoiseRange = (0, 0.001)
        elif submodel == "autoencoder":
            addingNoiseRange = (0, 0.001)
        elif submodel == "emotionPrediction":
            addingNoiseRange = (0, 0.001)
        else:
            assert False, "No model initialized"

        return self.generalMethods.biased_high_sample(*addingNoiseRange, randomValue=random.uniform(a=0, b=1)), addingNoiseRange

    def getTrainingBatchSize(self, submodel, metaDatasetName):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 75 experiments with 128 signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 99 signals.
        # Amigos: Found 12 (out of 12) well-labeled emotions across 178 experiments with 223 signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 41 signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1650 experiments with 87 signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 191 experiments with 183 signals.

        if submodel == "signalEncoder":
            totalMinBatchSize = 16
        elif submodel == "autoencoder":
            totalMinBatchSize = 16
        elif submodel == "emotionPrediction":
            totalMinBatchSize = 16
        else:
            raise Exception()

        # Adjust the batch size based on the number of gradient accumulations.
        gradientAccumulation = self.accelerator.gradient_accumulation_steps
        minimumBatchSize = totalMinBatchSize // gradientAccumulation
        # Assert that the batch size is divisible by the gradient accumulation steps.
        assert totalMinBatchSize % gradientAccumulation == 0, "The total batch size must be divisible by the gradient accumulation steps."
        assert gradientAccumulation <= totalMinBatchSize, "The gradient accumulation steps must be less than the total batch size."

        # Specify the small batch size datasets.
        if metaDatasetName in ['wesad']: return minimumBatchSize  # 1 times larger than the smallest dataset.

        # Specify the small-medium batch size datasets.
        if metaDatasetName in ['empatch']: return 2 * minimumBatchSize  # 2.5466 times larger than the smallest dataset.
        if metaDatasetName in ['amigos']: return 2 * minimumBatchSize  # 2.3733 times larger than the smallest dataset.

        # Specify the medium batch size datasets.
        if metaDatasetName in ['emognition']: return 4 * minimumBatchSize  # 5.4266 times larger than the smallest dataset.
        if metaDatasetName in ['dapper']: return 4 * minimumBatchSize  # 4.8533 times larger than the smallest dataset.

        # Specify the large batch size datasets.
        if metaDatasetName in ['case']: return 8 * minimumBatchSize  # 22 times larger than the smallest dataset.

        assert False, f"Dataset {metaDatasetName} not found for submodel {submodel}."

    def getInferenceBatchSize(self, submodel, numSignals):
        # Wesad: Found 32 (out of 32) well-labeled emotions across 75 experiments with 128 signals.
        # Emognition: Found 12 (out of 12) well-labeled emotions across 407 experiments with 99 signals.
        # Amigos: Found 12 (out of 12) well-labeled emotions across 178 experiments with 223 signals.
        # Dapper: Found 12 (out of 12) well-labeled emotions across 364 experiments with 41 signals.
        # Case: Found 2 (out of 2) well-labeled emotions across 1650 experiments with 87 signals.
        # Collected: Found 30 (out of 30) well-labeled emotions across 191 experiments with 183 signals.
        # Set the minimum batch size.
        minimumBatchSize = 16 if self.gpuFlag else 4

        if submodel == "signalEncoder":
            if self.userInputParams['numSigEncodingLayers'] <= 2 and self.userInputParams['numSigLiftedChannels'] <= 48: minimumBatchSize = 16
        elif submodel == "autoencoder":
            minimumBatchSize = 32 if self.userInputParams['deviceListed'].startswith("HPC") else 32
        elif submodel == "emotionPrediction":
            minimumBatchSize = 32 if self.userInputParams['deviceListed'].startswith("HPC") else 32
        else:
            raise Exception()

        maxNumSignals = 223
        # Adjust the batch size based on the number of signals used.
        maxBatchSize = int(minimumBatchSize * maxNumSignals / numSignals)
        maxBatchSize = min(maxBatchSize, numSignals)  # Ensure the maximum batch size is not larger than the number of signals.

        return maxBatchSize

    @staticmethod
    def getNumEpochs(submodel):
        if submodel == "signalEncoder":
            return 1000, 5  # numEpoch, numConstrainedEpochs
        elif submodel == "autoencoder":
            return 1000, 5  # numEpoch, numConstrainedEpochs
        elif submodel == "emotionPrediction":
            return 1000, 5  # numEpoch, numConstrainedEpochs
        else:
            raise Exception()

    @staticmethod
    def getSequenceLength(submodel, sequenceLength):
        if submodel == "signalEncoder":
            return 90, 240
        elif submodel == "autoencoder":
            return 90, 240
        elif submodel == "emotionPrediction":
            return sequenceLength, sequenceLength
        else:
            raise Exception()

    @staticmethod
    def getSignalMinMaxScale():
        return 1  # Some wavelets constrained to +/- 1.

    @staticmethod
    def getShiftInfo(submodel):
        if submodel == "signalEncoder":
            return ["wesad", "emognition", "amigos", "dapper", "case"], 40, 50
        elif submodel == "autoencoder":
            return ["wesad", "emognition", "amigos", "dapper", "case"], 40, 50
        elif submodel == "emotionPrediction":
            return ['case', 'amigos'], 4, 2
        else:
            raise Exception()

    @staticmethod
    def getExclusionCriteria(submodel):
        if submodel == "signalEncoder":
            return -1, 2  # Emotion classes dont matter.
        elif submodel == "autoencoder":
            return -1, 2  # Emotion classes dont matter.
        elif submodel == "emotionPrediction":
            return 2, 0.8
        else:
            raise Exception()

    @staticmethod
    def getSavingInformation(epoch, numConstrainedEpochs, numEpoch_toSaveFull, numEpoch_toPlot):
        # Initialize flags to False.
        saveFullModel = False

        # Determine if we should save or plot the model.
        if epoch < numConstrainedEpochs:
            plotSteps = True
        elif epoch == numConstrainedEpochs:
            saveFullModel = True
            plotSteps = True
        else:
            saveFullModel = (epoch % numEpoch_toSaveFull == 0)
            plotSteps = (epoch % numEpoch_toPlot == 0)

        return saveFullModel, plotSteps

    @staticmethod
    def getEpochInfo(submodel, useParamsHPC):
        if submodel == "signalEncoder":
            return 10, 10 if useParamsHPC else -1  # numEpoch_toPlot, numEpoch_toSaveFull
        elif submodel == "autoencoder":
            return 10, 10 if useParamsHPC else -1  # numEpoch_toPlot, numEpoch_toSaveFull
        elif submodel == "emotionPrediction":
            return 10, 10 if useParamsHPC else -1  # numEpoch_toPlot, numEpoch_toSaveFull
        else:
            raise Exception()

    @staticmethod
    def getSubmodelsSaving(submodel):
        # Get the submodels to save
        if submodel == "signalEncoder":
            submodelsSaving = ["trainingInformation", "signalEncoderModel"]
        elif submodel == "autoencoder":
            submodelsSaving = ["trainingInformation", "signalEncoderModel", "autoencoderModel"]
        elif submodel == "emotionPrediction":
            submodelsSaving = ["trainingInformation", "signalEncoderModel", "autoencoderModel", "signalMappingModel", "specificEmotionModel", "sharedEmotionModel"]
        else:
            assert False, "No model initialized"

        return submodelsSaving

    @staticmethod
    def maxNormL2(submodel):
        if submodel == "signalEncoder":
            return 10  # Empirically: StrongL 5 < maxNorm < 10; Weaker: 10 < maxNorm < 20
        elif submodel == "autoencoder":
            return 5
        elif submodel == "emotionPrediction":
            return 5
        else:
            assert False, "No maxL2Norm initialized"
