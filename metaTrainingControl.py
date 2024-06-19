""" Written by Samuel Solomon: https://scholar.google.com/citations?user=9oq12oMAAAAJ&hl=en """

import os

# Set specific environmental parameters.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING and ERROR, 3 = ERROR only)
os.environ["TORCH_COMPILE_DEBUG"] = "1"

# General
import accelerate
import argparse
import torch
import time

# Import files for machine learning
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.trainingProtocolHelpers import trainingProtocolHelpers
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from helperFiles.machineLearning.modelControl.Models.pyTorch.Helpers.modelMigration import modelMigration
from helperFiles.machineLearning.featureAnalysis.featureImportance import featureImportance  # Import feature analysis files.
from helperFiles.machineLearning.dataInterface.compileModelData import compileModelData  # Methods to organize model data.

# Configure cuDNN and PyTorch's global settings.
torch.backends.cudnn.deterministic = True  # If False: allow non-deterministic algorithms in cuDNN, which can enhance performance but reduce reproducibility.
torch.set_default_dtype(torch.float32)  # Set the default data type to float32, which is typical for neural network computations.
torch.backends.cudnn.benchmark = False  # If True: Enable cuDNN's auto-tuner to find the most efficient algorithm for the current configuration, potentially improving performance if fixed input size.

if __name__ == "__main__":
    # Define the accelerator parameters.
    accelerator = accelerate.Accelerator(
        dataloader_config=accelerate.DataLoaderConfiguration(split_batches=True),  # Whether to split batches across devices or not.
        cpu=torch.backends.mps.is_available(),  # Whether to use the CPU. MPS is NOT fully compatible yet.
        step_scheduler_with_optimizer=False,  # Whether to wrap the optimizer in a scheduler.
        gradient_accumulation_steps=8,  # The number of gradient accumulation steps.
        mixed_precision="no",  # FP32 = "no", BF16 = "bf16", FP16 = "fp16", FP8 = "fp8"
    )

    # General model parameters.
    trainingDate = "2024-06-11 wavelet analysis"  # The current date we are training the model. Unique identifier of this training set.
    modelName = "emotionModel"  # The emotion model's unique identifier. Options: emotionModel
    testSplitRatio = 0.2  # The percentage of testing points.

    # Training flags.
    useFinalParams = True  # If you want to use HPC parameters (and on the HPC).
    storeLoss = False  # If you want to record any loss values.
    fastPass = True  # If you want to only plot/train 240 points. No effect on training.

    # ----------------------- Parse Model Parameters ----------------------- #

    # Create the parser
    parser = argparse.ArgumentParser(description='Specify model parameters.')

    # Add arguments for the general model
    parser.add_argument('--submodel', type=str, default="signalEncoder", help='The component of the model we are training. Options: signalEncoder, autoencoder, emotionPrediction')
    parser.add_argument('--optimizerType', type=str, default='AdamW', help='The optimizerType used during training convergence: Options: RMSprop, Adam, AdamW, SGD, etc.')
    parser.add_argument('--deviceListed', type=str, default=accelerator.device.type, help='The device we are running the platform on')
    # Add arguments for the signal encoder prediction
    parser.add_argument('--signalEncoderWaveletType', type=str, default='bior3.7', help='The wavelet type for the wavelet transform: bior3.7, db3, dmey, etc')
    parser.add_argument('--numSigLiftedChannels', type=int, default=32, help='The number of channels to lift to during signal encoding. Range: (8, 16, 32, 48)')
    parser.add_argument('--numSigEncodingLayers', type=int, default=4, help='The number of operator layers during signal encoding. Range: (0, 6, 1)')
    parser.add_argument('--numExpandedSignals', type=int, default=2, help='The number of expanded signals in the encoder. Range: (2, 6, 1)')
    # Add arguments for the autoencoder
    parser.add_argument('--compressionFactor', type=float, default=1.5, help='The compression factor of the autoencoder')
    parser.add_argument('--expansionFactor', type=float, default=1.5, help='The expansion factor of the autoencoder')
    # Add arguments for the emotion prediction
    parser.add_argument('--numInterpreterHeads', type=int, default=4, help='The number of ways to interpret a set of physiological signals.')
    parser.add_argument('--numBasicEmotions', type=int, default=8, help='The number of basic emotions (basis states of emotions).')
    parser.add_argument('--sequenceLength', type=int, default=240, help='The maximum number of time series points to consider')
    # Parse the arguments
    args = parser.parse_args()

    # --------------------------- Setup Training --------------------------- #

    # Organize all the model parameters.
    sharedModelWeights = modelParameters.getSharedModels()  # The shared model weights.
    userInputParams, submodel = modelParameters.compileParameters(args)  # The user input parameters and the submodel.
    accelerator, storeLoss, fastPass = modelParameters.setParamsHPC(args, accelerator, userInputParams, storeLoss, fastPass, useFinalParams)  # Set the HPC parameters.
    fastPass = True

    # Initialize the model information classes.
    modelCompiler = compileModelData(submodel, userInputParams, useTherapyData=False, accelerator=accelerator)
    modelParameters = modelParameters(userInputParams, accelerator)
    modelInfoClass = compileModelInfo()

    # Specify training parameters
    datasetNames, metaDatasetNames, allDatasetNames = modelParameters.compileModelNames()  # Compile the model names.
    numEpoch_toPlot, numEpoch_toSaveFull = modelParameters.getEpochInfo(submodel, useFinalParams)  # The number of epochs to plot and save the model.
    numConstrainedEpochs, numEpochs = modelParameters.getNumEpochs(submodel)  # The number of epochs to train the model.
    trainingDate = modelCompiler.embedInformation(submodel, trainingDate)  # Embed training information into the name.
    submodelsSaving = modelParameters.getSubmodelsSaving(submodel)  # The submodels to save.

    # Initialize helper classes
    trainingProtocols = trainingProtocolHelpers(accelerator, sharedModelWeights, submodelsSaving)  # Initialize the training protocols.
    modelMigration = modelMigration(accelerator)  # Initialize the model migration class.
    featureAnalysis = featureImportance("")  # Initialize the feature analysis class.

    # -------------------------- Model Compilation ------------------------- #

    # Compile the final modules.
    allModels, allDataLoaders, allLossDataHolders, allMetaModels, allMetaDataLoaders, allMetaLossDataHolders, _ = modelCompiler.compileModelsFull(metaDatasetNames, modelName, submodel, testSplitRatio, datasetNames, useFinalParams)
    unifiedLayerData = modelMigration.copyModelWeights(allMetaModels[0], sharedModelWeights)  # Unify all the fixed weights in the models

    # -------------------------- Meta-model Training ------------------------- #

    # Store the initial loss information.
    trainingProtocols.calculateLossInformation(unifiedLayerData, allMetaLossDataHolders, allMetaModels, allModels, submodel, metaDatasetNames, fastPass, storeLoss, stepScheduler=True)

    # For each training epoch
    for epoch in range(1, numEpochs + 1):
        print(f"\nEpoch: {epoch}", flush=True)
        startEpochTime = time.time()

        # Get the saving information.
        saveFullModel, plotSteps = modelParameters.getSavingInformation(epoch, numConstrainedEpochs, numEpoch_toSaveFull, numEpoch_toPlot)
        constrainedTraining = epoch <= numConstrainedEpochs

        # Train the model for a single epoch.
        unifiedLayerData = trainingProtocols.trainEpoch(submodel, allMetaDataLoaders, allMetaModels, allModels, unifiedLayerData, constrainedTraining=constrainedTraining)

        # Store the initial loss information and plot.
        if storeLoss: trainingProtocols.calculateLossInformation(unifiedLayerData, allMetaLossDataHolders, allMetaModels, allModels, submodel, metaDatasetNames, fastPass, storeLoss, stepScheduler=False)
        if plotSteps: trainingProtocols.plotModelState(epoch, unifiedLayerData, allMetaLossDataHolders, allMetaModels, allModels, submodel, metaDatasetNames, trainingDate, fastPass=fastPass)

        # Save the model sometimes (only on the main device).
        if saveFullModel and accelerator.is_local_main_process:
            trainingProtocols.saveModelState(epoch, unifiedLayerData, allMetaModels, allModels, submodel, modelName, allDatasetNames, trainingDate)

        # Finalize the epoch parameters.
        accelerator.wait_for_everyone()  # Wait before continuing.
        endEpochTime = time.time()

        print("Total epoch time:", endEpochTime - startEpochTime)

    # -------------------------- Empatch Training ------------------------- #

    # # Unify all the fixed weights in the models
    # unifiedLayerData = modelMigration.copyModelWeights(modelPipeline, sharedModelWeights)
    # modelMigration.unifyModelWeights(allMetaModels, sharedModelWeights, unifiedLayerData)
    # modelMigration.unifyModelWeights(allModels, sharedModelWeights, unifiedLayerData)
    #
    # # SHAP analysis on the metalearning models.
    # featureAnalysis = _featureImportance.featureImportance(modelCompiler.saveTrainingData)
    #
    # # For each metatraining model.
    # for modelInd in metaModelIndices:
    #     dataLoader = allMetaDataLoaders[modelInd]
    #     modelPipeline = allMetaModels[modelInd]
    #     # Place model in eval mode.
    #     modelPipeline.model.eval()
    #
    #     # Extract all the data.
    #     allData, allLabels, allTrainingMasks, allTestingMasks = dataLoader.dataset.getAll()
    #
    #     # Stop gradient tracking.
    #     with torch.no_grad():
    #         # Convert time-series to features.
    #         compressedData, encodedData, transformedData, signalFeatures, subjectInds = modelPipeline.model.compileSignalFeatures(
    #             allData, fullDataPass=False)
    #
    #     # Reshape the signal features to match the SHAP format
    #     reshapedSignalFeatures = signalFeatures.view(len(signalFeatures), -1)
    #     # reshapedSignalFeatures.view(signalFeatures.shape)
    #
    #     featureAnalysis.shapAnalysis(modelPipeline.model, reshapedSignalFeatures, allLabels,
    #                                  featureNames=modelPipeline.featureNames, modelType="", shapSubfolder="")
