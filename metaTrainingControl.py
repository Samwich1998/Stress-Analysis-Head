# General
import os
import sys
import time
import warnings
import argparse
from accelerate import DataLoaderConfiguration

from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.trainingProtocolHelpers import trainingProtocolHelpers

# Set specific environmental parameters.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING and ERROR, 3 = ERROR only)

# Hugging Face
import accelerate
import torch

# Import files for machine learning
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from helperFiles.machineLearning.modelControl.Models.pyTorch.Helpers.modelMigration import modelMigration
from helperFiles.machineLearning.featureAnalysis.featureImportance import featureImportance  # Import feature analysis files.
from helperFiles.machineLearning.dataInterface.compileModelData import compileModelData  # Methods to organize model data.

# Import meta-analysis files.
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.emognitionInterface import emognitionInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.dapperInterface import dapperInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.amigosInterface import amigosInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.wesadInterface import wesadInterface
from helperFiles.dataAcquisitionAndAnalysis.metadataAnalysis.caseInterface import caseInterface

# Add the directory of the current file to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress specific PyTorch warnings about MPS fallback
warnings.filterwarnings("ignore", message="The operator 'aten::linalg_svd' is not currently supported on the MPS backend and will fall back to run on the CPU.")

# Configure cuDNN and PyTorch's global settings.
torch.backends.cudnn.deterministic = True  # If False: allow non-deterministic algorithms in cuDNN, which can enhance performance but reduce reproducibility.
torch.set_default_dtype(torch.float32)  # Set the default data type to float32, which is typical for neural network computations.
torch.backends.cudnn.benchmark = False  # If True: Enable cuDNN's auto-tuner to find the most efficient algorithm for the current configuration, potentially improving performance if fixed input size.

if __name__ == "__main__":
    # Define the accelerator parameters.
    accelerator = accelerate.Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=True),  # Whether to split batches across devices or not.
        step_scheduler_with_optimizer=False,  # Whether to wrap the optimizer in a scheduler.
        gradient_accumulation_steps=16,  # The number of gradient accumulation steps.
        mixed_precision="no",  # FP32 = "no", BF16 = "bf16", FP16 = "fp16", FP8 = "fp8"
    )

    # General model parameters.
    trainingDate = "2024-05-20"  # The current date we are training the model. Unique identifier of this training set.
    modelName = "emotionModel"  # The emotion model's unique identifier. Options: emotionModel
    testSplitRatio = 0.2  # The percentage of testing points.

    # Training flags.
    useParamsHPC = True  # If you want to use HPC parameters (and on the HPC).
    storeLoss = False  # If you want to record any loss values.
    fastPass = True  # If you want to only plot/train 240 points. No effect on training.

    # ---------------------------------------------------------------------- #
    # ----------------------- Parse Model Parameters ----------------------- #

    # Create the parser
    parser = argparse.ArgumentParser(description='Specify model parameters.')

    # Add arguments for the general model
    parser.add_argument('--submodel', type=str, default="signalEncoder", help='The component of the model we are training. Options: signalEncoder, autoencoder, emotionPrediction')
    parser.add_argument('--optimizerType', type=str, default='AdamW', help='The optimizerType used during training convergence: Options: RMSprop, Adam, AdamW, SGD, etc.')
    parser.add_argument('--deviceListed', type=str, default=accelerator.device.type, help='The device we are running the platform on')
    # Add arguments for the signal encoder prediction
    parser.add_argument('--numPosLiftedChannels', type=int, default=4, help='The number of channels to lift to during positional encoding. Range: (1, 4, 1)')
    parser.add_argument('--numSigLiftedChannels', type=int, default=32, help='The number of channels to lift to during signal encoding. Range: (16, 64, 16)')
    parser.add_argument('--numPosEncodingLayers', type=int, default=4, help='The number of operator layers during positional encoding. Range: (0, 4, 1)')
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

    # Organize the input information into a dictionary.
    userInputParams = {
        # Assign general model parameters
        'optimizerType': args.optimizerType,  # The optimizerType used during training convergence.
        'deviceListed': args.deviceListed,  # The device we are running the platform on.
        'submodel': args.submodel,  # The component of the model we are training.
        # Assign signal encoder parameters
        'numPosLiftedChannels': args.numPosLiftedChannels,  # The number of channels to lift to during positional encoding.
        'numSigLiftedChannels': args.numSigLiftedChannels,  # The number of channels to lift to during signa; encoding.
        'numPosEncodingLayers': args.numPosEncodingLayers,  # The number of operator layers during positional encoding.
        'numSigEncodingLayers': args.numSigEncodingLayers,  # The number of operator layers during signal encoding.
        'numExpandedSignals': args.numExpandedSignals,  # The number of signals to group when you begin compression or finish expansion.
        # Assign autoencoder parameters
        'compressionFactor': args.compressionFactor,  # The compression factor of the autoencoder.
        'expansionFactor': args.expansionFactor,  # The expansion factor of the autoencoder.
        # Assign emotion prediction parameters
        'numInterpreterHeads': args.numInterpreterHeads,  # The number of ways to interpret a set of physiological signals.
        'numBasicEmotions': args.numBasicEmotions,  # The number of basic emotions (basis states of emotions).
        'sequenceLength': args.sequenceLength,  # The maximum number of time series points to consider.
    }

    # Relay the inputs to the user.
    print("System Arguments:", userInputParams, flush=True)
    submodel = args.submodel

    # Self-check the hpc parameters.
    if userInputParams['deviceListed'].startswith("HPC") and useParamsHPC:
        accelerator.gradient_accumulation_steps = 16
        # storeLoss = True  # Turn on loss storage for HPC.
        # fastPass = False  # Turn off fast pass for HPC.

        if args.submodel == "signalEncoder":
            if args.numSigLiftedChannels <= 32 and args.numSigEncodingLayers <= 4:
                accelerator.gradient_accumulation_steps = 16
            if args.numSigLiftedChannels <= 32 and args.numSigEncodingLayers <= 2:
                accelerator.gradient_accumulation_steps = 16
            if args.numSigLiftedChannels <= 16 and args.numSigEncodingLayers <= 2:
                accelerator.gradient_accumulation_steps = 16

        print("HPC Parameters:", storeLoss, fastPass, accelerator.gradient_accumulation_steps, flush=True)

    # ---------------------------------------------------------------------- #
    # --------------------------- Setup Training --------------------------- #

    # Specify shared model parameters.
    # Possible models: ["trainingInformation", "signalEncoderModel", "autoencoderModel", "signalMappingModel", "specificEmotionModel", "sharedEmotionModel"]
    sharedModelWeights = ["signalEncoderModel", "autoencoderModel", "sharedEmotionModel"]

    # Initialize the model information classes.
    modelInfoClass = compileModelInfo(modelFile="_.pkl", modelTypes=[0, 1, 2])
    modelCompiler = compileModelData(submodel, userInputParams, accelerator)
    modelParameters = modelParameters(userInputParams, accelerator)

    # Specify training parameters
    numEpoch_toPlot, numEpoch_toSaveFull = modelParameters.getEpochInfo(submodel)
    trainingDate = modelCompiler.embedInformation(submodel, trainingDate)  # Embed training information into the name.
    submodelsSaving = modelParameters.getSubmodelsSaving(submodel)
    numEpochs, numConstrainedEpochs = modelParameters.getNumEpochs(submodel)

    # Initialize helper classes
    trainingProtocols = trainingProtocolHelpers(accelerator, sharedModelWeights, submodelsSaving)
    modelMigration = modelMigration(accelerator)
    featureAnalysis = featureImportance("")

    # ---------------------------------------------------------------------- #
    # ------------------------- Feature Compilation ------------------------ #

    # Specify the metadata analysis options.
    caseProtocolClass = caseInterface()
    wesadProtocolClass = wesadInterface()
    amigosProtocolClass = amigosInterface()
    dapperProtocolClass = dapperInterface()
    emognitionProtocolClass = emognitionInterface()
    # Specify which metadata analyses to compile
    metaProtocolInterfaces = [wesadProtocolClass, emognitionProtocolClass, amigosProtocolClass, dapperProtocolClass, caseProtocolClass]
    metaDatasetNames = ["wesad", "emognition", "amigos", "dapper", "case"]
    datasetNames = ['empatch']
    allDatasetNames = metaDatasetNames + datasetNames
    # Assert the integrity of dataset collection.
    assert len(metaProtocolInterfaces) == len(metaDatasetNames)
    assert len(datasetNames) == 1

    # Compile the metadata together.
    metaRawFeatureTimesHolders, metaRawFeatureHolders, metaRawFeatureIntervals, metaRawFeatureIntervalTimes, \
        metaAlignedFeatureTimes, metaAlignedFeatureHolder, metaAlignedFeatureIntervals, metaAlignedFeatureIntervalTimes, \
        metaSubjectOrder, metaExperimentalOrder, metaActivityNames, metaActivityLabels, metaFinalFeatures, metaFinalLabels, metaFeatureLabelTypes, metaFeatureNames, metaSurveyQuestions, \
        metaSurveyAnswersList, metaSurveyAnswerTimes, metaNumQuestionOptions, metaDatasetNames = modelCompiler.compileMetaAnalyses(metaProtocolInterfaces, loadCompiledData=True)
    # Compile the project data together
    allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
        allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
        subjectOrder, experimentalOrder, activityNames, activityLabels, allFinalFeatures, allFinalLabels, featureLabelTypes, featureNames, surveyQuestions, surveyAnswersList, \
        surveyAnswerTimes, numQuestionOptions = modelCompiler.compileProjectAnalysis(loadCompiledData=True)

    # ---------------------------------------------------------------------- #
    # -------------------------- Model Compilation ------------------------- #

    # Compile the meta-learning modules.
    allMetaModels, allMetaDataLoaders, allMetaLossDataHolders = modelCompiler.compileModels(metaAlignedFeatureIntervals, metaSurveyAnswersList, metaSurveyQuestions, metaActivityLabels, metaActivityNames, metaNumQuestionOptions,
                                                                                            metaSubjectOrder, metaFeatureNames, metaDatasetNames, modelName, submodel, testSplitRatio, metaTraining=True, specificInfo=None, useParamsHPC=useParamsHPC, random_state=42)
    # Compile the final modules.
    allModels, allDataLoaders, allLossDataHolders = modelCompiler.compileModels([allAlignedFeatureIntervals], [surveyAnswersList], [surveyQuestions], [activityLabels], [activityNames], [numQuestionOptions], [subjectOrder],
                                                                                [featureNames], datasetNames, modelName, submodel, testSplitRatio, metaTraining=False, specificInfo=None, useParamsHPC=useParamsHPC, random_state=42)
    # Create the meta-loss models and data loaders.
    allMetaLossDataHolders.extend(allLossDataHolders)

    # Clean up the code.
    del metaRawFeatureTimesHolders, metaRawFeatureHolders, metaRawFeatureIntervals, metaRawFeatureIntervalTimes, metaAlignedFeatureTimes, metaAlignedFeatureHolder, metaAlignedFeatureIntervals, metaAlignedFeatureIntervalTimes, metaSubjectOrder, \
        metaExperimentalOrder, metaActivityNames, metaActivityLabels, metaFinalFeatures, metaFinalLabels, metaFeatureLabelTypes, metaFeatureNames, metaSurveyQuestions, metaSurveyAnswersList, metaSurveyAnswerTimes, metaNumQuestionOptions
    del allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
        subjectOrder, experimentalOrder, activityNames, activityLabels, allFinalFeatures, allFinalLabels, featureLabelTypes, featureNames, surveyQuestions, surveyAnswersList, surveyAnswerTimes, numQuestionOptions

    # Unify all the fixed weights in the models
    unifiedLayerData = modelMigration.copyModelWeights(allMetaModels[0], sharedModelWeights)

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
    exit()

    # Unify all the fixed weights in the models
    unifiedLayerData = modelMigration.copyModelWeights(modelPipeline, sharedModelWeights)
    modelMigration.unifyModelWeights(allMetaModels, sharedModelWeights, unifiedLayerData)
    modelMigration.unifyModelWeights(allModels, sharedModelWeights, unifiedLayerData)

    # SHAP analysis on the metalearning models.
    featureAnalysis = _featureImportance.featureImportance(modelCompiler.saveTrainingData)

    # For each metatraining model.
    for modelInd in metaModelIndices:
        dataLoader = allMetaDataLoaders[modelInd]
        modelPipeline = allMetaModels[modelInd]
        # Place model in eval mode.
        modelPipeline.model.eval()

        # Extract all the data.
        allData, allLabels, allTrainingMasks, allTestingMasks = dataLoader.dataset.getAll()

        # Stop gradient tracking.
        with torch.no_grad():
            # Convert time-series to features.
            compressedData, encodedData, transformedData, signalFeatures, subjectInds = modelPipeline.model.compileSignalFeatures(
                allData, fullDataPass=False)

        # Reshape the signal features to match the SHAP format
        reshapedSignalFeatures = signalFeatures.view(len(signalFeatures), -1)
        # reshapedSignalFeatures.view(signalFeatures.shape)

        featureAnalysis.shapAnalysis(modelPipeline.model, reshapedSignalFeatures, allLabels,
                                     featureNames=modelPipeline.featureNames, modelType="", shapSubfolder="")
