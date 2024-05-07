# General
import os
import sys
import time
import warnings
import argparse
import numpy as np
from accelerate import DataLoaderConfiguration

# Set specific environmental parameters.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING and ERROR, 3 = ERROR only)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Hugging Face
import accelerate
import torch

# Import files for machine learning
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

# Use the same initialization every time.
torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    # Define the accelerator parameters.
    accelerator = accelerate.Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=True),  # Whether to split batches across devices or not.
        step_scheduler_with_optimizer=False,  # Whether to wrap the optimizer in a scheduler.
        gradient_accumulation_steps=4,  # The number of gradient accumulation steps.
        mixed_precision="no",  # FP32 = "no", BF16 = "bf16", FP16 = "fp16", FP8 = "fp8"
    )

    # General model parameters.
    trainingDate = "2024-05-05 bior37 db3 average"  # The current date we are training the model. Unique identifier of this training set.
    modelName = "emotionModel"  # The emotion model's unique identifier. Options: emotionModel
    trainTestSplit = 0.2  # The percentage of testing points.

    # Training flags.
    useFinalLearningParams = True  # If you want to use FINAL training parameters. The ONLY effect on training is LR.
    plotTrainingSteps = True  # If you want to plot any results from training.
    storeLoss = False  # If you want to record any loss values.
    fastPass = True  # If you want to only plot/train 240 points. No effect on training.

    # ---------------------------------------------------------------------- #
    # ----------------------- Parse Model Parameters ----------------------- #

    # Create the parser
    parser = argparse.ArgumentParser(description='Specify model parameters.')

    # Add arguments for the general model
    parser.add_argument('--submodel', type=str, default="signalEncoder", help='The component of the model we are training. Options: signalEncoder, autoencoder, emotionPrediction')
    parser.add_argument('--deviceListed', type=str, default=accelerator.device.type, help='The device we are running the platform on')
    # Add arguments for the signal encoder prediction
    parser.add_argument('--numLiftedChannels', type=int, default=48, help='The number of channels to lift before the fourier neural operator. Range: (16, 80, 16)')
    parser.add_argument('--numEncodingLayers', type=int, default=2, help='The number of layers in the transformer encoder. Range: (0, 6, 1)')
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
        'deviceListed': args.deviceListed,  # The device we are running the platform on.
        'submodel': args.submodel,  # The component of the model we are training.
        # Assign signal encoder parameters
        'numExpandedSignals': args.numExpandedSignals,  # The number of signals to group when you begin compression or finish expansion.
        'numEncodingLayers': args.numEncodingLayers,  # The number of transformer layers during signal encoding.
        'numLiftedChannels': args.numLiftedChannels,  # The number of channels to lift before the fourier neural operator.
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
    if userInputParams['deviceListed'].startswith("HPC"):
        accelerator.gradient_accumulation_steps = 16
        storeLoss = True  # Turn on loss storage for HPC.
        fastPass = False  # Turn off fast pass for HPC.

        print("HPC Parameters:", fastPass, accelerator, accelerator.gradient_accumulation_steps, flush=True)

    # ---------------------------------------------------------------------- #
    # --------------------------- Setup Training --------------------------- #

    # Specify shared model parameters.
    # Possible models: ["trainingInformation", "signalEncoderModel", "autoencoderModel", "signalMappingModel", "specificEmotionModel", "sharedEmotionModel"]
    sharedModelWeights = ["signalEncoderModel", "autoencoderModel", "sharedEmotionModel"]

    # Initialize the metaController class.
    modelCompiler = compileModelData(submodel, userInputParams, accelerator)
    modelInfoClass = compileModelInfo("_.pkl", [0, 1, 2])
    modelMigration = modelMigration(accelerator)
    featureAnalysis = featureImportance("")

    # Specify training parameters
    numEpoch_toPlot, numEpoch_toSaveFull = modelCompiler.getEpochInfo(submodel)
    trainingDate = modelCompiler.embedInformation(submodel, trainingDate)  # Embed training information into the name.

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
                                                                                            metaSubjectOrder, metaFeatureNames, metaDatasetNames, modelName, submodel, trainTestSplit, useFinalLearningParams, metaTraining=True,
                                                                                            specificInfo=None, random_state=42)
    # Compile the final modules.
    allModels, allDataLoaders, allLossDataHolders = modelCompiler.compileModels([allAlignedFeatureIntervals], [surveyAnswersList], [surveyQuestions], [activityLabels], [activityNames], [numQuestionOptions], [subjectOrder],
                                                                                [featureNames], datasetNames, modelName, submodel, trainTestSplit, useFinalLearningParams, metaTraining=False, specificInfo=None, random_state=42)
    # Create the meta-loss models and data loaders.
    allMetaLossDataHolders.extend(allLossDataHolders)

    # Clean up the code.
    del metaRawFeatureTimesHolders, metaRawFeatureHolders, metaRawFeatureIntervals, metaRawFeatureIntervalTimes, metaAlignedFeatureTimes, metaAlignedFeatureHolder, metaAlignedFeatureIntervals, metaAlignedFeatureIntervalTimes, metaSubjectOrder, \
        metaExperimentalOrder, metaActivityNames, metaActivityLabels, metaFinalFeatures, metaFinalLabels, metaFeatureLabelTypes, metaFeatureNames, metaSurveyQuestions, metaSurveyAnswersList, metaSurveyAnswerTimes, metaNumQuestionOptions
    del allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
        subjectOrder, experimentalOrder, activityNames, activityLabels, allFinalFeatures, allFinalLabels, featureLabelTypes, featureNames, surveyQuestions, surveyAnswersList, surveyAnswerTimes, numQuestionOptions

    # Unify all the fixed weights in the models
    unifiedLayerData = modelMigration.copyModelWeights(allMetaModels[0], sharedModelWeights)
    modelMigration.unifyModelWeights(allMetaModels, sharedModelWeights, unifiedLayerData)
    modelMigration.unifyModelWeights(allModels, sharedModelWeights, unifiedLayerData)

    t1 = time.time()
    # For each meta-training model.
    for modelInd in range(len(allMetaLossDataHolders)):
        lossDataLoader = allMetaLossDataHolders[modelInd]  # Contains the same information but with a different batch size.
        modelPipeline = allMetaModels[modelInd] if modelInd < len(metaDatasetNames) else allModels[0]  # Same pipeline instance in training loop.

        with torch.no_grad():
            # Calculate and store all the training and testing losses of the untrained model.
            if storeLoss:
                modelPipeline.organizeLossInfo.storeTrainingLosses(submodel, modelPipeline, lossDataLoader, fastPass)
            modelPipeline.scheduler.step()  # Update the learning rate.
    t2 = time.time()
    accelerator.print("Total loss calculation time:", t2 - t1)

    # For each training epoch
    for epoch in range(2, 1000):
        print(f"\nEpoch: {epoch}", flush=True)
        plotSteps = plotTrainingSteps and epoch % numEpoch_toPlot == 0
        saveFullModel = epoch % numEpoch_toSaveFull == 0
        numTrainingSteps = 1

        startEpochTime = time.time()
        # For each meta-training model.
        for modelInd in range(len(allMetaDataLoaders)):
            dataLoader = allMetaDataLoaders[modelInd]
            modelPipeline = allMetaModels[modelInd]

            # Load in the previous weights.
            modelMigration.unifyModelWeights([modelPipeline], sharedModelWeights, unifiedLayerData)

            # Train the model numTrainingSteps times and store training parameters.
            modelPipeline.trainModel(dataLoader, submodel, numTrainingSteps, trainingFlag=True)
            accelerator.wait_for_everyone()  # Wait for every device to reach this point before continuing.

            # Save and store the new model with its meta-trained weights.
            unifiedLayerData = modelMigration.copyModelWeights(modelPipeline, sharedModelWeights)
        # Unify all the model weights.
        modelMigration.unifyModelWeights(allMetaModels, sharedModelWeights, unifiedLayerData)
        modelMigration.unifyModelWeights(allModels, sharedModelWeights, unifiedLayerData)

        if storeLoss:
            t1 = time.time()
            # For each meta-training model.
            for modelInd in range(len(allMetaLossDataHolders)):
                lossDataLoader = allMetaLossDataHolders[modelInd]  # Contains the same information but with a different batch size.
                modelPipeline = allMetaModels[modelInd] if modelInd < len(metaDatasetNames) else allModels[0]  # Same pipeline instance in training loop.

                with torch.no_grad():
                    # Calculate and store all the training and testing losses of the untrained model.
                    modelPipeline.organizeLossInfo.storeTrainingLosses(submodel, modelPipeline, lossDataLoader, fastPass)
            t2 = time.time()
            accelerator.print("Total loss calculation time:", t2 - t1)

        if plotSteps:
            t1 = time.time()
            # For each meta-training model.
            for modelInd in range(len(allMetaLossDataHolders)):
                lossDataLoader = allMetaLossDataHolders[modelInd]  # Contains the same information but with a different batch size.
                modelPipeline = allMetaModels[modelInd] if modelInd < len(metaDatasetNames) else allModels[0]  # Same pipeline instance in training loop.

                with torch.no_grad():
                    numEpochs = modelPipeline.getTrainingEpoch(submodel) or epoch
                    modelPipeline.modelVisualization.plotAllTrainingEvents(submodel, modelPipeline, lossDataLoader, trainingDate, numEpochs, fastPass)
            allMetaModels[0].modelVisualization.plotDatasetComparison(submodel, allMetaModels + allModels, trainingDate, fastPass)
            t2 = time.time()
            accelerator.print("Total plotting time:", t2 - t1)

        endEpochTime = time.time()
        print("Total epoch time:", endEpochTime - startEpochTime)
        # Save the model sometimes (only on the main device).
        if saveFullModel and accelerator.is_local_main_process:
            # Prepare to save the model.
            modelPipeline = allMetaModels[-1]
            numEpochs = modelPipeline.getTrainingEpoch(submodel) or epoch
            submodelsSaving = modelPipeline.getSubmodelsSaving(submodel)
            modelMigration.unifyModelWeights(allMetaModels, sharedModelWeights, unifiedLayerData)

            # Create a copy of the pipelines together
            allPipelines = allMetaModels + allModels
            # Save the current version of the model.11
            modelMigration.saveModels(allPipelines, modelName, allDatasetNames, sharedModelWeights, submodelsSaving,
                                      submodel, trainingDate, numEpochs, metaTraining=True, saveModelAttributes=True)

        # Wait before continuing.
        accelerator.wait_for_everyone()
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

    exit()

    27 * 75 + 11 * 407 + 10 * 145 + 12 * 364 + 2 * 1650

    # Prepare our model for training.
    modelMigration.resetLayerStatistics(allModels, sharedModelWeights)  # Reset any statistics in the model.
    modelMigration.changeGradTracking(allModels, sharedModelWeights, requires_grad=False)  # Freeze the model weights.
    # Extract the features from the time-series model.
    unifiedDataLoaders = modelCompiler.convertSignalToFeatures(allModels, allDataLoaders, covShiftEpochs=100,
                                                               plotCovShift=True)
    # exit()

    # Holders for psych predictions.
    arrayLength = allModels[0].arrayLength
    allData, allLabels, allTrainingMasks, allTestingMasks = unifiedDataLoaders[0].dataset.getAll()
    emotionProbabilities = np.zeros((len(allModels), allData.shape[0], allLabels.shape[1],
                                     arrayLength))  # Dim: numModels, numExperiments, numEmotions, numAnswers
    trueEmotionProbabilities = np.zeros((len(allModels), allData.shape[0], allLabels.shape[1],
                                         arrayLength))  # Dim: numModels, numExperiments, numEmotions, numAnswers

    print("\nTraining models", flush=True)
    allFinalEmotionPredictions_Test = []
    allFinalEmotionPredictions_Train = []
    # For each model with data I collected.
    for modelInd in range(len(unifiedDataLoaders)):
        print(f"\tmodelInd: {modelInd}", flush=True)
        dataLoader = unifiedDataLoaders[modelInd]
        modelPipeline = allModels[modelInd]

        # Train he final emotion classification layer of the model.
        modelPipeline.trainModel(dataLoader, numEpochs=50, trainingFlag=False, plotSteps=True)
        modelPipeline.modelHelpers._saveModel(modelPipeline, f"checkpointModel_trainedModel_50Epochs_num{modelInd}.pth",
                                              appendedFolder="emotionModels/_checkpointModels/finalModels/",
                                              saveModelAttributes=True)

        # Load in all the data and labels of the model.
        allData, allLabels, allTrainingMasks, allTestingMasks = dataLoader.dataset.getAll()
        numEmotions = allLabels.shape[1]

        modelPipeline.model.eval()
        # Stop gradient tracking.
        with torch.no_grad():
            # Convert the signal data into features. 
            allPredictedLabels = modelPipeline.model.classifyEmotions(allData)
        modelPipeline.model.train()

        # For each emotion we are predicting.
        for emotionInd in range(numEmotions):
            testingMask = allTestingMasks[:, emotionInd]
            trainingMask = allTrainingMasks[:, emotionInd]

            # Transform the predictions from log-prob to probabilities.
            predictedLabels = allPredictedLabels[emotionInd].detach()
            if modelPipeline.model.lastLayer == "logSoftmax":
                predictedLabels = predictedLabels.exp()
            predictedLabels = predictedLabels.numpy()
            # Organize the results in training and testing data
            dataMask = torch.logical_or(testingMask, trainingMask)
            predictedLabels = predictedLabels[dataMask]

            # true labels
            trueEmotionLabels = modelPipeline.modelHelpers.gausEncoding(allLabels[:, emotionInd],
                                                                        modelPipeline.numClasses[emotionInd],
                                                                        modelPipeline.arrayLength)
            trueEmotionLabels = trueEmotionLabels[dataMask]

            # Predict the final emotion of the subject.
            emotionProbabilities[modelInd, :, emotionInd, :] = predictedLabels
            trueEmotionProbabilities[modelInd, :, emotionInd, :] = trueEmotionLabels

            # colors = np.array(['k', 'tab:blue', 'tab:red', 'tab:green', 'tab:brown'])
    # allData, allLabels, allTrainingMasks, allTestingMasks = dataLoader.dataset.getAll()
    # finalPredictions = modelPipeline.model(allData, trainingData = False)
    # emotionClasses = modelPipeline.numClasses
    # for emotionInd in range(len(finalPredictions)):
    #     trueLabels = allLabels[:, emotionInd]
    #     numClasses = emotionClasses[emotionInd]
    #     if (~torch.isnan(trueLabels)).sum().item() == 0: continue
    #     finalLabels  = trueLabels[~torch.isnan(trueLabels)]

    #     predictions = np.exp(finalPredictions[emotionInd].detach().numpy())
    #     predictions = predictions[~torch.isnan(trueLabels)].T

    #     finalColors = colors[finalLabels.to(int)]

    #     for classInd in range(numClasses):
    #         plt.plot(np.linspace(0, numClasses, predictions.shape[0]), predictions[], color=finalColors[classInd]);
    #     plt.show()

    import functools


    def convolveDistributions(arrs):
        conv = lambda x, y: np.convolve(x, y, mode='full')
        return functools.reduce(conv, arrs)


    def calculateFinalScores(psychProbs):
        numModels, numExperiments, numQuestions, numAnswers = psychProbs.shape
        finalSize = numModels, numExperiments, numQuestions * (numAnswers - 1) + 1
        finalProbs = np.zeros(finalSize)

        for modelInd in range(numModels):
            for pointInd in range(numExperiments):
                allLabelProbabilities = psychProbs[modelInd, pointInd]
                # if allLabelProbabilities[0]

                finalProbs[modelInd, pointInd] = convolveDistributions(allLabelProbabilities)

        return finalProbs


    # Calculate full psych probabilities
    staiProbabilities = calculateFinalScores(emotionProbabilities[:, :, np.arange(10, 30, 1), :])
    positiveAffectivityProbs = calculateFinalScores(
        emotionProbabilities[:, :, modelCompiler.modelInfoClass.posAffectInds, :])
    negativeAffectivityProbs = calculateFinalScores(
        emotionProbabilities[:, :, modelCompiler.modelInfoClass.negAffectInds, :])
    # Calculate full psych probabilities
    trueStaiProbabilities = calculateFinalScores(trueEmotionProbabilities[:, :, np.arange(10, 30, 1), :])
    truePositiveAffectivityProbs = calculateFinalScores(
        trueEmotionProbabilities[:, :, modelCompiler.modelInfoClass.posAffectInds, :])
    trueNegativeAffectivityProbs = calculateFinalScores(
        trueEmotionProbabilities[:, :, modelCompiler.modelInfoClass.negAffectInds, :])


    def plotPsychStatistics(allPredictedPsychDists, allTruePsychDists, offset, maxScore, examType="STAI"):
        numModels, numExperiments, numScores = allPredictedPsychDists.shape
        saveStatsFolder = modelCompiler.saveTrainingData + f"psychDistributions/{examType}/"
        os.makedirs(saveStatsFolder, exist_ok=True)

        # For each set of data.
        for modelInd in range(numModels):
            # For each prediction instance.
            for experimentInd in range(numExperiments):
                # Extract the predicted and true psych distributions.
                truePyschProbabilities = allTruePsychDists[modelInd][experimentInd]
                psychProbabilities = allPredictedPsychDists[modelInd][experimentInd]

                # Plot the psych probabilities.
                plt.plot(np.linspace(offset, maxScore, numScores), truePyschProbabilities, 'k', linewidth=2,
                         label="True Distribution")
                plt.plot(np.linspace(offset, maxScore, numScores), psychProbabilities, 'tab:red', linewidth=2,
                         label="Predicted Distribution")
                # Add figure aesthetics.
                plt.title(f"{examType} Distributions")
                plt.xlabel(f"{examType} Scores")
                plt.ylabel("Probability of Score")
                # Save and show.
                plt.savefig(saveStatsFolder + f"{examType}_modelNum{modelInd}_experimentNum{experimentInd}.png")
                plt.show()


    # Plot psych statistics.
    plotPsychStatistics(staiProbabilities, trueStaiProbabilities, offset=20, maxScore=80, examType="STAI")
    plotPsychStatistics(positiveAffectivityProbs, truePositiveAffectivityProbs, offset=5, maxScore=50, examType="PA")
    plotPsychStatistics(negativeAffectivityProbs, trueNegativeAffectivityProbs, offset=5, maxScore=50, examType="NA")

    exit()

    # DEPRECATED

    import matplotlib.pyplot as plt

    numDistinctLabels = len(set(allLabels))
    allIndices = np.arange(0, len(allLabels))
    assert (np.asarray(allTotalPoints) == allTotalPoints[0]).all()
    # For each distinct label
    for distinctLabelInd in range(numDistinctLabels):
        modelInds = allIndices[distinctLabelInd::numDistinctLabels]
        finalLabel = allLabels[distinctLabelInd]
        assert len(np.unique(allLabels[distinctLabelInd::numDistinctLabels])) == 1, \
            f"The expected survey question order is off. Did you cut some? I found: {allLabels[distinctLabelInd::numDistinctLabels]}"

        numClasses = allModels[modelInds[0]].numClasses
        finalPredictions = np.zeros((len(modelInds), allTotalPoints[0], numClasses))
        trueLabels = np.zeros(allTotalPoints[0])
        # For each model for that label.
        for signalInd in range(len(modelInds)):
            modelInd = modelInds[signalInd]

            trainingLoader = allTrainingLoaders[modelInd]
            trainingIndices = allTrainingIndices[modelInd]
            testingIndices = allTestingIndices[modelInd]
            testingLoader = allTestingLoaders[modelInd]
            modelPipeline = allModels[modelInd]
            finalLabel = allLabels[modelInd]

            # Load in all the data and labels for final predictions
            allTrainingData, allTrainingLabels = trainingLoader.dataset.getAll()
            allTestingData, allTestingLabels = testingLoader.dataset.getAll()
            # Apply the gaussian filter to the data
            allTestingLabels = modelPipeline.modelHelpers.gausEncoding(allTestingLabels, numClasses, arrayLength)
            allTrainingLabels = modelPipeline.modelHelpers.gausEncoding(allTrainingLabels, numClasses, arrayLength)

            # Predict the final labels
            predictedTestingLabels = modelPipeline.model(allTestingData).detach().numpy()
            predictedTrainingLabels = modelPipeline.model(allTrainingData).detach().numpy()

            # Map the shuffled data back into the original indices 
            finalPredictions[signalInd][testingIndices] = predictedTestingLabels
            finalPredictions[signalInd][trainingIndices] = predictedTrainingLabels
            # Map the labels back into the original indices.
            trueLabels[testingIndices] = allTestingLabels.argmax(axis=1)
            trueLabels[trainingIndices] = allTrainingLabels.argmax(axis=1)

        finalClassPredictions = finalPredictions.argmax(axis=2)
        averageClassPrecitions = finalClassPredictions.mean(axis=0)
        plt.plot(finalClassPredictions, trueLabels, 'ko')
        plt.xlabel("Predicted Emotion Rating")
        plt.ylabel("Emotion Rating")
        plt.title("Emotion Ratings")
        plt.xlim((-0.1, numClasses - 0.9))
        plt.ylim((-0.1, numClasses - 0.9))
        plt.show()

# -------------------------------------------------------------------------- #

0
