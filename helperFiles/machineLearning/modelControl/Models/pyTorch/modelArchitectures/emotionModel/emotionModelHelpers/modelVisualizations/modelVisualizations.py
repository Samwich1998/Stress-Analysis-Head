# General
import os

# Pytorch
import torch

# Import files for machine learning
from ..emotionDataInterface import emotionDataInterface

# Visualization protocols
from .........globalPlottingProtocols import globalPlottingProtocols
from ._signalEncoderVisualizations import signalEncoderVisualizations
from ._autoencoderVisualizations import autoencoderVisualizations
from ._generalVisualizations import generalVisualizations


class modelVisualizations(globalPlottingProtocols):

    def __init__(self, accelerator, generalTimeWindow, modelSubfolder):
        super(modelVisualizations, self).__init__()
        # General parameters.
        self.generalTimeWindow = generalTimeWindow
        self.accelerator = accelerator
        self.saveDataFolder = None

        # Initialize helper classes.
        self.dataInterface = emotionDataInterface()

        # Initialize visualization protocols.
        self.generalViz = generalVisualizations("")
        self.autoencoderViz = autoencoderVisualizations("")
        self.signalEncoderViz = signalEncoderVisualizations("")
        # Organize the visualization components.
        self.setSavingFolder(modelSubfolder)

    def setSavingFolder(self, modelSubfolder):
        # Compile and shorten the name of the model visualization folder.
        saveDataFolder = os.path.normpath(os.path.dirname(__file__) + f"/../../dataAnalysis/{modelSubfolder}") + "/"
        self.saveDataFolder = os.path.relpath(os.path.normpath(saveDataFolder), os.getcwd()) + "/"
        # Make the folder to save the data.
        os.makedirs(self.saveDataFolder, exist_ok=True)

        # Initialize visualization protocols.
        self.generalViz.setSavingFolder(self.saveDataFolder)
        self.autoencoderViz.setSavingFolder(self.saveDataFolder)
        self.signalEncoderViz.setSavingFolder(self.saveDataFolder)

    # ---------------------------------------------------------------------- #

    def plotDatasetComparison(self, submodel, allModelPipelines, trainingDate, fastPass):
        self.accelerator.print(f"\nCalculating loss for model comparison", flush=True)
        fastPass = True

        # Prepare the model/data for evaluation.
        self.setSavingFolder(f"trainingFigures/{submodel}/{trainingDate}/modelComparison/")  # Label the correct folder to save this analysis.
        timeWindows = allModelPipelines[0].model.timeWindows

        # Plot the loss on the primary GPU.
        if self.accelerator.is_local_main_process:

            if submodel in ["signalEncoder", "autoencoder"]:
                if submodel == "signalEncoder":
                    specificModels = [modelPipeline.model.signalEncoderModel for modelPipeline in allModelPipelines]
                else:
                    specificModels = [modelPipeline.model.autoencoderModel for modelPipeline in allModelPipelines]

                # For every time window.
                for timeWindowInd in range(len(timeWindows)):
                    timeWindow = timeWindows[timeWindowInd]

                    # If we are testing, only plot one time.
                    if fastPass and timeWindow != self.generalTimeWindow: continue

                    # Plot reconstruction loss.
                    self.generalViz.plotTrainingLosses([specificModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd] for specificModel in specificModels],
                                                       [specificModel.testingLosses_timeReconstructionAnalysis[timeWindowInd] for specificModel in specificModels],
                                                       lossLabels=[f"{modelPipeline.model.datasetName} Signal Encoding Reconstruction Loss"[timeWindowInd] for modelPipeline in allModelPipelines],
                                                       plotTitle="trainingLosses/All Signal Encoder Convergence Losses")

                    # Plot mean loss.
                    # self.generalViz.plotTrainingLosses([specificModel.trainingLosses_timeMeanAnalysis[timeWindowInd] for specificModel in specificModels],
                    #                                    [specificModel.testingLosses_timeMeanAnalysis[timeWindowInd] for specificModel in specificModels],
                    #                                    lossLabels=[f"{modelPipeline.model.datasetName} Signal Encoding Mean Loss"[timeWindowInd] for modelPipeline in allModelPipelines],
                    #                                    plotTitle="trainingLosses/All Signal Encoder Mean Losses")
                    #
                    # # Plot std loss.
                    # self.generalViz.plotTrainingLosses([specificModel.trainingLosses_timeSTDAnalysis[timeWindowInd] for specificModel in specificModels],
                    #                                    [specificModel.testingLosses_timeSTDAnalysis[timeWindowInd] for specificModel in specificModels],
                    #                                    lossLabels=[f"{modelPipeline.model.datasetName} Signal Encoding STD Loss"[timeWindowInd] for modelPipeline in allModelPipelines],
                    #                                    plotTitle="trainingLosses/All Signal Encoder STD Losses")

                    # Plot layer loss
                    self.generalViz.plotTrainingLosses([specificModel.trainingLosses_timeLayerAnalysis[timeWindowInd] for specificModel in specificModels],
                                                       [specificModel.testingLosses_timeLayerAnalysis[timeWindowInd] for specificModel in specificModels],
                                                       lossLabels=[f"{modelPipeline.model.datasetName} Signal Encoding Layer Loss"[timeWindowInd] for modelPipeline in allModelPipelines],
                                                       plotTitle="trainingLosses/All Signal Encoder Layer Losses")

                    # Plot path loss.
                    self.generalViz.plotTrainingLosses([specificModel.numEncodingsPath_timeAnalysis[timeWindowInd] for specificModel in specificModels], None,
                                                       lossLabels=[f"{modelPipeline.model.datasetName} Autoencoder Path" for modelPipeline in allModelPipelines],
                                                       plotTitle="trainingLosses/All Signal Encoder Path", logY=False)

                    # Plot buffer loss.
                    self.generalViz.plotTrainingLosses([specificModel.numEncodingsBufferPath_timeAnalysis[timeWindowInd] for specificModel in specificModels], None,
                                                       lossLabels=[f"{modelPipeline.model.datasetName} Autoencoder Path Buffer" for modelPipeline in allModelPipelines],
                                                       plotTitle="trainingLosses/All Signal Encoder Path Buffer", logY=False)

    def plotAllTrainingEvents(self, submodel, modelPipeline, lossDataLoader, trainingDate, currentEpoch, fastPass):
        self.accelerator.print(f"\nCalculating loss for {modelPipeline.model.datasetName} model", flush=True)
        fastPass = True

        # Prepare the model/data for evaluation.
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model

        # Load in all the data and labels for final predictions.
        allData, allLabels, allTrainingMasks, allTestingMasks = lossDataLoader.dataset.getAll()
        allSignalData, allDemographicData, allSubjectIdentifiers = self.dataInterface.separateData(allData, model.sequenceLength, model.numSubjectIdentifiers, model.demographicLength)
        reconstructionIndex = self.dataInterface.getReconstructionIndex(allTrainingMasks)
        assert reconstructionIndex is not None

        with torch.no_grad():
            # ---------------------- Time-Specific Plots ----------------------- #

            # For each time window we are analyzing.
            for timeAnalysisInd in range(len(model.timeWindows)):
                print(f"\tPlotting figures for the {model.timeWindows[timeAnalysisInd]} second time window")
                timeWindow = model.timeWindows[timeAnalysisInd]

                # If we are testing, only plot one time.
                if fastPass and timeWindow != model.timeWindows[5]: continue

                # Go through all the plots at this specific time window.
                self.plotTrainingEvent(model, currentEpoch, allSignalData, allSubjectIdentifiers, allTrainingMasks, allTestingMasks, reconstructionIndex, submodel, trainingDate, timeWindow, model.datasetName)

            # ---------------------- Time-Agnostic Plots ----------------------- #

            # Prepare the model/data for evaluation.
            self.setSavingFolder(f"trainingFigures/{submodel}/{trainingDate}/{model.datasetName}/")  # Label the correct folder to save this analysis.

            # Plot the loss on the primary GPU.
            if self.accelerator.is_local_main_process:

                if submodel == "signalEncoder":
                    # Plot autoencoder loss for each time.
                    self.generalViz.plotTrainingLosses(model.signalEncoderModel.trainingLosses_timeReconstructionAnalysis, model.signalEncoderModel.testingLosses_timeReconstructionAnalysis, lossLabels=model.timeWindows,
                                                       plotTitle="trainingLosses/Signal Encoder Convergence Loss")
                    self.generalViz.plotTrainingLosses(model.signalEncoderModel.trainingLosses_timeLayerAnalysis, model.signalEncoderModel.testingLosses_timeLayerAnalysis, lossLabels=model.timeWindows,
                                                       plotTitle="trainingLosses/Signal Encoder Layer Loss")
                    # self.generalViz.plotTrainingLosses(model.signalEncoderModel.trainingLosses_timeMeanAnalysis, model.signalEncoderModel.testingLosses_timeMeanAnalysis, lossLabels=model.timeWindows,
                    #                                    plotTitle="trainingLosses/Signal Encoder Mean Loss")
                    # self.generalViz.plotTrainingLosses(model.signalEncoderModel.trainingLosses_timeSTDAnalysis, model.signalEncoderModel.testingLosses_timeSTDAnalysis, lossLabels=model.timeWindows,
                    #                                    plotTitle="trainingLosses/Signal Encoder STD Loss")

                    # Plot signal encoding path.
                    self.generalViz.plotTrainingLosses(model.signalEncoderModel.numEncodingsBufferPath_timeAnalysis, None, lossLabels=model.timeWindows, plotTitle="trainingLosses/Signal Encoder Buffer Path Convergence", logY=False)
                    self.generalViz.plotTrainingLosses(model.signalEncoderModel.numEncodingsPath_timeAnalysis, None, lossLabels=model.timeWindows, plotTitle="trainingLosses/Signal Encoder Path Convergence", logY=False)

                if submodel == "autoencoder":
                    # Plot autoencoder loss for each time.
                    self.generalViz.plotTrainingLosses(model.autoencoderModel.trainingLosses_timeReconstructionAnalysis, model.autoencoderModel.testingLosses_timeReconstructionAnalysis, lossLabels=model.timeWindows,
                                                       plotTitle="trainingLosses/Autoencoder Convergence Loss")
                    self.generalViz.plotTrainingLosses(model.autoencoderModel.trainingLosses_timeLayerAnalysis, model.autoencoderModel.testingLosses_timeLayerAnalysis, lossLabels=model.timeWindows,
                                                       plotTitle="trainingLosses/Autoencoder Layer Loss")
                    # self.generalViz.plotTrainingLosses(model.autoencoderModel.trainingLosses_timeMeanAnalysis, model.autoencoderModel.testingLosses_timeMeanAnalysis, lossLabels=model.timeWindows,
                    #                                    plotTitle="trainingLosses/Autoencoder Mean Loss")
                    # self.generalViz.plotTrainingLosses(model.autoencoderModel.trainingLosses_timeSTDAnalysis, model.autoencoderModel.testingLosses_timeSTDAnalysis, lossLabels=model.timeWindows,
                    #                                    plotTitle="trainingLosses/Autoencoder STD Loss")

                    # Plot signal encoding path.
                    self.generalViz.plotTrainingLosses(model.autoencoderModel.numEncodingsBufferPath_timeAnalysis, None, lossLabels=model.timeWindows, plotTitle="trainingLosses/Autoencoder Buffer Path Convergence", logY=False)
                    self.generalViz.plotTrainingLosses(model.autoencoderModel.numEncodingsPath_timeAnalysis, None, lossLabels=model.timeWindows, plotTitle="trainingLosses/Autoencoder Path Convergence", logY=False)

            # Wait before continuing.
            self.accelerator.wait_for_everyone()

        # ------------------------------------------------------------------ #

    def plotTrainingEvent(self, model, currentEpoch, allSignalData, allSubjectIdentifiers, allTrainingMasks, allTestingMasks, reconstructionIndex, submodel, trainingDate, timeWindow, datasetName):
        # Prepare the model/data for evaluation.
        self.setSavingFolder(f"trainingFigures/{submodel}/{trainingDate}/{datasetName}/{timeWindow}/")  # Label the correct folder to save this analysis.
        model.eval()

        # General plotting parameters.
        _, numSignals, _ = allSignalData.shape

        # Augment the time axis.
        segmentedSignalData = self.dataInterface.getRecentSignalPoints(allSignalData, timeWindow)

        # ---------------------- Get the Data to Plot ---------------------- # 

        # Specify the amount of training/testing information.
        numTrainingInstances = 4
        numTestingInstances = 4

        # Get the reconstruction data mask
        reconstructionDataTrainingMask = self.dataInterface.getEmotionColumn(allTrainingMasks, reconstructionIndex)  # Dim: numExperiments
        reconstructionDataTestingMask = self.dataInterface.getEmotionColumn(allTestingMasks, reconstructionIndex)  # Dim: numExperiments
        # Only take a small set of training examples.
        reconstructionDataTrainingMask = reconstructionDataTrainingMask
        reconstructionDataTestingMask = reconstructionDataTestingMask

        # Compile the training data to plot
        trainingSignalData = segmentedSignalData[reconstructionDataTrainingMask][0: numTrainingInstances]
        trainingSubjectIdentifiers = allSubjectIdentifiers[reconstructionDataTrainingMask][0: numTrainingInstances]
        # Compile the training data to plot
        testingSignalData = segmentedSignalData[reconstructionDataTestingMask][0: numTestingInstances]
        testingSubjectIdentifiers = allSubjectIdentifiers[reconstructionDataTestingMask][0: numTestingInstances]

        # Stop gradient tracking
        with torch.no_grad():
            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            trainingEncodedData, trainingReconstructedData, trainingSignalEncodingLayerLoss, trainingCompressedData, trainingReconstructedEncodedData, trainingDenoisedDoubleReconstructedData, trainingAutoencoderLayerLoss, trainingMappedSignalData, \
                trainingReconstructedCompressedData, trainingFeatureData, trainingActivityDistributions, trainingBasicEmotionDistributions, trainingFinalEmotionDistributions \
                = model(trainingSignalData, trainingSubjectIdentifiers, trainingSignalData, reconstructSignals=True, compileVariables=False, submodel=submodel, trainingFlag=False)

            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            testingEncodedData, testingReconstructedData, testingSignalEncodingLayerLoss, testingCompressedData, testingReconstructedEncodedData, testingDenoisedDoubleReconstructedData, testingAutoencoderLayerLoss, testingMappedSignalData, \
                testingReconstructedCompressedData, testingFeatureData, testingActivityDistributions, testingBasicEmotionDistributions, testingFinalEmotionDistributions \
                = model(testingSignalData, testingSubjectIdentifiers, testingSignalData, reconstructSignals=True, compileVariables=False, submodel=submodel, trainingFlag=False)

        # ------------------- Plot the Data on One Device ------------------ # 

        # Plot the loss on the primary GPU.
        if self.accelerator.is_local_main_process:
            self.accelerator.print("Plotting")

            # ------------------- Signal Encoding Plots -------------------- # 

            if submodel == "signalEncoder":
                # Plot the encoding dimension.
                self.signalEncoderViz.plotOneSignalEncoding(testingEncodedData.detach().cpu(), currentEpoch, plotTitle="signalEncoding/Test Signal Encoding", numBatchPlots=1)
                self.signalEncoderViz.plotOneSignalEncoding(trainingEncodedData.detach().cpu(), currentEpoch, plotTitle="signalEncoding/Training Signal Encoding", numBatchPlots=1)

                # Plot all encoding dimensions.
                # self.signalEncoderViz.plotSignalEncoding(testingEncodedData.detach().cpu(), currentEpoch, plotTitle="signalEncoding/Test Signal Encoding Full Dimension")
                # self.signalEncoderViz.plotSignalEncoding(trainingEncodedData.detach().cpu(), currentEpoch, plotTitle="signalEncoding/All Training Signal Encoding Full Dimension")

            # Plot the autoencoder results.
            self.autoencoderViz.plotAutoencoder(testingSignalData.detach().cpu(), testingReconstructedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Signal Encoding Test Reconstruction", numSignalPlots=1)
            self.autoencoderViz.plotAutoencoder(trainingSignalData.detach().cpu(), trainingReconstructedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Signal Encoding Training Reconstruction", numSignalPlots=1)

            # Dont keep plotting untrained models.
            if submodel == "signalEncoder": return None

            # --------------------- Autoencoder Plots ---------------------- # 

            if submodel == "autoencoder":
                # Reconstruct the initial data.
                with torch.no_grad():
                    numSignalForwardPath = model.signalEncoderModel.encodeSignals.simulateSignalPath(numSignals, targetNumSignals=model.numEncodedSignals)[0]
                    _, _, _, propagatedReconstructedTestingData, _ = model.signalEncoderModel.reconstructEncodedData(testingReconstructedEncodedData.to(self.accelerator.device), numSignalForwardPath, signalEncodingLayerLoss=None,
                                                                                                                     calculateLoss=False, trainingFlag=False)
                    _, _, _, propagatedReconstructedTrainingData, _ = model.signalEncoderModel.reconstructEncodedData(trainingReconstructedEncodedData.to(self.accelerator.device), numSignalForwardPath, signalEncodingLayerLoss=None,
                                                                                                                      calculateLoss=False, trainingFlag=False)

                # Check the autoencoder propagated error.
                self.autoencoderViz.plotAutoencoder(testingSignalData.detach().cpu(), propagatedReconstructedTestingData.detach().cpu(), epoch=currentEpoch,
                                                    plotTitle="signalReconstruction/Reconstructed Signal Encoded Data from Autoencoder", numSignalPlots=1)
                self.autoencoderViz.plotAutoencoder(trainingSignalData.detach().cpu(), propagatedReconstructedTrainingData.detach().cpu(), epoch=currentEpoch,
                                                    plotTitle="signalReconstruction/Reconstructed Signal Encoded Data from Autoencoder", numSignalPlots=1)

                batchInd = 0
                numDistortedPlots = 2
                numSignalDistortions = 10
                # For each signal we are distorting.
                for distortedSignalInd in range(min(numDistortedPlots, segmentedSignalData.size(1))):
                    # Compile information about the distorted plots.
                    trueSignal = testingEncodedData[batchInd:batchInd + 1, distortedSignalInd:distortedSignalInd + 1, :]
                    distortedSignals = trueSignal.clone() + torch.randn((1, numSignalDistortions, testingEncodedData.size(2)), device=trainingSignalData.device) * 0.05
                    # If we don't have any signals to analyze, skip this step. This could happen with distributed training.
                    if distortedSignals.size(1) == 0: self.accelerator.print("No training data found on device"); continue

                    # Put the distorted signals through the autoencoder. 
                    with torch.no_grad():
                        compressedDistortedSignals, reconstructedDistortedEncodedSignals, _ = model.autoencoderModel(distortedSignals.to(self.accelerator.device), reconstructSignals=True, calculateLoss=False, trainingFlag=False)

                    # Plot the distorted signals
                    self.autoencoderViz.plotAllSignalComparisons(distortedSignals[0].detach().cpu(), reconstructedDistortedEncodedSignals[0].detach().cpu(), trueSignal[0][0].detach().cpu(), epoch=currentEpoch, signalInd=distortedSignalInd,
                                                                 plotTitle="signalReconstruction/Distorted Autoencoding Training Data")

            # Plot the autoencoder results.
            self.autoencoderViz.plotAutoencoder(testingEncodedData.detach().cpu(), testingReconstructedEncodedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Autoencoder Test Reconstruction", numSignalPlots=1)
            self.autoencoderViz.plotAutoencoder(trainingEncodedData.detach().cpu(), trainingReconstructedEncodedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Autoencoder Training Reconstruction", numSignalPlots=1)

            # Visualize autoencoding signal reconstruction.
            self.autoencoderViz.plotSignalComparison(testingEncodedData.detach().cpu(), testingCompressedData.detach().cpu(), epoch=currentEpoch, plotTitle="autoencoder/Autoencoding Test Compression", numSignalPlots=1)
            self.autoencoderViz.plotSignalComparison(trainingEncodedData.detach().cpu(), trainingCompressedData.detach().cpu(), epoch=currentEpoch, plotTitle="autoencoder/Autoencoding Training Compression", numSignalPlots=1)

            # Dont keep plotting untrained models.
            if submodel == "autoencoder": return None

            # ------------------ Manifold Projection Plots ----------------- # 

            if submodel == "emotionPrediction":
                # Plot latent encoding loss.
                self.generalViz.plotTrainingLosses([model.mapSignals.trainingLosses_timeReconstructionAnalysis, model.mapSignals.trainingLosses_mappedMean,
                                                    model.mapSignals.trainingLosses_mappedSTD],
                                                   [model.mapSignals.testingLosses_timeReconstructionAnalysis, model.mapSignals.testingLosses_mappedMean,
                                                    model.mapSignals.testingLosses_mappedSTD],
                                                   lossLabels=["Manifold Projection Reconstruction Loss", "Manifold Projection Mean Loss", "Manifold Projection Standard Deviation Loss"],
                                                   plotTitle="trainingLosses/Manifold Projection Loss")

                # Plot the common latent space.
                # self.latentEncoderViz.plotLatentSpace(allLatentData, allLatentData, currentEpoch, plotTitle = "Latent Space PCA", numSignalPlots = 8)

            # Plot the autoencoder results.
            self.autoencoderViz.plotAutoencoder(testingEncodedData.detach().cpu(), testingReconstructedEncodedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Manifold Test Reconstruction", numSignalPlots=1)
            self.autoencoderViz.plotAutoencoder(trainingEncodedData.detach().cpu(), trainingReconstructedEncodedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Manifold Training Reconstruction", numSignalPlots=1)

            # Dont keep plotting untrained models.
            if submodel == "emotionPrediction": return None

            # ------------------ Emotion Prediction Plots ------------------ # 

            # Organize activity information.
            # activityTestingMask = self.dataInterface.getActivityColumn(allTestingMasks)
            # activityTrainingMask = self.dataInterface.getActivityColumn(allTrainingMasks)
            # activityTestingLabels = self.dataInterface.getActivityLabels(allLabels, allTestingMasks)
            # activityTrainingLabels = self.dataInterface.getActivityLabels(allLabels, allTrainingMasks)

            # Activity plotting.
            # predictedActivityLabels = allActivityDistributions.argmax(dim=1).int()
            # self.plotPredictedMatrix(activityTrainingLabels, activityTestingLabels, predictedActivityLabels[activityTrainingMask], predictedActivityLabels[activityTestingMask], self.numActivities, currentEpoch, "Activities")
            # self.plotTrainingLosses(self.trainingLosses_activities, self.testingLosses_activities, plotTitle = "Activity Loss (Cross Entropy)")

            # Get the valid emotion indices (ones with training points).
            # emotionTrainingMask = self.dataInterface.getEmotionMasks(allTrainingMasks)
            # validEmotionInds = self.dataInterface.getLabelInds_withPoints(emotionTrainingMask)
            # For each emotion we are predicting that has training data.
            # for validEmotionInd in validEmotionInds:  
            #     testingMask = allTestingMasks[:, validEmotionInd]
            #     trainingMask = allTrainingMasks[:, validEmotionInd]
            #     emotionName = self.emotionNames[validEmotionInd]

            #     # # Organize the emotion's training/testing information.
            #     trainingEmotionLabels = self.dataInterface.getEmotionlabels(validEmotionInd, allLabels, allTrainingMasks)
            #     testingEmotionLabels = self.dataInterface.getEmotionlabels(validEmotionInd, allLabels, allTestingMasks)

            #     # Get the predicted and true emotion distributions.
            #     predictedTrainingEmotions, trueTrainingEmotions = self.dataInterface.getEmotionDistributions(validEmotionInd, allFinalEmotionDistributions, allLabels, allTrainingMasks)
            #     predictedTestingEmotions, trueTestingEmotions = self.dataInterface.getEmotionDistributions(validEmotionInd, allFinalEmotionDistributions, allLabels, allTestingMasks)

            #     # Get the class information from the testing and training data.
            #     allPredictedEmotionClasses = modelHelpers.extractClassIndex(allFinalEmotionDistributions[validEmotionInd], self.allEmotionClasses[validEmotionInd], axisDimension = 1, returnIndex = True)
            #     predictedTrainingEmotionClasses = allPredictedEmotionClasses[trainingMask]
            #     predictedTestingEmotionClasses = allPredictedEmotionClasses[testingMask]

            #     # Scale the emotion if log-softmax was the final layer.
            #     if model.lastEmotionLayer.lastEmotionLayer == "logSoftmax":
            #         predictedTestingEmotions = np.exp(predictedTestingEmotions)
            #         predictedTrainingEmotions = np.exp(predictedTrainingEmotions)

            # Get all the data predictions.
            # self.plotDistributions(trueTestingEmotions, predictedTestingEmotions, self.allEmotionClasses[validEmotionInd], plotTitle = "Testing Emotion Distributions")
            # self.plotDistributions(trueTrainingEmotions, predictedTrainingEmotions, self.allEmotionClasses[validEmotionInd], plotTitle = "Training Emotion Distributions")
            # # self.plotPredictions(trainingEmotionLabels, testingEmotionLabels, predictedTrainingEmotionClasses,
            # #                       predictedTestingEmotionClasses, self.allEmotionClasses[validEmotionInd], emotionName)
            # self.plotPredictedMatrix(trainingEmotionLabels, testingEmotionLabels, predictedTrainingEmotionClasses, predictedTestingEmotionClasses, self.allEmotionClasses[validEmotionInd], currentEpoch, emotionName)
            # # Plot model convergence curves.
            # self.plotTrainingLosses(self.trainingLosses_emotions[validEmotionInd], self.testingLosses_emotions[validEmotionInd], plotTitle = "Emotion Convergence Loss (KL)")
