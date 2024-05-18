# General
import torch
import time

# Helper classes.
from helperFiles.machineLearning.modelControl.Models.pyTorch.Helpers.modelMigration import modelMigration


class trainingProtocolHelpers:

    def __init__(self, accelerator, sharedModelWeights, submodelsSaving):
        # General parameters.
        self.sharedModelWeights = sharedModelWeights
        self.submodelsSaving = submodelsSaving
        self.accelerator = accelerator

        # Helper classes.
        self.modelMigration = modelMigration(accelerator)

    def trainEpoch(self, submodel, allMetaDataLoaders, allMetaModels, allModels, unifiedLayerData, linearTraining=False):
        # For each meta-training model.
        for modelInd in range(len(allMetaDataLoaders)):
            dataLoader = allMetaDataLoaders[modelInd]
            modelPipeline = allMetaModels[modelInd]

            # Load in the previous weights.
            self.modelMigration.unifyModelWeights(allModels=[modelPipeline], sharedModelWeights=self.sharedModelWeights, layerInfo=unifiedLayerData)

            # Train the model numTrainingSteps times and store training parameters.
            modelPipeline.trainModel(dataLoader, submodel, numEpochs=1, linearTraining=linearTraining)
            self.accelerator.wait_for_everyone()  # Wait for every device to reach this point before continuing.

            # Save and store the new model with its meta-trained weights.
            unifiedLayerData = self.modelMigration.copyModelWeights(modelPipeline, self.sharedModelWeights)

        # Unify all the model weights.
        self.modelMigration.unifyModelWeights(allModels=allMetaModels, sharedModelWeights=self.sharedModelWeights, layerInfo=unifiedLayerData)
        self.modelMigration.unifyModelWeights(allModels=allModels, sharedModelWeights=self.sharedModelWeights, layerInfo=unifiedLayerData)

        return unifiedLayerData

    def calculateLossInformation(self, unifiedLayerData, allMetaLossDataHolders, allMetaModels, allModels, submodel, metaDatasetNames, fastPass, storeLoss=True, stepScheduler=True):
        # Unify all the model weights.
        self.modelMigration.unifyModelWeights(allModels=allMetaModels, sharedModelWeights=self.sharedModelWeights, layerInfo=unifiedLayerData)
        self.modelMigration.unifyModelWeights(allModels=allModels, sharedModelWeights=self.sharedModelWeights, layerInfo=unifiedLayerData)

        t1 = time.time()
        # For each meta-training model.
        for modelInd in range(len(allMetaLossDataHolders)):
            lossDataLoader = allMetaLossDataHolders[modelInd]  # Contains the same information but with a different batch size.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(metaDatasetNames) else allModels[0]  # Same pipeline instance in training loop.

            with torch.no_grad():
                # Calculate and store all the training and testing losses of the untrained model.
                if storeLoss:
                    modelPipeline.organizeLossInfo.storeTrainingLosses(submodel, modelPipeline, lossDataLoader, fastPass)
                if stepScheduler: modelPipeline.scheduler.step()  # Update the learning rate.
        t2 = time.time()
        self.accelerator.print("Total loss calculation time:", t2 - t1)

    def plotModelState(self, epoch, unifiedLayerData, allMetaLossDataHolders, allMetaModels, allModels, submodel, metaDatasetNames, trainingDate, linearTraining=False, fastPass=True):
        # Unify all the model weights.
        self.modelMigration.unifyModelWeights(allModels=allMetaModels, sharedModelWeights=self.sharedModelWeights, layerInfo=unifiedLayerData)
        self.modelMigration.unifyModelWeights(allModels=allModels, sharedModelWeights=self.sharedModelWeights, layerInfo=unifiedLayerData)

        t1 = time.time()
        # For each meta-training model.
        for modelInd in range(len(allMetaLossDataHolders)):
            lossDataLoader = allMetaLossDataHolders[modelInd]  # Contains the same information but with a different batch size.
            modelPipeline = allMetaModels[modelInd] if modelInd < len(metaDatasetNames) else allModels[0]  # Same pipeline instance in training loop.

            with torch.no_grad():
                numEpochs = modelPipeline.getTrainingEpoch(submodel) or epoch
                modelPipeline.modelVisualization.plotAllTrainingEvents(submodel, modelPipeline, lossDataLoader, trainingDate, numEpochs, linearTraining, fastPass)
        allMetaModels[0].modelVisualization.plotDatasetComparison(submodel, allMetaModels + allModels, trainingDate, linearTraining, fastPass)
        t2 = time.time()
        self.accelerator.print("Total plotting time:", t2 - t1)

    def saveModelState(self, epoch, unifiedLayerData, allMetaModels, allModels, submodel, modelName, allDatasetNames, trainingDate):
        # Prepare to save the model.
        modelPipeline = allMetaModels[-1]
        numEpochs = modelPipeline.getTrainingEpoch(submodel) or epoch
        self.modelMigration.unifyModelWeights(allMetaModels, self.sharedModelWeights, unifiedLayerData)

        # Create a copy of the pipelines together
        allPipelines = allMetaModels + allModels
        # Save the current version of the model.
        self.modelMigration.saveModels(modelPipelines=allPipelines, modelName=modelName, datasetNames=allDatasetNames, sharedModelWeights=self.sharedModelWeights, submodelsSaving=self.submodelsSaving,
                                       submodel=submodel, trainingDate=trainingDate, numEpochs=numEpochs, metaTraining=True, saveModelAttributes=True, storeOptimizer=False)
