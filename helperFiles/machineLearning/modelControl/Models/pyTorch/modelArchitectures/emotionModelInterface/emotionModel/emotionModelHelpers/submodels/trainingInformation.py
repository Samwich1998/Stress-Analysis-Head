# General
import copy
from torch import nn


class trainingInformation(nn.Module):
    def __init__(self):
        super().__init__()
        # General model parameters.
        self.schedulerState = {}
        self.optimizerState = {}
        self.submodel = None

    def addSubmodel(self, submodel):
        # Set the submodel.
        self.submodel = submodel

        # If the submodel's information has never been set.
        if self.schedulerState.get(submodel, None) is None:
            self.schedulerState[submodel] = None
            self.optimizerState[submodel] = None

    def setSubmodelInfo(self, modelPipeline, submodelLoading):
        if submodelLoading is not None and self.optimizerState.get(submodelLoading, None) is not None:
            # Set the submodel.
            self.submodel = submodelLoading

            # Loading in the previous model's training state.
            modelPipeline.scheduler.load_state_dict(self.filterScheduler(modelPipeline, submodelLoading))
            modelPipeline.optimizer.load_state_dict(self.filterOptimizer(modelPipeline, submodelLoading))
        else:
            print("\t\tNot reading in training information")

    def storeOptimizer(self, optimizer, saveFullModel):
        if not saveFullModel:
            self.optimizerState[self.submodel] = None
            return None

        # Get the saved scheduler state.
        filteredOptimizerState = copy.deepcopy(optimizer.state_dict())

        # Do not save any model we are not/have trained.
        finalModelInd = self.getNumTrainedModels(self.submodel)
        filteredOptimizerState['param_groups'] = filteredOptimizerState['param_groups'][0:finalModelInd]

        # Save the optimizer's state dictionary
        self.optimizerState[self.submodel] = filteredOptimizerState

    def storeScheduler(self, scheduler, saveFullModel):
        if not saveFullModel:
            self.schedulerState[self.submodel] = None
            return None

        self.schedulerState[self.submodel] = scheduler.state_dict()

    def getOptimizer(self):
        return self.optimizerState[self.submodel]

    def getScheduler(self):
        return self.schedulerState[self.submodel]

    def filterOptimizer(self, modelPipeline, submodelLoading):
        # Get the saved state.
        optimizerState = self.getOptimizer()

        # Base case: we didn't save the optimizer
        if optimizerState is None:
            print("\tNo optimizer to overwrite.")
            return modelPipeline.optimizer.state_dict()
        
        # Initialize a blank state
        filteredOptimizerState = modelPipeline.optimizer.state_dict()
        finalModelInd = self.getNumTrainedModels(submodelLoading)
        assert finalModelInd <= len(optimizerState['param_groups'])
        
        # Do not load in any model we are not/have trained.
        filteredOptimizerState['param_groups'][0:finalModelInd] = optimizerState['param_groups'][0:finalModelInd]

        return filteredOptimizerState

    def filterScheduler(self, modelPipeline, submodelLoading):
        # Get the saved scheduler state.
        schedulerState = self.getScheduler()

        # Initialize a blank state
        filteredSchedulerState = modelPipeline.scheduler.state_dict()
        finalModelInd = self.getNumTrainedModels(submodelLoading)

        # Base case: we didn't save the scheduler
        if schedulerState is None:
            print("\tNo scheduler to overwrite.")
            return filteredSchedulerState

        # For each parameter to load.
        for key in filteredSchedulerState.keys():
            value = filteredSchedulerState[key]

            # Only change the value if its model-specific
            if not isinstance(value, list):
                continue


            # Update the state dictionary.
            filteredSchedulerState[key] = schedulerState[key][0:finalModelInd]

        return filteredSchedulerState

    @staticmethod
    def getNumTrainedModels(submodel):
        if submodel == "signalEncoder":
            return 1
        elif submodel == "autoencoder":
            return 2
        elif submodel == "decipherSignalMeaningModel":
            return 4
        elif submodel == "emotionPrediction":
            return 10
        else:
            assert False, "No model initialized"
