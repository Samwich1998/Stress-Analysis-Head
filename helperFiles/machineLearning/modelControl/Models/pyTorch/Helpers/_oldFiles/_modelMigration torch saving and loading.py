
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import dill

# PyTorch
import torch
import torch.nn as nn

# -------------------------------------------------------------------------- #
# -------------------------- Model Migration Class ------------------------- #

class _modelMigration:
    
    def __init__(self, accelerator = None):
        # Create folders to the save the data in.
        self.saveModelFolder = os.path.normpath(os.path.dirname(__file__) + "/../../../_finalModels/") + "/"
        self.saveModelFolder = os.path.relpath(os.path.normpath(self.saveModelFolder), os.getcwd()) + "/"
        os.makedirs(self.saveModelFolder, exist_ok=True) # Create the folders if they do not exist.
        
        # Specify the accelerator parameters.
        self.device = accelerator.device if accelerator else None
        self.accelerator = accelerator

        # Model identifiers.
        self.classAttibuteString = " Attributes"
        self.sharedWeightsName = "sharedData"
        
    # ---------------------------------------------------------------------- #
    # ------------------------- Specify the Device ------------------------- #
    
    def getModelDevice(self, accelerator = None):
        if accelerator:
            return accelerator.device
        
        else:
            # Find the pytorch device
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    # ---------------------------------------------------------------------- #
    # ------------------- Alter/Transfer Model Parameters ------------------ #
    
    def copyModelWeights(self, modelClass, sharedModelWeights):
        layerInfo = {}
        # For each parameter in the model
        for layerName, layerParams in modelClass.model.named_parameters():
            modelBlockType = layerName.split(".")[0]
            
            # If the layer should be saved.
            if modelBlockType in sharedModelWeights:
                # Save the layer (bias and weight indivisually)
                layerInfo[layerName] = layerParams.channelData.clone()
                
        return layerInfo
                
    def unifyModelWeights(self, allModels, sharedModelWeights, layerInfo):
        # For each model provided.
        for modelInd in range(len(allModels)):  
            pytorchModel = allModels[modelInd].model
            
            # For each parameter in the model
            for layerName, layerParams in pytorchModel.named_parameters():
                modelBlockType = layerName.split(".")[0]
                                
                # If the layer should be saved.
                if modelBlockType in sharedModelWeights:
                    assert layerName in layerInfo, print(layerName, layerInfo)
                    layerParams.channelData = layerInfo[layerName].clone()
                    
    def changeGradTracking(self, allModels, sharedModelWeights, requires_grad = False):
        # For each model provided.
        for modelInd in range(len(allModels)):  
            pytorchModel = allModels[modelInd].model
            
            # For each parameter in the model
            for layerName, layerParams in pytorchModel.named_parameters():
                modelBlockType = layerName.split(".")[0]
                
                # If the layer should be saved.
                if modelBlockType in sharedModelWeights:
                    layerParams.requires_grad = requires_grad
                    
    # ---------------------------------------------------------------------- #
    # ----------------------- Reset Model Parameters ----------------------- #
    
    def resetLayerStatistics(self, allModels):
        # For each model provided.
        for modelInd in range(len(allModels)):  
            pytorchModel = allModels[modelInd].model
            
            for module in pytorchModel.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    module.reset_running_stats()  # This will reset the running mean and variance
                    
    # ---------------------------------------------------------------------- #
    # ----------------- Common Model Saving/Loading Methods ---------------- #
    
    def _getAllModelChildren(self, model):
        # Iterate over all submodules using named_children() to include their names
        for name, submodule in model.named_children():
            print(name)
        
    def _filterStateDict(self, model, sharedModelWeights, submodelsSaving):
        # Initialize dictionaries to hold shared and specific parameters
        datasetSpecific_params = {}
        shared_params = {}
                
        # Iterate over all the parameters in the model's state_dict
        for name, param in model.state_dict().items():
            # Check if these weights are a part of a model we are saving.
            if not any(name.startswith(submodel) for submodel in submodelsSaving):
                continue
            
            # Check if the parameter name starts with any of the prefixes in sharedModelWeights
            if any(name.startswith(shared) for shared in sharedModelWeights):
                # If the parameter is shared, add it to the shared_params dictionary
                shared_params[name] = param
            else:
                # If the parameter is not shared, add it to the specific_params dictionary
                datasetSpecific_params[name] = param
                
        return shared_params, datasetSpecific_params
        
    def _filterClassAttributes(self, model, sharedModelWeights, submodelsSaving):
        # Initialize dictionaries to hold shared and specific attributes
        shared_attributes = {}
        dataset_specific_attributes = {}
    
        # Iterate over all submodules using named_children() to include their names
        for name, submodule in model.named_children():
            # Check if these weights are a part of a model we are saving.
            if not any(name.startswith(submodel) for submodel in submodelsSaving):
                continue
            
            # Check if the submodule name starts with any of the prefixes in sharedModelWeights
            if any(name.startswith(submodel) for submodel in sharedModelWeights):
                # If the submodule is shared, add its attributes to the shared_attributes dictionary
                shared_attributes[name] = self._removeBadAttributes(submodule.__dict__)
            else:
                # If the submodule is not shared, add its attributes to the specific_attributes dictionary
                dataset_specific_attributes[name] = self._removeBadAttributes(submodule.__dict__)
    
        return shared_attributes, dataset_specific_attributes
    
    def _removeBadAttributes(self, modelAttributes):
        newModelAttributes = {}
        
        # Iterate over all attributes in the model
        for attr_name, attr_value in modelAttributes.items():
            if not attr_name.startswith(("_")):
                newModelAttributes[attr_name] = attr_value
            if not attr_name.startswith(("module")):
                newModelAttributes[attr_name] = attr_value
        
        return newModelAttributes
        
    def _compileModelBaseName(self, modelName = "emotionModel", submodel = "autoencoder", datasetName = "sharedData",
                              trainingDate = "2023-11-22", numEpochs = 31, metaTraining = True):
        # Organize information about the model.
        trainingType = "metaTrainingModels" if metaTraining else "trainingModels"
        
        # Compile the location to save/load in the model.
        modelFilePath = self.saveModelFolder + f"{modelName}/{trainingType}/{submodel}/{trainingDate}/{datasetName}/Epoch {numEpochs}/"
                
        # Compile the filename to save/load in the model.
        modelBaseName = modelFilePath + f"{trainingDate} {datasetName} {submodel} at {numEpochs} Epochs"
        
        return modelBaseName
    
    def _createFolder(self, filePath):
        # Create the folders if they do not exist.
        os.makedirs(os.path.dirname(filePath), exist_ok=True) 
    
    # ---------------------------------------------------------------------- #
    # ------------------------ Saving Model Methods ------------------------ #
                    
    def saveModels(self, modelPipelines, modelName, datasetNames, sharedModelWeights, submodelsSaving,
                   submodel, trainingDate, numEpochs, metaTraining, saveModelAttributes = True):
        # Assert the integrity of the input variables.
        assert len(modelPipelines) == len(datasetNames), f"You provided {len(modelPipelines)} models to save, but only {len(datasetNames)} datasetNames."
        assert 0 < len(modelPipelines), "No models provided to save."
        subattributesSaving = submodelsSaving
        
        # For each model, save the shared and specific weights
        for datasetInd, (modelPipeline, datasetName) in enumerate(zip(modelPipelines, datasetNames)):

            # For non-shared models.
            if datasetInd == 1:
                # Don't save the shared weights again.
                submodelsSaving = [submodel for submodel in submodelsSaving if submodel not in sharedModelWeights]
                sharedModelWeights = [] 
            
            # Save the indivisual model's information.
            self._saveModel(modelPipeline.model, modelName, datasetName, sharedModelWeights, submodelsSaving, subattributesSaving,
                            submodel, trainingDate, numEpochs, metaTraining, saveModelAttributes)    
  
    def _saveModel(self, model, modelName, datasetName, sharedModelWeights, submodelsSaving, subattributesSaving,
                   submodel, trainingDate, numEpochs, metaTraining, saveModelAttributes = True):
        # Create a path to where we want to save the model.
        modelBaseName = self._compileModelBaseName(modelName, submodel, datasetName, trainingDate, numEpochs, metaTraining)
        sharedModelBaseName = self._compileModelBaseName(modelName, submodel, self.sharedWeightsName, trainingDate, numEpochs, metaTraining)
        
        # Filter the state_dict based on sharedModelWeights
        shared_params, specific_params = self._filterStateDict(model, sharedModelWeights, submodelsSaving)
        shared_attributes, specific_attributes = self._filterClassAttributes(model, [], subattributesSaving)

        # Save the pytorch models.
        self._savePyTorchModel(specific_params, modelBaseName, saveFullClass = False)       # Save dataset-specific parameters
        self._savePyTorchModel(shared_params, sharedModelBaseName, saveFullClass = False)   # Save shared parameters
        
        # If saving attributes.
        if saveModelAttributes:
            # Save the class attributes.
            self._saveModelAttributes(specific_attributes, modelBaseName) 
            self._saveModelAttributes(shared_attributes, sharedModelBaseName) 

    def _savePyTorchModel(self, model, modelBaseName, saveFullClass = False):
        # Prepare to save the model
        if not model: return None
        saveModelPath = modelBaseName + ".pth"
        
        # Create the folders if they do not exist.
        self._createFolder(modelBaseName)
        
        # Save the model
        if saveFullClass:
            # Note: You are saving the model with a PATH to the model class
            #       ANY directory restructure or code changes will break when loading.
            torch.save(model, saveModelPath)
        else:
            # Note: You are ONLY saving learnable parameters
            #       You ARE saving batch norm statistics such as running_mean (You may wanna check)
            self.accelerator.save_model(savingModel, saveModelPath)
            torch.save(model, saveModelPath)
            
    def _saveModelAttributes(self, attributes, modelBaseName):
        # Prepare to save the attributes
        if not attributes: return None

        # Create the folders if they do not exist.
        self._createFolder(modelBaseName)

        # Save the entire class along with its attributes.
        with open(modelBaseName + f"{self.classAttibuteString}.pkl", 'wb') as file:
            dill.dump(attributes, file)
            
    # ---------------------------------------------------------------------- #
    # ------------------------ Loading Model Methods ----------------------- #
    
    def loadModels(self, modelPipelines, modelName, datasetNames, sharedModelWeights, 
                   submodel, trainingDate, numEpochs, metaTraining, loadModelAttributes = True):
        # Assert the integrity of the input variables.
        assert len(modelPipelines) == len(datasetNames), f"You provided {len(modelPipelines)} models to load, but only {len(datasetNames)} datasetNames."
        # Update the user on the loading process.
        trainingType = "metaTrainingModels" if metaTraining else "trainingModels"
        print(f"Loading in previous {trainingType} weights and attributes")
        
        # Iterate over each model pipeline and dataset name
        for modelPipeline, datasetName in zip(modelPipelines, datasetNames):            
            # Save the indivisual model's information.
            self._loadModel(modelPipeline.model, modelName, datasetName, sharedModelWeights,
                            submodel, trainingDate, numEpochs, metaTraining, loadModelAttributes)  
            
    def _loadModel(self, model, modelName, datasetName, sharedModelWeights, submodel,
                   trainingDate, numEpochs, metaTraining, loadModelAttributes = True):
        # Construct base names for loading model and attributes
        modelBaseName = self._compileModelBaseName(modelName, submodel, datasetName, trainingDate, numEpochs, metaTraining)
        sharedModelBaseName = self._compileModelBaseName(modelName, submodel, self.sharedWeightsName, trainingDate, numEpochs, metaTraining)

        # Load in the pytorch models.
        self._loadPyTorchModel(model, modelBaseName)         # Load dataset-specific parameters
        self._loadPyTorchModel(model, sharedModelBaseName)   # Save shared parameters

        # If loading attributes.
        if loadModelAttributes:
            # Load the class attributes.
            self._loadModelAttributes(model, modelBaseName) 
            self._loadModelAttributes(model, sharedModelBaseName) 

    def _loadPyTorchModel(self, model, modelBaseName):
        # Prepare to save the attributes
        loadModelPath = modelBaseName + ".pth"
        
        # If the model exists.
        if model and os.path.exists(loadModelPath):
            model.eval()  # Set the model to evaluation mode
            
            # Load the state dict with 
            checkpoint = torch.load(loadModelPath, map_location=self.device)
            model.load_state_dict(checkpoint, strict=False) # strict=False to allow for loading only matching parts
            model.to(self.device)
        
    def _loadModelAttributes(self, model, modelBaseName):
        # Prepare to save the attributes
        loadModelPath = modelBaseName + f"{self.classAttibuteString}.pkl"

        # If the model exists.
        if model and os.path.exists(loadModelPath):
            # Load the class attributes.
            with open(loadModelPath, 'rb') as file:
                modelAttributes = dill.load(file)
                                
            # Iterate over all submodules using named_children() to include their names
            for name, submodule in model.named_children():
                # Check if the submodule name starts with any of the prefixes in sharedModelWeights
                if name in modelAttributes:
                    submodule.__dict__.update(modelAttributes[name])
                                                  
    def _updateModelInfo(self, model, modelBaseName):
        # Prepare to save the attributes
        loadModelPath = modelBaseName + f"{self.classAttibuteString}.pkl"
        
        # If the model exists.
        if model and os.path.exists(loadModelPath):
            # Load the class attributes.
            with open(loadModelPath, 'rb') as file:
                model.__dict__.update(dill.load(file)) 
                
    # ---------------------------------------------------------------------- #                    
# -------------------------------------------------------------------------- #


