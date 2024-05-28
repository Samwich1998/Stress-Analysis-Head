# General
import numpy as np

# Import Files for extracting model information
from .Models.generalModels.matrixFactorization import matrixFactorization
from .modelSpecifications.compileModelInfo import compileModelInfo
from .Models.generalModels.generalModels import generalModel
from .Models.tensorFlow.neuralNetwork import neuralNetwork


class modelControl:
    
    def __init__(self, modelFile, modelTypes, allFeatureNames, saveDataFolder):
        # Store constant parameters.
        self.modelFile = modelFile
        self.allFeatureNames = allFeatureNames
        # Store variable parameters.
        self.modelTypes = modelTypes  # List of all model keys: example, "KNN".
        self.modelClasses = []  # A list of all model classes
        self.modelPaths = None
                
        # Create models.
        self.createModels(modelTypes, overwriteModel = False)
        
    def createModels(self, modelTypes, overwriteModel = False):
        assert len(self.modelTypes) == len(modelTypes), "Please re-instantiate the model controller if you wish to change the model number. This is mostly for user-awareness."
        # Reset the model parameters
        self.modelTypes = modelTypes
        self.modelPaths = self.getModelPaths(self.modelFile, modelTypes) # List of all paths for saving models.

        modelClasses = []
        # For each incoming model
        for modelInd in range(len(self.modelTypes)):
            modelType = self.modelTypes[modelInd]
            modelPath = self.modelPaths[modelInd]
                                    
            # Call the correct model class
            if modelType == "NN":
                modelClasses.append(neuralNetwork(modelPath, modelType, self.allFeatureNames, overwriteModel))
            elif modelType == "MF":
                modelClasses.append(matrixFactorization(modelPath, modelType, self.allFeatureNames, overwriteModel))
            else:
                modelClasses.append(generalModel(modelPath, modelType, self.allFeatureNames, overwriteModel))
            
            # Transfer any standardization info over.
            if len(self.modelClasses) == len(modelTypes) and len(self.modelClasses) != 0:
                modelClasses[-1].setStandardizationInfo(self.allFeatureNames, self.modelClasses[modelInd].standardizeClass_Features, 
                                                        self.modelClasses[modelInd].standardizeClass_Labels)
        self.modelClasses = modelClasses
    
    def resetModels(self):
        # For each given model.
        for modelInd in range(len(self.modelClasses)):
            self.modelClasses[modelInd].resetModel()
        
    @staticmethod
    def getModelPaths(modelFile, modelTypes):
        # Instantiate model info class
        compileModelInfoClass = compileModelInfo()
        # Extract all model paths as list: this order is DEFINED in _compileModelInfo
        modelPaths = compileModelInfoClass.compileModelPaths(modelFile, modelTypes)
        
        return modelPaths    

    def _saveModels(self):
        # For each incoming model
        for modelInd in range(len(self.modelClasses)):
            self.modelClasses[modelInd].saveModelInfo()
            
    @staticmethod
    def getSpecificFeatures(allFeatureNames, getFeatureNames, featureData):
        featureData = np.asarray(featureData)
        
        newFeatureData = []
        for featureName in getFeatureNames:
            featureInd = list(allFeatureNames).index(featureName)
            
            if len(newFeatureData) == 0:
                newFeatureData = featureData[:,featureInd]
            else:
                newFeatureData = np.dstack((newFeatureData, featureData[:,featureInd]))
        
        if len(newFeatureData) == 0:
            print("No Features grouped")
            return []
        return newFeatureData[0]
    
# ---------------------------------------------------------------------------#
