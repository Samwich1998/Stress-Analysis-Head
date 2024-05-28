
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import os
import joblib
import torch
import numpy as np
# Abstract class
import abc

# -------------------------------------------------------------------------- #
# --------------------------- Global Model Class --------------------------- #

class globalModel(abc.ABC):
    
    def __init__(self, modelPath, modelType, allFeatureNames, overwriteModel = False):
        # Model information
        self.model = None
        self.modelType = modelType
        self.modelPath = modelPath
        self.overwriteModel = overwriteModel
                
        # Standardization information
        self.allFeatureNames = allFeatureNames
        self.finalFeatureNames = [] # The features the model was trained on. ONLY trained models will have finalFeatureNames
        self.standardizeClass_Labels = None
        self.standardizeClass_Features = None
        self.standardizationInfoPath = ".".join(modelPath.split(".")[0:-1]) + "_standardizationInfo.pkl"
        
        # Initialize model
        if not overwriteModel and os.path.exists(self.modelPath):
            # Load model if it exists
            self.loadModelInfo()
            self.verifyModelClass(allFeatureNames)
        else:
            # Create a new model
            self.createModel()

    def loadModelInfo(self):
        print(f"\tLoading Model {os.path.basename(self.modelPath)}")
        # Load in the model
        self._loadModel()
        # Load in standardization information
        with open(self.standardizationInfoPath, 'rb') as handle:
            standardizationInfo = joblib.load(handle, mmap_mode ='r')
            self.allFeatureNames, self.finalFeatureNames, self.standardizeClass_Labels, self.standardizeClass_Features = standardizationInfo
    
    def verifyModelClasses(self, featureNames):
        # If the model has been trained.
        if len(self.finalFeatureNames) != 0:
            # Assert the validity of each model
            assert len(featureNames) == len(self.allFeatureNames), print(len(featureNames), len(self.allFeatureNames))
            assert all(featureNames == self.allFeatureNames), "Feature names do not match from when building the model."

    def setStandardizationInfo(self, standardizedFeatureNames, standardizeClass_Features, standardizeClass_Labels):
        self.standardizeClass_Features = standardizeClass_Features
        self.standardizeClass_Labels = standardizeClass_Labels

        # print(self.allFeatureNames)
        assert all(self.allFeatureNames == standardizedFeatureNames)

    def saveModelInfo(self):
        """
        Save the standardization information as a pickle file

        Parameters
        ----------
        standardizationInfo : List of four elements: [all feature names, [final feature names], all feature standardization, [label standardizations]]
        """
        if len(self.finalFeatureNames) != 0:
            # Create folder to save the model
            modelPathFolder = os.path.dirname(self.modelPath)
            if len(modelPathFolder) != 0:
                os.makedirs(modelPathFolder, exist_ok=True)
            
            # Save the model
            self._saveModel()
            # Save the standardization methods/features
            standardizationInfo = self.allFeatureNames, self.finalFeatureNames, self.standardizeClass_Labels, self.standardizeClass_Features
            joblib.dump(standardizationInfo, self.standardizationInfoPath)
        else:
            print("\tModel not trained. Not saving model: " + os.path.basename(self.modelPath))
    
    def setupTraining(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames):
        # Assert the integrity of the input variables.
        assert len(Testing_Data) == len(Testing_Labels), "Testing points have to map 1:1 to testing labels."
        assert len(Training_Data) == len(Training_Labels), "Training points have to map 1:1 to training labels."
        assert len(featureNames) == len(Training_Data[0]) == len(Testing_Data[0]), "The featureNames should have the same length as all the features in testing/training sets."
        # Setup variables as numpy arrays
        if isinstance(Training_Data, (list, np.ndarray)):
            Training_Data = np.asarray(Training_Data.copy())
            Training_Labels = np.asarray(Training_Labels.copy())
            Testing_Data = np.asarray(Testing_Data.copy())
            Testing_Labels = np.asarray(Testing_Labels.copy())
        elif isinstance(Training_Data, (torch.Tensor)):
            Training_Data = Training_Data.clone().detach().numpy().copy()
            Training_Labels = Training_Labels.clone().detach().numpy().copy()
            Testing_Data = Testing_Data.clone().detach().numpy().copy()
            Testing_Labels = Testing_Labels.clone().detach().numpy().copy()
        # Save the information we trained on.
        self.finalFeatureNames = np.asarray(featureNames)
        # Start model from scratch
        self.resetModel()
        
        return Training_Data, Training_Labels, Testing_Data, Testing_Labels
    
    def getSpecificFeatures(self, allFeatureNames, getFeatureNames, featureData):
        featureData = np.array(featureData)
        
        newfeatureData = []
        for featureName in getFeatureNames:
            featureInd = list(allFeatureNames).index(featureName)
            
            if len(newfeatureData) == 0:
                newfeatureData = featureData[:,featureInd]
            else:
                newfeatureData = np.dstack((newfeatureData, featureData[:,featureInd]))
        
        if len(newfeatureData) == 0:
            print("No Features grouped")
            return []
        return newfeatureData[0]
        
    # ------------------------ Child Class Contract ------------------------ #
    
    @abc.abstractmethod
    def createModel(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")  
        
    @abc.abstractmethod
    def resetModel(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child") 
        
    @abc.abstractmethod
    def _loadModel(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")        

    @abc.abstractmethod
    def _saveModel(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")  

    @abc.abstractmethod
    def trainModel(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")  
        
    @abc.abstractmethod
    def scoreModel(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child") 
        
    @abc.abstractmethod
    def predict(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")
    
    # ---------------------------------------------------------------------- #


    
# -------------------------------------------------------------------------- #
