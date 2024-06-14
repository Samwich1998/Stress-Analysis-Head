import numpy as np
import os

# Import Files
from helperFiles.machineLearning.featureAnalysis.compiledFeatureNames.compileFeatureNames import compileFeatureNames
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo


class humanMachineInterface:
    
    def __init__(self, modelClasses, actionControl, extractFeaturesFrom):
        # General parameters.
        self.actionControl = actionControl
        self.modelClasses = modelClasses        # A list of machine learning models.

        # Initialize helper classes.
        self.compileFeatureNames = compileFeatureNames()  # Initialize the Feature Information
        self.compileModelInfo = compileModelInfo()  # Initialize the Model Information

        # Compile the feature information.
        self.featureNames, self.biomarkerFeatureNames, self.biomarkerFeatureOrder = self.compileFeatureNames.extractFeatureNames(extractFeaturesFrom)

        # Holder parameters.
        self.alignedFeatureLabels = None  # The FINAL predicted labels at the current timepoint.
        self.alignedFeatureTimes = None   # The interpolated timepoints of the ALIGNED feature.
        self.alignedFeatures = None       # Linearly interpolated features to align all at the same timepoint.
        self.alignedUserNames = []
        self.alignedItemNames = []
        self.userName = None
                
        # Initialize mutable variables.
        self.resetVariables_HMI()
        
    def resetVariables_HMI(self):        
        # Aligned feature data structure
        self.alignedFeatureLabels = [[] for _ in range(len(self.modelClasses))]  # The FINAL predicted labels at the current timepoint.
        self.alignedFeatureTimes = []   # The interpolated timepoints of the ALIGNED feature. Dimensions: [numTimepoints]
        self.alignedFeatures = [[] for _ in range(len(self.featureNames))]       # Interpolated features to align all at the same timepoint. Dimensions: [numFeatures, numTimepoints]

        # Subject information
        self.alignedUserNames = []
        self.alignedItemNames = []
        self.userName = None
        
    def setUserName(self, filePath):
        # Get user information
        fileName = os.path.basename(filePath).split(".")[0]
        self.userName = fileName.split(" ")[-1].lower()

    # Deprecated
    def predictLabels(self): 
        # Find the new final features, where no label has been predicted yet
        allNewFeaturesTimes = self.alignedFeatureTimes[len(self.alignedFeatureLabels[0]):].copy()
        allNewFeatures = self.alignedFeatures[:, len(self.alignedFeatureLabels[0]):].copy()

        # Find new subject information
        newItemNames = self.alignedItemNames[len(self.alignedFeatureLabels[0]):].copy()
        
        # For each prediction model
        for modelInd in range(len(self.modelClasses)):
            # If the model was never trained, don't use it.
            if len(self.modelClasses[modelInd].finalFeatureNames) == 0:
                continue
            
            modelClass = self.modelClasses[modelInd]  
            # Standardize the incoming features (if the model requires)
            standardizedFeatures = modelClass.standardizeClass_Features.standardize(allNewFeatures) if modelClass.standardizeClass_Features else allNewFeatures
            # Select the model features
            newFinalFeatures = modelClass.getSpecificFeatures(modelClass.allFeatureNames, modelClass.finalFeatureNames, standardizedFeatures)
            
            # Predict the final labels
            if modelClass.modelType == "MF":
                standardizedPredictions = modelClass.model.predict(newFinalFeatures, allNewFeaturesTimes, newUserNames, newItemNames)
            else:
                standardizedPredictions = modelClass.predict(newFinalFeatures)
            # Rescale up the labels (if the model requires)
            predictedLabels = modelClass.standardizeClass_Labels.unStandardize(standardizedPredictions) if modelClass.standardizeClass_Labels else standardizedPredictions
            
            # Save the final results
            self.alignedFeatureLabels[modelInd].extend(predictedLabels[-len(predictedLabels):])

        return self.alignedFeatureLabels
