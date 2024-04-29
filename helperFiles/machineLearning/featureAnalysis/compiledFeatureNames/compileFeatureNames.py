
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import numpy as np
import sys

# Import interfaces for reading/writing data
sys.path.append(os.path.dirname(__file__) + "/../../../dataAcquisitionAndAnalysis/excelProcessing/")
import extractDataProtocols

# -------------------------------------------------------------------------- #
# -------------------------- Compile Feature Names ------------------------- #

class compileFeatureNames:
    
    def __init__(self):
        # Store Extracted Features
        self.possibleFeatureNames = np.char.lower(["eog", "eeg", "ecg", "eda", "emg", "temp", "lowFreq", "highFreq"])
        self.featureNamesFolder = os.path.dirname(__file__) + "/All Features/"
        self.extractDataInterface = extractDataProtocols.extractData()
        
    def extractFeatureNames(self, extractFeaturesFrom):
        extractFeaturesFrom = [featureType.lower() for featureType in extractFeaturesFrom]
        # Compile feature neames
        featureNames = []; biomarkerOrder = []
        biomarkerFeatureNames = []
        
        # For each feature we are processing.
        for featureName in extractFeaturesFrom:
            # Assert that we have a protocol for this feature.
            assert featureName in self.possibleFeatureNames, f"Unknown feature neam: {featureName}"
            
            # Organize and store the feature name information.
            self.getFeatures(featureName, biomarkerFeatureNames, featureNames, biomarkerOrder)
       
        return np.array(featureNames, dtype=str), biomarkerFeatureNames, biomarkerOrder       
        
    def getFeatures(self, featureName, biomarkerFeatureNames, featureNames, biomarkerOrder):
        # Specify the Paths to the EOG Feature Names
        featuresFile = self.featureNamesFolder + f"{featureName}FeatureNames.txt"
        # Extract the EOG Feature Names we are Using
        currentFeatureNames = self.extractDataInterface.extractFeatureNames(featuresFile, prependedString = "finalFeatures.extend([", appendToName = f"_{featureName.upper()}")
        # Create Data Structure to Hold the Features
        biomarkerFeatureNames.append(currentFeatureNames)
        featureNames.extend(currentFeatureNames)
        biomarkerOrder.append(featureName)
        