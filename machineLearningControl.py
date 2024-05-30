"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Data Aquisition:
        
    Plotting:
        If Plotting, You Need an GUI Backend -> In Spyder IDE Use: %matplotlib qt5
        Some IDEs (Spyder Included) may Naturally Plot in GUI.
    --------------------------------------------------------------------------
    
    Modules to Import Before Running the Program (Some May be Missing):
        pip install -U numpy pandas scipy matplotlib seaborn natsort
        pip install -U openpyxl pyserial joblib pyexcel pyfirmata2
        pip install -U tensorflow keras torch torchvision scikit-learn 
        pip install -U shap xgboost sklearn lightgbm torchsummary
        pip install -U neurokit2 antropy pyeeg eeglib librosa
    
    Programs to Install:
        Vizard (If using Virtual Reality): https://www.worldviz.com/virtual-reality-software-downloads
        
    --------------------------------------------------------------------------
"""

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import sys
import numpy as np

# Import data aquisition and analysis files
sys.path.append(os.path.dirname(__file__) + "/Helper Files/Data Aquisition and Analysis/") 
import streamingProtocols      # Functions to Handle Data from Arduino

# Import Files for extracting feature names
sys.path.append(os.path.dirname(__file__) + "/Helper Files/Machine Learning/Feature Analysis/_compiledFeatureNames/") 
import _compileFeatureNames  # Functions to extract feature names

# Import Files for Machine Learning
sys.path.append(os.path.dirname(__file__) + "/Helper Files/Machine Learning/") 
import machineLearningInterface  # Class Header for All Machine Learning
import trainingProtocols     # Functions to Save/Read in Data from Excel
from _dataPreparation import standardizeData

# Import Files for Stress Therapy
sys.path.append(os.path.dirname(__file__) + "/Helper Files/Machine Learning/Feedback Control/Music Therapy/") 
# import musicTherapy

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # User options during the run: any number can be true.
    plotStreamedData = False        # Graph the Data to Show Incoming Signals + Analysis.
    useModelPredictions = False    # Apply the Learning Algorithm to Decode the Signals.
    
    # ---------------------------------------------------------------------- #
    
    # Analyze the data in batches.
    numPointsPerBatch = 2000        # The Number of Data Points to Display to the User at a Time.
    moveDataFinger = 200            # The Minimum Number of NEW Data Points to Plot/Analyze in Each Batch;
    
    # Specify biomarker information.
    streamingOrder = ["eog", "eeg", "eda", "temp"]  # A List Representing the Order of the Sensors being Streamed in.
    extractFeaturesFrom = ["eog", "eeg", "eda", "temp"] # A list with all the biomarkers from streamingOrder for feature extraction
    allAverageIntervals = [120, 75, 90, 45] # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60
    allAverageIntervals = [60, 30, 30, 30] # EOG: 120-180; EEG: 60-90; EDA: ?; Temp: 30 - 60
        
    # Compile feature names
    featureNames, biomarkerFeatureNames, biomarkerOrder = _compileFeatureNames.compileFeatureNames().extractFeatureNames(extractFeaturesFrom)
    
    featureAverageWindows = []
    # Compile feature average windows.
    for biomarker in biomarkerOrder:
        featureAverageWindows.append(allAverageIntervals[streamingOrder.index(biomarker)])
        
    # ML Flags
    actionControl = None        # NOT IMPLEMENTED YET
    reanalyzeData = True       # Reanalyze training files: don't use saved features
    plotTrainingData = False    # Plot all training information
    # If training, read the data as quickly as possible
    
    # Specify the machine learning information
    modelFile = "predictionModel.pkl" # Path to Model (Creates New if it Doesn't Exist)
    modelTypes = ["MF", "MF", "MF"]  # Model Options: linReg, logReg, ridgeReg, elasticNet, SVR_linear, SVR_poly, SVR_rbf, SVR_sigmoid, SVR_precomputed, SVC_linear, SVC_poly, SVC_rbf, SVC_sigmoid, SVC_precomputed, KNN, RF, ADA, XGB, XGB_Reg, lightGBM_Reg

    # If not streaming real-time
    numPointsPerBatch = 2048576
    moveDataFinger = 1048100 
                
    saveModel = True  # Save the Machine Learning Model for Later Use
    trainingFolder = "./Data/allSensors/_finalDataset/"   # Data Folder to Save the Excel Data; MUST END IN '/'

    # Get the Machine Learning Module
    performMachineLearning = machineLearningInterface.machineLearningInterface(modelTypes, modelFile, featureNames, trainingFolder)
    modelClasses = performMachineLearning.modelControl.modelClasses
    
    if True or useModelPredictions:
        # Specify the MTG-Jamendo dataset path
        soundInfoFile = 'raw_30s_cleantags_50artists.tsv'
        dataFolder = './Helper Files/Machine Learning/_Feedback Control/Music Therapy/Organized Sounds/MTG-Jamendo/'
        # Initialize the classes
        # soundManager = musicTherapy.soundController(dataFolder, soundInfoFile)  # Controls the music playing
        # soundManager.loadSound(soundManager.soundInfo[0][3])
        playGenres = [None, 'pop', 'jazz', 'heavymetal', 'classical', None]
        
        playGenres = [None, 'hiphop', 'blues', 'disco', 'ethno', None]
        
        playGenres = [None, 'funk', 'reggae', 'rap', 'classicrock', None]
        
        # playGenres = [None, 'hiphop', 'blues', 'hardrock', 'african', None]
        # soundManager.pickSoundFromGenres(playGenres)
        
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #

    # Initialize instance to analyze the data
    readData = streamingProtocols.streamingProtocols(None, modelClasses, actionControl, numPointsPerBatch, moveDataFinger, 
                                                     streamingOrder, biomarkerOrder, featureAverageWindows, plotStreamedData)
    
    # Take Preprocessed (Saved) Features from Excel Sheet
    trainingInterface = trainingProtocols.trainingProtocols(biomarkerFeatureNames, streamingOrder, biomarkerOrder, len(streamingOrder), trainingFolder, readData)

    checkFeatureWindow_EEG = False
    if checkFeatureWindow_EEG:
        featureTimeWindows = np.arange(5, 25, 5)
        # # featureTimeWindows = [5, 30, 60, 90, 120, 150, 180]
        excelFile = trainingFolder + '2022-12-16 Full Dataset TV.xlsx'
        allRawFeatureTimesHolders, allRawFeatureHolders = trainingInterface.varyAnalysisParam(excelFile, featureAverageWindows, featureTimeWindows)
            
    # Extract the features from the training files and organize them.
    allRawFeatureTimesHolders, allRawFeatureHolders, allRawFeatureIntervals, allRawFeatureIntervalTimes, \
        allAlignedFeatureTimes, allAlignedFeatureHolder, allAlignedFeatureIntervals, allAlignedFeatureIntervalTimes, \
            subjectOrder, experimentOrder, allFinalFeatures, allFinalLabels, featureLabelTypes, surveyQuestions, surveyAnswersList, surveyAnswerTimes \
                = trainingInterface.streamTrainingData(featureAverageWindows, plotTrainingData = plotTrainingData, reanalyzeData = reanalyzeData)    
    # Assert the validity of the feature extraction
    assert len(allAlignedFeatureHolder[0][0]) == len(featureNames), "Incorrect number of compiled features extracted"
    for analysisInd in range(len(allRawFeatureHolders[0])):
        assert len(allRawFeatureHolders[0][analysisInd][0]) == len(biomarkerFeatureNames[analysisInd]), "Incorrect number of fraw eatures extracted"
    print("\nFinished Feature Extraction")

    # Standardize data
    standardizeClass_Features = standardizeData(allFinalFeatures, threshold = 0)
    standardizedFeatures = standardizeClass_Features.standardize(allFinalFeatures)
    # Standardize labels
    standardizeClass_Labels = []; standardizedLabels = []; scoreTransformations = []
    for modelInd in range(len(performMachineLearning.modelControl.modelClasses)):
        if modelInd == 2:
            standardizeClass_Labels.append(standardizeData(allFinalLabels[modelInd], threshold = 0))
            standardizedLabels.append(standardizeClass_Labels[modelInd].standardize(allFinalLabels[modelInd]))
            
            scoreTransformation = np.diff(standardizedLabels[modelInd])/np.diff(allFinalLabels[modelInd])
            scoreTransformations.append(scoreTransformation[~np.isnan(scoreTransformation)][0])
        else:
            oddLabels = allFinalLabels[modelInd] #+ (np.mod(allFinalLabels[modelInd],2)==0)
            standardizeClass_Labels.append(standardizeData(oddLabels, threshold = 0))
            standardizedLabels.append(standardizeClass_Labels[modelInd].standardize(oddLabels))
            
            scoreTransformation = np.diff(standardizedLabels[modelInd])/np.diff(oddLabels)
            scoreTransformations.append(scoreTransformation[~np.isnan(scoreTransformation)][0])
        
        # Compile information into the model class
        performMachineLearning.modelControl.modelClasses[modelInd].setStandardizationInfo(featureNames, standardizeClass_Features, standardizeClass_Labels[modelInd])
    standardizedLabels = np.array(standardizedLabels)
    
    
    
    userNames = np.unique([i.split(" ")[-1].lower() for i in subjectOrder])
    activityNames = np.array(["Baseline", "Music", "CPT", "Exercise", "VR"])
            
    import itertools
    flattenedTimes = list(itertools.chain(*surveyAnswerTimes))

    newLabels = [[], [], []]
    for modelInd in range(3):
        for labelInd in range(len(standardizedLabels[modelInd])):
            userItemRating = standardizedLabels[modelInd][labelInd]
            experimentName = experimentOrder[labelInd]
            experimentLabel = subjectOrder[labelInd]
            timePoint = flattenedTimes[labelInd]
            
            # itemName = max(activityNames, key = lambda itemName: itemName in experimentLabel)
            itemName = experimentName
            if experimentName.isdigit():
                itemName = "Music"
            elif "Recovery" == experimentName:
                itemName = "Baseline"
            elif "VR" in itemName.split(" - "):
                itemName = "VR"
            elif itemName not in activityNames:
                print(itemName)
            itemInd = int(np.where(activityNames == itemName)[0][0])

            userName = experimentLabel.split(" ")[-1].lower()
            userInd = int(np.where(userNames == userName)[0][0])
            
            newLabels[modelInd].append([timePoint, userInd, itemInd, userItemRating])
    newLabels = np.array(newLabels)
    activityLabels = newLabels[2, :, 2]
    
    # Remove recovery
   # standardizedFeatures = standardizedFeatures[activityLabels != 5]
   # activityLabels = activityLabels[activityLabels != 5]
    
    
    
    sys.exit()    
    
    
    modelInd = 2
    modelTypes = ["NN", "lightGBM_Reg", "SVC_rbf"]  # Model Options: linReg, logReg, ridgeReg, elasticNet, SVR_linear, SVR_poly, SVR_rbf, SVR_sigmoid, SVR_precomputed, SVC_linear, SVC_poly, SVC_rbf, SVC_sigmoid, SVC_precomputed, KNN, RF, ADA, XGB, XGB_Reg, lightGBM_Reg
    performMachineLearning.createModels(modelTypes)
    featureNamesList = [featureNames]
    featureNamesListOrder = ["All"]
    
    featureLabels = activityLabels


    import matplotlib.pyplot as plt
    # For each group of features
    for currentFeatureNamesInd in range(len(featureNamesList)):
        featureType = featureNamesListOrder[currentFeatureNamesInd]
        currentFeatureNames = np.array(featureNamesList[currentFeatureNamesInd])
        
        # Define saving parameters
        saveFolder = trainingFolder + featureType + " Feature Combinations/"
        saveExcelName = featureType + " Feature Combinations.xlsx"
        # Group all the relevant features for this model
        standardizedFeatures_Cull = performMachineLearning.modelControl.getSpecificFeatures(featureNames, currentFeatureNames, standardizedFeatures)
        
        # thresholds = [0.35, 0.51, 0.6, 0.7, 0.8, 0.9, 0.95] KNN
        # thresholds = [0.45, 0.51, 0.6, 0.7, 0.8, 0.9, 0.95] SVC_rbf
        # thresholds = [0.45, 0.51, 0.6, 0.7, 0.8, 0.9, 0.95] SVC_poly
        
        
        thresholds = [0.44, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        numFeaturesCombineList = [1,2,3,4,5,6,7, 0.8]
        # Fit model to all feature combinations
        for combinationInd in range(len(numFeaturesCombineList)):
            numFeatures_perCombination = numFeaturesCombineList[combinationInd]
            print(saveExcelName, numFeatures_perCombination)
            modelScores, modelSTDs, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(modelInd, standardizedFeatures, featureLabels, currentFeatureNames, numFeatures_perCombination, numEpochs = 20, numModelsTrack = 500, 
                                                                                                                  saveData = True, imbalancedData = False, saveExcelName = "Feature Combination Accuracy.xlsx", printUpdateAfterTrial = 2000)
            
            plt.plot(modelScores); plt.show()
            # sys.exit()
            # Only use features that have some correlation
            if combinationInd < len(thresholds):
                currentFeatureNames = featureNames_Combinations[np.array(modelScores) >= thresholds[combinationInd]] 
                currentFeatureNames = np.asarray([currentFeatureName.split(" ") for currentFeatureName in currentFeatureNames])
                currentFeatureNames = np.unique(currentFeatureNames.flatten())
                standardizedFeatures_Cull = performMachineLearning.modelControl.getSpecificFeatures(featureNames, currentFeatureNames, standardizedFeatures)

    
    

