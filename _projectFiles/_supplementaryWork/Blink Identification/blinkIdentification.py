"""
    Written by Samuel Solomon
        
"""

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import sys
import collections
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(__file__) + "/../")
import trainingProtocols_Supp       # Functions to Save/Read in Data from Excel

sys.path.append(os.path.dirname(__file__) + "/../../../helperFiles/dataAcquisitionAndAnalysis/") 
import streamingProtocols      # Functions to Handle Data from Arduino

# Import Files for Machine Learning
sys.path.append(os.path.dirname(__file__) + "/../../../helperFiles/machineLearning/featureAnalysis/compiledFeatureNames/")
import compileFeatureNames  # Functions to extract feature names

sys.path.append(os.path.dirname(__file__) + "/../../../helperFiles/machineLearning/") 
import machineLearningInterface
from dataInterface.dataPreparation import standardizeData

import featureSelection
import plottingPipeline

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #
    featureLabelOptions = ["Blink", "Movement", "Wire"] # We only care about Blink and Wire
    
    # We only care about EOG data
    streamingOrder = ["eog"]  # A List Representing the Order of the Sensors being Streamed in.
    extractFeaturesFrom = ["eog"] # A list with all the biomarkers from streamingOrder for feature extraction

    # Compile Feature Names
    featureNamesFolder = "../../Helper Files/Machine Learning/_Compiled Feature Names/All Features/"
    featureNames, indivisualFeatureNames, biomarkerOrder, = compileFeatureNames.compileFeatureNames().extractFeatureNames(extractFeaturesFrom)
    eogFeatureNames = indivisualFeatureNames[0]
    
    dataExcelFolder = os.path.dirname(__file__) + "/Data/"
        
    # Train or Test the Data with the Machine Learning Model
    # plotFeatures = True   # Plot all training feature information
    
    # Reading/processing new data
    reanalyzeData = False  # Reanalyze Data from Scratch (Don't Use Saved Features)
    plotDistributions = False # Plot feature distributions (histograms)
    
    # ML model to select features
    preselectedFeatures = None # If None, model selects features, rather than pre-determined
    # Example of list of preselected features
    # preselectedFeatures = ['accel1_Accel2_Ratio_EOG', 'closingSlope_MaxVel_EOG', 'eyesClosedTime_EOG', 'blinkDuration_EOG', 'accelFullEntropy_EOG']
    
    # If doing machine learning, specify:
    modelTypes = ["KNN", "KNN", "KNN"] # What type of model to use
    numFeaturesCombine = 3 # How many features to select
    
    # Comparing performance of different types of models
    errorBars = True # Learning curves for model includes error bars
    testModels = None
    # Example list of models
    # testModels = ["KNN", "SVC_rbf", "SVC_rbf_1", "SVC_rbf_2", "SVC_rbf_4", "SVC_rbf_5", "RF", "ADA"]
    
    # Finding optimal bounds for selected features
    preselectedcullInfo = None # If none, uses grid search to optimize feature boundings
    
    # Example cullInfos
    # Previous/Example Machine Learning Attempt
    # preselectedcullInfo = [('accel1_Accel2_Ratio_EOG', ['>'], [-226.08898051797456]), ('closingSlope_MaxVel_EOG', ['>', '<'],
    #        [13.863705691193731, 27.503764452400503]), ('eyesClosedTime_EOG', ['<'], [0.2365096000000122]), ('blinkDuration_EOG', ['>', '<'],
    #        [0.10229590503286148, 0.3931081285961759]), ('accelFullEntropy_EOG', ['>', '<'],
    #        [3.469075904236875, 6.482192654566428])]
       
    # Original pipeline without Machine Learning
    # preselectedcullInfo = [("blinkDuration_EOG", [">", "<"], [0.008, 0.5]), ("tentDeviationX_EOG", [">"], [-0.2]), 
    #             ("tentDeviationY_EOG", [">", "<"], [-0.2, 0.6]), ("closingTime_Peak_EOG", [">", "<"], [0.04, 0.3]),
    #             ("openingTime_Peak_EOG", [">", "<"], [0.04, 0.4]), ("velRatio_EOG", ["<"], [-1]), ("peakSkew_EOG", [">"], [-0.75])]

    # Plot Optimized Culling Pipeline
    plotPipeline = True
    
    # Read the data as quickly as possible
    numPointsPerBatch = 2048576
    moveDataFinger = 1048100 

    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    #           Data Collection Program (Should Not Have to Edit)            #
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # Initialize instance to analyze the data
    readData = streamingProtocols.streamingProtocols(None, [], None, numPointsPerBatch, moveDataFinger, streamingOrder, biomarkerOrder, [0], False)

    # Extract the features from the training files and organize them.
    trainingClass = trainingProtocols_Supp.trainingProtocols(indivisualFeatureNames, biomarkerOrder, 1, dataExcelFolder, readData)
    subjectOrder, allFinalFeatures, allFinalLabels = trainingClass.streamBlinkData(featureLabelOptions, reanalyzeData)        
    breakpoint()
    # NOT INCLUDING MOVEMENT LABEL
    allFinalFeatures = allFinalFeatures[allFinalLabels != 1]
    allFinalLabels = allFinalLabels[allFinalLabels != 1]
    
    # finished feature extraction
    print("\nFinished Feature Extraction")
    
    allFinalLabelsCull = allFinalLabels.copy()
    allFinalFeaturesCull = allFinalFeatures.copy()
    
    # sys.exit()
    # ---------------------------------------------------------------------- #
    # PLOTTING FEATURE DISTRIBUTIONS --------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    plot = plottingPipeline.plotting(allFinalFeaturesCull, allFinalLabelsCull, featureLabelOptions, featureNames)
    
    if plotDistributions:
        plot.featureDistributions()
            
    # ---------------------------------------------------------------------- #
    # ML TO SELECT BEST COMBINATION OF FEATURES ---------------------------- #
    # ---------------------------------------------------------------------- #
    
    """
    NOTES:
        - Model selects the best combination of numFeaturesCombine number of features
            out of all the possibleFeatures specified.
        - Currently using a KNN, however through plotting learning curves, I found that
            SVM classifier (RBF kernel, 3rd degree) has the smallest gap between its
            training and testing curve, which may help if it is overfitting. I'm
            using KNN right now just because it runs faster, and I think the overfitting
            problem improved after adding more blink data.
        - The performance of each model is evaluated based on accuracy. I have a flag
            where you can indicate if the data is imbalanced or not. If the data is imbalanced,
            I use an sklearn function, balanced_accuracy_score, which I think takes
            the average recall of each class in order to account for imbalanced data. I
            am not super sure if this is what we are looking for, but it did seem to
            help when I implemented it. We definitly need to account for the data imbalance
            (right now there are 4866 blinks and 1056 wire), but not sure if this
            was the correct approach. There is a method I wrote within featureSelection's
            scoring class that returns the f1 score instead, which may be a better
            way to score a model's performance.
    """
    
    # Helper Functions + Setup --------------------------------------------- #
    
    modelFile = "KNN_BlinkIdentification.pkl" # Path to Model (Creates New if it Doesn't Exist)
    modelInd = 0
    
    saveModel = True  # Save the Machine Learning Model for Later Use
    trainingFolder = "./"   # Data Folder to Save the Excel Data; MUST END IN '/'
    # os.makedirs(trainingFolder, exist_ok = True)
    
    performMachineLearning = machineLearningInterface.machineLearningHead(modelTypes, modelFile, featureNames, trainingFolder)
    modelClasses = performMachineLearning.modelControl.modelClasses
    
    performMachineLearning.createModels(modelTypes)
    featureNames = [item for sublist in indivisualFeatureNames for item in sublist]
    # Standardize features
    standardizeClass_Features = standardizeData(allFinalFeatures)
    standardizedFeatures = standardizeClass_Features.standardize(allFinalFeatures)

    # Compile information into the model class
    modelClasses[modelInd].setStandardizationInfo(featureNames, standardizeClass_Features, allFinalLabels)

    # Define saving parameters
    saveFolder = trainingFolder + "Feature Combinations/"
    saveExcelName =  "Feature Combinations.xlsx"
    # Group all the relevant features for this model
    standardizedFeatures_Cull = standardizedFeatures
    currentFeatureNames = featureNames
    
    selectedFeatures = []
    
    # Initialize class from featureSelection.py
    scoring = featureSelection.scoring(featureNames)
    
    # ---------------------------------------------------------------------- #
    
    if preselectedFeatures != None:
        # Define selected Features here
        selectedFeatures = preselectedFeatures
    else:
        
        # List of possible features we want to consider selecting
        
        # This includes all features that are not y-dependent. (Not affected by normalizing peak height)
        nonAmplitudeFeatures = list(set(featureNames) - set(["peakHeight_EOG", "tentDeviationY_EOG", 'tentDeviationRatio_EOG', 'openingAmpVel_Loc_EOG', 
                             'maxClosingAccel_Loc_EOG', 'maxClosingVel_Loc_EOG', 'minBlinkAccel_Loc_EOG',
                             'openingAmpVel_Loc_EOG', 'maxOpeningAccel_firstHalfLoc_EOG', 'maxOpeningAccel_secondHalfLoc_EOG', 
                             'closingAmpSegment1_EOG', 'closingAmpSegment2_EOG', 'closingAmpSegmentFull_EOG', 
                             'openingAmpSegment1_EOG', 'openingAmpSegment2_EOG', 'openingAmpSegmentFull_EOG', 
                             'velocityAmpInterval_EOG', 'accelAmpInterval1_EOG', 'accelAmpInterval2_EOG', 
                             'blinkIntegral_EOG', 'portion1Integral_EOG', 'portion2Integral_EOG', 'portion3Integral_EOG', 
                             'portion4Integral_EOG', 'portion5Integral_EOG', 'portion6Integral_EOG', 'portion7Integral_EOG', 
                             'portion8Integral_EOG', 'velToVelIntegral_EOG', 'closingIntegral_EOG', 'openingIntegral_EOG', 
                             'closingSlopeIntegral_EOG', 'accel12Integral_EOG', 'openingAccelIntegral_EOG',
                             'condensedIntegral_EOG', 'peakToVel0Integral_EOG', 'peakToVel1Integral_EOG', 'peakAverage_EOG',
                             'peakSTD_EOG', 'peakCurvature_EOG', 'curvatureYDataAccel0_EOG', 'curvatureYDataAccel1_EOG',
                             'curvatureYDataAccel2_EOG', 'curvatureYDataAccel3_EOG', 'curvatureYDataVel0_EOG', 'curvatureYDataVel1_EOG', 
                             'velFullSTD_EOG', 'accelFullSTD_EOG', 'thirdDerivFullSTD_EOG', 'openingAmplitudeFull_EOG']))
        
        # This includes features that have been cited in previous literature
        goodFeatures = ["blinkDuration_EOG", "tentDeviationX_EOG", "closingTime_Peak_EOG", "openingTime_Peak_EOG", 
                        "eyesClosedTime_EOG", "accelToPeak_EOG", "peakToAccel_EOG", "velocityPeakInterval_EOG", "velToPeak_EOG", "peakToVel_EOG", "closingSlope_MaxAccel_EOG",
                        "closingSlope_MaxVel_EOG", "closingSlope_MinAccel_EOG", "openingSlope_MinVel_EOG", "closingAccel_MaxAccel_EOG", 
                        "closingAccel_MinAccel_EOG",
                        "accel0_Vel0_Ratio_EOG", "accel0_Accel1_Ratio_EOG", "accel0_Accel2_Ratio_EOG", 
                        "velRatio_EOG", "peakEntropy_EOG", "velFullEntropy_EOG", "accelFullEntropy_EOG"]
        
        # Features to debug with
        testFeatures = ["blinkDuration_EOG", "tentDeviationX_EOG", "closingTime_Peak_EOG", "openingTime_Peak_EOG", 
                        "eyesClosedTime_EOG"]
        
        possibleFeatures = list(set(goodFeatures))
        
        # Get data for all features we want to consider
        possibleFeatures_Cull = performMachineLearning.modelControl.getSpecificFeatures(featureNames, possibleFeatures, standardizedFeatures)
        # Returns performance for each combination of features
        # modelScores, modelSTDs, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(modelInd, possibleFeatures_Cull, allFinalLabels, possibleFeatures, numFeaturesCombine, subjectOrder, standardizedFeatures, saveData = True, saveExcelName = saveExcelName, printUpdateAfterTrial = 500, categorical = {0: featureLabelOptions[0], 2: featureLabelOptions[2]}, imbalancedData = True)
        #     def analyzeFeatureCombinations(self, modelInd, featureData, featureLabels, featureNames, numFeatures_perCombination, numEpochs = 10, numModelsTrack = 500,
        #                           saveData = True, imbalancedData = False, saveExcelName = "Feature Combination Accuracy.xlsx", printUpdateAfterTrial = 2000):
        modelScores, modelSTDs, featureNames_Combinations = performMachineLearning.analyzeFeatureCombinations(modelInd, possibleFeatures_Cull, allFinalLabels, possibleFeatures, numFeaturesCombine, saveData = True, saveExcelName = saveExcelName, printUpdateAfterTrial = 500, imbalancedData = True)
        # Chooses the highest performing feature combination to be the selected features
        selectedFeatures = featureNames_Combinations[0].split()
    
    if testModels is None:
        # Plot training and testing curves of model as we incporporate more of the selected features
        featureSelection.machineLearning().getTrainingCurves(featureNames, performMachineLearning, standardizedFeatures_Cull, selectedFeatures, allFinalLabels, modelTypes, errorBars, modelInd)
        
    else:
        # for each type of model we want to test
        for model in testModels:
            modelTypes = [model, model, model]
            modelInd = 0
            print(modelTypes[modelInd])
            
            performMachineLearning.createModels(modelTypes)
            standardizeClass_Features = standardizeData(allFinalFeatures)
            standardizedFeatures = standardizeClass_Features.standardize(allFinalFeatures)

            # Compile information into the model class
            modelClasses[modelInd].setStandardizationInfo(featureNames, standardizeClass_Features, allFinalLabels)
            
            # Plot training and testing curves of model as we incporporate more of the selected features
            featureSelection.machineLearning().getTrainingCurves(featureNames, performMachineLearning, standardizedFeatures_Cull, selectedFeatures, allFinalLabels, modelTypes, errorBars, modelInd)
    
    # ---------------------------------------------------------------------- #
    # PERFORMING GRID SEARCH FOR FEATURE BOUNDS ---------------------------- #
    # ---------------------------------------------------------------------- #
    
    """
    NOTES:
        - Given features, we now try to figure out the best values to cull out that
            result in the best performance. We need to perform some type of search.
        - I originally approached this by analyzing each feature individually. 
            Only looking at one feature, I would check every possible minimum and maximum
            value and save the boundaries that resulted in the highest accuracy score. 
        - I then switched to a BFS approach, which more hollistically looked at the
            features. Each value/node was a full culling sequence (so in the format of 
            cullInfo). The "neighbors" of a cullInfo are other cullInfos that are identical
            except one of the values of the inequality bounds on a feature has been 
            incremented/decremented.
            - Ex. Neighbors of[('closingTime_Peak_EOG', ['>', '<'], [0.04, 0.2]), ('eyesClosedTime_EOG', ['>', '<'], [0.01, 0.9])]
                could be: [('closingTime_Peak_EOG', ['>', '<'], [0.1, 0.2]), ('eyesClosedTime_EOG', ['>', '<'], [0.01, 0.9])]
                or [('closingTime_Peak_EOG', ['>', '<'], [0.04, 0.2]), ('eyesClosedTime_EOG', ['>', '<'], [0.01, 0.75])]
            - I start with a cullInfo with the values of the inequality bounds of each feature 
            being the maximum and minimum values of all the feature's data, and perform
            a search from there. If a neighboring cullInfo improves the performance score,
            it searches the neighbor's neighbors.
        - This accuracy score currently does not take the imbalance of data into account.
            Similar to my questions with the machine learning models, I'm not sure if
            accuracy is the best way to assess overall performance.
        - I print out the perfromance and final bounds of both versions of my search,
            but only the BFS result is saved and used for the final.
    """
    
    # Setup Helper Functions ----------------------------------------------- #
    
    gridSearch = featureSelection.gridSearch(featureNames, allFinalFeaturesCull, allFinalLabelsCull)
    
    # ---------------------------------------------------------------------- #
    
    if preselectedcullInfo != None:
        # Define new selected features and bounds here
        cullInfo = preselectedcullInfo
    else:
        # Otherwise, perform a grid search to find the bounds of the selected features
        best_accuracy= 0
        best_f1 = 0
        
        # Individual Search: all feature bounds are independently optimized
        cullInfo, final_performance = gridSearch.individual_search(selectedFeatures)
        cullInfo = gridSearch.simplify(cullInfo)
        print(str(cullInfo), final_performance)
        
        # Breadth First Search: all feature bounds are considered hollistically and overall performance is optimized
        cullInfo, final_performance = gridSearch.bfs(selectedFeatures)
        
        # We use the bfs method for our final culling bounds
        cullInfo = gridSearch.simplify(cullInfo)
        print(str(cullInfo), final_performance)
    
    # print cullInfo, formatted as code for the EOG Analysis file
    gridSearch.eog_pipeline_formatted(cullInfo)
    
    # ---------------------------------------------------------------------- #
    # PLOTTING CULLING PIPELINE -------------------------------------------- #
    # ---------------------------------------------------------------------- #
    
    if plotPipeline:
        plot.pipeline(cullInfo)