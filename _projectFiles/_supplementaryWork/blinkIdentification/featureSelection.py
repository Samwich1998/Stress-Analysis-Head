import os
import sys
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class scoring():
    # This class includes all helper functions that can be used to assess performance of culling technique
    
    def __init__(self, featureNames):
        self.featureNames = featureNames
    
    def f1_score(self, currDistribution):
        """
        Calculate f1 score using the distribution of blinks (Positive) versus wire movements (Negative)
        """
        
        tp = currDistribution[0][-1] / currDistribution[0][0]
        fp = currDistribution[2][-1] / currDistribution[2][0]
        fn = (currDistribution[0][0] - currDistribution[0][-1]) / currDistribution[0][0]
        
        return tp / (tp + (0.5 * (fp + fn)))
    
    def accuracy(self, currDistribution):
        """
        Calculate the accuracy using the distribution of blinks (Positive) versus wire movements (Negative)
        """
        return (currDistribution[0][-1] + currDistribution[2][0] - currDistribution[2][-1]) / (currDistribution[0][0] + currDistribution[2][0])
    
    def performance(self, curr_cullInfo, currFinalLabelsCull, currFinalFeaturesCull):
        """
        Given the culling steps, take the original data and return the performance of the entire culling process
        """
        
        currDistribution = {key: [] for key in list((collections.Counter(currFinalLabelsCull)).keys())}
        classDistribution = collections.Counter(currFinalLabelsCull)
        for key in currDistribution.keys():
            currDistribution[key].append(classDistribution[key])
        
        for cullInfoInd in range(len(curr_cullInfo)):
            featureCull, expressions, values = curr_cullInfo[cullInfoInd]
            
            colInd = list(self.featureNames).index(featureCull)
            classDistribution = collections.Counter(currFinalLabelsCull)
            
            assert len(expressions) == len(values), f"Invalid cullInfo step. Expected values to match {len(expressions)} number of expressions, got {len(values)} values."
            for setInd in range(len(expressions)):
                mask = eval("currFinalFeaturesCull[:, colInd] " + expressions[setInd] + str(values[setInd]))
                
                # Apply the mask
                currFinalLabelsCull = currFinalLabelsCull[mask]
                currFinalFeaturesCull = currFinalFeaturesCull[mask]
        
        classDistribution = collections.Counter(currFinalLabelsCull)
        
        for key in currDistribution.keys():
            if key in classDistribution:
                currDistribution[key].append(classDistribution[key])
            else:
                currDistribution[key].append(0)
        
        return self.accuracy(currDistribution), self.f1_score(currDistribution)
    
class machineLearning():
    # This class includes all helper functions useful for feature selection using machine learning
    
    def getTrainingCurves(self, featureNames, performMachineLearning, standardizedFeatures_Cull, selectedFeatures, allFinalLabels, modelTypes, errorBars, modelInd):
        """
        Plots the training and testing performance of the ML model with respect to the number of features used in the model
        """
        selectedFeatures_Cull = performMachineLearning.modelControl.getSpecificFeatures(featureNames, selectedFeatures, standardizedFeatures_Cull)
        Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(selectedFeatures_Cull, allFinalLabels, test_size=0.2, shuffle= True, stratify=allFinalLabels)
        
        avg_train_scores = []
        avg_test_scores = []
        
        std_train_scores = []
        std_test_scores = []
        
        # Train the model and save the accuracy
        
        # For each selected feature
        for i in range(1, len(selectedFeatures) + 1):
            
            # Get and split data for all features up to the currently selected feature
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(selectedFeatures_Cull[:, :i], allFinalLabels, test_size=0.2, shuffle= True, stratify=allFinalLabels)
            modelPerformance = performMachineLearning.modelControl.modelClasses[modelInd].trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, selectedFeatures[0:i])
            
            # If we want to include error bars
            if errorBars:
                train_scores = []
                test_scores = []
                
                # Get standard deviation by generating 100 models using the same data, with different split
                for j in range(100):
                    Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(selectedFeatures_Cull[:, :i], allFinalLabels, test_size=0.2, shuffle= True, stratify=allFinalLabels)
                    modelPerformance = performMachineLearning.modelControl.modelClasses[modelInd].trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, selectedFeatures[0:i], imbalancedData = True)           
                    train_scores.append(performMachineLearning.modelControl.modelClasses[modelInd].scoreModel(Training_Data, Training_Labels, imbalancedData = True))
                    
                    test_scores.append(modelPerformance)
                
                # append the average score across all 100 models to avg_train and avg_test
                avg_train_scores.append(np.mean(train_scores))
                avg_test_scores.append(np.mean(test_scores))
                
                # append the standard deviation across all 100 models
                std_train_scores.append(np.std(train_scores))
                std_test_scores.append(np.std(test_scores))
                
            else:
                # If we do not want to include error bars, we only generate 1 model, and do not keep track of standard deviation
                
                avg_train_scores.append(performMachineLearning.modelControl.modelClasses[modelInd].scoreModel(Testing_Data, Testing_Labels, imbalancedData = True))
                avg_test_scores.append(modelPerformance)
                    
        plt.figure()
        
        if errorBars: # plot learning curve with errorbars using the standard deviation scores
            plt.errorbar(np.arange(1, len(selectedFeatures)+1), avg_train_scores, yerr=std_train_scores, marker='o', capsize=5)
            plt.errorbar(np.arange(1, len(selectedFeatures)+1), avg_test_scores, yerr=std_test_scores, marker='o', capsize=5)
        else: # otherwise, just plot the learning curve with respect to the number of features used to generate the model
            plt.plot(np.arange(1, len(selectedFeatures)+1), avg_train_scores, marker='o')
            plt.plot(np.arange(1, len(selectedFeatures)+1), avg_test_scores, marker='o')
        
        # formatting plot
        plt.xlabel('Number of Selected Features')
        plt.ylabel('Model Performance (Accuracy)')
        plt.legend(["training", "testing"])
        plt.title('Learning Curves: ' + modelTypes[modelInd])
        plt.show()
        
        return avg_train_scores, avg_test_scores
    
class gridSearch():
    # This class includes all helper functions useful for finding the optimal bounds for culling datapoints
    def __init__(self, featureNames, allFinalFeaturesCull, allFinalLabelsCull):
        self.featureNames = featureNames
        self.allFinalFeaturesCull = allFinalFeaturesCull
        self.allFinalLabelsCull = allFinalLabelsCull
        self.scoring = scoring(featureNames)
        
    
    def simplify(self, curr_cullInfo):
        """
        Given the current culling steps, remove unnecessary bounds (when the lower bound is already the minimum value of all 
        datapoints, or the upper bound is the maximum value of all datapoints)
        """
        
        new_cullInfo = []
        # For each step in the culling pipeline
        for cull_step in curr_cullInfo:
            feature = cull_step[0]
            colInd = list(self.featureNames).index(feature)
            features = self.allFinalFeaturesCull[:,colInd]
            currmax = max(features)
            currmin = min(features)
            
            # If the value of the lower bound is the current minimum of all datapoints
            if cull_step[2][0] == currmin:
                # this culling step is not necessary, and we can simplify
                new_cullInfo.append((feature, ["<"], [cull_step[2][1]]))
            # If the value of the upper bound is the current maximum of all datapoints
            elif cull_step[2][1] == currmax:
                # this culling step is not necessary, and we can simplify.
                new_cullInfo.append((feature, [">"], [cull_step[2][0]]))
            else:
                # Otherwise, both boundaries are kept.
                new_cullInfo.append(cull_step)
                
        return new_cullInfo
    
    def eog_pipeline_formatted(self, cullInfo):
        """
        Puts cullInfo in this format:
            
        if not 0.008 < blinkDuration < 0.5:
            if debugBlinkDetection: print("\t\tBad Blink Duration:", blinkDuration, xData[peakInd])
            return [None]
        
        Which is how the EOG pipeline code is formatted in eogAnalysis.py (can be pasted after line 713).
        """
        
        print("---------- Code for Culling Bad Blinks in eogAnalysis.py ----------")
        
        for step in cullInfo:
            line = "if not "
            if '>' in step[1]:
                line += str(step[2][step[1].index('>')]) + " < "
            line += str(step[0][:-4])
            if '<' in step[1]:
                line += " < " + str(step[2][step[1].index('<')])
            line += ":"
            print(line)
            print("    " + "if debugBlinkDetection: print(\"\t\tBad " + str(step[0][:-4]) + ":\", " + str(step[0][:-4]) + ", xData[peakInd])")
            print("    return [None]")
            
        print("-------------------------------------------------------------------")
    
    def individual_search(self, curr_selectedFeatures):
        """
        Finds bound that optimizes accuracy for for each feature independently.
        and returns a list of all of the bounds found for the features.
        """
        bins = 50
        cullInfo = []
        # For each selected feature
        for feature in curr_selectedFeatures:
            
            # Get all data of given feature
            colInd = list(self.featureNames).index(feature)
            features = list(self.allFinalFeaturesCull[:,colInd])
            
            # Find maximum and minimum of all data
            currmax = max(features)
            currmin = min(features)
            
            # We only want to check bins number of boundaries. Calculate how much we need to increment by to assess the whole range of values
            delta = (max(features) - min(features)) / bins
            
            cullInfo.append((feature, [">"], [currmin]))
            
            scores = []
            
            # Loop over each possible lower bound, and compute the performance if the bound were to be placed at that value.
            for i in range(bins):
                cullInfo[-1] = (feature, [">"], [currmin + (i * delta)])
                curr_accuracy, curr_f1 = self.scoring.performance(cullInfo, self.allFinalLabelsCull, self.allFinalFeaturesCull)
                scores.append(curr_accuracy)
            
            # The lower bound is the bound that recieves the best performance score
            lower_bound = currmin + (np.argmax(scores) * delta)
            
            scores = []
            # Loop over each possible upper bound
            for i in range(bins):
                # Only loop until we have reached the lower bound, as the upper bound must be greater than the lower bound
                if currmax - (i * delta) <= lower_bound:
                    break
                cullInfo[-1] = (feature, [">", "<"], [lower_bound, currmax - (i * delta)])
                curr_accuracy, curr_f1 = self.scoring.performance(cullInfo, self.allFinalLabelsCull, self.allFinalFeaturesCull)
                scores.append(curr_accuracy)
            # The upper bound is the bound that recieves the best performance score
            upper_bound = currmax - (np.argmax(scores) * delta)
            
            cullInfo[-1] = (feature, [">", "<"], [lower_bound, upper_bound])
                
        # return all of the bounds, 
        return cullInfo, self.scoring.performance(cullInfo, self.allFinalLabelsCull, self.allFinalFeaturesCull)
   
        
    def bfs(self, curr_selectedFeatures):
        """
        Performs a breadth-first search, exploring possible bounds for all of the features.
        Attempts to maximize the total performance across all features.
        """
        
        # Initialize visited and queue
        visited = []
        queue = []
        best_accuracy = 0
        best_cullInfo = []
        
        # Set number of bins for discretization of feature values
        bins = 15
        deltas = []
        curr_cullInfo = []
        
        # For each selected feature, find the amount we need to increment/decrement by to reach its "neighbor". We want to check bounds closest to the current bound.
        for feature in curr_selectedFeatures:
            colInd = list(self.featureNames).index(feature)
            features = list(self.allFinalFeaturesCull[:,colInd])
            
            currmax = max(features)
            currmin = min(features)
            deltas.append((max(features) - min(features)) / bins)
            
            # Initialize the culling to at first, not cull any datapoints.
            curr_cullInfo.append((feature, [">", "<"], [currmin, currmax]))
            
        # Queue the current culling step
        queue.append(curr_cullInfo)
        visited.append(str(curr_cullInfo))

        
        while queue: # Creating loop to visit each node
            # Pop a culling pipeline off of the queue
            currCullInfo = queue.pop(0)
            curr_accuracy, curr_f1_score = self.scoring.performance(currCullInfo, self.allFinalLabelsCull, self.allFinalFeaturesCull)
            
            # If this culling pipeline has the best performance so far, continue to explore its "neighbors".
            if curr_accuracy >= best_accuracy:
                best_accuracy = curr_accuracy
                best_cullInfo = currCullInfo
                
                # For each selected feature
                for feature_ind in range(len(curr_selectedFeatures)):
                    newCullInfo = np.copy(currCullInfo)
                    # Create a new culling pipeline where the culling step that includes the feature has a slightly stricter lower bound (increases by delta value)
                    newCullInfo[feature_ind] = (curr_selectedFeatures[feature_ind], [">", "<"], [currCullInfo[feature_ind][2][0] + deltas[feature_ind], currCullInfo[feature_ind][2][1]])
                    # If this pipeline has not been visited, add it to the queue. We want to visit this eventually
                    if str(newCullInfo) not in visited:
                        queue.append(newCullInfo)
                        visited.append(str(newCullInfo))
                        
                    newCullInfo = np.copy(currCullInfo)
                    # Create a new culling pipeline where the culling step that includes the feature has a slightly stricter upper bound (decreases by delta value)
                    newCullInfo[feature_ind] = (curr_selectedFeatures[feature_ind], [">", "<"], [currCullInfo[feature_ind][2][0], currCullInfo[feature_ind][2][1] - deltas[feature_ind]])
                    # If this pipeline has not been visited, add it to the queue. We want to visit this eventually
                    if str(newCullInfo) not in visited:
                        queue.append(newCullInfo)
                        visited.append(str(newCullInfo))
                        
        return best_cullInfo, best_accuracy
    