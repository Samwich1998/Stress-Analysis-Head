from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import itertools
import bisect
import time
import os

# Import helper files.
from ..dataAcquisitionAndAnalysis.excelProcessing.saveDataProtocols import saveExcelData  # Import interfaces for reading/writing data
from .modelControl._modelControl import modelControl  # Import Files for extracting models


class machineLearningInterface:

    def __init__(self, modelTypes, modelFile, allFeatureNames, trainingFolder):
        # Store constant parameters.
        self.modelFile = modelFile
        self.allFeatureNames = allFeatureNames
        # Store variable parameters.
        self.modelTypes = modelTypes

        self.saveDataFolder = None
        # Create data folder to save the analysis.
        if trainingFolder is not None:
            self.saveDataFolder = trainingFolder + "dataAnalysis/Machine Learning/"
            os.makedirs(self.saveDataFolder, exist_ok=True)

        # Initialize classes.
        self.saveDataInterface = saveExcelData()
        self.modelControl = modelControl(self.modelFile, self.modelTypes, self.allFeatureNames, self.saveDataFolder)  # Create a controller for all models.

    def createModels(self, modelTypes):
        self.modelTypes = modelTypes
        self.modelControl.createModels(modelTypes)

    def resetModels(self, modelTypes):
        self.modelControl.createModels(modelTypes)
        # self.modelControl.resetModels()

    def averageModelAccuracy(self, modelInd, featureData, featureLabels, featureNames, numEpochs=200, stratifyBy=None, testSplitRatio=0.2, imbalancedData=False):

        modelPerformances = []
        # For each training round.
        for _ in range(numEpochs):
            # Randomly split the data into training and testing segments.
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(featureData, featureLabels, test_size=testSplitRatio, shuffle=True, stratify=stratifyBy)

            # Train the model and store the accuracy.
            modelPerformance = self.modelControl.modelClasses[modelInd].trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames, imbalancedData=imbalancedData)
            modelPerformances.append(modelPerformance)

        # Return the average classification accuracy
        averagePerformance = stats.trim_mean(modelPerformances, 0.3)
        stdPerformance = np.std(modelPerformances)

        return averagePerformance, stdPerformance

    @staticmethod
    def getMedianScore(scores, plotScores=False):
        # Plot the data distribution
        if plotScores:
            plt.hist(scores, 100, facecolor='blue', alpha=0.5)

        # Fit the skewed distribution and take the median
        ae, loce, scalee = stats.skewnorm.fit(scores)
        medianScore = np.round(loce * 100, 2)

        return medianScore

    def findWorstSubject(self, modelInd, featureData, featureLabels, featureNames, subjectOrder, currentFeatureNames_STR):
        subjectOrder = np.array(subjectOrder)
        featureLabels = np.array(featureLabels)

        # Get All Possible itertools.combinations
        if False:
            subjectCombinationInds = itertools.combinations(range(0, len(featureLabels)), len(featureLabels) - 2)
            allSubjectLabels = set(range(0, len(featureLabels)))
        else:
            uniqueSubjects = np.unique(subjectOrder)
            subjectCombinationInds = itertools.combinations(range(0, len(uniqueSubjects)), len(uniqueSubjects) - 3)
            allSubjectLabels = set(range(0, len(featureLabels)))

        removedInds = []
        removedSubjects = []
        finalPerformances = []
        for subjectInds in subjectCombinationInds:
            subjectInds = list(subjectInds)
            # Reset the Input Variables.
            self.resetModels(self.modelTypes)  # Reset the ML Model

            newInds = []
            for pointInd in range(len(subjectOrder)):
                if subjectOrder[pointInd] in uniqueSubjects[subjectInds]:
                    newInds.append(pointInd)
            subjectInds = newInds

            # Remove the subject features.
            culledSubjectData = featureData[subjectInds, :]
            culledSubjectLabels = featureLabels[subjectInds]

            # Score the model with this data set.
            modelPerformance = self.averageModelAccuracy(modelInd, culledSubjectData, culledSubjectLabels, featureNames, numEpochs=1, stratifyBy=None, testSplitRatio=0.3)

            # Save which subjects were removed.
            discardedSubjectInds = np.array(list(allSubjectLabels.difference(set(subjectInds))))
            removedSubject = np.unique(subjectOrder[discardedSubjectInds])

            insertionPoint = bisect.bisect(finalPerformances, -modelPerformance, key=lambda x: -x)
            # Save the model score and standard deviation.
            removedInds.insert(insertionPoint, discardedSubjectInds)
            removedSubjects.insert(insertionPoint, removedSubject)
            finalPerformances.insert(insertionPoint, modelPerformance)

        print(currentFeatureNames_STR)
        print("Worst Subject:", removedSubjects[0], finalPerformances[0])
        print("Best Subject:", removedSubjects[-1], finalPerformances[-1])
        print("")

    def analyzeFeatureCombinations(self, modelInd, featureData, featureLabels, featureNames, numFeatures_perCombination, numEpochs=10, numModelsTrack=500,
                                   saveData=True, imbalancedData=False, saveExcelName="Feature Combination Accuracy.xlsx", printUpdateAfterTrial=2000):
        """
        NOTES:
            - I start with a minimum threshold of 0.7. This means that I cull out any feature that,
                individually, produces a performance score less than 0.7, and save the rest.
            - Feature combinations are considered based on how many features the combination consists of.
            - After each group of combinations, I basically stop considering any combinations that
                include a feature that was not in any group that made it past the minimum threshold.
                (I explain this more towards the end of the function)
                    - Ex. Say you have combinations (openingTime_Peak_EOG closingTime_Peak_EOG), (blinkDuration_EOG closingTime_Peak_EOG),
                        and (blinkDuration_EOG, tentDeviationX_EOG), but only (openingTime_Peak_EOG closingTime_Peak_EOG) and (blinkDuration_EOG closingTime_Peak_EOG)
                        perform higher than the threshold. We create the next group, combinations of three features, only using features
                        that made it past the threshold in at least one combination: Thus, we can only create (openingTime_Peak_EOG closingTime_Peak_EOG blinkDuration_EOG).
                    - The minimum threshold is re-set after these new combinations are made. I currently
                        set it to be the performance of the best feature combination, meaning that a new
                        combination with n features must be better than the best combination I had with
                        n-1 features. I technically take three standard deviations lower than the mean,
                        because there is variation in performance and that value corresponds to the lowest
                        the score would reasonably actually be. Not sure if this is being too generous.
        """
        # Format the incoming data.
        featureLabels = np.asarray(featureLabels.copy())
        featureData = np.asarray(featureData.copy())
        featureNames = np.asarray(featureNames)

        # General method variables.
        featureNamesCombinations = []
        finalPerformancesSTDs = []
        combinationRoundInd = 0
        finalPerformances = []

        # Find the subject and feature combinations.
        subjectCombinationInds = list(itertools.combinations(range(0, len(featureLabels)), len(featureLabels)))  # DEPRECATED. Using all subjects
        featureCombinationInds = list(itertools.combinations(range(0, len(featureNames)), numFeatures_perCombination))
        numFeatureCombinations = len(featureCombinationInds)

        t1 = time.time()
        # For each feature combination.
        for featureInds in featureCombinationInds:
            featureInds = list(featureInds)
            modelPerformances = []
            modelSTDs = []

            # Separate out the new feature data.
            featureData_culledFeatures = featureData[:, featureInds]
            # Separate out the new feature names.
            currentFeatureNames = featureNames[featureInds]
            currentFeatureNames_STR = " ".join(currentFeatureNames)

            # For each subject combination.
            for subjectInds in subjectCombinationInds:
                subjectInds = list(subjectInds)

                # Reset the models.
                self.resetModels(self.modelTypes)  # Reset the ML Model

                # Collect the Signal Data for the Specific Subjects
                subjectFeatureData = featureData_culledFeatures[subjectInds, :]
                subjectFeatureLabels = featureLabels[subjectInds]

                # Score the model with this data set.
                modelPerformance, modelSTD = self.averageModelAccuracy(modelInd, subjectFeatureData, subjectFeatureLabels, currentFeatureNames, numEpochs,
                                                                       stratifyBy=None, testSplitRatio=0.2, imbalancedData=imbalancedData)
                modelPerformances.append(modelPerformance)
                modelSTDs.append(modelSTD)
            # Take the average of the performance.
            modelPerformance = stats.trim_mean(modelPerformances, 0.3)
            modelSTD = stats.trim_mean(modelSTDs, 0.3)

            # If the model's performance is one of the top scores.
            if len(finalPerformances) < numModelsTrack or finalPerformances[-1] < modelPerformance:

                # Find where to store the incoming model to keep the list sorted by model performance.
                insertionPoint = bisect.bisect(finalPerformances, -modelPerformance, key=lambda x: -x)
                # Save the model score and standard deviation
                featureNamesCombinations.insert(insertionPoint, currentFeatureNames_STR)
                finalPerformances.insert(insertionPoint, modelPerformance)
                finalPerformancesSTDs.insert(insertionPoint, modelSTD)

                # Only track the best models
                if numModelsTrack < len(finalPerformances):
                    finalPerformances.pop()
                    finalPerformancesSTDs.pop()
                    featureNamesCombinations.pop()

            # Update the user every so often about how long its taking.
            if (combinationRoundInd % printUpdateAfterTrial == 0 and combinationRoundInd != 0) or combinationRoundInd == 20:
                t2 = time.time()
                percentComplete = 100 * combinationRoundInd / numFeatureCombinations
                sectionPercent = 100 * min(combinationRoundInd or 1, printUpdateAfterTrial) / numFeatureCombinations
                print(str(np.round(percentComplete, 2)) + "% Complete; Estimated Time Remaining: " + str(np.round((t2 - t1) * (100 - percentComplete) / (sectionPercent * 60), 2)) + " Minutes")
                t1 = time.time()
            combinationRoundInd += 1

        print(finalPerformances[0], finalPerformancesSTDs[0], featureNamesCombinations[0])
        # Save the Data in Excel
        if saveData:
            headers = ["Mean Score", "STD", "Feature Combination"]
            data = np.dstack((finalPerformances, finalPerformancesSTDs, featureNamesCombinations))[0]

            # Save to excel. Formatted in order of the best performance by the mean score.
            self.saveDataInterface.saveFeatureComparison(data, [], headers, self.saveDataFolder, saveExcelName, sheetName=str(numFeatures_perCombination) + " Features in Combination", saveFirstSheet=True)

        return np.array(finalPerformances), np.array(finalPerformancesSTDs), np.array(featureNamesCombinations)
