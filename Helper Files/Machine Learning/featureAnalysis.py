#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:47:49 2021

@author: samuelsolomon
"""
# Basic Modules
import os
import collections
import numpy as np
from scipy import stats
from copy import deepcopy
# Modules for Plotting
import seaborn as sns
import matplotlib.pyplot as plt



class featureAnalysis:
    
    def __init__(self, featureNames, stimulusTime, saveDataFolder):
        # Store Extracted Features
        self.featureNames = featureNames            # Store Feature Names
        self.stimulusTime = list(stimulusTime)
        
        # Save Information
        self.saveDataFolder = saveDataFolder
        
        self.colorList = ['k', 'r', 'b', 'brown', 'purple', 'tab:green']
    
    def singleFeatureAnalysis(self, timePoints, featureList, averageIntervalList = [0.001, 30], folderName = "singleFeatureAnalysis/"):
        print("Plotting Each Features over Time")
        # Create Directory to Save the Figures
        saveDataFolder = self.saveDataFolder + folderName
        os.makedirs(saveDataFolder, exist_ok=True)
        
        # Loop Through Each Feature
        for featureInd in range(len(self.featureNames)):
            fig = plt.figure()
            ax = plt.gca()
            
            # Extract One Feature from the List
            allFeatures = featureList[:,featureInd]
            # Take Different Averaging Methods
            for ind, averageTogether in enumerate(averageIntervalList):
                features = []
                
                # Average the Feature Together at Each Point
                for pointInd in range(len(allFeatures)):
                    # Get the Interval of Features to Average
                    featureInterval = allFeatures[timePoints > timePoints[pointInd] - averageTogether]
                    timeMask = timePoints[timePoints > timePoints[pointInd] - averageTogether]
                    featureInterval = featureInterval[timeMask <= timePoints[pointInd]]
                    
                    # Take the Trimmed Average
                    feature = stats.trim_mean(featureInterval, 0.3)
                    features.append(feature)
                
                # Plot the Feature
                ax.plot(timePoints, features, 'o', c=self.colorList[ind], markersize=3)
            
            # Specify the Location of the Stimulus
            ax.vlines(self.stimulusTime, min(features), max(features), 'g', linewidth = 2, zorder=len(averageIntervalList) + 1)

            # Add Figure Labels
            ax.set_xlabel("Time (Seconds)")
            ax.set_ylabel(self.featureNames[featureInd])
            ax.set_title(self.featureNames[featureInd] + " Analysis")
            # Add Figure Legened
            ax.legend([str(averageTime) + " second average" for averageTime in averageIntervalList])
            # Save the Figure
            fig.savefig(saveDataFolder + self.featureNames[featureInd] + ".png", dpi=300, bbox_inches='tight')
            
            # Clear the Figure        
            fig.clear()
            plt.close(fig)
            plt.cla()
            plt.clf()
            
    
    def correlationMatrix(self, featureList, folderName = "correlationMatrix/"):
        print("Plotting the Correlation Matrix Amongst the Features")
        # Create Directory to Save the Figures
        saveDataFolder = self.saveDataFolder + folderName
        os.makedirs(saveDataFolder, exist_ok=True)
        
        # Perform Deepcopy to Not Edit Features
        signalData = deepcopy(featureList); signalLabels = deepcopy(self.featureNames)
        
        # Standardize the Feature
        for i in range(len(signalData[0])):
             signalData[:,i] = (signalData[:,i] - np.mean(signalData[:,i]))/np.std(signalData[:,i], ddof=1)
        
        matrix = np.array(np.corrcoef(signalData.T)); 
        sns.set_theme(); ax = sns.heatmap(matrix, cmap='icefire', xticklabels=signalLabels, yticklabels=signalLabels)
        # Save the Figure
        sns.set(rc={'figure.figsize':(50,35)})
        fig = ax.get_figure(); fig.savefig(saveDataFolder + "correlationMatrixFull.png", dpi=300)
        fig.show()
        
        # Cluster the Similar Features
        signalLabelsX = deepcopy(signalLabels)
        signalLabelsY = deepcopy(signalLabels)
        for i in range(1,len(matrix)):
            signalLabelsX = signalLabelsX[matrix[:,i].argsort()]
            matrix = matrix[matrix[:,i].argsort()]
        for i in range(1,len(matrix[0])):
            signalLabelsY = signalLabelsY[matrix[i].argsort()]
            matrix = matrix [ :, matrix[i].argsort()]
        # Plot the New Cluster
        sns.set_theme(); ax = sns.heatmap(matrix, cmap='icefire', xticklabels=signalLabelsX, yticklabels=signalLabelsY)
        # Save the Figure
        sns.set(rc={'figure.figsize':(50,35)})
        fig = ax.get_figure(); fig.savefig(saveDataFolder + "correlationMatrixSorted.png", dpi=300)
        fig.show()

        # Remove Small Correlations
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if abs(matrix[i][j]) < 0.96:
                    matrix[i][j] = 0
        # Plot the New Correlations
        sns.set_theme(); ax = sns.heatmap(matrix, cmap='icefire', xticklabels=signalLabelsX, yticklabels=signalLabelsY)
        # Save the Figure
        sns.set(rc={'figure.figsize':(50,35)})
        fig = ax.get_figure(); fig.savefig(saveDataFolder + "correlationMatrixSortedCull.png", dpi=300)            
        fig.show()

    def featureComparison(self, featureList1, featureList2, featureLabels, featureNames, xChemical, yChemical):
        # Create Directory to Save the Figures
        saveDataFolder = self.saveDataFolder + "chemicalFeatureComparison/"
        os.makedirs(saveDataFolder, exist_ok=True)
        
        featureList1 = np.array(featureList1)
        featureList2 = np.array(featureList2)
        
        labelList = ['Cold', 'Exercise', 'VR']

        for featureInd1 in range(len(featureNames)):
            
            features1 = featureList1[:, featureInd1]
            
            for featureInd2 in range(len(featureNames)):
                features2 = featureList2[:, featureInd2]
                
                fig = plt.figure()
                ax = plt.gca()
                for ind in range(len(featureLabels)):
                    labelInd = featureLabels[ind]
                    ax.plot(features1[ind], features2[ind], 'o', c=self.colorList[labelInd], label=labelList[labelInd])
                
                ax.set_xlabel(xChemical + ": " + featureNames[featureInd1])
                ax.set_ylabel(yChemical + ": " + featureNames[featureInd2])
                ax.set_title("Feature Comparison")
                ax.legend()
                # Save the Figure
                fig.savefig(saveDataFolder + featureNames[featureInd1] + "_" + featureNames[featureInd2] + ".png", dpi=300, bbox_inches='tight')
            
                plt.show()
    
    def singleFeatureComparison(self, featureListFull, featureLabelFull, chemicalOrder, featureNames):
        # Create Directory to Save the Figures
        saveDataFolder = self.saveDataFolder + "singleChemicalFeatureComparison/"
        os.makedirs(saveDataFolder, exist_ok=True)
        
        #labelList = ['Cold', 'Exercise', 'VR']
        for chemicalInd in range(len(chemicalOrder)):
            chemicalName = chemicalOrder[chemicalInd]
            featureList = featureListFull[chemicalInd]
            featureLabels = featureLabelFull[chemicalInd]
            
            saveDataFolderChemical = saveDataFolder + chemicalName + "/"
            os.makedirs(saveDataFolderChemical, exist_ok=True)
            
            for featureInd in range(len(featureNames)):
                
                features = featureList[:, featureInd]
                
                fig = plt.figure()
                ax = plt.gca()
                for ind in range(len(featureLabels)):
                    labelInd = featureLabels[ind]
                    ax.plot(features[ind], [0], self.colorList[labelInd])
                
                ax.set_xlabel(chemicalName + ": " + featureNames[featureInd])
                ax.set_ylabel("Constant")
                ax.set_title("Feature Comparison")
               # plt.legend()
                # Save the Figure
                fig.savefig(saveDataFolderChemical + featureNames[featureInd] + ".png", dpi=300, bbox_inches='tight')
                # Clear the Figure        
                fig.clear()
                plt.close(fig)
                plt.cla()
                plt.clf()
    
    def featureDistribution(self, signalData, signalLabels, featureLabels, folderName = "Feature Distribution/"):
        print("Plotting Feature Distributions")
        classDistribution = collections.Counter(signalLabels)
        print("\tClass Distribution:", classDistribution)
        print("\tNumber of Data Points = ", len(classDistribution))
        
        # Create Directory to Save the Figures
        saveDataFolder = self.saveDataFolder + folderName
        os.makedirs(saveDataFolder, exist_ok=True)
        
        signalData = np.array(signalData); signalLabels = np.array(signalLabels)
        for featureInd in range(len(self.featureNames)):
            fig = plt.figure()
            
            for label in range(len(featureLabels)):
                features = signalData[:,featureInd][signalLabels == label]
            
                plt.hist(features, bins=100, alpha=0.5, label = featureLabels[label],  align='mid', density=True)

            plt.legend()
            plt.ylim(0,20)
            plt.ylabel(self.featureNames[featureInd])
            fig.savefig(saveDataFolder + self.featureNames[featureInd] + ".png", dpi=300, bbox_inches='tight')       
            # Clear the Figure        
            fig.clear()
            plt.close(fig)
            plt.cla()
            plt.clf()

            
            
            
            
            
            
            
            
            
            
            
            
            