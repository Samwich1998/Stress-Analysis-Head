#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:38:50 2023

@author: jadelynndaomac
"""

import os
import collections
import numpy as np
import matplotlib.pyplot as plt

# This is for version control on the culling pipeline plots.
from datetime import date

import featureSelection

class plotting():
    def __init__(self, allFinalFeaturesCull, allFinalLabelsCull, featureLabelOptions, featureNames):
        self.allFinalFeaturesCull = allFinalFeaturesCull
        self.allFinalLabelsCull = allFinalLabelsCull
        self.featureLabelOptions = featureLabelOptions
        self.featureNames = featureNames
        
    def featureDistributions(self):
        print("Plotting Feature Distributions")
        classDistribution = collections.Counter(self.allFinalLabelsCull)
        print("\tClass Distribution:", classDistribution)
        print("\tNumber of Unique Points = ", len(classDistribution))
          
        # Create/verify a directory to save the figures
        saveDataFolder = "./featureDistributions/"
        os.makedirs(saveDataFolder, exist_ok = True)
        
        # for every feature
        for featureInd in range(len(self.featureNames)):
            fig = plt.figure()
            
            # for each label (blinks and wire)
            for label in classDistribution.keys():
                # plot histogram of feature, using only data with specific label
                features = self.allFinalFeaturesCull[:,featureInd][self.allFinalLabelsCull == label]
                
                plt.hist(features, bins=min(100, len(features)), alpha=0.5, label = self.featureLabelOptions[label],  align='mid', density=True, stacked=True)
    
            plt.legend()
            plt.ylabel(self.featureNames[featureInd])
            # save histogram as featureName.png in saveDataFolder
            fig.savefig(saveDataFolder + self.featureNames[featureInd] + ".png", dpi=300, bbox_inches='tight')       
            plt.show()
            fig.clear()
            plt.cla(); plt.clf()
    
    def pipeline(self, cullInfo):
        scoring = featureSelection.scoring(self.featureNames)
        allDistributions = {key: [] for key in list((collections.Counter(self.allFinalLabelsCull)).keys())}
            
        # Setup plotting
        fig = plt.figure()
        fig.set_size_inches(12, 9)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        num_subplots = len(cullInfo) + 1
        num_cols = 3
        num_bins = 35
        gs = fig.add_gridspec(int(np.ceil(num_subplots/ num_cols)), num_cols)
        axs = gs.subplots()
        
        fig.suptitle('Blink Data Culling Pipeline')
        
        # Blinks = Green Line, Wire = Blue Line
        colors = {0: 'green', 1: 'orange', 2: 'royalblue'}
        plt.setp(axs, ylim=[0, 0.15])
        
        # Get original distribution of all the labels
        classDistribution = collections.Counter(self.allFinalLabelsCull)
        
        # print distribution
        print("Original Feature Distribution")
        print("\tClass Distribution:", classDistribution)
        print("\tNumber of Unique Points = ", len(classDistribution))
        
        # Store the distribution by their label
        for key in allDistributions.keys():
            allDistributions[key].append(classDistribution[key])
        
        
        def histogram_subplot(index):
            """
            Generate subplot for a given step in the culling pipeline. 
            """
            
            currInd_x = int(index / num_cols)
            currInd_y = index % num_cols
            
            # For each type of label in the distribution
            for label in classDistribution.keys():
                # Get all of the feature's datapoints for the given label
                features = self.allFinalFeaturesCull[:,featureInd][self.allFinalLabelsCull == label]
                # Create histogram
                n, bins = np.histogram(features, bins=num_bins)
                
                # Format subplot
                axs[currInd_x, currInd_y].fill_between(bins[:-1], n / allDistributions[label][0], step='pre', alpha=0.6, color=colors[label], label=self.featureLabelOptions[label])
                axs[currInd_x, currInd_y].set_title('Culling Step #' + str(index))
                axs[currInd_x, currInd_y].set_xlabel(self.featureNames[featureInd])
                axs[currInd_x, currInd_y].set_ylabel('normalized density')
            
            axs[currInd_x, currInd_y].legend()
                
            # Draw dashed, vertical red lines to represent the boundaries we are culling at
            for value in values:
                axs[currInd_x, currInd_y].axvline(value, color='red', linestyle='dashed')
            
        # ---------------------------------------------------------------------- #
    
        # For each step in culling pipeline (cullInfo)
        for cullInfoInd in range(len(cullInfo)):
            
            featureCull, expressions, values = cullInfo[cullInfoInd]
            
            # Get the column of the feature we are culling
            colInd = list(self.featureNames).index(featureCull)
            featureInd = colInd
            
            # Generate subplot
            histogram_subplot(cullInfoInd)
            
            assert len(expressions) == len(values), f"Invalid cullInfo step. Expected values to match {len(expressions)} number of expressions, got {len(values)} values."
            
            # For each boundary in the culling step (upper and/or lower bound for the given feature)
            for setInd in range(len(expressions)):
                
                # Create a mask that only returns the datapoints that are inside of the boundary
                mask = eval("self.allFinalFeaturesCull[:, colInd] " + expressions[setInd] + str(values[setInd]))
                
                # Apply the mask
                self.allFinalLabelsCull = self.allFinalLabelsCull[mask]
                self.allFinalFeaturesCull = self.allFinalFeaturesCull[mask]
            
            # Get new distribution of all the labels, after the culling step
            print("Plotting Feature Distributions")
            classDistribution = collections.Counter(self.allFinalLabelsCull)
            print("\tClass Distribution:", classDistribution)
            print("\tNumber of Unique Points = ", len(classDistribution))
            
            # Store the distribution with the rest of the previous distributions
            for key in allDistributions.keys():
                if key in classDistribution:
                    allDistributions[key].append(classDistribution[key])
                else:
                    allDistributions[key].append(0)
            
        # Creating subplot to show distributions over each culling step
        
        # For each type of label
        for label in allDistributions.keys():
            # Plot the proportion of the datapoints that have not been culled versus the original number of datapoints, after each culling step was performed
            axs[-1, -1].plot([*range(0, len(cullInfo) + 1)], allDistributions[label] / np.max(allDistributions[label]), color=colors[label], label=self.featureLabelOptions[label])
            axs[-1, -1].set_ylim([0, 1])
            
        # formatting
        axs[-1, -1].set_title('Peaks Preserved Over \n Culling Pipeline')
        axs[-1, -1].set_ylabel('normalized density')
        axs[-1, -1].set_xlabel('step in pipeline')
        axs[-1, -1].legend()
        
        fig.text(0.5, 0.01, "Accuracy: " + str(scoring.accuracy(allDistributions)) + "\n" + "F1 Score: " + str(scoring.f1_score(allDistributions)), ha='center')
        #fig.legend()
        plt.show()
        
        saveDataFolder = "./featureDistributions/EOGCullingPipeline/"
        os.makedirs(saveDataFolder, exist_ok = True)
        
        today = date.today()
        fig.savefig(saveDataFolder + today.strftime("%m_%d") + ".svg", dpi=300, bbox_inches='tight')