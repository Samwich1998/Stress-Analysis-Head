#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:47:49 2021

@author: samuelsolomon
"""
# Basic Modules
import os
import sys
import numpy as np
import collections
import pandas as pd
from scipy import stats
# Modules for Plotting
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
# Machine Learning Modules
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
# Neural Network Modules
from sklearn.model_selection import train_test_split
# Feature Importance
import shap

# Import Machine Learning Files
sys.path.append('./Helper Files/Machine Learning/_Classification Methods/')
sys.path.append('./Machine Learning/_Classification Methods/') # Folder with Machine Learning Files
import supportVectorRegression as SVR   # Functions for Support Vector Regression Algorithm
import neuralNetwork as NeuralNet       # Functions for Neural Network Algorithm
import logisticRegression as LR         # Functions for Linear Regression Algorithm
import ridgeRegression as Ridge         # Functions for Ridge Regression Algorithm
import elasticNet as elasticNet         # Functions for Elastic Net Algorithm
import randomForest                     # Functions for the Random Forest Algorithm
import KNN as KNN                       # Functions for K-Nearest Neighbors' Algorithm
import SVM as SVM                       # Functions for Support Vector Machine algorithm

sys.path.append('./Helper Files/Data Aquisition and Analysis/_Plotting/')  # Folder with Machine Learning Files
sys.path.append('./Data Aquisition and Analysis/_Plotting/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network

class predictionModelHead:
    
    def __init__(self, modelType, modelPath, numFeatures, machineLearningClasses, saveDataFolder, supportVectorKernel = ""):
        # Store Parameters
        self.modelType = modelType
        self.modelPath = modelPath
        self.saveDataFolder = saveDataFolder
        self.machineLearningClasses = machineLearningClasses
        self.numClasses = len(machineLearningClasses)
        self.testSize = 0.4
        self.supportVectorKernel = supportVectorKernel
        
        self.possibleModels = ['RF', 'LR', 'KNN', 'SVM', 'RG', 'EN', "SVR"]
        if modelType not in self.possibleModels:
            exit("The Model Type is Not Found")
        
        # Holder Variables
        self.map2D = []
        # Get Prediction Model
        self.predictionModel = self.getModel(modelType, modelPath, numFeatures)
        if saveDataFolder:
            # Create Output File Directory to Save Data: If None
            os.makedirs(self.saveDataFolder, exist_ok=True)
        
    
    def getModel(self, modelType, modelPath, numFeatures):
        # Get the Machine Learning Model
        if modelType == "NN":
            # numFeatures = The dimensionality of one data point
            predictionModel = NeuralNet.Neural_Network(modelPath = modelPath, numFeatures = numFeatures)
        elif modelType == "RF":
            predictionModel = randomForest.randomForest(modelPath = modelPath)
        elif modelType == "LR":
            predictionModel = LR.logisticRegression(modelPath = modelPath)
        elif modelType == "RG":
            predictionModel = Ridge.ridgeRegression(modelPath = modelPath)
        elif modelType == "EN":
            predictionModel = elasticNet.elasticNet(modelPath = modelPath)
        elif modelType == "KNN":
            predictionModel = KNN.KNN(modelPath = modelPath, numClasses = self.numClasses)
        elif modelType == "SVM":
            predictionModel = SVM.SVM(modelPath = modelPath, modelType = self.supportVectorKernel, polynomialDegree = 3)
            # Section off SVM Data Analysis Into the Type of Kernels
            if self.saveDataFolder:
                self.saveDataFolder += self.supportVectorKernel +"/"
                os.makedirs(self.saveDataFolder, exist_ok=True)
        elif modelType == "SVR":
            predictionModel = SVR.supportVectorRegression(modelPath = modelPath, modelType = self.supportVectorKernel, polynomialDegree = 3)
            # Section off SVM Data Analysis Into the Type of Kernels
            if self.saveDataFolder:
                self.saveDataFolder += self.supportVectorKernel +"/"
                os.makedirs(self.saveDataFolder, exist_ok=True)
        else:
            print("No Matching Machine Learning Model was Found for '", modelType, "'");
            sys.exit()
        # Return the Precition Model
        return predictionModel
    
    def scoreClassificationModel(self, signalData, signalLabels, stratifyBy = []):
        # Extract a list of unique labels
        # possibleClassifications = list(set(signalLabels))
        classificationScores = []
        # Taking the Average Score Each Time
        for _ in range(300):
            # Train the Model with the Training Data
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.4, shuffle= True, stratify=stratifyBy)
            self.predictionModel.model.fit(Training_Data, Training_Labels)
            
            classAccuracies = []
            # if not testStressScores:
            #     for classification in possibleClassifications:
            #         testClassData = Testing_Data[Testing_Labels == classification]
            #         testClassLabels = self.predictionModel.model.predict(testClassData)
                    
            #         classAccuracy = len(testClassLabels[testClassLabels == classification])/len(testClassLabels)
            #         classAccuracies.append(classAccuracy)
            testClassLabels = self.predictionModel.model.predict(Testing_Data)
            classAccuracy = len(testClassLabels[testClassLabels == Testing_Labels])/len(testClassLabels)
            classAccuracies.append(classAccuracy)
            
            classificationScores.append(classAccuracies)
        
        averageClassAccuracy = stats.trim_mean(classificationScores, 0.4)
        return averageClassAccuracy
        
        
    def trainModel(self, signalData, signalLabels, featureLabels = [], returnScore = False, stratifyBy = []):
        if len(featureLabels) != 0 and not len(featureLabels) == len(signalData[0]):
            print("The Number of Feature Labels Provided Does Not Match the Number of Features")
            print("Removing Feature Labels")
            featureLabels = []
            
        signalData = np.array(signalData); signalLabels = np.array(signalLabels); featureLabels = np.array(featureLabels)
        # Find the Data Distribution
        classDistribution = collections.Counter(signalLabels)
        print("Class Distribution:", classDistribution)
        print("Number of Data Points = ", len(classDistribution))
        
        if self.modelType in self.possibleModels:
            # Train the Model Multiple Times
            modelScores = []
            # Taking the Average Score Each Time
            for _ in range(300):
                # Train the Model with the Training Data
                Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.5, shuffle= True, stratify=stratifyBy)
                modelScores.append(self.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels))
            if returnScore:
                #print("Mean Testing Accuracy (Return):", meanScore)
                return stats.trim_mean(modelScores, 0.4)
            # Display the Spread of Scores
            plt.hist(modelScores, 100, facecolor='blue', alpha=0.5)
            # Fit the Mean Distribution and Save the Mean
            ae, loce, scalee = stats.skewnorm.fit(modelScores)
            # Take the Median Score as the True Score
            meanScore = np.round(loce*100, 2)
            if returnScore:
                #print("Mean Testing Accuracy (Return):", meanScore)
                return meanScore
            #print("Mean Testing Accuracy:", meanScore)
            # Label Accuracy
            self.accuracyDistributionPlot_Average(signalData, signalLabels, self.machineLearningClasses, "Test")
            self.accuracyDistributionPlot_Average(signalData, signalLabels, self.machineLearningClasses, "Full")
            # Extract Feature Importance
            #self.featureImportance(signalData, signalLabels, signalData, signalLabels, featureLabels = featureLabels, numTrials = 100)
            self.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels)
            
        if self.modelType == "NN":
            # Plot the training loss    
            self.predictionModel.plotStats()
        
        
    def mapTo2DPlot(self, signalData, signalLabels, name = "Channel Map"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(signalData, signalLabels)
        
        mds = MDS(n_components=2,random_state=0, n_init = 4)
        X_2d = mds.fit_transform(X_scaled)
        
        X_2d = self.rotatePoints(X_2d, -np.pi/2).T
        
        figMap = plt.scatter(X_2d[:,0], X_2d[:,1], c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 130, marker='.', edgecolors='k')        
        
        # Figure Aesthetics
        fig.colorbar(figMap, ticks=range(self.numClasses), label='digit value')
        figMap.set_clim(-0.5, 5.5)
        plt.title('Channel Feature Map');
        fig.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=200, bbox_inches='tight')
        plt.show() # Must be the Last Line
        
        return X_2d
    
    def rotatePoints(self, rotatingMatrix, theta_rad = -np.pi/2):

        A = np.matrix([[np.cos(theta_rad), -np.sin(theta_rad)],
                       [np.sin(theta_rad), np.cos(theta_rad)]])
        
        m2 = np.zeros(rotatingMatrix.shape)
        
        for i,v in enumerate(rotatingMatrix):
          w = A @ v.T
          m2[i] = w
        m2 = m2.T
        
        return m2
    
    
    def plot3DLabels(self, signalData, signalLabels, name = "Channel Feature Distribution"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)
        ax = plt.axes(projection='3d')
        
        # Scatter Plot
        ax.scatter3D(signalData[:, 3], signalData[:, 1], signalData[:, 2], c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 100, edgecolors='k')
        
        ax.set_title('Channel Feature Distribution');
        ax.set_xlabel("Channel 4")
        ax.set_ylabel("Channel 2")
        ax.set_zlabel("Channel 3")
        #fig.tight_layout()
        fig.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=200, bbox_inches='tight')
        plt.show() # Must be the Last Line
    
    def plot3DLabelsMovie(self, signalData, signalLabels, name = "Channel Feature Distribution Movie"):
        # Plot and Save
        fig = plt.figure()
        #fig.set_size_inches(15,15,10)
        ax = plt.axes(projection='3d')
        
        # Initialize Relevant Channel 4 Range
        errorPoint = 0.01; # Width of Channel 4's Values
        channel4Vals = np.arange(min(signalData[:, 3]), max(signalData[:, 3]), 2*errorPoint)
        
        # Initialize Movie Writer for Plots
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=name + " " + self.modelType, artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=2, metadata=metadata)
        
        with writer.saving(fig, self.saveDataFolder + name + " " + self.modelType + ".mp4", 300):
            for channel4Val in channel4Vals:
                channelPoints1 = signalData[:, 0][abs(signalData[:, 3] - channel4Val) < errorPoint]
                channelPoints2 = signalData[:, 1][abs(signalData[:, 3] - channel4Val) < errorPoint]
                channelPoints3 = signalData[:, 2][abs(signalData[:, 3] - channel4Val) < errorPoint]
                currentLabels = signalLabels[abs(signalData[:, 3] - channel4Val) < errorPoint]
                
                if len(currentLabels) != 0:
                    # Scatter Plot
                    figMap = ax.scatter3D(channelPoints1, channelPoints2, channelPoints3, "o", c = currentLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 50, edgecolors='k')
        
                    ax.set_title('Channel Feature Distribution; Channel 4 = ' + str(channel4Val) + " ± " + str(errorPoint));
                    ax.set_xlabel("Channel 1")
                    ax.set_ylabel("Channel 2")
                    ax.set_zlabel("Channel 3")
                    ax.yaxis._axinfo['label']['space_factor'] = 20
                    
                    ax.set_xlim3d(0, max(signalData[:, 0]))
                    ax.set_ylim3d(0, max(signalData[:, 1]))
                    ax.set_zlim3d(0, max(signalData[:, 2]))
                    
                    # Figure Aesthetics
                    cb = fig.colorbar(figMap, ticks=range(self.numClasses), label='digit value')
                    plt.rcParams['figure.dpi'] = 300
                    figMap.set_clim(-0.5, 5.5)
                    
                    # Write to Video
                    writer.grab_frame()
                    # Clear Previous Frame
                    plt.cla()
                    cb.remove()
                
        plt.show() # Must be the Last Line
                 
    def accuracyDistributionPlot_Average(self, signalData, signalLabels, machineLearningClasses, analyzeType = "Full", name = "Accuracy Distribution"):
        numAverage = 200
        
        accMat = np.zeros((len(machineLearningClasses), len(machineLearningClasses)))
        # Taking the Average Score Each Time
        for roundInd in range(1,numAverage+1):
            # Train the Model with the Training Data
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.5, shuffle= True, stratify=signalLabels)
            self.predictionModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels)
            
            if analyzeType == "Full":
                inputData = signalData; inputLabels = signalLabels
            elif analyzeType == "Test":
                inputData = Testing_Data; inputLabels = Testing_Labels
            else:
                sys.exit("Unsure which data to use for the accuracy map");

            testingLabelsML = self.predictionModel.predictData(inputData)
            # Calculate the Accuracy Matrix
            accMat_Temp = np.zeros((len(machineLearningClasses), len(machineLearningClasses)))
            for ind, channelFeatures in enumerate(inputData):
                # Sum(Row) = # of Gestures Made with that Label
                # Each Column in a Row = The Number of Times that Gesture Was Predicted as Column Label #
                accMat_Temp[inputLabels[ind]][testingLabelsML[ind]] += 1
        
            # Scale Each Row to 100
            for label in range(len(machineLearningClasses)):
                accMat_Temp[label] = 100*accMat_Temp[label]/(np.sum(accMat_Temp[label]))
            
                # Scale Each Row to 100
            for label in range(len(machineLearningClasses)):
                accMat[label] = (accMat[label]*(roundInd-1) + accMat_Temp[label])/roundInd

        
        # Make plot
        fig, ax = plt.subplots()
        fig.set_size_inches(6,6)
        
        # Make heatmap on plot
        im, cbar = createMap.heatmap(accMat, machineLearningClasses, machineLearningClasses, ax=ax,
                           cmap="copper", cbarlabel="Gesture Accuracy (%)")
        createMap.annotate_heatmap(im, accMat, valfmt="{x:.2f}",)
        
        # Style the Fonts
        font = {'family' : 'verdana',
                'weight' : 'bold',
                'size'   : 10}
        matplotlib.rc('font', **font)
        
        # Format, save, and show
        fig.tight_layout()
        plt.savefig(self.saveDataFolder + name + " " + analyzeType + " " + self.modelType + ".png", dpi=130, bbox_inches='tight')
        plt.show()
    
    
    def plotImportance(self, perm_importance_result, featureLabels, name = "Relative Feature Importance"):
        """ bar plot the feature importance """
    
        fig, ax = plt.subplots()
    
        indices = perm_importance_result['importances_mean'].argsort()
        plt.barh(range(len(indices)),
                 perm_importance_result['importances_mean'][indices],
                 xerr=perm_importance_result['importances_std'][indices])
    
        ax.set_yticks(range(len(indices)))
        if len(featureLabels) != 0:
            _ = ax.set_yticklabels(np.array(featureLabels)[indices])
      #      headers = np.array(featureLabels)[indices]
      #      for i in headers:
      #          print('%s Weight: %.5g' % (str(i),v))
        plt.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=150, bbox_inches='tight')
        
    
    def featureImportance(self, signalData, signalLabels, Testing_Data, Testing_Labels, featureLabels = [], numTrials = 100):
        """
        Randomly Permute a Feature's Column and Return the Average Deviation in the Score: |oldScore - newScore|
        NOTE: ONLY Compare Feature on the Same Scale: Time and Distance CANNOT be Compared
        """
      #  if self.modelType not in ["NN"]:
      #      importanceResults = permutation_importance(self.predictionModel.model, signalData, signalLabels, n_repeats=numTrials)
      #      self.plotImportance(importanceResults, featureLabels)
        
        if self.modelType == "RF":
            # get importance
            importance = self.predictionModel.model.feature_importances_
            # summarize feature importance
            for i,v in enumerate(importance):
                if len(featureLabels) != 0:
                    i = featureLabels[i]
                    print('%s Weight: %.5g' % (str(i),v))
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            freq_series = pd.Series(importance)
            ax = freq_series.plot(kind="bar")
            
            # Specify Figure Aesthetics
            ax.set_title("Feature Importance in Model")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Feature Importance")
            
            # Set X-Labels
            if len(featureLabels) != 0:
                ax.set_xticklabels(featureLabels)
                self.add_value_labels(ax)
            # Show Plot
            name = "Feature Importance"
            plt.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=150, bbox_inches='tight')
            pyplot.show()
             
        
        if len(featureLabels) != 0:
            featureLabels = np.array(featureLabels)
            print("Entering SHAP Analysis")
            # Make Output Folder for SHAP Values
            os.makedirs(self.saveDataFolder + "SHAP Values/", exist_ok=True)
            # Create Panda DataFrame to Match Input Type for SHAP
            testingDataPD = pd.DataFrame(signalData, columns = featureLabels)
            
            # More General Explainer
            explainerGeneral = shap.Explainer(self.predictionModel.model.predict, testingDataPD)
            shap_valuesGeneral = explainerGeneral(testingDataPD)
            
            # MultiClass (Only For Tree)
            if self.modelType == "RF":
                explainer = shap.TreeExplainer(self.predictionModel.model)
                shap_values = explainer.shap_values(testingDataPD)
                
                # misclassified = Testing_Labels != self.predictionModel.model.predict(signalData)
                # shap.multioutput_decision_plot(list(explainer.expected_value), list(shap_values), row_index = 0, features = testingDataPD, feature_names = list(featureLabels), feature_order = "importance", highlight = misclassified)
                # #shap.decision_plot(explainer.expected_value, shap_values, features = testingDataPD, feature_names = list(featureLabels), feature_order = "importance", highlight = misclassified)

            else:
                # Calculate Shap Values
                explainer = shap.KernelExplainer(self.predictionModel.model.predict, testingDataPD)
                shap_values = explainer.shap_values(testingDataPD, nsamples=len(signalData))
            
            # Specify Indivisual Sharp Parameters
            dataPoint = 10
            featurePoint = 20
            
            # Summary Plot
            name = "Summary Plot"
            summaryPlot = plt.figure()
            if self.modelType == "RF":
                shap.summary_plot(shap_values, testingDataPD, plot_type="bar", class_names=self.machineLearningClasses, feature_names = featureLabels, max_display = len(featureLabels))
            else:
                shap.summary_plot(shap_valuesGeneral, testingDataPD, class_names=self.machineLearningClasses, feature_names = featureLabels)
            summaryPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Dependance Plot
            name = "Dependance Plot"
            dependancePlot, dependanceAX = plt.subplots()
            shap.dependence_plot(featurePoint, shap_values, features = testingDataPD, feature_names = featureLabels, ax = dependanceAX)
            dependancePlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Indivisual Force Plot
            name = "Indivisual Force Plot"
            forcePlot = shap.force_plot(explainer.expected_value, shap_values[dataPoint,:], features = np.round(testingDataPD.iloc[dataPoint,:], 5), feature_names = featureLabels, matplotlib = True, show = False)
            forcePlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Full Force Plot. NOTE: CANNOT USE matplotlib = True to See
            name = "Full Force Plot"
            fullForcePlot = shap.force_plot(explainer.expected_value, shap_values, features = testingDataPD, feature_names = featureLabels, matplotlib = False, show = True)
            shap.save_html(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".htm", fullForcePlot)
            
            # WaterFall Plot
            name = "Waterfall Plot"
            waterfallPlot = plt.figure()
            #shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0dataPoint], feature_names = featureLabels, max_display = len(featureLabels), show = True)
            shap.plots.waterfall(shap_valuesGeneral[dataPoint],  max_display = len(featureLabels), show = True)
            waterfallPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
 
            # Indivisual Decision Plot
            misclassified = Testing_Labels != self.predictionModel.model.predict(signalData)
            decisionFolder = self.saveDataFolder + "SHAP Values/Decision Plots/"
            os.makedirs(decisionFolder, exist_ok=True) 
            for dataPoint1 in range(min(50, len(testingDataPD))):
                name = "Indivisual Decision Plot DataPoint Num " + str(dataPoint1)
                decisionPlot = plt.figure()
                shap.decision_plot(explainer.expected_value, shap_values[dataPoint1,:], features = testingDataPD.iloc[dataPoint1,:], feature_names = featureLabels, feature_order = "importance", highlight = misclassified[dataPoint1])
                decisionPlot.savefig(decisionFolder + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Decision Plot
            name = "Decision Plot"
            decisionPlotOne = plt.figure()
            shap.decision_plot(explainer.expected_value, shap_values, features = testingDataPD, feature_names = featureLabels, feature_order = "importance", highlight = misclassified)
            decisionPlotOne.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Bar Plot
            name = "Bar Plot"
            barPlot = plt.figure()
            shap.plots.bar(shap_valuesGeneral, max_display = len(featureLabels), show = True)
            barPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)

            # HeatMap Plot
            name = "Heatmap Plot"
            heatmapPlot = plt.figure()
            shap.plots.heatmap(shap_valuesGeneral, max_display = len(featureLabels), show = True, instance_order=shap_valuesGeneral.sum(1))
            heatmapPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
                
            # Scatter Plot
            scatterFolder = self.saveDataFolder + "SHAP Values/Scatter Plots/"
            os.makedirs(scatterFolder, exist_ok=True)
            for featurePoint1 in range(len(featureLabels)):
                for featurePoint2 in range(len(featureLabels)):
                    name = "Scatter Plot (" + featureLabels[featurePoint1] + " VS " + featureLabels[featurePoint2] + ")" 
                    scatterPlot, scatterAX = plt.subplots()
                    shap.plots.scatter(shap_valuesGeneral[:, featureLabels[featurePoint1]], color = shap_valuesGeneral[:, featureLabels[featurePoint2]], ax = scatterAX)
                    scatterPlot.savefig(scatterFolder + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
            
            # Monitoring Plot (The Function is a Beta Test As of 11-2021)
            if len(signalData) > 150:  # They Skip Every 50 Points I Believe
                name = "Monitor Plot"
                monitorPlot = plt.figure()
                shap.monitoring_plot(featurePoint, shap_values, features = testingDataPD, feature_names = featureLabels)
                monitorPlot.savefig(self.saveDataFolder + "SHAP Values/" + name + " " + self.modelType + ".png", bbox_inches='tight', dpi=300)
                        
    def add_value_labels(self, ax, spacing=5):
        """Add labels to the end of each bar in a bar chart.
    
        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
        """
    
        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
    
            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'
    
            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'
    
            # Use Y value as label and format number with one decimal place
            label = "{:.3f}".format(y_value)
    
            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.
        
        