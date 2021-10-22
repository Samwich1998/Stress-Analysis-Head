"""
Code Written by Samuel Solomon

SKLearn SVM Guide: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Import Basic Modules
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Import Modules for Plotting
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt

# Import Machine Learning Modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold

# Import Python Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network
    
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class randomForest:
    def __init__(self, modelPath):        
        # Initialize Model
        if os.path.exists(modelPath):
            # If Model Exists, Load it
            self.loadModel(modelPath)
        else:
            # Else, Create the Model
            self.createModel()
    
    def saveModel(self, modelPath = "./SVM.sav"):
        joblib.dump(self.model, modelPath)    
    
    def loadModel(self, modelPath):
        with open(modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        print("Random Forest Model Loaded")
            
    def createModel(self):
        """
        criteria: “gini” for the Gini impurity and “entropy” for the information gain.
        """
        self.model = RandomForestClassifier(n_estimators=100,criterion='gini', max_depth=None, min_samples_split=2,
                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',max_leaf_nodes=None,
                    min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                    verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
        print("Random Forest Model Created")
        
    def trainModel(self, Training_Data, Training_Labels, newData = [], newLabels = [], scoreType = "Score:"):
        # Train the Model
        self.model.fit(Training_Data, Training_Labels)
        # Score the Model
        if len(newData) > 0:
            self.scoreModel(newData, newLabels, scoreType)
    
    def scoreModel(self, signalData, signalLabels, scoreType = "Score:"):
        print(scoreType, self.model.score(signalData, signalLabels))
        
        # Evaluate the model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
        n_scores = cross_val_score(self.model, signalData, signalLabels, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # report performance
        print('Cross-Validation Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


    
    def predictData(self, New_Data):
        # Predict Label based on new Data
        return self.model.predict(New_Data)
        
    def accuracyDistributionPlot(self, signalData, trueLabels, predicatedLabels, signalLabelsTitles, saveFolder = "./Output Data/Machine Learning Results/", name = "Accuracy Distribution"):
        
        # Calculate the Accuracy Matrix
        accMat = np.zeros((len(signalLabelsTitles), len(signalLabelsTitles)))
        for ind, channelFeatures in enumerate(signalData):
            # Sum(Row) = # of Gestures Made with that Label
            # Each Column in a Row = The Number of Times that Gesture Was Predicted as Column Label #
            accMat[trueLabels[ind]][predicatedLabels[ind]] += 1
        
        # Scale Each Row to 100
        for label in range(len(signalLabelsTitles)):
            accMat[label] = 100*accMat[label]/np.sum(accMat[label])
        
        # Make plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8,8)
        
        # Make heatmap on plot
        im, cbar = createMap.heatmap(accMat, signalLabelsTitles, signalLabelsTitles, ax=ax,
                           cmap="copper", cbarlabel="Label Accuracy (%)")
        createMap.annotate_heatmap(im, accMat, valfmt="{x:.2f}",)
        
        # Style the Fonts
        font = {'family' : 'verdana',
                'weight' : 'bold',
                'size'   : 9}
        matplotlib.rc('font', **font)
                
        # Format, save, and show
        fig.tight_layout()
        plt.savefig(saveFolder + name + ".png", dpi=300, bbox_inches='tight')
        plt.show()


    def plotImportance(self, perm_importance_result, headerTitles):
        """ bar plot the feature importance """
    
        fig, ax = plt.subplots()
    
        indices = perm_importance_result['importances_mean'].argsort()
        plt.barh(range(len(indices)),
                 perm_importance_result['importances_mean'][indices],
                 xerr=perm_importance_result['importances_std'][indices])
    
        ax.set_yticks(range(len(indices)))
        if headerTitles:
            _ = ax.set_yticklabels(np.array(headerTitles)[indices])
    
    def featureImportance(self, signalData, signalLabels, headerTitles = [], numTrials = 30):
        """
        Randomly Permute a Feature's Column and Return the Average Deviation in the Score: |oldScore - newScore|
        NOTE: ONLY Compare Feature on the Same Scale: Time and Distance CANNOT be Compared
        """
        importanceResults = permutation_importance(self.model, signalData, signalLabels, n_repeats=numTrials)
        self.plotImportance(importanceResults, headerTitles)
        
        
        # get importance
        importance = self.model.feature_importances_
        # summarize feature importance
        for i,v in enumerate(importance):
            if headerTitles:
                i = headerTitles[i]
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
        if headerTitles:
            ax.set_xticklabels(headerTitles)
            self.add_value_labels(ax)
        # Show Plot
        pyplot.show()
        
        
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


