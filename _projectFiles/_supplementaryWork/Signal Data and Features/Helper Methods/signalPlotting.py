
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import Files for Machine Learning
sys.path.append(os.path.dirname(__file__) + "/../../../Helper Files/Machine Learning/") 
from _dataPreparation import standardizeData

# Import Files for Machine Learning
sys.path.append(os.path.dirname(__file__) + "/../../../Helper Files/Machine Learning/Model Specifications/")
import _compileModelInfo    # Functions with model information

# -------------------------------------------------------------------------- #
# --------------------- Plotting the data and Features --------------------- #

class plotData:
    def __init__(self):
        # Change the font style and size.
        plt.rcParams['font.sans-serif'] = ['Arial']  # Additional fallback option
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12
        
        self.savingFolder = "./Figures/"
        plt.ioff() # prevent memory leaks; plt.ion()
        
        self.modelInfoClass = _compileModelInfo.compileModelInfo()
        
        self.rawDataOrder = ['EOG', 'EEG', 'EDA', 'Temp']
        self.rawDataColors = [
                        '#3498db',  # Blue shades
                        '#9ED98F',  # Green shades
                        '#918ae1',  # Purple shades
                        '#fc827f',   # Red shades
                    ]
        
        self.activityOrder = ['CPT', 'Exersice', 'Music', 'VR']
        self.activityColors = [
                        '#3498db',  # Blue shades
                        '#fc827f',   # Red shades
                        '#9ED98F',  # Green shades
                        '#918ae1',  # Purple shades
                    ]
        
        
    def clearFigure(self, fig, legend):
        plt.show()
        if legend != None: legend.remove()
        # Clear plots
        if fig != None: fig.clear()
        plt.cla(); plt.clf()
        plt.close(fig); plt.close('all')
        
    # ---------------------------------------------------------------------- #
    # ------------------------- Feature Label Plots ------------------------ #
        
    def plotSurveyInfo(self, readDatas, surveyQuestions):
        # Initialize saving folder
        surveyInfoDirectory = self.savingFolder + "/Survey Information/"
        os.makedirs(surveyInfoDirectory, exist_ok=True)
        
        print("\nPlotting the Survey Information")
        # Initialize holders of the survey information
        finalLabels = []; experimentalOrder = []
        surveyAnswersList = []
        
        # For each analysis file.
        for fileInd in range(len(readDatas)):
            readData = readDatas[fileInd]
            
            # -------------------------------------------------------------- #
            # -------------- Extract Data into this Namespace -------------- #

            # Extract the feature labels.
            surveyAnswerTimes = np.array(readData.surveyAnswerTimes) # A list of times associated with each feature label.
            currentSurveyAnswersList = np.array(readData.surveyAnswersList) 
            # Extract the experiment information
            experimentTimes = np.array(readData.experimentTimes)
            experimentNames = np.array(readData.experimentNames)
            
            featureLabelTypes, finalLabels = self.modelInfoClass.extractFinalLabels(currentSurveyAnswersList, finalLabels)
            
            # Keep track of all the experiments.
            experimentalOrder.extend(experimentNames)
            surveyAnswersList.extend(currentSurveyAnswersList)   
        surveyAnswersList = np.array(surveyAnswersList)
        
        # Get all the activity information from the experiment.
        activityNames, activityLabels = self.modelInfoClass.extractActivityInformation(experimentalOrder, distinguishBaselines = True)
        experimentalOrder_byActivity = self.modelInfoClass.labelExperimentalOrder(experimentalOrder, distinguishBaselines = True)

        # ------------------------------------------------------------------ #
        # -------------------- Plot the aligned features ------------------- #
                                        
        # # Initialize the figure.
        # xlims = [[4, 26], [4, 26], [19, 81]]
        # figSize = (1, 3)
                
        # # For each survey administered.
        # for labelTypeInd in range(len(featureLabelTypes)):
        #     surveyName = featureLabelTypes[labelTypeInd]
        #     surveyAnswers = finalLabels[labelTypeInd]
            
        #     # Compartmentalized each label into its respective activity.
        #     allDataX, allDataY = self.organizeSurveyAnswers_toPlot(surveyAnswers, experimentalOrder_byActivity)
            
        #     fig, axes = plt.subplots(4, 1, figsize=figSize, sharex=True)    
        #     savePlottingFolder = surveyInfoDirectory + f"{surveyName}/"
        #     os.makedirs(savePlottingFolder, exist_ok=True)
    
        #     for axisInd in range(4):
        #         # Set plotting labels/aesthetics.
        #         axes[axisInd].set_ylabel(self.activityOrder[axisInd])
        #         axes[axisInd].set_yticks([])  # Hide y-axis ticks
        #         axes[axisInd].set_ylim([-1, 1])  # Hide y-axis ticks
        #         axes[axisInd].set_xlim(xlims[labelTypeInd])  # Hide y-axis ticks
                
        #         allTrialsX = []; allTrialsY = []
        #         for segmentInd in range(3):
        #             xData = allDataX[axisInd][segmentInd]
        #             yData = allDataY[axisInd][segmentInd]
                    
        #             allTrialsX.append()
                    
        #             axes[axisInd].plot(xData, yData, 'o', markersize=1, color=self.rawDataColors[axisInd])
        
                        
        #     # Set plotting labels/aesthetics.
        #     plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
        #     axes[-1].set_xlabel(f"{surveyName} Results")
            
        #     # Save and show the figure.
        #     plt.savefig(savePlottingFolder + f"{surveyName} Results.pdf", dpi=500, bbox_inches='tight')
        #     self.clearFigure(fig, None)
        
        # ------------------------------------------------------------------ #
        # -------------------- Plot the aligned features ------------------- #
                                        
        # # Initialize the figure.
        # ylims = [[4, 26], [4, 26], [19, 81]]
        # bins = [8, 8, 15]
        # figSize = (1, 3)
                
        # # For each survey administered.
        # for labelTypeInd in range(len(featureLabelTypes)):
        #     surveyName = featureLabelTypes[labelTypeInd]
        #     surveyAnswers = finalLabels[labelTypeInd]
            
        #     # Compartmentalized each label into its respective activity.
        #     allDataY, activityGroups = self.organizeSurveyAnswers_forCurves(surveyAnswers, experimentalOrder_byActivity)
            
        #     fig, axes = plt.subplots(len(activityGroups), 1, figsize=figSize, sharex=True)    
        #     savePlottingFolder = surveyInfoDirectory + f"{surveyName}/"
        #     os.makedirs(savePlottingFolder, exist_ok=True)
        
        #     for axisInd in range(len(activityGroups)):
        #         # Set plotting labels/aesthetics.
        #         axes[axisInd].set_ylabel(self.activityOrder[axisInd])
        #         # axes[axisInd].set_yticks([])  # Hide y-axis ticks
        #         axes[axisInd].set_ylim(ylims[labelTypeInd])  # Hide y-axis ticks
                
        #         for pointInd in range(len(allDataY[axisInd])):
        #             axes[axisInd].plot(allDataY[axisInd][pointInd], linewidth=1, color=self.rawDataColors[axisInd], alpha=0.3)
                                                               
        #     # Set plotting labels/aesthetics.
        #     plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
        #     axes[-1].set_xlabel(f"{surveyName} Results")
            
        #     # Save and show the figure.
        #     plt.savefig(surveyInfoDirectory + f"{surveyName} Results.pdf", dpi=500, bbox_inches='tight')
        #     self.clearFigure(fig, None)
            
            
            
        
        # Initialize the figure.
        ylims = [[4, 26], [4, 26], [19, 81]]
        bins = [8, 8, 15]
        figSize = (2, 3)
        
        fig, axes = plt.subplots(len(featureLabelTypes), 2, figsize=figSize, sharex=True, sharey='row')    
                
        # For each survey administered.
        for labelTypeInd in range(len(featureLabelTypes)):
            surveyName = featureLabelTypes[labelTypeInd]
            surveyAnswers = finalLabels[labelTypeInd]
            
            # Compartmentalized each label into its respective activity.
            allDataY, activityGroups = self.organizeSurveyAnswers_forCurves(surveyAnswers, experimentalOrder_byActivity)
            axes[labelTypeInd][0].set_ylim(ylims[labelTypeInd])  # Hide y-axis ticks
            axes[labelTypeInd][1].set_ylim(ylims[labelTypeInd])  # Hide y-axis ticks
            
            savePlottingFolder = surveyInfoDirectory + f"{surveyName}/"
            os.makedirs(savePlottingFolder, exist_ok=True)
            
            for groupInd in range(2):
                for pointInd in range(len(allDataY[groupInd])):
                    axes[labelTypeInd][groupInd].plot(allDataY[groupInd][pointInd], linewidth=1, color=self.rawDataColors[groupInd], alpha=0.7)
                                                               
            # Set plotting labels/aesthetics.
            plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
            # axes[-1].set_xlabel(f"{surveyName} Results")
            
            # Save and show the figure.
        
        plt.savefig(surveyInfoDirectory + f"Labels Results.pdf", dpi=500, bbox_inches='tight')
        self.clearFigure(fig, None)
                
                
        # # ------------------------------------------------------------------ #
        # # -------------------- Plot the aligned features ------------------- #
        
        # violinTitles = ["STAI Emotions", "Positive STAI Emotions", "Negative STAI Emotions", "PANAS Emotions", "Positive PANAS Emotions", "Negative PANAS Emotions", "All Emotions"]
        # allEmotionInds = [self.modelInfoClass.staiInds, self.modelInfoClass.staiInds_Pos, self.modelInfoClass.staiInds_Neg, 
        #                   self.modelInfoClass.panasInds, self.modelInfoClass.posAffectInds, self.modelInfoClass.negAffectInds,
        #                   self.modelInfoClass.allInds]
        
        # for violinInd in range(len(allEmotionInds)):
        #     emotionInds =  allEmotionInds[violinInd]
            
        #     # Get the emotion names and answers.
        #     currentSurveyQuestions = surveyQuestions[emotionInds]
        #     currentSurveyAnswers = surveyAnswersList[:, emotionInds]
            
        #     # Flatten the answers to make it 1-dimensional
        #     flattened_answers = currentSurveyAnswers.T.ravel()
            
        #     # Repeat each question to match the length of flattened_answers
        #     questions_repeated = np.repeat(currentSurveyQuestions, len(currentSurveyAnswers))
            
        #     # melted_data = [(val, emotion) for val_list, emotion in zip(currentSurveyAnswers.T, currentSurveyQuestions) for val in val_list]
        #     melted_data = pd.DataFrame({"Emotion": questions_repeated, "Value": flattened_answers})
            
        #     # Create a combined violin plot
        #     plt.figure(figsize=(12, 8))  # Set the figure size
            
        #     sns.violinplot(x="Emotion", y="Value", data=melted_data, inner='box', scale='count')
        #     plt.title(f'{violinTitles[violinInd]}')
        #     plt.xlabel('Emotions')
        #     plt.ylabel('User Ratings')
            
        #     plt.tight_layout()  # Adjust layout for better spacing
        #     plt.xticks(rotation=45)  # Rotate x-axis labels if needed
            
        #     # Save and show the figure.
        #     plt.savefig(surveyInfoDirectory + f"{violinTitles[violinInd]}.pdf", dpi=500, bbox_inches='tight')
        #     self.clearFigure(None, None)

    def organizeSurveyAnswers(self, surveyAnswers, activityNames, activityLabels):
        assert len(surveyAnswers) == len(activityLabels)
        activitySurveyAnswers = [[] for _ in range(len(activityNames))]
        
        # For each type of activity.
        for activityInd in range(len(activityLabels)):
            activityIndex = activityLabels[activityInd]
            surveyAnswer = surveyAnswers[activityInd]
            
            # Get the activityIndex and store the activity
            activitySurveyAnswers[activityIndex].append(surveyAnswer)
        
        return activitySurveyAnswers
    
    def organizeSurveyAnswers_toPlot(self, surveyAnswers, experimentalOrder_byActivity):
        activityGroups = ['CPT', 'Exercise', 'Music', 'VR']
        segments = ["Recovery", "Activity", "Baseline"]

        xData = [[[],[],[]] for _ in range(len(activityGroups))]        
        yData = [[[],[],[]] for _ in range(len(activityGroups))]        
        # For each type of activity.
        for experimentInd in range(len(experimentalOrder_byActivity)):
            experimentName = experimentalOrder_byActivity[experimentInd]
            activityGroup, activityName = experimentName.split(" ")
            surveyAnswer = surveyAnswers[experimentInd]
                        
            # Get the placement information
            activityGroupInd = activityGroups.index(activityGroup)
            segmentInd = segments.index(activityName)
            # Store the final label
            xData[activityGroupInd][segmentInd].append(surveyAnswer)
            yData[activityGroupInd][segmentInd].append((segmentInd - 1)/2)
            
        return xData, yData, segments
    
    def organizeSurveyAnswers_forCurves(self, surveyAnswers, experimentalOrder_byActivity):
        activityGroups = ['CPT', 'Exercise', 'Music', 'VR']
        segments = ["Recovery", "Activity", "Baseline"]

        # xData = [[] for _ in range(len(activityGroups))]        
        yData = [[] for _ in range(len(activityGroups))]        
        # For each type of activity.
        for experimentInd in range(len(experimentalOrder_byActivity)):
            experimentName = experimentalOrder_byActivity[experimentInd]
            activityGroup, activityName = experimentName.split(" ")
            surveyAnswer = surveyAnswers[experimentInd]
                        
            # Get the placement information
            activityGroupInd = activityGroups.index(activityGroup)
            segmentInd = segments.index(activityName)
            
            # If a new experiment
            if segmentInd == 2:
                # xData[activityGroupInd].append([])
                yData[activityGroupInd].append([])
            
            # Store the final label
            # xData[activityGroupInd][-1].append(surveyAnswer)
            yData[activityGroupInd][-1].append(surveyAnswer)
            
        return yData, activityGroups
        
    # ---------------------------------------------------------------------- #
    # ---------------------------- Signal Plots ---------------------------- #

    def plotFigures(self, readDatas, testDataExcelFiles, featureNames, biomarkerFeatureNames):
        print("\nPlotting data")
        # For each analysis file.
        for fileInd in range(len(testDataExcelFiles)):
            testDataExcelFile = testDataExcelFiles[fileInd]
            saveName  = testDataExcelFile.split("/")[-1].split(".")[0]
            readData = readDatas[fileInd]
            
            print(f"\tPlotting data for {saveName}")
            
            # Initialize saving folder
            savePlottingFolder = self.savingFolder + f"/{saveName}/"
            os.makedirs(savePlottingFolder, exist_ok=True)
            
            # -------------------------------------------------------------- #
            # -------------- Extract Data into this Namespace -------------- #
            
            # # Extract raw features
            eogFeatures, eegFeatures, edaFeatures, tempFeatures = readData.rawFeatureHolder
            eogFeatureTimes, eegFeatureTimes, edaFeatureTimes, tempFeatureTimes = readData.rawFeatureTimesHolder
            
            # Extract the features
            alignedFeatures = np.array(readData.alignedFeatures)
            alignedFeatureTimes = np.array(readData.alignedFeatureTimes)
            
            # Extract the feature labels.
            surveyAnswerTimes = np.array(readData.surveyAnswerTimes) # A list of times associated with each feature label.
            surveyAnswersList = np.array(readData.surveyAnswersList) 
            # Extract the experiment information
            experimentTimes = np.array(readData.experimentTimes)
            experimentNames = np.array(readData.experimentNames)
            
            modelInfoClass = _compileModelInfo.compileModelInfo()
            featureLabelTypes, finalLabels = modelInfoClass.extractFinalLabels(surveyAnswersList, [])
            
            # General parameters.
            numFeatureSignals = len(readData.analysisList_WithFeatures)
            
            # Organize all the data events.
            rawDataTimes, standardizedRawData, rawData = self.getRawData(readData) # Get the raw data.
            allFilteredTimes, allFilteredData = self.getFilteredData(readData, rawDataTimes, standardizedRawData, rawData) # Get the filtered signal data.
            alignedFeatureTimes, alignedFeatures = self.getAlignedFeatures(readData, experimentTimes) # get the aligned features.
            
            # -------------------------------------------------------------- #
            # ------------------ Plot the aligned features ----------------- #
                
            plottingFeatureNames = ["blinkDuration_EOG", "halfClosedTime_EOG", 
                                    "alphaPower_EEG", "engagementLevelEst_EEG", 
                                    "hjorthActivity_EDA", "firstDerivVariance_EDA", 
                                    "mean_TEMP", "firstDerivativeMean_TEMP"]
            shortenedNames = ["BD", "HCT", "HA", "EL", "HA", "FDV", "M", "FDM"]
            
            plottingColors = [
                '#3498db', '#3498db',  # Blue shades
                '#9ED98F', '#9ED98F',  # Green shades
                '#918ae1', '#918ae1',  # Purple shades
                '#fc827f', '#fc827f'   # Red shades
            ]
        
            alignedFeatureInds = [np.where(plottingFeatureNames[i] == featureNames)[0][0] for i in range(len(plottingFeatureNames))]
            
            # Initialize the figure.
            figSize = (min(3, round(len(alignedFeatureTimes) / 500)), 6)
            fig, axes = plt.subplots(len(plottingFeatureNames), 1, figsize=figSize, sharex=True)
            yLim = [-3.5, 3.5]
        
            # For each axis in the plot.
            for axisInd in range(len(plottingFeatureNames)):
                featureInd = alignedFeatureInds[axisInd]
                featureName = shortenedNames[axisInd]
                color = plottingColors[axisInd]
                
                # Plot all the aligned features together.
                axes[axisInd].plot(alignedFeatureTimes, alignedFeatures[:, featureInd], linewidth=1, color=color)
                axes[axisInd].set_yticks([])  # Hide y-axis ticks
                axes[axisInd].set_ylabel(featureName)
                axes[axisInd].set_ylim(yLim)  # Hide y-axis ticks 
                
            axes[-1].set_xlabel('Time')
            plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
            plt.savefig(savePlottingFolder + f"{saveName} no experiments.pdf", dpi=500, bbox_inches='tight')
        
            # Shade in the experimental sections.
            self.addExperimentalSections(axes, experimentTimes, surveyAnswerTimes, yLim)
            plt.savefig(savePlottingFolder + f"{saveName}.pdf", dpi=500, bbox_inches='tight')
            self.clearFigure(fig, None)
            
            # -------------------------------------------------------------- #
            # -------------------- Plot Single features -------------------- #
            
            figSize = (min(3, round(len(alignedFeatureTimes) / 500)), 1)
            for i in range(len(plottingFeatureNames)):
                singleAlignedFig, singleAlignedAx = plt.subplots(1, 1, figsize=figSize, sharex=True)
        
                featureInd = alignedFeatureInds[i]
                featureName = shortenedNames[i]
                color = plottingColors[i]
        
                singleAlignedAx.plot(alignedFeatureTimes, alignedFeatures[:, featureInd], linewidth=1, color=color)
                # Remove plotting border.
                self.removeBorder(singleAlignedAx)
                # Set plotting labels/aesthetics.
                plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
            
                # Shade in the experimental sections.
                #self.addExperimentalSections(axes, experimentTimes, surveyAnswerTimes, yLim)
        
                # ax.set_xlabel('Time')
                plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
                plt.savefig(savePlottingFolder + f"{featureName}_{color}.pdf", dpi=500, bbox_inches='tight')
                self.clearFigure(singleAlignedFig, None)
                
            # -------------------------------------------------------------- #
            # ---------------------- Plot All Features --------------------- #
            
            plottingFeatureNames = ["blinkDuration_EOG", 
                                    "alphaPower_EEG", 
                                    "hjorthActivity_EDA", 
                                    "mean_TEMP"]
            shortenedNames = ["BD", "HA", "HA", "M"]
            
            plottingColors = [
                '#3498db',  # Blue shades
                '#9ED98F',  # Green shades
                '#918ae1',  # Purple shades
                '#fc827f',   # Red shades
            ]
        
            alignedFeatureInds = [np.where(plottingFeatureNames[i] == featureNames)[0][0] for i in range(len(plottingFeatureNames))]
            
            figSize_AllFeatures = (min(3, round(len(alignedFeatureTimes) / 500)), 3)
            fig, axes = plt.subplots(len(shortenedNames), 1, figsize=figSize_AllFeatures, sharex=True)
            
            # Plot the data
            self.plotColumnData(axes, [alignedFeatureTimes]*4, alignedFeatures[:, alignedFeatureInds].T, experimentTimes, 
                                surveyAnswerTimes, shortenedNames, plottingColors, yLim, ['-']*4, savePlottingFolder)
            
            # Set plotting labels/aesthetics.
            axes[0].set_title("Aligned Features")
            plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
            axes[-1].set_xlabel('Time')
            
            # Save and show the figure.
            plt.savefig(savePlottingFolder + "Aligned Features.pdf", dpi=500, bbox_inches='tight')
            self.clearFigure(fig, None)
            
            
            # -------------------------------------------------------------- #
            # ------------------- Plot Features with Raw ------------------- #
            
            # plottingFeatureNames = ["blinkDuration_EOG", 
            #                         "hjorthActivity_EEG", 
            #                         "hjorthActivity_EDA", 
            #                         "mean_TEMP"]
            plottingFeatureNames = ["blinkDuration_EOG", 
                                    "alphaPower_EEG", 
                                    "hjorthActivity_EDA", 
                                    "mean_TEMP"]
            shortenedNames = ["EOG", "BD", "EEG", "HA", "EDA", "HA", "Temp", "M"]
            
            plottingColors = [
                '#3498db', '#3498db',  # Blue shades
                '#9ED98F', '#9ED98F',  # Green shades
                '#918ae1', '#918ae1',  # Purple shades
                '#fc827f', '#fc827f',  # Red shades
            ]
            
            alignedFeatureInds = [np.where(plottingFeatureNames[i] == featureNames)[0][0] for i in range(len(plottingFeatureNames))]
            
            columnData = [
                allFilteredData[0], alignedFeatures[:, alignedFeatureInds[0]].T,
                allFilteredData[1], alignedFeatures[:, alignedFeatureInds[1]].T,
                allFilteredData[2], alignedFeatures[:, alignedFeatureInds[2]].T,
                allFilteredData[3], alignedFeatures[:, alignedFeatureInds[3]].T,
            ]
            
            columnTimes = [
                allFilteredTimes[0], alignedFeatureTimes,
                allFilteredTimes[1], alignedFeatureTimes,
                allFilteredTimes[2], alignedFeatureTimes,
                allFilteredTimes[3], alignedFeatureTimes,
            ]
            
            figSize_AllFeatures = (min(3, round(len(alignedFeatureTimes) / 500)), 3)
            fig, axes = plt.subplots(len(shortenedNames), 1, figsize=figSize_AllFeatures, sharex=True)
            
            # Plot the data
            self.plotColumnData(axes, columnTimes, columnData, experimentTimes,  surveyAnswerTimes, shortenedNames, plottingColors, yLim, ['-']*len(axes), savePlottingFolder)
            
            # Set plotting labels/aesthetics.
            plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
            axes[-1].set_xlabel('Time')
            
            # Save and show the figure.
            plt.savefig(savePlottingFolder + "Filtered with Features.pdf", dpi=500, bbox_inches='tight')
            self.clearFigure(fig, None)
            
            # -------------------------------------------------------------- #
            # ---------------------- Plot Raw Features --------------------- #
            
            shortenedNames = ["BD", "HA", "HA", "M"]
            
            plottingColors = [
                '#3498db',  # Blue shades
                '#9ED98F',  # Green shades
                '#918ae1',  # Purple shades
                '#fc827f',  # Red shades
            ]
            
            # Get all the raw data.
            allRawFeatureTimes, allRawFeatures = self.getRawFeatures(readData, experimentTimes, biomarkerFeatureNames, plottingFeatureNames) # get the raw feature information.
            
            # figSize = (min(3, round(len(alignedFeatureTimes) / 500)), 3)
            fig, axes = plt.subplots(len(shortenedNames), 1, figsize=figSize_AllFeatures, sharex=True)
            
            for rawFeatureInd in range(numFeatureSignals):
                # Plot the data.
                axes[rawFeatureInd].plot(allRawFeatureTimes[rawFeatureInd], allRawFeatures[rawFeatureInd], 'o', markersize=0.5, color=plottingColors[rawFeatureInd])
                # Set plotting labels/aesthetics.
                axes[rawFeatureInd].set_ylabel(shortenedNames[rawFeatureInd])
                axes[rawFeatureInd].set_yticks([])  # Hide y-axis ticks   
                axes[rawFeatureInd].set_ylim(yLim)  # Hide y-axis ticks 
                
            # Shade in the experimental sections.
            self.addExperimentalSections(axes, experimentTimes, surveyAnswerTimes, yLim)  

            # Set plotting labels/aesthetics.
            axes[0].set_title("Raw Features")
            plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
            axes[-1].set_xlabel('Time')
            
            # Save and show the figure.
            plt.savefig(savePlottingFolder + "Raw Features.pdf", dpi=500, bbox_inches='tight')
            self.clearFigure(fig, None)
            
                            
            # -------------------------------------------------------------- #
            # ----------------------- Plot Single Raw ---------------------- #
                
            figSize = (min(3, round(len(alignedFeatureTimes) / 500)), 3)            
            for rawFeatureInd in range(len(readData.rawFeatureHolder)):
                # Plot the data.
                fig, ax = plt.subplots(1, 1, figsize=figSize, sharex=True)
                ax.plot(allRawFeatureTimes[rawFeatureInd], allRawFeatures[rawFeatureInd], linewidth=1, color=plottingColors[rawFeatureInd], alpha = 1)
                
                # Remove plotting border.
                self.removeBorder(ax)
                # Set plotting labels/aesthetics.
                plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
                
                # Save and show the figure.
                plt.savefig(savePlottingFolder + f"{shortenedNames[rawFeatureInd]} Raw Features.pdf", dpi=500, bbox_inches='tight')
                self.clearFigure(fig, None)
                
            # -------------------------------------------------------------- #
            # -------------------- Plot Compiled Features ------------------ #
            
            # Get all the raw data.
            allCompiledFeatureTimes, allCompiledFeatures = self.getCompiledFeatures(readData, experimentTimes, biomarkerFeatureNames, plottingFeatureNames) # Get the compiled features.
            
            # figSize = (min(3, round(len(alignedFeatureTimes) / 500)), 3)
            fig, axes = plt.subplots(len(shortenedNames), 1, figsize=figSize_AllFeatures, sharex=True)
            
            for analysisInd in range(len(readData.analysisList_WithFeatures)):
                compiledFeatures = allCompiledFeatures[analysisInd]
                compiledFeatureTimes = allCompiledFeatureTimes[analysisInd]
                endTimeIndex = self.findTimeIndex(alignedFeatureTimes[-1], compiledFeatureTimes)

                # Plot the data.
                axes[analysisInd].plot(compiledFeatureTimes[:endTimeIndex], compiledFeatures[:endTimeIndex], 'o', markersize=0.5, color=plottingColors[analysisInd])
                # Set plotting labels/aesthetics.
                axes[analysisInd].set_ylabel(shortenedNames[analysisInd])
                axes[analysisInd].set_yticks([])  # Hide y-axis ticks 
                axes[analysisInd].set_ylim(yLim)  # Hide y-axis ticks 
                
            # Shade in the experimental sections.
            self.addExperimentalSections(axes, experimentTimes, surveyAnswerTimes, yLim)  
        
            # Set plotting labels/aesthetics.
            axes[0].set_title("Compiled Features")
            plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
            axes[-1].set_xlabel('Time')
            
            # Save and show the figure.
            plt.savefig(savePlottingFolder + "Compiled Features.pdf", dpi=500, bbox_inches='tight')
            self.clearFigure(fig, None)
                                                    
            # -------------------------------------------------------------- #
            # ------------------------ Plot Raw Data ----------------------- #
            
            yLim = [-4.5, 4.5]           
            shortenedNames = ["EOG", "EEG", "EDA", "Temp"]     
            plottingColors = [
                '#3498db',  # Blue shades
                '#9ED98F',  # Green shades
                '#918ae1',  # Purple shades
                '#fc827f',   # Red shades
            ]
            
            # figSize = (((rawDataTimes[-1] - rawDataTimes[0]) / 600), 3)
            fig, axes = plt.subplots(len(shortenedNames), 1, figsize=figSize_AllFeatures, sharex=True)
            
            # Plot the data
            self.plotColumnData(axes, [rawDataTimes]*4, standardizedRawData, experimentTimes, surveyAnswerTimes,
                                shortenedNames, plottingColors, yLim, ['-']*4, savePlottingFolder)
        
            axes[-1].set_xlabel('Time')
            axes[0].set_title("Raw Signals")
            plt.ylim(yLim)
            plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
            plt.savefig(savePlottingFolder + "Raw Signals.pdf", dpi=500, bbox_inches='tight')
            self.clearFigure(fig, None)
            
            # -------------------------------------------------------------- #
            # --------------- Plot Feature Extraction Process -------------- #
            
            signalNames = ["EOG", "EEG", "EDA", "Temp"]     
            shortenedNames = ["Raw Signal", "Filtered Signal", "Raw Feature", "Compiled Feature", "Aligned Feature"]     
            plottingColors = [
                '#3498db',  # Blue shades
                '#9ED98F',  # Green shades
                '#918ae1',  # Purple shades
                '#fc827f',   # Red shades
            ]
            

            # For each signal with features.
            for signalInd in range(numFeatureSignals):
                color = plottingColors[signalInd]
                featureInd = alignedFeatureInds[signalInd]
                
                compiledFeatures = allCompiledFeatures[signalInd]
                compiledFeatureTimes = allCompiledFeatureTimes[signalInd]
                endTimeIndex = self.findTimeIndex(alignedFeatureTimes[-1], compiledFeatureTimes)
                compiledFeatureTimes = compiledFeatureTimes[:endTimeIndex]
                compiledFeatures = compiledFeatures[:endTimeIndex]
                                       
                plottingDataTimes = [
                    rawDataTimes,
                    allFilteredTimes[signalInd],
                    allRawFeatureTimes[signalInd], 
                    compiledFeatureTimes,
                    alignedFeatureTimes, 
                ]
                                
                plottingData = [
                    standardizedRawData[signalInd],
                    allFilteredData[signalInd],
                    allRawFeatures[signalInd],
                    compiledFeatures,
                    alignedFeatures[:, featureInd]
                ]
                
                linetype = ['-', '-', 'o', 'o', '-']
                
                figSize = (((rawDataTimes[-1] - rawDataTimes[0]) / 600), 3)
                fig, axes = plt.subplots(len(shortenedNames), 1, figsize=figSize_AllFeatures, sharex=True)
                
                
                # For each feature extraction process.
                for axisInd in range(len(shortenedNames)):
                    
                    # Plot the data.
                    if linetype[axisInd] == '-':
                        axes[axisInd].plot(plottingDataTimes[axisInd], plottingData[axisInd], '-', linewidth=1, color=plottingColors[signalInd])
                    else:
                        axes[axisInd].plot(plottingDataTimes[axisInd], plottingData[axisInd], 'o', markersize=0.25, color=plottingColors[signalInd])
                    # Set plotting labels/aesthetics.
                    # axes[axisInd].set_ylabel(shortenedNames[axisInd])
                    axes[axisInd].set_yticks([])  # Hide y-axis ticks 
                    # axes[axisInd].set_ylim(yLim)  # Hide y-axis ticks 
                    # Customize border box linewidth (change the width as desired)
                    ax.spines['top'].set_linewidth(.25)     # Width of top border line
                    ax.spines['right'].set_linewidth(.25)   # Width of right border line
                    ax.spines['bottom'].set_linewidth(.25)  # Width of bottom border line
                    ax.spines['left'].set_linewidth(.25)    # Width of left border line
                                                        
                    # axes[signalInd].set_title(shortenedNames[signalInd])
                
                axes[-1].set_xlabel('Time')
                plt.subplots_adjust(hspace=0)  # No vertical spacing between subplots
                plt.savefig(savePlottingFolder + f"Feature Extraction Process for {signalNames[signalInd]}.pdf", dpi=500, bbox_inches='tight')
                self.clearFigure(fig, None)
            
            
    def getRawData(self, readData):
        # Extract the raw data
        timePoints = np.array(readData.analysisList[0].data[0])
        eogReadings = np.array(readData.analysisProtocols['eog'].data[1][0])
        eegReadings = np.array(readData.analysisProtocols['eeg'].data[1][0])
        edaReadings = np.array(readData.analysisProtocols['eda'].data[1][0])
        tempReadings = np.array(readData.analysisProtocols['temp'].data[1][0])
        # Organize the raw data.
        rawData = np.array([eogReadings, eegReadings, edaReadings, tempReadings])
        
        # Standardize data
        standardizeClass_Features = standardizeData(rawData, axisDimension = 1, threshold = 0)
        standardizedRawData = standardizeClass_Features.standardize(rawData)
        
        return timePoints, standardizedRawData, rawData
    
    def getFilteredData(self, readData, timePoints, standardizedRawData, rawData):
        # Initialize holders for the filtered information.
        allFilteredData = []; allFilteredTimes = []
        
        # For each signal with features.
        for analysisInd in range(len(readData.analysisList_WithFeatures)):
            standardizedData_toFilter = standardizedRawData[analysisInd]
            rawData_toFilter = rawData[analysisInd]
            
            # Extract the filtered information.
            _, _, goodIndicesMask = readData.analysisList[analysisInd].filterData(timePoints, rawData_toFilter, removePoints = True)
            filteredTimes, filteredData, _ = readData.analysisList[analysisInd].filterData(timePoints, standardizedData_toFilter, removePoints = False)

            # Store the filtered information.
            allFilteredTimes.append(filteredTimes[goodIndicesMask])
            allFilteredData.append(filteredData[goodIndicesMask])
        
        return allFilteredTimes, allFilteredData

    def getRawFeatures(self, readData, experimentTimes, biomarkerFeatureNames, plottingFeatureNames):
        # Initialize holders for the raw features.
        allRawFeatures = []; allRawFeatureTimes = []

        # For each signal with features.
        for rawFeatureInd in range(len(readData.rawFeatureTimesHolder)):
            # Get the raw feature data.
            rawFeatureTimes = np.array(readData.rawFeatureTimesHolder[rawFeatureInd])
            rawFeatures = np.array(readData.rawFeatureHolder[rawFeatureInd])
            featureInd = biomarkerFeatureNames[rawFeatureInd].index(plottingFeatureNames[rawFeatureInd])
            startTimeIndex = self.findTimeIndex(experimentTimes[0][0], rawFeatureTimes)
    
            # Standardize data
            standardizeClass_Features = standardizeData(rawFeatures[startTimeIndex:, featureInd], axisDimension = 0, threshold = 0)
            standardizedFeatures = standardizeClass_Features.standardize(rawFeatures[:, featureInd])
            
            # Store the raw feature information.
            allRawFeatureTimes.append(rawFeatureTimes)
            allRawFeatures.append(standardizedFeatures)
            
        return allRawFeatureTimes, allRawFeatures
    
    
    def getCompiledFeatures(self, readData, experimentTimes, biomarkerFeatureNames, plottingFeatureNames):
        # Initialize holders for the raw features.
        allCompiledFeatures = []; allCompiledFeatureTimes = []
        
        # For each signal with features.
        for analysisInd in range(len(readData.analysisList_WithFeatures)):
            # Get the compiled feature data.
            compiledFeatures = np.array(readData.analysisList_WithFeatures[analysisInd].compiledFeatures[0])
            compiledFeatureTimes = np.array(readData.analysisList_WithFeatures[analysisInd].featureTimes[0])
            featureInd = biomarkerFeatureNames[analysisInd].index(plottingFeatureNames[analysisInd])
            startTimeIndex = self.findTimeIndex(experimentTimes[0][0], compiledFeatureTimes)
            # endTimeIndex = self.findTimeIndex(alignedFeatureTimes[-1], compiledFeatureTimes)
    
            # Standardize data
            standardizeClass_Features = standardizeData(compiledFeatures[startTimeIndex:, featureInd], axisDimension = 0, threshold = 0)
            standardizedFeatures = standardizeClass_Features.standardize(compiledFeatures[:, featureInd])
            
            # Store the raw feature information.
            allCompiledFeatureTimes.append(compiledFeatureTimes)
            allCompiledFeatures.append(standardizedFeatures)

        return allCompiledFeatureTimes, allCompiledFeatures
        
    def getAlignedFeatures(self, readData, experimentTimes):
        # Extract the features
        alignedFeatures = np.array(readData.alignedFeatures)
        alignedFeatureTimes = np.array(readData.alignedFeatureTimes)
    
        startTimeIndex = self.findTimeIndex(experimentTimes[0][0], alignedFeatureTimes)
        # Cull the endpoints as they are not well formed.
        alignedFeatures = alignedFeatures[startTimeIndex:-30, :] # signalLength, numSignals
        alignedFeatureTimes = alignedFeatureTimes[startTimeIndex:-30] # SignalLength
        
        # Standardize data
        standardizeClass_Features = standardizeData(alignedFeatures, axisDimension = 0, threshold = 0)
        standardizedFeatures = standardizeClass_Features.standardize(alignedFeatures)
        
        return alignedFeatureTimes, standardizedFeatures
    
            
    def removeBorder(self, ax):
        # Set plotting labels/aesthetics.
        ax.set_yticks([])  # Hide y-axis ticks   
        ax.set_xticks([])  # Hide y-axis ticks   
        # Remove axis spines (border lines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
                    
    def plotColumnData(self, axes, xData, yData, experimentTimes, surveyAnswerTimes,
                       columnNames, plottingColors, yLim, linetypes, savePlottingFolder):
    
        # For each plotting axis.
        for axisInd in range(len(axes)):
            columnName = columnNames[axisInd]
            color = plottingColors[axisInd]
            linetype = linetypes[axisInd]
            
            # Plot the data.
            if linetype == "-":
                axes[axisInd].plot(xData[axisInd], yData[axisInd], linewidth=1, color=color)
            else:
                axes[axisInd].plot(xData[axisInd], yData[axisInd], 'o', markersize=0.25, color=color)
            # Set plotting labels/aesthetics.
            axes[axisInd].set_ylabel(columnName)
            axes[axisInd].set_yticks([])  # Hide y-axis ticks
            axes[axisInd].set_ylim(yLim)  # Hide y-axis ticks 
        
        if None not in experimentTimes:
            # Shade in the experimental sections.
            self.addExperimentalSections(axes, experimentTimes, surveyAnswerTimes, yLim)     

    def addExperimentalSections(self, axes, experimentTimes, surveyAnswerTimes, yLim):
        for experimentInd in range(len(experimentTimes)):
            for ax in axes:
                ax.axvline(experimentTimes[experimentInd][0], color='gray', linestyle='--', linewidth=0.2)
                ax.axvline(surveyAnswerTimes[experimentInd], color='gray', linestyle='--', linewidth=0.2)
    
                # ax.fill_betweenx(np.array(yLim), experimentTimes[experimentInd][0], surveyAnswerTimes[experimentInd], color="lightblue", alpha=0.03)
                
    def findTimeIndex(self, timePoint, timePoints):
        timePoints = np.asarray(timePoints)
        return (abs(timePoints - timePoint)).argmin()
        
        
        
    
    
    
