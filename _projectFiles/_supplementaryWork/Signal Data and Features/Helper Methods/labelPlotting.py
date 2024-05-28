
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import numpy as np
import pandas as pd
from natsort import natsorted

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# PlotAPI
import plotapi
from plotapi import Chord
from plotapi import Sankey


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
        
        # Specify the plotAPI licence keys
        plotapi.api_key("ff69182a-324b-46b9-90c5-77a6da9bd050")
        Sankey.api_key("ff69182a-324b-46b9-90c5-77a6da9bd050")
        Chord.api_key("ff69182a-324b-46b9-90c5-77a6da9bd050")
        
        
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
        
        
        # PlotAPI diagrams
        # self.plotChordDiagram(surveyAnswersList, surveyQuestions, surveyInfoDirectory) # Chord Diagrams
        
        # Get all the activity information from the experiment.
        activityNames, activityLabels = self.modelInfoClass.extractActivityInformation(experimentalOrder, distinguishBaselines = True)
        experimentalOrder_byActivity = self.modelInfoClass.labelExperimentalOrder(experimentalOrder, distinguishBaselines = True)
        # experimentalOrder_byActivity: "{Exercise CPT Music or VR} {Baseline Activity or Recovery}". Dim: numExperiments
        # activityLabels: classification label of the activity name (0-5). Dim: numExperiments
        # activityNames: ['Baseline' 'Music' 'CPT' 'Exercise' 'VR' 'Recovery']. Dim: 6
        
        # Initialize the figure.
        allLabelBins = [2, 2, 5]
        allLabelRanges = [(5, 25), (5, 25), (20, 80)]
        alphabet_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", 
                         "J", "K", "L", "M", "N", "O", "P", "Q", "R", 
                         "S", "T", "U", "V", "W", "X", "Y", "Z", 
                         "ZA", "ZB", "ZC", "ZD", "ZE", "ZF", "ZG", "ZH", "ZI"
                         "ZJ", "ZK", "ZL", "ZM", "ZN", "ZO", "ZP", "ZQ", "ZR", 
                         "ZS", "ZT", "ZU", "ZV", "ZW", "ZX", "ZY", "ZZ",
                         "ZZA", "ZZB", "ZZC", "ZZD", "ZZE", "ZZF", "ZZG", "ZZH", "ZZI"
                         "ZZJ", "ZZK", "ZZL", "ZZM", "ZZN", "ZZO", "ZZP", "ZZQ", "ZZR", 
                         "ZZS", "ZZT", "ZZU", "ZZV", "ZZW", "ZZX", "ZZY", "ZZZ",
                         "ZZZA", "ZZZB", "ZZZC", "ZZZD", "ZZZE", "ZZZF", "ZZZG", "ZZZH", "ZZZI"
                         "ZZZJ", "ZZZK", "ZZZL", "ZZZM", "ZZZN", "ZZZO", "ZZZP", "ZZZQ", "ZZZR", 
                         "ZZZS", "ZZZT", "ZZZU", "ZZZV", "ZZZW", "ZZZX", "ZZZY", "ZZZZ",
                         "ZZZZA", "ZZZZB", "ZZZZC", "ZZZZD", "ZZZZE", "ZZZZF", "ZZZZG", "ZZZZH", "ZZZZI"
                         "ZZZZJ", "ZZZZK", "ZZZZL", "ZZZZM", "ZZZZN", "ZZZZO", "ZZZZP", "ZZZZQ", "ZZZZR", 
                         "ZZZZS", "ZZZZT", "ZZZZU", "ZZZZV", "ZZZZW", "ZZZZX", "ZZZZY", "ZZZZZ",
                         "ZZZZZA", "ZZZZZB", "ZZZZZC", "ZZZZZD", "ZZZZZE", "ZZZZZF", "ZZZZZG", "ZZZZZH", "ZZZZZI"
                         "ZZZZZJ", "ZZZZZK", "ZZZZZL", "ZZZZZM", "ZZZZZN", "ZZZZZO", "ZZZZZP", "ZZZZZQ", "ZZZZZR", 
                         "ZZZZZS", "ZZZZZT", "ZZZZZU", "ZZZZZV", "ZZZZZW", "ZZZZZX", "ZZZZZY", "ZZZZZZ",
                         ]
        # Initialize saving folder
        savingPlotAPI = surveyInfoDirectory + "/PlotAPI/Sankey/"
        os.makedirs(savingPlotAPI, exist_ok=True)
                        
        # For each survey administered (PA, NA, STAI).
        for labelTypeInd in range(len(featureLabelTypes)):
            surveyName = featureLabelTypes[labelTypeInd]
            surveyAnswers = finalLabels[labelTypeInd]
            numScores_perBin = allLabelBins[labelTypeInd]
            labelRange = allLabelRanges[labelTypeInd]
            
            allPossibleBins = np.arange(0, (labelRange[1] - labelRange[0])//numScores_perBin + 1, 1, dtype = int)
            
            # Compartmentalized each label into its respective activity.
            allCompiledActivityLabels, activityGroups = self.organizeSurveyAnswers_forCurves(surveyAnswers, experimentalOrder_byActivity)
            # allCompiledActivityLabels: Dim: numActivityGroups (4), numActivityExperiments, numTrials_perActivity (3-6)
            # activityGroups:['CPT', 'Exercise', 'Music', 'VR']. Dim: 4
            
            # Assert the integrity of the activity compilation.
            assert len(activityGroups) == len(allCompiledActivityLabels)
            
            # For each type of activity performed.
            for activityInd in range(len(activityGroups)):
                activityName = activityGroups[activityInd]
                compiledActivityLabels = np.array(allCompiledActivityLabels[activityInd])
                # Initialize the parameters for creatings the Sankey diagram.
                numActivityExperiments, numTrials = compiledActivityLabels.shape
                sankeyLinks = []
                
                # For each trial (flow in the Sankey).
                for trialInd in range(numTrials - 1):
                    # Compile the in starting and ending scores.
                    sources = ((compiledActivityLabels[:, trialInd] - labelRange[0]) // numScores_perBin).astype(int)
                    targets = ((compiledActivityLabels[:, trialInd+1] - labelRange[0]) // numScores_perBin).astype(int)
                    numBinOptions = labelRange[1]//numScores_perBin + 1
                    
                    recordedFlows = {}
                    # For each possible source bin.
                    for possibleSource in allPossibleBins:
                        sourceName = f"G{trialInd} {alphabet_list[possibleSource]}"
                        # For each possible target bin.
                        for possibleTarget in allPossibleBins:
                            # Organize the bins into their flows.
                            targetName = f"G{trialInd+1} {alphabet_list[possibleTarget]}"
                            # Record how many times this appeared
                            flowIdentifier = f"{sourceName}_{targetName}"
                            recordedFlows[flowIdentifier] = 0
                                                
                    # For each source and target flow.
                    for expInd in range(len(sources)):
                        # Organize the bins into their flows.
                        sourceName = f"G{trialInd} {alphabet_list[sources[expInd]]}"
                        targetName = f"G{trialInd+1} {alphabet_list[targets[expInd]]}"
                        # Record how many times this appeared
                        flowIdentifier = f"{sourceName}_{targetName}"
                        
                        currentFlows = recordedFlows.get(flowIdentifier, 0)
                        recordedFlows[flowIdentifier] = int(currentFlows + 1)
                    
                    # For each source and target flow.
                    for flowIdentifier in  recordedFlows.keys():
                        # Organize the bins into their flows.
                        sourceName, targetName = flowIdentifier.split("_")
                        
                        # = f"{trialInd} {sources[expInd]}"
                        # targetName = f"{trialInd+1} {targets[expInd]}"
                        # flowIdentifier = f"{sourceName} {targetName}"
                        
                        # Add this flow to the Sankey diagram.
                        sankeyLinks.append(
                            {"source": sourceName, 
                             "target": targetName, 
                             "value": recordedFlows[flowIdentifier]},
                        )
                
                # Sort the links data alphabetically by source and target nodes
                sankeyLinks = natsorted(sankeyLinks, key=lambda x: (x["source"], x["target"]))

                # Create the Sankey diagram
                sankeyDiagram = Sankey(
                    sankeyLinks,
                    margin=100,
                    thumbs_margin=1,
                    link_opacity = 0.45,
                    link_background_opacity = 1,
                    node_opacity = 1,
                    popup_width=1000,
                    node_width = 24,
                    width = 800,
                    height = 500, # 500
                    animated_intro=True,
                    node_sort="alpha",
                    reverse_gradients = False,
                    noun="Number of Instances",
                    title=f"{surveyName} Flow through {activityName}",
                )
                # Save the chord diagram
                sankeyDiagram.to_pdf(savingPlotAPI + f"Sankey diagram for {surveyName} during {activityName}.pdf")
                sankeyDiagram.to_html(savingPlotAPI + f"Sankey diagram for {surveyName} during {activityName}.html")
                # print(f"{surveyName} Flow through {activityName}")
                # print(sankeyLinks)
                # sys.exit()
        
        sys.exit()
        
        
        
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
        #     allDataX, allCompiledActivityLabels = self.organizeSurveyAnswers_toPlot(surveyAnswers, experimentalOrder_byActivity)
            
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
        #             yData = allCompiledActivityLabels[axisInd][segmentInd]
                    
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
        #     allCompiledActivityLabels, activityGroups = self.organizeSurveyAnswers_forCurves(surveyAnswers, experimentalOrder_byActivity)
            
        #     fig, axes = plt.subplots(len(activityGroups), 1, figsize=figSize, sharex=True)    
        #     savePlottingFolder = surveyInfoDirectory + f"{surveyName}/"
        #     os.makedirs(savePlottingFolder, exist_ok=True)
        
        #     for axisInd in range(len(activityGroups)):
        #         # Set plotting labels/aesthetics.
        #         axes[axisInd].set_ylabel(self.activityOrder[axisInd])
        #         # axes[axisInd].set_yticks([])  # Hide y-axis ticks
        #         axes[axisInd].set_ylim(ylims[labelTypeInd])  # Hide y-axis ticks
                
        #         for pointInd in range(len(allCompiledActivityLabels[axisInd])):
        #             axes[axisInd].plot(allCompiledActivityLabels[axisInd][pointInd], linewidth=1, color=self.rawDataColors[axisInd], alpha=0.3)
                                                               
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
            allCompiledActivityLabels, activityGroups = self.organizeSurveyAnswers_forCurves(surveyAnswers, experimentalOrder_byActivity)
            # allCompiledActivityLabels: Dim: numActivityGroups (4), numExp_perGroup, numTrials_perActivity (3-6)
            # activityGroups:['CPT', 'Exercise', 'Music', 'VR']. Dim: 4
            
            axes[labelTypeInd][0].set_ylim(ylims[labelTypeInd])  # Hide y-axis ticks
            axes[labelTypeInd][1].set_ylim(ylims[labelTypeInd])  # Hide y-axis ticks
            
            sys.exit()
            
            savePlottingFolder = surveyInfoDirectory + f"{surveyName}/"
            os.makedirs(savePlottingFolder, exist_ok=True)
            
            for groupInd in range(2):
                for pointInd in range(len(allCompiledActivityLabels[groupInd])):
                    axes[labelTypeInd][groupInd].plot(allCompiledActivityLabels[groupInd][pointInd], linewidth=1, color=self.rawDataColors[groupInd], alpha=0.7)
                                                               
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
        
        
        
        
        
    def plotChordDiagram(self, surveyAnswersList, columnNames, surveyInfoDirectory):  
        # Change the survey questions
        shortenedNames_STAI = ["Calm", "Secure", "Tense", "Strained", "Ease", "Upset", 
                          "Satisfied", "Worrying", "Frightened", "Comfortable", 
                          "Self-confident", "Nervous", "Jittery", "Indecisive", 
                          "Relaxed", "Content", "Worried", "Confused", "Steady", "Pleasant"]
        columnNames[10:] = shortenedNames_STAI

        # Define an order of the emotions to group positive and negative emotions.
        column_order = (self.modelInfoClass.posAffectInds + self.modelInfoClass.negAffectInds +
            self.modelInfoClass.staiInds_Pos + self.modelInfoClass.staiInds_Neg)
        # Reorder the emotions to group positive and negative
        surveyAnswersList = surveyAnswersList[:, column_order]        
        columnNames = columnNames[column_order]

        # Calculate the correlations between the emotiuons
        correlationMatrix = np.corrcoef(surveyAnswersList, rowvar=False)
        correlationMatrix = abs(correlationMatrix)
        np.fill_diagonal(correlationMatrix, 0.0)
        columnNames = list(columnNames)
        
        # Initialize saving folder
        savingPlotAPI = surveyInfoDirectory + "/PlotAPI/Chord/"
        os.makedirs(savingPlotAPI, exist_ok=True)
        
        # Create the chord diagram
        panasChord = Chord(
            correlationMatrix[0:10, 0:10].tolist(),
            columnNames[0:10],
            margin=100,
            thumbs_margin=1,
            popup_width=1000,
            directed=False,
            symmetric=True,
            arc_numbers=False,
            animated_intro=True,
            ticks_font_size = 5,
            curved_labels = True,
            reverse_gradients = False,
            noun="percent correlated",
            data_table_show_indices=True,
            title="",
        )
        # Save the chord diagram
        panasChord.to_pdf(savingPlotAPI + "panasCorrelation_chord.pdf")
        panasChord.to_html(savingPlotAPI + "panasCorrelation_chord.html")
        
        # Create the chord diagram
        staiChord = Chord(
            correlationMatrix[10:30, 10:30].tolist(),
            columnNames[10:30],
            margin=100,
            thumbs_margin=1,
            popup_width=1000,
            directed=False,
            symmetric=True,
            arc_numbers=False,
            animated_intro=True,
            ticks_font_size = 5,
            curved_labels = False,
            reverse_gradients = False,
            noun="percent correlated",
            data_table_show_indices=True,
            title="",
        )
        # Save the chord diagram
        staiChord.to_pdf(savingPlotAPI + "staiCorrelation_chord.pdf")
        staiChord.to_html(savingPlotAPI + "staiCorrelation_chord.html")
        
        # Create the chord diagram
        emotionChord = Chord(
            correlationMatrix.tolist(),
            columnNames,
            margin=100,
            thumbs_margin=1,
            popup_width=1000,
            directed=False,
            symmetric=True,
            arc_numbers=False,
            animated_intro=True,
            ticks_font_size = 5,
            curved_labels = False,
            reverse_gradients = False,
            noun="percent correlated",
            data_table_show_indices=True,
            title="",
        )
        # Save the chord diagram
        emotionChord.to_pdf(savingPlotAPI + "emotionCorrelation_chord.pdf")
        emotionChord.to_html(savingPlotAPI + "emotionCorrelation_chord.html")
        
        

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
                
    def findTimeIndex(self, timePoint, timePoints):
        timePoints = np.asarray(timePoints)
        return (abs(timePoints - timePoint)).argmin()
        
        
        
"""
        # For each survey administered (PA, NA, STAI).
        for labelTypeInd in range(len(featureLabelTypes)):
            surveyName = featureLabelTypes[labelTypeInd]
            surveyAnswers = finalLabels[labelTypeInd]
            numScores_perBin = allLabelBins[labelTypeInd]
            labelRange = allLabelRanges[labelTypeInd]
            
            # Compartmentalized each label into its respective activity.
            allCompiledActivityLabels, activityGroups = self.organizeSurveyAnswers_forCurves(surveyAnswers, experimentalOrder_byActivity)
            # allCompiledActivityLabels: Dim: numActivityGroups (4), numActivityExperiments, numTrials_perActivity (3-6)
            # activityGroups:['CPT', 'Exercise', 'Music', 'VR']. Dim: 4
            
            # Assert the integrity of the activity compilation.
            assert len(activityGroups) == len(allCompiledActivityLabels)
            
            # For each type of activity performed.
            for activityInd in range(len(activityGroups)):
                activityName = activityGroups[activityInd]
                compiledActivityLabels = np.array(allCompiledActivityLabels[activityInd])
                # Initialize the parameters for creatings the Sankey diagram.
                numActivityExperiments, numTrials = compiledActivityLabels.shape
                sankeyLinks = []
                
                # For each trial (flow in the Sankey).
                for trialInd in range(numTrials - 1):
                    # Compile the in starting and ending scores.
                    sources = (compiledActivityLabels[:, trialInd] // numScores_perBin).astype(int)
                    targets = (compiledActivityLabels[:, trialInd+1] // numScores_perBin).astype(int)
                    allPossibleBins = np.arange(0, labelRange[1]//numScores_perBin + 1, 1, dtype = int)
                    
                    recordedFlows = {}
                    # For each possible source bin.
                    for possibleSource in allPossibleBins:
                        sourceName = f"{trialInd} {possibleSource}"
                        # For each possible target bin.
                        for possibleTarget in allPossibleBins:
                            # Organize the bins into their flows.
                            targetName = f"{trialInd+1} {possibleTarget}"
                            # Record how many times this appeared
                            flowIdentifier = f"{sourceName}_{targetName}"
                            recordedFlows[flowIdentifier] = 0.00001   
                                                
                    # For each source and target flow.
                    for expInd in range(len(sources)):
                        # Organize the bins into their flows.
                        sourceName = f"{trialInd} {sources[expInd]}"
                        targetName = f"{trialInd+1} {targets[expInd]}"
                        # Record how many times this appeared
                        flowIdentifier = f"{sourceName}_{targetName}"
                        recordedFlows[flowIdentifier] += 1
                    
                    # For each source and target flow.
                    for flowIdentifier in  recordedFlows.keys():
                        # Organize the bins into their flows.
                        sourceName, targetName = flowIdentifier.split("_")
                        
                        # = f"{trialInd} {sources[expInd]}"
                        # targetName = f"{trialInd+1} {targets[expInd]}"
                        # flowIdentifier = f"{sourceName} {targetName}"
                        
                        # Add this flow to the Sankey diagram.
                        sankeyLinks.append(
                            {"source": sourceName, 
                             "target": targetName, 
                             "value": recordedFlows[flowIdentifier]},
                        )
                    
                # Create the Sankey diagram
                sankeyDiagram = Sankey(
                    links = sankeyLinks,
                    margin=100,
                    thumbs_margin=1,
                    popup_width=1000,
                    animated_intro=True,
                    reverse_gradients = False,
                    noun="Number of Instances",
                    title=f"{surveyName} Flow through {activityName}",
                )
                # Save the chord diagram
                sankeyDiagram.to_pdf(savingPlotAPI + f"Sankey diagram for {surveyName} during {activityName}.pdf")
                sankeyDiagram.to_html(savingPlotAPI + f"Sankey diagram for {surveyName} during {activityName}.html")
"""  
    
