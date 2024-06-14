# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import copy
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd


# -------------------------------------------------------------------------- #
# --------------------- Plotting the data and Features --------------------- #

class varyWindows:
    def __init__(self):
        # Change the font style and size.
        plt.rcParams['font.sans-serif'] = ['Arial']  # Additional fallback option
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12

        self.savingFolder = "./Figures/"
        plt.ioff()  # prevent memory leaks; plt.ion()

    @staticmethod
    def clearFigure(fig, legend):
        if legend is not None: legend.remove()
        # Clear plots
        fig.clear()
        plt.cla();
        plt.clf()
        plt.close(fig);
        plt.close('all')

    # ---------------------------------------------------------------------- #
    # ---------------------------- Signal Plots ---------------------------- #

    def varyAnalysisParam(self, readDatas, compiledRawDatas, testDataExcelFiles, featureTimeWindows,
                          featureAverageWindows, biomarkerFeatureOrder, biomarkerFeatureNames):
        savingWindowsDirectory = self.savingFolder + "/Feature Time Windows/smallWindow/"
        os.makedirs(savingWindowsDirectory, exist_ok=True)

        for analysisFileInd in range(len(testDataExcelFiles)):
            print(analysisFileInd)
            saveName = testDataExcelFiles[analysisFileInd].split("/")[-1].split(".")[0]
            compiledRawData = compiledRawDatas[analysisFileInd].copy()
            readData = copy.deepcopy(readDatas[analysisFileInd])

            # Create and start a thread for each analysis
            self.varyAnalysisFile(readData, compiledRawData, saveName, featureTimeWindows, featureAverageWindows,
                                  biomarkerFeatureOrder, biomarkerFeatureNames, savingWindowsDirectory)

    def varyAnalysisFile(self, readData, compiledRawData, saveName, featureTimeWindows, featureAverageWindows,
                         biomarkerFeatureOrder, biomarkerFeatureNames, savingWindowsDirectory):

        # Initialize holders for the features.
        experimentTimes = readData.experimentTimes
        allRawFeatureTimesHolders = []
        allRawFeatureHolders = []

        # For each new time window.
        for featureTimeWindow in featureTimeWindows:
            # Set the parameter in the analysis
            readData.unifyFeatureTimeWindows(featureTimeWindow)
            readData.resetGlobalVariables()

            # Analyze the data, extracting features with the time window.
            readData.streamExcelData(compiledRawData, [], [], [], [], [], [], [], "")
            # Extract information from the streamed data.
            allRawFeatureTimesHolders.append(readData.rawFeatureTimesHolder.copy())
            allRawFeatureHolders.append(readData.rawFeatureHolder.copy());
            # Remove all previous information from this trial
            readData.resetGlobalVariables()

            # Assert the integrity of data streaming.
            assert len(biomarkerFeatureOrder) == len(allRawFeatureHolders[-1]), f"Incorrect number of channels: {len(allRawFeatureHolders[-1])} {biomarkerFeatureOrder}"

        # Initialize saving folder
        savingUserWindowsDirectory = savingWindowsDirectory + f"{saveName}/"
        os.makedirs(savingUserWindowsDirectory, exist_ok=True)

        # For each biomarker with time windows.
        for biomarker in ["eeg", "eda", "temp"]:
            biomarkerInd = biomarkerFeatureOrder.index(biomarker)

            # Initialize saving folder
            finalSavingWindowsDirectory = savingUserWindowsDirectory + f"{biomarker}/"
            os.makedirs(finalSavingWindowsDirectory, exist_ok=True)

            # For each EEG Feature
            for featureInd in range(len(biomarkerFeatureNames[biomarkerInd])):
                featureName = biomarkerFeatureNames[biomarkerInd][featureInd]
                finalTimePoints = np.arange(max(allRawFeatureTimesHolders, key=lambda x: x[biomarkerInd][0])[biomarkerInd][0],
                                            min(allRawFeatureTimesHolders, key=lambda x: x[biomarkerInd][-1])[biomarkerInd][-1], 0.1)

                heatMap = [];
                mpl.rcdefaults()
                # For each feature-variation within a certain time window
                for trialInd in range(len(allRawFeatureHolders)):
                    averageWindow = featureAverageWindows[biomarkerInd]
                    rawFeatures = np.asarray(allRawFeatureHolders[trialInd][biomarkerInd])[:, featureInd]
                    rawFeatureTimes = np.asarray(allRawFeatureTimesHolders[trialInd][biomarkerInd])

                    # Perform the feature averaging
                    compiledFeatures = readData.averageFeatures_static(rawFeatureTimes, rawFeatures, averageWindow, startTimeInd=0)

                    # Interpolate all the features within the same time-window
                    featurePolynomial = scipy.interpolate.interp1d(rawFeatureTimes, compiledFeatures, kind='linear')
                    finalFeatures = featurePolynomial(finalTimePoints)

                    # Track the heatmap
                    heatMap.append(finalFeatures)

                # Find the start and end of the experiments.
                startTimeIndex = self.findTimeIndex(experimentTimes[0][0], finalTimePoints)
                endTimeIndex = self.findTimeIndex(experimentTimes[-1][-1], finalTimePoints)
                # Find the upper and lower bounds of the heatmap
                vMin = np.asarray(heatMap)[:, startTimeIndex:endTimeIndex].min()
                vMax = np.asarray(heatMap)[:, startTimeIndex:endTimeIndex].max()

                # for finalFeatures in heatMap:
                #     plt.plot(finalTimePoints, finalFeatures, linewidth=2)
                # plt.ylim((vMin, vMax))
                # plt.savefig(finalSavingWindowsDirectory + f"{saveName}_{featureName} Sliding Windows Plot.pdf", dpi=500, bbox_inches='tight')
                # self.clearFigure(None, None)

                # Plot the heatmap
                ax = sns.heatmap(pd.DataFrame(heatMap, index=featureTimeWindows, columns=np.round(finalTimePoints, 2)), robust=True, vmin=vMin, vmax=vMax, cmap='icefire')
                ax.set(title='Feature:' + biomarkerFeatureNames[biomarkerInd][featureInd])
                # Save the Figure
                sns.set(rc={'figure.figsize': (7, 9)})
                plt.xlabel("Time (Seconds)")
                plt.ylabel("Sliding Window Size")
                fig = ax.get_figure();
                fig.savefig(finalSavingWindowsDirectory + f"{saveName}_{featureName} Sliding Windows Heatmap.pdf", dpi=500, bbox_inches='tight')
                fig.savefig(finalSavingWindowsDirectory + f"{saveName}_{featureName} Sliding Windows Heatmap.png", dpi=500, bbox_inches='tight')
                self.clearFigure(fig, None)

    def findTimeIndex(self, timePoint, timePoints):
        timePoints = np.asarray(timePoints)
        return (abs(timePoints - timePoint)).argmin()
