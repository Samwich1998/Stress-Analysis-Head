# General
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os

# Import files for machine learning
from .trainingPlots import trainingPlots
from scipy.ndimage import uniform_filter1d


class signalEncoderPlots(trainingPlots):
    def __init__(self, modelName, datasetNames, sharedModelWeights, savingBaseFolder, accelerator=None):
        super(signalEncoderPlots, self).__init__(modelName, datasetNames, sharedModelWeights, savingBaseFolder, accelerator)
        # General parameters
        self.savingSignalEncoderFolder = savingBaseFolder + "signalEncoderPlots/"  # The folder to save the figures

        # Define saving folder locations.
        self.heatmapFolder = self.savingSignalEncoderFolder + "heatmapParamsPlots/"

        cmap = plt.get_cmap('tab10')
        self.timeWindowColors = [cmap(i) for i in range(8)]

    # ---------------------------------------------------------------- #

    def signalEncoderParamHeatmap(self, numExpandedSignalBounds=(2, 10), numEncodingLayerBounds=(0, 12), numLiftedChannelBounds=(16, 64, 16),
                                  finalTrainingDataString="2024-04-21 numLiftedChannels ZZ at numExpandedSignals XX at numEncodingLayers YY"):
        print("\nPlotting the signal encoder heatmaps")
        os.makedirs(self.heatmapFolder, exist_ok=True)

        # Get the losses for the signal encoder
        lossStrings = ["trainingLosses_timeReconstructionSVDAnalysis", "trainingLosses_timeReconstructionAnalysis", "testingLosses_timeReconstructionAnalysis"]
        lossHolders = self.getSignalEncoderLosses(finalTrainingDataString, numLiftedChannelBounds=numLiftedChannelBounds, numExpandedSignalBounds=numExpandedSignalBounds, numEncodingLayerBounds=numEncodingLayerBounds,
                                                  lossStrings=lossStrings)
        # Dimension: (len(lossStrings), numTimeWindows, numDatasets, numLiftedChannelsTested, numExpandedSignalsTested, numEncodingLayersTested)
        optimalLossHolders, trainingLossHolders, testingLossHolders = lossHolders

        # Prepare the data for plotting.
        numLiftedChannelsTested, numExpandedSignalsTested, numEncodingLayersTested = trainingLossHolders[0][0].shape

        # Plot the heatmaps for each combination of losses
        for time_index, time_window in enumerate(self.timeWindows):
            for dataset_index, dataset_name in enumerate(self.datasetNames):
                data = trainingLossHolders[time_index, dataset_index, :, :, :]

                # For each combination of losses, plot the heatmap
                for numLiftedChannelsInd in range(numLiftedChannelsTested):
                    numLiftedChannels = numLiftedChannelBounds[0] + numLiftedChannelsInd * numLiftedChannelBounds[2]
                    accuracy = data[numLiftedChannelsInd, :, :]

                    # Prepare the heatmap parameters.
                    saveFigurePath = os.path.join(self.heatmapFolder, f"TimeWindow_{time_window}_{dataset_name}_numLiftedChannels{numLiftedChannels}.pdf")
                    column_labels = [f"{numExpandedSignals}" for numExpandedSignals in range(numExpandedSignalsTested)]
                    row_labels = [f"{numEncodingLayers}" for numEncodingLayers in range(numEncodingLayersTested)]
                    title = f"{dataset_name} - Time Window: {time_window} - Lifted Channels: {numLiftedChannels}"
                    columnLabel = "Expanded Signals"
                    rowLabel = "Encoding Layers"

                    # Plot the heatmap
                    self.plot_heatmap(accuracy, column_labels, row_labels, columnLabel, rowLabel, title=title, color_map='icefire', cbar_label='Accuracy', saveFigurePath=saveFigurePath, useLogNorm=True)

                # For each combination of losses, plot the heatmap
                for numExpandedSignalsInd in range(numExpandedSignalsTested):
                    numExpandedSignals = numExpandedSignalBounds[0] + numExpandedSignalsInd
                    accuracy = data[:, numExpandedSignalsInd, :]

                    # Prepare the heatmap parameters.
                    saveFigurePath = os.path.join(self.heatmapFolder, f"TimeWindow_{time_window}_{dataset_name}_numExpandedSignals{numExpandedSignals}.pdf")
                    column_labels = [f"{numLiftedChannels}" for numLiftedChannels in range(numLiftedChannelsTested)]
                    row_labels = [f"{numEncodingLayers}" for numEncodingLayers in range(numEncodingLayersTested)]
                    title = f"{dataset_name} - Time Window: {time_window} - numExpandedSignals: {numExpandedSignals}"
                    columnLabel = "Lifted Channels"
                    rowLabel = "Encoding Layers"

                    # Plot the heatmap
                    self.plot_heatmap(accuracy, column_labels, row_labels, columnLabel, rowLabel, title=title, color_map='icefire', cbar_label='Accuracy', saveFigurePath=saveFigurePath, useLogNorm=True)

                # For each combination of losses, plot the heatmap
                for numEncodingLayersInd in range(numEncodingLayersTested):
                    numEncodingLayers = numEncodingLayerBounds[0] + numEncodingLayersInd
                    accuracy = data[:, :, numEncodingLayersInd]

                    # Prepare the heatmap parameters.
                    saveFigurePath = os.path.join(self.heatmapFolder, f"TimeWindow_{time_window}_{dataset_name}_numEncodingLayers{numEncodingLayers}.pdf")
                    column_labels = [f"{numLiftedChannels}" for numLiftedChannels in range(numLiftedChannelsTested)]
                    row_labels = [f"{numExpandedSignals}" for numExpandedSignals in range(numExpandedSignalsTested)]
                    title = f"{dataset_name} - Time Window: {time_window} - numEncodingLayers: {numEncodingLayers}"
                    columnLabel = "Lifted Channels"
                    rowLabel = "Expanded Signals"

                    # Plot the heatmap
                    self.plot_heatmap(accuracy, column_labels, row_labels, columnLabel, rowLabel, title=title, color_map='icefire', cbar_label='Accuracy', saveFigurePath=saveFigurePath, useLogNorm=True)

    def getSignalEncoderLosses(self, finalTrainingDataString, numLiftedChannelBounds=(16, 64, 16), numExpandedSignalBounds=(2, 10), numEncodingLayerBounds=(0, 12), lossStrings=[]):
        # Define the bounds for the number of expanded signals and encoding layers.
        numLiftedChannelsTested = (numLiftedChannelBounds[1] - numLiftedChannelBounds[0]) // numLiftedChannelBounds[2] + 1  # Boundary inclusive
        numExpandedSignalsTested = numExpandedSignalBounds[1] - numExpandedSignalBounds[0] + 1  # Boundary inclusive
        numEncodingLayersTested = numEncodingLayerBounds[1] - numEncodingLayerBounds[0] + 1  # Boundary inclusive

        lossHolders = []
        for _ in lossStrings:
            # Initialize the holders.
            lossHolders.append(np.zeros((len(self.timeWindows), len(self.datasetNames), numLiftedChannelsTested, numExpandedSignalsTested, numEncodingLayersTested)))
            # Dimension: (len(lossStrings), numTimeWindows, numDatasets, numLiftedChannelsTested, numExpandedSignalsTested, numEncodingLayersTested)

        allDummyModelPipelines = []
        # For each lifted channel value.
        for numLiftedChannelInd in range(numLiftedChannelsTested):
            numLiftedChannels = numLiftedChannelBounds[0] + numLiftedChannelInd * numLiftedChannelBounds[2]

            # For each expanded signal value.
            for numExpandedSignalInd in range(numExpandedSignalsTested):
                numExpandedSignals = numExpandedSignalBounds[0] + numExpandedSignalInd

                # For each encoding layer value.
                for numEncodingLayerInd in range(numEncodingLayersTested):
                    numEncodingLayers = numEncodingLayerBounds[0] + numEncodingLayerInd

                    # Load in the previous model attributes.
                    loadSubmodelDate = finalTrainingDataString.replace("XX", str(numLiftedChannels)).replace("YY", str(numExpandedSignals)).replace("ZZ", str(numEncodingLayers))
                    allDummyModelPipelines = self.modelCompiler.onlyPreloadModelAttributes(self.modelName, self.datasetNames, loadSubmodel="signalEncoder", loadSubmodelDate=loadSubmodelDate, loadSubmodelEpochs=-1, allDummyModelPipelines=allDummyModelPipelines)

                    # For each model, get the losses.
                    for modelInd in range(len(allDummyModelPipelines)):
                        currentModel = self.getSubmodel(allDummyModelPipelines[modelInd], submodel="signalEncoder")
                        assert self.timeWindows == currentModel.timeWindows, f"Time windows do not match: {self.timeWindows} != {currentModel.timeWindows}"

                        # For each loss value we want:
                        for lossInd, lossString in enumerate(lossStrings):
                            lossValues = getattr(currentModel, lossString)
                            lossHolders[lossInd][:, modelInd, numLiftedChannelInd, numExpandedSignalInd, numEncodingLayerInd] = self.getSmoothedFinalLosses(lossValues)

        return lossHolders
