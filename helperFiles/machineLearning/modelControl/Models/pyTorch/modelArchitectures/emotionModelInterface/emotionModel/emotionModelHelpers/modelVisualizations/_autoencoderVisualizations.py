# General
import matplotlib.pyplot as plt
import numpy as np
import os

# Visualization protocols
from ..........globalPlottingProtocols import globalPlottingProtocols


class autoencoderVisualizations(globalPlottingProtocols):

    def __init__(self, saveDataFolder):
        super(autoencoderVisualizations, self).__init__()
        # General parameters
        self.saveDataFolder = None

        # Set the location for saving the models.
        self.setSavingFolder(saveDataFolder)

    def setSavingFolder(self, saveDataFolder):
        self.saveDataFolder = saveDataFolder

    # ---------------------------------------------------------------------- #
    # --------------------- Visualize Model Parameters --------------------- #

    def plotEncoder(self, initialSignal, comparisonSignal, epoch, plotTitle="Encoder Prediction", numSignalPlots=1):
        # Assert the integrity of the incoming data
        assert initialSignal.shape[0:2] == comparisonSignal.shape[0:2], f"{initialSignal.shape} {comparisonSignal.shape}"
        batchSize, numSignals, numEncodedPoints = comparisonSignal.shape
        if batchSize == 0: return None

        # Get the signals to plot.
        plottingSignals = np.arange(0, numSignalPlots)
        plottingSignals = np.concatenate((plottingSignals, np.sort(numSignals - plottingSignals - 1)))
        assert plottingSignals[-1] == numSignals - 1, f"{plottingSignals} {numSignals}"

        batchInd = 0
        for signalInd in plottingSignals:
            # Plot the signal reconstruction.
            plt.plot(initialSignal[batchInd, signalInd, :], 'k', linewidth=2, alpha=0.5, label="Initial Signal")
            plt.plot(comparisonSignal[batchInd, signalInd, :], 'tab:blue', linewidth=2, alpha=0.8, label="Reconstructed Signal")
            plt.xlabel("Points")
            plt.ylabel("Signal (AU)")
            plt.title(f"{plotTitle.split('/')[-1]} - Signal {signalInd + 1}")
            plt.legend(loc="best")
            if self.saveDataFolder:
                self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf")
            plt.show()

    def plotSignalComparison(self, originalSignal, comparisonSignal, epoch, plotTitle, numSignalPlots=1):
        """ originalSignal dimension: batchSize, numSignals, numTotalPoints """
        # Assert the integrity of the incoming data
        assert originalSignal.shape[0:2] == comparisonSignal.shape[0:2], f"{originalSignal.shape} {comparisonSignal.shape}"

        # Extract the shapes of the data
        batchSize, numSignals, numTotalPoints = originalSignal.shape
        batchSize, numSignals, numEncodedPoints = comparisonSignal.shape

        if batchSize == 0: return None

        batchInd = 0
        # For each signal
        for signalInd in range(numSignals):
            # Plot both the signals alongside each other.
            plt.plot(originalSignal[batchInd, signalInd], 'k', marker='o', linewidth=2)
            plt.plot(np.linspace(0, numTotalPoints, numEncodedPoints), comparisonSignal[batchInd, signalInd], 'tab:red', marker='o', linewidth=1)

            # Format the plotting      
            plt.ylabel("Arbitrary Axis (AU)")
            plt.xlabel("Points")
            plt.title(plotTitle)

            # Save the plot
            if self.saveDataFolder:
                self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs = {epoch} signalInd = {signalInd}.pdf")
            plt.show()

            # There are too many signals to plot.
            if signalInd + 1 == numSignalPlots: break

    def plotAllSignalComparisons(self, distortedSignals, reconstructedDistortedSignals, trueSignal, epoch, signalInd, plotTitle):
        numSignals, numTotalPoints = reconstructedDistortedSignals.shape
        alphas = np.linspace(0.1, 1, numSignals)

        # Plot all signals in 'distortedSignals'
        for i in range(numSignals):
            plt.plot(distortedSignals[i], '-', color='k', alpha=alphas[i], linewidth=1, markersize=2, zorder=0)
            plt.plot(trueSignal, 'o', color='tab:blue', linewidth=1, markersize=2, zorder=10)
            plt.plot(reconstructedDistortedSignals[i], '-', color='tab:red', linewidth=1, markersize=2, alpha=alphas[i], zorder=5)

        plt.title(plotTitle + " at Epoch " + str(epoch))
        plt.xlabel('Time (Seconds)')
        plt.ylabel("Arbitrary Axis (AU)")
        plt.legend(['Noisy Signal', 'True Signal', 'Reconstructed Signal'], loc='best')
        # Save the plot
        if self.saveDataFolder:
            self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs = {epoch} signalInd{signalInd}.pdf")
        plt.show()

    def plotSignalComparisonHeatmap(self, originalSignal, comparisonSignal):
        # Assert the integrity of the incoming data
        assert originalSignal.shape[0:2] == comparisonSignal.shape[0:2]

        # Extract the shapes of the data
        batchSize, numSignals, numTotalPoints = originalSignal.shape
        batchSize, numSignals, numEncodedPoints = comparisonSignal.shape

        # Seperate out all the signals
        signalData = originalSignal.reshape(batchSize * numSignals, numTotalPoints)
        encodedData = comparisonSignal.reshape(batchSize * numSignals, numEncodedPoints)

        # Plot the heatmap
        self.heatmap(signalData[0:25], saveDataPath=None, title="Initial Signals", xlabel="Points", ylabel="Signal Index")
        self.heatmap(encodedData[0:25], saveDataPath=None, title="Encoded Signals", xlabel="Points", ylabel="Signal Index")

    def plotAutoencoderWeights(self, autoencoderWeightsCNN, autoencoderWeightsFC):
        # For each convolutional weight.
        for layerInd in range(len(autoencoderWeightsCNN)):
            autoencoderWeightCNN = autoencoderWeightsCNN[layerInd]
            # Dim: numOutChannels, numInchannels/Groups, kernelSize
            numOutChannels, numInchannels, kernelSize = autoencoderWeightCNN.shape

            if numOutChannels == 1:
                plt.plot(autoencoderWeightCNN.reshape(-1))
                plt.title("numOutChannels = 1")
            if numInchannels == 1:
                plt.plot(autoencoderWeightCNN.reshape(-1))
                plt.title("numInchannels = 1")
            if kernelSize == 1:
                plt.plot(autoencoderWeightCNN.reshape(-1))
                plt.title("kernelSize = 1")
            # else:
            #     # Create meshgrids for x, y, and z coordinates
            #     x, y = np.meshgrid(np.arange(numOutChannels), np.arange(numInchannels))
            #     z = np.zeros_like(x)

            #     # Create separate 2D heatmaps for each color channel
            #     fig = plt.figure(figsize=(12, 6))

            #     for kernelInd in range(kernelSize):
            #         ax = fig.add_subplot(131 + kernelInd, projection='3d')
            #         z = autoencoderWeightCNN[:, :, kernelInd]  # Choose the current color channel

            #         ax.plot_surface(x, y, z, cmap='bwr')  # Adjust the cmap for different colormaps

            #         ax.set_title(f'kernelInd: {kernelInd}')
            #         ax.set_xlabel('numOutChannels')
            #         ax.set_ylabel('numInchannels')
            #         ax.set_zlabel('Weight (A.U.)')

            plt.tight_layout()
            plt.show()
            plt.rcdefaults()

    def plotAllAutoencoderWeights(self, allAutoencoderWeightsCNN, allAutoresponderWeightsFC):
        """
        Dim allAutoencoderWeightsCNN: numEpochs, numLayers, numOutChannels, numInchannels/Groups, kernelSize
        Dim: allAautoencoderWeightsFC: numEpochs, numLayers, numOutFeatures, numInFeatures
        """
        assert len(allAutoencoderWeightsCNN) == len(allAutoresponderWeightsFC)
        numEpochs = len(allAutoencoderWeightsCNN)

        # Create folder to save the data
        if self.saveDataFolder:
            saveAutoencoderFolder = self.saveDataFolder + "/Autoencoder/"
            os.makedirs(saveAutoencoderFolder, exist_ok=True)

        # For each epoch
        for layerInd in range(len(allAutoencoderWeightsCNN[0])):
            autoencoderWeightsCNN = np.array([epoch[layerInd] for epoch in allAutoencoderWeightsCNN])
            # Dim: numEpochs, numOutChannels, numInchannels/Groups, kernelSize
            autoencoderWeightsCNN = autoencoderWeightsCNN.reshape(numEpochs, -1)

            # Plot the heatmap
            self.heatmap(autoencoderWeightsCNN, saveDataPath=saveAutoencoderFolder + f"Autoencoder CNN Weights at Layer {layerInd}.pdf",
                         title=f'Autoencoder CNN Weights at Layer {layerInd}', xlabel="Kernel Index", ylabel="Epoch")

        for layerInd in range(len(allAutoresponderWeightsFC[0])):
            encoderWeightsFC = np.array([epoch[layerInd] for epoch in allAutoresponderWeightsFC])
            encoderWeightsFC = encoderWeightsFC.reshape(len(encoderWeightsFC), -1)

            # Plot the heatmap
            self.heatmap(encoderWeightsFC, saveDataPath=saveAutoencoderFolder + f"Autoencoder FC Weights at Layer {layerInd}.pdf",
                         title=f'Autoencoder FC Weights at Layer {layerInd}', xlabel="FC Weights", ylabel="Epoch")

    # def plotLatentSpace(self, latentData, latentLabels, epoch, plotTitle="Latent Space PCA", numSignalPlots=1):
    #     batchSize, numSignals, latentDimension = latentData.shape
    #
    #     # Use PCA to reduce the dimensionality of latent space to 2D for visualization
    #     pca = PCA(n_components=2, random_state=42)
    #     latent_2d = pca.fit_transform(latentData.reshape(batchSize * numSignals, latentDimension))
    #     latent_2d = latent_2d.reshape(batchSize, numSignals, 2)
    #
    #     for batchInd in range(batchSize):
    #         # Plot the 2D projection of latent space
    #         plt.plot(latent_2d[:, batchInd, 0], latent_2d[:, batchInd, 1], 'o', markersize=5, alpha=0.5)
    #
    #         if batchInd == numSignalPlots: break
    #
    #     plt.xlabel('PCA Component 1')
    #     plt.ylabel('PCA Component 2')
    #     plt.title('2D PCA Visualization of Latent Space')
    #     # Save the plot
    #     if self.saveDataFolder:
    #         self.saveFigure(self.saveDataFolder + f"{plotTitle} epochs = {epoch}.pdf")
    #     plt.show()
