# General
import matplotlib.pyplot as plt
import math

import numpy as np

# Visualization protocols
from ..........globalPlottingProtocols import globalPlottingProtocols


class signalEncoderVisualizations(globalPlottingProtocols):

    def __init__(self, saveDataFolder):
        super(signalEncoderVisualizations, self).__init__()
        # General parameters
        self.saveDataFolder = None

        # Set the location for saving the models.
        self.setSavingFolder(saveDataFolder)

    def setSavingFolder(self, saveDataFolder):
        self.saveDataFolder = saveDataFolder

    # ---------------------------------------------------------------------- #
    # --------------------- Visualize Model Parameters --------------------- #

    def plotSignalEncoding(self, allEncodedData, epoch, plotTitle="Signal Encoding"):
        # allEncodedData dimension: batchSize, numCondensedSignals, compressedLength
        # Plot the signal reconstruction.
        plt.plot(allEncodedData[0].view(-1), 'k', linewidth=2, alpha=1)
        plt.plot(allEncodedData[1].view(-1), 'k', linewidth=2, alpha=0.6)
        plt.plot(allEncodedData[-2].view(-1), 'tab:red', linewidth=2, alpha=1)
        plt.plot(allEncodedData[-1].view(-1), 'tab:red', linewidth=2, alpha=0.6)

        plt.xlabel("All Encoding Dimensions (Points)")
        plt.ylabel("Signal (AU)")
        plt.title(f"{plotTitle}")
        if self.saveDataFolder:
            self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch}.pdf")
        plt.show()

    def plotOneSignalEncoding(self, allEncodedData, referenceEncodedData=None, epoch=0, plotTitle="Signal Encoding", numSignalPlots=1):
        batchSize, numCondensedSignals, compressedLength = allEncodedData.shape
        # allEncodedData dimension: batchSize, numCondensedSignals, compressedLength

        # Get the signals to plot.
        plottingSignals = np.arange(0, numSignalPlots)
        plottingSignals = np.concatenate((plottingSignals, np.sort(numCondensedSignals - plottingSignals - 1)))

        batchInd = 0
        for signalInd in plottingSignals:
            # Plot the signal reconstruction.
            plt.plot(allEncodedData[batchInd, 0], 'k', label=f"batchInd{batchInd}-signalInd0", linewidth=2, alpha=1)
            plt.plot(allEncodedData[batchInd, 1], 'tab:blue', label=f"batchInd{batchInd}-signalInd1", linewidth=2, alpha=1)
            plt.plot(allEncodedData[batchInd, -2], 'tab:red', label=f"batchInd{batchInd}-signalInd-2", linewidth=2, alpha=1)
            plt.plot(allEncodedData[batchInd, -1], 'tab:green', label=f"batchInd{batchInd}-signalInd-1", linewidth=2, alpha=1)

            if referenceEncodedData is not None:
                # Plot the signal reconstruction.
                plt.plot(referenceEncodedData[batchInd, 0], 'k', label=f"Ref-batchInd{batchInd}-signalInd0", linewidth=1, alpha=0.6)
                plt.plot(referenceEncodedData[batchInd, 1], 'tab:blue', label=f"Ref-batchInd{batchInd}-signalInd1", linewidth=1, alpha=0.6)
                plt.plot(referenceEncodedData[batchInd, -2], 'tab:red', label=f"Ref-batchInd{batchInd}-signalInd-2", linewidth=1, alpha=0.6)
                plt.plot(referenceEncodedData[batchInd, -1], 'tab:green', label=f"Ref-batchInd{batchInd}-signalInd-1", linewidth=1, alpha=0.6)
            plt.legend()

            plt.xlabel("Encoding Dimension (Points)")
            plt.ylabel("Signal (AU)")
            plt.title(f"{plotTitle}")
            if self.saveDataFolder:
                self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch} batchInd{batchInd}.pdf")
            plt.show()

            # There are too many signals to plot.
            if batchInd + 1 == numSignalPlots: break

    def plotSignalEncodingMap(self, model, allEncodedData, allSignalData, epoch, plotTitle="Signal Encoding", numBatchPlots=1):
        batchSize, numSignals, signalDimension = allSignalData.shape
        # allEncodedData dimension: batchSize, numCondensedSignals, compressedLength

        # Find the number of relevant signals to the first encoding.
        numSignalForwardPath = model.signalEncoderModel.encodeSignals.simulateSignalPath(allSignalData.size(1), allEncodedData.size(1))[0]
        roughEstNumSignals = (model.signalEncoderModel.encodeSignals.expansionFactor ** (len(numSignalForwardPath)-1)) / model.signalEncoderModel.encodeSignals.numCompressedSignals
        print(numSignalForwardPath, roughEstNumSignals, len(numSignalForwardPath), model.signalEncoderModel.encodeSignals.numCompressedSignals)
        roughEstNumSignals = math.ceil(roughEstNumSignals)

        for batchInd in range(batchSize):
            # Plot the signals.
            for signalInd in range(roughEstNumSignals):
                plt.plot(allSignalData[batchInd, signalInd], 'k', linewidth=2, alpha=1/(signalInd+1))

            for signalInd in range(model.signalEncoderModel.encodeSignals.numCompressedSignals):
                plt.plot(allEncodedData[batchInd, signalInd], 'tab:red', linewidth=2, alpha=1/(signalInd+1))

            plt.xlabel("Signal Dimension (Points)")
            plt.ylabel("Signal (AU)")
            plt.title(f"{plotTitle}")
            if self.saveDataFolder:
                self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch} batchInd{batchInd}.pdf")
            plt.show()

            # There are too many signals to plot.
            if batchInd + 1 == numBatchPlots: break
