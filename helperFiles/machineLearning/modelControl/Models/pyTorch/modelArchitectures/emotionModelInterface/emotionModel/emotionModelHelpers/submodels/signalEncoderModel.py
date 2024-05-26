# General
import matplotlib.pyplot as plt
import random
import torch

from helperFiles.globalPlottingProtocols import globalPlottingProtocols
# Import files for machine learning
from .modelComponents.generalSignalEncoder import generalSignalEncoding  # Framework for encoding/decoding of all signals.
from .helperModules.trainingSignalEncoder import trainingSignalEncoder
from ..generalMethods.generalMethods import generalMethods
from ....._globalPytorchModel import globalModel


class signalEncoderModel(globalModel):
    def __init__(self, sequenceBounds, signalMinMaxScale, maxNumSignals, numEncodedSignals, numExpandedSignals, numSigEncodingLayers, numSigLiftedChannels, waveletType, timeWindows, accelerator, plotDataFlow=False, debuggingResults=False):
        super(signalEncoderModel, self).__init__()
        # General model parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.plotDataFlow = plotDataFlow  # Whether to plot the encoding process. Type: bool
        self.timeWindows = timeWindows  # A list of all time windows to consider for the encoding.
        self.accelerator = accelerator  # Hugging face interface for model and data optimizations.

        # Signal encoder parameters.
        self.numSigLiftedChannels = numSigLiftedChannels  # The number of channels to lift to during signal encoding.
        self.numSigEncodingLayers = numSigEncodingLayers  # The number of operator layers during signal encoding.
        self.numExpandedSignals = numExpandedSignals  # The number of signals in the expanded form for encoding to numExpandedSignals - 1.
        self.numEncodedSignals = numEncodedSignals  # The final number of signals to accept, encoding all signal information.
        self.signalMinMaxScale = signalMinMaxScale  # The minimum and maximum signal values to scale the data.
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence lengths to consider.
        self.maxNumSignals = maxNumSignals  # The maximum number of signals to consider.
        self.waveletType = waveletType  # The type to use during the signal encoder.

        # Method to converge to the final number of signals.
        self.encodeSignals = generalSignalEncoding(
            numSigEncodingLayers=self.numSigEncodingLayers,
            numSigLiftedChannels=self.numSigLiftedChannels,
            numExpandedSignals=self.numExpandedSignals,
            signalMinMaxScale=self.signalMinMaxScale,
            debuggingResults=self.debuggingResults,
            sequenceBounds=self.sequenceBounds,
            waveletType=self.waveletType,
        )

        # Initialize helper classes.
        self.trainingMethods = trainingSignalEncoder(numEncodedSignals, self.encodeSignals.expansionFactor, self.encodeSignals.positionalEncodingInterface.maxNumEncodedSignals)

        # Initialize loss holders.
        self.trainingLosses_timeReconstructionOptimalAnalysis = None
        self.testingLosses_timeReconstructionOptimalAnalysis = None
        self.trainingLosses_timeReconstructionAnalysis = None
        self.testingLosses_timeReconstructionAnalysis = None
        self.trainingLosses_timePosEncAnalysis = None
        self.testingLosses_timePosEncAnalysis = None
        self.trainingLosses_timeMeanAnalysis = None
        self.testingLosses_timeMeanAnalysis = None
        self.trainingLosses_timeMinMaxAnalysis = None
        self.testingLosses_timeMinMaxAnalysis = None

        # Reset the model.
        self.resetModel()

    def resetModel(self):
        # Signal encoder reconstructed loss holders.
        self.trainingLosses_timeReconstructionAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeReconstructionAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

        # Signal encoder mean loss holders.
        self.trainingLosses_timeMeanAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded mean training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeMeanAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded mean testing losses. Dim: numTimeWindows, numEpochs
        # Signal encoder standard deviation loss holders.
        self.trainingLosses_timeMinMaxAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded standard deviation training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeMinMaxAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded standard deviation testing losses. Dim: numTimeWindows, numEpochs

        # Positional encoding analysis.
        self.trainingLosses_timePosEncAnalysis = [[] for _ in self.timeWindows]  # List of list of positional encoding reconstruction losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timePosEncAnalysis = [[] for _ in self.timeWindows]  # List of list of positional encoding reconstruction losses. Dim: numTimeWindows, numEpochs

        # Signal encoder optimal reconstruction loss holders
        self.trainingLosses_timeReconstructionOptimalAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeReconstructionOptimalAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

    def setDebuggingResults(self, debuggingResults):
        self.encodeSignals.debuggingResults = debuggingResults
        self.debuggingResults = debuggingResults

    def forward(self, signalData, initialSignalData, decodeSignals=False, calculateLoss=False, trainingFlag=False):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        if self.debuggingResults: print("\nEntering signal encoder model")

        # ----------------------- Data Preprocessing ----------------------- #  

        # Prepare the data for compression/expansion
        batchSize, numSignals, sequenceLength = signalData.size()
        # signalData dimension: batchSize, numSignals, sequenceLength

        # Create placeholders for the final variables.
        denoisedReconstructedData = torch.zeros_like(signalData, device=signalData.device)
        signalEncodingLoss = torch.zeros((batchSize,), device=signalData.device)
        # denoisedReconstructedData dimension: batchSize, numSignals, sequenceLength
        # signalEncodingLoss dimension: batchSize

        # Initialize training parameters
        reconstructedData = None
        decodedData = None

        # ---------------------- Training Augmentation --------------------- #  

        # Initialize augmentation parameters
        numEncodedSignals = self.numEncodedSignals

        if trainingFlag:
            if self.accelerator.sync_gradients:
                # Randomly change encoding directions.
                self.trainingMethods.randomlyChangeDirections()

            # Set up the training parameters
            numEncodedSignals = self.trainingMethods.augmentFinalTarget(numSignals)

        # ------------------- Learned Signal Compression ------------------- #

        # Learn how to add positional encoding to each signal's position.
        positionEncodedData = self.encodeSignals.positionalEncodingInterface.addPositionalEncoding(signalData)
        # positionEncodedData dimension: batchSize, numSignals, sequenceLength

        # Predict the positional encoding index.
        predictedIndexProbabilities = self.encodeSignals.positionalEncodingInterface.predictSignalIndex(positionEncodedData)
        predictedIndexProbabilities = self.encodeSignals.denoiseSignals.applySmoothing_forPosPreds(predictedIndexProbabilities)  # Smooth over the encoding space.
        # predictedIndexProbabilities dimension: batchSize, numSignals, maxNumEncodingSignals

        # Compress the signal space into numEncodedSignals.
        initialEncodedData, numSignalForwardPath, signalEncodingLayerLoss = self.encodeSignals(signalData=positionEncodedData, targetNumSignals=numEncodedSignals, signalEncodingLayerLoss=None, calculateLoss=calculateLoss)
        # initialEncodedData dimension: batchSize, numEncodedSignals, sequenceLength

        # Allow the model to adjust the incoming signals
        encodedData = self.encodeSignals.finalVarianceInterface.adjustSignalVariance(initialEncodedData)
        # adjustedData dimension: batchSize, numEncodedSignals, sequenceLength

        # ---------------------- Signal Reconstruction --------------------- #

        if self.debuggingResults: print("Signal Encoding Downward Path:", numSignals, numSignalForwardPath, numEncodedSignals)

        if decodeSignals:
            # Perform the reverse operation.
            decodedData, reconstructedData, denoisedReconstructedData, signalEncodingLayerLoss = self.reconstructEncodedData(encodedData, numSignalForwardPath, signalEncodingLayerLoss=signalEncodingLayerLoss, calculateLoss=calculateLoss)
            signalEncodingLayerLoss = signalEncodingLayerLoss / len(numSignalForwardPath)  # Normalize each encoding loss.

        # ------------------------ Loss Calculations ----------------------- #

        if calculateLoss and decodeSignals:
            # Prepare for loss calculations.
            removedStampEncoding = self.encodeSignals.positionalEncodingInterface.removePositionalEncoding(positionEncodedData)
            # Calculate the immediately reconstructed data.
            potentialEncodedData, _, _ = self.encodeSignals(signalData=signalData, targetNumSignals=numEncodedSignals, signalEncodingLayerLoss=None, calculateLoss=False)
            potentialDecodedData, _, _ = self.reverseEncoding(signalEncodingLayerLoss=None, numSignalPath=numSignalForwardPath, decodedData=potentialEncodedData, calculateLoss=False)

            # Calculate the loss by comparing encoder/decoder outputs.
            encodingReconstructionStateLoss = (positionEncodedData - decodedData).pow(2).mean(dim=2).mean(dim=1)
            finalReconstructionStateLoss = (signalData - reconstructedData).pow(2).mean(dim=2).mean(dim=1)
            finalDenoisedReconstructionStateLoss = (initialSignalData - denoisedReconstructedData).pow(2).mean(dim=2).mean(dim=1)
            if self.debuggingResults: print("State Losses (EF-D):", encodingReconstructionStateLoss.detach().mean().item(), finalReconstructionStateLoss.detach().mean().item(), finalDenoisedReconstructionStateLoss.detach().mean().item())
            # Calculate the loss from taking other routes
            positionReconstructionLoss = (signalData - removedStampEncoding).pow(2).mean(dim=2).mean(dim=1)
            encodingReconstructionLoss = (signalData - potentialDecodedData).pow(2).mean(dim=2).mean(dim=1)
            if self.debuggingResults: print("Path Losses (P-E-S):", positionReconstructionLoss.detach().mean().item(), encodingReconstructionLoss.detach().mean().item(), signalEncodingLayerLoss.detach().mean().item())

            # Always add the final reconstruction loss (not denoised).
            signalEncodingLoss = signalEncodingLoss + finalReconstructionStateLoss

            # Add up all the state losses together.
            if positionReconstructionLoss.mean() < 0.1 < encodingReconstructionStateLoss.mean():
                signalEncodingLoss = signalEncodingLoss + encodingReconstructionStateLoss
            # Add up all the path losses together.
            if 0.001 < encodingReconstructionLoss.mean():
                signalEncodingLoss = signalEncodingLoss + encodingReconstructionLoss
            if 0.001 < positionReconstructionLoss.mean():
                signalEncodingLoss = signalEncodingLoss + positionReconstructionLoss
            # Add up all the layer losses together.
            if 0.01 < signalEncodingLayerLoss.mean():
                signalEncodingLoss = signalEncodingLoss + signalEncodingLayerLoss

            if self.plotDataFlow and random.random() < 0.01:
                self.plotDataFlowDetails(initialSignalData, positionEncodedData, initialEncodedData, encodedData, decodedData, reconstructedData, denoisedReconstructedData)

        return encodedData, denoisedReconstructedData, predictedIndexProbabilities, signalEncodingLoss

        # ------------------------------------------------------------------ #  

    def reverseEncoding(self, decodedData, numSignalPath, signalEncodingLayerLoss, calculateLoss):
        reversePath = []
        # Follow the path back to the original signal.
        for pathInd in range(len(numSignalPath) - 1, -1, -1):
            # Reconstruct to the current signal number in the path.
            decodedData, miniPath, signalEncodingLayerLoss \
                = self.encodeSignals(signalEncodingLayerLoss=signalEncodingLayerLoss,
                                     targetNumSignals=numSignalPath[pathInd],
                                     calculateLoss=calculateLoss,
                                     signalData=decodedData)
            reversePath.extend(miniPath)

        return decodedData, reversePath, signalEncodingLayerLoss

    def reconstructEncodedData(self, encodedData, numSignalForwardPath, signalEncodingLayerLoss=None, calculateLoss=False):
        # Undo the signal encoding.
        decodedData, reversePath, signalEncodingLayerLoss = self.reverseEncoding(
            signalEncodingLayerLoss=signalEncodingLayerLoss,
            numSignalPath=numSignalForwardPath,
            calculateLoss=calculateLoss,
            decodedData=encodedData,
        )
        # reconstructedInitEncodingData dimension: batchSize, numSignals, sequenceLength
        if self.debuggingResults: print("Signal Encoding Upward Path:", encodedData.size(1), reversePath, decodedData.size(1))
        assert reversePath[1:] == numSignalForwardPath[1:][::-1], f"Signal encoding path mismatch: {reversePath[1:]} != {numSignalForwardPath[1:][::-1]} reversed"

        # Learn how to remove positional encoding to each signal's position.
        reconstructedData = self.encodeSignals.positionalEncodingInterface.removePositionalEncoding(decodedData)

        # Denoise the final signals.
        denoisedReconstructedData = self.encodeSignals.denoiseSignals.applyDenoiser(reconstructedData)

        return decodedData, reconstructedData, denoisedReconstructedData, signalEncodingLayerLoss

    def calculateOptimalLoss(self, initialSignalData, printLoss=True):
        with torch.no_grad():
            # Perform the optimal compression via PCA and embed channel information (for reconstruction).
            pcaProjection, principal_components = generalMethods.svdCompression(initialSignalData, self.numEncodedSignals, standardizeSignals=True)
            # Loss for PCA reconstruction
            pcaReconstruction = torch.matmul(principal_components, pcaProjection)
            pcaReconstruction = (pcaReconstruction + initialSignalData.mean(dim=-1, keepdim=True)) * initialSignalData.std(dim=-1, keepdim=True)
            pcaReconstructionLoss = (initialSignalData - pcaReconstruction).pow(2).mean(dim=2).mean(dim=1)
            if printLoss: print("\tFIRST Optimal Compression Loss STD:", pcaReconstructionLoss.mean().item())

            return pcaReconstructionLoss

    @staticmethod
    def plotDataFlowDetails(initialSignalData, positionEncodedData, initialEncodedData, encodedData, decodedData, reconstructedData, denoisedReconstructedData):
        fig = plt.figure()
        plt.plot(initialSignalData[0][0].cpu().detach().numpy(), 'k', linewidth=2, label="Initial Data")
        plt.plot(positionEncodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2, label="Positional Encoding Data")
        plt.title("Positional Encoding"); plt.legend()
        globalPlottingProtocols.clearFigure(fig=fig)

        fig = plt.figure()
        plt.plot(positionEncodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2, label="Positional Encoding Data")
        plt.plot(initialEncodedData[0][0].cpu().detach().numpy(), 'tab:blue', linewidth=2, label="Encoded Data (before variance)")
        plt.title("Signal Encoding"); plt.legend()
        globalPlottingProtocols.clearFigure(fig=fig)

        fig = plt.figure()
        plt.plot(initialEncodedData[0][0].cpu().detach().numpy(), 'tab:blue', linewidth=2, label="Encoded Data (before variance)")
        plt.plot(encodedData[0][0].cpu().detach().numpy(), 'tab:green', linewidth=2, label="Variance Adjusted Data")
        plt.title("Variance Adjustment"); plt.legend()
        globalPlottingProtocols.clearFigure(fig=fig)

        fig = plt.figure()
        plt.plot(positionEncodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2, label="Positional Encoding Data")
        plt.plot(decodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2, alpha=0.5, label="Decoded Data (Backward Path)")
        plt.title("Position Decoding"); plt.legend()
        globalPlottingProtocols.clearFigure(fig=fig)

        fig = plt.figure()
        plt.plot(initialSignalData[0][0].cpu().detach().numpy(), 'k', linewidth=2, label="Initial Data")
        plt.plot(reconstructedData[0][0].cpu().detach().numpy(), 'k', linewidth=2, alpha=0.5, label="Reconstructed Data")
        plt.plot(denoisedReconstructedData[0][0].cpu().detach().numpy(), 'k', linewidth=2, alpha=0.25, label="Denoised Reconstructed Data")
        plt.title("Final Denoising"); plt.legend()
        globalPlottingProtocols.clearFigure(fig=fig)
