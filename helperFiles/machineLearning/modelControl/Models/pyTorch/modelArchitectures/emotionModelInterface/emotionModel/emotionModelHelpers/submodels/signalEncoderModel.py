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
    def __init__(self, sequenceBounds, maxNumSignals, numEncodedSignals, numExpandedSignals, numPosEncodingLayers, numSigEncodingLayers, numPosLiftedChannels, numSigLiftedChannels, timeWindows, accelerator, plotDataFlow=False, debuggingResults=False):
        super(signalEncoderModel, self).__init__()
        # General model parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.plotDataFlow = plotDataFlow  # Whether to plot the encoding process. Type: bool
        self.timeWindows = timeWindows  # A list of all time windows to consider for the encoding.
        self.accelerator = accelerator  # Hugging face interface for model and data optimizations.

        # Signal encoder parameters.
        self.numPosLiftedChannels = numPosLiftedChannels  # The number of channels to lift to during positional encoding.
        self.numSigLiftedChannels = numSigLiftedChannels  # The number of channels to lift to during signal encoding.
        self.numPosEncodingLayers = numPosEncodingLayers  # The number of operator layers during positional encoding.
        self.numSigEncodingLayers = numSigEncodingLayers  # The number of operator layers during signal encoding.
        self.numExpandedSignals = numExpandedSignals  # The number of signals in the expanded form for encoding to numExpandedSignals - 1.
        self.numEncodedSignals = numEncodedSignals  # The final number of signals to accept, encoding all signal information.
        self.sequenceBounds = sequenceBounds  # The minimum and maximum sequence lengths to consider.
        self.maxNumSignals = maxNumSignals  # The maximum number of signals to consider.

        # Method to converge to the final number of signals.
        self.encodeSignals = generalSignalEncoding(
            numPosEncodingLayers=self.numPosEncodingLayers,
            numSigEncodingLayers=self.numSigEncodingLayers,
            numPosLiftedChannels=self.numPosLiftedChannels,
            numSigLiftedChannels=self.numSigLiftedChannels,
            numExpandedSignals=self.numExpandedSignals,
            debuggingResults=self.debuggingResults,
            sequenceBounds=self.sequenceBounds,
        )

        # Initialize helper classes.
        self.trainingMethods = trainingSignalEncoder(numEncodedSignals, self.encodeSignals.expansionFactor)

        # Initialize loss holders.
        self.trainingLosses_timeReconstructionOptimalAnalysis = None
        self.testingLosses_timeReconstructionOptimalAnalysis = None
        self.trainingLosses_timeReconstructionAnalysis = None
        self.testingLosses_timeReconstructionAnalysis = None
        self.numEncodingsBufferPath_timeAnalysis = None
        self.trainingLosses_timeLayerAnalysis = None
        self.testingLosses_timeLayerAnalysis = None
        self.trainingLosses_timeMeanAnalysis = None
        self.testingLosses_timeMeanAnalysis = None
        self.trainingLosses_timeSTDAnalysis = None
        self.testingLosses_timeSTDAnalysis = None
        self.numEncodingsPath_timeAnalysis = None

        # Reset the model.
        self.resetModel()

    def resetModel(self):
        # Signal encoder reconstructed loss holders.
        self.trainingLosses_timeReconstructionAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeReconstructionAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs
        # Time analysis loss methods.
        self.trainingLosses_timeLayerAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeLayerAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

        # Signal encoder mean loss holders.
        self.trainingLosses_timeMeanAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded mean training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeMeanAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded mean testing losses. Dim: numTimeWindows, numEpochs
        # Signal encoder standard deviation loss holders.
        self.trainingLosses_timeSTDAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded standard deviation training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeSTDAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded standard deviation testing losses. Dim: numTimeWindows, numEpochs

        # Compression analysis.
        self.numEncodingsBufferPath_timeAnalysis = [[] for _ in self.timeWindows]  # List of list of buffers at each epoch. Dim: numTimeWindows, numEpochs
        self.numEncodingsPath_timeAnalysis = [[] for _ in self.timeWindows]  # List of list of the number of compressions at each epoch. Dim: numTimeWindows, numEpochs

        # Signal encoder optimal reconstruction loss holders
        self.trainingLosses_timeReconstructionOptimalAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeReconstructionOptimalAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

    def setDebuggingResults(self, debuggingResults):
        self.encodeSignals.channelEncodingInterface.debuggingResults = debuggingResults
        self.encodeSignals.positionalEncodingInterface.debuggingResults = debuggingResults
        self.encodeSignals.finalVarianceInterface.debuggingResults = debuggingResults
        self.encodeSignals.denoiseSignals.debuggingResults = debuggingResults
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
        initialDecodedData = None
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

        # Smooth over the encoding space.
        positionEncodedData = self.encodeSignals.denoiseSignals.applySmoothing_forPosEnc(positionEncodedData)

        # Compress the signal space into numEncodedSignals.
        initialEncodedData, numSignalForwardPath, signalEncodingLayerLoss = self.encodeSignals(signalData=positionEncodedData, targetNumSignals=numEncodedSignals, signalEncodingLayerLoss=None, calculateLoss=calculateLoss)
        # initialEncodedData dimension: batchSize, numEncodedSignals, sequenceLength

        # Allow the model to adjust the incoming signals
        encodedData = self.encodeSignals.finalVarianceInterface.adjustSignalVariance(initialEncodedData)
        # adjustedData dimension: batchSize, numEncodedSignals, sequenceLength

        # Smooth over the encoding space.
        encodedData = self.encodeSignals.denoiseSignals.applySmoothing_forVar(encodedData)

        # ---------------------- Signal Reconstruction --------------------- #

        if self.debuggingResults: print("Signal Encoding Downward Path:", numSignals, numSignalForwardPath, numEncodedSignals)

        if decodeSignals:
            # Perform the reverse operation.
            initialDecodedData, decodedData, reconstructedData, denoisedReconstructedData, signalEncodingLayerLoss = \
                self.reconstructEncodedData(encodedData, numSignalForwardPath, signalEncodingLayerLoss=signalEncodingLayerLoss, calculateLoss=calculateLoss)

            # Normalize each encoding loss.
            signalEncodingLayerLoss = signalEncodingLayerLoss / len(numSignalForwardPath)

        # ------------------------ Loss Calculations ----------------------- #

        if calculateLoss and decodeSignals:
            # Prepare for loss calculations.
            removedStampEncoding = self.encodeSignals.positionalEncodingInterface.removePositionalEncoding(positionEncodedData)
            # Calculate the immediately reconstructed data.
            potentialEncodedData, _, _ = self.encodeSignals(signalData=signalData, targetNumSignals=numEncodedSignals, signalEncodingLayerLoss=None, calculateLoss=False)
            potentialDecodedData, _, _ = self.reverseEncoding(signalEncodingLayerLoss=None, numSignalPath=numSignalForwardPath, decodedData=potentialEncodedData, calculateLoss=False)
            # Prepare for loss calculations.
            potentialEncodedData = self.encodeSignals.finalVarianceInterface.adjustSignalVariance(signalData)
            potentialEncodedData = self.encodeSignals.denoiseSignals.applySmoothing_forVar(potentialEncodedData)
            potentialSignalData = self.encodeSignals.finalVarianceInterface.unAdjustSignalVariance(potentialEncodedData)

            # Calculate the loss by comparing encoder/decoder outputs.
            varReconstructionStateLoss = (initialEncodedData - initialDecodedData).pow(2).mean(dim=2).mean(dim=1)
            encodingReconstructionStateLoss = (positionEncodedData - decodedData).pow(2).mean(dim=2).mean(dim=1)
            finalReconstructionStateLoss = (signalData - reconstructedData).pow(2).mean(dim=2).mean(dim=1)
            finalDenoisedReconstructionStateLoss = (initialSignalData - denoisedReconstructedData).pow(2).mean(dim=2).mean(dim=1)
            if self.debuggingResults: print("State Losses (VEF-D):", varReconstructionStateLoss.detach().mean().item(), encodingReconstructionStateLoss.detach().mean().item(), finalReconstructionStateLoss.detach().mean().item(), finalDenoisedReconstructionStateLoss.detach().mean().item())
            # Calculate the loss from taking other routes
            positionReconstructionLoss = (signalData - removedStampEncoding).pow(2).mean(dim=2).mean(dim=1)
            encodingReconstructionLoss = (signalData - potentialDecodedData).pow(2).mean(dim=2).mean(dim=1)
            potentialVarReconstructionStateLoss = (signalData - potentialSignalData).pow(2).mean(dim=2).mean(dim=1)
            if self.debuggingResults: print("Path Losses (P-E-V2-S):", positionReconstructionLoss.detach().mean().item(), encodingReconstructionLoss.detach().mean().item(), potentialVarReconstructionStateLoss.detach().mean().item(), signalEncodingLayerLoss.detach().mean().item())

            # Always add the final reconstruction loss (not denoised).
            signalEncodingLoss = signalEncodingLoss + finalReconstructionStateLoss

            # Add up all the state losses together.
            if (positionReconstructionLoss.mean() < 0.1 and potentialVarReconstructionStateLoss.mean() < 0.1) and 0.1 < encodingReconstructionStateLoss.mean():
                signalEncodingLoss = signalEncodingLoss + encodingReconstructionStateLoss
            if (positionReconstructionLoss.mean() < 0.1 and encodingReconstructionStateLoss.mean() < 0.1) and 0.1 < varReconstructionStateLoss.mean():
                signalEncodingLoss = signalEncodingLoss + varReconstructionStateLoss
            # Add up all the path losses together.
            if 0.001 < potentialVarReconstructionStateLoss.mean():
                signalEncodingLoss = signalEncodingLoss + potentialVarReconstructionStateLoss
            if 0.001 < encodingReconstructionLoss.mean():
                signalEncodingLoss = signalEncodingLoss + encodingReconstructionLoss
            if 0.001 < positionReconstructionLoss.mean():
                signalEncodingLoss = signalEncodingLoss + positionReconstructionLoss
            # Add up all the layer losses together.
            if 0.01 < signalEncodingLayerLoss.mean():
                signalEncodingLoss = signalEncodingLoss + signalEncodingLayerLoss

            if self.plotDataFlow and random.random() < 0.015:
                self.plotDataFlowDetails(initialSignalData, positionEncodedData, initialEncodedData, encodedData, initialDecodedData, decodedData, reconstructedData, denoisedReconstructedData)

        return encodedData, denoisedReconstructedData, signalEncodingLoss

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
        # Undo what was done in the initial adjustment.
        initialDecodedData = self.encodeSignals.finalVarianceInterface.unAdjustSignalVariance(encodedData)

        # Undo the signal encoding.
        decodedData, reversePath, signalEncodingLayerLoss = self.reverseEncoding(
            signalEncodingLayerLoss=signalEncodingLayerLoss,
            numSignalPath=numSignalForwardPath,
            decodedData=initialDecodedData,
            calculateLoss=calculateLoss,
        )
        # reconstructedInitEncodingData dimension: batchSize, numSignals, sequenceLength
        if self.debuggingResults: print("Signal Encoding Upward Path:", encodedData.size(1), reversePath, decodedData.size(1))
        assert reversePath[1:] == numSignalForwardPath[1:][::-1], f"Signal encoding path mismatch: {reversePath[1:]} != {numSignalForwardPath[1:][::-1]} reversed"

        # Learn how to remove positional encoding to each signal's position.
        reconstructedData = self.encodeSignals.positionalEncodingInterface.removePositionalEncoding(decodedData)

        # Denoise the final signals.
        denoisedReconstructedData = self.encodeSignals.denoiseSignals.applyDenoiser(reconstructedData)

        return initialDecodedData, decodedData, reconstructedData, denoisedReconstructedData, signalEncodingLayerLoss

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
    def plotDataFlowDetails(initialSignalData, positionEncodedData, initialEncodedData, encodedData, initialDecodedData, decodedData, reconstructedData, denoisedReconstructedData):
        plt.plot(initialSignalData[0][0].cpu().detach().numpy(), 'k', linewidth=2)
        plt.plot(positionEncodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2)
        plt.show()

        plt.plot(positionEncodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2)
        plt.plot(initialEncodedData[0][0].cpu().detach().numpy(), 'tab:blue', linewidth=2)
        plt.show()

        plt.plot(initialEncodedData[0][0].cpu().detach().numpy(), 'tab:blue', linewidth=2)
        plt.plot(encodedData[0][0].cpu().detach().numpy(), 'tab:green', linewidth=2)
        plt.show()

        plt.plot(initialEncodedData[0][0].cpu().detach().numpy(), 'tab:blue', linewidth=2)
        plt.plot(initialDecodedData[0][0].cpu().detach().numpy(), 'tab:blue', linewidth=2, alpha=0.5)
        plt.show()

        plt.plot(positionEncodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2)
        plt.plot(decodedData[0][0].cpu().detach().numpy(), 'tab:red', linewidth=2, alpha=0.5)
        plt.show()

        plt.plot(initialSignalData[0][0].cpu().detach().numpy(), 'k', linewidth=2)
        plt.plot(reconstructedData[0][0].cpu().detach().numpy(), 'k', linewidth=2, alpha=0.5)
        plt.plot(denoisedReconstructedData[0][0].cpu().detach().numpy(), 'k', linewidth=2, alpha=0.25)
        plt.show()

        # Clear the figure
        globalPlottingProtocols.clearFigure()
