# PyTorch
import torch

# Import files for machine learning
from .helperModules.trainingAutoEncoder import trainingAutoEncoder
from .modelComponents.generalAutoencoder import generalAutoencoder
from ...._globalPytorchModel import globalModel


class autoencoderModel(globalModel):
    def __init__(self, compressedLength, timeWindows, compressionFactor, expansionFactor, accelerator, debuggingResults=False):
        super(autoencoderModel, self).__init__()
        # General model parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.timeWindows = timeWindows  # A list of all time windows to consider for the encoding. Type: list
        self.accelerator = accelerator  # Hugging face model optimizations.

        # Autoencoder parameters.
        self.compressionFactor = compressionFactor  # The expansion factor of the autoencoder. Type: float
        self.compressedLength = compressedLength  # The final length of the compressed signal after the autoencoder. Type: int
        self.expansionFactor = expansionFactor  # The expansion factor of the autoencoder. Type: float

        # Gradient accumulation parameters.
        self.numAccumulations = 0   # The number of gradient accumulations.
        self.accumulatedLoss = 0    # The accumulated loss for gradient accumulation.

        # Method to reconstruct the original signal.
        self.generalAutoencoder = generalAutoencoder(accelerator, compressionFactor, expansionFactor)

        # Initialize helper classes.
        self.trainingMethods = trainingAutoEncoder(compressedLength, compressionFactor, expansionFactor)

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

        self.trainingLosses_timeReconstructionSVDAnalysis = None
        self.testingLosses_timeReconstructionSVDAnalysis = None

        # Reset the model.
        self.resetModel()

    def resetModel(self):
        # Autoencoder reconstructed loss holders.
        self.trainingLosses_timeReconstructionAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeReconstructionAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs
        # Time analysis loss methods.
        self.trainingLosses_timeLayerAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeLayerAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

        # Autoencoder mean loss holders.
        self.trainingLosses_timeMeanAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded mean training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeMeanAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded mean testing losses. Dim: numTimeWindows, numEpochs
        # Autoencoder standard deviation loss holders.
        self.trainingLosses_timeSTDAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded standard deviation training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeSTDAnalysis = [[] for _ in self.timeWindows]  # List of list of encoded standard deviation testing losses. Dim: numTimeWindows, numEpochs

        # Compression analysis.
        self.numEncodingsBufferPath_timeAnalysis = [[] for _ in self.timeWindows]  # List of list of buffers at each epoch. Dim: numTimeWindows, numEpochs
        self.numEncodingsPath_timeAnalysis = [[] for _ in self.timeWindows]  # List of list of the number of compressions at each epoch. Dim: numTimeWindows, numEpochs

        # Autoencoder optimal reconstruction loss holders
        self.trainingLosses_timeReconstructionOptimalAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numTimeWindows, numEpochs
        self.testingLosses_timeReconstructionOptimalAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numTimeWindows, numEpochs

        # Keep track of gradient accumulation.
        self.numAccumulations = 0
        self.accumulatedLoss = 0

    def forward(self, encodedData, reconstructSignals=False, calculateLoss=False, trainingFlag=False):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        print("\nEntering autoencoder model", flush=True)

        # ----------------------- Data Preprocessing ----------------------- #  

        # Prepare the data for compression/expansion
        batchSize, numSignals, initialSequenceLength = encodedData.size()
        # encodedData dimension: batchSize, numSignals, initialSequenceLength

        # Create placeholders for the final variables.
        bridgedReconstructedEncodedData = torch.zeros_like(encodedData, device=encodedData.device)
        autoencoderLoss = torch.zeros(batchSize, device=encodedData.device)
        # reconstructedEncodedData dimension: batchSize, numSignals, initialSequenceLength
        # autoencoderLoss dimension: batchSize

        # Initialize training parameters
        reconstructedInitialCompressedData = None
        reconstructedEncodedData = None

        # ---------------------- Training Augmentation --------------------- #

        # Initialize augmentation parameters
        compressedLength = self.compressedLength
        forwardDirection = None
        totalNumEncodings = 0

        if trainingFlag:
            # Set up the training parameters
            assert reconstructSignals and calculateLoss, f"Training requires decoding and loss calculations. reconstructSignals: {reconstructSignals}, calculateLoss: {calculateLoss}"
            compressedLength, totalNumEncodings, forwardDirection = self.trainingMethods.augmentFinalTarget(initialSequenceLength)

        # --------------------- Signal Compression --------------------- # 

        # Data reduction: remove unnecessary timepoints from the signals.
        noisyEncodedData = self.generalAutoencoder.dataInterface.addNoise(encodedData, trainingFlag, noiseSTD=0.01)
        initialCompressedData, numSignalPath, autoencoderLayerLoss = self.generalAutoencoder(inputData=noisyEncodedData, targetSequenceLength=compressedLength, initialSequenceLength=initialSequenceLength, autoencoderLayerLoss=None, calculateLoss=calculateLoss)
        # compressedData dimension: batchSize, numSignals, compressedLength
        print("Autoencoder Downward path:", encodedData.size(2), numSignalPath, initialCompressedData.size(2), flush=True)

        # Adjust the final statistics of the data.
        compressedData = self.generalAutoencoder.adjustSignalVariance(initialCompressedData)

        # -------------------- Signal Reconstruction ------------------- #  

        if reconstructSignals:
            # Reconstruct the original signal points.
            noisyCompressedData, reconstructedInitialCompressedData, reconstructedEncodedData, bridgedReconstructedEncodedData, numSignalPath, autoencoderLayerLoss = \
                    self.decompressData(compressedData, initialSequenceLength, initialSequenceLength, autoencoderLayerLoss=autoencoderLayerLoss, calculateLoss=calculateLoss)

        # ------------------------ Loss Calculations ----------------------- #

        if calculateLoss and reconstructSignals:
            # Calculate the immediately reconstructed data.
            halfReconstructedData, _, _ = self.generalAutoencoder(initialSequenceLength=initialSequenceLength, targetSequenceLength=initialSequenceLength, inputData=initialCompressedData, calculateLoss=False, autoencoderLayerLoss=None)
            # Calculate the immediately reconstructing variance.
            varianceHolder = self.generalAutoencoder.adjustSignalVariance(encodedData)
            varReconstructedEncodedData = self.generalAutoencoder.unAdjustSignalVariance(varianceHolder)

            # Calculate the loss by comparing encoder/decoder outputs.
            finalReconstructionStateLoss = (encodedData - reconstructedEncodedData).pow(2).mean(dim=2).mean(dim=1)
            finalDenoisedReconstructionStateLoss = (encodedData - bridgedReconstructedEncodedData).pow(2).mean(dim=2).mean(dim=1)
            varReconstructionStateLoss = (initialCompressedData - reconstructedInitialCompressedData).pow(2).mean(dim=-1).mean(dim=1)
            print("State Losses (VF-D):", varReconstructionStateLoss.detach().mean().item(), finalReconstructionStateLoss.detach().mean().item(), finalDenoisedReconstructionStateLoss.detach().mean().item())
            # Calculate the loss from taking other routes
            autoencoderLayerLoss = autoencoderLayerLoss.view(batchSize, numSignals).mean(dim=1)
            encodingReconstructionLoss = (encodedData - halfReconstructedData).pow(2).mean(dim=-1).mean(dim=1)
            varReconstructionLoss = (encodedData - varReconstructedEncodedData).pow(2).mean(dim=-1).mean(dim=1)
            print("Path Losses (E2-V2-S):", encodingReconstructionLoss.detach().mean().item(), varReconstructionLoss.detach().mean().item(), autoencoderLayerLoss.detach().mean().item())

            # Add up all the losses together.
            if 0.001 < encodingReconstructionLoss.mean():
                autoencoderLoss = autoencoderLoss + 0.1*encodingReconstructionLoss
            if 0.001 < finalReconstructionStateLoss.mean():
                autoencoderLoss = autoencoderLoss + finalReconstructionStateLoss
            if 0.001 < varReconstructionStateLoss.mean():
                autoencoderLoss = autoencoderLoss + varReconstructionStateLoss
            if 0.001 < varReconstructionLoss.mean():
                autoencoderLoss = autoencoderLoss + 0.1*varReconstructionLoss
            if 0.001 < autoencoderLayerLoss.mean():
                autoencoderLoss = autoencoderLoss + 0.1*autoencoderLayerLoss

            if trainingFlag:
                self.trainingMethods.adjustNumEncodings(totalNumEncodings, autoencoderLayerLoss, finalDenoisedReconstructionStateLoss, forwardDirection)

        return compressedData, bridgedReconstructedEncodedData, autoencoderLoss

        # ------------------------------------------------------------------ #  

    def decompressData(self, compressedData, initialSequenceLength, targetSequenceLength, autoencoderLayerLoss=None, calculateLoss=False):
        # Remove any noise from the inner network.
        reconstructedInitialCompressedData = self.generalAutoencoder.unAdjustSignalVariance(compressedData)

        # Reconstruct to the current signal number in the path.
        reconstructedEncodedData, numSignalPath, autoencoderLayerLoss = self.generalAutoencoder(inputData=reconstructedInitialCompressedData,
                                                                                                initialSequenceLength=initialSequenceLength,
                                                                                                targetSequenceLength=targetSequenceLength,
                                                                                                autoencoderLayerLoss=autoencoderLayerLoss,
                                                                                                calculateLoss=calculateLoss)
        # reconstructedEncodedData dimension: batchSize, numSignals, sequenceLength
        print("Autoencoder Upward path:", numSignalPath, flush=True)

        # Denoise the final signals.
        bridgedReconstructedEncodedData = self.generalAutoencoder.applyAutoencoderDenoiser(reconstructedEncodedData)

        return compressedData, reconstructedInitialCompressedData, reconstructedEncodedData, bridgedReconstructedEncodedData, numSignalPath, autoencoderLayerLoss
