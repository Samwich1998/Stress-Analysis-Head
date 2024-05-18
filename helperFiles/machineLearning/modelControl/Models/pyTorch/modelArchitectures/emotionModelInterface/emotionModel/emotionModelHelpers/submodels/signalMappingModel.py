# General
import torch

# Import files for machine learning
from .modelComponents.signalMappingHead import signalMappingHead
from ....._globalPytorchModel import globalModel


class signalMappingModel(globalModel):
    def __init__(self, compressedLength, numEncodedSignals, featureNames, timeWindows):
        super(signalMappingModel, self).__init__()
        # General model parameters.
        self.numEncodedSignals = numEncodedSignals  # The dimensionality of the encoded data.
        self.compressedLength = compressedLength  # The initial length of each incoming signal.
        self.numSignals = len(featureNames)  # The number of original signals from the specific dataset.
        self.featureNames = featureNames  # The names of the original feature/signal in the model. Dim: numSignals
        self.timeWindows = timeWindows

        # Method to find a common signal-agnostic space.
        self.mapSignals = signalMappingHead(maxSequenceLength=self.compressedLength)

        # Initialize loss holders.
        self.trainingLosses_signalReconstruction = None
        self.testingLosses_signalReconstruction = None
        self.trainingLosses_timeAnalysis = None
        self.testingLosses_timeAnalysis = None
        self.trainingLosses_mappedMean = None
        self.testingLosses_mappedMean = None
        self.trainingLosses_mappedSTD = None
        self.testingLosses_mappedSTD = None

        # Reset the model
        self.resetModel()

    def resetModel(self):
        # Autoencoder manifold reconstructed loss holders.
        self.trainingLosses_signalReconstruction = []  # List of manifold reconstruction (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_signalReconstruction = []  # List of manifold reconstruction (autoencoder) testing losses. Dim: numEpochs

        # Autoencoder manifold mean loss holders.
        self.trainingLosses_mappedMean = []  # List of manifold mean (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_mappedMean = []  # List of manifold mean (autoencoder) testing losses. Dim: numEpochs
        # Autoencoder manifold standard deviation loss holders.
        self.trainingLosses_mappedSTD = []  # List of manifold standard deviation (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_mappedSTD = []  # List of manifold standard deviation (autoencoder) testing losses. Dim: numEpochs

        # Time analysis loss methods.
        self.trainingLosses_timeAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction training losses. Dim: numEpochs
        self.testingLosses_timeAnalysis = [[] for _ in self.timeWindows]  # List of list of data reconstruction testing losses. Dim: numEpochs

    def forward(self, compressedData, remapSignals=False, trainingFlag=False):
        """ The shape of inputData: (batchSize, combinedSignalLength) """

        # ----------------------- Data Preprocessing ----------------------- #  

        # Extract the incoming data's dimension.
        batchSize, numEncodedSignals, compressedLength = compressedData.size()
        # compressedData dimension: batchSize, numEncodedSignals, compressedLength

        # Assert the integrity of the incoming data.
        assert numEncodedSignals == self.numEncodedSignals, f"The model was expecting {self.numEncodedSignals} signals, but received {numEncodedSignals}"
        assert compressedLength == self.compressedLength, f"The signals have length {compressedLength}, but the model expected {self.compressedLength} points."

        # Initialize holders for final values.
        reconstructedCompressedData = torch.zeros_like(compressedData, device=compressedData.device)
        # reconstructedCompressedData dimension: batchSize, numEncodedSignals, compressedLength

        # ------------------------ Signal Encoding ------------------------- # 

        # Inject model-specific signal information.
        mappedSignalData = self.mapSignals.addSignalInfo(compressedData)
        # reconstructedCompressedData dimension: batchSize, numEncodedSignals, compressedLength

        # ---------------------- Signal Reconstruction --------------------- #  

        # If we are reconstructing the signals.
        if remapSignals:
            # Reconstruct the original signals.
            reconstructedCompressedData = self.mapSignals.removeSignalInfo(mappedSignalData)
            # reconstructedEncodedData dimension: batchSize, numEncodedSignals, compressedLength

        return mappedSignalData, reconstructedCompressedData

        # ------------------------------------------------------------------ #
