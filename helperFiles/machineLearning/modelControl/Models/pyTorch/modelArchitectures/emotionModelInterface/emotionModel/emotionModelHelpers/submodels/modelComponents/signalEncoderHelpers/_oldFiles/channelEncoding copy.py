# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class channelEncoding(signalEncoderModules):

    def __init__(self, numCompressedSignals, numExpandedSignals, expansionFactor, numEncoderLayers):
        super(channelEncoding, self).__init__()
        # General parameters
        self.numCompressedSignals = numCompressedSignals
        self.numExpandedSignals = numExpandedSignals
        self.numEncoderLayers = numEncoderLayers
        self.expansionFactor = expansionFactor

        # Initialize initial models.
        self.initialCompressionModel = self.preProcessChannels(inChannel=self.numExpandedSignals, outChannel=self.numCompressedSignals)
        self.initialExpansionModel = self.preProcessChannels(inChannel=self.numCompressedSignals, outChannel=self.numExpandedSignals)

        # Initialize encoder models.
        self.channelExpanders = nn.ModuleList([])
        self.channelCompressors = nn.ModuleList([])
        # Initialize learner models.
        self.postprocessCompression = nn.ModuleList([])
        self.postprocessExpansion = nn.ModuleList([])

        # For each encoder model.
        for modelInd in range(self.numEncoderLayers):
            # Create the encoders and decoders
            self.channelExpanders.append(self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals))
            self.channelCompressors.append(self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals))

            # Learn channel information.
            self.postprocessCompression.append(self.learnModuleEncoding(inChannel=self.numCompressedSignals))
            self.postprocessExpansion.append(self.learnModuleEncoding(inChannel=self.numExpandedSignals))

        # Initialize final models.
        self.finalCompressionModel = self.postProcessChannels(inChannel=self.numCompressedSignals)
        self.finalExpansionModel = self.postProcessChannels(inChannel=self.numExpandedSignals)

    # ---------------------------------------------------------------------- #
    # ----------------------- Signal Encoding Methods ---------------------- #

    def compressionAlgorithm(self, inputData):
        # Learn the initial signal.
        processedData = checkpoint(self.initialCompressionModel, inputData, use_reentrant=False)

        # For each encoder model.
        for modelInd in range(self.numEncoderLayers):
            # Learn how to add the input data into the model.
            encodedData = self.applyLearningConv(processedData, inputData, self.channelCompressors[modelInd], addCheckpoint=True) + processedData

            # Learn how to synthesize the data from the model.
            processedData = checkpoint(self.postprocessCompression[modelInd], encodedData, use_reentrant=False)

        # Learn the final signal.
        processedData = checkpoint(self.finalCompressionModel, processedData, use_reentrant=False)

        return processedData

    def expansionAlgorithm(self, inputData):
        # Learn the initial signal.
        processedData = checkpoint(self.initialExpansionModel, inputData, use_reentrant=False)

        # For each encoder model.
        for modelInd in range(self.numEncoderLayers):
            # Learn how to add the input data into the model.
            encodedData = self.applyLearningConv(processedData, inputData, self.channelExpanders[modelInd], addCheckpoint=True) + processedData

            # Learn how to synthesize the data from the model.
            processedData = checkpoint(self.postprocessExpansion[modelInd], encodedData, use_reentrant=False)

        # Learn the final signal.
        processedData = checkpoint(self.finalExpansionModel, processedData, use_reentrant=False)

        return processedData

    @staticmethod
    def applyLearningConv(learnedData, inputData, combinationModule, addCheckpoint=False):
        # Concatenate the information as different channels
        processedData = torch.cat((learnedData, inputData), dim=1)

        if addCheckpoint:
            return checkpoint(combinationModule, processedData, use_reentrant=False)
        return combinationModule(processedData)
