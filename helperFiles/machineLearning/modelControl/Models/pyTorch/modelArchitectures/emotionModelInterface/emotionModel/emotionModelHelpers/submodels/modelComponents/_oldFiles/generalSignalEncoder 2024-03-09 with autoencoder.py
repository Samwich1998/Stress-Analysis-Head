# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import time

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

from ._modelHelpers._convolutionalHelpers import ResNet
# Import helper models
from ._signalEncoderModules._signalEncoderModules import signalEncoderModules

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class signalEncoderBase(signalEncoderModules):
    def __init__(self, maxSequenceLength=64, numExpandedSignals=2, accelerator=None):
        super(signalEncoderBase, self).__init__(numExpandedSignals)
        # General parameters.
        self.maxSequenceLength = maxSequenceLength  # The maximum number of points in any signal.
        self.accelerator = accelerator  # Hugging face model optimizations.
        self.numEncodingStamps = 10  # The number of binary bits in the encoding (010 = 2 signals; 3 encodings).

        # A list of parameters to encode each signal.
        self.encodedStamp = nn.ParameterList()  # A list of learnable parameters for learnable signal positions.

        # For each encoding bit.
        for stampInd in range(self.numEncodingStamps):
            # Assign a learnable parameter to the signal.
            self.encodedStamp.append(torch.nn.Parameter(torch.randn(maxSequenceLength)))

        # Learn how to embed the positional information into the signals.
        self.learnSignalPositions = self.learnPositionalEncodingModule(inChannel=1)
        self.unlearnSignalPositions = self.learnPositionalEncodingModule(inChannel=1)
        # Learn how to embed the positional information into the signals.
        self.learnSignalPositionsCombine = self.combinePositionalEncodingModule(inChannel=2, outChannel=1)
        self.unlearnSignalPositionsCombine = self.combinePositionalEncodingModule(inChannel=2, outChannel=1)

        # Specify the ladder operators to integrate the learned information from the encoding location.
        self.raisingModuleExpansionCombine = self.combineLadderModule(inChannel=2*self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.loweringModuleExpansionCombine = self.combineLadderModule(inChannel=2*self.numExpandedSignals, outChannel=self.numExpandedSignals)
        # Specify the ladder operators to integrate the learned information from the encoding location.
        self.raisingModuleExpansion = self.learnLadderModule(inChannel=self.numCompressedSignals)
        self.loweringModuleExpansion = self.learnLadderModule(inChannel=self.numExpandedSignals)
        # Specify the ladder operators to account for dilation in expansion.
        self.loweringParamsExpansion = torch.nn.Parameter(torch.randn(8))
        self.raisingParamsExpansion = torch.nn.Parameter(torch.randn(8))

        # Specify the ladder operators to integrate the learned information from the encoding location.
        self.loweringModuleCompressionCombine = self.combineLadderModule(inChannel=2*self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.raisingModuleCompressionCombine = self.combineLadderModule(inChannel=2*self.numExpandedSignals, outChannel=self.numExpandedSignals)
        # Specify the ladder operators to integrate the learned information from the encoding location.
        self.loweringModuleCompression = self.learnLadderModule(inChannel=self.numCompressedSignals)
        self.raisingModuleCompression = self.learnLadderModule(inChannel=self.numExpandedSignals)
        # Specify the ladder operators to account for dilation in compression.
        self.loweringParamsCompression = torch.nn.Parameter(torch.randn(8))
        self.raisingParamsCompression = torch.nn.Parameter(torch.randn(8))

        # Learned compression via CNN.
        self.compressChannelsCNN_preprocessChannels = self.learnChannelModule(inChannel=self.numExpandedSignals)
        self.compressChannelsCNN_preprocessCombine = self.combineChannelModule(inChannel=2*self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.compressChannelsCNN = self.encodeChannels(inChannel=self.numExpandedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN_postprocessCombine = self.combineChannelModule(inChannel=2*self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN_postprocessChannels = self.learnChannelModule(inChannel=self.numCompressedSignals)

        # Learned expansion via CNN.
        self.expandChannelsCNN_preprocessChannels = self.learnChannelModule(inChannel=self.numCompressedSignals)
        self.expandChannelsCNN_preprocessCombine = self.combineChannelModule(inChannel=2*self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.expandChannelsCNN = self.encodeChannels(inChannel=self.numCompressedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN_postprocessCombine = self.combineChannelModule(inChannel=2*self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN_postprocessChannels = self.learnChannelModule(inChannel=self.numExpandedSignals)

        self.reduceSignalDimension = self.learnSignalChannelReductionPre(inChannel=self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.reconstructSignalDimension1 = self.learnSignalChannelReduction(inChannel=self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.reconstructSignalDimension2 = self.learnDimensionExpansion(inChannel=self.numExpandedSignals)
        self.reduceSignals1 = self.learnSignalChannelReduction(inChannel=self.numExpandedSignals, outChannel=self.numCompressedSignals)
        self.reduceSignals2 = self.learnSignalReduction(inChannel=self.numCompressedSignals)

        # Map the initial signals into a common subspace.
        self.adjustSignals = self.minorSubspaceTransformation(inChannel=1, numMidChannels=4)
        self.removeSignalAdjustment = self.minorSubspaceTransformation(inChannel=1, numMidChannels=4)

    # ---------------------------------------------------------------------- #
    # ------------------- Machine Learning Architectures ------------------- #

    def learnSignalChannelReductionPre(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            nn.Conv1d(in_channels=inChannel, out_channels=outChannel, kernel_size=3, stride=2, dilation=1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),

            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),
        )

    def learnSignalReduction(self, inChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),
        )

    def learnSignalChannelReduction(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            nn.ConvTranspose1d(in_channels=inChannel, out_channels=outChannel, kernel_size=3, stride=2, dilation=1, padding=1, padding_mode='zeros', groups=1, bias=True, output_padding=0),
            nn.SELU(),
        )

    def learnDimensionExpansion(self, inChannel=1):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),
        )

    # ------------------- Positional Encoding Architectures ------------------- #

    def learnPositionalEncodingModule(self, inChannel=1):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
        )

    def combinePositionalEncodingModule(self, inChannel=2, outChannel=1):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),

            # Convolution architecture: change channels
            self.convolutionalOneFilters(numChannels=[inChannel, outChannel], kernel_size=3, dilation=1, group=1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFiltersBlock(numChannels=[outChannel, 4 * outChannel, 2 * outChannel, outChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[outChannel, 2 * outChannel, 2 * outChannel, outChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[outChannel, 2 * outChannel, 2 * outChannel, outChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
        )

    # ------------------- Ladder Architectures ------------------- #

    def learnLadderModule(self, inChannel=1):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
        )

    def combineLadderModule(self, inChannel=1, outChannel=None):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),

            # Convolution architecture: change channels
            self.convolutionalOneFilters(numChannels=[inChannel, outChannel], kernel_size=3, dilation=1, group=1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFiltersBlock(numChannels=[outChannel, 4 * outChannel, 2 * outChannel, outChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[outChannel, 2 * outChannel, 2 * outChannel, outChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[outChannel, 2 * outChannel, 2 * outChannel, outChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
        )

    def learnChannelModule(self, inChannel=1):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
        )

    def combineChannelModule(self, inChannel=1, outChannel=1):
        return nn.Sequential(
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),

            # Convolution architecture: change channels
            self.convolutionalOneFilters(numChannels=[inChannel, outChannel], kernel_size=3, dilation=1, group=1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFiltersBlock(numChannels=[outChannel, 4 * outChannel, 2 * outChannel, outChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[outChannel, 2 * outChannel, 2 * outChannel, outChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            self.convolutionalThreeFiltersBlock(numChannels=[outChannel, 2 * outChannel, 2 * outChannel, outChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
        )

    def encodeChannels(self, inChannel=1, outChannel=2):
        return nn.Sequential(
            # Convolution architecture: change channels
            self.convolutionalOneFilters(numChannels=[inChannel, outChannel], kernel_size=3, dilation=1, group=1),
        )

    def minorSubspaceTransformation(self, inChannel=1, numMidChannels=4):
        return nn.Sequential(
            # Convolution architecture: feature engineering
            ResNet(module=nn.Sequential(
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 4 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
                self.convolutionalThreeFiltersBlock(numChannels=[inChannel, 2 * inChannel, 2 * inChannel, inChannel], kernel_sizes=3, dilations=[1, 1, 1], groups=[1, 1, 1]),
            ), numCycles=1),
        )

    # ---------------------------------------------------------------------- #
    # ----------------------- Signal Encoding Methods ---------------------- #

    def compressionAlgorithm(self, processedData, stateMap, stateMapInd, numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, trainingFlag):
        # Prepare the data for signal reduction.
        # import matplotlib.pyplot as plt
        # plt.plot(processedData[0][1].detach().cpu(), label="initCompress")
        processedData = self.ladderOperator(processedData, stateMap[stateMapInd], numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, self.raisingModuleCompression, self.raisingModuleCompressionCombine, self.raisingParamsCompression)
        # plt.plot(processedData[0][1].detach().cpu(), label="ladderCompress")
        processedData = self.synthesizeChannelInfo(processedData, self.compressChannelsCNN_preprocessChannels, self.compressChannelsCNN_preprocessCombine)
        # plt.plot(processedData[0][1].detach().cpu(), label="synthCompress")

        # Learned downsampling via CNN network.
        # processedData, expandedData, unExpandedData = self.compressSignalsModel(processedData, trainingFlag)
        processedData = self.compressChannelsCNN(processedData)
        # plt.plot(processedData[0][1].detach().cpu(), label="afterCompress")
        expandedData, unExpandedData = None, None

        # Process the reduced data.
        nextNumActiveSignals = int(numActiveSignals/self.expansionFactor)
        processedData = self.synthesizeChannelInfo(processedData, self.compressChannelsCNN_postprocessChannels, self.compressChannelsCNN_postprocessCombine)
        # plt.plot(processedData[0][1].detach().cpu(), label="afterSynth")
        processedData = self.ladderOperator(processedData, stateMap[stateMapInd+1], nextNumActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, self.loweringModuleCompression, self.loweringModuleCompressionCombine, self.loweringParamsCompression)
        # plt.plot(processedData[0][1].detach().cpu(), label="afterLadder")
        # plt.legend()
        # plt.show()

        return processedData, expandedData, unExpandedData

    def expansionAlgorithm(self, processedData, stateMap, stateMapInd, numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals):
        import matplotlib.pyplot as plt
        # plt.plot(processedData[0][1].detach().cpu(), label="initExpand")

        # Prepare the data for signal expansion.
        processedData = self.ladderOperator(processedData, stateMap[stateMapInd], numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, self.raisingModuleExpansion, self.raisingModuleExpansionCombine, self.raisingParamsExpansion)
        # plt.plot(processedData[0][1].detach().cpu(), label="ladderExpand")
        processedData = self.synthesizeChannelInfo(processedData, self.expandChannelsCNN_preprocessChannels, self.expandChannelsCNN_preprocessCombine)
        # plt.plot(processedData[0][1].detach().cpu(), label="synthExpand")

        # Learned upsampling via CNN network.
        processedData = self.expandChannelsCNN(processedData)
        # plt.plot(processedData[0][1].detach().cpu(), label="Expanded")

        # Process the expanded data.
        nextNumActiveSignals = int(numActiveSignals*self.expansionFactor)
        processedData = self.synthesizeChannelInfo(processedData, self.expandChannelsCNN_postprocessChannels, self.expandChannelsCNN_postprocessCombine)
        # plt.plot(processedData[0][1].detach().cpu(), label="afterSynthExpand")
        processedData = self.ladderOperator(processedData, stateMap[stateMapInd+1], nextNumActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, self.loweringModuleExpansion, self.loweringModuleExpansionCombine, self.loweringParamsExpansion)
        # plt.plot(processedData[0][1].detach().cpu(), label="afterLadderExpand")
        # plt.legend()
        # plt.show()

        return processedData

    def compressSignalsModel(self, inputData, trainingFlag = False):
        # Reduce the dimensionality of the data by half.
        processedData = self.reduceSignalDimension(inputData)
        import matplotlib.pyplot as plt
        # plt.plot(processedData[0][1].detach().cpu(), label="reduceSignalDimension")

        expandedData = None
        if trainingFlag:
            # Reconstruct the original signal.
            expandedData = self.reconstructSignalDimension1(processedData)
            expandedData = nn.functional.interpolate(expandedData, size=inputData.size(2), mode='linear', align_corners=True, antialias=False)
            expandedData = self.reconstructSignalDimension2(expandedData)

        # Expand the data by two as you remove a signal.
        processedData = self.reduceSignals1(processedData)
        # plt.plot(processedData[0][1].detach().cpu(), label="reduceSignals1")
        processedData = nn.functional.interpolate(processedData, size=inputData.size(2), mode='linear', align_corners=True, antialias=False)
        # plt.plot(processedData[0][1].detach().cpu(), label="interp")
        processedData = self.reduceSignals2(processedData)
        # plt.plot(processedData[0][1].detach().cpu(), label="reduceSignals2")

        return processedData, expandedData, inputData

    def ladderOperator(self, inputData, currentStateMap, numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, learningModule, combinationModule, ladderParams):
        # Extract the dimension information.
        fullSignalBatchSize, encodingSize, sequenceLength = inputData.size()
        signalBatchSize = int(numActiveSignals/encodingSize)
        batchSize = int(fullSignalBatchSize / signalBatchSize)
        
        # Extract the information from the map
        numBackwardNodes = currentStateMap[:numActiveSignals, 1].view(signalBatchSize, encodingSize).expand(batchSize, signalBatchSize, encodingSize).contiguous().view(fullSignalBatchSize, encodingSize, 1)
        numForwardNodes = currentStateMap[:numActiveSignals, 0].view(signalBatchSize, encodingSize).expand(batchSize, signalBatchSize, encodingSize).contiguous().view(fullSignalBatchSize, encodingSize, 1)
        nodesInView = numBackwardNodes + numForwardNodes
        # Normalize the information
        compressionFactor = initialNumSignals / (finalNumSignals + initialNumSignals)
        numForwardNodesPercent = numForwardNodes / nodesInView
        nodesInViewPercent = nodesInView / totalNodesInMap
        
        # Learn how to scale the data given the state in the route.
        processedData = inputData * (ladderParams[0] +
                                     ladderParams[1] * numForwardNodesPercent + ladderParams[2] * compressionFactor + ladderParams[3] * nodesInViewPercent + 
                                     # Linear two terms
                                     ladderParams[4] * compressionFactor*numForwardNodesPercent + ladderParams[5] * nodesInViewPercent*compressionFactor + ladderParams[6] * nodesInViewPercent*numForwardNodesPercent +
                                     # Linear three terms
                                     ladderParams[7] * numForwardNodesPercent*compressionFactor*nodesInViewPercent)

        # Non-linear learning.
        processedData = learningModule(processedData)
        processedData = self.applyLearningConv(processedData, inputData, combinationModule)

        return processedData

    def synthesizeChannelInfo(self, inputData, learningModule, combinationModule):
        # Apply the module to the data.
        processedData = learningModule(inputData)

        # Learn how to integrate this information.
        processedData = self.applyLearningConv(processedData, inputData, combinationModule)

        return processedData

    @staticmethod
    def applyLearningConv(learnedData, inputData, combinationModule):
        # Concatenate the information as different channels
        processedData = torch.cat((inputData, learnedData), dim=1)

        # Non-linear learning.
        processedData = combinationModule(processedData) + inputData

        return processedData
    
    # ---------------------------------------------------------------------- #
    # -------------------- Learned Positional Encoding --------------------- #

    def positionalEncoding(self, inputData, learningModule, combinationModule, addingData = 1):
        # Set up the variables for signal encoding.
        batchSize, numSignals, signalDimension = inputData.size()
        positionEncodedData = inputData.clone()

        # Extract the size of the input parameter.
        bitInds = torch.arange(self.numEncodingStamps).to(positionEncodedData.device)
        signalInds = torch.arange(numSignals).to(positionEncodedData.device)

        # Generate the binary encoding of signalInds in a batched manner
        binary_encoding = signalInds[:, None].bitwise_and(2 ** bitInds).bool()
        # binary_encoding dim: numSignals, numEncodingStamps

        # If we are removing the encoding, learn how to remove the encoding.
        if addingData == -1: positionEncodedData = self.applyPositionalEncodingModule(positionEncodedData, learningModule, combinationModule)

        # For each stamp encoding
        for stampInd in range(self.numEncodingStamps):
            # Check each signal if it is using this specific encoding.
            usingStampEncoding = binary_encoding[:, stampInd:stampInd + 1]
            encodingVector = usingStampEncoding.float() * self.encodedStamp[stampInd][-signalDimension:]
            # encodingVector dim: numSignals, signalDimension

            # Add the stamp encoding to all the signals in all the batches.
            positionEncodedData = positionEncodedData + (stampInd % 2 == 0)*addingData*encodingVector.unsqueeze(0)/10

        # If we are encoding the data, learn how to apply the encoding.
        if addingData == 1: positionEncodedData = self.applyPositionalEncodingModule(positionEncodedData, learningModule, combinationModule)

        return positionEncodedData

    def applyPositionalEncodingModule(self, inputData, learningModule, combinationModule):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.size()

        # Apply the module to the data.
        processedData = self.encodingInterface(inputData, learningModule)

        # Reshape the data to process each signal separately.
        reshapedInputData = inputData.view(batchSize * numSignals, 1, signalDimension)
        processedData = processedData.view(batchSize * numSignals, 1, signalDimension)
        # Concatenate the information as different channels.
        processedData = torch.cat((reshapedInputData, processedData), dim=1)

        # Apply a CNN network.
        processedData = combinationModule(processedData)
        processedData = processedData.view(batchSize, numSignals, signalDimension) + inputData

        return processedData

    # ---------------------------------------------------------------------- #
    # ---------------------------- Loss Methods ---------------------------- #   

    def calculateEncodingLoss(self, originalData, encodedData, sectionStateMap, initSectionStateMap, totalNodesInMap, initialNumSignals, finalNumSignals):
        # originalData  encodedDecodedOriginalData
        #          \         /
        #          encodedData

        # Set up the variables for signal encoding.
        originalNumSignals = originalData.size(1)
        numEncodedSignals = encodedData.size(1)

        # If we are training, add noise to the final state to ensure continuity of the latent space.
        noisyEncodedData = self.dataInterface.addNoise(encodedData, trainingFlag=True, noiseSTD=0.001)
        expandedDataLoss = None

        # Calculate the number of active signals in each path.
        numActiveSignals = originalNumSignals - self.simulateSignalPath(originalNumSignals, numEncodedSignals)[1]

        # Reverse operation
        if numEncodedSignals < originalNumSignals:
            encodedDecodedOriginalData = self.expansionModel(noisyEncodedData, originalNumSignals, sectionStateMap, stateMapInd = 0, totalNodesInMap = totalNodesInMap, initialNumSignals=initialNumSignals, finalNumSignals=finalNumSignals)
        else:
            encodedDecodedOriginalData, expandedDataLoss = self.compressionModel(noisyEncodedData, originalNumSignals, sectionStateMap, stateMapInd = 0, totalNodesInMap = totalNodesInMap, initialNumSignals=initialNumSignals, finalNumSignals=finalNumSignals, trainingFlag=True)

        # Assert the integrity of the expansions/compressions.
        assert encodedDecodedOriginalData.size(1) == originalData.size(1)

        # Calculate the squared error loss for this layer of compression/expansion.
        squaredErrorLoss_forward = (originalData - encodedDecodedOriginalData)[:, :numActiveSignals, :].pow(2).mean(dim=2).mean(dim=1)
        print("\tSignal encoder reverse operation loss:", squaredErrorLoss_forward.mean().item(), expandedDataLoss.mean().item() if expandedDataLoss is not None else 0)

        # Compile all the loss information together into one value.
        # finalLoss = 0.5*squaredErrorLoss_forward + 0.1*squaredErrorLoss_Encoding + 0.4*(squaredErrorLoss_forward2 + squaredErrorLoss_forwardMid)
        finalLoss = squaredErrorLoss_forward
        if expandedDataLoss is not None: finalLoss = finalLoss + expandedDataLoss/5

        return finalLoss

    def updateLossValues(self, originalData, encodedData, signalEncodingLayerLoss, sectionStateMap, initSectionStateMap, totalNodesInMap, initialNumSignals, finalNumSignals):
        # Keep tracking of the loss through each loop.
        layerLoss = self.calculateEncodingLoss(originalData, encodedData, sectionStateMap, initSectionStateMap, totalNodesInMap, initialNumSignals, finalNumSignals)

        if layerLoss.mean() < 0.01:
            return signalEncodingLayerLoss
        else:
            return signalEncodingLayerLoss + layerLoss

    # ---------------------------------------------------------------------- #
    # -------------------------- Data Organization ------------------------- #

    def expansionModel(self, originalData, targetNumSignals, stateMap, stateMapInd, totalNodesInMap, initialNumSignals, finalNumSignals):
        # Unpair the signals with their neighbors.
        unpairedData, frozenData, numActiveSignals = self.unpairSignals(originalData, targetNumSignals)
        # activeData dimension: batchSize*numActiveSignals/numCompressedSignals, numCompressedSignals, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension

        # Increase the number of signals.
        expandedData = self.expansionAlgorithm(unpairedData, stateMap, stateMapInd, numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals)
        # expandedData dimension: batchSize*numSignalPairs, 2, signalDimension

        # Recompile the signals to their original dimension.
        signalData = self.recompileSignals(expandedData, frozenData)
        # signalData dimension: batchSize, 2*numSignalPairs + numFrozenSignals, signalDimension

        # Free up memory.
        freeMemory()

        return signalData

    def compressionModel(self, originalData, targetNumSignals, stateMap, stateMapInd, totalNodesInMap, initialNumSignals, finalNumSignals, trainingFlag = False):
        # Pair up the signals with their neighbors.
        pairedData, frozenData, numActiveSignals = self.pairSignals(originalData, targetNumSignals)
        # pairedData dimension: batchSize*numActiveSignals/numExpandedSignals, numExpandedSignals, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension
        
        # Reduce the number of signals.
        reducedPairedData, expandedData, unExpandedData = self.compressionAlgorithm(pairedData, stateMap, stateMapInd, numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, trainingFlag)
        # reducedPairedData dimension: batchSize*numSignalPairs, 1, signalDimension

        expandedDataLoss = None
        if trainingFlag and expandedData is not None:
            expandedData = self.recompileSignals(expandedData, frozenData)
            unExpandedData = self.recompileSignals(unExpandedData, frozenData)

            # Calculate the loss.
            expandedDataLoss = (expandedData - unExpandedData)[:, :numActiveSignals, :].pow(2).mean(dim=2).mean(dim=1)

        # Recompile the signals to their original dimension.
        signalData = self.recompileSignals(reducedPairedData, frozenData)
        # signalData dimension: batchSize, numSignalPairs + numFrozenSignals, signalDimension

        # Free up memory.
        freeMemory()

        return signalData, expandedDataLoss


# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class generalSignalEncoding(signalEncoderBase):
    def __init__(self, maxSequenceLength=64, numExpandedSignals=2, accelerator=None):
        super(generalSignalEncoding, self).__init__(maxSequenceLength, numExpandedSignals, accelerator)

    def forward(self, signalData, targetNumSignals=32, stateMap = None, totalNodesInMap = None, initialNumSignals = None, finalNumSignals = None, signalEncodingLayerLoss=None, calculateLoss=True):
        """ The shape of signalData: (batchSize, numSignals, compressedLength) """
        # Initialize first time parameters for signal encoding.
        if signalEncodingLayerLoss is None: signalEncodingLayerLoss = torch.zeros((signalData.size(0),), device=signalData.device)
        if stateMap is None: stateMap, totalNodesInMap = self.getStateMapfromRoot(signalData.size(1), targetNumSignals, signalData.device)
        if initialNumSignals is None: initialNumSignals = signalData.size(1); finalNumSignals = targetNumSignals
        expandedDataLoss = 0

        # Set up the variables for signal encoding.
        batchSize, numSignals, signalDimension = signalData.size()
        numSignalPath = [numSignals]  # Keep track of the signal's at each iteration.

        # Assert that we have the expected data format.
        assert self.numCompressedSignals <= targetNumSignals, f"At the minimum, we cannot go lower than compressed signal batch. You provided {targetNumSignals} signals."
        assert self.maxSequenceLength >= signalDimension, f"Can only process signals that are <= {self.maxSequenceLength}. You provided a sequence of length {signalDimension}"
        assert self.numCompressedSignals <= numSignals, f"We cannot compress or expand if we dont have at least the compressed signal batch. You provided {numSignals} signals."

        # ------------- Signal Compression/Expansion Algorithm ------------- #  
        
        stateMapInd = 0
        # While we have the incorrect number of signals.
        while targetNumSignals != signalData.size(1):
            compressedDataFlag = targetNumSignals < signalData.size(1)

            # Keep track of the initial state
            originalData = signalData.clone()

            # Compress the signals down to the targetNumSignals.
            if compressedDataFlag: signalData, expandedDataLoss = self.compressionModel(signalData, targetNumSignals, stateMap, stateMapInd, totalNodesInMap, initialNumSignals, finalNumSignals, trainingFlag=calculateLoss)

            # Expand the signals up to the targetNumSignals.
            else: signalData = self.expansionModel(signalData, targetNumSignals, stateMap, stateMapInd, totalNodesInMap, initialNumSignals, finalNumSignals)

            # Keep track of the error during each compression/expansion.
            if calculateLoss and compressedDataFlag and expandedDataLoss is not None: signalEncodingLayerLoss = signalEncodingLayerLoss + expandedDataLoss/5
            if calculateLoss and signalEncodingLayerLoss.mean() < 0.5: signalEncodingLayerLoss = self.updateLossValues(originalData, signalData, signalEncodingLayerLoss, [stateMap[stateMapInd+1], stateMap[stateMapInd]],
                                                                                                                       [stateMap[stateMapInd], stateMap[stateMapInd+1]], totalNodesInMap, initialNumSignals, finalNumSignals)
            
            # Keep track of the signal's at each iteration.
            numSignalPath.append(signalData.size(1))
            stateMapInd += 1
            
        # ------------------------------------------------------------------ # 

        # Assert the integrity of the expansion/compression.
        if numSignals != targetNumSignals:
            assert all(numSignalPath[i] <= numSignalPath[i + 1] for i in range(len(numSignalPath) - 1)) \
                   or all(numSignalPath[i] >= numSignalPath[i + 1] for i in
                          range(len(numSignalPath) - 1)), "List is not sorted up or down"

        # Remove the target signal from the path.
        numSignalPath.pop()

        return signalData, numSignalPath, signalEncodingLayerLoss

    def printParams(self, numSignals=50, maxSequenceLength=300):
        # generalSignalEncoding(numExpandedSignals = 3, maxSequenceLength = 300).to('cpu').printParams(numSignals = 100, maxSequenceLength = 300)
        t1 = time.time()
        summary(self, (numSignals, maxSequenceLength))
        t2 = time.time()
        print(t2 - t1)

        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
