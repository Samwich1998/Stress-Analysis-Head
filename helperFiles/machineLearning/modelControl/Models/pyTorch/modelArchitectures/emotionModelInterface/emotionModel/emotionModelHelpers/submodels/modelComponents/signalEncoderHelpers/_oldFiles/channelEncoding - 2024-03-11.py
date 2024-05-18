# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch

# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class channelEncoding(signalEncoderModules):

    def __init__(self, numCompressedSignals, numExpandedSignals, expansionFactor):
        super(channelEncoding, self).__init__()
        # General parameters
        self.numCompressedSignals = numCompressedSignals
        self.numExpandedSignals = numExpandedSignals
        self.expansionFactor = expansionFactor

        # Specify the ladder operators to integrate the learned information from the encoding location.
        self.raisingModuleExpansionCombine = self.combineModuleSimpleEncoding(inChannel=2 * self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.loweringModuleExpansionCombine = self.combineModuleSimpleEncoding(inChannel=2 * self.numExpandedSignals, outChannel=self.numExpandedSignals)
        # Specify the ladder operators to integrate the learned information from the encoding location.
        self.raisingModuleExpansion = self.learnModuleSimpleEncoding(inChannel=self.numCompressedSignals)
        self.loweringModuleExpansion = self.learnModuleSimpleEncoding(inChannel=self.numExpandedSignals)
        # Specify the ladder operators to account for dilation in expansion.
        self.loweringParamsExpansion = torch.nn.Parameter(torch.randn(8))
        self.raisingParamsExpansion = torch.nn.Parameter(torch.randn(8))

        # Specify the ladder operators to integrate the learned information from the encoding location.
        self.loweringModuleCompressionCombine = self.combineModuleSimpleEncoding(inChannel=2 * self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.raisingModuleCompressionCombine = self.combineModuleSimpleEncoding(inChannel=2 * self.numExpandedSignals, outChannel=self.numExpandedSignals)
        # Specify the ladder operators to integrate the learned information from the encoding location.
        self.loweringModuleCompression = self.learnModuleSimpleEncoding(inChannel=self.numCompressedSignals)
        self.raisingModuleCompression = self.learnModuleSimpleEncoding(inChannel=self.numExpandedSignals)
        # Specify the ladder operators to account for dilation in compression.
        self.loweringParamsCompression = torch.nn.Parameter(torch.randn(8))
        self.raisingParamsCompression = torch.nn.Parameter(torch.randn(8))

        # Learned compression via CNN.
        self.compressChannelsCNN_preprocessChannels = self.learnModuleEncoding(inChannel=self.numExpandedSignals)
        self.compressChannelsCNN_preprocessCombine = self.combineModuleEncoding(inChannel=2 * self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.compressChannelsCNN1 = self.encodeChannels(inChannel=2 * self.numExpandedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN2 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN3 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN4 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN5 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN6 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN7 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN8 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN9 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN10 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN11 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN12 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN13 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN14 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN15 = self.encodeChannels(inChannel=self.numExpandedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN_postprocessCombine = self.combineModuleSimpleEncoding(inChannel=2 * self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.compressChannelsCNN_postprocessChannels = self.learnModuleSimpleEncoding(inChannel=self.numCompressedSignals)
        self.compressCombineEnd = self.encodeChannels(inChannel=self.numCompressedSignals + self.numCompressedSignals, outChannel=self.numCompressedSignals)

        # Learned expansion via CNN.
        self.expandChannelsCNN_preprocessChannels = self.learnModuleEncoding(inChannel=self.numCompressedSignals)
        self.expandChannelsCNN_preprocessCombine = self.combineModuleEncoding(inChannel=2 * self.numCompressedSignals, outChannel=self.numCompressedSignals)
        self.expandChannelsCNN1 = self.encodeChannels(inChannel=2 * self.numCompressedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN2 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN3 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN4 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN5 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN6 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN7 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN8 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN9 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN10 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN11 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN12 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN13 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN14 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN15 = self.encodeChannels(inChannel=self.numCompressedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN_postprocessCombine = self.combineModuleSimpleEncoding(inChannel=2 * self.numExpandedSignals, outChannel=self.numExpandedSignals)
        self.expandChannelsCNN_postprocessChannels = self.learnModuleSimpleEncoding(inChannel=self.numExpandedSignals)
        self.expandCombineEnd = self.encodeChannels(inChannel=self.numExpandedSignals + self.numExpandedSignals, outChannel=self.numExpandedSignals)

    # ---------------------------------------------------------------------- #
    # ----------------------- Signal Encoding Methods ---------------------- #

    def compressionAlgorithm(self, inputData, stateMap, stateMapInd, numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals):
        # Prepare the data for signal reduction.
        # import matplotlib.pyplot as plt
        # plt.plot(inputData[0][1].detach().cpu(), label="initCompress")
        # processedData = self.ladderOperator(inputData, stateMap[stateMapInd], numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, self.raisingModuleCompression, self.raisingModuleCompressionCombine,
        #                                     self.raisingParamsCompression)
        # plt.plot(processedData[0][1].detach().cpu(), label="ladderCompress")
        processedData = self.synthesizeChannelInfo(inputData, self.compressChannelsCNN_preprocessChannels, self.compressChannelsCNN_preprocessCombine)
        # plt.plot(processedData[0][1].detach().cpu(), label="synthCompress")

        # Learned downsampling via CNN network.
        # self.compressChannelsCNN(processedData)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN1, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN2, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN3, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN4, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN5, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN6, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN7, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN8, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN9, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN10, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN11, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN12, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN13, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN14, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.compressChannelsCNN15, addInput = False)
        # plt.plot(processedData[0][1].detach().cpu(), label="afterCompress")

        # Process the reduced data.
        nextNumActiveSignals = int(numActiveSignals / self.expansionFactor)
        processedData = self.synthesizeChannelInfo(processedData, self.compressChannelsCNN_postprocessChannels, self.compressChannelsCNN_postprocessCombine)
        # plt.plot(processedData[0][1].detach().cpu(), label="afterSynth")
        # processedData = self.ladderOperator(processedData, stateMap[stateMapInd + 1], nextNumActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, self.loweringModuleCompression, self.loweringModuleCompressionCombine,
        #                                     self.loweringParamsCompression)
        # plt.plot(processedData[0][1].detach().cpu(), label="afterLadder")
        # plt.legend()
        # plt.show()

        # processedData = self.applyLearningConv(processedData, compressedData, self.compressCombineEnd, addInput = False)

        return processedData

    def expansionAlgorithm(self, inputData, stateMap, stateMapInd, numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals):
        # import matplotlib.pyplot as plt
        # plt.plot(inputData[0][1].detach().cpu(), label="initExpand")

        # Prepare the data for signal expansion.
        # processedData = self.ladderOperator(inputData, stateMap[stateMapInd], numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, self.raisingModuleExpansion, self.raisingModuleExpansionCombine,
        #                                     self.raisingParamsExpansion)
        # plt.plot(processedData[0][1].detach().cpu(), label="ladderExpand")
        processedData = self.synthesizeChannelInfo(inputData, self.expandChannelsCNN_preprocessChannels, self.expandChannelsCNN_preprocessCombine)
        # plt.plot(processedData[0][1].detach().cpu(), label="synthExpand")

        # Learned upsampling via CNN network.
        # processedData = self.expandChannelsCNN(processedData)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN1, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN2, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN3, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN4, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN5, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN6, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN7, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN8, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN9, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN10, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN11, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN12, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN13, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN14, addInput = False)
        processedData = self.applyLearningConv(processedData, inputData, self.expandChannelsCNN15, addInput = False)
        # plt.plot(processedData[0][1].detach().cpu(), label="Expanded")

        # Process the expanded data.
        nextNumActiveSignals = int(numActiveSignals * self.expansionFactor)
        processedData = self.synthesizeChannelInfo(processedData, self.expandChannelsCNN_postprocessChannels, self.expandChannelsCNN_postprocessCombine)
        # plt.plot(processedData[0][1].detach().cpu(), label="afterSynthExpand")
        # processedData = self.ladderOperator(processedData, stateMap[stateMapInd + 1], nextNumActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, self.loweringModuleExpansion, self.loweringModuleExpansionCombine,
        #                                     self.loweringParamsExpansion)
        # plt.plot(processedData[0][1].detach().cpu(), label="afterLadderExpand")
        # plt.legend()
        # plt.show()

        # processedData = self.applyLearningConv(processedData, expandedData, self.expandCombineEnd, addInput = False)

        return processedData

    def ladderOperator(self, inputData, currentStateMap, numActiveSignals, totalNodesInMap, initialNumSignals, finalNumSignals, learningModule, combinationModule, ladderParams):
        # Extract the dimension information.
        fullSignalBatchSize, encodingSize, sequenceLength = inputData.size()
        signalBatchSize = int(numActiveSignals / encodingSize)
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
                                     ladderParams[4] * compressionFactor * numForwardNodesPercent + ladderParams[5] * nodesInViewPercent * compressionFactor + ladderParams[6] * nodesInViewPercent * numForwardNodesPercent +
                                     # Linear three terms
                                     ladderParams[7] * numForwardNodesPercent * compressionFactor * nodesInViewPercent)

        # Non-linear learning.
        processedData = learningModule(processedData)
        processedData = self.applyLearningConv(processedData, inputData, combinationModule, addInput = True)

        return processedData

    def synthesizeChannelInfo(self, inputData, learningModule, combinationModule):
        # Apply the module to the data.
        processedData = learningModule(inputData)

        # Learn how to integrate this information.
        processedData = self.applyLearningConv(processedData, inputData, combinationModule, addInput = True)

        return processedData

    @staticmethod
    def applyLearningConv(learnedData, inputData, combinationModule, addInput = False):
        # Concatenate the information as different channels
        processedData = torch.cat((inputData, learnedData), dim=1)

        # Non-linear learning.
        if addInput: processedData = combinationModule(processedData) + inputData
        else: processedData = combinationModule(processedData)

        return processedData
