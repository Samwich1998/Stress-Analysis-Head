# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import time

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModel.universalModelHelpers import freeMemory

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/signalEncoderHelpers/")
import _signalEncoderModules

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class signalEncoderBase(_signalEncoderModules.signalEncoderModules):
    def __init__(self, signalDimension = 64, numExpandedSignals = 2, accelerator = None):
        super(signalEncoderBase, self).__init__(signalDimension, numExpandedSignals)  
        # General parameters.
        self.encodedStamp = nn.ParameterList() # A list of learnable parameters for learnable signal positions.
        self.accelerator = accelerator      # Hugging face model optimizations.
        self.numEncodingStamps = 10         # The number of binary bits in the encoding (010 = 2 signals; 3 encodings).

        # A list of modules to encode each signal.
        self.learnSignalPositions = nn.ModuleList()  # Use ModuleList to store child modules.
        self.unlearnSignalPositions = nn.ModuleList()  # Use ModuleList to store child modules.
        
        # For each encoding bit
        for stampInd in range(self.numEncodingStamps):
            # Assign a learnable parameter to the signal.
            self.encodedStamp.append(torch.nn.Parameter(torch.randn(signalDimension)))
            
            # Learn how to embed the positional information into the signals.
            self.learnSignalPositions.append(self.addPositionalInfoModule(inChannel = 1, channelIncrease = 8))
            self.unlearnSignalPositions.append(self.removePositionalInfoModule(inChannel = 1, channelIncrease = 8))
            
        # Learned compression via CNN.
        self.compressChannelsCNN_preprocessSignals = self.signalEncodingModule(inChannel = 1, channelIncrease = 8)
        self.compressChannelsCNN_preprocessChannels = self.channelEncodingModule(inChannel = self.numExpandedSignals, channelIncrease = 8)
        self.compressChannelsCNN = self.channelCombination(inChannel = self.numExpandedSignals, outChannel = self.numCompressedSignals, channelIncrease = 8)
        self.compressChannelsCNN_postprocessChannels = self.channelEncodingModule(inChannel = self.numCompressedSignals, channelIncrease = 8)
        self.compressChannelsCNN_postprocessSignals = self.signalEncodingModule(inChannel = 1, channelIncrease = 8)

        # Learned expansion via CNN.
        self.expandChannelsCNN_preprocessSignals = self.signalEncodingModule(inChannel = 1, channelIncrease = 8)
        self.expandChannelsCNN_preprocessChannels = self.channelEncodingModule(inChannel = self.numCompressedSignals, channelIncrease = 8)
        self.expandChannelsCNN = self.channelCombination(inChannel = self.numCompressedSignals, outChannel = self.numExpandedSignals, channelIncrease = 8)
        self.expandChannelsCNN_postprocessChannels = self.channelEncodingModule(inChannel = self.numExpandedSignals, channelIncrease = 8)
        self.expandChannelsCNN_postprocessSignals = self.signalEncodingModule(inChannel = 1, channelIncrease = 8)

        # Linear parameters to account for dilation.
        self.raisingParams = torch.nn.Parameter(torch.randn(3))
        self.loweringParams = torch.nn.Parameter(torch.randn(3))
        # Specify the ladder operators to account for dilation.
        self.raisingModule = self.ladderModules(inChannel = 1, channelIncrease = 8)
        self.loweringModule = self.ladderModules(inChannel = 1, channelIncrease = 8)
        
        # Map the initial signals into a common subspace.
        self.initialTransformation = self.minorSubspaceTransformationInitial(inChannel = 1, numMidChannels = 4)
        self.finalTransformation = self.minorSubspaceTransformationFinal(inChannel = 1, numMidChannels = 4)

    # ---------------------------------------------------------------------- #
    # ------------------- Machine Learning Architectures ------------------- #
    
    def addPositionalInfoModule(self, inChannel = 1, channelIncrease = 8): 
        # Calculate the number of intermediate channels.
        numExpandedChannels = int(channelIncrease*inChannel)
        
        return nn.Sequential( 
            # Convolution architecture: signal-specific feature engineering
            self.convolutionalThreeFilters_noActivationBlock(numChannels = [inChannel, numExpandedChannels, numExpandedChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),    
            
            self.restNet(module = nn.Sequential(
                    # Convolution architecture: channel expansion
                    self.convolutionalThreeFilters_noActivation(numChannels = [inChannel, 2*inChannel, 4*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    
                    # Convolution architecture: feature engineering
                    self.convolutionalThreeFilters_noActivationBlock(numChannels = [8*inChannel, 8*inChannel, 8*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    self.convolutionalThreeFilters_noActivationBlock(numChannels = [8*inChannel, 4*inChannel, 4*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    
                    # Convolution architecture: channel compression
                    self.convolutionalThreeFilters_noActivation(numChannels = [8*inChannel, 4*inChannel, 2*inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_noActivationBlock(numChannels = [inChannel, inChannel, inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
        )
    
    def removePositionalInfoModule(self, inChannel = 1, channelIncrease = 8): 
        # Calculate the number of intermediate channels.
        numExpandedChannels = int(channelIncrease*inChannel)
        
        return nn.Sequential( 
            # Convolution architecture: signal-specific feature engineering
            self.convolutionalThreeFilters_noActivationBlock(numChannels = [inChannel, numExpandedChannels, numExpandedChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),    
            
            self.restNet(module = nn.Sequential(
                    # Convolution architecture: channel expansion
                    self.convolutionalThreeFilters_noActivation(numChannels = [inChannel, 2*inChannel, 4*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    
                    # Convolution architecture: feature engineering
                    self.convolutionalThreeFilters_noActivationBlock(numChannels = [8*inChannel, 8*inChannel, 8*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    self.convolutionalThreeFilters_noActivationBlock(numChannels = [8*inChannel, 4*inChannel, 4*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    
                    # Convolution architecture: channel compression
                    self.convolutionalThreeFilters_noActivation(numChannels = [8*inChannel, 4*inChannel, 2*inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_noActivationBlock(numChannels = [inChannel, inChannel, inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
        )
    
    def ladderModules(self, inChannel = 1, channelIncrease = 8):     
        # Calculate the number of intermediate channels.
        numExpandedChannels = int(channelIncrease*inChannel)
        
        return nn.Sequential( 
            # Convolution architecture: signal-specific feature engineering
            self.convolutionalThreeFilters_noActivationBlock(numChannels = [inChannel, numExpandedChannels, numExpandedChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),    
            
            self.restNet(module = nn.Sequential(
                    # Convolution architecture: channel expansion
                    self.convolutionalThreeFilters_noActivation(numChannels = [inChannel, 2*inChannel, 4*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    
                    # Convolution architecture: feature engineering
                    self.convolutionalThreeFilters_noActivationBlock(numChannels = [8*inChannel, 8*inChannel, 8*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    self.convolutionalThreeFilters_noActivationBlock(numChannels = [8*inChannel, 4*inChannel, 4*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    
                    # Convolution architecture: channel compression
                    self.convolutionalThreeFilters_noActivation(numChannels = [8*inChannel, 4*inChannel, 2*inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_noActivationBlock(numChannels = [inChannel, inChannel, inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
        )

    def signalEncodingModule(self, inChannel = 1, channelIncrease = 8):
        # Calculate the number of intermediate channels.
        numInitChannelIncrease = int(channelIncrease*inChannel)

        return nn.Sequential(
            # Convolution architecture: signal-specific feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            
            self.restNet(module = nn.Sequential(
                    # Convolution architecture: channel expansion
                    self.convolutionalThreeFilters(numChannels = [inChannel, 2*inChannel, 4*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    
                    # Convolution architecture: feature engineering
                    self.convolutionalThreeFiltersBlock(numChannels = [8*inChannel, 8*inChannel, 8*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    self.convolutionalThreeFiltersBlock(numChannels = [8*inChannel, 4*inChannel, 4*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    
                    # Convolution architecture: channel compression
                    self.convolutionalThreeFilters(numChannels = [8*inChannel, 4*inChannel, 2*inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, inChannel, inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
        )
    
    def channelEncodingModule(self, inChannel = 1, channelIncrease = 8):
        # Calculate the number of intermediate channels.
        numInitChannelIncrease = int(channelIncrease*inChannel)

        return nn.Sequential(
            # Convolution architecture: signal-specific feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            
            self.restNet(module = nn.Sequential(
                    # Convolution architecture: channel expansion
                    self.convolutionalThreeFilters(numChannels = [inChannel, 2*inChannel, 4*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    
                    # Convolution architecture: feature engineering
                    self.convolutionalThreeFiltersBlock(numChannels = [8*inChannel, 8*inChannel, 8*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    self.convolutionalThreeFiltersBlock(numChannels = [8*inChannel, 4*inChannel, 4*inChannel, 8*inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                    
                    # Convolution architecture: channel compression
                    self.convolutionalThreeFilters(numChannels = [8*inChannel, 4*inChannel, 2*inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),

            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, inChannel, inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
        )
    
    def channelCombination(self, inChannel = 1, outChannel = 2, channelIncrease = 8):
        # Calculate the number of intermediate channels.
        numInitChannelIncrease = int(channelIncrease*inChannel)

        return nn.Sequential(
            # Convolution architecture: change channels
            self.convolutionalThreeFilters(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, outChannel], kernel_sizes = 3, dilations = 1, groups = 1),
        )
    
    def minorSubspaceTransformationInitial(self, inChannel = 1, numMidChannels = 8):        
        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = 3, dilations = 1, groups = 1),
        )
    
    def minorSubspaceTransformationFinal(self, inChannel = 1, numMidChannels = 8):        
        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = 3, dilations = 1, groups = 1),
        )
    
    # ---------------------------------------------------------------------- #
    # ----------------------- Signal Encoding Methods ---------------------- # 
    
    def updateCompressionMap(self, numActiveCompressionsMap, numFinalSignals):
        # Keep track of the compressions/expansions.
        numActiveCompressionsMap = numActiveCompressionsMap.sum(dim=1, keepdim=True) / numFinalSignals
        numActiveCompressionsMap = numActiveCompressionsMap.expand(numActiveCompressionsMap.size(0), numFinalSignals).contiguous()
        
        return numActiveCompressionsMap
    
    def expansionAlgorithm(self, compressedData, numActiveCompressionsMap, updatedActiveCompressionMap = None):   
        # Prepare the data for signal reduction.
        processedData = self.raisingOperator(compressedData, numActiveCompressionsMap)
        # processedData = compressedData

        # Preprocess signals
        processedData = self.encodingInterface(processedData, self.expandChannelsCNN_preprocessSignals)
        processedData = self.expandChannelsCNN_preprocessChannels(processedData)
        # Learned upsampling via CNN network.
        processedData = self.expandChannelsCNN(processedData)
        # Postprocess signals
        processedData = self.expandChannelsCNN_postprocessChannels(processedData)
        processedData = self.encodingInterface(processedData, self.expandChannelsCNN_postprocessSignals)
        
        # Keep track of the compressions/expansions.
        if updatedActiveCompressionMap == None:
            updatedActiveCompressionMap = self.updateCompressionMap(numActiveCompressionsMap, self.numExpandedSignals)
        
        # Process the reduced data.
        expandedData = self.loweringOperator(processedData, updatedActiveCompressionMap)
        # expandedData = processedData
                
        return expandedData, updatedActiveCompressionMap
    
    def compressionAlgorithm(self, expandedData, numActiveCompressionsMap, updatedActiveCompressionMap = None): 
        # Prepare the data for signal reduction.
        processedData = self.raisingOperator(expandedData, numActiveCompressionsMap)
        # processedData = expandedData
        
        # Preprocess signals
        processedData = self.encodingInterface(processedData, self.compressChannelsCNN_preprocessSignals)
        processedData = self.compressChannelsCNN_preprocessChannels(processedData)
        # Learned downsampling via CNN network.
        processedData = self.compressChannelsCNN(processedData)
        # Postprocess signals
        processedData = self.compressChannelsCNN_postprocessChannels(processedData)
        processedData = self.encodingInterface(processedData, self.compressChannelsCNN_postprocessSignals)
        
        # Keep track of the compressions/expansions.
        if updatedActiveCompressionMap == None:
            updatedActiveCompressionMap = self.updateCompressionMap(numActiveCompressionsMap, self.numCompressedSignals)

        # Process the reduced data.
        compressedData = self.loweringOperator(processedData, updatedActiveCompressionMap)
        # compressedData = processedData
        
        return compressedData, updatedActiveCompressionMap
    
    def raisingOperator(self, inputData, numCompressionsMap):
        # Extract the dimension information.
        signalBatchSize, encodingSize = numCompressionsMap.size()
        fullSignalBatchSize, encodingSize, sequenceLength = inputData.size()
        batchSize = int(fullSignalBatchSize / signalBatchSize)

        # Learn how to scale the data given the signal dilation.
        numFullCompressionsMap = numCompressionsMap.expand(batchSize, signalBatchSize, encodingSize).contiguous().view(fullSignalBatchSize, encodingSize, 1)
        processedData = inputData * (self.raisingParams[0] + self.raisingParams[1] * numFullCompressionsMap + self.raisingParams[2] * numFullCompressionsMap**2)
                        
        # Non-linear learning. 
        processedData = self.encodingInterface(processedData, self.raisingModule) + inputData
                
        return processedData

    def loweringOperator(self, inputData, numCompressionsMap):        
        # Extract the dimension information.
        signalBatchSize, encodingSize = numCompressionsMap.size()
        fullSignalBatchSize, encodingSize, sequenceLength = inputData.size()
        batchSize = int(fullSignalBatchSize / signalBatchSize)

        # Learn how to scale the data given the signal dilation.
        numFullCompressionsMap = numCompressionsMap.expand(batchSize, signalBatchSize, encodingSize).contiguous().view(fullSignalBatchSize, encodingSize, 1)
        processedData = inputData * (self.loweringParams[0] + self.loweringParams[1] * numFullCompressionsMap + self.loweringParams[2] * numFullCompressionsMap**2)
        
        # Non-linear learning.
        processedData = self.encodingInterface(processedData, self.loweringModule) + inputData
        
        return processedData
    
    # ---------------------------------------------------------------------- #
    # -------------------- Learned Positional Encoding --------------------- #

    def addPositionalEncoding(self, inputData):
        # Setup the variables for signal encoding.
        batchSize, numSignals, signalDimension = inputData.size()
        
        # Extract the size of the input parameter.
        bitInds = torch.arange(self.numEncodingStamps).to(inputData.device)
        signalInds = torch.arange(numSignals).to(inputData.device)
        
        # Generate the binary encoding of signalInds in a batched manner
        binary_encoding = signalInds[:, None].bitwise_and(2**bitInds).bool()
        # positionEncodedData dim: numSignals, numEncodingStamps
                    
        # For each stamp encoding
        for stampInd in range(self.numEncodingStamps):
            # Check each signal if it is using this specific encoding.
            usingStampEncoding = binary_encoding[:, stampInd:stampInd+1]
            encodingVector = usingStampEncoding.float() * self.encodedStamp[stampInd]
            # encodingVector dim: numSignals, signalDimension
            
            # Add the stamp encoding to all the signals in all the batches.
            encodedData = inputData + encodingVector.view(1, numSignals, self.signalDimension)
            # encodedData dim: batchSize, numSignals, signalDimension
            
            # Label the signals that have been stamped.
            stampedSignalsMask = usingStampEncoding.view(-1)
            numSignalsStamped = stampedSignalsMask.sum()
            # Prepare only the stamped signals for filtering.
            stampEncodingData = encodedData[:, stampedSignalsMask, :]
            stampEncodingData = stampEncodingData.view(batchSize*numSignalsStamped, 1, signalDimension)
            reshapedData = inputData[:, stampedSignalsMask, :].view(batchSize*numSignalsStamped, 1, signalDimension)
            
            # Apply a filter to the new encoded data.
            stampEncodingData =  reshapedData + self.learnSignalPositions[stampInd](stampEncodingData)
            # stampEncodingData dim: batchSize*numSignalsStamped, 1, signalDimension
            
            # Remake the original data with the stamped signals.
            stampEncodingData = stampEncodingData.view(batchSize, numSignalsStamped, signalDimension)
            inputData[:, stampedSignalsMask, :] = stampEncodingData
                    
        return inputData

    def removePositionalEncoding(self, inputData):
        # Setup the variables for signal encoding.
        batchSize, numSignals, signalDimension = inputData.size()
        
        # Extract the size of the input parameter.
        bitInds = torch.arange(self.numEncodingStamps).to(inputData.device)
        signalInds = torch.arange(numSignals).to(inputData.device)
        
        # Generate the binary encoding of signalInds in a batched manner
        binary_encoding = signalInds[:, None].bitwise_and(2**bitInds).bool()
        # positionEncodedData dim: numSignals, numEncodingStamps
                    
        # For each stamp encoding
        for stampInd in range(self.numEncodingStamps - 1, -1, -1):
            # Label the signals that have been stamped.
            usingStampEncoding = binary_encoding[:, stampInd:stampInd+1]
            stampedSignalsMask = usingStampEncoding.view(-1)
            numSignalsStamped = stampedSignalsMask.sum()
            # Prepare only the stamped signals for filtering.
            stampEncodingData = inputData[:, stampedSignalsMask, :]
            stampEncodingData = stampEncodingData.view(batchSize*numSignalsStamped, 1, signalDimension)
            
            # Apply a filter to the new encoded data.
            stampEncodingData = stampEncodingData + self.unlearnSignalPositions[stampInd](stampEncodingData)
            # inputData dim: batchSize*numSignalsStamped, 1, signalDimension
            
            # Remake the original data with the stamped signals.
            stampEncodingData = stampEncodingData.view(batchSize, numSignalsStamped, signalDimension)
            inputData[:, stampedSignalsMask, :] = stampEncodingData
                    
        return inputData
    
    # ---------------------------------------------------------------------- #
    # ---------------------------- Loss Methods ---------------------------- #   
    
    def calculateStandardizationLoss(self, inputData, expectedMean = 0, expectedStandardDeviation = 1, dim=-1):
        # Calculate the data statistics on the last dimension.
        standardDeviationData = inputData.std(dim=dim, keepdim=False)
        meanData = inputData.mean(dim=dim, keepdim=False)
        
        # Calculate the squared deviation from mean = 0; std = 1.
        standardDeviationError = (standardDeviationData - expectedStandardDeviation).pow(2).mean(dim=1)
        meanError = (meanData - expectedMean).pow(2).mean(dim=1)
                
        return meanError + standardDeviationError
    
    def calculateEncodingLoss(self, originalData, encodedData, numCompressionsMap, initialCompressionMap):
        # originalData  encodedDecodedOriginalData
        #          \         /
        #          encodedData
        
        # Setup the variables for signal encoding.
        originalNumSignals = originalData.size(1)
        numEncodedSignals = encodedData.size(1)
        
        # If we are training, add noise to the final state to ensure continuity of the latent space.
        noisyEncodedData = self.dataInterface.addNoise(encodedData, trainingFlag = True, noiseSTD = 0.005)
        
        # Reverse operation
        if numEncodedSignals < originalNumSignals:
            encodedDecodedOriginalData, _ = self.expansionModel(noisyEncodedData, originalNumSignals, numCompressionsMap, initialCompressionMap)
        else:            
            encodedDecodedOriginalData, _ = self.compressionModel(noisyEncodedData, originalNumSignals, numCompressionsMap, initialCompressionMap)
        # Assert the integrity of the expansions/compressions.
        assert encodedDecodedOriginalData.size(1) == originalData.size(1)
        
        # Calculate the number of active signals in each path.
        numActiveSignals = originalNumSignals - self.simulateNumFrozenSignals(originalNumSignals, numEncodedSignals)

        # Calculate the squared error loss of this layer of compression/expansion.
        squaredErrorLoss_forward = (originalData - encodedDecodedOriginalData)[:, :numActiveSignals, :].pow(2).mean(dim=-1).mean(dim=1)
        print("\tSignal encoder reverse operation loss:", squaredErrorLoss_forward.mean().item())
        
        # Compile all the loss information together into one value.
        return squaredErrorLoss_forward
    
    def updateLossValues(self, originalData, encodedData, numCompressionsMap, initialCompressionMap, signalEncodingLayerLoss):
        # It is a waste to go backward if we lost the initial signal.
        if 0.25 < signalEncodingLayerLoss.mean(): return signalEncodingLayerLoss
        
        # Keep tracking of the loss through each loop.
        layerLoss = self.calculateEncodingLoss(originalData, encodedData, numCompressionsMap, initialCompressionMap)
        
        return signalEncodingLayerLoss + layerLoss
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Data Organization ------------------------- #
    
    def expansionModel(self, originalData, targetNumSignals, numCompressionsMap, nextCompressionMap):
        # Unpair the signals with their neighbors.
        unpairedData, frozenData, numActiveSignals = self.unpairSignals(originalData, targetNumSignals)
        # activeData dimension: batchSize*numActiveSignals/numCompressedSignals, numCompressedSignals, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension
        
        # Keep track of the compressions/expansions.
        numActiveCompressionsMap, numFrozenCompressionsMap = self.segmentCompressionMap(numCompressionsMap, numActiveSignals, self.numCompressedSignals)
        updatedActiveCompressionMap, _ = self.segmentCompressionMap(nextCompressionMap, int(numActiveSignals*self.expansionFactor), self.numExpandedSignals)
        # numActiveCompressionsMap dimension: numActiveSignals/numCompressedSignals, numCompressedSignals
        # numFrozenCompressionsMap dimension: numFrozenSignals
                  
        # Increase the number of signals.
        expandedData, numActiveCompressionsMap = self.expansionAlgorithm(unpairedData, numActiveCompressionsMap, updatedActiveCompressionMap)
        # expandedData dimension: batchSize*numSignalPairs, 2, signalDimension
        
        # Keep track of the compressions/expansions.
        if updatedActiveCompressionMap == None: 
            nextCompressionMap = self.recombineCompressionMap(numActiveCompressionsMap, numFrozenCompressionsMap)

        # Recompile the signals to their original dimension.
        signalData = self.recompileSignals(expandedData, frozenData)
        # signalData dimension: batchSize, 2*numSignalPairs + numFrozenSignals, signalDimension
        
        # Free up memory.
        freeMemory()
        
        return signalData, nextCompressionMap
    
    def compressionModel(self, originalData, targetNumSignals, numCompressionsMap, nextCompressionMap):
        # Pair up the signals with their neighbors.
        pairedData, frozenData, numActiveSignals = self.pairSignals(originalData, targetNumSignals)
        # pairedData dimension: batchSize*numActiveSignals/numExpandedSignals, numExpandedSignals, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension
        
        # Keep track of the compressions/expansions.
        numActiveCompressionsMap, numFrozenCompressionsMap = self.segmentCompressionMap(numCompressionsMap, numActiveSignals, self.numExpandedSignals)
        updatedActiveCompressionMap, _ = self.segmentCompressionMap(nextCompressionMap, int(numActiveSignals/self.expansionFactor), self.numCompressedSignals)
        # numActiveCompressionsMap dimension: numActiveSignals/numExpandedSignals, numExpandedSignals
        # numFrozenCompressionsMap dimension: numFrozenSignals
        
        # Reduce the number of signals.
        reducedPairedData, numActiveCompressionsMap = self.compressionAlgorithm(pairedData, numActiveCompressionsMap, updatedActiveCompressionMap)
        # reducedPairedData dimension: batchSize*numSignalPairs, 1, signalDimension
        
        # Keep track of the compressions/expansions.
        if updatedActiveCompressionMap == None: 
            nextCompressionMap = self.recombineCompressionMap(numActiveCompressionsMap, numFrozenCompressionsMap)
            
        # Recompile the signals to their original dimension.
        signalData = self.recompileSignals(reducedPairedData, frozenData)
        # signalData dimension: batchSize, numSignalPairs + numFrozenSignals, signalDimension
        
        # Free up memory.
        freeMemory()
                
        return signalData, nextCompressionMap
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numExpandedSignals = 2, accelerator = None):
        super(signalEncoding, self).__init__(signalDimension, numExpandedSignals, accelerator) 
                        
    def forward(self, signalData, targetNumSignals = 32, numCompressionsMap = None, nextCompressionMap = None, signalEncodingLayerLoss = None, calculateLoss = True):
        """ The shape of signalData: (batchSize, numSignals, compressedLength) """
        # Initialize first time parameters for signal encoding.
        if numCompressionsMap == None: numCompressionsMap = torch.ones((signalData.size(1),), device=signalData.device)
        if signalEncodingLayerLoss == None: signalEncodingLayerLoss = torch.zeros((signalData.size(0),), device=signalData.device)
        
        # Setup the variables for signal encoding.
        batchSize, numSignals, signalDimension = signalData.size()
        numCompressionsPath = [numCompressionsMap.clone()] # Keep track of the compression amount at each iteration.
        numSignalPath = [numSignals] # Keep track of the signal's at each iteration.

        # Assert that we have the expected data format.
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."
        assert self.numCompressedSignals <= targetNumSignals, f"At the minimum, we cannot go lower than compressed signal batch. You provided {targetNumSignals} signals."
        assert self.numCompressedSignals <= numSignals, f"We cannot compress or expand if we dont have at least the compressed signal batch. You provided {numSignals} signals."
        
        # ------------- Signal Compression/Expansion Algorithm ------------- #  
        
        if targetNumSignals == signalData.size(1): print("Issue", targetNumSignals, signalData.size()); sys.exit()
        
        # While we have the incorrect number of signals.
        while targetNumSignals != signalData.size(1):
            compressedDataFlag = targetNumSignals < signalData.size(1)
            
            # Keep track of the initial state
            initialCompressionMap = numCompressionsMap.clone()
            originalData = signalData.clone()
                                    
            # Compress the signals down to the targetNumSignals.
            if compressedDataFlag: signalData, numCompressionsMap = self.compressionModel(signalData, targetNumSignals, numCompressionsMap, nextCompressionMap)
            
            # Expand the signals up to the targetNumSignals.
            else: signalData, numCompressionsMap = self.expansionModel(signalData, targetNumSignals, numCompressionsMap, nextCompressionMap)

            # Keep track of the error during each compression/expansion.
            if calculateLoss: signalEncodingLayerLoss = self.updateLossValues(originalData, signalData, numCompressionsMap, initialCompressionMap, signalEncodingLayerLoss)
        
            # Keep track of the signal's at each iteration.
            numCompressionsPath.append(numCompressionsMap.clone())
            numSignalPath.append(signalData.size(1))
                    
        # ------------------------------------------------------------------ # 
        
        # Assert the integrity of the expansion/compression.
        if numSignals != targetNumSignals:
            assert all(numSignalPath[i] <= numSignalPath[i + 1] for i in range(len(numSignalPath) - 1)) \
                or all(numSignalPath[i] >= numSignalPath[i + 1] for i in range(len(numSignalPath) - 1)), "List is not sorted up or down"
        
        # Remove the target signal from the path.
        numSignalPath.pop()
        
        return signalData, numSignalPath, numCompressionsPath, signalEncodingLayerLoss
        
    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64, numExpandedSignals = 3).to('cpu').printParams(numSignals = 4)
        t1 = time.time()
        summary(self, (numSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
