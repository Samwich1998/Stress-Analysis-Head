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

        # Map the initial signals into a common subspace.
        self.initialTransformation = self.minorSubspaceTransformationInitial(inChannel = 1, numMidChannels = 4)
        self.finalTransformation = self.minorSubspaceTransformationFinal(inChannel = 1, numMidChannels = 4)
        
        # Learn how to embed the positional information into the signals.
        self.learnSignalPositions = self.addPositionalInfoModule(inChannel = self.numEncodingStamps + 1, outChannel = 1, channelIncrease = 1)
        self.unlearnSignalPositions = self.removePositionalInfoModule(inChannel = 1, outChannel = 1, channelIncrease = 32)

        # For each encoding bit
        for stampInd in range(self.numEncodingStamps):
            # Assign a learnable parameter to the signal.
            self.encodedStamp.append(torch.nn.Parameter(torch.randn(signalDimension)))
            
        # Learned compression via CNN.
        self.compressChannelsCNN_preprocess = self.channelEncodingModule_preprocess(inChannel = 1, outChannel = 1, channelIncrease = 8)
        self.compressChannelsCNN = self.channelEncodingModule(inChannel = self.numExpandedSignals, outChannel = self.numCompressedSignals, channelIncrease = 8)
        self.compressChannelsCNN_postprocess = self.channelEncodingModule_postprocess(inChannel = 1, outChannel = 1, channelIncrease = 8)
        # Learned expansion via CNN.
        self.expandChannelsCNN_preprocess = self.channelEncodingModule_preprocess(inChannel = 1, outChannel = 1, channelIncrease = 4)
        self.expandChannelsCNN = self.channelEncodingModule(inChannel = self.numCompressedSignals, outChannel = self.numExpandedSignals, channelIncrease = 4)
        self.expandChannelsCNN_postprocess = self.channelEncodingModule_postprocess(inChannel = 1, outChannel = 1, channelIncrease = 4)
        
        # Linear parameters to account for dilation.
        self.raisingParams = torch.nn.Parameter(torch.randn(3))
        self.loweringParams = torch.nn.Parameter(torch.randn(3))
        # Specify the ladder operators to account for dilation.
        self.raisingModule = self.ladderModules(inChannel = 1, channelIncrease = 4)
        self.loweringModule = self.ladderModules(inChannel = 1, channelIncrease = 4)

    # ---------------------------------------------------------------------- #
    # ------------------- Machine Learning Architectures ------------------- #
    
    def addPositionalInfoModule(self, inChannel = 2, outChannel = 1, channelIncrease = 8): 
        numChannelIncrease = int(channelIncrease*inChannel)
        numChannelOutcrease = int(4*outChannel)

        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters(numChannels = [inChannel, numChannelIncrease, numChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters(numChannels = [inChannel, numChannelIncrease, numChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters(numChannels = [inChannel, numChannelIncrease, numChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            
            # Convolution architecture: change channels
            self.convolutionalOneFilters(numChannels = [inChannel, outChannel], kernel_size = 3, dilation = 1, group = 1),
            
            self.convolutionalThreeFilters(numChannels = [inChannel, numChannelOutcrease, numChannelOutcrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters(numChannels = [inChannel, numChannelOutcrease, numChannelOutcrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters(numChannels = [inChannel, numChannelOutcrease, numChannelOutcrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
        )
    
    def removePositionalInfoModule(self, inChannel = 1, outChannel = 1, channelIncrease = 8): 
        numChannelIncrease = int(channelIncrease*inChannel)

        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numChannelIncrease, numChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numChannelIncrease, numChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numChannelIncrease, numChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numChannelIncrease, numChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numChannelIncrease, numChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
        )
    
    def ladderModules(self, inChannel = 1, channelIncrease = 8): 
        numChannelIncrease = int(channelIncrease*inChannel)
    
        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters(numChannels = [inChannel, numChannelIncrease, numChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters(numChannels = [inChannel, numChannelIncrease, numChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
        )
    
    def channelEncodingModule_preprocess(self, inChannel = 1, outChannel = 2, channelIncrease = 8):
        # Calculate the number of intermediate channels.
        numInitChannelIncrease = int(channelIncrease*inChannel)

        return nn.Sequential(
            # Convolution architecture: shared feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
        )
    
    def channelEncodingModule_postprocess(self, inChannel = 1, outChannel = 2, channelIncrease = 8):
        # Calculate the number of intermediate channels.
        numInitChannelIncrease = int(channelIncrease*inChannel)

        return nn.Sequential(
            # Convolution architecture: shared feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
        )
    
    def channelEncodingModule(self, inChannel = 1, outChannel = 2, channelIncrease = 8):
        # Calculate the number of intermediate channels.
        numInitChannelIncrease = int(channelIncrease*inChannel)
        numChannelOutcrease = int(channelIncrease*outChannel)

        return nn.Sequential(
            # Convolution architecture: shared feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),

            # Convolution architecture: change channels
            self.convolutionalOneFilters(numChannels = [inChannel, outChannel], kernel_size = 3, dilation = 1, group = 1),
            
            # Convolution architecture: shared feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelOutcrease, numInitChannelIncrease, outChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelOutcrease, numInitChannelIncrease, outChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelOutcrease, numInitChannelIncrease, outChannel], kernel_sizes = 3, dilations = 1, groups = [1, 1, 1]),
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

        # Learned downsampling via CNN network.
        processedData = self.encodingInterface(processedData, self.expandChannelsCNN_preprocess)
        processedData = self.expandChannelsCNN(processedData)
        processedData = self.encodingInterface(processedData, self.expandChannelsCNN_postprocess)
        
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
        
        # Learned upsampling via CNN network.
        processedData = self.encodingInterface(processedData, self.compressChannelsCNN_preprocess)
        processedData = self.compressChannelsCNN(processedData)
        processedData = self.encodingInterface(processedData, self.compressChannelsCNN_postprocess)
        
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

    def getEncodingStamp(self, numSignals, batchSize, device):
        # Extract the size of the input parameter.
        bitInds = torch.arange(self.numEncodingStamps).to(device)
        signalInds = torch.arange(numSignals).to(device)

        # Setup the parameters for the encoding.
        stampEncodings = torch.zeros(batchSize, numSignals, self.numEncodingStamps, self.signalDimension, device=device)
        
        # Generate the binary encoding of signalInds in a batched manner
        binary_encoding = signalInds[:, None].bitwise_and(2**bitInds).bool()
        # positionEncodedData dim: numSignals, numEncodingStamps
                    
        # For each stamp encoding
        for stampInd in range(self.numEncodingStamps):
            # Find the encoding vector for the signal.
            usingStampInd = binary_encoding[:, stampInd:stampInd+1].float()
            encodingVector = (usingStampInd * self.encodedStamp[stampInd].expand(numSignals, self.signalDimension))
            # encodingVector dim: numSignals, signalDimension

            encodingVector = encodingVector.view(1, numSignals, 1, self.signalDimension)
            stampEncodings[:, :, stampInd:stampInd+1, :] = stampEncodings[:, :, stampInd:stampInd+1, :] + encodingVector.expand(batchSize, numSignals, 1, self.signalDimension)
            # stampEncodings dim: batchSize, numSignals, self.numEncodingStamps, self.signalDimension

        return stampEncodings.contiguous()
    
    def applyEncodingStamps(self, inputData):
        # Setup the variables for signal encoding.
        batchSize, numSignals, signalDimension = inputData.size()
        
        # Calculate the positional encodings of the signals.
        positionalEncoding = self.getEncodingStamp(numSignals, batchSize, inputData.device)
        # positionalEncoding dim: batchSize, numSignals, self.numEncodingStamps, self.signalDimension
        
        # Move all the signals into their own batches.
        positionalEncoding = positionalEncoding.view(batchSize*numSignals, self.numEncodingStamps, signalDimension)
        inputData = inputData.view(batchSize*numSignals, 1, signalDimension)
        # Add the positional encoding as a seperate channel.
        positionEncodedData = torch.cat((inputData, positionalEncoding), dim=1)
        # positionEncodedData dim: batchSize*numSignals, numEncodingStamps + 1, signalDimension

        # Map the positional encodings onto the data.
        positionEncodedData = inputData + self.learnSignalPositions(positionEncodedData)
        # positionEncodedData dim: batchSize*numSignals, 1, signalDimension
        
        # Move all the signals back into their respective batches.
        positionEncodedData = positionEncodedData.view(batchSize, numSignals, signalDimension)
        # positionEncodedData dim: batchSize, numSignals, signalDimension
   
        return positionEncodedData
    
    def removeEncodingStamps(self, positionEncodedData):
        # Map the positional encodings onto the data.
        originalData = self.encodingInterface(positionEncodedData, self.unlearnSignalPositions)
        # positionEncodedData dim: batchSize, numSignals, signalDimension

        return originalData
    
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
        noisyEncodedData = self.dataInterface.addNoise(encodedData, trainingFlag = True, noiseSTD = 0.05)
        
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
        if 0.1 < signalEncodingLayerLoss.mean(): return signalEncodingLayerLoss
        
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
        
