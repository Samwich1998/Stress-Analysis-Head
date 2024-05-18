# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import gc
import sys
import time

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/signalEncoderHelpers/")
import _signalEncoderModules

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import _convolutionalHelpers

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class signalEncoderBase(_signalEncoderModules.signalEncoderModules):
    def __init__(self, signalDimension = 64, numExpandedSignals = 2):
        super(signalEncoderBase, self).__init__(signalDimension, numExpandedSignals)        

        # Map the initial signals into a common subspace.
        self.initialTransformation = self.minorSubspaceTransformationInitial(inChannel = 1, numMidChannels = 8)
        self.finalTransformation = self.minorSubspaceTransformationFinal(inChannel = 1, numMidChannels = 8)
        # self.initMidTransformation = self.minorSubspaceTransformationFinal(inChannel = 1, numMidChannels = 16)
        # self.finalMidTransformation = self.minorSubspaceTransformationFinal(inChannel = 1, numMidChannels = 16)
        
        # Learned expansion/compression via CNN.
        self.compressChannelsCNN = self.signalEncodingModule_compression(inChannel = self.numExpandedSignals, outChannel = self.numCompressedSignals, channelIncrease = 16)
        self.expandChannelsCNN = self.signalEncodingModule_expansion(inChannel = self.numCompressedSignals, outChannel = self.numExpandedSignals, channelIncrease = 16)  

        # Initialize the pooling layers to upsample/downsample
        # self.downsampleChannels = nn.Upsample(size=self.numCompressedSignals, mode='linear', align_corners=True)
        # self.upsampleChannels = nn.Upsample(size=self.numExpandedSignals, mode='linear', align_corners=True)
        # Initialize method for learned upsampling/downsampling.
        # self.learnedUpsampling = self.semiLearnedChangeChannels(inChannel = self.numCompressedSignals, outChannel = self.numExpandedSignals)
        # self.learnedDownsampling = self.semiLearnedChangeChannels(inChannel = self.numExpandedSignals, outChannel = self.numCompressedSignals)

    def expansionAlgorithm(self, compressedData):   
        # Learned downsampling via CNN network.
        expandedData = self.expandChannelsCNN(compressedData)
                
        return expandedData
    
    def compressionAlgorithm(self, expandedData): 
        # Learned upsampling via CNN network.
        compressedData = self.compressChannelsCNN(expandedData)
        
        return compressedData
    
    # ---------------------------------------------------------------------- #
    # ------------------- Machine Learning Architectures ------------------- #
    
    def signalEncodingModule_compression(self, inChannel = 1, outChannel = 2, channelIncrease = 8):
        # Calculate the number of intermediate channels.
        numInitChannelIncrease = int(channelIncrease*inChannel)
        numChannelIncrease = int(channelIncrease*outChannel)

        return nn.Sequential(_convolutionalHelpers.addModules(
            firstModule = nn.Sequential(
                # Convolution architecture: signal-specific feature engineering
                self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
                self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
                self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
                self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),

                # Convolution architecture: shared feature engineering
                self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

                # Convolution architecture: change channels
                self.convolutionalThreeFilters(numChannels = [inChannel, inChannel, outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            ),
            
            secondModule = nn.Sequential(
                # Convolution architecture: change channels
                self.pointwiseConv(numChannels = [inChannel, outChannel]),
            ),
            
            scalingFactor = 1,),
            
            # Convolution architecture: shared feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelIncrease, numChannelIncrease, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelIncrease, numChannelIncrease, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelIncrease, numChannelIncrease, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

            # Convolution architecture: signal-specific feature engineering
            self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
            self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
            self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
            self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
        )
    
    def signalEncodingModule_expansion(self, inChannel = 2, outChannel = 1, channelIncrease = 8):
        # Calculate the number of intermediate channels.
        numInitChannelIncrease = int(channelIncrease*inChannel)
        numChannelIncrease = int(channelIncrease*outChannel)

        return nn.Sequential(_convolutionalHelpers.addModules(
            firstModule = nn.Sequential(
                # Convolution architecture: signal-specific feature engineering
                self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
                self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
                self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
                self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),

                # Convolution architecture: shared feature engineering
                self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

                # Convolution architecture: change channels
                self.convolutionalThreeFilters(numChannels = [inChannel, inChannel, outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            ),
            
            secondModule = nn.Sequential(
                # Convolution architecture: change channels
                self.pointwiseConv(numChannels = [inChannel, outChannel]),
            ),
            
            scalingFactor = 1,),
            
            # Convolution architecture: shared feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelIncrease, numChannelIncrease, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelIncrease, numChannelIncrease, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelIncrease, numChannelIncrease, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

            # Convolution architecture: signal-specific feature engineering
            self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
            self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
            self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
            self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
        )

    # def signalEncodingModule_compression(self, inChannel = 2, outChannel = 1, channelIncrease = 8):
    #     # Calculate the number of intermediate channels.
    #     numInitChannelIncrease = int(channelIncrease*inChannel)
    #     numChannelIncrease = int(channelIncrease*outChannel)

    #     return nn.Sequential(_convolutionalHelpers.addModules(
    #         firstModule = nn.Sequential(
    #             # Convolution architecture: shared feature engineering
    #             self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
    #             self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

    #             # Convolution architecture: change channels
    #             self.convolutionalThreeFilters(numChannels = [inChannel, inChannel, outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
    #         ),
            
    #         secondModule = nn.Sequential(
    #             # Convolution architecture: signal-specific feature engineering
    #             self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
    #             self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),

    #             # Convolution architecture: channel-specific feature engineering
    #             self.pointwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel]),
    #             self.pointwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel]),
    
    #             self.pointwiseConv(numChannels = [inChannel, outChannel]),
    #         ),
            
    #         scalingFactor = 1,),
            
    #         # Convolution architecture: channel-specific feature engineering
    #         self.pointwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel]),
    #         self.pointwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel]),

    #         # Convolution architecture: shared feature engineering
    #         self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelIncrease, numChannelIncrease, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
    #         self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelIncrease, numChannelIncrease, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

    #         # Convolution architecture: signal-specific feature engineering
    #         self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
    #         self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
    #     )
    
    # def signalEncodingModule_expansion(self, inChannel = 2, outChannel = 1, channelIncrease = 8):
    #     # Calculate the number of intermediate channels.
    #     numInitChannelIncrease = int(channelIncrease*inChannel)
    #     numChannelIncrease = int(channelIncrease*outChannel)

    #     return nn.Sequential(_convolutionalHelpers.addModules(
    #         firstModule = nn.Sequential(
    #             # Convolution architecture: shared feature engineering
    #             self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
    #             self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numInitChannelIncrease, numInitChannelIncrease, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

    #             # Convolution architecture: change channels
    #             self.convolutionalThreeFilters(numChannels = [inChannel, inChannel, outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
    #         ),
            
    #         secondModule = nn.Sequential(
    #             # Convolution architecture: signal-specific feature engineering
    #             self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
    #             self.depthwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),

    #             # Convolution architecture: channel-specific feature engineering
    #             self.pointwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel]),
    #             self.pointwiseConvBlock_resNetBlock(numChannels = [inChannel, inChannel]),
    
    #             self.pointwiseConv(numChannels = [inChannel, outChannel]),
    #         ),
            
    #         scalingFactor = 1,),
            
    #         # Convolution architecture: channel-specific feature engineering
    #         self.pointwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel]),
    #         self.pointwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel]),

    #         # Convolution architecture: shared feature engineering
    #         self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelIncrease, numChannelIncrease, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
    #         self.convolutionalThreeFilters_resNetBlock(numChannels = [outChannel, numChannelIncrease, numChannelIncrease, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

    #         # Convolution architecture: signal-specific feature engineering
    #         self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
    #         self.depthwiseConvBlock_resNetBlock(numChannels = [outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1]),
    #     )
      
    def minorSubspaceTransformationInitial(self, inChannel = 1, numMidChannels = 8):        
        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
        )
    
    def minorSubspaceTransformationFinal(self, inChannel = 1, numMidChannels = 8):        
        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
        )
    
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
    
    def calculateEncodingLoss(self, originalData, encodedData, numSignalPath, compressedDataFlag):
        #          decodedData
        #    
        #          /         \
        #                decodedEncodedOriginalData
        # originalData          
        #                encodedDecodedOriginalData
        #          \         /
        #                                            
        #          encodedData
        
        # Setup the variables for signal encoding.
        originalNumSignals = originalData.size(1)
        numEncodedSignals = encodedData.size(1)
        
        # If we compressed the data
        if compressedDataFlag: 
            # Setup the variables for signal encoding.
            decodedNumSignals = self.getMaxSignals_Expansion(originalNumSignals)

            # Traveling along the top of the pyramid.
            _, decodedData = self.expansionModel(originalData, decodedNumSignals)
            _, decodedEncodedOriginalData = self.compressionModel(decodedData, originalNumSignals)

            # # Traveling along the bottom of the pyramid.
            _, encodedDecodedOriginalData = self.expansionModel(encodedData, originalNumSignals)
        else:
            # Setup the variables for signal encoding.
            decodedNumSignals = self.getMaxSignals_Compression(originalNumSignals)

            # Traveling along the top-middle of the pyramid.
            _, decodedData = self.compressionModel(originalData, decodedNumSignals)
            _, decodedEncodedOriginalData = self.expansionModel(decodedData, originalNumSignals)

            # # Traveling along the bottom-middle of the pyramid.
            _, encodedDecodedOriginalData = self.compressionModel(encodedData, originalNumSignals)
        # Assert the integrity of the expansions/compressions.
        assert decodedEncodedOriginalData.size(1) == encodedDecodedOriginalData.size(1)
        assert decodedEncodedOriginalData.size(1) == originalData.size(1)
        assert decodedData.size(1) == decodedNumSignals
        
        # Calculate the number of active signals in each path.
        numActiveSignals_toEncoding = originalNumSignals - self.simulateNumFrozenSignals(originalNumSignals, numEncodedSignals)
        numActiveSignals_toDecoding = originalNumSignals - self.simulateNumFrozenSignals(originalNumSignals, decodedNumSignals)
        maxActiveSignals = min(numActiveSignals_toEncoding, numActiveSignals_toDecoding)

        # Calculate the squared error loss of this layer of compression/expansion.
        squaredErrorLoss_middle = (decodedEncodedOriginalData - encodedDecodedOriginalData)[:, :maxActiveSignals, :].pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_forward = (originalData - encodedDecodedOriginalData)[:, :numActiveSignals_toEncoding, :].pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_backward = (originalData - decodedEncodedOriginalData)[:, :numActiveSignals_toDecoding, :].pow(2).mean(dim=-1).mean(dim=1)
        print(squaredErrorLoss_forward.mean().item(), squaredErrorLoss_middle.mean().item(), squaredErrorLoss_backward.mean().item())
        
        # Compile all the loss information together into one value.
        return squaredErrorLoss_forward + squaredErrorLoss_middle + squaredErrorLoss_backward
    
    def updateLossValues(self, originalData, encodedData, numSignalPath, compressedDataFlag, signalEncodingLayerLoss, signalEncodingLayer):
        # Keep tracking of the loss through each loop.
        layerLoss = self.calculateEncodingLoss(originalData, encodedData, numSignalPath, compressedDataFlag)

        if signalEncodingLayer == 1:
            signalEncodingLayerLoss = signalEncodingLayerLoss + 5*layerLoss
        else:
            signalEncodingLayerLoss = signalEncodingLayerLoss + layerLoss
        
        return signalEncodingLayerLoss
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Data Organization ------------------------- #
    
    def expansionModel(self, originalData, targetNumSignals):
        # Unpair the signals with their neighbors.
        unpairedData, frozenData, numActiveSignals = self.unpairSignals(originalData, targetNumSignals)
        # activeData dimension: batchSize*numSignalPairs, 1, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension
                    
        # Increase the number of signals.
        expandedData = self.expansionAlgorithm(unpairedData)
        # expandedData dimension: batchSize*numSignalPairs, 2, signalDimension
        
        # Recompile the signals to their original dimension.
        signalData = self.recompileSignals(expandedData, frozenData)
        # signalData dimension: batchSize, 2*numSignalPairs + numFrozenSignals, signalDimension
        
        # Free up memory.
        gc.collect(); torch.cuda.empty_cache();
        
        return originalData, signalData
    
    def compressionModel(self, originalData, targetNumSignals):
        # Pair up the signals with their neighbors.
        pairedData, frozenData, numActiveSignals = self.pairSignals(originalData, targetNumSignals)
        # pairedData dimension: batchSize*numSignalPairs, 2, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension
        
        # Reduce the number of signals.
        reducedPairedData = self.compressionAlgorithm(pairedData)
        # reducedPairedData dimension: batchSize*numSignalPairs, 1, signalDimension

        # Recompile the signals to their original dimension.
        signalData = self.recompileSignals(reducedPairedData, frozenData)
        # signalData dimension: batchSize, numSignalPairs + numFrozenSignals, signalDimension
        
        # Free up memory.
        gc.collect(); torch.cuda.empty_cache();
                
        return originalData, signalData
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numExpandedSignals = 2):
        super(signalEncoding, self).__init__(signalDimension, numExpandedSignals) 
        
    def signalEncodingInterface(self, signalData, transformation):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = signalData.size()
        
        # Reshape the data to process each signal seperately.
        signalData = signalData.view(batchSize*numSignals, 1, signalDimension)
        
        # Apply a CNN network.
        signalData = transformation(signalData)
        
        # Return to the initial dimension of the input.
        signalData = signalData.view(batchSize, numSignals, signalDimension)
        
        return signalData
                        
    def forward(self, signalData, targetNumSignals = 32, signalEncodingLayerLoss = 0, calculateLoss = True):
        """ The shape of signalData: (batchSize, numSignals, compressedLength) """
        # Setup the variables for signal encoding.
        batchSize, numSignals, signalDimension = signalData.size()
        numSignalPath = [numSignals] # Keep track of the signal's at each iteration.

        # Assert that we have the expected data format.
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."
        assert self.numCompressedSignals <= targetNumSignals, f"At the minimum, we cannot go lower than compressed signal batch. You provided {targetNumSignals} signals."
        assert self.numCompressedSignals <= numSignals, f"We cannot compress or expand if we dont have at least the compressed signal batch. You provided {numSignals} signals."
        
        # ------------- Signal Compression/Expansion Algorithm ------------- #             
        
        signalEncodingLayer = 0
        # While we have the incorrect number of signals.
        while targetNumSignals != signalData.size(1):
            compressedDataFlag = targetNumSignals < signalData.size(1)
            signalEncodingLayer = signalEncodingLayer + 1
            
            # Compress the signals down to the targetNumSignals.
            if compressedDataFlag: originalData, signalData = self.compressionModel(signalData, targetNumSignals)
            
            # Expand the signals up to the targetNumSignals.
            else: originalData, signalData = self.expansionModel(signalData, targetNumSignals)
                        
            # Keep track of the error during each compression/expansion.
            if calculateLoss: signalEncodingLayerLoss = self.updateLossValues(originalData, signalData, numSignalPath, compressedDataFlag,
                                                                              signalEncodingLayerLoss, signalEncodingLayer)
        
            # Keep track of the signal's at each iteration.
            numSignalPath.append(signalData.size(1))
        
        # ------------------------------------------------------------------ # 
        
        # Assert the integrity of the expansion/compression.
        if numSignals != targetNumSignals:
            assert all(numSignalPath[i] <= numSignalPath[i + 1] for i in range(len(numSignalPath) - 1)) \
                or all(numSignalPath[i] >= numSignalPath[i + 1] for i in range(len(numSignalPath) - 1)), "List is not sorted up or down"
        
        # Remove the target signal from the path.
        numSignalPath.pop()
        
        return signalData, numSignalPath, signalEncodingLayerLoss
        
    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64, numExpandedSignals = 3).to('cpu').printParams(numSignals = 4)
        t1 = time.time()
        summary(self, (numSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
