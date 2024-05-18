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
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import _convolutionalHelpers

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class signalEncoderBase(_convolutionalHelpers.convolutionalHelpers):
    def __init__(self, signalDimension = 64, numExpandedSignals = 4):
        super(signalEncoderBase, self).__init__()        
        # General shape parameters.
        self.signalDimension = signalDimension  # The incoming dimension of each signal.
        
        # Compression/Expansion parameters.
        self.numExpandedSignals = numExpandedSignals        # The final number of signals in any expansion
        self.numCompressedSignals = numExpandedSignals - 1  # The final number of signals in any compression.
        self.expansionFactor = self.numExpandedSignals/self.numCompressedSignals  # The percent expansion.
        
        # Assert the integrity of the input parameters.
        assert self.numExpandedSignals - self.numCompressedSignals == 1, "You should only gain 1 channel when expanding or else you may overshoot."
        
        # Create model for learning local information.
        self.expandChannels = self.signalEncodingModule(inChannel = self.numCompressedSignals, outChannel = self.numExpandedSignals, channelIncrease = self.numCompressedSignals*4)
        self.compressChannels = self.signalEncodingModule(inChannel = self.numExpandedSignals, outChannel = self.numCompressedSignals, channelIncrease = self.numExpandedSignals*4)
        
        # Delta learning modules to predict the residuals.
        self.deltaCompression_multiChannelInfo = nn.ModuleList()  # Use ModuleList to store child modules.
        self.deltaExpansion_multiChannelInfo = nn.ModuleList()    # Use ModuleList to store child modules.
        self.deltaCompression_channelInfo = nn.ModuleList()  # Use ModuleList to store child modules.
        self.deltaExpansion_channelInfo = nn.ModuleList()    # Use ModuleList to store child modules.
        
        # For each delta channel module to add.
        for deltaExpansionModelInd in range(self.numExpandedSignals):
            self.deltaCompression_channelInfo.append(self.signalEncodingModule(inChannel = 1, outChannel = self.numCompressedSignals, channelIncrease = self.numCompressedSignals))
            # self.deltaExpansion_multiChannelInfo.append(self.signalEncodingModule(inChannel = self.numCompressedSignals, outChannel = 1, channelIncrease = self.numCompressedSignals*2))
            
        # For each delta channel module to add.
        for deltaExpansionModelInd in range(self.numCompressedSignals):
            # self.deltaCompression_multiChannelInfo.append(self.signalEncodingModule(inChannel = self.numExpandedSignals, outChannel = 1, channelIncrease = self.numExpandedSignals*2))
            self.deltaExpansion_channelInfo.append(self.signalEncodingModule(inChannel = 1, outChannel = self.numExpandedSignals, channelIncrease = self.numExpandedSignals))
            
        # Initialize the pooling layers to upsample/downsample
        self.interpolateChannels = nn.Upsample(size=self.numExpandedSignals, mode='linear', align_corners=True)

    def signalCompression(self, expandedData):
        # Compile the information learned from each channel seperately. 
        singleChannelInfo, multiChannelInfo = self.compileChannelInfo(expandedData, self.deltaCompression_channelInfo, self.deltaCompression_multiChannelInfo)
                
        # Predict how the signals will compress together.
        estimatedCompression = self.compressChannels(expandedData)
        
        # Crude estimation of how the channels will compress.
        averagePrediction = (expandedData[:, :-1, :] + expandedData[:, 1:, :])/2
        
        # Return the summation of each prediction.
        return singleChannelInfo + multiChannelInfo + estimatedCompression + averagePrediction
    
    def signalExpansion(self, compressedData):
        # Compile the information learned from each channel seperately. 
        singleChannelInfo, multiChannelInfo = self.compileChannelInfo(compressedData, self.deltaExpansion_channelInfo, self.deltaExpansion_multiChannelInfo)
                
        # Predict how the signals will expand together.
        estimatedExpansion = self.expandChannels(compressedData)
        
        # Crude estimation of how the channels will expand.
        averagePrediction = self.interpolateChannels(compressedData.transpose(1, 2)).transpose(1, 2)
        
        # Return the summation of each prediction.
        return singleChannelInfo + multiChannelInfo + estimatedExpansion + averagePrediction
    
    def compileChannelInfo(self, inputData, singleChannelModels, multiChannelModels):
        # Setup channel info holders.
        singleChannelInfo = 0
        multiChannelInfo = []
        
        # For each delta channel module to compress the data.
        for deltaModelInd in range(len(singleChannelModels)):
            channelData = inputData[:, deltaModelInd:deltaModelInd+1, :]
            # channelData dimension: batchSize, 1, signalDimension
            
            # Delta learn information to add to each channel.
            singleChannelInfo = singleChannelInfo + singleChannelModels[deltaModelInd](channelData)
            # singleChannelInfo dimension: batchSize, numSignals, signalDimension
            
        # # For each delta channel module to compress the data.
        # for deltaModelInd in range(len(multiChannelModels)):
        #     # Delta learn information to add to each channel.
        #     multiChannelInfo.append(multiChannelModels[deltaModelInd](inputData))            
        # # Stack the multi-channel information together.
        # multiChannelInfo = torch.cat(multiChannelInfo, dim=1).contiguous()
        # # multiChannelInfo dimension: batchSize, numSignals, signalDimension
            
        return singleChannelInfo, 0
        # return singleChannelInfo, multiChannelInfo
        
    # ---------------------------------------------------------------------- #
    # ------------------- Machine Learning Architectures ------------------- #

    def signalEncodingModule(self, inChannel = 2, outChannel = 1, channelIncrease = 8):
        return nn.Sequential( 
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [2, 2], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [3, 3], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [2, 2], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                                
                # Convolution architecture: channel expansion
                self.deconvolutionalFilter(numChannels = [inChannel, 4*inChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                # self.deconvolutionalFilter_resNet(numChannels = [outChannel, 2*outChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [outChannel, outChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
        )
    
    def deconvolutionalFilter_resNet(self, numChannels = [6, 6, 6], kernel_sizes = [3, 3], dilations = [1, 2, 1], groups = [1, 1]):
        return _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter(numChannels = numChannels, kernel_sizes = kernel_sizes, dilations = dilations, groups = groups),
                ), numCycles = 1)
        
    def deconvolutionalFilter(self, numChannels = [6, 6, 6], kernel_sizes = [3, 3], dilations = [1, 2], groups = [1, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
                # Convolution architecture: feature engineering
                nn.Conv1d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=kernel_sizes[0], stride=1, 
                          dilation = dilations[0], padding=paddings[0], padding_mode='reflect', groups=groups[0], bias=True),
                nn.SELU(),
                
                # Convolution architecture: feature engineering
                nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=kernel_sizes[1], stride=1, 
                          dilation = dilations[1], padding=paddings[1], padding_mode='reflect', groups=groups[1], bias=True),
                nn.SELU(),
        )
    
    def changeChannels(self, numChannels = [6, 6], kernel_sizes = [3], dilations = [1], groups = [1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
                # Convolution architecture: feature engineering
                nn.Conv1d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=kernel_sizes[0], stride=1, 
                          dilation = dilations[0], padding=paddings[0], padding_mode='reflect', groups=groups[0], bias=True),
                nn.SELU(),
        )
    
    # ---------------------------------------------------------------------- #
    # ---------------------------- Loss Methods ---------------------------- #   
    
    def calculateStandardizationLoss(self, inputData, batchSize, finalEncodingLayer, expectedMean = 0, expectedStandardDeviation = 1, dim=-1):
        # Base case: we dont want a normalization loss on the last layer.
        if finalEncodingLayer: return 0
        
        # Reorganize the data for the loss calculation.
        inputData = inputData.view(batchSize, -1, self.signalDimension)

        # Calculate the data statistics on the last dimension.
        standardDeviationData = inputData.std(dim=dim, keepdim=False)
        meanData = inputData.mean(dim=dim, keepdim=False)

        # Calculate the squared deviation from mean = 0; std = 1.
        standardDeviationError = (standardDeviationData - expectedStandardDeviation).pow(2).mean(dim=1)
        meanError = (meanData - expectedMean).pow(2).mean(dim=1)
                
        return 0.5*meanError + 0.5*standardDeviationError
    
    def calculateEncodingLoss(self, originalData, encodedData, compressingDataFlag, batchSize): 
        # If we compressed the data
        if compressingDataFlag:
            # Predict the original expanded form.
            predictedData = self.signalExpansion(encodedData)
            reencodedData = self.signalCompression(predictedData)
            predictedOriginalData = self.signalExpansion(reencodedData)
        else:
            # Predict the original compressed form.
            predictedData = self.signalCompression(encodedData)
            reencodedData = self.signalExpansion(predictedData)
            predictedOriginalData = self.signalCompression(reencodedData)
        
        # Reorganize the data for the loss calculation.
        predictedOriginalData = predictedOriginalData.view(batchSize, -1, self.signalDimension)
        predictedData = predictedData.view(batchSize, -1, self.signalDimension)
        originalData = originalData.view(batchSize, -1, self.signalDimension)
        reencodedData = reencodedData.view(batchSize, -1, self.signalDimension)
        encodedData = encodedData.view(batchSize, -1, self.signalDimension)
        
        # Calculate the squared error loss.
        squaredErrorLoss_forward = (originalData - predictedData).pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_backward = (reencodedData - encodedData).pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_forwardLayer2 = (originalData - predictedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        # squaredErrorLoss dimension: batchSize
        
        # Give compression a higher loss.
        if compressingDataFlag:
            squaredErrorLoss_forward = squaredErrorLoss_forward*2
        else:
            squaredErrorLoss_backward = squaredErrorLoss_backward*2
        
        # Return a summation of the losses (they are two examples).
        return squaredErrorLoss_forward + squaredErrorLoss_backward + 2*squaredErrorLoss_forwardLayer2
    
    def updateLossValues(self, originalData, encodedData, compressingDataFlag, batchSize, signalEncodingLayerLoss, finalEncodingLayer):
        # Keep tracking on the normalization loss through each loop.
        # normalizationLayerLoss = self.calculateStandardizationLoss(encodedData, batchSize, finalEncodingLayer, expectedMean = 0, expectedStandardDeviation = 1, dim=-1)
        layerLoss = self.calculateEncodingLoss(originalData, encodedData, compressingDataFlag, batchSize)
        
        # Update the signal encoding layer loss.
        if not finalEncodingLayer:
            signalEncodingLayerLoss = signalEncodingLayerLoss + layerLoss #+ 0.1*normalizationLayerLoss
        else:
            signalEncodingLayerLoss = signalEncodingLayerLoss + layerLoss
        
        return signalEncodingLayerLoss
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Data Organization ------------------------- #
    
    def seperateActiveData(self, inputData, targetNumSignals):
        # Extract the signal number.
        numSignals = inputData.size(1)
                
        # If we are upsampling the signals as much as I can..
        if numSignals*self.expansionFactor <= targetNumSignals:
            # Upsample the max number of signals.
            numActiveSignals = numSignals - (numSignals%self.numCompressedSignals)
        
        # If we are only slightly below the final number of signals.
        elif numSignals < targetNumSignals < numSignals*self.expansionFactor:
            # Find the number of signals to expand.
            numSignalsGained = targetNumSignals - numSignals
            numExpansions = numSignalsGained/(self.numExpandedSignals - self.numCompressedSignals)
            numActiveSignals = numExpansions*self.numCompressedSignals
            assert numActiveSignals <= numSignals, "This must be true if the logic is working."
        
        # If we are only slightly above the final number of signals.
        elif targetNumSignals < numSignals < targetNumSignals*self.expansionFactor:
            # Find the number of signals to reduce.
            numSignalsLossed = numSignals - targetNumSignals
            numCompressions = numSignalsLossed/(self.numExpandedSignals - self.numCompressedSignals)
            numActiveSignals = numCompressions*self.numExpandedSignals  
            assert numActiveSignals <= numSignals, "This must be true if the logic is working."
                    
        # If we are reducing the signals as much as I can..
        elif targetNumSignals*self.expansionFactor <= numSignals:
            # We can only pair up an even number.
            numActiveSignals = numSignals - (numSignals%self.numExpandedSignals)
            
        # Base case: numSignals == targetNumSignals
        else: numActiveSignals = 0
        numActiveSignals = int(numActiveSignals)
                        
        # Segment the tensor into its frozen and active components.
        activeData = inputData[:, 0:numActiveSignals].contiguous() # Reducing these signals.
        frozenData = inputData[:, numActiveSignals:numSignals].contiguous()  # These signals are finalized.
        # Only the last rows of signals are frozen.
        
        return activeData, frozenData
        
    def pairSignals(self, inputData, targetNumSignals):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.shape
        
        # Assert that this method is only for signal reduction.
        assert targetNumSignals < numSignals, f"Why are you trying to reduce {numSignals} signals to {targetNumSignals} signals"
                        
        # Seperate out the active and frozen data.
        activeData, frozenData = self.seperateActiveData(inputData, targetNumSignals)
        # activeData dimension: batchSize, numActiveSignals, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension

        # Pair up the signals.
        numSignalPairs = int(activeData.size(1)/self.numExpandedSignals)
        pairedData = activeData.view(batchSize, numSignalPairs, self.numExpandedSignals, signalDimension)
        pairedData = pairedData.view(batchSize*numSignalPairs, self.numExpandedSignals, signalDimension)
        # pairedData dimension: batchSize*numSignalPairs, numExpandedSignals, signalDimension
                
        return pairedData, frozenData
    
    def recompileSignals(self, pairedData, frozenData):
        # Extract the incoming data's dimension.
        batchPairedSize, numChannels, signalDimension = pairedData.shape
        batchSize, numFrozenSignals, signalDimension = frozenData.shape
                
        # Seperate out the paired data into its batches.
        unpairedData = pairedData.view(batchSize, int(numChannels*batchPairedSize/batchSize), signalDimension)
        # unpairedData dimension: batchSize, numSignalPairs, signalDimension
                                    
        # Recombine the paired and frozen data.
        recombinedData = torch.cat((unpairedData, frozenData), dim=1).contiguous()
        # recombinedData dimension: batchSize, numSignalPairs + numFrozenSignals, signalDimension
                                        
        return recombinedData
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numExpandedSignals = 4):
        super(signalEncoding, self).__init__(signalDimension, numExpandedSignals)        
                        
    def forward(self, signalData, targetNumSignals = 64, signalEncodingLayerLoss = 0, numIterations = 1e-15, calculateLoss = True):
        """ The shape of signalData: (batchSize, numSignals, compressedLength) """
        # Setup the variables for signal encoding.
        batchSize, numSignals, signalDimension = signalData.size()
        numSignalPath = [numSignals] # Keepn track of the signal's at each iteration.

        # Assert that we have the expected data format.
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."
        assert self.numCompressedSignals <= targetNumSignals, f"At the minimum, we cannot go lower than compressed signal batch. You provided {targetNumSignals} signals."
        assert self.numCompressedSignals <= numSignals, f"We cannot compress or expand if we dont have at least the compressed signal batch. You provided {numSignals} signals."

        # ------------------ Signal Compression Algorithm ------------------ #             
        
        # While we have too many signals.
        while targetNumSignals < signalData.size(1):
            # Pair up the signals with their neighbors.
            pairedData, frozenData = self.pairSignals(signalData, targetNumSignals)
            # pairedData dimension: batchSize*numSignalPairs, 2, signalDimension
            # frozenData dimension: batchSize, numFrozenSignals, signalDimension
            
            # Pool half the paired data together.
            reducedPairedData = self.signalCompression(pairedData)
            # reducedPairedData dimension: batchSize*numSignalPairs, 1, signalDimension

            # Recompile the signals to their original dimension.
            signalData = self.recompileSignals(reducedPairedData, frozenData)
            # signalData dimension: batchSize, numSignalPairs + numFrozenSignals, signalDimension

            # Keepn track of the signal's at each iteration.
            numSignalPath.append(signalData.size(1))
            
            if calculateLoss:
                # Keep track of the encodings.
                numIterations = numIterations + 1
                
                # Aggregate all the layer loss values together.
                finalEncodingLayer = targetNumSignals == signalData.size(1)
                signalEncodingLayerLoss = self.updateLossValues(pairedData, reducedPairedData, True, batchSize,
                                                                signalEncodingLayerLoss, finalEncodingLayer)

        # ------------------- Signal Expansion Algorithm ------------------- # 
            
        # While we have too many signals to process.
        while signalData.size(1) < targetNumSignals: 
            # Seperate out the active and frozen data.
            activeData, frozenData = self.seperateActiveData(signalData, targetNumSignals)
            activeData = activeData.view(int(activeData.size(0)*activeData.size(1)/self.numCompressedSignals), self.numCompressedSignals, signalDimension) # Create a channel for the CNN.
            # activeData dimension: batchSize*numSignalPairs, 1, signalDimension
            # frozenData dimension: batchSize, numFrozenSignals, signalDimension
                        
            # Upsample to twice as many signals.
            expandedData = self.signalExpansion(activeData)
            # expandedData dimension: batchSize*numSignalPairs, 2, signalDimension
            
            # Recompile the signals to their original dimension.
            signalData = self.recompileSignals(expandedData, frozenData)
            # signalData dimension: batchSize, 2*numSignalPairs + numFrozenSignals, signalDimension
                        
            # Keepn track of the signal's at each iteration.
            numSignalPath.append(signalData.size(1))
            
            if calculateLoss:
                numIterations = numIterations + 1

                # Aggregate all the layer loss values together.
                finalEncodingLayer = targetNumSignals == signalData.size(1)
                signalEncodingLayerLoss = self.updateLossValues(activeData, expandedData, False, batchSize,
                                                                signalEncodingLayerLoss, finalEncodingLayer)
                                
        # ------------------------------------------------------------------ # 
        
        # Normalize the losses.
        # signalEncodingLayerLoss = signalEncodingLayerLoss/numIterations
        # print(signalEncodingLayerLoss)
        
        # Remove the target signal from the path.
        numSignalPath.pop()
        
        return signalData, numSignalPath, signalEncodingLayerLoss
        
    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64).to('cpu').printParams(numSignals = 4)
        t1 = time.time()
        summary(self, (numSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
