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
    def __init__(self, signalDimension = 64):
        super(signalEncoderBase, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Specify the CPU or GPU capabilities.
        
        # General shape parameters.
        self.signalDimension = signalDimension
        self.numCompressedSignals = 3
        self.numExpandedSignals = 4
        
        # Specify channel encoding
        self.channelEncoding = torch.ones(2, self.signalDimension).to(self.device)
        self.channelEncoding[1] = self.channelEncoding[1]*-1
        # channelEncoding dimension: 2, signalDimension
        
        # Create model for learning local information.
        self.expandChannels = self.signalEncodingModule(inChannel = 1, outChannel = 2)
        self.compressChannels = self.signalEncodingModule(inChannel = 2, outChannel = 1)
        self.delatCompressChannels = self.signalEncodingModule(inChannel = 2, outChannel = 1)
        
        # Delta learning modules to predict the residuals.
        self.deltaLearning_compressChannel1 = self.deltaLearningModule(inChannel = 1)
        self.deltaLearning_compressChannel2 = self.deltaLearningModule(inChannel = 1)
        self.deltaLearning_expansionchannel1 = self.deltaLearningModule(inChannel = 1)
        self.deltaLearning_expansionchannel2 = self.deltaLearningModule(inChannel = 1)
        self.deltaLearning_expansionchannel3 = self.deltaLearningModule(inChannel = 2)

    def signalCompression(self, expandedData):
        # Delta learning what to add to each channel.
        channelOneInfo = self.deltaLearning_compressChannel1(expandedData[:, 0:1, :])/2
        channelTwoInfo = self.deltaLearning_compressChannel2(expandedData[:, 1:2, :])/2
        
        # Add the channels together.
        deltaChannelData = torch.cat((channelOneInfo, channelTwoInfo), dim=1).contiguous()
        deltaChannelData = self.delatCompressChannels(deltaChannelData)/2
        # deltaChannelData dimension: batchSize, 2, signalDimension
        
        # Add channel positional encoding.
        expandedData = expandedData + self.channelEncoding.expand(expandedData.size(0), 2, self.signalDimension)
        # Predict how the signals will condense together.
        averagePrediction = self.compressChannels(expandedData)/2
        
        # Compress the data
        return channelOneInfo + channelTwoInfo + deltaChannelData + averagePrediction
    
    def signalExpansion(self, compressedData):        
        # Delta learning what to add to each channel.
        channel1 = self.deltaLearning_expansionchannel1(compressedData)
        channel2 = self.deltaLearning_expansionchannel2(compressedData)
        # channel2 = 2*compressedData - channel1 # Such that we assume (channel1 + channel2)/2 = expandedData
        
        # Add the channels together.
        channelInfo = torch.cat((channel1, channel2), dim=1).contiguous()
        deltaChannelData = self.deltaLearning_expansionchannel3(channelInfo)
        # deltaChannelData dimension: batchSize, 2, signalDimension
        
        return channelInfo + deltaChannelData/2 + self.expandChannels(compressedData)/2
        
    # ---------------------------------------------------------------------- #
    # ------------------- Machine Learning Architectures ------------------- #

    def signalEncodingModule(self, inChannel = 2, outChannel = 1):
        return nn.Sequential( 
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 16*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 16*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [2, 2], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 16*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [3, 3], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 16*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [4, 4], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 16*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [5, 5], groups = [1, 1]),                                
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 8*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [4, 4], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 8*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [3, 3], groups = [1, 1]),

                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 8*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [2, 2], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 8*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                                
                # Convolution architecture: channel expansion
                self.deconvolutionalFilter(numChannels = [inChannel, inChannel*8, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [outChannel, 4*outChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
        )

    def deltaLearningModule(self, inChannel = 1):
        return nn.Sequential( 
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 16*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 16*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [2, 2], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 16*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [3, 3], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 16*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [4, 4], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 16*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [5, 5], groups = [1, 1]),                                
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 8*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [4, 4], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 8*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [3, 3], groups = [1, 1]),

                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 8*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [2, 2], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, 8*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
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
            # Switch the signals and retry the compression.
            switchedPairedData = originalData[:, [1, 0], :]
            switchedEncodedData = self.signalCompression(switchedPairedData)
            
            # PRedict the original expanded form.
            predictedData = self.signalExpansion(encodedData)
            switchedPredictedData = self.signalExpansion(switchedEncodedData)
            
            # Reorganize the data for the loss calculation.
            switchedPairedData = switchedPairedData.view(batchSize, -1, self.signalDimension)
            switchedPredictedData = switchedPredictedData.view(batchSize, -1, self.signalDimension)
        else:
            # Predict the original compressed form.
            predictedData = self.signalCompression(encodedData)
        
        # Reorganize the data for the loss calculation.
        predictedData = predictedData.view(batchSize, -1, self.signalDimension)
        originalData = originalData.view(batchSize, -1, self.signalDimension)
        
        # Calculate the squared error loss.
        squaredErrorLoss = (originalData - predictedData).pow(2).mean(dim=-1).mean(dim=1)
        # squaredErrorLoss dimension: batchSize
        
        if compressingDataFlag:
            # Calculate the squared error loss.
            switchedSquaredErrorLoss = (switchedPairedData - switchedPredictedData).pow(2).mean(dim=-1).mean(dim=1)
            
            # Average both loss values.
            squaredErrorLoss = squaredErrorLoss + switchedSquaredErrorLoss
        
        return squaredErrorLoss
    
    def updateLossValues(self, originalData, encodedData, compressingDataFlag, batchSize, signalEncodingLayerLoss, finalEncodingLayer):
        # Keep tracking on the normalization loss through each loop.
        normalizationLayerLoss = self.calculateStandardizationLoss(encodedData, batchSize, finalEncodingLayer, expectedMean = 0, expectedStandardDeviation = 1, dim=-1)
        layerLoss = self.calculateEncodingLoss(originalData, encodedData, compressingDataFlag, batchSize)
        
        # Update the signal encoding layer loss.
        if not finalEncodingLayer:
            signalEncodingLayerLoss = signalEncodingLayerLoss + layerLoss + 0.05*normalizationLayerLoss
        else:
            signalEncodingLayerLoss = signalEncodingLayerLoss + layerLoss
        
        return signalEncodingLayerLoss
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Data Organization ------------------------- #
    
    def seperateActiveData(self, inputData, targetNumSignals):
        # Extract the signal number.
        numSignals = inputData.size(1)
        
        # If we are upsampling the signals by 2.
        if numSignals*2 <= targetNumSignals:
            # Upsample the max number of signals.
            numActiveSignals = numSignals
        
        # If we are only slightly below the final number of signals.
        elif numSignals < targetNumSignals < numSignals*2:
            # Find the number of signals to double.
            numActiveSignals = targetNumSignals - numSignals
        
        # If we are only slightly above the final number of signals.
        elif targetNumSignals < numSignals < targetNumSignals*2:
            # Find the number of signals to pair up and reduce.
            numActiveSignals = (numSignals - targetNumSignals)*2
        
        # If we are reducing the signals by 2.
        elif targetNumSignals*2 <= numSignals:
            # We can only pair up an even number.
            numActiveSignals = numSignals - (numSignals % 2)
            
        # Base case: numSignals == targetNumSignals
        else: numActiveSignals = 0
        
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
        numSignalPairs = int(activeData.size(1)/2)
        pairedData = activeData.view(batchSize, numSignalPairs, 2, signalDimension)
        pairedData = pairedData.view(batchSize*numSignalPairs, 2, signalDimension)
        # pairedData dimension: batchSize*numSignalPairs, 2, signalDimension
                
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
    def __init__(self, signalDimension = 64):
        super(signalEncoding, self).__init__(signalDimension)        
                        
    def forward(self, signalData, targetNumSignals = 64, signalEncodingLayerLoss = 0, numIterations = 1e-15, calculateLoss = True):
        """ The shape of signalData: (batchSize, numSignals, compressedLength) """
        # Setup the variables for signal encoding.
        batchSize, numSignals, signalDimension = signalData.size()
        numSignalPath = [numSignals] # Keepn track of the signal's at each iteration.

        # Assert that we have the expected data format.
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."

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
            activeData = activeData.view(activeData.size(0)*activeData.size(1), 1, signalDimension) # Create a channel for the CNN.
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
        
        
