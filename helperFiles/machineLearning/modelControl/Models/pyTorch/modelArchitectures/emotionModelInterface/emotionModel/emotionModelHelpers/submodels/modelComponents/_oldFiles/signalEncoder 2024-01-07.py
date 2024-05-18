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
        # General shape parameters.
        self.signalDimension = signalDimension
        
        # Create model for learning local information.
        self.signalExpansion = self.signalEncodingModule(inChannel = 1, outChannel = 2)
        self.signalCompression = self.signalEncodingModule(inChannel = 2, outChannel = 1)
        
    # ---------------------------------------------------------------------- #
    # ------------------- Machine Learning Architectures ------------------- #

    def signalEncodingModule(self, inChannel = 1, outChannel = 2):
        return nn.Sequential(            
                # Convolution architecture: channel expansion
                self.deconvolutionalFilter(numChannels = [inChannel, 2, 4], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 3], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),
                
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 3], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),
                
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 3], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),
                
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 3], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),
                
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 3], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [4, 4, 4], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),
                
                # Convolution architecture: channel compression
                self.deconvolutionalFilter(numChannels = [4, 4, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                       
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),
                
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),
                
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),
                
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),
                
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]),
                        self.deconvolutionalFilter_resNet(numChannels = [2, 2, 2], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                ), numCycles = 1),

                # Convolution architecture: channel compression
                self.deconvolutionalFilter(numChannels = [2, 2, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [outChannel, outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [outChannel, outChannel, outChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
        )
    
    def deconvolutionalFilter_resNet(self, numChannels = [6, 6, 6, 6], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]):
        return _convolutionalHelpers.ResNet(
                    module = nn.Sequential( 
                        # Convolution architecture: feature engineering
                        self.deconvolutionalFilter(numChannels = numChannels, kernel_sizes = kernel_sizes, dilations = dilations, groups = groups),
                ), numCycles = 1)
        
    def deconvolutionalFilter(self, numChannels = [6, 6, 6], kernel_sizes = [3, 3, 3], dilations = [1, 2, 1], groups = [1, 1, 1]):
        # Calculate the required padding for no information loss.
        paddings = [dilation * (kernel_size - 1) // 2 for kernel_size, dilation in zip(kernel_sizes, dilations)]
        
        return nn.Sequential(
                # Convolution architecture: feature engineering
                nn.Conv1d(in_channels=numChannels[0], out_channels=numChannels[1], kernel_size=kernel_sizes[0], stride=1, 
                          dilation = dilations[0], padding=paddings[0], padding_mode='reflect', groups=groups[0], bias=True),
                nn.SELU(),
                
                _convolutionalHelpers.ResNet(
                    module = nn.Sequential(
                        # Convolution architecture: feature engineering
                        nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[1], kernel_size=kernel_sizes[1], stride=1, 
                                  dilation = dilations[1], padding=paddings[1], padding_mode='reflect', groups=groups[1], bias=True),
                        nn.SELU(),
                ), numCycles = 1),
                
                # Convolution architecture: feature engineering
                nn.Conv1d(in_channels=numChannels[1], out_channels=numChannels[2], kernel_size=kernel_sizes[2], stride=1, 
                          dilation = dilations[2], padding=paddings[2], padding_mode='reflect', groups=groups[2], bias=True),
                nn.SELU(),
        )
    
    # ---------------------------------------------------------------------- #
    # ---------------------------- Loss Methods ---------------------------- #   
    
    def calculateStandardizationLoss(self, inputData, finalEncodingLayer, expectedMean = 0, expectedStandardDeviation = 1, dim=-1):
        # Base case: we dont want a normalization loss on the last layer.
        if finalEncodingLayer: return 0
        
        # Calculate the data statistics on the last dimension.
        standardDeviationData = inputData.std(dim=dim, keepdim=False)
        meanData = inputData.mean(dim=dim, keepdim=False)

        # Calculate the squared deviation from mean = 0; std = 1.
        standardDeviationError = ((standardDeviationData - expectedStandardDeviation) ** 2).mean(dim=(1,))
        meanError = ((meanData - expectedMean) ** 2).mean(dim=(1,))
        
        if 0.1 < standardDeviationError.mean():
            return 0.1*meanError + 0.9*standardDeviationError
        else:
            return 0.5*meanError + 0.5*standardDeviationError
    
    def calculateEncodingLoss(self, trueData, predictedData):        
        # Calculate the squared error loss.
        return  ((trueData - predictedData) ** 2).mean(dim=-1).mean(dim=1)
    
    def updateLossValues(self, inputData, encodedData, inverseEncoding, batchSize, signalEncodingLayerLoss, numIterations, finalEncodingLayer):
        # Predict the upsampled data.
        decodedData = inverseEncoding(encodedData)
        
        # Reorganize the data for the loss calculation.
        encodedData = self.recompileSignals(encodedData, torch.empty(batchSize, 0, self.signalDimension).to(encodedData.device))
        decodedData = self.recompileSignals(decodedData, torch.empty(batchSize, 0, self.signalDimension).to(encodedData.device))
        inputData = self.recompileSignals(inputData, torch.empty(batchSize, 0, self.signalDimension).to(encodedData.device))

        # Keep tracking on the normalization loss through each loop.
        normalizationLayerLoss = self.calculateStandardizationLoss(encodedData, finalEncodingLayer, expectedMean = 0, expectedStandardDeviation = 1, dim=-1)
        layerLoss = self.calculateEncodingLoss(inputData, decodedData)
        
        # Update the signal encoding layer loss.
        if finalEncodingLayer:
            signalEncodingLayerLoss = signalEncodingLayerLoss + 0.6*layerLoss + 0.4*normalizationLayerLoss
        else:
            signalEncodingLayerLoss = signalEncodingLayerLoss + layerLoss
        numIterations = numIterations + 1
        
        return signalEncodingLayerLoss, numIterations
    
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
        frozenData = inputData[:, numActiveSignals:].contiguous()  # These signals are finalized.
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
                        
    def forward(self, signalData, targetNumSignals = 6, signalEncodingLayerLoss = 0, numIterations = 1e-15, trainingFlag = True):
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

            if trainingFlag:
                # Aggregate all the layer loss values together.
                finalEncodingLayer = targetNumSignals == signalData.size(1)
                signalEncodingLayerLoss, numIterations = self.updateLossValues(pairedData, reducedPairedData, self.signalExpansion, batchSize,
                                                                               signalEncodingLayerLoss, numIterations, finalEncodingLayer)
                            
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
            
            if trainingFlag:
                # Aggregate all the layer loss values together.
                finalEncodingLayer = targetNumSignals == signalData.size(1)
                signalEncodingLayerLoss, numIterations = self.updateLossValues(activeData, expandedData, self.signalCompression, batchSize, 
                                                                               signalEncodingLayerLoss, numIterations, finalEncodingLayer)
                                
        # ------------------------------------------------------------------ # 
        
        # Normalize the losses.
        signalEncodingLayerLoss = signalEncodingLayerLoss/numIterations
        
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
        
        