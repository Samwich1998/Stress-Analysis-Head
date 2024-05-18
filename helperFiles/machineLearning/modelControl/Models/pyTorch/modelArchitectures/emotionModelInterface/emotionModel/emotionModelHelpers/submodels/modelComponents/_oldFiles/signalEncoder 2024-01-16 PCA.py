# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import time

# Plotting
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import _convolutionalHelpers
import positionEncodings

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class signalEncoderBase(_convolutionalHelpers.convolutionalHelpers):
    def __init__(self, signalDimension = 64, numExpandedSignals = 4, plotSignalEncoding = False):
        super(signalEncoderBase, self).__init__()        
        # General shape parameters.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Specify the CPU or GPU capabilities.
        self.signalDimension = signalDimension  # The incoming dimension of each signal.
        
        # Compression/Expansion parameters.
        self.numExpandedSignals = numExpandedSignals        # The final number of signals in any expansion
        self.numCompressedSignals = numExpandedSignals - 1  # The final number of signals in any compression.
        self.expansionFactor = self.numExpandedSignals/self.numCompressedSignals  # The percent expansion.
        
        # Positional encoding.
        self.channelEncoding = positionEncodings.positionalEncoding.to(self.device).T[-self.numExpandedSignals:]
        # channelEncoding dimension: numExpandedSignals, signalDimension
        
        # Assert the integrity of the input parameters.
        assert self.numExpandedSignals - self.numCompressedSignals == 1, "You should only gain 1 channel when expanding or else you may overshoot."
        
        # Create model for learning local information.
        self.expandChannels = self.signalEncodingModule(inChannel = self.numCompressedSignals, outChannel = self.numExpandedSignals, channelIncrease = 4)
        self.compressChannels = self.signalEncodingModule(inChannel = self.numExpandedSignals, outChannel = self.numCompressedSignals, channelIncrease = 2)
        self.compressChannelsPCA = self.signalEncodingModule(inChannel = self.numCompressedSignals, outChannel = self.numCompressedSignals, channelIncrease = 8)
        
        # Delta learning modules to predict the residuals.
        # self.deltaCompression_channelInfo = self.channelSpecificModule(inChannel = 1, outChannel = self.numCompressedSignals, channelIncrease = 2, numGroups = self.numExpandedSignals)
        # self.deltaExpansion_channelInfo = self.channelSpecificModule(inChannel = 1, outChannel = self.numExpandedSignals, channelIncrease = 2, numGroups = self.numCompressedSignals)

        # Initialize the pooling layers to upsample/downsample
        self.interpolateChannels = nn.Upsample(size=self.numExpandedSignals, mode='linear', align_corners=True)
        
        if plotSignalEncoding:
            plt.plot(self.channelEncoding.T)
            plt.show()
            
    def signalCompression(self, expandedData):
        # Compile the information learned from each channel seperately. 
        # singleChannelInfo = self.compileChannelInfo(expandedData, self.deltaCompression_channelInfo, self.numExpandedSignals)

        # Predict how the signals will expand together.
        estimatedSignalArrangement2 = self.compressChannels(expandedData)
        
        # Perform the optimal compression via PCA and embed channel information (for reconstruction).
        pcaProjection, principal_components = self.pcaCompression(expandedData)
        channelEmbeddedPCA = self.embedChannelInfo(pcaProjection, principal_components)
        # Predict how the signals will compress together.
        estimatedSignalArrangement = self.compressChannelsPCA(channelEmbeddedPCA)
                
        # Crude estimation of how the channels will compress.
        potentialSignalArrangement = (expandedData[:, :-1, :] + expandedData[:, 1:, :])/2
                
        # Return the summation of each prediction.
        return estimatedSignalArrangement + estimatedSignalArrangement2 + potentialSignalArrangement
    
    def signalExpansion(self, compressedData):
        # Compile the information learned from each channel seperately. 
        # singleChannelInfo = self.compileChannelInfo(compressedData, self.deltaExpansion_channelInfo, self.numCompressedSignals)

        # Predict how the signals will expand together.
        estimatedExpansion = self.expandChannels(compressedData)
        
        # Crude estimation of how the channels will expand.
        averagePrediction = self.interpolateChannels(compressedData.transpose(1, 2)).transpose(1, 2)
        
        # Return the summation of each prediction.
        return estimatedExpansion + averagePrediction
    
    def compileChannelInfo(self, inputData, singleChannelModels, numChunks):
        # Delta learn information to add to each channel.
        allSignleChannelInfo = singleChannelModels(inputData)
        # Compile and add all the delta learning channel information together.
        groupedSingleChannelInfo = torch.chunk(allSignleChannelInfo, chunks=numChunks, dim=1)
        singleChannelInfo = torch.stack(groupedSingleChannelInfo).sum(dim=0)   
            
        return singleChannelInfo
        
    def pcaCompression(self, signals):
        # Extract the incoming data's dimension.
        batch_size, num_signals, signal_dimension = signals.shape

        # Calculate means and standard deviations
        signalMeans = signals.mean(dim=-1, keepdim=True)
        signalSTDs = signals.std(dim=-1, keepdim=True)
        # Standardize the signals
        standardized_signals = (signals - signalMeans) / signalSTDs

        # Calculate the covariance matrix
        covariance_matrix = torch.matmul(standardized_signals, standardized_signals.transpose(1,2)) / (signal_dimension - 1)

        # Perform eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # Select the top n_components eigenvectors
        principal_components = eigenvectors[:, :, -self.numCompressedSignals:]

        # Project the original signals to the new subspace
        projected_signals = torch.matmul(principal_components.transpose(1, 2), standardized_signals)
        
        return projected_signals, principal_components
    
    def embedChannelInfo(self, projected_signals, principal_components):
        # Extract the incoming data's dimension.
        batchSize, nunComponents, signalDimension = projected_signals.size()
        batchSize, numExpandedSignals, nunComponents = principal_components.size()
        # Assert the integrity of the input parameters.
        assert numExpandedSignals == self.numExpandedSignals
        assert nunComponents == self.numCompressedSignals
        assert signalDimension == self.signalDimension
                
        # Normalize the principle components.
        principal_components = nn.functional.softmax(principal_components, dim=-1)
                
        # Add up the positional encodings per each component.
        batchPositionalEncoding = self.channelEncoding.expand(batchSize, numExpandedSignals, signalDimension)
        finalChannelEncoding = torch.matmul(principal_components.transpose(1, 2), batchPositionalEncoding)
        # finalChannelEncoding dimension: batchSize, nunComponents, signalDimension
        
        # Add the positional encoding to the pca analysis.
        encodedProjection = projected_signals + finalChannelEncoding
        
        return encodedProjection
    
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
                self.deconvolutionalFilter(numChannels = [inChannel, 2*outChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [outChannel, 2*outChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
        )
    
    def channelSpecificModule(self, inChannel = 2, outChannel = 1, channelIncrease = 8, numGroups = 1):
        # Setup the problem for each group.
        outChannel = int(outChannel*numGroups)
        inChannel = int(inChannel*numGroups)
        
        return nn.Sequential( 
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [2, 2], groups = numGroups),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups),

                # Convolution architecture: channel expansion
                self.deconvolutionalFilter(numChannels = [inChannel, 2*outChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [outChannel, 2*outChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups),
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
        if not isinstance(groups, list): groups = [groups, groups]
        
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
    
    def calculateStandardizationLoss(self, inputData, expectedMean = 0, expectedStandardDeviation = 1, dim=-1):
        # Calculate the data statistics on the last dimension.
        standardDeviationData = inputData.std(dim=dim, keepdim=False)
        meanData = inputData.mean(dim=dim, keepdim=False)

        # Calculate the squared deviation from mean = 0; std = 1.
        standardDeviationError = (standardDeviationData - expectedStandardDeviation).pow(2).mean(dim=1)
        meanError = (meanData - expectedMean).pow(2).mean(dim=1)
                
        return 0.5*meanError + 0.5*standardDeviationError
    
    def calculateEncodingLoss(self, originalData, encodedData, compressingDataFlag): 
        # Setup the variables for signal encoding.
        originalNumSignals = originalData.size(1)
        encodedNumSignals = encodedData.size(1)
        
        #               doubleDecodedData   
        #
        #               //             \\       
        #
        #          decodedData         doubleDecodedEncodedData
        #    
        #          /         \                         \\
        #                decodedEncodedOriginalData    doubleDecodedOriginalData
        # originalData          
        #                encodedDecodedOriginalData    doubleEncodedOriginalData
        #          \         /                         //
        #                                            
        #          encodedData         doubleEncodedDecodedData
        #
        #               \\             //       
        #    
        #               doubleEncodedData   
        
        # If we compressed the data
        if compressingDataFlag: 
            # Setup the variables for signal encoding.
            decodedNumSignals = self.getMaxSignals_Expansion(originalNumSignals)
            doubleDecodedNumSignals = self.getMaxSignals_Expansion(decodedNumSignals)
            doubleEncodedNumSignals = self.getMaxSignals_Compression(encodedNumSignals)
            
            # Traveling along the top-middle of the pyramid.
            _, decodedData = self.expansionAlgorithm(originalData, decodedNumSignals)
            _, decodedEncodedOriginalData = self.compressionAlgorithm(decodedData, originalNumSignals)

            # Traveling along the top of the pyramid.
            _, doubleDecodedData = self.expansionAlgorithm(decodedData, doubleDecodedNumSignals)
            _, doubleDecodedEncodedData = self.compressionAlgorithm(doubleDecodedData, decodedNumSignals)
            _, doubleDecodedOriginalData = self.compressionAlgorithm(doubleDecodedEncodedData, originalNumSignals)
            
            # Traveling along the bottom-middle of the pyramid.
            _, encodedDecodedOriginalData = self.expansionAlgorithm(encodedData, originalNumSignals)
            
            # Traveling along the bottom of the pyramid.
            _, doubleEncodedData = self.compressionAlgorithm(encodedData, doubleEncodedNumSignals)
            _, doubleEncodedDecodedData = self.expansionAlgorithm(doubleEncodedData, encodedNumSignals)
            _, doubleEncodedOriginalData = self.expansionAlgorithm(doubleEncodedDecodedData, originalNumSignals)
        else:
            # Setup the variables for signal encoding.
            decodedNumSignals = self.getMaxSignals_Compression(originalNumSignals)
            doubleDecodedNumSignals = self.getMaxSignals_Compression(decodedNumSignals)
            doubleEncodedNumSignals = self.getMaxSignals_Expansion(encodedNumSignals)
            
            # Traveling along the top-middle of the pyramid.
            _, decodedData = self.compressionAlgorithm(originalData, decodedNumSignals)
            _, decodedEncodedOriginalData = self.expansionAlgorithm(decodedData, originalNumSignals)

            # Traveling along the top of the pyramid.
            _, doubleDecodedData = self.compressionAlgorithm(decodedData, doubleDecodedNumSignals)
            _, doubleDecodedEncodedData = self.expansionAlgorithm(doubleDecodedData, decodedNumSignals)
            _, doubleDecodedOriginalData = self.expansionAlgorithm(doubleDecodedEncodedData, originalNumSignals)
            
            # Traveling along the bottom-middle of the pyramid.
            _, encodedDecodedOriginalData = self.compressionAlgorithm(encodedData, originalNumSignals)
            
            # Traveling along the bottom of the pyramid.
            _, doubleEncodedData = self.expansionAlgorithm(encodedData, doubleEncodedNumSignals)
            _, doubleEncodedDecodedData = self.compressionAlgorithm(doubleEncodedData, encodedNumSignals)
            _, doubleEncodedOriginalData = self.compressionAlgorithm(doubleEncodedDecodedData, originalNumSignals)

        # Setup the loss variables.
        squaredErrorLoss_middle = 0
        squaredErrorLoss_forward = 0
        squaredErrorLoss_backward = 0

        # Calculate the squared error loss: middle level forward
        squaredErrorLoss_forward = squaredErrorLoss_forward + (originalData - encodedDecodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_forward = squaredErrorLoss_forward + (originalData - doubleEncodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        # Calculate the squared error loss: middle level backward
        squaredErrorLoss_backward = squaredErrorLoss_backward + (originalData - decodedEncodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_backward = squaredErrorLoss_backward + (originalData - doubleDecodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        # Calculate the squared error loss: middle level middle
        squaredErrorLoss_middle = squaredErrorLoss_middle + (decodedEncodedOriginalData - encodedDecodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_middle = squaredErrorLoss_middle + (decodedEncodedOriginalData - doubleDecodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_middle = squaredErrorLoss_middle + (decodedEncodedOriginalData - doubleEncodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_middle = squaredErrorLoss_middle + (encodedDecodedOriginalData - doubleDecodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_middle = squaredErrorLoss_middle + (encodedDecodedOriginalData - doubleEncodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        squaredErrorLoss_middle = squaredErrorLoss_middle + (doubleDecodedOriginalData - doubleEncodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        
        # Calculate the squared error loss: top level
        # squaredErrorLoss_backward = squaredErrorLoss_backward + (decodedData - doubleDecodedEncodedData).pow(2).mean(dim=-1).mean(dim=1)
        
        # Calculate the squared error loss: top level
        # squaredErrorLoss_forward = squaredErrorLoss_forward + (encodedData - doubleEncodedDecodedData).pow(2).mean(dim=-1).mean(dim=1)

        
        # Give compression a higher loss.
        if compressingDataFlag:
            squaredErrorLoss_forward = squaredErrorLoss_forward*2
        else:
            squaredErrorLoss_backward = squaredErrorLoss_backward*2
        
        # Return a summation of the losses.
        return squaredErrorLoss_forward + squaredErrorLoss_backward + squaredErrorLoss_middle
    
    def updateLossValues(self, originalData, encodedData, compressingDataFlag, signalEncodingLayerLoss):
        # Keep tracking on the normalization loss through each loop.
        # normalizationLayerLoss = self.calculateStandardizationLoss(encodedData, finalEncodingLayer, expectedMean = 0, expectedStandardDeviation = 1, dim=-1)
        layerLoss = self.calculateEncodingLoss(originalData, encodedData, compressingDataFlag)
        
        # Update the signal encoding layer loss.
        signalEncodingLayerLoss = signalEncodingLayerLoss + layerLoss
        
        return signalEncodingLayerLoss
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Data Organization ------------------------- #
    
    def getMaxSignals_Expansion(self, numSignals):
        numActiveSignals = self.getMaxActiveSignals_Expansion(numSignals)
        numFrozenSignals = numSignals - numActiveSignals
        
        return numActiveSignals*self.expansionFactor + numFrozenSignals
    
    def getMaxSignals_Compression(self, numSignals):
        numActiveSignals = self.getMaxActiveSignals_Compression(numSignals)
        numFrozenSignals = numSignals - numActiveSignals
                
        return numActiveSignals/self.expansionFactor + numFrozenSignals
    
    def getMaxActiveSignals_Expansion(self, numSignals):
        return numSignals - (numSignals%self.numCompressedSignals)
    
    def getMaxActiveSignals_Compression(self, numSignals):
        return numSignals - (numSignals%self.numExpandedSignals)
    
    def seperateActiveData(self, inputData, targetNumSignals):
        # Extract the signal number.
        numSignals = inputData.size(1)
                
        # If we are upsampling the signals as much as I can..
        if numSignals*self.expansionFactor <= targetNumSignals:
            # Upsample the max number of signals.
            numActiveSignals = self.getMaxActiveSignals_Expansion(numSignals)
        
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
            numActiveSignals = self.getMaxActiveSignals_Compression(numSignals)
            
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
    
    def unpairSignals(self, inputData, targetNumSignals):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = inputData.shape
                
        # Seperate out the active and frozen data.
        activeData, frozenData = self.seperateActiveData(inputData, targetNumSignals)
        # activeData dimension: batchSize, numActiveSignals, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension
                
        # Unpair up the signals.
        numUnpairedBatches = int(activeData.size(0)*activeData.size(1)/self.numCompressedSignals)
        unpairedData = activeData.view(numUnpairedBatches, self.numCompressedSignals, signalDimension) # Create a channel for the CNN.
        # unpairedData dimension: batchSize*numSignalPairs, 1, signalDimension
        
        return unpairedData, frozenData
    
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
    
    
    def expansionAlgorithm(self, originalData, targetNumSignals):
        # Unpair the signals with their neighbors.
        unpairedData, frozenData = self.unpairSignals(originalData, targetNumSignals)
        # activeData dimension: batchSize*numSignalPairs, 1, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension
                    
        # Increase the number of signals.
        expandedData = self.signalExpansion(unpairedData)
        # expandedData dimension: batchSize*numSignalPairs, 2, signalDimension
        
        # Recompile the signals to their original dimension.
        signalData = self.recompileSignals(expandedData, frozenData)
        # signalData dimension: batchSize, 2*numSignalPairs + numFrozenSignals, signalDimension
        
        return originalData, signalData
        
    def compressionAlgorithm(self, originalData, targetNumSignals):
        # Pair up the signals with their neighbors.
        pairedData, frozenData = self.pairSignals(originalData, targetNumSignals)
        # pairedData dimension: batchSize*numSignalPairs, 2, signalDimension
        # frozenData dimension: batchSize, numFrozenSignals, signalDimension
        
        # Reduce the number of signals.
        reducedPairedData = self.signalCompression(pairedData)
        # reducedPairedData dimension: batchSize*numSignalPairs, 1, signalDimension

        # Recompile the signals to their original dimension.
        signalData = self.recompileSignals(reducedPairedData, frozenData)
        # signalData dimension: batchSize, numSignalPairs + numFrozenSignals, signalDimension
        
        return originalData, signalData
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numExpandedSignals = 6):
        super(signalEncoding, self).__init__(signalDimension, numExpandedSignals)        
                        
    def forward(self, signalData, targetNumSignals = 64, signalEncodingLayerLoss = 0, calculateLoss = True):
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
            originalData, signalData = self.compressionAlgorithm(signalData, targetNumSignals)

            # Keep track of the signal's at each iteration.
            numSignalPath.append(signalData.size(1))
            
            # Aggregate all the layer loss values together.
            if calculateLoss: signalEncodingLayerLoss = self.updateLossValues(originalData, signalData, True, signalEncodingLayerLoss)

        # ------------------- Signal Expansion Algorithm ------------------- # 
            
        # While we have too many signals to process.
        while signalData.size(1) < targetNumSignals:
            originalData, signalData = self.expansionAlgorithm(signalData, targetNumSignals)
                        
            # Keep track of the signal's at each iteration.
            numSignalPath.append(signalData.size(1))
            
            # Aggregate all the layer loss values together.
            if calculateLoss: signalEncodingLayerLoss = self.updateLossValues(originalData, signalData, False, signalEncodingLayerLoss)
                                
        # ------------------------------------------------------------------ # 
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
        
        
