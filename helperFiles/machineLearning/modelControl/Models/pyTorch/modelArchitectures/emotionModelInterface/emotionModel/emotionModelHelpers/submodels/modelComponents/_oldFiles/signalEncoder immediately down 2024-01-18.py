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

# -------------------------------------------------------------------------- #
# -------------------------- Shared Architecture --------------------------- #

class signalEncoderBase(_convolutionalHelpers.convolutionalHelpers):
    def __init__(self, signalDimension = 64, numEncodedSignals = 32, numExpandedSignals = 2, plotSignalEncoding = False):
        super(signalEncoderBase, self).__init__()        
        # General shape parameters.
        self.numEncodedSignals = numEncodedSignals  # The final number of signals to accept, encoding all signal information.
        self.signalDimension = signalDimension  # The incoming dimension of each signal.
        
        # Compression/Expansion parameters.
        self.numExpandedSignals = numExpandedSignals        # The final number of signals in any expansion
        self.numCompressedSignals = numExpandedSignals - 1  # The final number of signals in any compression.
        self.expansionFactor = self.numExpandedSignals/self.numCompressedSignals  # The percent expansion.
        # Assert the integrity of the input parameters.
        assert self.numExpandedSignals - self.numCompressedSignals == 1, "You should only gain 1 channel when expanding or else you may overshoot."
                
        # Map the signals into a common subspace.
        self.initialTransformation = self.minorSubspaceTransformation(inChannel = 1, channelIncrease = 32, numGroups = 1)
        self.finalTransformation = self.minorSubspaceTransformation(inChannel = 1, channelIncrease = 32, numGroups = 1)
        
        # self.expandChannels_principleComp = self.signalEncodingModule(inChannel = self.numCompressedSignals, outChannel = self.numExpandedSignals, channelIncrease = 16)      
        # self.expandChannels_projectedSigs = self.subspaceTransformation(inChannel = self.numCompressedSignals, channelIncrease = 16, numGroups = 1)
        
        self.expandChannels_principleComp = self.signalEncodinANN(inputDimension = self.signalDimension, outputDimension = self.signalDimension)      
        self.expandChannels_projectedSigs = self.signalEncodingModule(inChannel = self.numCompressedSignals, outChannel = self.numExpandedSignals, channelIncrease = 32)  
        
        # Create model for learning local information.

        self.finalizeCompressionInfo = self.minorSubspaceTransformation(inChannel = self.numEncodedSignals, channelIncrease = 1, numGroups = 1)
        self.finalizeExpansionInfo = self.minorSubspaceTransformation(inChannel = self.numExpandedSignals, channelIncrease = 32, numGroups = 1)
        
        # Create model for learning complex from the channels.
        self.embedPrincipleComponents = self.signalEncodinANN(inputDimension = self.numEncodedSignals, outputDimension = self.numEncodedSignals*self.signalDimension)

        # Initialize the pooling layers to upsample/downsample
        self.downsampleChannels = nn.Upsample(size=self.numEncodedSignals, mode='linear', align_corners=True)
        self.upsampleChannels = nn.Upsample(size=self.numExpandedSignals, mode='linear', align_corners=True)
        
        if plotSignalEncoding:
            plt.plot(self.channelEncoding.detach().cpu().T[:, 0:25:2])
            plt.show()
            
            plt.plot(self.channelEncoding.detach().cpu().T[:, -25::2])
            plt.show()
    
    def signalExpansion(self, compressedData):   
        # Predict the projected signals using principle components info.
        learnedPrincipleComponents = self.expandChannels_principleComp(compressedData) + compressedData
        estimatedProjectedSig = self.expandChannels_projectedSigs(learnedPrincipleComponents)
                
        # Crude estimation of how the channels will expand.
        averagePrediction = self.upsampleChannels(compressedData.transpose(1, 2)).transpose(1, 2)
        
        # Add up all the predictions.
        expandedData = estimatedProjectedSig + averagePrediction
        
        # Synthesize the information from both algorithms.
        expandedData = self.finalizeExpansionInfo(expandedData)
                
        return expandedData
    
    def compressionAlgorithm(self, signalData, calculateLoss = True):
        # Perform the optimal compression via PCA and embed channel information (for reconstruction).
        pcaProjection, principal_components = self.svdCompression(signalData, self.numEncodedSignals, standardizeSignals = True)
        encodedProjection = self.embedChannelInfo(pcaProjection, principal_components)
        # principal_components dimension: batchSize, numOriginalSignals, numComponents
        # pcaProjection dimension: batchSize, numComponents, signalDimension

        if calculateLoss:
            # Loss for PCA reconstruction
            pcaReconstruction = torch.matmul(principal_components, pcaProjection)
            pcaReconstructionLoss = (signalData - pcaReconstruction).pow(2).mean(dim=-1).mean(dim=1)
            print("Optimal Compression Loss:", pcaReconstructionLoss.mean().item())
        else: pcaReconstructionLoss = 0
        
        # Crude estimation of how the channels will compress with spatial significance.
        averagePrediction = self.downsampleChannels(signalData.transpose(1, 2)).transpose(1, 2)
        
        # Add up and synthesize the compressed information.
        compressedData = encodedProjection + averagePrediction
        compressedData = self.finalizeCompressionInfo(compressedData)
        
        return compressedData, pcaReconstructionLoss
        
    def pcaCompression(self, signals, numComponents, standardizeSignals = True):
        # Extract the incoming data's dimension.
        batch_size, num_signals, signal_dimension = signals.shape
        
        if standardizeSignals:
            # Calculate means and standard deviations
            signalMeans = signals.mean(dim=-1, keepdim=True)
            signalSTDs = signals.std(dim=-1, keepdim=True) + 1e-5
            # Standardize the signals
            standardized_signals = (signals - signalMeans) / signalSTDs
        else:
            standardized_signals = signals

        # Calculate the covariance matrix
        covariance_matrix = torch.matmul(standardized_signals, standardized_signals.transpose(1,2)) / (signal_dimension - 1)
        
        # Ensure covariance matrix is symmetric to avoid numerical issues
        covariance_matrix = (covariance_matrix + covariance_matrix.transpose(1, 2)) / 2
        # Add a small value to the diagonal for numerical stability (regularization)
        regularization_term = 1e-5 * torch.eye(num_signals, device=signals.device)
        covariance_matrix += regularization_term

        # Perform eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # Select the top n_components eigenvectors
        principal_components = eigenvectors[:, :, -numComponents:].contiguous()

        # Project the original signals to the new subspace
        projected_signals = torch.matmul(principal_components.transpose(1, 2), standardized_signals)
        
        return projected_signals, principal_components
    
    def svdCompression(self, signals, numComponents, standardizeSignals = True):
        # Extract the incoming data's dimension.
        batch_size, num_signals, signal_dimension = signals.shape
        
        if standardizeSignals:
            # Calculate means and standard deviations
            signalMeans = signals.mean(dim=-1, keepdim=True)
            signalSTDs = signals.std(dim=-1, keepdim=True) + 1e-5
            # Standardize the signals
            standardized_signals = (signals - signalMeans) / signalSTDs
        else:
            standardized_signals = signals
        
        # Perform Singular Value Decomposition
        U, S, V = torch.linalg.svd(standardized_signals)

        # Select the top n_components eigenvectors
        principal_components = U[:, :, :numComponents]
        
        # Project the original signals to the new subspace
        projected_signals = torch.matmul(principal_components.transpose(1, 2), standardized_signals)
        
        return projected_signals, principal_components
    
    def embedChannelInfo(self, projected_signals, principal_components):
        # Extract the incoming data's dimension.
        batchSize, numComponents, signalDimension = projected_signals.size()
        batchSize, numOriginalSignals, numComponents = principal_components.size()
        # Assert the integrity of the input parameters.
        assert numComponents == self.numEncodedSignals
        assert signalDimension == self.signalDimension

        # Project each signal's principle components onto the projected signal's dimension
        embeddedPrincipleComponents = self.embedPrincipleComponents(principal_components)
        # embeddedPrincipleComponents dimension: batchSize, numOriginalSignals, numComponents*signalDimension
        
        # Compile the informationn from each original signal.
        embeddedPrincipleComponents = embeddedPrincipleComponents.mean(1)
        embeddedPrincipleComponents = embeddedPrincipleComponents.view(batchSize, numComponents, signalDimension)
        # Add the principle component information to the projected signals.
        encodedProjection = projected_signals + embeddedPrincipleComponents
        # encodedProjection dimension: batchSize, numComponents, signalDimension
        
        return encodedProjection
    
    # ---------------------------------------------------------------------- #
    # ------------------- Machine Learning Architectures ------------------- #

    def signalEncodingModule(self, inChannel = 2, outChannel = 1, channelIncrease = 8):
        return nn.Sequential( 
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 2], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 3], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 2], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, channelIncrease*inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),

                # Convolution architecture: channel expansion
                self.deconvolutionalFilter(numChannels = [inChannel, 2*outChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [outChannel, 2*outChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
                self.deconvolutionalFilter_resNet(numChannels = [outChannel, 2*outChannel, outChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = [1, 1]),
        )
    
    def signalEncodinANN(self, inputDimension, outputDimension):        
        return nn.Sequential( 
                # Neural architecture: Layer 1.
                nn.Linear(inputDimension, outputDimension, bias=True),
                nn.SELU(),
            )
    
    def subspaceTransformation(self, inChannel = 1, channelIncrease = 8, numGroups = [1, 1, 1, 1, 1, 1, 1]):  
        if type(numGroups) != list: numGroups = [numGroups]*7
        numMidChannels = int(channelIncrease*inChannel)
        
        return nn.Sequential( 
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups[0]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 2], groups = numGroups[1]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 3], groups = numGroups[2]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups[3]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups[4]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups[5]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, inChannel, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups[6]),
        )
    
    def minorSubspaceTransformation(self, inChannel = 1, channelIncrease = 8, numGroups = [1, 1, 1, 1, 1]):  
        if type(numGroups) != list: numGroups = [numGroups]*5
        numMidChannels = int(channelIncrease*inChannel)
        
        return nn.Sequential( 
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups[0]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 2], groups = numGroups[1]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 3], groups = numGroups[2]),
                
                # Convolution architecture: feature engineering
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups[3]),
                self.deconvolutionalFilter_resNet(numChannels = [inChannel, numMidChannels, inChannel], kernel_sizes = [3, 3], dilations = [1, 1], groups = numGroups[4]),
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
    
    def updateLossValues(self, startingData, encodedData, signalEncodingLayerLoss):
        # Calculate the normalization loss.
        normalizationLayerLoss = self.calculateStandardizationLoss(encodedData, expectedMean = 0, expectedStandardDeviation = 1, dim=-1)
        
        # Make sure you can recompress and expand the data again (the same way).
        predictedStartingData, pcaReconstructionLoss = self.compressionAlgorithm(encodedData, calculateLoss = True)
        reEncodedData, _ = self.forward(predictedStartingData, targetNumSignals = encodedData.size(1), calculateLoss = False)
        
        # Check that we can compress and expand to the same location.
        recompressionLoss = (startingData - predictedStartingData).pow(2).mean(dim=-1).mean(dim=1)
        reexpansionLoss = (encodedData - reEncodedData).pow(2).mean(dim=-1).mean(dim=1)

        # Update the signal encoding layer loss.
        signalEncodingLayerLoss = signalEncodingLayerLoss + 10*recompressionLoss + 10*reexpansionLoss
        signalEncodingLayerLoss = signalEncodingLayerLoss + 0.1*normalizationLayerLoss + 5*pcaReconstructionLoss
        print(normalizationLayerLoss.mean().item(), reexpansionLoss.mean().item(), recompressionLoss.mean().item(), pcaReconstructionLoss.mean().item())
        
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
        
        return signalData
    
# -------------------------------------------------------------------------- #
# -------------------------- Encoder Architecture -------------------------- #

class signalEncoding(signalEncoderBase):
    def __init__(self, signalDimension = 64, numEncodedSignals = 32, numExpandedSignals = 3):
        super(signalEncoding, self).__init__(signalDimension, numEncodedSignals, numExpandedSignals) 
        
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
                        
    def forward(self, signalData, targetNumSignals = 32, calculateLoss = True):
        """ The shape of signalData: (batchSize, numSignals, compressedLength) """
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = signalData.size()

        # Assert that we have the expected data format.
        assert signalDimension == self.signalDimension, f"You provided a signal of length {signalDimension}, but we expected {self.signalDimension}."
        assert self.numCompressedSignals <= targetNumSignals, f"At the minimum, we cannot go lower than compressed signal batch. You provided {targetNumSignals} signals."
        assert self.numCompressedSignals <= numSignals, f"We cannot compress or expand if we dont have at least the compressed signal batch. You provided {numSignals} signals."
        
        # Setup the training variables.
        startingData = signalData.clone()
        signalEncodingLayerLoss = 0

        # ------------------ Signal Compression Algorithm ------------------ #             
        
        # While we have too many signals.
        while targetNumSignals < signalData.size(1):
            # Compress the signals down to the targetNumSignals.
            signalData, pcaReconstructionLoss = self.compressionAlgorithm(signalData, calculateLoss)
            assert targetNumSignals == signalData.size(1), f"We should have done the compression in one step: {targetNumSignals}, {signalData.size(1)}"
            
            # Keep track of the error during each compression.
            if calculateLoss: signalEncodingLayerLoss = signalEncodingLayerLoss + 10*pcaReconstructionLoss
            
        # ------------------- Signal Expansion Algorithm ------------------- # 
                            
        # While we have too many signals.
        while signalData.size(1) < targetNumSignals:
            # Expand the signals up to the targetNumSignals.
            signalData = self.expansionAlgorithm(signalData, targetNumSignals)
            
            # Keep track of the error during each expansion.
            if calculateLoss: signalEncodingLayerLoss = self.updateLossValues(startingData, signalData, signalEncodingLayerLoss)
                                            
        # ------------------------------------------------------------------ # 
        
        return signalData, signalEncodingLayerLoss
        
    def printParams(self, numSignals = 50):
        # signalEncoding(signalDimension = 64, numEncodedSignals = 32, numExpandedSignals = 2).to('cpu').printParams(numSignals = 4)
        t1 = time.time()
        summary(self, (numSignals, self.signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
        
        
        
        
        
        
        
