# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import gc
import sys
import time
import math

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# Import helper models
sys.path.append(os.path.dirname(__file__) + "/modelHelpers/")
import _convolutionalHelpers

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class generalAutoencoderBase(_convolutionalHelpers.convolutionalHelpers):
    def __init__(self):
        super(generalAutoencoderBase, self).__init__()
        # Autoencoder modules for preprocessing.
        self.compressDataCNN_preprocessing = self.signalEncodingModule_preprocess(inChannel = 1, channelIncrease = 8)
        self.expandDataCNN_preprocessing = self.signalEncodingModule_preprocess(inChannel = 1, channelIncrease = 8)
        
        # Autoencoder modules for postprocessing.
        self.compressDataCNN_postprocessing = self.signalEncodingModule_postprocess(inChannel = 1, channelIncrease = 8)
        self.expandDataCNN_postprocessing = self.signalEncodingModule_postprocess(inChannel = 1, channelIncrease = 8)
        
        # Map the initial signals into a common subspace.
        self.initialTransformation = self.minorSubspaceTransformationInitial(inChannel = 1, numMidChannels = 4)
        self.finalTransformation = self.minorSubspaceTransformationFinal(inChannel = 1, numMidChannels = 4)
        
        # Signal compression/expansion parameters.
        self.compressionFactor = 1.5
        self.expansionFactor = 1.2
        
        # Initialize linking module holders.
        self.compressionLinkingModules = nn.ModuleList()
        self.expansionLinkingModules = nn.ModuleList()
        
        self.numMaxCompressions = len(self.simulateEncoding(300, 64))
        for encodingLayer in range(self.numMaxEncodings):
            self.compressionLinkingModules.append(self.autoencoderLinkingModule(inChannel = 1, numMidChannels = 8))
            
        self.numMaxExpansions = len(self.simulateEncoding(64, 300))
        for encodingLayer in range(self.numMaxEncodings):
            self.expansionLinkingModules.append(self.autoencoderLinkingModule(inChannel = 1, numMidChannels = 8))
    
    
    def signalEncodingModule_preprocess(self, inChannel = 1, channelIncrease = 4):
        # Calculate the number of intermediate channels.
        numExpandedChannels = int(channelIncrease*inChannel)

        return nn.Sequential(
            # Convolution architecture: signal-specific feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numExpandedChannels, numExpandedChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numExpandedChannels, numExpandedChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numExpandedChannels, numExpandedChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
        )
    
    def signalEncodingModule_postprocess(self, inChannel = 1, channelIncrease = 4):
        # Calculate the number of intermediate channels.
        numExpandedChannels = int(channelIncrease*inChannel)

        return nn.Sequential(                        
            # Convolution architecture: signal-specific feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numExpandedChannels, numExpandedChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numExpandedChannels, numExpandedChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numExpandedChannels, numExpandedChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
        )
    
    def autoencoderLinkingModule(self, inChannel = 1, numMidChannels = 8):        
        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
        )
    
    def minorSubspaceTransformationInitial(self, inChannel = 1, numMidChannels = 8):        
        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, inChannel, inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
        )
    
    def minorSubspaceTransformationFinal(self, inChannel = 1, numMidChannels = 8):        
        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, inChannel, inChannel, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
        )
    
    def simulateEncoding(self, initialSequenceLength, targetSequenceLength):
        encodingPath = [initialSequenceLength]
        
        # While we havent converged to the target length.
        while initialSequenceLength != targetSequenceLength:
            # Simulate how the sequence will change during the next iteration.
            initialSequenceLength = self.getNextSequenceLength(initialSequenceLength, targetSequenceLength)
            
            # Store the path the sequence takes.
            encodingPath.append(initialSequenceLength)
        
        return encodingPath
    
    def getLinkingModelInd(self, currentSequenceLength, targetSequenceLength):
        numStepsAway = len(self.simulateEncoding(currentSequenceLength, targetSequenceLength)) - 1
        
        return max(0, numStepsAway - 1)
                    
    def getNextSequenceLength(self, initialSequenceLength, targetSequenceLength):
        # If we are a factor of 2 away from the target length
        if initialSequenceLength*self.expansionFactor < targetSequenceLength:
            return math.floor(initialSequenceLength*self.expansionFactor)
        
        # If we are less than halfway to the target length
        elif initialSequenceLength <= targetSequenceLength:
            return targetSequenceLength
        
        # If we are a factor of 2 away from the target length
        elif targetSequenceLength <= initialSequenceLength/self.compressionFactor:
            return math.ceil(initialSequenceLength/self.compressionFactor)
        
        # If we are less than halfway to the target length
        elif targetSequenceLength <= initialSequenceLength:
            return targetSequenceLength  
        
    def expansionAlgorithm(self, compressedData, nextSequenceLength, layerInd):   
        # Apply a layer-specific encoding.
        processedData = self.expansionLinkingModules[layerInd](compressedData)
        
        # Prepare the data for signal reduction.
        processedData = self.expandDataCNN_preprocessing(processedData)
        
        # Convolution architecture: change signal's dimension.
        processedData = nn.Upsample(size=nextSequenceLength, mode='linear', align_corners=True)(processedData)
        
        # Process the reduced data.
        encodedData = self.expandDataCNN_postprocessing(processedData)
                
        return compressedData, encodedData
    
    def compressionAlgorithm(self, expandedData, nextSequenceLength, layerInd): 
        # Apply a layer-specific encoding.
        processedData = self.compressionLinkingModules[layerInd](expandedData)
        
        # Prepare the data for signal reduction.
        processedData = self.compressDataCNN_preprocessing(processedData)
        
        # Convolution architecture: change signal's dimension.
        processedData = nn.Upsample(size=nextSequenceLength, mode='linear', align_corners=True)(processedData)
        
        # Process the reduced data.
        encodedData = self.compressDataCNN_postprocessing(processedData)
        
        return expandedData, encodedData
                
    # ---------------------------- Loss Methods ---------------------------- #   
    
    def calculateEncodingLoss(self, originalData, encodedData, compressedDataFlag, autoencoderLayerInd):
        # originalData  encodedDecodedOriginalData
        #          \         /
        #          encodedData
        
        # Setup the variables for signal encoding.
        originalSignalDimension = originalData.size(2)
        reverseEncodingInd = self.numMaxEncodings - 1 - autoencoderLayerInd
        
        noisyEncodedData = encodedData + torch.randn_like(encodedData, device=encodedData.device) * 0.05
        
        # Reverse operation
        if compressedDataFlag: 
            _, encodedDecodedOriginalData = self.expansionAlgorithm(noisyEncodedData, originalSignalDimension, reverseEncodingInd)
        else:
            _, encodedDecodedOriginalData = self.compressionAlgorithm(noisyEncodedData, originalSignalDimension, reverseEncodingInd)
        # Assert the integrity of the expansions/compressions.
        assert encodedDecodedOriginalData.size(2) == originalSignalDimension
        
        # Calculate the squared error loss of this layer of compression/expansion.
        squaredErrorLoss_forward = (originalData - encodedDecodedOriginalData).pow(2).mean(dim=-1).mean(dim=1)
        print("Reverse operation loss:", squaredErrorLoss_forward.mean().item())
        
        # Compile all the loss information together into one value.
        return squaredErrorLoss_forward #+ squaredErrorLoss_middle + squaredErrorLoss_backward
    
    def updateLossValues(self, originalData, encodedData, compressedDataFlag, autoencoderLayerLoss, autoencoderLayerInd):
        # Keep tracking of the loss through each loop.
        layerLoss = self.calculateEncodingLoss(originalData, encodedData, compressedDataFlag, autoencoderLayerInd)
        if autoencoderLayerInd == 0: layerLoss = layerLoss*10
        
        return autoencoderLayerLoss + layerLoss
    
class generalAutoencoder(generalAutoencoderBase):
    def __init__(self):
        super(generalAutoencoder, self).__init__() 
        
    def encodingInterface(self, signalData, transformation):
        # Extract the incoming data's dimension.
        batchSize, numSignals, signalDimension = signalData.size()
        
        # Reshape the data to process each signal seperately.
        signalData = signalData.view(batchSize*numSignals, 1, signalDimension)
        
        # Apply a CNN network.
        signalData = transformation(signalData)
        
        # Return to the initial dimension of the input.
        signalData = signalData.view(batchSize, numSignals, signalDimension)
        
        return signalData
                        
    def forward(self, signalData, targetSequenceLength = 64, autoencoderLayerLoss = 0, calculateLoss = True):
        """ The shape of signalData: (batchSize, numSignals, compressedLength) """
        # Setup the variables for signal encoding.
        batchSize, numSignals, signalDimension = signalData.size()
        numSignalPath = [signalDimension] # Keep track of the signal's length at each iteration.
        
        # Initialize a holder for the loss values.
        reshapedAutoencoderLoss = torch.zeros((batchSize*numSignals), device=signalData.device)
        
        # Reshape the data to the expected input into the CNN architecture.
        signalData = signalData.view(batchSize * numSignals, 1, signalDimension) # Seperate out indivisual signals.
        # signalData dimension: batchSize*numSignals, 1, sequenceLength 
        
        # ------------- Signal Compression/Expansion Algorithm ------------- #       
        
        autoencoderLayerInd = 0
        # While we have the incorrect number of signals.
        while targetSequenceLength != signalData.size(2):
            nextSequenceLength = self.getNextSequenceLength(signalData.size(2), targetSequenceLength)
            compressedDataFlag = targetSequenceLength < signalData.size(2)
            
            # Compress the signals down to the targetSequenceLength.
            if compressedDataFlag: originalData, signalData = self.compressionAlgorithm(signalData, nextSequenceLength, autoencoderLayerInd)
            
            # Expand the signals up to the targetSequenceLength.
            else: originalData, signalData = self.expansionAlgorithm(signalData, nextSequenceLength, autoencoderLayerInd)
                        
            # Keep track of the error during each compression/expansion.
            if calculateLoss: reshapedAutoencoderLoss = self.updateLossValues(originalData, signalData, compressedDataFlag, reshapedAutoencoderLoss, autoencoderLayerInd)
        
            # Keep track of the signal's at each iteration.
            numSignalPath.append(signalData.size(2))
            autoencoderLayerInd += 1
            
            # Free up memory.
            gc.collect(); torch.cuda.empty_cache();
                                
        # ------------------------------------------------------------------ # 
        
        # Seperate put each signal into its respective batch.
        encodedSignalData = signalData.view(batchSize, numSignals, targetSequenceLength) 
        reshapedAutoencoderLoss = reshapedAutoencoderLoss.view(batchSize, numSignals).mean(dim=1)
        # compressedData dimension: batchSize, numSignals, compressedLength
        
        # Update the autoencoder loss.
        autoencoderLayerLoss = autoencoderLayerLoss + reshapedAutoencoderLoss
        
        # Assert the integrity of the expansion/compression.
        if numSignals != targetSequenceLength:
            assert all(numSignalPath[i] <= numSignalPath[i + 1] for i in range(len(numSignalPath) - 1)) \
                or all(numSignalPath[i] >= numSignalPath[i + 1] for i in range(len(numSignalPath) - 1)), "List is not sorted up or down"
        
        # Remove the target signal from the path.
        numSignalPath.pop()
        
        return encodedSignalData, numSignalPath, autoencoderLayerLoss
        
    def printParams(self, numSignals = 50, signalDimension = 300):
        # generalAutoencoder().to('cpu').printParams(numSignals = 100, signalDimension = 300)
        t1 = time.time()
        summary(self, (numSignals, signalDimension))
        t2 = time.time(); print(t2-t1)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

