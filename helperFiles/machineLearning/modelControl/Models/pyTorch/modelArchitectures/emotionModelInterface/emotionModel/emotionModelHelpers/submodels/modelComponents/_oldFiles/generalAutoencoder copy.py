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
sys.path.append(os.path.dirname(__file__) + "/autoencoderHelpers/")
import _autoencoderModules

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class generalAutoencoderBase(_autoencoderModules.autoencoderModules):
    def __init__(self, accelerator = None):
        super(generalAutoencoderBase, self).__init__(accelerator, compressionFactor = 2, expansionFactor = 2)        
        # Allow the final signals to reach the final variance. 
        self.initialTransformation = self.minorSubspaceTransformationInitial(inChannel = 1, numMidChannels = 4)
        self.finalTransformation = self.minorSubspaceTransformationFinal(inChannel = 1, numMidChannels = 4)

        # Autoencoder modules for preprocessing.
        self.compressDataCNN_preprocessing = self.signalEncodingModule_preprocess(inChannel = 1, channelIncrease = 8)
        self.expandDataCNN_preprocessing = self.signalEncodingModule_preprocess(inChannel = 1, channelIncrease = 8)
        # Autoencoder modules for postprocessing.
        self.compressDataCNN_postprocessing = self.signalEncodingModule_postprocess(inChannel = 1, channelIncrease = 8)
        self.expandDataCNN_postprocessing = self.signalEncodingModule_postprocess(inChannel = 1, channelIncrease = 8)
                
        # Linear parameters to account for dilation.
        self.raisingParams = torch.nn.Parameter(torch.randn(2))
        self.loweringParams = torch.nn.Parameter(torch.randn(2))
        # Specify the ladder operators to account for dilation.
        self.raisingModule = self.ladderModules(inChannel = 1, channelIncrease = 8)
        self.loweringModule = self.ladderModules(inChannel = 1, channelIncrease = 8)
        
    # ---------------------- Encoder-Specific Modules ---------------------- #
    
    def signalEncodingModule_preprocess(self, inChannel = 1, channelIncrease = 8):
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
    
    def signalEncodingModule_postprocess(self, inChannel = 1, channelIncrease = 8):
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
    
    def minorSubspaceTransformationInitial(self, inChannel = 1, numMidChannels = 4):        
        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
        )
    
    def minorSubspaceTransformationFinal(self, inChannel = 1, numMidChannels = 4):        
        return nn.Sequential( 
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [inChannel, numMidChannels, numMidChannels, inChannel], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = 1),
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
    
    # --------------------------- Encoder Methods -------------------------- #
  
    def expansionAlgorithm(self, compressedData, nextSequenceLength, initialSequenceLength):  
        # Prepare the data for signal reduction.
        # processedData = compressedData
        processedData = self.raisingOperator(compressedData, initialSequenceLength)
        processedData = self.expandDataCNN_preprocessing(processedData)
        
        # Convolution architecture: change signal's dimension.
        processedData = nn.Upsample(size=nextSequenceLength, mode='linear', align_corners=True)(processedData)
        
        # Process the reduced data.
        processedData = self.expandDataCNN_postprocessing(processedData)
        encodedData = self.loweringOperator(processedData, initialSequenceLength)
        # encodedData = processedData
        
        # Free up memory.
        freeMemory()
        
        return compressedData, encodedData
    
    def compressionAlgorithm(self, expandedData, nextSequenceLength, initialSequenceLength): 
        # Prepare the data for signal reduction.
        # processedData = expandedData
        processedData = self.raisingOperator(expandedData, initialSequenceLength)
        processedData = self.compressDataCNN_preprocessing(processedData)
        
        # Convolution architecture: change signal's dimension.
        processedData = nn.Upsample(size=nextSequenceLength, mode='linear', align_corners=True)(processedData)
        
        # Process the reduced data.
        processedData = self.compressDataCNN_postprocessing(processedData)
        encodedData = self.loweringOperator(processedData, initialSequenceLength)
        # encodedData = processedData
    
        # Free up memory.
        freeMemory()

        return expandedData, encodedData
    
    def raisingOperator(self, inputData, initialSequenceLength):
        # Learn how to scale the data given the time dilation.
        dilationFraction = inputData.size(2) / initialSequenceLength  # Can be less than or greater than 0 (it depends on the starting position)
        processedData = (self.raisingParams[0] + self.raisingParams[1] * dilationFraction) * inputData
        
        # Non-linear learning.
        processedData = self.raisingModule(processedData) + inputData
                
        return processedData

    def loweringOperator(self, inputData, initialSequenceLength):        
        # Learn how to scale the data given the time dilation.
        dilationFraction = inputData.size(2) / initialSequenceLength  # Can be less than or greater than 0 (it depends on the starting position)
        processedData = (self.loweringParams[0] + self.loweringParams[1] * dilationFraction) * inputData
        
        # Non-linear learning.
        processedData = self.loweringModule(processedData) + inputData
        
        return processedData
                
    # ---------------------------- Loss Methods ---------------------------- #   
    
    def calculateEncodingLoss(self, originalData, encodedData, initialSequenceLength):
        # originalData    decodedData
        #          \         /
        #          encodedData
        
        # Setup the variables for signal encoding.
        originalSignalDimension = originalData.size(2)
        # reverseEncodingInd = self.numMaxEncodings - 1 - autoencoderLayerInd
        
        # Add noise to the encoded data before the reverse operation.
        decodedData = self.dataInterface.addNoise(encodedData, trainingFlag = True, noiseSTD = 0.005)
        
        # Reconstruct the original data.
        if encodedData.size(2) < originalSignalDimension:
            while decodedData.size(2) != originalSignalDimension:
                nextSequenceLength = self.getNextSequenceLength(decodedData.size(2), originalSignalDimension)
                _, decodedData = self.expansionAlgorithm(decodedData, nextSequenceLength, initialSequenceLength)
        else:
            while decodedData.size(2) != originalSignalDimension:
                nextSequenceLength = self.getNextSequenceLength(decodedData.size(2), originalSignalDimension)
                _, decodedData = self.compressionAlgorithm(decodedData, nextSequenceLength, initialSequenceLength)
        # Assert the integrity of the expansions/compressions.
        assert decodedData.size(2) == originalSignalDimension
        
        # Calculate the squared error loss of this layer of compression/expansion.
        squaredErrorLoss_forward = (originalData - decodedData).pow(2).mean(dim=-1).mean(dim=1)
        print("\tAutoencoder reverse operation loss:", squaredErrorLoss_forward.mean().item())
        
        return squaredErrorLoss_forward
    
    def updateLossValues(self, originalData, encodedData, autoencoderLayerLoss, initialSequenceLength):
        # It is a waste to go backward if we lost the initial signal.
        if 0.25 < autoencoderLayerLoss.mean(): return autoencoderLayerLoss

        # Keep tracking of the loss through each loop.
        layerLoss = self.calculateEncodingLoss(originalData, encodedData, initialSequenceLength)
                
        return autoencoderLayerLoss + layerLoss
    
    # ---------------------------------------------------------------------- #   
    
class generalAutoencoder(generalAutoencoderBase):
    def __init__(self, accelerator = None):
        super(generalAutoencoder, self).__init__(accelerator) 
                        
    def forward(self, signalData, targetSequenceLength = 64, initialSequenceLength = 300, autoencoderLayerLoss = None, calculateLoss = True):
        """ The shape of signalData: (batchSize, numSignals, compressedLength) """
        # Setup the variables for signal encoding.
        batchSize, numSignals, inputSequenceLength = signalData.size()
        numSignalPath = [inputSequenceLength] # Keep track of the signal's length at each iteration.
        
        # Initialize a holder for the loss values.
        if autoencoderLayerLoss == None: autoencoderLayerLoss = torch.zeros((batchSize*numSignals), device=signalData.device)
        
        # Reshape the data to the expected input into the CNN architecture.
        signalData = signalData.view(batchSize * numSignals, 1, inputSequenceLength) # Seperate out indivisual signals.
        # signalData dimension: batchSize*numSignals, 1, sequenceLength 
        
        # ------------- Signal Compression/Expansion Algorithm ------------- #       
        
        # While we have the incorrect number of signals.
        while targetSequenceLength != signalData.size(2):
            nextSequenceLength = self.getNextSequenceLength(signalData.size(2), targetSequenceLength)
            
            # Compress the signals to the target length.
            if targetSequenceLength < signalData.size(2): 
                originalData, signalData = self.compressionAlgorithm(signalData, nextSequenceLength, initialSequenceLength)
            
            # Expand the signals to the target length.
            else: originalData, signalData = self.expansionAlgorithm(signalData, nextSequenceLength, initialSequenceLength)
            
            # Keep track of the error during each compression/expansion.
            if calculateLoss: autoencoderLayerLoss = self.updateLossValues(originalData, signalData, autoencoderLayerLoss, initialSequenceLength)
                                
            # Keep track of the signal's at each iteration.
            numSignalPath.append(signalData.size(2))
                                            
        # ------------------------------------------------------------------ # 
        
        # Seperate put each signal into its respective batch.
        encodedSignalData = signalData.view(batchSize, numSignals, targetSequenceLength) 
        # compressedData dimension: batchSize, numSignals, compressedLength
                
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

