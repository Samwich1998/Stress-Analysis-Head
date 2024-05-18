
# General
import time

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# Import files for machine learning
from .autoencoderHelpers.autoencoderModules import autoencoderModules


class generalAutoencoderBase(autoencoderModules):
    def __init__(self, accelerator, compressionFactor, expansionFactor):
        super(generalAutoencoderBase, self).__init__(accelerator, compressionFactor, expansionFactor)
        # Autoencoder modules for preprocessing.
        self.compressDataCNN_preprocessing = self.signalEncodingModule()
        self.expandDataCNN_preprocessing = self.signalEncodingModule()
        # Autoencoder modules for postprocessing.
        self.compressDataCNN_postprocessing = self.signalEncodingModule()
        self.expandDataCNN_postprocessing = self.signalEncodingModule()

        # Linear parameters to account for dilation.
        self.raisingParams = torch.nn.Parameter(torch.randn(4))
        self.loweringParams = torch.nn.Parameter(torch.randn(4))
        # Specify the ladder operators to account for dilation.
        self.raisingModule = self.ladderModules()
        self.loweringModule = self.ladderModules()

        # Allow the final signals to reach the final variance.
        self.addVarianceAdjustment = self.varianceTransformation()
        self.removeVarianceAdjustment = self.varianceTransformation()

        # Allow the final signals to denoise after the sequential model.
        self.denoiseSignalsLast = self.denoiserModel()
        self.denoiseSignals = self.denoiserModel()

    def applyDenoiserLast(self, inputData):
        return self.encodingInterface(inputData, self.denoiseSignalsLast)

    def applyAutoencoderDenoiser(self, inputData):
        return self.encodingInterface(inputData, self.denoiseSignals)

    # --------------------------- Encoder Methods -------------------------- #

    def expansionAlgorithm(self, compressedData, nextSequenceLength, initialSequenceLength):
        # Prepare the data for signal reduction.
        processedData = self.raisingOperator(compressedData, initialSequenceLength)
        processedData = self.expandDataCNN_preprocessing(processedData)

        # Convolution architecture: change signal's dimension.
        processedData = nn.functional.interpolate(processedData, size=nextSequenceLength, mode='linear', align_corners=True, antialias=False)

        # Process the reduced data.
        processedData = self.expandDataCNN_postprocessing(processedData)
        encodedData = self.loweringOperator(processedData, initialSequenceLength)

        return encodedData

    def compressionAlgorithm(self, expandedData, nextSequenceLength, initialSequenceLength):
        # Prepare the data for signal reduction.
        processedData = self.raisingOperator(expandedData, initialSequenceLength)
        processedData = self.compressDataCNN_preprocessing(processedData)

        # Convolution architecture: change signal's dimension.
        processedData = nn.functional.interpolate(processedData, size=nextSequenceLength, mode='linear', align_corners=True, antialias=False)

        # Process the reduced data.
        processedData = self.compressDataCNN_postprocessing(processedData)
        encodedData = self.loweringOperator(processedData, initialSequenceLength)

        return encodedData

    def raisingOperator(self, inputData, initialSequenceLength):
        # Learn how to scale the data given the time dilation.
        dilationFraction = inputData.size(2) / initialSequenceLength  # Can be less than or greater than 0 (it depends on the starting position)
        processedData = inputData * (self.raisingParams[0] + self.raisingParams[1] * dilationFraction + self.raisingParams[2] * dilationFraction ** 2 + self.raisingParams[3] * dilationFraction * 3)

        # Non-linear learning.
        processedData = self.raisingModule(processedData) + inputData

        return processedData

    def loweringOperator(self, inputData, initialSequenceLength):
        # Learn how to scale the data given the time dilation.
        dilationFraction = inputData.size(2) / initialSequenceLength  # Can be less than or greater than 0 (it depends on the starting position)
        processedData = inputData * (self.loweringParams[0] + self.loweringParams[1] * dilationFraction + self.loweringParams[2] * dilationFraction ** 2 + self.loweringParams[3] * dilationFraction ** 3)

        # Non-linear learning.
        processedData = self.loweringModule(processedData) + inputData

        return processedData

    def adjustSignalVariance(self, inputData):
        return self.encodingInterface(inputData, self.addVarianceAdjustment)

    def unAdjustSignalVariance(self, inputData):
        return self.encodingInterface(inputData, self.removeVarianceAdjustment)

    # ---------------------------- Loss Methods ---------------------------- #   

    def calculateEncodingLoss(self, originalData, encodedData, initialSequenceLength):
        # originalData    decodedData
        #          \         /
        #          encodedData

        # Set up the variables for signal compression.
        originalSignalDimension = originalData.size(2)

        # Add noise to the encoded data before the reverse operation.
        decodedData = self.dataInterface.addNoise(encodedData, trainingFlag=True, noiseSTD=0.001)

        # Reconstruct the original data.
        if encodedData.size(2) < originalSignalDimension:
            while decodedData.size(2) != originalSignalDimension:
                nextSequenceLength = self.getNextSequenceLength(decodedData.size(2), originalSignalDimension)
                decodedData = self.expansionAlgorithm(decodedData, nextSequenceLength, initialSequenceLength)
        else:
            while decodedData.size(2) != originalSignalDimension:
                nextSequenceLength = self.getNextSequenceLength(decodedData.size(2), originalSignalDimension)
                decodedData = self.compressionAlgorithm(decodedData, nextSequenceLength, initialSequenceLength)
        # Assert the integrity of the expansions/compressions.
        assert decodedData.size(2) == originalSignalDimension
        assert decodedData.size(1) == 1

        # Calculate the squared error loss for this layer of compression/expansion.
        squaredErrorLoss_forward = (originalData - decodedData).pow(2).mean(dim=-1).mean(dim=1)
        print("\tAutoencoder reverse operation loss:", squaredErrorLoss_forward.mean().item(), flush=True)

        return squaredErrorLoss_forward

    def updateLossValues(self, originalData, encodedData, autoencoderLayerLoss, initialSequenceLength):
        # Keep tracking of the loss through each loop.
        layerLoss = self.calculateEncodingLoss(originalData, encodedData, initialSequenceLength)

        return autoencoderLayerLoss + layerLoss


class generalAutoencoder(generalAutoencoderBase):
    def __init__(self, accelerator=None, compressionFactor=1.5, expansionFactor=1.5):
        super(generalAutoencoder, self).__init__(accelerator, compressionFactor, expansionFactor)

    def forward(self, inputData, targetSequenceLength=64, initialSequenceLength=300, autoencoderLayerLoss=None, calculateLoss=True):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """
        # Set up the variables for signal encoding.
        batchSize, numSignals, inputSequenceLength = inputData.size()
        signalData = inputData.view(batchSize * numSignals, 1, inputSequenceLength)
        numSignalPath = [inputSequenceLength]  # Keep track of the signal's length at each iteration.

        # Initialize a holder for the loss values.
        if autoencoderLayerLoss is None: autoencoderLayerLoss = torch.zeros((batchSize * numSignals), device=inputData.device)
        else: autoencoderLayerLoss = autoencoderLayerLoss.view(batchSize * numSignals)

        # ------------- Signal Compression/Expansion Algorithm ------------- #   

        # While we have the incorrect number of signals.
        while targetSequenceLength != signalData.size(2):
            nextSequenceLength = self.getNextSequenceLength(signalData.size(2), targetSequenceLength)
            originalData = signalData.clone()  # Keep track of the initial state

            # Compress the signals to the target length.
            if targetSequenceLength < signalData.size(2):
                signalData = self.compressionAlgorithm(signalData, nextSequenceLength, initialSequenceLength)

            # Expand the signals to the target length.
            else: signalData = self.expansionAlgorithm(signalData, nextSequenceLength, initialSequenceLength)

            # Wrap up this module's layer information.
            numSignalPath.append(signalData.size(2))  # Keep track of the signal's at each iteration.

            # Keep track of the error during each compression/expansion.
            if calculateLoss:
                autoencoderLayerLoss = self.updateLossValues(originalData, signalData, autoencoderLayerLoss, initialSequenceLength)

        # Save the results.
        autoencoderLayerLoss = autoencoderLayerLoss.view(batchSize, numSignals)
        compressedSignalData = signalData.view(batchSize, numSignals, targetSequenceLength)

        # Assert the integrity of the expansion/compression.
        if numSignals != targetSequenceLength:
            assert all(numSignalPath[i] <= numSignalPath[i + 1] for i in range(len(numSignalPath) - 1)) \
                   or all(numSignalPath[i] >= numSignalPath[i + 1] for i in range(len(numSignalPath) - 1)), "List is not sorted up or down"

        # Remove the target signal from the path.
        numSignalPath.pop()

        return compressedSignalData, numSignalPath, autoencoderLayerLoss

    def printParams(self, numSignals=50, signalDimension=300):
        # generalAutoencoder().to('cpu').printParams(numSignals = 100, signalDimension = 300)
        t1 = time.time()
        summary(self, (numSignals, signalDimension))
        t2 = time.time()
        print(t2 - t1)

        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
