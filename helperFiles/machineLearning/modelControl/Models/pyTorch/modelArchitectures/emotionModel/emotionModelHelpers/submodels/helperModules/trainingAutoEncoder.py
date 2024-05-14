
# General
import random
import math


class trainingAutoEncoder:
    def __init__(self, compressedLength, compressionFactor, expansionFactor):
        super(trainingAutoEncoder, self).__init__()
        # General model parameters.
        self.compressionFactor = compressionFactor  # The expansion factor of the autoencoder
        self.compressedLength = compressedLength  # The final length of the compressed signal after the autoencoder. MUST BE CHANGED IN AUTOENCODER.py
        self.expansionFactor = expansionFactor  # The expansion factor of the autoencoder

        # Specify the training parameters.
        self.maxKeepNumEncodingBuffer = 5
        self.keepNumEncodingBuffer = 0
        self.numEncodings = 1  # The number of compressions/expansions possible for this dataset.

    def augmentFinalTarget(self, signalLength):
        """ The shape of inputData: (batchSize, signalLength, sequenceLength) """
        # Set up the training parameters
        forwardDirection = 0 <= self.numEncodings
        compressingSignalFlag = forwardDirection + (self.compressedLength < signalLength) != 1
        numEncodings = self.numEncodings  # The number of compressions/expansions.
        compressedLength = signalLength  # Initialize starting point.
        totalNumEncodings = 0

        if random.random() < 0.1:
            # Randomly change the direction sometimes.
            compressingSignalFlag = not compressingSignalFlag
            forwardDirection = not forwardDirection
        elif random.random() < 0.25:
            # Randomly compress/expand more.
            numEncodings = numEncodings + 1

        # For each compression/expansion, we are training.
        for numEncodingInd in range(abs(numEncodings)):
            totalNumEncodings = numEncodingInd + 1

            if compressingSignalFlag:
                compressedLength = math.ceil(compressedLength / self.compressionFactor)
                # Stop compressing once you are below the number of signals
                if compressedLength < self.compressedLength: break  # Ensures upper/lower bounds
            else:
                compressedLength = math.floor(compressedLength * self.expansionFactor)
                # Stop compressing once you are above the number of signals
                if self.compressedLength < compressedLength: break  # Ensures upper/lower bounds

        # It's not useful to train on nothing.
        if compressedLength == signalLength: compressedLength = compressedLength + 1
        print(f"\tTraining Augmentation Stage (numEncodings totalNumEncodings): {'' if forwardDirection else '-'}{numEncodings} {totalNumEncodings}")

        return compressedLength, totalNumEncodings, forwardDirection

    def adjustNumEncodings(self, totalNumEncodings, finalReconstructionStateLoss, forwardDirection):
        encodingDirection = forwardDirection*2 - 1
        finalLoss = finalReconstructionStateLoss.mean()
        # If we can keep going forwards.
        if finalLoss < 0.1 or (self.numEncodings in [-1] and finalLoss < 0.2):
            if encodingDirection*totalNumEncodings == self.numEncodings:
                self.keepNumEncodingBuffer = max(0, self.keepNumEncodingBuffer - 1)

                # If we have a proven track record.
                if self.keepNumEncodingBuffer == 0:
                    self.numEncodings = max(self.numEncodings, encodingDirection*totalNumEncodings + 1)
                    if self.numEncodings == 0: self.numEncodings = 1  # Zero is not useful.

        elif 0.3 < finalLoss:
            # If we cannot complete the current goal, then record the error.
            self.keepNumEncodingBuffer = min(self.maxKeepNumEncodingBuffer, self.keepNumEncodingBuffer + 1)
