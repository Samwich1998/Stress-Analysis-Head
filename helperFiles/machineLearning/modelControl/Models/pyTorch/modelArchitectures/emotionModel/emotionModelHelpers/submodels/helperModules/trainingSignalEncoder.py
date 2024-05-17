# General
import random
from ..modelComponents.signalEncoderHelpers.signalEncoderHelpers import signalEncoderHelpers


class trainingSignalEncoder:
    def __init__(self, numEncodedSignals, expansionFactor, accelerator):
        super(trainingSignalEncoder, self).__init__()
        # General model parameters
        self.numEncodedSignals = numEncodedSignals  # The final number of signals to accept, encoding all signal information.
        self.expansionFactor = expansionFactor
        self.accelerator = accelerator

        # Specify the training parameters.
        self.maxKeepNumEncodingBuffer = 5
        self.keepNumEncodingBuffer = 0
        self.numEncodings = 4  # The number of compressions/expansions possible for this dataset.

        # Gradient accumulation parameters.
        self.numAccumulatedPoints = 0  # The number of points accumulated.
        self.numAccumulations = 0   # The number of gradient accumulations.
        self.accumulatedLoss = 0    # The accumulated loss for gradient accumulation.

    def augmentFinalTarget(self, numSignals):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        # Set up the training parameters
        forwardDirection = 0 <= self.numEncodings
        compressingSignalFlag = forwardDirection + (self.numEncodedSignals < numSignals) != 1
        numEncodings = self.numEncodings  # The number of compressions/expansions.
        numEncodedSignals = numSignals  # Initialize starting point.
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
                numEncodedSignals = signalEncoderHelpers.roundNextSignal(compressingSignalFlag, numEncodedSignals, self.expansionFactor)
                # Stop compressing once you are below the number of signals
                if numEncodedSignals <= self.numEncodedSignals: break  # Ensures upper/lower bounds
            else:
                numEncodedSignals = signalEncoderHelpers.roundNextSignal(compressingSignalFlag, numEncodedSignals, self.expansionFactor)
                # Stop compressing once you are above the number of signals
                if self.numEncodedSignals <= numEncodedSignals: break  # Ensures upper/lower bounds

        # Adjust the number of encodings.
        if numEncodedSignals == numSignals: numEncodedSignals = numEncodedSignals + 1  # It's not useful to train on nothing.
        numEncodedSignals = max(numEncodedSignals, self.numEncodedSignals)   # Ensure that we are not over-training.
        print(f"\tTraining Augmentation Stage (numEncodings totalNumEncodings): {'' if forwardDirection else '-'}{self.numEncodings} {totalNumEncodings}")

        return numEncodedSignals, totalNumEncodings, forwardDirection

    def adjustNumEncodings(self, totalNumEncodings, denoisedReconstructedData, forwardDirection):
        # Assert the integrity of the input data.
        assert len(denoisedReconstructedData.size()) == 1, f"The shape of the data must be (batchSize,) not {denoisedReconstructedData.size()}"

        # Set up the training parameters.
        self.numAccumulations = self.numAccumulations + 1
        encodingDirection = forwardDirection*2 - 1

        # Accumulate the loss if you have done enough encodings.
        if encodingDirection * totalNumEncodings == self.numEncodings:
            self.numAccumulatedPoints = self.numAccumulatedPoints + denoisedReconstructedData.size(0)
            self.accumulatedLoss = self.accumulatedLoss + denoisedReconstructedData.detach().sum()

        # If we have accumulated enough gradients for a full batch.
        if self.accelerator.gradient_accumulation_steps <= self.numAccumulations:

            # If the batch has enough relevant points.
            if self.numAccumulatedPoints != 0:
                accumulatedLoss = self.accumulatedLoss / self.numAccumulatedPoints

                # And the average loss for this batch is good enough.
                if accumulatedLoss < 0.1 or (self.numEncodings in [-1] and accumulatedLoss < 0.2):
                    self.keepNumEncodingBuffer = max(0, self.keepNumEncodingBuffer - 1)  # Move the buffer down.

                    # If we have a proven track record.
                    if self.keepNumEncodingBuffer == 0:
                        self.numEncodings = self.numEncodings + 1
                        if self.numEncodings == 0: self.numEncodings = 1  # Zero is not useful.
                        self.keepNumEncodingBuffer = 1  # Assume the best for the next round with a small buffer.

                elif 0.3 < accumulatedLoss:
                    # If we cannot complete the current goal, then record the error.
                    self.keepNumEncodingBuffer = min(self.maxKeepNumEncodingBuffer, self.keepNumEncodingBuffer + 1)

            # Reset the accumulation counter.
            self.numAccumulatedPoints = 0
            self.numAccumulations = 0
            self.accumulatedLoss = 0
