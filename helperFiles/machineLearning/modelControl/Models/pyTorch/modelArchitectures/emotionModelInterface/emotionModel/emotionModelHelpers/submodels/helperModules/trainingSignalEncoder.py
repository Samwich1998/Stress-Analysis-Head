# General
import random
from ..modelComponents.signalEncoderHelpers.signalEncoderHelpers import signalEncoderHelpers


class trainingSignalEncoder:
    def __init__(self, numEncodedSignals, expansionFactor):
        super(trainingSignalEncoder, self).__init__()
        # General model parameters
        self.numEncodedSignals = numEncodedSignals  # The final number of signals to accept, encoding all signal information.
        self.expansionFactor = expansionFactor
        self.switchDirections = False

    def randomlyChangeDirections(self):
        if random.random() < 0.1:
            self.switchDirections = True
        else:
            self.switchDirections = False

    def augmentFinalTarget(self, numSignals):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        # Set up the training parameters
        compressingSignalFlag = self.numEncodedSignals < numSignals
        numEncodedSignals = numSignals  # Initialize starting point.
        forwardDirection = True
        totalNumEncodings = 0

        if self.switchDirections:
            # Randomly change the direction sometimes.
            compressingSignalFlag = not compressingSignalFlag
            forwardDirection = not forwardDirection

        while True:
            totalNumEncodings = totalNumEncodings + 1

            if compressingSignalFlag:
                numEncodedSignals = signalEncoderHelpers.roundNextSignal(compressingSignalFlag, numEncodedSignals, self.expansionFactor)
                # Stop compressing once you are below the number of signals
                if not self.switchDirections and numEncodedSignals <= self.numEncodedSignals: break  # Ensures upper/lower bounds
            else:
                numEncodedSignals = signalEncoderHelpers.roundNextSignal(compressingSignalFlag, numEncodedSignals, self.expansionFactor)
                # Stop compressing once you are above the number of signals
                if not self.switchDirections and self.numEncodedSignals <= numEncodedSignals: break  # Ensures upper/lower bounds

            if self.switchDirections and totalNumEncodings == 2:
                break

        # Adjust the number of encodings.
        if numEncodedSignals == numSignals: numEncodedSignals = numEncodedSignals + 1  # It's not useful to train on nothing.
        numEncodedSignals = max(numEncodedSignals, self.numEncodedSignals)   # Ensure that we are not over-training.
        print(f"\tTraining Augmentation Stage (totalNumEncodings): {'' if forwardDirection else '-'} {totalNumEncodings}")

        return numEncodedSignals
