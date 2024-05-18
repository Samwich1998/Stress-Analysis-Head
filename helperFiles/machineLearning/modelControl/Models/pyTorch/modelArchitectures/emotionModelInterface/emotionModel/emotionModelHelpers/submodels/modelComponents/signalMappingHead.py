
# PyTorch
import torch
import torch.nn as nn

# Import machine learning files
from .signalInfoMappingHelpers.signalInfoMappingModules import signalInfoMappingModules


class signalMappingHead(signalInfoMappingModules):
    def __init__(self, maxSequenceLength=64):
        super(signalMappingHead, self).__init__()
        # General parameters.
        self.maxSequenceLength = maxSequenceLength  # The maximum number of points in any signal.
        self.numEncodingStamps = 6  # The number of binary bits in the encoding (010 = 2 signals; 3 encodings).

        # A list of parameters to encode each signal.
        self.encodedStamp = nn.ParameterList()  # A list of learnable parameters for learnable signal positions.

        # For each encoding bit.
        for stampInd in range(self.numEncodingStamps):
            # Assign a learnable parameter to the signal.
            self.encodedStamp.append(torch.nn.Parameter(torch.randn(maxSequenceLength)))

        # Learn how to embed the positional information into the signals.
        self.learnSignalInfo = self.learnSignalInfoModule(inChannel=1)
        self.unlearnSignalInfo = self.learnSignalInfoModule(inChannel=1)

    # ---------------------------------------------------------------------- #
    # -------------------- Learned Positional Encoding --------------------- #

    def addSignalInfo(self, inputData):
        return self.positionalEncoding(inputData, self.learnSignalInfo, addingData=1)

    def removeSignalInfo(self, inputData):
        return self.positionalEncoding(inputData, self.unlearnSignalInfo, addingData=-1)

    def positionalEncoding(self, inputData, learningModule, addingData=1):
        # Set up the variables for signal encoding.
        batchSize, numSignals, signalDimension = inputData.size()
        positionEncodedData = inputData.clone()

        # Extract the size of the input parameter.
        bitInds = torch.arange(self.numEncodingStamps).to(positionEncodedData.device)
        signalInds = torch.arange(numSignals).to(positionEncodedData.device)

        # Generate the binary encoding of signalInds in a batched manner
        binary_encoding = signalInds[:, None].bitwise_and(2 ** bitInds).bool()
        # binary_encoding dim: numSignals, numEncodingStamps

        # If we are removing the encoding, learn how to remove the encoding.
        if addingData == -1: positionEncodedData = self.encodingInterface(positionEncodedData, learningModule, useCheckpoint=False)

        # For each stamp encoding
        for stampInd in range(self.numEncodingStamps):
            # Check each signal if it is using this specific encoding.
            usingStampEncoding = binary_encoding[:, stampInd:stampInd + 1]
            encodingVector = usingStampEncoding.float() * self.encodedStamp[stampInd][-signalDimension:]
            # encodingVector dim: numSignals, signalDimension

            # Add the stamp encoding to all the signals in all the batches.
            positionEncodedData = positionEncodedData + (stampInd % 2 == 0) * addingData * encodingVector.unsqueeze(0) / 10

        # If we are encoding the data, learn how to apply the encoding.
        if addingData == 1: positionEncodedData = self.encodingInterface(positionEncodedData, learningModule, useCheckpoint=False)

        return positionEncodedData
