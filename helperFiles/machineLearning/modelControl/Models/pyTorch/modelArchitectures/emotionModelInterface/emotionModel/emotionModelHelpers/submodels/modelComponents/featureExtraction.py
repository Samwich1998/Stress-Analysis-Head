# General
from torchsummary import summary

# Import files for machine learning
from .emotionActivityPredictionHelpers.featureExtractionModules import featureExtractionModules


class featureExtraction(featureExtractionModules):
    def __init__(self, numEncodedSignals=32, compressedLength=64, numCommonSignals=8):
        super(featureExtraction, self).__init__()
        # General parameters.
        self.numEncodedSignals = numEncodedSignals  # The dimensionality of the encoded data.
        self.compressedLength = compressedLength    # The final length of each signal after projection.

        # Reduce the size of the features.
        self.extractSignalFeatures = self.commonSignalFeatureExtraction(inChannel=1)
        self.reduceChannelSignals = self.commonChannelFeatureExtraction(inChannel=numEncodedSignals, outChannel=numCommonSignals)

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numEncodedSignals, compressedLength) """
        # Extract the incoming data's dimension and ensure a proper data format.
        batchSize, numEncodedSignals, compressedLength = inputData.size()

        # Assert we have the expected data format.
        assert compressedLength == self.compressedLength, f"Expected manifold dimension of {self.compressedLength}, but got {compressedLength}."
        assert numEncodedSignals == self.numEncodedSignals, f"Expected {self.numEncodedSignals} signals, but got {numEncodedSignals} signals."

        # Apply CNN architecture to extract features from the data.
        signalData = inputData.view(batchSize * numEncodedSignals, 1, compressedLength)
        signalFeatureData = self.extractSignalFeatures(signalData)
        signalFeatureData = signalFeatureData.view(batchSize, numEncodedSignals, compressedLength)
        # signalFeatureData dimension: batchSize, numEncodedSignals, numSignalFeatures

        # Apply CNN architecture to extract features from the data.
        signalFeatureData = self.reduceChannelSignals(signalFeatureData)
        # signalFeatureData dimension: batchSize, numCommonSignals, numSignalFeatures

        return signalFeatureData

    def printParams(self):
        # featureExtraction(numCommonFeatures = 64, numEncodedSignals = 64, compressedLength = 32).to('cpu').printParams()
        summary(self, (self.numEncodedSignals, self.compressedLength,))
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
