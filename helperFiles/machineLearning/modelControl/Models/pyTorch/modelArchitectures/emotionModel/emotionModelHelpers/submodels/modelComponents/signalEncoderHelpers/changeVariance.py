# ---------------------------- Imported Modules ---------------------------- #

# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class changeVariance(signalEncoderModules):

    def __init__(self, debuggingResults=False):
        super(changeVariance, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool

        # Map the initial signals into a common subspace.
        self.adjustSignals = self.varianceTransformation(inChannel=1)
        self.removeSignalAdjustment = self.varianceTransformation(inChannel=1)

        self.smoothenEncodingSpace = self.smoothDataModel(inChannel=1)

    def adjustSignalVariance(self, inputData):
        adjustedData = self.encodingInterface(inputData, self.adjustSignals)
        adjustedData = self.encodingInterface(adjustedData, self.smoothenEncodingSpace)

        return adjustedData

    def unAdjustSignalVariance(self, inputData):
        return self.encodingInterface(inputData, self.removeSignalAdjustment)
