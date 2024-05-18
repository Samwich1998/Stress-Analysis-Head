# ---------------------------- Imported Modules ---------------------------- #

# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class changeVariance(signalEncoderModules):

    def __init__(self, debuggingResults=False):
        super(changeVariance, self).__init__()
        # General parameters.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool

        # Map the initial signals into a common subspace.
        self.removeSignalAdjustment = self.varianceTransformation(inChannel=1)
        self.adjustSignals = self.varianceTransformation(inChannel=1)

    def adjustSignalVariance(self, inputData):
        return self.encodingInterface(inputData, self.adjustSignals)

    def unAdjustSignalVariance(self, inputData):
        return self.encodingInterface(inputData, self.removeSignalAdjustment)
