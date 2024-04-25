# Import files for machine learning
from .trainingPlots import trainingPlots


class autoencoderPlots(trainingPlots):
    def __init__(self, modelName, datasetNames, sharedModelWeights, savingBaseFolder, accelerator=None):
        super(autoencoderPlots, self).__init__(modelName, datasetNames, sharedModelWeights, savingBaseFolder, accelerator)
        # General parameters
        self.savingFolder = savingBaseFolder + "autoencoderPlots/"  # The folder to save the figures
