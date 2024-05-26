# General
import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModelInterface.emotionModel.emotionModelHelpers.generalMethods.generalMethods import generalMethods
# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class changeVariance(signalEncoderModules):

    def __init__(self, signalMinMaxScale):
        super(changeVariance, self).__init__()
        # General parameters.
        self.signalMinMaxScale = signalMinMaxScale  # The minimum and maximum signal values.
        self.generalMethods = generalMethods()

    def adjustSignalVariance(self, inputData):
        # Apply the neural operator and the skip connection.
        return inputData  #self.generalMethods.minMaxScale_noInverse(inputData, scale=self.signalMinMaxScale)
