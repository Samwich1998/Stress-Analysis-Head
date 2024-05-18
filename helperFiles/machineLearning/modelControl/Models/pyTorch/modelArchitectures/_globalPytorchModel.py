# PyTorch
import torch.nn as nn
import torch.nn.functional as F

# Abstract class
import abc


class globalModel(nn.Module, abc.ABC):
    def __init__(self):
        super(globalModel, self).__init__()
        # Set general model parameters.
        self.numClasses = None

    @staticmethod
    def applyFinalActivation(inputData, lastLayer):
        # Normalize the distributions.
        if lastLayer == "logSoftmax":
            inputData = F.log_softmax(inputData, dim=-1)        # Apply log-softmax activation to the last dimension to get class probabilities.
        elif lastLayer == "softmax":
            inputData = F.softmax(inputData, dim=-1)            # Apply softmax activation to the last dimension to get class probabilities.
        elif lastLayer == "normalize":
            inputData = F.normalize(inputData, dim=-1, p=1)     # Apply normalization activation to the last dimension to get normalized distributions.
        elif lastLayer is not None:
            assert False

        return inputData

    # ------------------------ Child Class Contract ------------------------ #

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")

    # @abc.abstractmethod
    # def shapInterface(self):
    #     """ Create contract for child class method """
    #     raise NotImplementedError("Must override in child")  

    # @abc.abstractmethod
    # def _resetModel(self):
    #     """ Create contract for child class method """
    #     raise NotImplementedError("Must override in child") 

    # @abc.abstractmethod
    # def _loadModel(self):
    #     """ Create contract for child class method """
    #     raise NotImplementedError("Must override in child")        

    # @abc.abstractmethod
    # def _saveModel(self):
    #     """ Create contract for child class method """
    #     raise NotImplementedError("Must override in child")  

    # @abc.abstractmethod
    # def trainModel(self):
    #     """ Create contract for child class method """
    #     raise NotImplementedError("Must override in child")  

    # @abc.abstractmethod
    # def scoreModel(self):
    #     """ Create contract for child class method """
    #     raise NotImplementedError("Must override in child") 

    # @abc.abstractmethod
    # def predict(self):
    #     """ Create contract for child class method """
    #     raise NotImplementedError("Must override in child")

    # ---------------------------------------------------------------------- #
