# PyTorch
import torch.nn as nn
from torchsummary import summary

# Import files for machine learning
from signalInfoMappingHelpers.signalInfoMappingModules import signalInfoMappingModules


class signalMappingHead(signalInfoMappingModules):
    def __init__(self, signalDimension=64, numEncodedSignals=32):
        super(signalMappingHead, self).__init__()
        # General shape parameters.
        self.numEncodedSignals = numEncodedSignals
        self.signalDimension = signalDimension


class addSignalInfo(signalMappingHead):
    def __init__(self, signalDimension=64, numEncodedSignals=32):
        super(addSignalInfo, self).__init__(signalDimension, numEncodedSignals)
        # Define a trainable weight that is signal-specific.
        # self.signalWeights = nn.Parameter(torch.rand((1, self.numEncodedSignals, 1)))
        # signalWeights: How does each signal contribute to the biological profile (latent manifold).

        # Assert the integrity of the input parameters.
        # assert manifoldLength == int(signalDimension/2), "You must change manifoldProjection.py architecture. I only reduce the signal dimension by 2."

        # Encode spatial features.
        self.projectSignals = nn.Sequential(

            # --- Dimension: batchSize, numEncodedSignals, signalDimension -- # 

            # Convolution architecture: feature engineering
            self.twoConvolutionalFilter_resNet(numChannels=[numEncodedSignals, 2 * numEncodedSignals, numEncodedSignals], kernel_sizes=[3, 3, 3], dilations=[1, 2, 3],
                                               groups=[numEncodedSignals / 2, numEncodedSignals / 2, numEncodedSignals / 2]),

            # Convolution architecture: channel and signal reduction
            self.changeChannels(numChannels=[numEncodedSignals, finalNumSignals], groups=[1]),
            # self.signalReduction(numChannels = [finalNumSignals, finalNumSignals], groups = [1]),

            # --------- Dimension: batchSize, 64, signalDimension --------- # 

            # Convolution architecture: feature engineering
            self.twoConvolutionalFilter_resNet(numChannels=[finalNumSignals, 2 * finalNumSignals, finalNumSignals], kernel_sizes=[3, 3, 3], dilations=[1, 2, 3], groups=[finalNumSignals / 2, finalNumSignals / 2, finalNumSignals / 2]),

            # -- Dimension: batchSize, finalNumSignals, signalDimension/2 -- #  
        )

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numEncodedSignals, compressedLength) """
        # Extract the incoming data's dimension.
        batchSize, numEncodedSignals, compressedLength = inputData.size()
        # Assert that we have the expected data format.
        assert numEncodedSignals == self.numEncodedSignals, \
            f"You initialized {self.numEncodedSignals} signals but only provided {numEncodedSignals} signals."
        assert compressedLength == self.signalDimension, \
            f"You provided a signal of length {compressedLength}, but we expected {self.signalDimension}."

        # ------------------------ CNN Architecture ------------------------ # 

        # Apply CNN architecture to compress the data.
        manifoldData = self.projectSignals(inputData)
        # projectedSignals dimension: batchSize, finalNumSignals, manifoldLength

        # ------------------------------------------------------------------ # 

        return manifoldData

    def printParams(self):
        # manifoldProjection(signalDimension = 64, manifoldLength = 32, numEncodedSignals = 128, finalNumSignals = 64).to('cpu').printParams()
        summary(self, (self.numEncodedSignals, self.signalDimension))

        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')


# -------------------------------------------------------------------------- #
# -------------------------- Decoder Architecture -------------------------- #   

class removeSignalInfo(signalMappingHead):
    def __init__(self, signalDimension=64, manifoldLength=32, numEncodedSignals=128, finalNumSignals=64):
        super(removeSignalInfo, self).__init__(signalDimension, manifoldLength, numEncodedSignals)

        # Assert the integrity of the input parameters.
        # assert manifoldLength == int(signalDimension/2), "You must change manifoldProjection.py architecture. I only reduce the signal dimension by 2."

        # Encode spatial features.
        self.unprojectSignals = nn.Sequential(

            # --- Dimension: batchSize, numEncodedSignals, signalDimension -- # 

            # Convolution architecture: feature engineering
            self.twoConvolutionalFilter_resNet(numChannels=[finalNumSignals, 2 * finalNumSignals, finalNumSignals], kernel_sizes=[3, 3, 3], dilations=[1, 2, 3], groups=[finalNumSignals / 2, finalNumSignals / 2, finalNumSignals / 2]),

            # Convolution architecture: channel and signal reduction
            self.changeChannels(numChannels=[finalNumSignals, numEncodedSignals], groups=[1]),
            # nn.Upsample(size=self.signalDimension, mode='linear', align_corners=True),

            # --------- Dimension: batchSize, 64, signalDimension --------- # 

            # Convolution architecture: feature engineering
            self.twoConvolutionalFilter_resNet(numChannels=[numEncodedSignals, 2 * numEncodedSignals, numEncodedSignals], kernel_sizes=[3, 3, 3], dilations=[1, 2, 3],
                                               groups=[numEncodedSignals / 2, numEncodedSignals / 2, numEncodedSignals / 2]),

            # -- Dimension: batchSize, finalNumSignals, signalDimension/2 -- #  
        )

    def forward(self, manifoldData):
        """ The shape of manifoldData: (batchSize, finalNumSignals, manifoldLength) """
        batchSize, finalNumSignals, manifoldLength = manifoldData.size()

        # Assert we have the expected data format.
        assert manifoldLength == self.manifoldLength, f"Expected manifold dimension of {self.manifoldLength}, but got {manifoldLength}."
        assert finalNumSignals == self.finalNumSignals, f"Expected {self.finalNumSignals} signals, but got {finalNumSignals} signals."

        # ------------------------ CNN Architecture ------------------------ # 

        # Apply CNN architecture to decompress the data.
        reconstructedData = self.unprojectSignals(manifoldData)
        # reconstructedSignals dimension: batchSize, numEncodedSignals, signalDimension

        # ------------------------------------------------------------------ # 

        return reconstructedData

    def printParams(self):
        # manifoldReconstruction(signalDimension = 64, manifoldLength = 32, numEncodedSignals = 128, finalNumSignals = 64).printParams()
        summary(self, (self.finalNumSignals, self.manifoldLength))

        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
