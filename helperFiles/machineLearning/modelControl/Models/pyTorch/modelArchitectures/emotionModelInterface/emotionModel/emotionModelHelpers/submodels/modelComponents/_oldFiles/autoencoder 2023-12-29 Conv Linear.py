# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch.nn as nn
from torchsummary import summary

# Import models
import _convolutionalLinearLayer

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class autoencoderParameters(nn.Module):
    def __init__(self, sequenceLength, compressedLength):
        super(autoencoderParameters, self).__init__()
        # General shape parameters.
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength
        
        # Layern dimensions.
        self.firstCompressionDim = 200
        self.secondCompressionDim = 100
        
class encodingLayer(autoencoderParameters):
    def __init__(self, sequenceLength, compressedLength):
        super(encodingLayer, self).__init__(sequenceLength, compressedLength)   

        # Initialize the convolutional linear layer.
        self.convLinear_1 = _convolutionalLinearLayer.convolutionalLinearLayer(
                compressedDim = self.firstCompressionDim,
                initialDimension = self.sequenceLength,
                compressedEmbeddedDim = 20,
                embeddingStride = 3,
                embeddedDim = 30,
        )
        
        self.convLinearArchitecture_1 = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.convLinear_1.embeddedDim, self.convLinear_1.compressedEmbeddedDim, bias = True),
            nn.SELU(),
        )
        
                
        # Initialize the convolutional linear layer.
        self.convLinear_2 = _convolutionalLinearLayer.convolutionalLinearLayer(
                initialDimension = self.firstCompressionDim,
                compressedDim = self.secondCompressionDim,
                compressedEmbeddedDim = 12,
                embeddingStride = 2,
                embeddedDim = 24,
        )
        
        self.convLinearArchitecture_2 = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.convLinear_2.embeddedDim, self.convLinear_2.compressedEmbeddedDim, bias = True),
            nn.SELU(),
        )
        
        # Initialize the convolutional linear layer.
        self.convLinear_3 = _convolutionalLinearLayer.convolutionalLinearLayer(
                initialDimension = self.secondCompressionDim,
                compressedDim = self.secondCompressionDim,
                compressedEmbeddedDim = 16,
                embeddingStride = 1,
                embeddedDim = 16,
        )
        
        self.convLinearArchitecture_3 = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.convLinear_3.embeddedDim, self.convLinear_3.compressedEmbeddedDim, bias = True),
            nn.SELU(),
        )
        
        self.finalPooling = nn.AdaptiveAvgPool1d(self.compressedLength)

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert self.sequenceLength == sequenceLength
        
        # Reshape the data to the expected input into the ANN architecture.
        signalData = inputData.view(batchSize * numSignals, sequenceLength) # Seperate out indivisual signals.
        # signalData dimension: batchSize*numSignals, sequenceLength
                
        # ------------------------ ANN Architecture ------------------------ # 
        
        compressedSignals_Linear1 = self.convLinear_1(signalData, self.convLinearArchitecture_1)
        # embeddedSignalData dimension: batchSize*numSignals, firstCompressionDim
        
        compressedSignals_Linear2 = self.convLinear_2(compressedSignals_Linear1, self.convLinearArchitecture_2)
        # embeddedSignalData dimension: batchSize*numSignals, secondCompressionDim
        
        compressedSignals_Linear3 = self.convLinear_3(compressedSignals_Linear2, self.convLinearArchitecture_3)
        compressedSignals_Linear3 = self.finalPooling(compressedSignals_Linear3)
        # embeddedSignalData dimension: batchSize*numSignals, compressedLength
        
        # ------------------------------------------------------------------ # 
        
        # Seperate put each signal into its respective batch.
        compressedData = compressedSignals_Linear3.view(batchSize, numSignals, self.compressedLength) 
        # compressedData dimension: batchSize, numSignals, self.compressedLength

        return compressedData
    
    def printParams(self, numSignals = 50):
        #encodingLayer(sequenceLength = 300, compressedLength = 64).printParams(numSignals = 50)
        summary(self, (numSignals, self.sequenceLength,)) # summary(model, inputShape)
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        
    
class decodingLayer(autoencoderParameters):
    def __init__(self, compressedLength, sequenceLength):
        super(decodingLayer, self).__init__(sequenceLength, compressedLength)
                
        # Upsampling layers.
        self.initialPooling = nn.Upsample(size=self.secondCompressionDim, mode='linear', align_corners=True)    
        
        # Initialize the convolutional linear layer.
        self.deconvLinear_1 = _convolutionalLinearLayer.convolutionalLinearLayer(
                initialDimension = self.secondCompressionDim,
                compressedDim = self.secondCompressionDim,
                compressedEmbeddedDim = 16,
                embeddingStride = 1,
                embeddedDim = 16,
        )
        
        self.deconvLinearArchitecture_1 = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.deconvLinear_1.embeddedDim, self.deconvLinear_1.compressedEmbeddedDim, bias = True),
            nn.SELU(),
        )
        
        # Initialize the convolutional linear layer.
        self.deconvLinear_2 = _convolutionalLinearLayer.convolutionalLinearLayer(
                initialDimension = self.secondCompressionDim,
                compressedDim = self.firstCompressionDim,
                compressedEmbeddedDim = 24,
                embeddingStride = 2,
                embeddedDim = 12,
        )
        
        self.deconvLinearArchitecture_2 = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.deconvLinear_2.embeddedDim, self.deconvLinear_2.compressedEmbeddedDim, bias = True),
            nn.SELU(),
        )
                
        # Initialize the convolutional linear layer.
        self.deconvLinear_3 = _convolutionalLinearLayer.convolutionalLinearLayer(
                initialDimension = self.firstCompressionDim,
                compressedDim = self.sequenceLength,
                compressedEmbeddedDim = 30,
                embeddingStride = 2,
                embeddedDim = 20,
        )
        
        self.deconvLinearArchitecture_3 = nn.Sequential(
            # Neural architecture: Layer 1.
            nn.Linear(self.deconvLinear_3.embeddedDim, self.deconvLinear_3.compressedEmbeddedDim, bias = True),
            nn.SELU(),
        )
                        
    def forward(self, compressedData):
        """ The shape of compressedData: (batchSize, compressedLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, compressedLength = compressedData.size()
        assert self.compressedLength == compressedLength
        
        # Expand the compressed signals through an FC network.
        compressedSignals = compressedData.view(batchSize*numSignals, self.compressedLength) # Seperate put each signal into its respective batch.
        # compressedSignals dimension: batchSize*numSignals, compressedLength
        
        # ------------------------ ANN Architecture ------------------------ # 
    
        decompressedSignals_1 = self.initialPooling(compressedSignals.unsqueeze(1)).squeeze(1)        
        decompressedSignals_1 = self.deconvLinear_1(decompressedSignals_1, self.deconvLinearArchitecture_1)
        # embeddedSignalData dimension: batchSize*numSignals, secondCompressionDim 
        
        decompressedSignals_2 = self.deconvLinear_2(decompressedSignals_1, self.deconvLinearArchitecture_2)
        # embeddedSignalData dimension: batchSize*numSignals, firstCompressionDim
        
        decompressedSignals_3 = self.deconvLinear_3(decompressedSignals_2, self.deconvLinearArchitecture_3)
        # embeddedSignalData dimension: batchSize*numSignals, sequenceLength

        # ------------------------------------------------------------------ # 

        # Reconstruct the original signals.
        reconstructedData = decompressedSignals_3.view(batchSize, numSignals, self.sequenceLength)   # Organize the signals into the original batches.
        # reconstructedData dimension: batchSize, numSignals, sequenceLength

        return reconstructedData
    
    def printParams(self, numSignals = 50):
        #decodingLayer(compressedLength = 64, sequenceLength = 300).printParams()
        summary(self, (numSignals, self.compressedLength,))
    
    
    
    
    
    