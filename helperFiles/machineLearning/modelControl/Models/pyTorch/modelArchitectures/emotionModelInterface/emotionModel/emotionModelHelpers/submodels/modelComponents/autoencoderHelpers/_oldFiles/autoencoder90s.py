# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import time

# PyTorch
import torch.nn as nn
from torchsummary import summary

# Import autoencoder head.
import _autoencoderModules

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #


class autoencoder90s(_autoencoderModules.autoencoderModules):
    def __init__(self, sequenceLength = 240, compressedLength = 64):
        super(autoencoder90s, self).__init__(sequenceLength, compressedLength)
        # General shape parameters.
        self.compressedLength = compressedLength
        self.sequenceLength = sequenceLength    
        
class encodingLayer(autoencoder90s):
    def __init__(self, sequenceLength, compressedLength):
        super(encodingLayer, self).__init__(sequenceLength, compressedLength)   
        # Autoencoder notes:
        #   padding: the number of added values around the image borders. padding = dilation * (kernel_size - 1) // 2
        #   dilation: the number of indices skipped between kernel points.
        #   kernel_size: the number of indices within the sliding filter.
        #   stride: the number of indices skipped when sliding the filter.
        
        # ------------------------ CNN Architecture ------------------------ # 
        
        self.compressSignals = nn.Sequential(
            # ---------- Dimension: batchSize, 1, sequenceLength ----------- # 
            
            # Convolution architecture: channel expansion
            self.convolutionalThreeFilters_resNet(numChannels = [1, 2, 4, 6], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            
            # ---------- Dimension: batchSize, 6, sequenceLength ----------- # 
            
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [6, 6, 6, 6], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),            
            
            # Apply a pooling layer and interpolation to reduce the signal's dimension.
            nn.Upsample(size=self.compressedLength, mode='linear', align_corners=True),
            #try other upsample techniques 
            
            # ---------- Dimension: batchSize, 6, compressedLength --------- # 

            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [6, 6, 6, 6], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [6, 3, 3, 6], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [6, 1, 1, 6], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

            # Convolution architecture: channel compression
            self.convolutionalThreeFilters_semiResNet(numChannels = [6, 4, 2, 1], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1], scalingFactor = 1),

            # ---------- Dimension: batchSize, 1, compressedLength --------- # 
            
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [1, 1, 1, 1], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            
            # ---------- Dimension: batchSize, 1, compressedLength --------- # 
        )
        
        # ------------------------------------------------------------------ # 

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert self.sequenceLength == sequenceLength
                
        # ------------------------ CNN Architecture ------------------------ # 
        
        # Reshape the data to the expected input into the CNN architecture.
        signalData = inputData.view(batchSize * numSignals, 1, sequenceLength) # Seperate out indivisual signals.
        # signalData dimension: batchSize*numSignals, 1, sequenceLength

        # Apply CNN architecture to compress the data.
        compressedSignals = self.compressSignals(signalData)
        # signalData dimension: batchSize*numSignals, 1, compressedLength
        
        # Seperate put each signal into its respective batch.
        compressedData = compressedSignals.view(batchSize, numSignals, self.compressedLength) 
        # compressedData dimension: batchSize, numSignals, compressedLength
        
        # ------------------------------------------------------------------ # 
        
        return compressedData
    
    def printParams(self, numSignals = 2):
        # encodingLayer(sequenceLength = 90, compressedLength = 64).to('cpu').printParams(numSignals = 2)
        t1 = time.time()
        summary(self, (numSignals, self.sequenceLength,)) # summary(model, inputShape)
        t2 = time.time()
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        print("Total time:", t2 - t1)
        
        
class decodingLayer(autoencoder90s):
    def __init__(self, sequenceLength, compressedLength):
        super(decodingLayer, self).__init__(sequenceLength, compressedLength)

        # ------------------------ CNN Architecture ------------------------ # 

        self.decompressSignals = nn.Sequential(
            # --------- Dimension: batchSize, 1, compressedLength ---------- # 

            # Convolution architecture: channel expansion
            self.convolutionalThreeFilters_resNet(numChannels = [1, 2, 4, 6], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
                        
            # --------- Dimension: batchSize, 6, compressedLength ---------- # 
            
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [6, 6, 6, 6], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            
            # Apply a pooling layer to increase the signal's dimension.
            nn.Upsample(size=self.sequenceLength, mode='linear', align_corners=True),
            
            # ---------- Dimension: batchSize, 6, compressedLength --------- # 

            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [6, 6, 6, 6], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [6, 3, 3, 6], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            self.convolutionalThreeFilters_resNetBlock(numChannels = [6, 1, 1, 6], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),

            # Convolution architecture: channel compression
            self.convolutionalThreeFilters_semiResNet(numChannels = [6, 4, 2, 1], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1], scalingFactor = 1),

            # ---------- Dimension: batchSize, 1, compressedLength --------- # 
            
            # Convolution architecture: feature engineering
            self.convolutionalThreeFilters_resNetBlock(numChannels = [1, 1, 1, 1], kernel_sizes = [3, 3, 3], dilations = [1, 1, 1], groups = [1, 1, 1]),
            
            # -------------------------------------------------------------- # 
        )
               
         
    def forward(self, compressedData):
        """ The shape of compressedData: (batchSize, numSignals, compressedLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, compressedLength = compressedData.size()
        assert self.compressedLength == compressedLength
        
        # ------------------------ CNN Architecture ------------------------ # 

        # Reshape the signals.
        compressedSignals = compressedData.view(batchSize*numSignals, 1, self.compressedLength) 
        # compressedSignals dimension: batchSize*numSignals, 1, self.compressedLength

        # Apply CNN architecture to decompress the data.
        decompressedSignals = self.decompressSignals(compressedSignals)
        # decompressedSignals dimension: batchSize*numSignals, 1, self.sequenceLength
        
        # Organize the signals into the original batches.
        reconstructedData = decompressedSignals.view(batchSize, numSignals, self.sequenceLength)
        # reconstructedData dimension: batchSize, numSignals, self.sequenceLength
        
        # ------------------------------------------------------------------ # 

        return reconstructedData
    
    def printParams(self, numSignals = 2):
        # decodingLayer(compressedLength = 64, sequenceLength = 90).to('cpu').printParams(numSignals = 2)
        summary(self, (numSignals, self.compressedLength,))
    

