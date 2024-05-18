# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class signalEncoding(nn.Module):
    def __init__(self, inputDimension = 25, outputDimension = 25, numSignals = 75):
        super(signalEncoding, self).__init__()
        # General parameters.
        self.numSignals = numSignals # The number of independant signals in the model.
        self.inputDimension = inputDimension
        
        # A list of modules to encode each signal.
        # self.signalEncodingModules_FC = nn.ModuleList()  # Use ModuleList to store child modules.
        self.signalEncodingModules_CNN1 = nn.ModuleList()  # Use ModuleList to store child modules.
        self.signalEncodingModules_CNN2 = nn.ModuleList()  # Use ModuleList to store child modules.
        # signalEncodingModules dimension: self.numSignals
        
        # Normalize the distributions.
        self.batchNorm = nn.BatchNorm1d(outputDimension, affine = True, momentum = 0.1, track_running_stats=True)
        
        # Encoding notes:
        #   padding: the number of added values around the image borders. padding = dilation * (kernel_size - 1) // 2
        #   dilation: the number of indices skipped between kernel points.
        #   kernel_size: the number of indices within the sliding filter.
        #   stride: the number of indices skipped when sliding the filter.

        # For each signal.
        for signalInd in range(self.numSignals):                        
            # Encode spatial features.
            self.signalEncodingModules_CNN1.append(
                nn.Sequential(
                    # Convolution architecture: Layer 1, Conv 1-2
                    nn.Conv1d(in_channels=1, out_channels=1, kernel_size=11, stride=1, dilation = 1, padding=5, padding_mode='circular', groups=1, bias=True),
                    nn.Conv1d(in_channels=1, out_channels=1, kernel_size=11, stride=1, dilation = 1, padding=5, padding_mode='circular', groups=1, bias=True),
                    nn.LayerNorm(self.inputDimension, eps = 1E-10),
                    nn.SELU(),
                )
            )
            
            # Encode spatial features.
            self.signalEncodingModules_CNN2.append(
                nn.Sequential(
                    # Convolution architecture: Layer 1, Conv 1
                    nn.Conv1d(in_channels=1, out_channels=1, kernel_size=9, stride=1, dilation = 2, padding=8, padding_mode='circular', groups=1, bias=True),
                    nn.Conv1d(in_channels=1, out_channels=1, kernel_size=9, stride=1, dilation = 2, padding=8, padding_mode='circular', groups=1, bias=True),
                    nn.LayerNorm(self.inputDimension, eps = 1E-10),
                    nn.SELU(),
                )
            )
            
    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """  
        # Assert that we have the expected data format.
        assert inputData.shape[1] == self.numSignals, \
            f"You initialized {self.numSignals} signals but only provided {inputData.shape[1]} signals."

        # Create a new tensor to hold updated values.
        updatedInputData = torch.zeros_like(inputData, requires_grad=False)
        
        # For each signal.
        for signalInd in range(self.numSignals):
            signalData = inputData[:, signalInd, :]
            
            # Prepare for CNN network.
            signalData = signalData.unsqueeze(1) # Add one channel to the signal.
            # Apply CNN network to encode the signals.
            encodedSignals = (signalData + self.signalEncodingModules_CNN1[signalInd](signalData))/2
            encodedSignals = (encodedSignals + self.signalEncodingModules_CNN2[signalInd](encodedSignals))/2
            # Remove the extra CNN channel.
            encodedSignals = encodedSignals.squeeze(1) # Remove one channel from the signal.

            # Store the encoded signals.
            updatedInputData[:, signalInd, :] = encodedSignals
            # Dimension: batchSize, numSignals, compressedLength

        return updatedInputData
    
    def printParams(self, inputDimension = 75):
        #signalEncoding(inputDimension = 75, outputDimension = 75, numSignals = 50).printParams()
        summary(self, (self.numSignals, inputDimension))
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')

    
    