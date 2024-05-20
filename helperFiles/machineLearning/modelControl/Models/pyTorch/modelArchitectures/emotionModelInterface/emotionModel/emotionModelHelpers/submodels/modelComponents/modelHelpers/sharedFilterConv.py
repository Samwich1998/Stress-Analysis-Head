# PyTorch
import torch.nn.functional as F
import torch.nn as nn


class sharedFilterConv1D(nn.Module):
    def __init__(self, numInChannels, numOutChannels, kernel_size, stride, padding, dilation, groups):
        super(sharedFilterConv1D, self).__init__()
        # General parameters.
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        # Define the convolutional parameter.
        self.conv1d = nn.Conv1d(in_channels=numInChannels, out_channels=numOutChannels, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=groups, padding_mode='reflect', bias=True)

    # --------------------------- Helper Methods --------------------------- #

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numChannels, initialDimension) """
        batchSize, numChannels, initialDimension = inputData.size()

        # Expand the convolutional weights to match the input size.
        kernelWeights = self.conv1d.weight.expand(numChannels, 1, self.kernel_size)
        bias = self.conv1d.bias.expand(numChannels)

        # Perform the convolution.
        inputData = F.conv1d(inputData, kernelWeights, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=numChannels) + bias

        return inputData
