# General
import torch.nn as nn
import torch
import math

from torch.nn.init import calculate_gain


class weightInitialization:

    def initialize_weights(self, modelParam, activationMethod='selu', layerType='conv1D'):
        # Extract the linearity of the layer.
        linearity = layerType.split("_")[0]

        extraGain = 1.0
        # Apply the extra gain to the weights
        if layerType.split("_")[-1] == 'WNO':
            extraGain = math.sqrt(1/6.0)

        # Assert the validity of the input parameters.
        assert linearity in ['conv1D', 'fc', 'pointwise'], "I have not considered this layer's initialization strategy yet."

        if linearity == 'conv1D':
            self.kaiming_uniform_weights(modelParam, a=math.sqrt(5), nonlinearity='conv1d', extraGain=extraGain)
        elif linearity == 'pointwise':
            self.kaiming_uniform_weights(modelParam, a=math.sqrt(5), nonlinearity='conv1d', extraGain=extraGain)
        elif linearity == 'fc':
            self.kaiming_uniform_weights(modelParam, a=math.sqrt(5), nonlinearity='linear', extraGain=extraGain)
        else:
            modelParam.reset_parameters()

        return modelParam

    @staticmethod
    def reset_weights(model):
        """ Resetting model weights. """
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    # -------------------------- Layer Weights -------------------------- #

    @staticmethod
    def pointwise_uniform_weights(m):
        # Calculate the bounds for the pointwise operation.
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        bound = 2/(fan_in + fan_out)

        # Set the weights for the pointwise operation.
        nn.init.constant_(m.weight, bound)
        if m.bias is not None:
            bound = 1 / math.sqrt(fan_in) if fan_in != 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)

    @staticmethod
    def identityFC(layer):
        nn.init.eye_(layer.weight)
        nn.init.zeros_(layer.bias)

        with torch.no_grad():
            # Adding small random perturbation to the identity weights
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            perturbation = torch.randn_like(layer.weight) * (1 / fan_in)
            layer.weight.add_(perturbation)

    @staticmethod
    def initialize_weights_uniform(m):
        if hasattr(m, 'weight'):
            # Assuming the weight tensor is of shape [out_features, in_features]
            num_input_units = m.weight.size(1)  # Get the number of input units
            bound = 1 / torch.sqrt(torch.tensor(num_input_units, dtype=torch.float))
            nn.init.uniform_(m.weight, -bound, bound)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    @staticmethod
    def kaiming_uniform_weights(m, a=math.sqrt(5), nonlinearity='relu', extraGain=1.0):
        # Taken from pytorch default documentation: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L44-L48
        # Pytorch default for linear layers.
        nn.init.kaiming_uniform_(m.weight, a=a, mode='fan_in', nonlinearity=nonlinearity)

        # Apply the extra gain to the weights
        with torch.no_grad():  # Ensure we do not track this operation in the computational graph
            m.weight *= extraGain

        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in != 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)

    @staticmethod
    def xavier_uniform_weights(m, nonlinearity='relu', extraGain=1.0):
        gain = nn.init.calculate_gain(nonlinearity=nonlinearity) * extraGain
        nn.init.xavier_uniform_(m.weight, gain=gain)

        if hasattr(m, 'bias') and m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in != 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)

    @staticmethod
    def initialize_weights_lecun(m):
        if hasattr(m, 'weight'):
            # Proper LeCun Normal initialization
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = 1 / fan_in ** 0.5
            nn.init.normal_(m.weight, mean=0.0, std=std)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    @staticmethod
    def initialize_weights_kaimingLecun(m, a=math.sqrt(5), nonlinearity='relu'):
        if hasattr(m, 'weight'):
            nn.init.kaiming_uniform_(m.weight, a=a, mode='fan_in', nonlinearity=nonlinearity)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    # -------------------------- Parameter Weights -------------------------- #

    @staticmethod
    def uniformInitialization(parameter, bound):
        return nn.init.uniform_(parameter, -bound, bound)

    @staticmethod
    def uniformParamInitialization(parameter, numUnits):
        return nn.init.uniform_(parameter, -1 / math.sqrt(numUnits), 1 / math.sqrt(numUnits))

    @staticmethod
    def default_init_conv1d(parameter, fan_in):
        # Calculate the bound for the uniform distribution
        bound = math.sqrt(6 / fan_in)
        # Apply the initialization
        nn.init.uniform_(parameter, -bound, bound)
        return parameter

    @staticmethod
    def lecunParamInitialization(parameter, fan_in):
        # Initialize the weights with a normal distribution.
        std = (1 / fan_in) ** 0.5  # Adjusted fan_in for SELU according to LeCun's recommendation
        nn.init.normal_(parameter, mean=0.0, std=std)

        return parameter

    @staticmethod
    def kaimingUniformBiasInit(bias, fan_in):
        if bias is not None:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)
        return bias

    @staticmethod
    def kaimingUniformInit(parameter, a=math.sqrt(5), fan_in=1, nonlinearity='leaky_relu'):
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return parameter.uniform_(-bound, bound)

    @staticmethod
    def xavierUniformInit(parameter, fan_in, fan_out):
        # Calculate the limit for the Xavier uniform distribution
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(parameter, -limit, limit)
        return parameter

    @staticmethod
    def averagingInit(parameter, num_inputs):
        # Initialize the weights to 1/num_inputs to compute the average of the inputs
        init_value = 1.0 / num_inputs
        nn.init.constant_(parameter, init_value)
        return parameter

    @staticmethod
    def xavierNormalInit(parameter, fan_in, fan_out):
        # Calculate standard deviation for the Xavier normal distribution
        std = math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(parameter, mean=0.0, std=std)
        return parameter

    @staticmethod
    def heUniformInit(parameter, fan_in):
        # Initialize the weights with a uniform distribution using He initialization
        limit = math.sqrt(6 / fan_in)
        nn.init.uniform_(parameter, -limit, limit)
        return parameter

    @staticmethod
    def heNormalInit(parameter, fan_in):
        # Initialize the weights with a normal distribution using He initialization
        std = math.sqrt(2 / fan_in)
        nn.init.normal_(parameter, mean=0.0, std=std)
        return parameter

    # -------------------------- Custom Kernels Weights -------------------------- #

    @staticmethod
    def smoothingKernel(kernelSize=3, averageWeights=None):
        if averageWeights is not None:
            assert len(averageWeights) == kernelSize, "The kernel size and the average weights must be the same size."
            averageWeights = torch.tensor(averageWeights, dtype=torch.float32)
        else:
            averageWeights = torch.ones([kernelSize], dtype=torch.float32)
        # Initialize kernel weights.
        averageWeights = averageWeights / averageWeights.sum()

        # Set the parameter weights
        averageKernel = nn.Parameter(
            averageWeights.view(1, 1, kernelSize),
            requires_grad=False,  # Do not learn/change these weights.
        )

        return averageKernel

    def custom_kernel_initialization(self, conv_layer, activation='conv1d'):
        """
        Custom kernel initialization with a normal distribution centered around the kernel's middle
        and normalized by the number of input and output channels.
        """
        # Get the parameters of the conv layer
        num_output_channels, num_input_channels, kernel_size = conv_layer.weight.size()
        weight = conv_layer.weight
        device = weight.device
        dtype = weight.dtype

        # Determine the gain based on the activation function
        gain = nn.init.calculate_gain(activation)

        # Calculate variance considering the kernel size
        variance = 2 / (num_input_channels + num_output_channels)
        center = kernel_size // 2

        # Create a 1D tensor for distance to center calculation
        distance_to_center = (torch.arange(kernel_size, dtype=dtype, device=device) - center).float() / kernel_size

        # Generate the Gaussian kernel centered at the middle
        kernel = torch.exp(-0.5 * distance_to_center.pow(2) / variance)
        kernel = kernel / kernel.abs().sum() / kernel_size

        # Normalize the kernel using the calculated gain
        normalization_factor = gain * variance
        kernel = kernel * normalization_factor

        # Repeat the kernel across input and output channels
        kernel = kernel.view(1, 1, kernel_size).repeat(num_output_channels, num_input_channels, 1)

        # Add random initialization to the kernel.
        self.kaiming_uniform_weights(conv_layer, a=math.sqrt(5), nonlinearity='selu')
        kernel = 0.1*kernel + 0.9*conv_layer.weight

        # Assign the initialized weights to the conv layer
        with torch.no_grad():
            weight.copy_(kernel)
