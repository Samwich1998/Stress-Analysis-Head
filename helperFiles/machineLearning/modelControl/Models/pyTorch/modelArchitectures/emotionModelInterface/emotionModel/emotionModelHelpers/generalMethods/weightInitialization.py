# General
import torch.nn as nn
import torch
import math


class weightInitialization:

    def initialize_weights(self, modelParam, activationMethod='selu', layerType='conv1D'):
        assert layerType in ['conv1D', 'conv1D_gausInit', 'fc', 'pointwise'], "I have not considered this layer's initialization strategy yet."

        if layerType == 'conv1D':
            self.kaiming_uniform_weights(modelParam, a=math.sqrt(5), nonlinearity='selu')
        elif layerType == 'fc':
            self.kaiming_uniform_weights(modelParam, a=math.sqrt(5), nonlinearity='selu')
        elif layerType == 'conv1D_gausInit':
            self.custom_kernel_initialization(modelParam)
        elif layerType == 'pointwise':
            self.pointwise_uniform_weights(modelParam)
        else:
            modelParam.reset_parameters()

        return modelParam

    @staticmethod
    def reset_weights(model):
        """ Resetting model weights. """
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    @staticmethod
    def smoothWeights(model, kernelSize=3):
        from ..submodels.modelComponents.signalEncoderHelpers.signalEncoderModules import signalEncoderModules
        smoothingKernel = signalEncoderModules.smoothingKernel(kernelSize=kernelSize)

        for param in model.parameters():
            if param.requires_grad:
                param.data = signalEncoderModules.applySmoothing(param.data, smoothingKernel)

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
    def kaiming_uniform_weights(m, a=math.sqrt(5), nonlinearity='relu'):
        # Taken from pytorch default documentation: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L44-L48
        # Pytorch default for linear layers.
        nn.init.kaiming_uniform_(m.weight, a=a, mode='fan_in', nonlinearity=nonlinearity)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in != 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)

    @staticmethod
    def initialize_weights_xavier(m):
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

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
    def kaimingUniformInit(parameter, a=math.sqrt(5), nonlinearity='relu'):
        parameter = nn.init.kaiming_uniform_(parameter, a=a, mode='fan_in', nonlinearity=nonlinearity)
        return parameter

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

    @staticmethod
    def custom_kernel_initialization(conv_layer):
        """
        Custom kernel initialization with a normal distribution that has a maximum
        around the center and normalized by the number of input channels.
        """
        # Get the parameters of the conv layer
        num_output_channels, num_input_channels, kernel_size = conv_layer.weight.size()
        variance = 1 / (3 * kernel_size)
        weight = conv_layer.weight
        center = kernel_size // 2

        # Fill the kernel with a normal distribution centered at the center
        distanceToCenter = (torch.arange(kernel_size, dtype=weight.dtype, device=weight.device) - center) / kernel_size / 2
        kernel = torch.exp(-0.5 * distanceToCenter.pow(2) / variance)  # Normal distribution centered at the center of the kernel.
        kernel = kernel / kernel.abs().sum()

        # Normalize the kernel.
        normalizationFactor = 2 / (num_input_channels + num_output_channels)
        normalizationFactor = normalizationFactor / kernel_size
        kernel = kernel * normalizationFactor

        # Create the final weight tensor by repeating the kernel across input and output channels
        weight_np = kernel.repeat(num_output_channels, num_input_channels, 1)

        # Assign the initialized weights to the conv layer
        with torch.no_grad():
            weight.copy_(weight_np)
