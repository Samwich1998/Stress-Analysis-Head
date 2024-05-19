# General
import torch.nn as nn
import torch
import math


class weightInitialization:

    def initialize_weights(self, modelParam, activationMethod='selu', layerType='conv'):
        assert layerType in ['conv', 'fc'], "I have not considered this layer's initialization strategy yet."

        if activationMethod == 'selu':
            if layerType == 'conv':
                self.kaiming_uniform_weights(modelParam, a=math.sqrt(5), nonlinearity='linear')
            elif layerType == 'fc':
                self.kaiming_uniform_weights(modelParam, a=math.sqrt(5), nonlinearity='linear')
        elif activationMethod == 'boundedExp':
            if layerType == 'conv':
                self.kaiming_uniform_weights(modelParam, a=math.sqrt(5), nonlinearity='linear')
            elif layerType == 'fc':
                self.kaiming_uniform_weights(modelParam, a=math.sqrt(5), nonlinearity='linear')
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
