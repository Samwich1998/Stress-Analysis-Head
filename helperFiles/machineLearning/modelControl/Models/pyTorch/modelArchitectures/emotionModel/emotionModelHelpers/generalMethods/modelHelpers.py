import math

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class modelHelpers:

    # ---------------------- Validate Model Parameters --------------------- #

    def checkModelParams(self, model):
        # For each trainable parameter in the model.
        for layerName, layerParams in model.named_parameters():
            self.assertVariableIntegrity(layerParams, layerName, assertGradient=False)

    def assertVariableIntegrity(self, variable, variableName, assertGradient=False):
        if variable is not None:
            # Assert that the variable has a discrete value.
            assert not variable.isnan().any().item(), f"NaNs present in {variableName}: {variable}"
            assert not variable.isinf().any().item(), f"Infs present in {variableName}: {variable}"

            if variable.is_leaf:
                # Assert a valid gradient exists if needed.
                assert not assertGradient or variable.grad is not None, "No gradient present in {variableName}: {variable}"
                self.assertVariableIntegrity(variable.grad, variableName + " gradient", assertGradient=False)

    # -------------------------- Model Weights -------------------------- #

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
    def initialize_weights_kaiming(m):
        if hasattr(m, 'weight'):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

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
            std = 1 / fan_in**0.5
            nn.init.normal_(m.weight, mean=0.0, std=std)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    @staticmethod
    def uniformParamInitialization(parameter, numUnits):
        return nn.init.uniform_(parameter, -1/math.sqrt(numUnits), 1/math.sqrt(numUnits))

    @staticmethod
    def lecunParamInitialization(parameter, fan_in):
        # Initialize the weights with a normal distribution.
        std = (1 / fan_in) ** 0.5  # Adjusted fan_in for SELU according to LeCun's recommendation
        nn.init.normal_(parameter, mean=0.0, std=std)

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

    def initialize_weights(self, model, activationMethod='selu'):
        method_map = {
            'selu': self.initialize_weights_lecun,
            'relu': self.initialize_weights_kaiming,
            'sigmoid': self.initialize_weights_xavier,
            'tanh': self.initialize_weights_xavier,
            'none': self.initialize_weights_uniform,
        }

        # Get the initialization function.
        init_function = method_map.get(activationMethod, None)
        assert init_function is not None, activationMethod

        # Apply the initialization function to the model.
        for modelParam in model.modules():
            modelParam.apply(init_function)

        return model

    @staticmethod
    def getAutoencoderWeights(model):
        weightsCNN = []
        weightsFC = []

        # For each trainable parameter in the model.
        for layerName, layerParams in model.named_parameters():
            # Get the autoencoder components
            modelAttributes = layerName.split(".")
            autoencoderComponent = modelAttributes[1].split("_")[0]

            # If we have a CNN network
            if autoencoderComponent in ["compressSignalsCNN"] and modelAttributes[-1] == 'weight' and len(layerParams.data.shape) == 3:
                # layerParams Dim: numOutChannels, numInChannels/Groups, numInFeatures -> (1, 1, 15)
                weightsCNN.append(layerParams.data.numpy().copy())

            # If we have an FC network
            if autoencoderComponent in ["compressSignalsFC"] and modelAttributes[-1] == 'weight':
                # layerParams Dim: numOutFeatures, numInFeatures.
                weightsFC.append(layerParams.data.numpy().copy())

        return weightsCNN, weightsFC

    @staticmethod
    def printModelParams(model):
        for name, param in model.named_parameters():
            print(name, param.data)

    # -------------------------- Model Interface -------------------------- #

    @staticmethod
    def getLastActivationLayer(lossType, predictingProb=False):
        # Predict probabilities for classification problems
        if lossType in ["weightedKLDiv", "NLLLoss", "KLDivLoss"]:
            return "logSoftmax"
        elif lossType in ["diceLoss", "FocalLoss"] or predictingProb:
            # For CrossEntropyLoss, no activation function is needed as it applies softmax internally
            # For diceLoss and FocalLoss, you might use sigmoid or softmax based on the problem (binary/multi-class)
            return "softmax"
        else:
            return None

    # -------------------------- Model Updates -------------------------- #

    @staticmethod
    def power_iteration(W, num_iterations: int = 50, eps: float = 1e-10):
        """
        Approximates the largest singular value (spectral norm) of weight matrix W using power iteration.
        """
        assert num_iterations > 0, "Power iteration should be a positive integer"
        sigma = None

        v = torch.randn(W.size(1)).to(W.device)
        v = v / torch.norm(v, p=2) + eps
        for i in range(num_iterations):
            u = torch.mv(W, v)
            u_norm = torch.norm(u, p=2)
            u = u / u_norm
            v = torch.mv(W.t(), u)
            v_norm = torch.norm(v, p=2)
            v = v / v_norm
            # Monitor the change for debugging
            sigma = torch.dot(u, torch.mv(W, v))

        return sigma.item()

    @staticmethod
    def addSpectralNormalization(model, n_power_iterations=5):
        for name, module in model.named_children():
            # Apply recursively to submodules
            modelHelpers.addSpectralNormalization(module, n_power_iterations=n_power_iterations)

            # Check if it's an instance of the modules we want to normalize
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                # Apply spectral normalization
                spectrally_normalized_module = spectral_norm(module, n_power_iterations=n_power_iterations)
                setattr(model, name, spectrally_normalized_module)

        return model

    @staticmethod
    def spectralNormalization(model, maxSpectralNorm=2, fastPath=False):
        # For each trainable parameter in the model.
        for layerParams in model.parameters():
            # If the parameters are 2D
            if layerParams.ndim > 1:

                if fastPath:
                    spectralNorm = modelHelpers.power_iteration(layerParams, num_iterations=5)
                else:
                    # Calculate the spectral norm.
                    singular_values = torch.linalg.svdvals(layerParams)
                    spectralNorm = singular_values.max().item()  # Get the maximum singular value (spectral norm)

                # Constrain the spectral norm.
                if maxSpectralNorm < spectralNorm != 0:
                    layerParams.data = layerParams * (maxSpectralNorm / spectralNorm)

    @staticmethod
    def l2Normalization(model, maxNorm=2):
        # For each trainable parameter in the model.
        for layerParams in model.parameters():
            # Calculate the L2 norm. THIS IS NOT SN, except for the 1D case.
            paramNorm = torch.norm(layerParams, p='fro').item()

            # Constrain the spectral norm.
            if maxNorm < paramNorm != 0:
                layerParams.data = layerParams * (maxNorm / paramNorm)

    @staticmethod
    def apply_spectral_normalization(model, max_spectral_norm=2, power_iterations=1):
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
                weight = layer.weight
                weight_shape = weight.shape
                flattened_weight = weight.view(weight_shape[0], -1)

                u = torch.randn(weight_shape[0], 1, device=weight.device)
                v = None

                for _ in range(power_iterations):
                    v = torch.nn.functional.normalize(torch.mv(flattened_weight.t(), u), dim=0)
                    u = torch.nn.functional.normalize(torch.mv(flattened_weight, v), dim=0)

                sigma = torch.dot(u.view(-1), torch.mv(flattened_weight, v))
                weight.data = weight.data * (max_spectral_norm / sigma)
