import torch
import torch.nn as nn
import torch.nn.init as init
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
    def reset_weights(model):
        """ Resetting model weights. """
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    @staticmethod
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for layers followed by Sigmoid or Tanh
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                # Kaiming/He initialization for layers followed by ReLU
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                # Orthogonal initialization for RNNs
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                # BatchNorm initialization
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

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
            if autoencoderComponent in ["compressSignalsCNN"] and modelAttributes[-1] == 'weight' and len(
                    layerParams.data.shape) == 3:
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
                if maxSpectralNorm < spectralNorm:
                    layerParams.data = layerParams * (maxSpectralNorm / spectralNorm)

    @staticmethod
    def l2Normalization(model, maxNorm=2):
        # For each trainable parameter in the model.
        for layerParams in model.parameters():
            if layerParams.ndim > 1:
                # Calculate the L2 norm. THIS IS NOT SN, except for the 1D case.
                paramNorm = torch.norm(layerParams, p=2).item()

                # Constrain the spectral norm.
                if maxNorm < paramNorm:
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
