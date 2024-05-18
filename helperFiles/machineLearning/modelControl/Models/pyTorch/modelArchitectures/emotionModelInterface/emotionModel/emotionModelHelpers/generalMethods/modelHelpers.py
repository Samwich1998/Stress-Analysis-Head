import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Import helper classes.
from ..optimizerMethods.activationFunctions import switchActivation


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

    # -------------------------- Model Interface -------------------------- #

    @staticmethod
    def calculate_weight_variance(modelPipelines):
        # Initialize variance holder.
        params_variance = {}

        with torch.no_grad():
            # Assume all models have the same architecture and parameter names
            param_names = list(modelPipelines[0].model.state_dict().keys())

            for name in param_names:
                # Collect the weights for this parameter from all models
                weights = [modelPipeline.model.state_dict()[name].unsqueeze(0) for modelPipeline in modelPipelines]
                weights_tensor = torch.cat(weights, dim=0)  # Shape: (num_models, *param_shape)

                # Calculate the variance along the first dimension (num_models)
                variance = torch.var(weights_tensor, dim=0)
                params_variance[name] = variance.mean().item()

        return params_variance

    @staticmethod
    def getCurrentSwitchActivationLayers(model):
        # Set the initial switch state to None.
        switchState = None

        for name, module in model.named_modules():
            if isinstance(module, switchActivation):
                if switchState is None:
                    switchState = module.switchState
                assert switchState == module.switchState, "Switch state is not consistent across the model."
        if switchState is None:
            switchState = False

        return switchState

    @staticmethod
    def switchActivationLayers(model, switchState=True):
        for name, module in model.named_modules():
            if isinstance(module, switchActivation):
                module.switchState = switchState

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
    def l2Normalization(model, maxNorm=2, checkOnly=False):
        # For each trainable parameter in the model with its name.
        for name, layerParams in model.named_parameters():
            # Calculate the L2 norm. THIS IS NOT SN, except for the 1D case.
            paramNorm = torch.norm(layerParams, p='fro').item()

            # Constrain the spectral norm.
            if maxNorm < paramNorm != 0:
                print("You should fix this with weight initialization:", paramNorm, name)
                if not checkOnly:
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
