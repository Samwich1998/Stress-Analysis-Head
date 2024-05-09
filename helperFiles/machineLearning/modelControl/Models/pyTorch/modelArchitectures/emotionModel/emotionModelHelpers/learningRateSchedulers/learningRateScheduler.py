import torch
import tensorly as tl
from tensorly.decomposition import parafac

from helperFiles.machineLearning.modelControl.Models.pyTorch.modelArchitectures.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface


class CustomLRScheduler:
    def __init__(self, numPrincipleComponents, numParamSamples=5, addingNoiseSTD=1E-3):
        # General parameters
        self.numPrincipleComponents = numPrincipleComponents  # Number of principal components to retain.
        self.emotionDataInterface = emotionDataInterface()  # Initialize the data interface.
        self.numParamSamples = numParamSamples  # Number of samples to evaluate the model.
        self.addingNoiseSTD = addingNoiseSTD  # Deviation from the mean for the principal components.
        self.directions = [0]

        # Assert the validity of the input.
        assert numPrincipleComponents > 0, "The number of principal components must be greater than 0."
        assert numParamSamples > 2, "The number of samples must be at least 1, as it represents the first forward pass."

    def apply_pca_and_update(self, model, data, targets, loss_fn):
        """
        Apply PCA to the learnable parameters of the model and update the weights.

        Args:
        model (nn.Module): PyTorch model whose parameters will be modified.
        K (int): Number of principal components to retain.
        R (float): Scaling factor for the principal component in the update rule.
        """
        # Set up the update parameters.
        original_training_state = model.training  # Store the original training state.
        finalParamDatas = []  # Initialize the final parameter data.
        model.eval()  # Set the model to evaluation mode.

        with torch.no_grad():  # Ensure no gradients are computed in this block.
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Prepare the data for the model.
                    originalParam = param.data.clone()  # Store the original parameters.
                    paramLosses = torch.zeros((self.numParamSamples - 1), device=data.device)  # Initialize the loss history.
                    gradient_tensor = param.grad.data.detach().clone()

                    # Decompose the gradient tensor, selecting the top 'k' components.
                    decomposedGradientTensorInfo = parafac(gradient_tensor, rank=self.numPrincipleComponents, n_iter_max=100, init='svd', svd='truncated_svd',
                                                           normalize_factors=False, orthogonalise=False, tol=1e-08, random_state=None, l2_reg=0, cvg_criterion='abs_rec_error', linesearch=True)
                    # principleComponents dimension: (numDimensions, gradient_tensor.size(dimensionInd), numPrincipleComponents).
                    # decomposedGradientTensorInfo dimension: weights, principleComponents.
                    # weights dimension: numPrincipleComponents.
                    # numDimensions: len(gradient_tensor.size()).

                    # Reconstruct the gradient tensor from the principal components only.
                    reconstructed_grad = tl.cp_to_tensor(decomposedGradientTensorInfo)
                    # reconstructed_grad dimension: gradient_tensor.size().

                    # Perform updates and evaluate the model.
                    for sampleInd in range(1, self.numParamSamples + 1):
                        # Update the parameter along the direction of the principal components.
                        walkingDistance = sampleInd / self.numParamSamples  # Scaling factor for step size.
                        param.data = originalParam + walkingDistance * reconstructed_grad

                        # Calculate the loss.
                        output = model(data)  # Forward pass.
                        loss = loss_fn(output, targets)  # Compute the loss.
                        # Store the loss. The first sample is the original parameter.

                        # Store the loss. The first sample is the original parameter.
                        paramLosses[sampleInd - 1] = loss.mean().item()
                        # paramLosses dimension: (numParamSamples-1, 3).

                    # Calculate the gradients of the loss with respect to each parameter.
                    dL_dP, dL_dX = torch.gradient(paramLosses, spacing=[reconstructed_grad.norm(p='fro') / self.numParamSamples])
                    # Average the gradients over the noise.
                    smoothedParamLosses = paramLosses.mean(dim=1)
                    dL_dP = dL_dP.mean(dim=1)
                    dL_dX = dL_dX.mean(dim=1)

                    # Find the best magnitude to update the parameter.
                    learningRate = 0.6 * (smoothedParamLosses.argmin() + 1) / self.numParamSamples + \
                                   0.3 * (dL_dP.argmin() + 1) / self.numParamSamples + \
                                   0.1 * (dL_dX.argmin() + 1) / self.numParamSamples
                    # Update the parameter.
                    finalParamDatas.append(originalParam + learningRate * reconstructed_grad)

                    # Reset the parameter to the original state.
                    param.data = originalParam

        # Restore the original training state.
        model.train(original_training_state)

        return model
