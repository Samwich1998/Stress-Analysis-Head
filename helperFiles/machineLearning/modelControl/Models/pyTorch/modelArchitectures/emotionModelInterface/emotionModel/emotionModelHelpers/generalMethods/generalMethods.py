import torch
import matplotlib.pyplot as plt
import random


class generalMethods:

    @staticmethod
    def minMaxScale_noInverse(inputData, scale=1, buffer=0):
        # Find the minimum and maximum along the last dimension
        min_val = inputData.min(dim=-1, keepdim=True).values
        max_val = inputData.max(dim=-1, keepdim=True).values

        # Check if it is within the buffer range of the scale
        min_val[(min_val + scale).abs() < buffer] = -scale
        max_val[(max_val - scale).abs() < buffer] = scale

        # Handle the case when max_val == min_val (avoid division by zero)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Prevent division by zero

        # Normalize the data.
        normalized = (inputData - min_val) / range_val  # Normalize to [0, 1]
        scaled = 2 * normalized - 1  # Then scale to [-1, 1]
        scaled = scale * scaled  # Apply additional scaling if needed

        return scaled

    @staticmethod
    def pcaCompression(signals, numComponents, standardizeSignals=True):
        # Extract the incoming data's dimension.
        batch_size, num_signals, signal_dimension = signals.shape

        if standardizeSignals:
            # Calculate means and standard deviations
            signalMeans = signals.mean(dim=-1, keepdim=True)
            signalSTDs = signals.std(dim=-1, keepdim=True) + 1e-5
            # Standardize the signals
            standardized_signals = (signals - signalMeans) / signalSTDs
        else:
            standardized_signals = signals

        # Calculate the covariance matrix
        covariance_matrix = torch.matmul(standardized_signals, standardized_signals.transpose(1, 2)) / (signal_dimension - 1)

        # Ensure covariance matrix is symmetric to avoid numerical issues
        covariance_matrix = (covariance_matrix + covariance_matrix.transpose(1, 2)) / 2
        # Add a small value to the diagonal for numerical stability (regularization)
        regularization_term = 1e-5 * torch.eye(num_signals, device=signals.device)
        covariance_matrix += regularization_term

        # Perform eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # Select the top n_components eigenvectors
        principal_components = eigenvectors[:, :, -numComponents:].contiguous()

        # Project the original signals to the new subspace
        projected_signals = torch.matmul(principal_components.transpose(1, 2), standardized_signals)

        return projected_signals, principal_components

    @staticmethod
    def svdCompression(signals, numComponents, standardizeSignals=True):
        if standardizeSignals:
            # Calculate means and standard deviations
            signalMeans = signals.mean(dim=-1, keepdim=True)
            signalSTDs = signals.std(dim=-1, keepdim=True) + 1e-5
            # Standardize the signals
            standardized_signals = (signals - signalMeans) / signalSTDs
        else:
            standardized_signals = signals

        # Perform Singular Value Decomposition
        U, S, V = torch.linalg.svd(standardized_signals)

        # Select the top n_components eigenvectors
        principal_components = U[:, :, :numComponents]

        # Project the original signals to the new subspace
        projected_signals = torch.matmul(principal_components.transpose(1, 2), standardized_signals)

        return projected_signals, principal_components

    @staticmethod
    def biased_high_sample(range_start, range_end, randomValue=None):
        if randomValue is None:
            randomValue = random.uniform(0, 1)

        # Normalizing the range to [0, 1]
        normalized_sample = 1 - randomValue ** 4  # Square to bias towards higher values
        # Scaling up to the original range
        biased_sample = range_start + (range_end - range_start) * normalized_sample
        return biased_sample

    @staticmethod
    def create_gaussian_array(array_length, num_classes, mean_gaus_indices, gaus_stds, relative_amplitudes, device):
        # Convert the inputs into the expected data structures.
        relative_amplitudes = torch.asarray(relative_amplitudes, device=device).reshape(-1)
        mean_gaus_indices = torch.asarray(mean_gaus_indices, device=device).reshape(-1)
        gaus_stds = torch.asarray(gaus_stds, device=device).reshape(-1)

        # Assert the validity of the input.
        assert num_classes - 1 < array_length, f"You cannot have {num_classes} classes with an array length of {array_length}"
        assert (mean_gaus_indices.shape == gaus_stds.shape == relative_amplitudes.shape), \
            f"Specify all the stds ({gaus_stds}) for every mean ({mean_gaus_indices})"

        # Normalize the relative amplitudes
        relative_amplitudes = relative_amplitudes / relative_amplitudes.sum()

        # Adjust indices and stds for the array length
        sampling_freq = array_length / num_classes
        gaussian_array_inds = torch.arange(array_length, device=device) - sampling_freq * 0.5
        mean_gaus_indices = sampling_freq * mean_gaus_indices
        gaus_stds = sampling_freq * gaus_stds

        # Create the Gaussian array
        normalized_gaus_array = torch.zeros(array_length, device=device)
        for mean_index, std, amplitude in zip(mean_gaus_indices, gaus_stds, relative_amplitudes):
            gaussian_distribution = torch.exp(-0.5 * ((gaussian_array_inds - mean_index) / std) ** 2)
            normalized_gaus_array += gaussian_distribution * amplitude

        # Normalize the Gaussian array
        normalized_gaus_array /= normalized_gaus_array.sum()

        return normalized_gaus_array

    @staticmethod
    def createNegativeBinomialArray(array_length, num_classes, r_parameters, p_parameters, relative_amplitudes, device):
        # Convert the inputs into the expected data structures.
        relative_amplitudes = torch.asarray(relative_amplitudes, device=device).reshape(-1)
        r_parameters = torch.asarray(r_parameters, device=device).reshape(-1)
        p_parameters = torch.asarray(p_parameters, device=device).reshape(-1)

        # Assert the validity of the input.
        assert num_classes - 1 < array_length, f"You cannot have {num_classes} classes with an array length of {array_length}"
        assert (r_parameters.shape == p_parameters.shape == relative_amplitudes.shape), \
            "Specify all the parameters for every distribution"

        # Create an array of n elements
        samplingFreq = array_length / num_classes
        distributionIndices = torch.arange(array_length, device=device) - samplingFreq * 0.5
        # Convert the class information into points
        relative_amplitudes = relative_amplitudes / relative_amplitudes.sum()

        normalizedDistributionArray = torch.zeros(array_length, device=device)
        for r_param, p_param, relativeAmplitude in zip(r_parameters, p_parameters, relative_amplitudes):
            # Generate Negative Binomial distribution
            negative_binomial_distribution = torch.exp(
                torch.lgamma(r_param + distributionIndices) - torch.lgamma(r_param) -
                torch.lgamma(distributionIndices + 1)) * (p_param ** r_param) * (
                                                     (1 - p_param) ** distributionIndices)
            normalizedDistributionArray += negative_binomial_distribution * relativeAmplitude

        normalizedDistributionArray /= normalizedDistributionArray.sum()

        return normalizedDistributionArray

    def testGaussianArray(self):
        relative_amplitudes = 1
        mean_gaus_indices = 5
        array_length = 100
        num_classes = 10
        gaus_stds = 0.4

        xAxis = torch.arange(0, num_classes, num_classes / array_length) - 0.5
        predictionArray = self.create_gaussian_array(array_length, num_classes, mean_gaus_indices, gaus_stds,
                                                     relative_amplitudes, device='cpu')

        plt.plot(xAxis, predictionArray, 'k', linewidth=2)
        plt.show()
