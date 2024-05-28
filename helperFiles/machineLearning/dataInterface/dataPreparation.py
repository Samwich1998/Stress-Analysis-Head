import numpy as np
import torch


# Standardize data class
def minMaxScale_noInverse(X, scale=1, torchFlag=False):
    # Ensure the proper data type
    X = torch.as_tensor(X) if torchFlag else np.asarray(X)

    # Find the minimum and maximum along the last dimension
    min_val = X.min(axis=-1, keepdims=True)
    max_val = X.max(axis=-1, keepdims=True)

    # Torch specific operations
    if torchFlag:
        min_val = min_val.values
        max_val = max_val.values

    # Handle the case when max_val == min_val (avoid division by zero)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1  # Prevent division by zero

    # Normalize the data.
    normalized = (X - min_val) / range_val  # Normalize to [0, 1]
    scaled = 2 * normalized - 1  # Then scale to [-1, 1]
    scaled = scale * scaled  # Apply additional scaling if needed

    return scaled


class standardizeData:
    def __init__(self, X, axisDimension=0, threshold=10E-15):
        self.axisDimension = axisDimension

        self.mu_ = np.mean(X, axis=axisDimension)
        self.sigma_ = np.std(X, ddof=1, axis=axisDimension)

        if (self.sigma_ <= threshold).any():
            print(self.sigma_, np.asarray(X).shape)
            assert False, "Cannot Standardize the Data. The standard deviation is too small."

        if self.axisDimension == 1:
            self.mu_ = self.mu_.reshape(-1, 1)
            self.sigma_ = self.sigma_.reshape(-1, 1)

    def standardize(self, X):
        return (X - self.mu_) / self.sigma_

    def unStandardize(self, Xhat):
        return self.mu_ + self.sigma_ * Xhat

    def getStatistics(self):
        return self.mu_, self.sigma_


# -------------------------------------------------------------------------- #
# ------------------------------ Test Methods ------------------------------ #

def testRestandardization(data):
    # Initialize the standardization class.
    standardizationClass = standardizeData(data)
    # Standardize the data
    standardizedData = standardizationClass.standardize(data)
    restandardizedData = standardizationClass.unStandardize(standardizedData)
    # Assert that the standardization worked
    assert np.allclose(restandardizedData, data)


# -------------------------------------------------------------------------- #
# ------------------------------ User Testing ------------------------------ #

if __name__ == "__main__":
    # Specify data params
    numDataPoints = 10
    numFeatures = 5
    # Initialize random data
    featureData = np.random.uniform(0, 1, (numDataPoints, numFeatures))
    featureLabels = np.random.uniform(0, 1, (numDataPoints, 1))

    # Test forward-backward method.
    testRestandardization(featureData)
    testRestandardization(featureLabels)
