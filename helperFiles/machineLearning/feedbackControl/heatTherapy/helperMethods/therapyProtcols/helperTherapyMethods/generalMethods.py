# General
import torch
from scipy.ndimage import gaussian_filter
import numpy as np

# Import helper files.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.dataInterface.dataInterface import dataInterface


class generalMethods:

    def __init__(self):
        # Initialize helper classes.
        self.dataInterface = dataInterface

    @staticmethod
    def smoothenArray(deltaFunctionMatrix, sigma):
        return gaussian_filter(deltaFunctionMatrix, sigma=sigma)

    @staticmethod
    def createGaussianArray(inputData, gausMean, gausSTD, torchFlag=True):
        library = torch if torchFlag else np

        xValues = library.arange(len(inputData), dtype=library.float32)
        gaussianArray = library.exp(-0.5 * ((xValues - gausMean) / gausSTD) ** 2)
        gaussianArray = gaussianArray / gaussianArray.sum()  # Normalize the Gaussian array
        
        return gaussianArray

    @staticmethod
    def createGaussianMap(temp_bins, loss_bins, gausMean, gausSTD):
        # Generate a grid for Gaussian distribution calculations
        x, y = np.meshgrid(loss_bins, temp_bins)

        # Calculate Gaussian distribution values across the grid
        gaussMatrix = np.exp(-0.5 * ((x - gausMean[0]) ** 2 / gausSTD[0] ** 2 + (y - gausMean[1]) ** 2 / gausSTD[1] ** 2))
        gaussMatrix = gaussMatrix / gaussMatrix.sum()  # Normalize the Gaussian matrix

        return gaussMatrix

    def getProbabilityMatrix(self, initialData, temp_bins, loss_bins, gausSTD, noise=0.0, applyGaussianFilter=True):
        """ initialData: numPoints, (T, L); 2D array"""
        # Initialize probability matrix holder.
        probabilityMatrix = np.zeros((len(temp_bins), len(loss_bins)))

        # Calculate the probability matrix.
        for initialDataPoints in initialData:
            currentUserTemp, currentUserLoss = initialDataPoints

            if applyGaussianFilter:
                # Generate a delta function probability.
                tempBinIndex = self.dataInterface.getBinIndex(temp_bins, currentUserTemp)
                lossBinIndex = self.dataInterface.getBinIndex(loss_bins, currentUserLoss)
                probabilityMatrix[tempBinIndex, lossBinIndex] += 1  # map out bins and fill out with discrete values
            else:
                # Generate 2D gaussian matrix.
                gaussianMatrix = self.createGaussianMap(temp_bins, loss_bins, gausMean=(currentUserLoss, currentUserTemp), gausSTD=gausSTD)
                probabilityMatrix += gaussianMatrix  # Add the gaussian map to the matrix

        if applyGaussianFilter:
            # Smoothen the probability matrix.
            probabilityMatrix = self.smoothenArray(probabilityMatrix, sigma=gausSTD[::-1])

        # Normalize the probability matrix.
        probabilityMatrix += noise * np.random.randn(*probabilityMatrix.shape)  # Add random noise
        probabilityMatrix = np.clip(probabilityMatrix, 0, None)  # Ensure no negative probabilities
        probabilityMatrix = probabilityMatrix / probabilityMatrix.sum()

        return probabilityMatrix
