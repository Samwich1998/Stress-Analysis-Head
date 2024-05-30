# General
from scipy.ndimage import gaussian_filter
import torch

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
        library = torch if torchFlag else None

        xValues = library.arange(len(inputData), dtype=library.float32)
        gaussianArray = library.exp(-0.5 * ((xValues - gausMean) / gausSTD) ** 2)
        gaussianArray = gaussianArray / gaussianArray.sum()  # Normalize the Gaussian array
        
        return gaussianArray

    @staticmethod
    def createGaussianMap(allParameterBins, allPredictionBins, gausMean, gausSTD):
        # Generate a grid for Gaussian distribution calculations
        x, y = torch.meshgrid(allPredictionBins, allParameterBins)

        # Calculate Gaussian distribution values across the grid
        gaussMatrix = torch.exp(-0.5 * ((x - gausMean[0]) ** 2 / gausSTD[0] ** 2 + (y - gausMean[1]) ** 2 / gausSTD[1] ** 2))
        gaussMatrix = gaussMatrix / gaussMatrix.sum()  # Normalize the Gaussian matrix

        return gaussMatrix

    def getProbabilityMatrix(self, initialData, allParameterBins, allPredictionBins, gausSTD, noise=0.0, applyGaussianFilter=True):
        probabilityMatrix = []
        # For each input parameter.
        for parameterInd in range(len(allParameterBins)):
            parameterBins = allParameterBins[parameterInd]
            probabilityMatrix.append([])

            # For each prediction.
            for predictionInd in range(len(allPredictionBins)):
                predictionBins = allPredictionBins[predictionInd]

                # Initialize a probability matrix: p(predictionBin | *paramBin).
                probabilityMatrix.append(torch.zeros(len(parameterBins), len(predictionBins)))

        probabilityMatrix = [torch.zeros((len(allParameterBins), len(allPredictionBins[predictionInd]))) for predictionInd in range(len(allPredictionBins))]

        # Calculate the probability matrix.
        for initialDataPoints in initialData:
            currentUserTemp, currentUserLoss = initialDataPoints

            if applyGaussianFilter:
                # Generate a delta function probability.
                tempBinIndex = self.dataInterface.getBinIndex(allParameterBins, currentUserTemp)
                lossBinIndex = self.dataInterface.getBinIndex(allPredictionBins, currentUserLoss)
                probabilityMatrix[tempBinIndex][lossBinIndex] += 1  # map out bins and fill out with discrete values
            else:
                # Generate 2D gaussian matrix.
                gaussianMatrix = self.createGaussianMap(allParameterBins, allPredictionBins, gausMean=(currentUserLoss, currentUserTemp), gausSTD=gausSTD)
                probabilityMatrix += gaussianMatrix  # Add the gaussian map to the matrix

        if applyGaussianFilter:
            # Smoothen the probability matrix.
            probabilityMatrix = self.smoothenArray(probabilityMatrix, sigma=gausSTD[::-1])

        # Normalize the probability matrix.
        probabilityMatrix += noise * torch.randn(*probabilityMatrix.size())  # Add random noise
        probabilityMatrix = torch.clamp(probabilityMatrix, min=0, max=None)  # Ensure no negative probabilities
        probabilityMatrix = probabilityMatrix / probabilityMatrix.sum()

        return probabilityMatrix
