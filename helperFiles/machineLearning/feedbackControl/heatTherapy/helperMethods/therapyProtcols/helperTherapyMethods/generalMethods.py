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
    def createGaussianMap(allParameterBins, predictionBins, gausMean, gausSTD):
        # Generate a grid for Gaussian distribution calculations
        x, y = torch.meshgrid(predictionBins, allParameterBins)

        # Calculate Gaussian distribution values across the grid
        gaussMatrix = torch.exp(-0.5 * ((x - gausMean[0]) ** 2 / gausSTD[0] ** 2 + (y - gausMean[1]) ** 2 / gausSTD[1] ** 2))
        gaussMatrix = gaussMatrix / gaussMatrix.sum()  # Normalize the Gaussian matrix

        return gaussMatrix

    @staticmethod
    def separateUneven2DArray(inputArray, index):
        return inputArray[index]

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
        prob_matrix_PA = probabilityMatrix[0]
        prob_matrix_NA = probabilityMatrix[1]
        prob_matrix_SA = probabilityMatrix[2]

        # Calculate the probability matrix.
        for initialDataPoints in initialData:
            currentUserTemp = initialDataPoints[0] # within loop: torch.Size([1, 1])
            currentUserLoss = initialDataPoints[1:4] # within loop: torch.Size([3, 1. 1])

            if applyGaussianFilter:
                # Generate a delta function probability.
                tempBinIndex = self.dataInterface.getBinIndex(allParameterBins, currentUserTemp)
                #print('allPredictionBins: ', allPredictionBins)
                # separate out uneven 2D arrays
                PA_list_separated = generalMethods.separateUneven2DArray(allPredictionBins, 0)
                NA_list_separated = generalMethods.separateUneven2DArray(allPredictionBins, 1)
                SA_list_separated = generalMethods.separateUneven2DArray(allPredictionBins, 2)

                lossBinIndex_PA = self.dataInterface.getBinIndex(PA_list_separated, currentUserLoss)
                lossBinIndex_NA = self.dataInterface.getBinIndex(NA_list_separated, currentUserLoss)
                lossBinIndex_SA = self.dataInterface.getBinIndex(SA_list_separated, currentUserLoss)
                prob_matrix_PA[tempBinIndex][lossBinIndex_PA] += 1  # map out bins and fill out with discrete values
                prob_matrix_NA[tempBinIndex][lossBinIndex_NA] += 1  # map out bins and fill out with discrete values
                prob_matrix_SA[tempBinIndex][lossBinIndex_SA] += 1  # map out bins and fill out with discrete values
            else:
                # separate out uneven 2D arrays
                PA_list_separated = generalMethods.separateUneven2DArray(allPredictionBins, 0)
                NA_list_separated = generalMethods.separateUneven2DArray(allPredictionBins, 1)
                SA_list_separated = generalMethods.separateUneven2DArray(allPredictionBins, 2)
                # Generate 2D gaussian matrix.
                gaussianMatrix_PA = self.createGaussianMap(allParameterBins, PA_list_separated, gausMean=(currentUserLoss, currentUserTemp), gausSTD=gausSTD)
                gaussianMatrix_NA = self.createGaussianMap(allParameterBins, NA_list_separated, gausMean=(currentUserLoss, currentUserTemp), gausSTD=gausSTD)
                gaussianMatrix_SA = self.createGaussianMap(allParameterBins, SA_list_separated, gausMean=(currentUserLoss, currentUserTemp), gausSTD=gausSTD)
                prob_matrix_PA += gaussianMatrix_PA  # Add the gaussian map to the matrix
                prob_matrix_NA += gaussianMatrix_NA  # Add the gaussian map to the matrix
                prob_matrix_SA += gaussianMatrix_SA  # Add the gaussian map to the matrix

        if applyGaussianFilter:
            # Smoothen the probability matrix.
            prob_matrix_PA = self.smoothenArray(prob_matrix_PA, sigma=gausSTD[::-1])
            prob_matrix_NA = self.smoothenArray(prob_matrix_NA, sigma=gausSTD[::-1])
            prob_matrix_SA = self.smoothenArray(prob_matrix_SA, sigma=gausSTD[::-1])

        # Normalize the probability matrix.
        prob_matrix_PA += noise * torch.randn(*prob_matrix_PA.size())  # Add random noise
        prob_matrix_NA += noise * torch.randn(*prob_matrix_NA.size())  # Add random noise
        prob_matrix_SA += noise * torch.randn(*prob_matrix_SA.size())  # Add random noise
        prob_matrix_PA = torch.clamp(prob_matrix_PA, min=0, max=None)  # Ensure no negative probabilities
        prob_matrix_NA = torch.clamp(prob_matrix_NA, min=0, max=None)  # Ensure no negative probabilities
        prob_matrix_SA = torch.clamp(prob_matrix_SA, min=0, max=None)  # Ensure no negative probabilities

        prob_matrix_PA = prob_matrix_PA / prob_matrix_PA.sum()
        prob_matrix_NA = prob_matrix_NA / prob_matrix_NA.sum()
        prob_matrix_SA = prob_matrix_SA / prob_matrix_SA.sum()

        # TODO: combine three probability matrices


        return probabilityMatrix



