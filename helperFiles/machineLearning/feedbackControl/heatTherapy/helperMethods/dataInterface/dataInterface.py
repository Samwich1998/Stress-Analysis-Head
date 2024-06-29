# General
import torch


class dataInterface:

    def __init__(self, predictionWeights, optimalNormalizedState):
        # General parameters
        self.optimalNormalizedState = optimalNormalizedState
        self.predictionWeights = predictionWeights
        self.numPredictions = 3 # PA, NA, SA
        self.numParameters = 1 # 1 for heat therapy, predicting 1 temperature
        self.singleBatch = 1 # numTrue sample processed in a single batch (with all the same weights and other parameters)

    def calculateCompiledLoss(self, initialMentalStates):
        """ initialStates: numPoints, (PA, NA, SA); 2D array"""
        initialMentalStates = torch.as_tensor(initialMentalStates) # dim = numPoints, (PA, NA, SA), 1, 1; torch.Size([30, 3, 1, 1])
        predictionWeights_tensor = torch.as_tensor(self.predictionWeights)
        predictionWeights_tensor = predictionWeights_tensor.view(self.singleBatch, self.numPredictions, 1, 1) # dim = 1, 3, 1, 1; torch.Size([1, 3, 1, 1])
        self.optimalNormalizedState = self.optimalNormalizedState.view(self.singleBatch, self.numPredictions, 1, 1) # dim = 1, 3, 1, 1; torch.Size([1, 3, 1, 1])
        # print the sizes for perdictionWeights and optimalNormalizedState and initialMentalStates

        # Calculate the compiled loss.
        compiledLoss = predictionWeights_tensor * (initialMentalStates - self.optimalNormalizedState).pow(2) # weighted MSE; torch.Size([30, 3, 1, 1])

        compiledLoss = compiledLoss.sum(dim=1) # dim = numPoints, 1, 1; torch.Size([30, 1, 1])

        # Normalize the loss values.
        compiledLoss = compiledLoss / compiledLoss.sum() # dim = numPoints, 1, 1; torch.Size([30, 1, 1])
        # compiledLoss dimension: numPoints

        return compiledLoss

    @staticmethod
    def multiDimensionalParameterInterface(allParameterBounds):
        # Ensure the input is a torch array.
        allParameterBounds = torch.as_tensor(allParameterBounds)

        # Multidimensional interface.
        if allParameterBounds.ndim == 1:
            assert len(allParameterBounds) == 2, f"Parameter bounds must be a list of length 2, found {allParameterBounds}"
            assert len(allParameterBounds) == 2, f"Parameter bounds must be a list of length 2, found {allParameterBounds}"
            initialLowerBounds, initialUpperBounds = allParameterBounds
        else:
            assert allParameterBounds.size(1) == 2, f"Each row of parameter bounds must be of length 2, found {allParameterBounds}"
            initialLowerBounds, initialUpperBounds = allParameterBounds[:, 0], allParameterBounds[:, 1]

        return initialLowerBounds, initialUpperBounds

    @staticmethod
    def boundParameter(allParameterBounds, newValues):
        # Extract the lower and upper bounds from the bounds, accounting for dimension.
        initialLowerBounds, initialUpperBounds = dataInterface.multiDimensionalParameterInterface(allParameterBounds)

        # Apply bounds to newValues
        boundedValues = torch.max(initialLowerBounds, torch.min(initialUpperBounds, newValues))

        return boundedValues

    @staticmethod
    def normalizeParameters(currentParamBounds, normalizedParamBounds, currentParamValues):
        # Extract the lower and upper bounds from the bounds, accounting for dimension.
        currentLowerBounds, currentUpperBounds = dataInterface.multiDimensionalParameterInterface(currentParamBounds)
        normalizedLowerBounds, normalizedUpperBounds = dataInterface.multiDimensionalParameterInterface(normalizedParamBounds)

        # Apply bounds to currentValues.
        normalizedValues = (currentParamValues - currentLowerBounds) / (currentUpperBounds - currentLowerBounds)  # Normalize between [0, 1]
        normalizedValues = normalizedLowerBounds + normalizedValues*(normalizedUpperBounds - normalizedLowerBounds)  # Normalize between [normalizedLowerBounds, normalizedUpperBounds]

        return normalizedValues

    @staticmethod
    def unNormalizeParameters(normalizedParamBounds, originalParamBounds, currentParamValues):
        return dataInterface.normalizeParameters(currentParamBounds=normalizedParamBounds, normalizedParamBounds=originalParamBounds, currentParamValues=currentParamValues)

    @staticmethod
    def getBinIndex(allBins, binValue):
        # Ensure allBins is a PyTorch tensor
        allBins = torch.as_tensor(allBins, dtype=torch.float32) # torch.Size([1, 4])
        binValue = torch.as_tensor([binValue]) # torch.Size([1])

        # Find the index where binValue should be inserted
        # ensure dimension matches
        allBins = allBins.squeeze() # torch.Size([4])
        binIndex = torch.searchsorted(allBins, binValue, right=True).item()
        binIndex = min(binIndex, len(allBins) - 1)

        return binIndex

    @staticmethod
    def initializeAllBins(allBounds, binWidths):
        # Base case: if binWidths is a 1D tensor.
        if allBounds.ndim == 1:
            allBounds = allBounds.unsqueeze(0).expand(len(binWidths), len(allBounds))
        assert allBounds.size(0) == binWidths.size(0), f"Bounds and binWidths must have the same length, found {allBounds.size(0)} and {len(binWidths)}"

        # Extract the lower and upper bounds from the bounds, accounting for dimension.
        currentLowerBounds, currentUpperBounds = dataInterface.multiDimensionalParameterInterface(allBounds)

        finalBins = []
        # For each parameter's bins information, initialize the bins.
        for lower, upper, binWidth in zip(currentLowerBounds, currentUpperBounds, binWidths):
            currentBins = dataInterface.initializeBins(bounds=[lower, upper], binWidth=binWidth)

            # Store the results.
            finalBins.append(currentBins.tolist())

        return finalBins

    @staticmethod
    def initializeBins(bounds, binWidth):
        # Generate the temperature bins
        finalBins = torch.arange(bounds[0], bounds[1], step=binWidth)

        # Edge case: the last bin must be the upper bound
        if finalBins[-1] != bounds[1]:
            finalBins = torch.cat((finalBins, torch.tensor([bounds[1]])))

        return finalBins
