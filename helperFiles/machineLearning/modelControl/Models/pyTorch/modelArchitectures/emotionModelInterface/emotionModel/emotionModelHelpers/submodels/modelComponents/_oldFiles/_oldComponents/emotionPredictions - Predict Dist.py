# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class basicEmotionPredictions(nn.Module):
    def __init__(self, inputDimension, emotionLength, numBasicEmotions, numInterpreterHeads, numSignals):
        super(basicEmotionPredictions, self).__init__()
        # General model parameters.
        self.numInterpreterHeads = numInterpreterHeads
        self.emotionLength = emotionLength
        self.numBasicEmotions = numBasicEmotions
        self.numSignals = numSignals
                
        # List of machine learning modules to train.
        self.allBasicEmotionModels = nn.ModuleList()  # Use ModuleList to store child modules.
        # allBasicEmotionModels dimension: self.numInterpreterHeads, self.numBasicEmotions
        
        self.numGaussians = 1
        self.numParamsDist = 3 * self.numGaussians
        # For each interpretation of the data.
        for interpreterHeadInd in range(self.numInterpreterHeads):
            interpreterModelList = nn.ModuleList()  # Use ModuleList to store child modules.
            
            # For each basic emotion to interpret.
            for modelInd in range(self.numBasicEmotions):
                # Store aan emotion model.
                interpreterModelList.append( 
                    # Create model.
                    nn.Sequential(
                        # Neural architecture: Layer 1.
                        nn.Linear(inputDimension, 128, bias=True),
                        nn.BatchNorm1d(128, track_running_stats=True),
                        nn.GELU(),
                        nn.Dropout(0.5),
                        
                        # Neural architecture: Layer 2.
                        nn.Linear(128, 64, bias=True),
                        nn.BatchNorm1d(64, track_running_stats=True),
                        nn.GELU(),
                        nn.Dropout(0.4),
                        
                        # Neural architecture: Layer 3.
                        # nn.Linear(64, 32, bias=True),
                        # nn.BatchNorm1d(32, track_running_stats=True),
                        # nn.GELU(),
                        # nn.Dropout(0.3),
                        
                        # # Neural architecture: Layer 4.
                        # nn.Linear(32, 16, bias=True),
                        # nn.BatchNorm1d(16, track_running_stats=True),
                        # nn.GELU(),
                        
                        # Neural architecture: Layer 5.
                        nn.Linear(64, self.emotionLength*self.numGaussians, bias=True),
                    )
                )
                           
            self.allBasicEmotionModels.append(interpreterModelList)
        
        # Assert the integrity of the model array.
        assert len(self.allBasicEmotionModels) == self.numInterpreterHeads
        assert len(self.allBasicEmotionModels[0]) == self.numBasicEmotions
        
    def forward(self, inputData, allSignalWeights, batchSize):
        """ 
        inputData: (batchSize*numSignals, compressedLength) 
        allSubjectWeights: (batchSize, self.numInterpreterHeads, 1, 1) 
        allSignalWeights: (self.numInterpreterHeads, self.numBasicEmotions, 1, self.numSignals, 1) 
        """
        # Create a holder for the signals.
        allBasicEmotionDistributions = torch.zeros(batchSize, self.numInterpreterHeads, self.numBasicEmotions, self.emotionLength)
        
        # For each interpretation of the data (mindset).
        for interpreterHeadInd in range(self.numInterpreterHeads):            
            # For each basic emotion prediction model (basis state).
            for basicEmotionInd in range(self.numBasicEmotions):

                # Predict the normalized emotion distribution.
                basicEmotionParameters = self.allBasicEmotionModels[interpreterHeadInd][basicEmotionInd](inputData)
                
                # basicEmotionDistribution = self.createGaussianArray(self.emotionLength, 
                #                                                     numClasses = 10, 
                #                                                     meanGausIndices = basicEmotionParameters[:, 0:self.numGaussians]*10,
                #                                                     gausSTDs = basicEmotionParameters[:, self.numGaussians:self.numGaussians*2]*10,
                #                                                     gausAmps = basicEmotionParameters[:, self.numGaussians*2:self.numGaussians*3]*10,
                #                           )
                # assert not basicEmotionDistribution.isnan().any().item()
               
                basicEmotionDistribution = F.softmax(basicEmotionParameters, dim=1) # Normalize the distribution.
                # basicEmotionDistribution dimension: batchSize*numSignals, emotionLength
                assert (basicEmotionDistribution >= 0).all()
              
                # import matplotlib.pyplot as plt
                # plt.plot(torch.arange(0, 10, 10/self.emotionLength).detach().numpy() - 0.5, basicEmotionDistribution[0].detach().numpy());

                # Recombine the basic emotions into their respective batches.
                basicEmotionDistribution = basicEmotionDistribution.view(batchSize, self.numSignals, self.emotionLength)
                # basicEmotionDistribution dimension: batchSize, numSignals, self.emotionLength

                # Apply a weighted average to recombine the basic emotions.
                basicEmotionDistribution = (basicEmotionDistribution * allSignalWeights[interpreterHeadInd][basicEmotionInd]).sum(dim=1)
                # basicEmotionDistribution dimension: batchSize, self.emotionLength
                assert (basicEmotionDistribution >= 0).all(), allSignalWeights[interpreterHeadInd][basicEmotionInd]
                
                # import matplotlib.pyplot as plt
                # plt.plot(torch.arange(0, 10, 10/self.emotionLength).detach().numpy() - 0.5, basicEmotionDistribution[0].detach().numpy()); plt.show()
                
                # Store the basic emotion distributions.
                allBasicEmotionDistributions[:, interpreterHeadInd, basicEmotionInd, :] += basicEmotionDistribution
                # allBasicEmotionDistributions dimension: batchSize, self.numInterpreterHeads, self.numBasicEmotions, self.emotionLength
        
        
    
        import matplotlib.pyplot as plt
        plt.plot(torch.arange(0, 10, 10/self.emotionLength).detach().numpy() - 0.5, basicEmotionDistribution[0].detach().numpy()); plt.show()
            
        
        return allBasicEmotionDistributions
    
    def createGaussianArray(self, arrayLength, numClasses, meanGausIndices, gausSTDs, gausAmps, eps=1E-20):
        assert numClasses - 1 < arrayLength, f"You cannot have {numClasses} classes with an array length of {arrayLength}"
        assert (meanGausIndices.shape == gausSTDs.shape == gausAmps.shape), \
                    f"Specify all the stds ({gausSTDs}) for every mean ({meanGausIndices})"
        
        batchSize, numGaussians = meanGausIndices.size()
        
        # Create an array of n elements
        samplingFreq = arrayLength / numClasses
        gaussianArrayInds = torch.arange(arrayLength).unsqueeze(0) - samplingFreq * 0.5
        
        # Bound the gaussian parameters
        relativeAmplitudes = torch.clamp(gausAmps, min = eps)
        relativeStds = torch.clamp(samplingFreq * gausSTDs, min = 0.01)
        relativeMeanInds = torch.clamp(meanGausIndices, min = -0.5, max = numClasses + 0.5)*samplingFreq
        # Normalize the amplitudes
        relativeAmplitudes = relativeAmplitudes / relativeAmplitudes.sum(dim=1, keepdim=True)
        
        # Generate Gaussian distribution using broadcasting and element-wise operations
        gaussianExponents = -0.5 * ((gaussianArrayInds - relativeMeanInds.unsqueeze(2)) / relativeStds.unsqueeze(2)) ** 2
        gaussianDistribution = gaussianExponents.exp() * relativeAmplitudes.unsqueeze(2)
        
        normalizedGausArray = gaussianDistribution.sum(dim=1)
        normalizedGausArray = F.softmax(normalizedGausArray, dim=1)
        
        return normalizedGausArray


class emotionClassification(nn.Module):
    def __init__(self, allFeatures, allNumAnswers):
        super(emotionClassification, self).__init__()
        self.numEmotionModels = len(allNumAnswers)
        self.emotionModels = nn.ModuleList()  # Use ModuleList to store child modules.

        for modelInd in range(self.numEmotionModels):
            # Classify emotions.
            self.emotionModels.append(
                nn.Sequential(
                    # Neural architecture: Layer 1.
                    nn.Linear(allFeatures, 8, bias = True),
                    nn.BatchNorm1d(8, track_running_stats = True),
                    nn.GELU(),
                    #nn.Dropout(0.3),

                    # Neural architecture: Layer 3.
                    nn.Linear(8, allNumAnswers[modelInd], bias = True),
                    # nn.Linear(32, 2, bias = True),
                )
            )
    
    def forward(self, signalFeatures):
        emotionProbabilities = []
        # For each emotion prediction model.
        for modelInd in range(self.numEmotionModels):
            emotionProbability = self.emotionModels[modelInd](signalFeatures)
            
            # For classification problems
            if self.lastLayer == "logSoftmax":
                emotionProbability = F.log_softmax(emotionProbability, dim=1)  # Apply log-softmax activation to get class probabilities.
            if self.lastLayer == "softmax":
                emotionProbability = F.softmax(emotionProbability, dim=1)  # Apply log-softmax activation to get class probabilities.
            
            emotionProbabilities.append(emotionProbability)

        return emotionProbabilities
    
    
    
        