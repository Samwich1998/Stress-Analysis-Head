# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class basicEmotionPredictions(nn.Module):
    def __init__(self, numCommonFeatures = 64, emotionLength = 25, numBasicEmotions = 12, numInterpreterHeads = 8):
        super(basicEmotionPredictions, self).__init__()
        # General model parameters.
        self.emotionLength = emotionLength
        self.numBasicEmotions = numBasicEmotions
        self.numInterpreterHeads = numInterpreterHeads
        
        self.numEmotionFeatures = 32
        # Initialize the base model for emotion prediction.
        self.extractEmotionFeatures = nn.Sequential(
                # Neural architecture: Layer 1.
                nn.Linear(numCommonFeatures, 32, bias=True),
                nn.SELU(),
                
                # Neural architecture: Layer 2.
                nn.Linear(32, self.numEmotionFeatures, bias=True),
                nn.BatchNorm1d(32, affine = True, momentum = 0.1, track_running_stats=True),
                nn.SELU(),
        )
                
        # List of list of machine learning modules to train.
        self.allBasicEmotionModels = nn.ModuleList()  # Use ModuleList to store child modules.
        # allBasicEmotionModels dimension: self.numInterpreterHeads, self.numBasicEmotions

        # For each interpretation of the data.
        for interpreterHeadInd in range(self.numInterpreterHeads):
            interpreterModelList = nn.ModuleList()  # Use ModuleList to store child modules.
            
            # For each basic emotion to interpret.
            for basicEmotionInd in range(self.numBasicEmotions):
                # Store an emotion model.
                interpreterModelList.append(nn.Sequential(
                        # Neural architecture: Layer 1.
                        nn.Linear(self.numEmotionFeatures, self.emotionLength, bias=True),
                        nn.SELU(),
                    )
                )
                           
            self.allBasicEmotionModels.append(interpreterModelList)
        
        # Assert the integrity of the model array.
        assert len(self.allBasicEmotionModels) == self.numInterpreterHeads
        assert len(self.allBasicEmotionModels[0]) == self.numBasicEmotions
        
    def forward(self, inputData):
        """ 
        inputData: (batchSize, numFeatures) 
        """
        # Create a holder for the signals.
        allBasicEmotionDistributions = torch.zeros(inputData.shape[0], self.numInterpreterHeads, self.numBasicEmotions, self.emotionLength, device=inputData.device)
        
        # Send the signals through the main model.
        emotionFeatures = self.extractEmotionFeatures(inputData)
        
        # For each interpretation of the data (mindset).
        for interpreterHeadInd in range(self.numInterpreterHeads):            
            # For each basic emotion prediction model (basis state).
            for basicEmotionInd in range(self.numBasicEmotions):

                # Predict the normalized emotion distribution.
                basicEmotionDistribution = self.allBasicEmotionModels[interpreterHeadInd][basicEmotionInd](emotionFeatures)
                # basicEmotionDistribution dimension: batchSize, emotionLength
                
                # Normalize the emotion distribution.
                distributionSign = basicEmotionDistribution.sign() # Keep track of the sign.
                basicEmotionDistribution = distributionSign*F.normalize(basicEmotionDistribution.abs(), dim=1, p=1)
                # basicEmotionDistribution dimension: batchSize, emotionLength

                # Store the basic emotion distributions.
                allBasicEmotionDistributions[:, interpreterHeadInd, basicEmotionInd, :] += basicEmotionDistribution
                # allBasicEmotionDistributions dimension: batchSize, self.numInterpreterHeads, self.numBasicEmotions, self.emotionLength

        return allBasicEmotionDistributions
    
    def printParams(self, numCommonFeatures = 64):
        # basicEmotionPredictions(numCommonFeatures = 64, emotionLength = 25, numBasicEmotions = 6, numInterpreterHeads = 4).to('cpu').printParams()
        summary(self, (numCommonFeatures,))
    
    
    
    
    
        # import matplotlib.pyplot as plt
        # plt.plot(torch.arange(0, 10, 10/self.emotionLength).detach().numpy() - 0.5, allBasicEmotionDistributions[0][0][0].detach().numpy()); 
        # plt.plot(torch.arange(0, 10, 10/self.emotionLength).detach().numpy() - 0.5, allBasicEmotionDistributions[0][1][0].detach().numpy()); 
        # plt.plot(torch.arange(0, 10, 10/self.emotionLength).detach().numpy() - 0.5, allBasicEmotionDistributions[0][0][1].detach().numpy()); 
        # plt.plot(torch.arange(0, 10, 10/self.emotionLength).detach().numpy() - 0.5, allBasicEmotionDistributions[0][1][1].detach().numpy()); 
        # plt.show()
    
    
    
    
    
    
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
        normalizedGausArray = F.normalize(normalizedGausArray, dim=1, p=1)
        
        return normalizedGausArray
    
    
    
                # basicEmotionDistribution = self.createGaussianArray(self.emotionLength, 
                #                                                     numClasses = 10, 
                #                                                     meanGausIndices = basicEmotionParameters[:, 0:self.numGaussians]*10,
                #                                                     gausSTDs = basicEmotionParameters[:, self.numGaussians:self.numGaussians*2]*10,
                #                                                     gausAmps = basicEmotionParameters[:, self.numGaussians*2:self.numGaussians*3]*10,
                #                           )
                # assert not basicEmotionDistribution.isnan().any().item()
        