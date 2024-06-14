
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Plotting
import matplotlib.pyplot as plt

import _generateDataset

sys.path.append(os.path.dirname(__file__) + "/../../")
import equationGenerator # Global model class

# Import pytorch helper files
sys.path.append(os.path.dirname(__file__) + "/../Helpers/")
import _dataLoaderPyTorch

# Import files
sys.path.append(os.path.dirname(__file__) + "/../../Model Helpers/Expression Tree/")
import expressionTreeModel # Global model class

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class ANN(nn.Module):
    def __init__(self, numFeatures, output_size):
        super(ANN, self).__init__()
        # Specify general model parameters
        self.output_size = output_size   # For classification this is numClasses. For regression this is 1.
        self.input_size = numFeatures    # The number of features going into the model.
        self.bias = True                 # Introduce a shift or offset in the activation of each unit or hidden state.
        
        self.dropout = nn.Dropout(p=0.75)
        self.dropout1 = nn.Dropout(p=0.75)

        # Pass the data through the neural architecture: Layer 1.
        self.fc1 = nn.Linear(self.input_size, 16, bias = self.bias)
        self.bn1 = nn.BatchNorm1d(16)  # Batch normalization after the linear layer
        # Pass the data through the neural architecture: Layer 2.
        self.fc2 = nn.Linear(16, 16, bias = self.bias)
        self.bn2 = nn.BatchNorm1d(16)  # Batch normalization after the linear layer
        # Pass the data through the neural architecture: Layer 3.
        self.fc3 = nn.Linear(16, 16, bias = self.bias)
        self.bn3 = nn.BatchNorm1d(16)  # Batch normalization after the linear layer
        # Pass the data through the neural architecture: Layer 4.
        self.fc4 = nn.Linear(16, 8, bias = self.bias)
        self.bn4 = nn.BatchNorm1d(8)  # Batch normalization after the linear layer
        # Pass the data through the neural architecture: Layer 5.
        self.fc5 = nn.Linear(8, 8, bias = self.bias)
        self.bn5 = nn.BatchNorm1d(8)  # Batch normalization after the linear layer
        # Pass the data through the neural architecture: Layer 5.
        self.fc6 = nn.Linear(8, self.output_size)

    def forward(self, inputData):
        inputData = torch.from_numpy(np.asarray(inputData)).float()
        # Pass the data through the neural architecture: Layer 1.
        output = self.fc1(inputData)
        output = self.bn1(output)  # Apply batch normalization
        output = self.dropout(output)
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 2.
        output = self.fc2(output)
        output = self.bn2(output)  # Apply batch normalization
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 3.
        output = self.fc3(output)
        output = self.bn3(output)  # Apply batch normalization
        output = self.dropout1(output)
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 4.
        output = self.fc4(output)
        output = self.bn4(output)  # Apply batch normalization
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 4.
        output = self.fc5(output)
        output = self.bn5(output)  # Apply batch normalization
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 5.
        output = self.fc6(output)
        
        return output

class pytorchPipeline:
    
    def __init__(self, modelType, numFeatures, output_size, featureNames):
        # Specify the CPU or GPU capabilities.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create an instance of the real model.
        self.resetRL(featureNames)
        self.trainingLosses = [[] for _ in range(len(self.equationGeneratorClass.metricTypes))]
        self.testingLosses = [[] for _ in range(len(self.equationGeneratorClass.metricTypes))]
        
        if modelType == "ANN":
            self.model = ANN(numFeatures, output_size)
            
        # Specify the optimizer and loss.
        self.addOptimizer()
        self.addLossFunction()
                    
        # Send the model to GPU/CPU
        self.model.to(self.device)
        
        self.colors = ["k", "tab:blue", "tab:red", "tab:green"]
        
        self.saveWeightInfo = ["fc1", "bn1", "fc2", "bn2", "fc3", "bn3", "fc4", "bn4"]
        
    def copyModelWeights(self, pytorchModel):
        layerInfo = {}
        # For each parameter in the model
        for layerName, layerParams in pytorchModel.model.named_parameters():
            
            # If the layer should be saved.
            if layerName.split(".")[0] in self.saveWeightInfo:
                # Save the layer (bias and weight indivisually)
                layerInfo[layerName] = layerParams.channelData
                
        return layerInfo
                
    def unifyModelWeights(self, allModels, layerInfo):
        # For each model provided.
        for modelInd in range(len(allModels)):  
            pytorchModel = allModels[modelInd]
            
            # For each parameter in the model
            for layerName, layerParams in pytorchModel.model.named_parameters():
                
                # If the layer should be saved.
                if layerName.split(".")[0] in self.saveWeightInfo:
                    assert layerName in layerInfo, print(layerName, layerInfo)
                    layerParams.channelData = layerInfo[layerName]
        
    def addOptimizer(self):
        # Define the optimizer
        adamOptimizer = optim.Adam(
            params = self.model.parameters(), 
            weight_decay = 0.001, # Common values: 10E-6 - 1
            maximize = False,
            amsgrad = True,
            lr=0.001,            # Common values: 0.1 - 0.001
        )
        # Set the optimizer
        self.optimizer = adamOptimizer

    def addLossFunction(self):
        # Specify potential loss functions.
        mseLoss = nn.MSELoss()
        # Save the final loss function
        self.loss_fn = mseLoss
        
    def resetRL(self, featureNames):
        self.equationGeneratorClass = equationGenerator.equationGenerator("metaTrain.pkl", "EGM", featureNames, overwriteModel = True)
        
    def trainModel(self, trainingLoader, testingLoader, featureNames, numEpochs = 300, plotSteps = True):
        # Load in all the data and labels for final predictions
        allTrainingFeatures, allTrainingSolutions = trainingLoader.dataset.getAll()
        allTestingFeatures, allTestingSolutions = testingLoader.dataset.getAll()
        # Make numpy arrays for plotting
        allTrainingFeatures = allTrainingFeatures.detach().numpy()
        allTrainingSolutions = allTrainingSolutions.detach().numpy()
        allTestingFeatures = allTestingFeatures.detach().numpy()
        allTestingSolutions = allTestingSolutions.detach().numpy()
                
        self.model.train()
        # For each training epoch.
        for epoch in range(numEpochs):
            # For each batch of data points.
            for batchInd, data in enumerate(trainingLoader):
                # print("\nEntering Batch")
                # Extract the batch data and labels
                inputFeatures, finalPoints = data
                
                # Zero your gradients for every batch.
                self.optimizer.zero_grad()
                
                # Interface with the equation generation model
                currentExpressionTree, batchDataScores, trueBatchLabels = self.equationGeneratorClass.metaTrain(self.model, inputFeatures, finalPoints, featureNames, maxEpochs = 3)
                trueBatchLabels = torch.as_tensor(trueBatchLabels).float()
                
                # print("Batch Tree")
                # currentExpressionTree.prettyPrint()
                                
                # Calculate the prediction error for this batch.
                predictedBatchLabels = self.model(batchDataScores) 
                batchLoss = self.loss_fn(predictedBatchLabels, trueBatchLabels)
                # Assert the integrity of the loss function.
                assert not batchLoss.isnan().any(), print(batchDataScores, predictedBatchLabels, trueBatchLabels.long(), batchLoss)

                # Backpropogate the gradient.
                batchLoss.backward()    # Calculate the gradients.
                self.optimizer.step()   # Adjust learning weights.
                                
            # Store the loss information.
            finalExpressionTree, _, _ = self.equationGeneratorClass.metaTrain(self.model, allTrainingFeatures, allTrainingSolutions, featureNames, maxEpochs = 1)
            epochLossValues_train = self.equationGeneratorClass.getStateValues(finalExpressionTree, allTrainingFeatures, allTrainingSolutions)
            epochLossValues_test = self.equationGeneratorClass.getStateValues(finalExpressionTree, allTestingFeatures, allTestingSolutions)
            assert len(epochLossValues_train) == len(epochLossValues_test)
            
            for lossInd in range(len(epochLossValues_train)):
                self.trainingLosses[lossInd].append(epochLossValues_train[lossInd])
                self.testingLosses[lossInd].append(epochLossValues_test[lossInd])
            
            # print("Final")
            # finalExpressionTree.prettyPrint()
                    
        if plotSteps:
            # Plot training loss.
            self.plotTrainingLoss()
                        
    def printModelParams(self):
        for name, param in self.model.named_parameters():
            print(name, param.data)
            break
        
    def plotTrainingLoss(self):
        assert len(self.equationGeneratorClass.metricTypes) == len(self.trainingLosses)
        
        # Plot the training loss.
        for lossInd in range(len(self.trainingLosses)):
            plt.plot(self.trainingLosses[lossInd], 'o', color = self.colors[lossInd], label = self.equationGeneratorClass.metricTypes[lossInd])
            plt.plot(self.testingLosses[lossInd], 'o', color = self.colors[lossInd], alpha = 0.4)
        plt.xlabel("Epoch Number")
        plt.ylabel("Metric Loss")
        plt.title("Model Loss Convergence")
        plt.legend()
        plt.show()  
    
    def saveModel(self, modelPath = "model.pth"):
        self.model.eval() # Set the model to evaluation mode
        # Save the model
        torch.save(self.model.state_dict(), modelPath)
        
    def loadModel(self, modelPath):
        # Load the saved model parameters
        self.model.load_state_dict(torch.load(modelPath))
        self.model.eval() # Set the model to evaluation mode
        
def generateFeatureData(numPoints = 1000):
    # Specify input features.
    x = np.random.uniform(-3, 3, numPoints)
    y = np.random.uniform(-3, 3, numPoints)
    z = np.random.uniform(-3, 3, numPoints)
    a = np.random.uniform(-3, 3, numPoints)
    b = np.random.uniform(-3, 3, numPoints)
    c = np.random.uniform(-3, 3, numPoints)
    
    # Compile feature data.
    metaFeatureData = np.array([x, y, z, a, b, c]).T
    metaFeatureNames = np.array(['x', 'y', 'z', 'a', 'b', 'c'])
    
    return metaFeatureData, metaFeatureNames
    

if __name__ == "__main__":  
    # Set random seed
    np.random.seed(1234)
            
    # Initialize the classes.
    equationGeneratorClass = _generateDataset.datasetGenerator()
    expressionTreeModel = expressionTreeModel.expressionTreeModel()
    
    # Read in metadataset and retrieve the equations.
    saveEquationFile = "./metadata/generatedEquations.txt"
    equationLabels = equationGeneratorClass.getEquationList(saveEquationFile)
    
    allModels = []
    allMetalabels = []
    finalEquations = []
    allTreeEquations = []
    allTestingLoaders = []
    allTrainingLoaders = []
    allMetaFeatureData = []
    allMetaFeatureNames = []
    # For each string equation
    for stringEquation in equationLabels[0:200:10]:
        # Generate metadata
        metaFeatureData, metaFeatureNames = generateFeatureData(numPoints = 1000)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # Get the tree format.
                treeEquation = expressionTreeModel.equationInterface(stringEquation, metaFeatureNames)
                metalabels = expressionTreeModel.expressionTreeInterface(treeEquation, metaFeatureData)
        except:
            continue
        
        # Randomly split the data
        Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(metaFeatureData, metalabels, test_size=0.2, shuffle= True)
        
        # Organize the data into the pytorch format.
        pytorchDataClass = _dataLoaderPyTorch.pytorchDataInterface(batch_size=100, num_workers=8, shuffle=True)
        trainingLoader = pytorchDataClass.getDataLoader_variableLength(Training_Data, Training_Labels)
        testingLoader = pytorchDataClass.getDataLoader_variableLength(Testing_Data, Testing_Labels)

        # Initialize and train the model class.
        pytorchModel = pytorchPipeline(modelType = "ANN", numFeatures = 5, output_size = 1, featureNames = metaFeatureNames)
        
        # Store the information.
        allModels.append(pytorchModel)
        allMetalabels.append(metalabels)
        allTreeEquations.append(treeEquation)
        finalEquations.append(stringEquation)
        allTestingLoaders.append(testingLoader)
        allTrainingLoaders.append(trainingLoader)
        allMetaFeatureData.append(metaFeatureData)
        allMetaFeatureNames.append(metaFeatureNames)
    # Unify all the fixed weights in the models
    unifiedLayerData = pytorchModel.copyModelWeights(allModels[0])
    pytorchModel.unifyModelWeights(allModels, unifiedLayerData)
                
    print("\n\nStarting to Train")
    for epoch in range(500):
        print(f"Epoch: {epoch}")
        
        plotSteps = epoch%20 == 0 and epoch != 0
        # For each model
        # for modelInd in range(len(allModels)): 
        for modelInd in range(4): 
            pytorchModel = allModels[modelInd]
            finalEquation = finalEquations[modelInd]
            trainingLoader = allTrainingLoaders[modelInd]
            testingLoader = allTestingLoaders[modelInd]
            metaFeatureNames = allMetaFeatureNames[modelInd]
            # Load in the previous weights.
            pytorchModel.unifyModelWeights([pytorchModel], unifiedLayerData)
            
            # if epoch%50 == 0:
            #     pytorchModel.resetRL(metaFeatureNames)
            
            # Train the model for some epochs.
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    pytorchModel.trainModel(trainingLoader, testingLoader, metaFeatureNames, numEpochs = 5, plotSteps = plotSteps)
            except:
                continue

            # Store the new model weights.
            unifiedLayerData = pytorchModel.copyModelWeights(pytorchModel)
            
            # print(pytorchModel.equationGeneratorClass.stateActionRewards_Root[1] / pytorchModel.equationGeneratorClass.numActionsTaken_Root[1])
            


    
    
    