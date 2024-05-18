
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
# PyTorch
import torch
import torch.optim as optim
import torch.nn.functional as F

# Plotting
import matplotlib.pyplot as plt

# Import data aquisition and analysis files
sys.path.append(os.path.dirname(__file__) + "/Helpers/") 
import _lossFunctions

# Import models
sys.path.append(os.path.dirname(__file__) + "/Model Architectures/") 
import pytorchGRU
import pytorchANN
import pytorchLSTM
import compiledGRUs

# Import models
sys.path.append(os.path.dirname(__file__) + "/Model Architectures/Transformers/") 
import emotionDecoder

# -------------------------------------------------------------------------- #
# ---------------------------- Pytorch Pipeline ---------------------------- #

class pytorchPipeline:
    
    def __init__(self, modelInd, modelType, output_size, input_size = None, numSignals = None, emotionNames = None, featureNames = [], saveDataFolder = None, metaTraining = True):
        # Specify the CPU or GPU capabilities.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lossType = "weightedKL" # "weightedKL", "NLLLoss", "KLDivLoss", "CrossEntropyLoss", "diceLoss", "FocalLoss"
        
        # Predict probabilities for classification problems
        if self.lossType in ["weightedKL", "NLLLoss", "KLDivLoss"]:
            lastLayer = "logSoftmax"
        elif self.lossType in ["CrossEntropyLoss", "diceLoss", "FocalLoss"]:
            lastLayer = "softmax"
        else:
            lastLayer = None
            assert output_size == 1, "I may be assuming this output_size 1 is only for regression"
                
        # Create the ML model
        if modelType == "LSTM":
            self.model = pytorchLSTM.LSTM(output_size, numSignals)
        elif modelType == "GRU":
            self.model = pytorchGRU.GRU(output_size, numSignals)
        elif modelType == "ANN":
            self.model = pytorchANN.ANN(output_size, input_size)
        elif modelType == "compiledGRU":
            self.model = compiledGRUs.compiledGRUs(output_size, numSignals, metaTraining = metaTraining)
        elif modelType == "transformer":
            self.model = emotionDecoder.modelHead(output_size, numSignals, emotionNames = emotionNames, maxSeqLength = 120, metaTraining = metaTraining)
            self.model.classifyEmotions.lastLayer = lastLayer
            
            self.reconstructionLoss = _lossFunctions.pytorchLossMethods(lossType = "MeanSquaredError").loss_fn
        # Apply information on the last layer
        self.model.lastLayer = lastLayer
    
        # If we have a regression problem
        if output_size == 1:
            self.numClasses = 0
            self.loss_fn = _lossFunctions.pytorchLossMethods(lossType = self.lossType).loss_fn
        # If we have a classification problem
        else:
            self.numClasses = output_size
        # Specify the optimizer and loss.
        self.addOptimizer()
            
        # Send the model to GPU/CPU
        self.model.to(self.device)
        
        # Create folders to the save the data in.
        self.saveDataFolder = saveDataFolder + ("metatraining/" if metaTraining else "training/")
        self.saveModelFolder = os.path.join(os.path.dirname(__file__), r"../../../_finalModels/")
        # Create the folders if they do not exist.
        os.makedirs(self.saveDataFolder, exist_ok=True)
        os.makedirs(self.saveModelFolder, exist_ok=True)
        
        # Model identifiers.
        self.modelInd = modelInd   
        self.featureNames = featureNames
        self.classAttibuteString = "_class_attributes"
        self.modelInfoString = f"_modelNum{self.modelInd}"
        
        # Reset the model's variable parameters
        self.resetModel()
    
    def resetModel(self):
        # Setup training holders.
        self.trainingLosses = []
        self.testingLosses = []
        self.trainingLosses_autoencoder = []
        self.testingLosses_autoencoder = []
        
    def addOptimizer(self):
        # Define the optimizer
        adamOptimizer = optim.Adam(
            params = self.model.parameters(), 
            weight_decay = 0.005, # Common values: 10E-6 - 1
            maximize = False,
            amsgrad = True,
            lr=0.002,            # Common values: 0.1 - 0.001
        )
        # Set the optimizer
        self.optimizer = adamOptimizer
        
    def trainModel(self, trainingLoader, testingLoader, emotionQuestion, numEpochs = 300, metaTraining = True, plotSteps = True):
        # Load in all the data and labels for final predictions
        allTrainingData, allTrainingLabels = trainingLoader.dataset.getAll()
        allTestingData, allTestingLabels = testingLoader.dataset.getAll()
        # If classification.
        if self.numClasses != 0:
            # Take into consideration the class weights.
            self.trainingClassWeights = self.assignClassWeights(allTrainingLabels, self.numClasses)
            self.testingClassWeights = self.assignClassWeights(allTestingLabels, self.numClasses)
            classWeights = 0.8*self.trainingClassWeights + 0.2*self.testingClassWeights
            self.loss_fn = _lossFunctions.pytorchLossMethods(lossType = self.lossType, class_weights = classWeights).loss_fn
            # Apply the gaussian filter to the data
            allTestingLabels = self.gausEncoding(allTestingLabels)
            allTrainingLabels = self.gausEncoding(allTrainingLabels)

        # For each training epoch.
        for epoch in range(numEpochs):
            
            self.model.train()
            # For each batch of data points.
            for batchInd, data in enumerate(trainingLoader):
                # Extract the batch data and labels
                batchData, trueBatchLabels = data
                if self.numClasses != 0:
                    trueBatchLabels = self.gausEncoding(trueBatchLabels)
                # Zero your gradients for every batch.
                self.optimizer.zero_grad()
                
                if metaTraining:
                    # Calculate the predictions for this batch.
                    predictedBatchLabels, reconstructedBatchData = self.model(batchData, trainingData = True)
                
                    # Calculate the error in both predictions
                    batchLoss = self.loss_fn(predictedBatchLabels, trueBatchLabels.float())
                    reconstructedLoss = self.reconstructionLoss(reconstructedBatchData, batchData.float())
                    # Assert the integrity of the loss function.
                    assert not batchLoss.isnan().any() or not batchLoss.isinf().any(), \
                        print(predictedBatchLabels, trueBatchLabels.float(), batchLoss)
                    assert not reconstructedLoss.isnan().any() or not reconstructedLoss.isinf().any(), \
                        print(reconstructedBatchData, batchData.float(), reconstructedLoss)
                    # Average the loss together
                    finalLoss = batchLoss + reconstructedLoss
                else:
                    # Calculate the predictions for this batch.
                    predictedBatchLabels = self.model.classifyEmotions(batchData.float())
                
                    # Calculate the error in the emotion predictions
                    batchLoss = self.loss_fn(predictedBatchLabels, trueBatchLabels.float())
                    assert not batchLoss.isnan().any() or not batchLoss.isinf().any(), \
                        print(predictedBatchLabels, trueBatchLabels.float(), batchLoss)
                    # Average the loss together
                    finalLoss = batchLoss
                
                # Backpropogate the gradient.
                finalLoss.backward()    # Calculate the gradients.
                self.optimizer.step()   # Adjust the weights.
            
            self.model.eval()
            # Stop gradient tracking.
            with torch.no_grad():
                if metaTraining:
                    # Predict the training and testing data.
                    predictedTestingLabels, reconstructedTestingData = self.model(allTestingData, trainingData = True, allTrainingData = False)
                    predictedTrainingLabels, reconstructedTrainingData = self.model(allTrainingData, trainingData = True, allTrainingData = True)
                    
                    # Calculate the training and testing loss.
                    epochTrainingLoss_autoencoder = self.reconstructionLoss(reconstructedTrainingData, allTrainingData.float()).mean(axis=0).detach().clone()
                    epochTestingLoss_autoencoder = self.reconstructionLoss(reconstructedTestingData, allTestingData.float()).mean(axis=0).detach().clone()
                    # Store the loss information.
                    self.testingLosses_autoencoder.append(epochTestingLoss_autoencoder)
                    self.trainingLosses_autoencoder.append(epochTrainingLoss_autoencoder)
                else:
                    # Predict the training and testing data.
                    predictedTestingLabels = self.model.classifyEmotions(allTestingData)
                    predictedTrainingLabels = self.model.classifyEmotions(allTrainingData)
                    
                # Calculate the training and testing loss.
                epochTestingLoss = self.loss_fn(predictedTestingLabels, allTestingLabels.float()).mean(axis=0).detach().clone()
                epochTrainingLoss = self.loss_fn(predictedTrainingLabels, allTrainingLabels.float()).mean(axis=0).detach().clone()
                # Store the loss information.
                self.testingLosses.append(epochTestingLoss.item())
                self.trainingLosses.append(epochTrainingLoss.item())
            
        if plotSteps:
            # Get the class information from the testing and training data.
            allTestingClasses = allTestingLabels.argmax(dim=1).detach().numpy()
            allTrainingClasses  = allTrainingLabels.argmax(dim=1).detach().numpy()
            allPredictedTestingClasses = predictedTestingLabels.argmax(dim=1).detach().numpy()
            allPredictedTrainingClasses = predictedTrainingLabels.argmax(dim=1).detach().numpy()
            # Get all the data predictions.
            # self.plotPredictions(allTrainingClasses, allTestingClasses, allPredictedTrainingClasses, 
            #                      allPredictedTestingClasses, emotionQuestion)
            self.plotPredictedMatrix(allTrainingClasses, allTestingClasses, allPredictedTrainingClasses, 
                                     allPredictedTestingClasses, emotionQuestion)
            # Plot model convergence curves.
            self.plotTrainingLoss(self.trainingLosses, self.testingLosses, plotTitle = "Emotion Convergence Loss")
            if metaTraining:
                # Plot model convergence curves.
                self.plotTrainingLoss(self.trainingLosses_autoencoder, self.testingLosses_autoencoder, plotTitle = "Autoencoder Convergence Loss")
                # Plot the autoencoder results.
                self.plotAutoencoder(allTestingData, reconstructedTestingData, plotTitle = "Autoencoder Test Prediction")
                self.plotAutoencoder(allTrainingData, reconstructedTrainingData, plotTitle = "Autoencoder Training Prediction")
                
    def printModelParams(self):
        for name, param in self.model.named_parameters():
            print(name, param.data)
            break
            
    def plotPredictions(self, allTrainingLabels, allTestingLabels, allPredictedTrainingLabels, allPredictedTestingLabels, plotTitle = "Emotion Prediction"):
        # Plot the data correlation.
        plt.plot(allPredictedTrainingLabels, allTrainingLabels, 'ko', markersize=6, alpha = 0.3,label = "Training Points")
        plt.plot(allPredictedTestingLabels, allTestingLabels, '*', color = 'tab:blue', markersize=6, alpha = 0.6, label = "Testing Poitns")
        plt.xlabel("Predicted Emotion Rating")
        plt.ylabel("Emotion Rating")
        plt.title(f"{plotTitle}")
        plt.legend(loc="best")
        plt.xlim((-0.1, self.numClasses-0.9))
        plt.ylim((-0.1, self.numClasses-0.9))
        plt.show()
                    
    def plotAutoencoder(self, inputData, outputData, plotTitle = "Autoencoder Prediction"):
        
        batchInd = 0
        for signalInd in range(inputData.size(2)):
            # Plot the signal reconstruction.
            plt.plot(inputData[batchInd, :, signalInd], 'k', linewidth=2, alpha = 0.1, label = "Initial Signal")
            plt.plot(outputData[batchInd, :, signalInd], 'tab:blue', linewidth=2, alpha = 0.6, label = "Reconstructed Signal")
            plt.xlabel("Points")
            plt.ylabel("Signal (AU)")
            plt.title(f"{plotTitle}")
            plt.legend(loc="best")
            if self.saveDataFolder:
                plt.savefig(self.saveDataFolder + f"{plotTitle} epochs = {len(self.trainingLosses_autoencoder)} signalInd = {signalInd}.png")
            plt.show()
            
            if signalInd == 2:break
                
    def plotPredictedMatrix(self, allTrainingLabels, allTestingLabels, allPredictedTrainingLabels, allPredictedTestingLabels, plotTitle="Emotion Prediction"):
        # Calculate confusion matrices
        training_confusion_matrix = confusion_matrix(allTrainingLabels, allPredictedTrainingLabels, labels=np.arange(self.numClasses), normalize = 'true')
        testing_confusion_matrix = confusion_matrix(allTestingLabels, allPredictedTestingLabels, labels=np.arange(self.numClasses), normalize = 'true')
    
        # Plot the confusion matrices as heatmaps
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(training_confusion_matrix, cmap='Blues', vmin=0, vmax=1)
        plt.title('Training Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.colorbar(format='%.2f')
        plt.gca().invert_yaxis()  # Reverse the order of y-axis ticks
    
        # Display percentages in the boxes
        for i in range(self.numClasses):
            for j in range(self.numClasses):
                plt.text(j, i, f'{training_confusion_matrix[i, j] * 100:.2f}%', ha='center', va='center', color='black')
    
        plt.subplot(122)
        plt.imshow(testing_confusion_matrix, cmap='Blues', vmin=0, vmax=1)
        plt.title('Testing Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.colorbar(format='%.2f')
    
        # Display percentages in the boxes
        for i in range(self.numClasses):
            for j in range(self.numClasses):
                plt.text(j, i, f'{testing_confusion_matrix[i, j] * 100:.2f}%', ha='center', va='center', color='black')
                
        plt.gca().invert_yaxis()  # Reverse the order of y-axis ticks
        plt.tight_layout()
        if self.saveDataFolder:
            plt.savefig(self.saveDataFolder + f"{plotTitle} epochs = {len(self.trainingLosses)}.png")
        plt.show()

        
    def plotTrainingLoss(self, trainingLosses, testingLosses, plotTitle = "Model Convergence Loss"):
        # Plot the training loss.
        plt.plot(trainingLosses, color = 'k', linewidth = 2, label = "Training Loss")
        plt.plot(testingLosses, color = 'tab:red', linewidth = 2, label = "Testing Loss")
        plt.legend(loc="upper right")
        plt.ylabel("Training Loss")
        plt.xlabel("Epoch")
        plt.title(f"{plotTitle}")
        if self.saveDataFolder:
            plt.savefig(self.saveDataFolder + f"{plotTitle} epochs = {len(trainingLosses)}.png")
        plt.show()  

    def assignClassWeights(self, classLabels, numClasses):
        # Count the occurrences of each class
        class_counts = torch.bincount(classLabels.to(torch.int64), minlength=numClasses)
        # Calculate the total number of samples
        total_samples = torch.sum(class_counts)
        # Calculate the class weights
        class_weights = total_samples / (class_counts.float() + 1e-60)  # Add small constant to avoid division by zero
        
        # Normalize the class weights
        class_weights[class_weights == torch.inf] = 0
        class_weights /= (class_weights != torch.inf).sum()
    
        return class_weights
    
    def oneHotEncoding(self, classLabels):
        return F.one_hot(classLabels.long(), num_classes=self.numClasses).float()
    
    def gausEncoding(self, classLabels):
        finalLabels = np.zeros((len(classLabels), self.numClasses))
        for finalLabelInd in range(len(classLabels)):
            classInd = int(classLabels[finalLabelInd])
            finalLabels[finalLabelInd] = self.create_gaussian_array(self.numClasses, classInd, 0.3)
        return torch.tensor(finalLabels)
    
    def create_gaussian_array(self, numElements, meanIndex, std_dev):
        # Create an array of n elements
        arr = np.arange(numElements)  

        # Generate Gaussian distribution
        gaussian_values = np.exp(-0.5 * ((arr - meanIndex) / std_dev) ** 2)
        normalized_values = gaussian_values / np.sum(gaussian_values)  # Normalize the values to sum up to 1

        return normalized_values

    def _saveModel(self, modelFileName = "model.pth", appendedFolder = None, saveModelAttributes = True):
        self.model.eval()  # Set the model to evaluation mode
        saveModelFolder = os.path.normpath(self.saveModelFolder + "" if appendedFolder is None else appendedFolder)

        # Save the model
        os.makedirs(saveModelFolder, exist_ok=True)
        saveModelPath = os.path.normpath(saveModelFolder + modelFileName + self.modelInfoString + ".pkl")
        torch.save(self.model.state_dict(), saveModelPath)
        
        if saveModelAttributes:
            # Save the entire class along with its attributes using pickle
            with open(os.path.normpath(saveModelFolder + modelFileName + self.modelInfoString + self.classAttibuteString + ".pkl"), "wb") as f:
                pickle.dump(self.__dict__, f)            

    def _loadModel(self, modelFileName, appendedFolder = None, loadModelAttributes = True):
        if modelFileName is not None:
            # Store information to check model translation.
            currentFeatureNames = self.featureNames.copy()
            currentModelInd = self.modelInd
            
            # Load the saved model parameters.
            saveModelFolder = self.saveModelFolder + ("" if appendedFolder is None else appendedFolder)
            # Load in the model.
            loadModelPath = os.path.normpath(saveModelFolder + modelFileName + self.modelInfoString + ".pkl")
            self.model.load_state_dict(torch.load(loadModelPath))
            self.model.eval()  # Set the model to evaluation mode
            
            if loadModelAttributes:
                # Load the entire class along with its attributes using pickle
                with open(os.path.normpath(saveModelFolder + modelFileName + self.modelInfoString + self.classAttibuteString + ".pkl"), "rb") as f:
                    self.__dict__.update(pickle.load(f))     
        
            # Assert that the loaded model is what was expected.
            assert (currentFeatureNames == self.model.featureNames).all(), "You loaded a model with different features."
            assert currentModelInd == self.modelInd, "You loaded a model with a different index."

if __name__ == "__main__":  
    # Specify data params
    timePointRange = (2, 5)
    numLabelOptions = 2
    numDataPoints = 60
    numSignals = 4
    numClasses = 5

    alignedData = []
    # For each data point.
    for _ in range(numDataPoints):
        # Randomly select the number of time points
        numTimePoints = torch.randint(low=timePointRange[0], high=timePointRange[1], size=(1,)).item()
        
        # Generate random data for the selected number of time points
        data = np.random.uniform(5, 8, (numSignals, numTimePoints)).tolist()
        
        # Append the data to the list
        alignedData.append(data)
    # featureData = convertAlignedSignals(alignedData)
        
    # Initialize random labels
    allFeatureLabels = torch.randint(low=0, high=numClasses, size=(numDataPoints, numLabelOptions))
    
    