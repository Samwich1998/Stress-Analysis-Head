"""




pip3 install torch torchvision torchaudio

"""


# Import Basic Modules
import os
import joblib
import numpy as np

# Import Modules for Plotting
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

# Import Machine Learning Modules
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor, Lambda, Compose
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



# PyTorch models inherit from torch.nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self, featureDimension, numClasses):
        super(NeuralNetwork, self).__init__()
        
        # Initialize Model Parameters
        self.featureDimension = featureDimension
        self.numClasses = numClasses
        
    def activationLayers(self, inputData):
        # List of Possible Activation Layers
            # nn.Linear(in_features, out_features, bias=False, device=self.device)
            # nn.Relu(inplace=False)

        return nn.Sequential(
            nn.Linear(in_features=self.featureDimension, out_features=self.featureDimension, bias=False, device=self.device),
            nn.Linear(in_features=self.featureDimension, out_features=self.numClasses, bias=False, device=self.device),
        )(inputData)
    
    def forward(self, inputData):
        return self.activationLayers(inputData)
    
    def lossFunction(self):
        # List of Possible Loss Functions
            # torch.nn.CrossEntropyLoss()
        return torch.nn.CrossEntropyLoss()
    
    def optimizerFunction(self, model):
        # Optimizers specified in the torch.optim package
        # List of Possible Optimizer Functions
            # torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        return torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train_one_epoch(model, training_loader, lossFunction, optimizerFunction, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
    
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
    
            # Zero your gradients for every batch!
            optimizerFunction.zero_grad()
    
            # Make predictions for this batch
            outputs = model(inputs)
    
            # Compute the loss and its gradients
            loss = lossFunction(outputs, labels)
            loss.backward()
    
            # Adjust learning weights
            optimizerFunction.step()
    
            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
    
        return last_loss

    def runTraining(self, model, signalData, signalLabels, EPOCHS):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0
        
        # Create data loaders for our datasets; shuffle for training, not for validation
        Training_Data, Validation_Data, Training_Labels, Validation_Labels = train_test_split(signalData, signalLabels, test_size=0.1, shuffle= True, stratify=signalLabels)
        training_loader = torch.utils.data.DataLoader(Training_Data, batch_size=4, shuffle=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(Validation_Data, batch_size=4, shuffle=False, num_workers=2)
        
        # Calculate the Loss
        lossFunction = self.lossFunction()
        #loss = lossFunction(Training_Data, Training_Labels)
        # Create Optimizer
        optimizerFunction = self.optimizerFunction(model)
        
        
        best_vloss = 1_000_000.
        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))
        
            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = self.train_one_epoch(model, training_loader, lossFunction, optimizerFunction, epoch_number, writer)
        
            # We don't need gradients on to do reporting
            model.train(False)
        
            running_vloss = 0.0
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = lossFunction(voutputs, vlabels)
                running_vloss += vloss
        
            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        
            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()
        
            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)
        
            epoch_number += 1


class ANN():
    def __init__(self, modelPath, featureDimension, numClasses):
        super(ANN, self).__init__()
        
        # Get cpu or gpu device for training.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Model Parameters
        self.featureDimension = featureDimension
        self.numClasses = numClasses
        self.model = None
        
        # Initialize Model
        if os.path.exists(modelPath):
            # If Model Exists, Load it
            self.loadModel(modelPath)
        else:
            # Else, Create the Model
            self.createModel()
        
    def createModel(self):
        self.model = NeuralNetwork(self.featureDimension, self.numClasses).to(self.device)
        print("ANN Model Created")
        
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels):
        # Train the Model
        self.model.runTraining(self.model, Training_Data, Training_Labels, EPOCHS = 500)
        
        # Score the Model
        modelScore = self.scoreModel(Testing_Data, Testing_Labels)
        return modelScore
        
    def loadModel(self, modelPath):
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load(modelPath))
        print("ANN Model Loaded")
        
    def saveModel(self, modelPath = "./KNN.pkl"):
        with open(modelPath, 'wb') as handle:
            joblib.dump(self.model, handle)
    
    def scoreModel(self, signalData, signalLabels):
        return self.model.score(signalData, signalLabels)        
        
    def predictData(self, New_Data):
        logits = self.model(New_Data)
        predictProb = nn.Softmax(dim=1)(logits) # 'dim' is the axis which the values must sum to 1
        classPrediction = predictProb.argmax(1)
        return classPrediction



        

            