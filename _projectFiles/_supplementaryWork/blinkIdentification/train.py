import numpy as np
import pandas as pd
import sys

# ---------- Model ---------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import torch.nn.functional as F
import dataLoader.dataLoader as dataLoader

class trainModel:
    def __init__(self, model, model_type, train_loader, test_loader, class_weights_tensor,
                 num_epochs=30,
                 learning_rate=0.001,
                 dropout=0.2,
                 debug=False):
        self.model = model
        self.model_type = model_type
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.class_weights_tensor = class_weights_tensor
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.debug = debug

        self.train_losses = []; self.train_accuracies = []; self.train_f1_scores = []
        self.test_accuracies = []; self.test_f1_scores = []

    def train(self):
        min_test_accuracy = float('inf')
        criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        optimizer = optim.AdamW(self.model.parameters(), lr=1E-4, weight_decay=1E-4)

        # ---------- Training Loop ---------

        for epoch in range(self.num_epochs):
            # Lists to store labels and predictions for each batch, for F1 and accuracy calculations
            all_preds = []
            all_labels = []

            self.model.train()  # Ensure the model is in training mode
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], batches: {len(self.train_loader)}')
            for inputs, labels in self.train_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Convert outputs to predictions
                _, predicted = torch.max(outputs.channelData, 1)
                all_preds.extend(predicted.tolist())
                all_labels.extend(labels.tolist())

            # Calculate training metrics
            train_accuracy = balanced_accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds)
            self.train_losses.append(loss.item())
            self.train_accuracies.append(train_accuracy)
            self.train_f1_scores.append(train_f1)

            # get test metrics
            all_val_labels, all_val_preds, test_accuracy, test_f1 = self.test()

            self.test_accuracies.append(test_accuracy)
            self.test_f1_scores.append(test_f1)

            if self.test_accuracies[-1] < min(self.test_accuracies):
                torch.save(self.model.state_dict(), f'models/best_model_{self.model_type}_{self.dropout}.pth')
            if self.debug:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}, Train Acc: {train_accuracy * 100:.2f}%, Train F1: {train_f1:.4f}, Test Acc: {test_accuracy * 100:.2f}%, Test F1: {test_f1:.4f}')

    def get_predicted_labels(self):
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.channelData, 1)
                all_val_preds.extend(predicted.tolist())
                all_val_labels.extend(labels.tolist())
        return all_val_labels, all_val_preds
    def test(self):
        # Validation metrics
        self.model.eval()  # Set model to evaluation mode
        all_val_labels, all_val_preds = self.get_predicted_labels()

        test_accuracy = balanced_accuracy_score(all_val_labels, all_val_preds)
        test_f1 = f1_score(all_val_labels, all_val_preds)
        self.confusion_matrix(all_val_labels, all_val_preds)

        return all_val_labels, all_val_preds, test_accuracy, test_f1

    def confusion_matrix(self, actual_labels, predicted_labels):
        # Confusion Matrix calculation
        TP = sum((np.array(actual_labels) == 0) & (np.array(predicted_labels) == 0))
        TN = sum((np.array(actual_labels) == 1) & (np.array(predicted_labels) == 1))
        FP = sum((np.array(actual_labels) == 1) & (np.array(predicted_labels) == 0))
        FN = sum((np.array(actual_labels) == 0) & (np.array(predicted_labels) == 1))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
        if self.debug:
            print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, TPR: {TPR:.4f}, FPR: {FPR:.4f}')
        return TP, TN, FP, FN, TPR, FPR

    def loadModel(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        all_val_labels, all_val_preds, test_accuracy, test_f1 = self.test()
        print(f'Test Acc: {test_accuracy * 100:.2f}%, Test F1: {test_f1:.4f}')
        print(self.confusion_matrix(all_val_labels, all_val_preds))
