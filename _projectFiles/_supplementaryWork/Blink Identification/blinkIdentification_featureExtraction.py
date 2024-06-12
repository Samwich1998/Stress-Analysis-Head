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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = torch.softmax(out, dim=1)
        return out

# ---------- Datasets + Inits ---------

df_all_data = pd.read_csv("alldata.csv") # all features data
df_good_data = pd.read_csv("dataLoader/goodfeaturesdata.csv") # non-amplitude dependent features data
df_dataset = df_good_data
df_dataset.loc[df_dataset['Label'] == 2, 'Label'] = 1

df_train = df_dataset[df_dataset['IsTrial'] == 0]
df_test = df_dataset[df_dataset['IsTrial'] == 1]

# Convert numpy arrays to torch tensors
trainX_tensor = torch.tensor(df_train.drop(columns=['IsTrial', 'Label']).values, dtype=torch.float32)
trainY_tensor = torch.tensor(df_train['Label'].values, dtype=torch.long)
testX_tensor = torch.tensor(df_test.drop(columns=['IsTrial', 'Label']).values, dtype=torch.float32)
testY_tensor = torch.tensor(df_test['Label'].values, dtype=torch.long)

# Create datasets + dataloaders
train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
test_dataset = TensorDataset(testX_tensor, testY_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

def train_eval_model(hidden_size, dropout, debug=False, min_test_accuracy=float('inf')):
    model = MLP(input_size, hidden_size, num_classes, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1E-4, weight_decay=1E-4)

    # Initialize lists to track metrics
    train_losses = []; train_accuracies = []; train_f1_scores = []
    test_accuracies = []; test_f1_scores = []

    # ---------- Training Loop ---------

    num_epochs = 30
    for epoch in range(num_epochs):
        # Lists to store labels and predictions for each batch, for F1 and accuracy calculations
        all_preds = []
        all_labels = []

        model.train()  # Ensure the model is in training mode
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Convert outputs to predictions
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

        # Calculate training metrics
        train_accuracy = balanced_accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        train_losses.append(loss.item())
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)

        # Validation metrics
        model.eval()  # Set model to evaluation mode
        all_val_preds = []
        all_val_labels = []
        all_outputs = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                all_outputs.extend(outputs.tolist())
                _, predicted = torch.max(outputs.data, 1)
                all_val_preds.extend(predicted.tolist())
                all_val_labels.extend(labels.tolist())

        test_accuracy = balanced_accuracy_score(all_val_labels, all_val_preds)
        test_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        test_accuracies.append(test_accuracy)
        test_f1_scores.append(test_f1)

        if test_accuracy < min_test_accuracy:
            min_test_accuracy = test_accuracy
            torch.save(model.state_dict(), f'models/best_model_feature_extraction_{hidden_size}_{dropout}.pth')

        if debug:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Acc: {train_accuracy * 100:.2f}%, Train F1: {train_f1:.4f}, Test Acc: {test_accuracy * 100:.2f}%, Test F1: {test_f1:.4f}')

    plt.figure(figsize=(12, 5))
    plt.suptitle(f'Model Performance\nHidden Size: {hidden_size}, Dropout: {dropout}')
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_f1_scores, label='Train F1 Score')
    plt.plot(range(1, num_epochs+1), test_f1_scores, label='Test F1 Score')
    plt.title('F1 Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.show()

    # AUC using outputs for each class

    # Assuming outputs are from your model and testY_tensor is available
    probabilities = np.array(all_outputs)

    thresholds = np.linspace(0, 1, num=25)
    curr_method_probs = probabilities[:, 0]

    TP_rates = []; FP_rates = []; TN_rates = []; FN_rates = []

    for threshold in thresholds:
        predicted = [0 if p >= threshold else 1 for p in curr_method_probs]

        TP = 0; FP = 0; TN = 0; FN = 0
        for i in range(len(predicted)):
            if predicted[i] == 0:
                if all_val_labels[i] == 0:
                    TP += 1
                else:
                    FP += 1
            else:
                if all_val_labels[i] == 1:
                    TN += 1
                else:
                    FN += 1

        #
        TP_rates.append(TP / (len(predicted) - np.count_nonzero(all_val_labels)))
        FP_rates.append(FP / np.count_nonzero(all_val_labels))
        TN_rates.append(TN / np.count_nonzero(all_val_labels))
        FN_rates.append(FN / (len(predicted) - np.count_nonzero(all_val_labels)))

    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, TP_rates, label='True Positives', color='green')
    plt.plot(thresholds, FP_rates, label='False Positives', color='red')
    plt.plot(thresholds, TN_rates, label='True Negatives', color='blue')
    plt.plot(thresholds, FN_rates, label='False Negatives', color='orange')
    plt.title(f'Classification Metrics as Function of Thresholding\nHidden Size: {hidden_size}, Dropout: {dropout}')
    plt.xlabel('Threshold Value')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    load_prev_model = f'models/best_model_feature_extraction_{hidden_size}_{dropout}.pth'
    if load_prev_model:
        model = MLP(input_size, hidden_size, num_classes, dropout=dropout)
        model.load_state_dict(torch.load(load_prev_model))
        model.eval()
        all_val_preds = []
        all_val_labels = []
        all_outputs = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                all_outputs.extend(outputs.tolist())
                _, predicted = torch.max(outputs.data, 1)
                all_val_preds.extend(predicted.tolist())
                all_val_labels.extend(labels.tolist())

        test_accuracy = balanced_accuracy_score(all_val_labels, all_val_preds)
        test_f1 = f1_score(all_val_labels, all_val_preds)
        print(f'Balanced Test Acc: {test_accuracy * 100:.2f}%, Test F1: {test_f1:.4f}')

        # Confusion Matrix calculation
        TP = sum((np.array(all_val_labels) == 0) & (np.array(all_val_preds) == 0))
        TN = sum((np.array(all_val_labels) == 1) & (np.array(all_val_preds) == 1))
        FP = sum((np.array(all_val_labels) == 1) & (np.array(all_val_preds) == 0))
        FN = sum((np.array(all_val_labels) == 0) & (np.array(all_val_preds) == 1))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate

        print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, TPR: {TPR:.4f}, FPR: {FPR:.4f}')

    return stats.trim_mean(test_f1_scores, 0.3), stats.trim_mean(test_accuracies, 0.3), min_test_accuracy

# Model parameters
input_size = trainX_tensor.shape[1]
num_classes = 2  # For binary classification
# ------------------------ Grid Search ------------------------
all_f1 = []
all_acc = []
all_min_test_accuracy = float('inf')
for hidden_size in [32, 64, 128, 256]:
    curr_size_f1 = []
    curr_size_acc = []
    for dropout in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        curr_f1, curr_acc, min_test_accuracy = train_eval_model (hidden_size, dropout)
        all_min_test_accuracy = min(all_min_test_accuracy, min_test_accuracy)
        curr_size_f1.append(curr_f1)
        curr_size_acc.append(curr_acc)
    all_f1.append(curr_size_f1)
    all_acc.append(curr_size_acc)

# Plotting as heatmap
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(all_f1, cmap='GnBu', interpolation='nearest')
plt.title('F1 Score')
plt.xlabel('Dropout')
plt.ylabel('Hidden Size')
plt.xticks(range(6), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
plt.yticks(range(4), [32, 64, 128, 256])
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(all_acc, cmap='GnBu', interpolation='nearest')
plt.title('Accuracy')
plt.xlabel('Dropout')
plt.ylabel('Hidden Size')
plt.xticks(range(6), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
plt.yticks(range(4), [32, 64, 128, 256])
plt.colorbar()
plt.show()