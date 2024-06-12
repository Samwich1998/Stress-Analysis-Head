# take blink data raw, turn into image/cnn, train on that
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
import convolutionalHelpers # todo import convolutionalHelpers

class blinkIdentificationModel(convolutionalHelpers):
    def __init__(self):
        super(blinkIdentificationModel, self).__init__()
        # Initialize the CNN model.
        self.convolutionalModel = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
            nn.Conv1d(4, 4, kernel_size=3, padding=1),
            nn.Linear(4 * 147, 256),
            nn.Linear(256, 2) # Assuming binary classification
        )

    def forward(self, x):
        x = self.convolutionalModel(x)
        return x
