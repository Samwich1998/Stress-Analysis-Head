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
import dataLoader.dataLoader as dataLoader
from train import trainModel
import visualizations
from blinkIdentification_models import blinkIdentificationModel

if __name__ == "__main__":
    trainLoader, testLoader, class_weights_tensor = dataLoader.get_dataLoader(dataset='finalFullBlinksData')
    debug = True
    plotting = True
    for dropout in [0.1, 0.2, 0.3, 0.4, 0.6]:
        model = blinkIdentificationModel()
        model_type = 'CNN'
        myTraining = trainModel(model, model_type, trainLoader, testLoader, class_weights_tensor, dropout=dropout, debug=debug)
        myTraining.train()
        if plotting:
            myPlotting = visualizations.plotting(myTraining)
            myPlotting.model_performance()
            myPlotting.classification_threshold()