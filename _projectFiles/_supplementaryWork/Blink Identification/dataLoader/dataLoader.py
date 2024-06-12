import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

def get_dataLoader(dataset='finalFullBlinksData', batch_size=64, weighted=True):
    df_dataset = pd.read_csv(dataset + '.csv')  # all features data
    df_dataset = df_dataset.fillna(0)
    df_dataset = df_dataset.loc[:, ~df_dataset.columns.str.contains('^Unnamed')]

    df_dataset.loc[df_dataset['Label'] == 2, 'Label'] = 1

    df_train = df_dataset[df_dataset['IsTrial'] == 0]
    df_test = df_dataset[df_dataset['IsTrial'] == 1]

    trainX_tensor, trainY_tensor, testX_tensor, testY_tensor = get_dataTensors(df_train, df_test, dataset)

    sampler = None
    # Create datasets + dataloaders
    class_counts = df_train['Label'].value_counts().to_dict()
    total_samples = sum(class_counts.values())
    class_weights = {class_label: total_samples/class_count for class_label, class_count in class_counts.items()}
    class_weights_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32)
    if weighted:
        weights = [class_weights[label] for label in df_train['Label']]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    # Create datasets + dataloaders with sampler
    train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
    test_dataset = TensorDataset(testX_tensor, testY_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, class_weights_tensor
def get_dataTensors(df_train, df_test, dataset):
    if Path(f'{dataset}/trainX_tensor.pt').exists() and Path(f'{dataset}/trainY_tensor.pt').exists() and Path(
            f'{dataset}/testX_tensor.pt').exists() and Path(f'{dataset}/testY_tensor.pt').exists():
        trainX_tensor = torch.load(f'{dataset}/trainX_tensor.pt')
        trainY_tensor = torch.load(f'{dataset}/trainY_tensor.pt')
        testX_tensor = torch.load(f'{dataset}/testX_tensor.pt')
        testY_tensor = torch.load(f'{dataset}/testY_tensor.pt')
    else:
        # convert each row to a 2D image
        feature_columns = df_train.columns.difference(['Label', 'IsTrial'])

        # Determine image dimensions (example: 28x28)
        img_height, img_width = 1, len(feature_columns)
        num_features = len(feature_columns)  # Each feature corresponds to one column

        # Ensure the number of features matches the width of the image
        if len(feature_columns) != num_features:
            raise ValueError(f"Number of features ({len(feature_columns)}) does not match the required width ({img_width}) for the image.")

        train_images = np.array([row for _, row in df_train[feature_columns].iterrows()])
        test_images = np.array([row for _, row in df_test[feature_columns].iterrows()])

        # Convert numpy images to torch tensors
        trainX_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        trainY_tensor = torch.tensor(df_train['Label'].values, dtype=torch.long)
        testX_tensor = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        testY_tensor = torch.tensor(df_test['Label'].values, dtype=torch.long)

        # save tensors to file
        torch.save(trainX_tensor, f'{dataset}/trainX_tensor.pt')
        torch.save(trainY_tensor, f'{dataset}/trainY_tensor.pt')
        torch.save(testX_tensor, f'{dataset}/testX_tensor.pt')
        torch.save(testY_tensor, f'{dataset}/testY_tensor.pt')

    return trainX_tensor, trainY_tensor, testX_tensor, testY_tensor


def plot_images(images, labels, n=10):
    plt.figure(figsize=(20, 20))  # Increase the figure size for better visibility
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        ax.imshow(np.expand_dims(images[i], axis=1), cmap='gray', aspect='auto')
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')
    plt.show()