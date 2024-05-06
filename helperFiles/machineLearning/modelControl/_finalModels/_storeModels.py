import os
import shutil

# Define the root directory where your trainingData and model folders are located
root_dir = "./emotionModel/metaTrainingModels/signalEncoder/"  # Adjust to your actual root folder path
save_dir = "./storedModels/emotionModel/metaTrainingModels/signalEncoder/"  # Adjust to your actual save folder path

# Create the save directory if it doesn't already exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through the main categories of interest: trainingData and model folders
for trainingFolder in os.listdir(root_dir):
    pathToTrainingFolder = os.path.join(root_dir, trainingFolder)
    if trainingFolder.startswith((".", "_")):  # Skip hidden or unwanted folders
        continue

    for modelFolder in os.listdir(pathToTrainingFolder):
        pathToModelFolder = os.path.join(pathToTrainingFolder, modelFolder)
        if modelFolder.startswith("."):  # Skip hidden folders
            continue

        if "2024-05-04" not in pathToModelFolder: continue

        # Initialize variables to find the last epoch
        max_epoch = -1
        pathToMaxEpochFolder = None

        # Evaluate all epochs to find the maximum
        for epochFolder in os.listdir(pathToModelFolder):
            pathToEpochFolder = os.path.join(pathToModelFolder, epochFolder)
            if epochFolder.startswith("."):  # Skip hidden folders and ensure folder name is numeric
                continue
            epoch = int(epochFolder.split(" ")[-1])  # Parse the epoch number from the folder name

            if epoch > max_epoch:
                max_epoch = epoch
                pathToMaxEpochFolder = pathToEpochFolder

        # If a max epoch was found, copy its directory
        if pathToMaxEpochFolder:
            destinationFolder = os.path.join(save_dir, f"{trainingFolder}/{modelFolder}/Epoch {max_epoch}/")
            if os.path.exists(destinationFolder):
                shutil.rmtree(destinationFolder)  # Remove the folder if it exists
            shutil.copytree(pathToMaxEpochFolder, destinationFolder)
