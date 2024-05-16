
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from natsort import natsorted

# -------------------------------------------------------------------------- #
# ----------------------- Quantify Image Similarities ---------------------- #

class imageSimilarities:
    
    def __init__(self):
        # Feature extraction parameters
        self.imageSize = (1024, 1024)
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')        
    
    def getImagePaths(self, imageFolder):
        imagePaths = []
        # For each file in the folder
        for filename in natsorted(os.listdir(imageFolder)):
            # If the file is an image
            if filename.endswith((".jpeg", ".jpg")):
                # Save the file location
                imagePaths.append(imageFolder + filename)
        
        return imagePaths
    
    def get_image_features(self, image_paths):
        features = []
        for path in image_paths:
            img = Image.open(path)
            img = img.resize(self.imageSize)
            x = np.array(img)  
            print(x[:,:,0])

            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            
            print(x.shape)
            sys.exit()
            feature = self.model.predict(x)
            features.append(feature)
        features = np.concatenate(features, axis=0)
        return features
    
    def get_similarity_matrix(self, image_paths):
        features = self.get_image_features(image_paths)
        similarities = np.zeros((len(image_paths), len(image_paths)))
        for i in range(len(image_paths)):
            for j in range(i, len(image_paths)):
                similarities[i, j] = np.dot(features[i], features[j]) / (np.linalg.norm(features[i]) * np.linalg.norm(features[j]))
                similarities[j, i] = similarities[i, j]
        return similarities #/ np.sum(similarities, axis=0)


    # ---------------------------------------------------------------------- #

if __name__ == "__main__":
    # Instantiate class.
    imageComparison = imageSimilarities()
    # Collect the image files.
    imageFolder = os.path.dirname(__file__) + "/Virtual Images/"
    imagePaths = imageComparison.getImagePaths(imageFolder)
    # Calculate similarity matrix
    similarity_matrix = imageComparison.get_similarity_matrix(imagePaths)
    
    
    # Plot the similarity matrix as a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(similarity_matrix, cmap='hot')
    
    filenames = [os.path.basename(path).split(".")[0] for path in imagePaths]
    # Add labels to the x and y axis
    ax.set_xticks(np.arange(len(imagePaths)))
    ax.set_yticks(np.arange(len(imagePaths)))
    ax.set_xticklabels(filenames)
    ax.set_yticklabels(filenames)
    
    # Rotate the x-axis labels to make them easier to read
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set the title of the plot
    ax.set_title("Image similarity matrix")
    plt.tight_layout()
    
    # Show the plot
    fig.savefig(imageFolder + "imageSimilarity.png", dpi=300)
    plt.show()
    

