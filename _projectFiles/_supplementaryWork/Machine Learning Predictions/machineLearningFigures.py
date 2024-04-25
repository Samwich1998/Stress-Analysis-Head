# General
import os
import sys
import accelerate

# Set specific environmental parameters.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING and ERROR, 3 = ERROR only)

# Add the directory of the current file to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import plotting methods
from helperMethods.signalEncoderPlots import signalEncoderPlots


if __name__ == "__main__":
    # Specify the general model information.
    sharedModelWeights = ["signalEncoderModel", "autoencoderModel", "sharedEmotionModel"]  # Possible models: ["trainingInformation", "signalEncoderModel", "autoencoderModel", "signalMappingModel", "specificEmotionModel", "sharedEmotionModel"]
    datasetNames = ["wesad", "emognition", "amigos", "dapper", "case", 'empatch']  # Specify which metadata analyses to compile
    modelName = "emotionModel"  # The emotion model's unique identifier. Options: emotionModel

    # Testing parameters.
    numLiftedChannelBounds = (16, 64, 16)  # Boundary inclusive
    numExpandedSignalBounds = (2, 5)    # Boundary inclusive
    numEncodingLayerBounds = (0, 6)     # Boundary inclusive

    # --------------------------- Figure Plotting -------------------------- #

    # Initialize plotting classes.
    saveFolder = os.path.dirname(os.path.abspath(__file__)) + "/finalFigures/"
    signalEncoderPlots = signalEncoderPlots(modelName, datasetNames, sharedModelWeights, savingBaseFolder=saveFolder, accelerator=accelerate.Accelerator(cpu=True))

    # Define extra parameters for the plotting protocols.
    finalSignalEncoderTrainingDataString = "2024-04-24 final signalEncoder on HPC-GPU at numLiftedChannels XX at numExpandedSignals YY at numEncodingLayers ZZ"

    # Heatmap num Expanded Signals by numEncoding Layers
    signalEncoderPlots.signalEncoderParamHeatmap(numExpandedSignalBounds, numEncodingLayerBounds, numLiftedChannelBounds, finalSignalEncoderTrainingDataString)



    # signalEncoderPlots.signalEncoderLossComparison(numExpandedSignalBounds, numEncodingLayerBounds, numLiftedChannelBounds, finalSignalEncoderTrainingDataString, plotTitle="Signal Encoder Loss Comparison")

    # Plot the loss comparisons over time.
    # plottingProtocols.timeLossComparison(allModelPipelines, metaLearnedInfo, userInputParams, plotTitle="Auto Encoder Loss Plots")
    # plottingProtocols.reconstructionLossComparison(allModelPipelines, metaLearnedInfo, userInputParams, plotTitle="Signal Encoder Reconstruction Loss Plots")
    # plottingProtocols.meanLossComparison(allModelPipelines, metaLearnedInfo, userInputParams, plotTitle="Auto Encoder Mean Loss Plots")
    # plottingProtocols.stdLossComparison(allModelPipelines, metaLearnedInfo, userInputParams, plotTitle="Auto Encoder Standard Deviation Loss Plots")

    # Plot the loss comparisons
    # plottingProtocols.autoencoderLossComparison(allMetaModelPipelines, metaLearnedInfo, modelComparisonInfo, comparingModelInd=0, plotTitle="Autoencoder Loss Plots")
