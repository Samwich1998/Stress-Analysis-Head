# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import gc
import sys

# PyTorch
import torch

# Import helper files.
sys.path.append(os.path.dirname(__file__) + "/modelComponents/")
import generalAutoencoder   # Framwork for compression/decompression of time-series signals.

# Import global model
sys.path.append(os.path.dirname(__file__) + "/../../../")
import _globalPytorchModel

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class autoencoderModel(_globalPytorchModel.globalModel):
    def __init__(self, compressedLength):
        super(autoencoderModel, self).__init__()
        # General model parameters.
        self.compressedLength = compressedLength  # The final length of the compressed signal after the autoencoder. MUST BE CHANGED IN AUTOENCODER.py
        
        # ------------------- General Autoencoder Module ------------------- #  
        
        # Method to reconstruct the original signal.
        self.generalAutoencoder = generalAutoencoder.generalAutoencoder(compressedLength)
        
        # ------------------------------------------------------------------ #
        
        # Reset the model
        self.resetModel()
    
    def resetModel(self):        
        # Autoencoder signal reconstructed loss holders.
        self.trainingLosses_signalReconstruction = []    # List of signal reconstruction (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_signalReconstruction = []     # List of signal reconstruction (autoencoder) testing losses. Dim: numEpochs
        
        # Autoencoder compressed mean loss holders.
        self.trainingLosses_compressedMean = []    # List of compressed mean (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_compressedMean = []     # List of compressed mean (autoencoder) testing losses. Dim: numEpochs
        # Autoencoder compressed standard deviation loss holders.
        self.trainingLosses_compressedSTD = []    # List of compressed standard deviation (autoencoder) training losses. Dim: numEpochs
        self.testingLosses_compressedSTD = []     # List of compressed standard deviation (autoencoder) testing losses. Dim: numEpochs      

    def forward(self, signalData, reconstructSignals = False, calculateLoss = False, trainingFlag = False):
        """ The shape of inputData: (batchSize, numSignals, sequenceLength) """
        
        # ----------------------- Data Preprocessing ----------------------- #  

        # Prepare the data for compression/expansion
        signalData = signalData.to(torch.float32) # Floats are required for gradient tracking.
        autoencoderLayerLoss = torch.zeros((signalData.size(0)), device=signalData.device)
        # signalData dimension: batchSize, numSignals, sequenceLength
        # autoencoderLayerLoss dimension: batchSize
            
        # --------------------- Signal Compression --------------------- # 
            
        # Data reduction: remove unnecessary timepoints from the signals.
        initialCompressedData, numSignalPath, encoderLoss = self.generalAutoencoder(signalData, targetSequenceLength = self.compressedLength, autoencoderLayerLoss = 0, calculateLoss = calculateLoss)
        # compressedData dimension: batchSize, numSignals, compressedLength
        
        compressedData = self.generalAutoencoder.encodingInterface_reshapeMethod(initialCompressedData.clone(), self.generalAutoencoder.initialTransformation)
        
        # -------------------- Signal Reconstruction ------------------- #  
            
        # If we are reconstructing the data.
        if reconstructSignals:
            print("\nGoing back up:", numSignalPath) 
            # Clear the cache memory
            gc.collect(); torch.cuda.empty_cache();
            
            # If we are training, add noise to the final state to ensure continuity of the latent space.
            noisyCompressedData = compressedData.clone() + torch.randn_like(compressedData, device=compressedData.device) * 0.05 if trainingFlag else compressedData.clone()
            
            finalCompressedData = self.generalAutoencoder.encodingInterface_reshapeMethod(noisyCompressedData.clone(), self.generalAutoencoder.finalTransformation)

            decoderLoss = 0  
            reconstructedData = finalCompressedData.clone()
            # Follow the path back to the original signal.
            for pathInd in range(len(numSignalPath)-1, -1, -1):
                # Reconstruct to the current signal number in the path.
                reconstructedData, _, decoderLoss = self.generalAutoencoder(targetSequenceLength = numSignalPath[pathInd], 
                                                                               autoencoderLayerLoss = decoderLoss,
                                                                               signalData = reconstructedData,
                                                                               calculateLoss = calculateLoss)
            # reconstructedData dimension: batchSize, numSignals, sequenceLength
            
            if calculateLoss:
                # Calculate the loss in reconstructing the encoded data.
                innerReconstructionLoss = (initialCompressedData - finalCompressedData).pow(2).mean(dim=-1).mean(dim=1)

                autoencoderLayerLoss = decoderLoss + encoderLoss #+ innerReconstructionLoss
                print("Component Loss:", encoderLoss.detach().mean().item(), decoderLoss.mean().item())
                print("Reconstruction Loss:", innerReconstructionLoss.detach().mean().item())
            
        # Clear the cache memory
        gc.collect(); torch.cuda.empty_cache();

        return compressedData, reconstructedData, autoencoderLayerLoss
            
        # ------------------------------------------------------------------ #  
    
    def shapInterface(self, signalData):
        # Ensure proper data format.
        batchSize, sequenceLength = signalData.shape
        signalDataTensor = torch.tensor(signalData.tolist())
                
        # Reshape the inputs to integrate into the model's expected format.
        signalDataTensor = signalDataTensor.unsqueeze(1)
        
        # predict the activities.
        compressedData, reconstructedData = self.forward(signalDataTensor, reconstructSignals = True)

        return reconstructedData.detach().cpu().numpy()
    
    
    