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
import signalEncoder   # Framwork for encoding/decoding of all signals.

# Import global model
sys.path.append(os.path.dirname(__file__) + "/../../../")
import _globalPytorchModel

# Model helper classes
sys.path.append(os.path.dirname(__file__) + "/../")
import _modelHelpers

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class signalEncoderModel(_globalPytorchModel.globalModel):
    def __init__(self, compressedLength, numEncodedSignals, numExpandedSignals):
        super(signalEncoderModel, self).__init__()
        # General model parameters.
        self.numExpandedSignals = numExpandedSignals
        self.numEncodedSignals = numEncodedSignals  # The final number of signals to accept, encoding all signal information.
        self.compressedLength = compressedLength   # The initial length of each incoming signal.

        # ------------------------- Signal Encoding ------------------------ #  
        
        # Method to remove unnecessary timepoints.
        self.encodeSignals = signalEncoder.signalEncoding(
                    numExpandedSignals = self.numExpandedSignals,
                    signalDimension = self.compressedLength, 
        )
        
        # ------------------------------------------------------------------ #
        
        # Initialize helper classes.
        self.modelHelpers = _modelHelpers._modelHelpers()
        
        # Reset the model
        self.resetModel()
    
    def resetModel(self):   
        # Signal encoder reconstructed loss holders.
        self.trainingLosses_compressedReconstruction = []   # List of compressed data reconstruction (signal encoder) training losses. Dim: numEpochs
        self.testingLosses_compressedReconstruction = []    # List of compressed data reconstruction (signal encoder) testing losses. Dim: numEpochs  
        
        # Signal encoder mean loss holders.
        self.trainingLosses_encodedMean = []         # List of encoded mean (signal encoder) training losses. Dim: numEpochs
        self.testingLosses_encodedMean = []          # List of encoded mean (signal encoder) testing losses. Dim: numEpochs
        # Signal encoder standard deviation loss holders.
        self.trainingLosses_encodedSTD = []          # List of encoded standard deviation (signal encoder) training losses. Dim: numEpochs
        self.testingLosses_encodedSTD = []           # List of encoded standard deviation (signal encoder) testing losses. Dim: numEpochs  
        
        # Signal encoder layer normalization loss holders.
        self.trainingLosses_signalEncodingLayerInfo = []     # List of iterative layer (signal encoder) training losses. Dim: numEpochs
        self.testingLosses_signalEncodingLayerInfo = []      # List of iterative layer (signal encoder) testing losses. Dim: numEpochs 

    def forward(self, compressedData, decodeSignals = False, calculateLoss = False, trainingFlag = False):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """
        # Initialize parameters for signal encoding.
        batchSize, numSignals, compressedLength = compressedData.size()
        signalEncodingLayerLoss = torch.zeros((batchSize,), device=compressedData.device)
                        
        # Assert the integrity of the incoming data.
        assert compressedLength == self.compressedLength, f"The signals have length {compressedLength}, but the model expected {self.compressedLength} points."
        
        # ---------------------- Training Augmentation --------------------- #  
        
        # Set the initial target signals.
        numEncodedSignals = self.numEncodedSignals
                
        if trainingFlag: 
            if self.numEncodedSignals == numSignals:
                numEncodedSignals = numEncodedSignals + torch.randint(-int(numEncodedSignals/4), int(numEncodedSignals/4), (1,)).item()
            elif  self.numEncodedSignals < numSignals:
                numEncodedSignals = numEncodedSignals + torch.randint(-int(numEncodedSignals/4), 0, (1,)).item()
            else:
                numEncodedSignals = numEncodedSignals + torch.randint(0, int(numEncodedSignals/4), (1,)).item()
            
            # Its not useful to train on nothing.
            if numEncodedSignals == numSignals: numEncodedSignals + 2
                        
        # ------------------- Optimal Signal Compression ------------------- #  
        
        with torch.no_grad():
            # Perform the optimal compression via PCA and embed channel information (for reconstruction).
            pcaProjection, principal_components = self.modelHelpers.svdCompression(compressedData.clone(), numEncodedSignals, standardizeSignals = True)
            # Loss for PCA reconstruction
            pcaReconstruction = torch.matmul(principal_components, pcaProjection)
            pcaReconstructionLoss = (compressedData - pcaReconstruction).pow(2).mean(dim=-1).mean(dim=1)
            print("\nFIRST Optimal Compression Loss:", pcaReconstructionLoss.mean().item())
            
            pcaReconstruction = (pcaReconstruction + compressedData.mean(dim=-1, keepdim=True)) * compressedData.std(dim=-1, keepdim=True)
            pcaReconstructionLoss = (compressedData - pcaReconstruction).pow(2).mean(dim=-1).mean(dim=1)
            print("FIRST Optimal Compression Loss STD:", pcaReconstructionLoss.mean().item())
            
        # ------------------------ Data Preparation ------------------------ #  
        
        # Calculate the number of signals changed.
        numFrozenSignals = self.encodeSignals.simulateNumFrozenSignals(numSignals, numEncodedSignals)
        numOriginalSignalsModified = numSignals - numFrozenSignals
        
        # Split the data into active and frozen components.
        activeCompressedData = compressedData[:, :numOriginalSignalsModified, :].contiguous()
        frozenCompressedData = compressedData[:, numOriginalSignalsModified:, :].contiguous()
        
        # ------------------- Learned Signal Compression ------------------- #  
            
        # Allow the model to adjust the incoming signals
        adjustedCompressedData = self.encodeSignals.signalEncodingInterface(activeCompressedData.clone(), self.encodeSignals.initialTransformation)
        # adjustedCompressedData dimension: batchSize, numSignals, compressedLength
                               
        # Compress the signal space into numEncodedSignals.
        encodedData, numSignalPath, batchsignalEncodingLayerLoss = self.encodeSignals(signalData = adjustedCompressedData.clone(), targetNumSignals = numEncodedSignals, calculateLoss = calculateLoss)
        # encodedData dimension: batchSize, numEncodedSignals, compressedLength

        # ---------------------- Signal Reconstruction --------------------- # 
                
        if decodeSignals:
            print("\nGoing back up:", numSignalPath) 
            # Clear the cache memory
            gc.collect(); torch.cuda.empty_cache();
            
            # If we are training, add noise to the final state to ensure continuity of the latent space.
            decodedData = encodedData.clone() + torch.rand_like(encodedData, device=encodedData.device).uniform_(-0.05, 0.05) if trainingFlag else encodedData.clone()
            
            pathSignalDecodingLoss = 0            
            # Follow the path back to the original signal.
            for pathInd in range(len(numSignalPath)-1, -1, -1):
                # Reconstruct to the current signal number in the path.
                decodedData, _, pathSignalDecodingLoss \
                        = self.encodeSignals(signalEncodingLayerLoss = pathSignalDecodingLoss,
                                            targetNumSignals = numSignalPath[pathInd], 
                                            calculateLoss = calculateLoss,
                                            signalData = decodedData)
            # reconstructedInitEncodingData dimension: batchSize, numSignals, compressedLength'
            
            # Undo what was done in the initial adjustment.
            reconstructedCompressedData = self.encodeSignals.signalEncodingInterface(decodedData.clone(), self.encodeSignals.finalTransformation)
            
            # Help smoothen out the noise added by the signal encoding/decoding.
            # reconstructedAlteredCompressedData = self.encodeSignals.signalEncodingInterface(decodedData[:, 0:numOriginalSignalsModified, :].contiguous(), self.encodeSignals.finalTransformation)
            # reconstructedCompressedData = torch.cat((reconstructedAlteredCompressedData, decodedData[:, numOriginalSignalsModified:, :]), dim=1).contiguous()
            # reconstructedCompressedData = decodedData.clone()
            
        # Clear the cache memory
        gc.collect(); torch.cuda.empty_cache();
            
        # ------------------------ Loss Calculations ----------------------- # 
                    
        if calculateLoss:    
            # Calculate the loss in reconstructing the initial encoded data.
            innerReconstructionLoss = (adjustedCompressedData - decodedData).pow(2).mean(dim=-1).mean(dim=1)
            
            # Try and reconstruct the compressed data from the shallow (outer) CNN network.
            if trainingFlag: adjustedCompressedData = adjustedCompressedData + torch.rand_like(adjustedCompressedData, device=adjustedCompressedData.device).uniform_(-0.05, 0.05)
            shallowReconstruction = self.encodeSignals.signalEncodingInterface(adjustedCompressedData.clone(), self.encodeSignals.finalTransformation)
            # shallowReconstruction dimension: batchSize, numSignals, compressedLength
            
            # Calculate the loss in reconstructing the outer data.
            outerReconstructionLoss = (activeCompressedData - shallowReconstruction).pow(2).mean(dim=-1).mean(dim=1)
            
            # Calculate the loss in reconstructing all the data.
            finalReconstructionLoss = (reconstructedCompressedData - activeCompressedData).pow(2).mean(dim=-1).mean(dim=1)
            
            # Print the loss information.
            print("Component Loss:", batchsignalEncodingLayerLoss.detach().mean().item(), pathSignalDecodingLoss.mean().item())
            print("Decoding Loss:", innerReconstructionLoss.detach().mean().item(), outerReconstructionLoss.mean().item(), finalReconstructionLoss.mean().item())
            
            # # Add up all the losses together.
            signalEncodingLayerLoss = signalEncodingLayerLoss + (batchsignalEncodingLayerLoss + pathSignalDecodingLoss)/(len(numSignalPath) + 1)
            signalEncodingLayerLoss = signalEncodingLayerLoss + innerReconstructionLoss + outerReconstructionLoss
        
        # Recombine the active and frozen data.
        reconstructedCompressedData = torch.cat((reconstructedCompressedData, frozenCompressedData), dim=1).contiguous()
    
        return encodedData, reconstructedCompressedData, signalEncodingLayerLoss
            
        # ------------------------------------------------------------------ #  
    
    # DEPRECATED
    def shapInterface(self, signalData):
        # Ensure proper data format.
        batchSize, compressedLength = signalData.shape
        signalDataTensor = torch.tensor(signalData.tolist())
                
        # Reshape the inputs to integrate into the model's expected format.
        signalDataTensor = signalDataTensor.unsqueeze(1)
        
        # predict the activities.
        compressedData, reconstructedCompressedData = self.forward(signalDataTensor, reconstructData = True)

        return reconstructedCompressedData.detach().cpu().numpy()
    
    
    