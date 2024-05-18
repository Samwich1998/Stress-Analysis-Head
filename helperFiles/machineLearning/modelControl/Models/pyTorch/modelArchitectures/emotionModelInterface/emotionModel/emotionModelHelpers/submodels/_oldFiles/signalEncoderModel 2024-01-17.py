# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import gc
import sys
import math

# PyTorch
import torch

# Import helper files.
sys.path.append(os.path.dirname(__file__) + "/modelComponents/")
import signalEncoder   # Framwork for encoding/decoding of all signals.

# Import global model
sys.path.append(os.path.dirname(__file__) + "/../../../")
import _globalPytorchModel

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class signalEncoderModel(_globalPytorchModel.globalModel):
    def __init__(self, compressedLength, numEncodedSignals):
        super(signalEncoderModel, self).__init__()
        # General model parameters.
        self.numEncodedSignals = numEncodedSignals  # The final number of signals to accept, encoding all signal information.
        self.compressedLength = compressedLength   # The initial length of each incoming signal.

        # ------------------------- Signal Encoding ------------------------ #  
        
        # Method to remove unnecessary timepoints.
        self.encodeSignals = signalEncoder.signalEncoding(
                    signalDimension = self.compressedLength, 
        )
        
        # ------------------------------------------------------------------ #
        
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

    def forward(self, compressedData, decodeSignals = False, calculateLoss = False, trainingFlag = False, maxBatchSignals = 10000):
        """ The shape of inputData: (batchSize, numSignals, compressedLength) """
        
        # ----------------------- Data Preprocessing ----------------------- #  

        # Extract the incoming data's dimension.
        batchSize, numSignals, compressedLength = compressedData.size()
        # signalData dimension: batchSize, numSignals, compressedLength
        
        # Assert the validity
        assert self.encodeSignals.numCompressedSignals <= numSignals
        
        
        
        
        # Perform the optimal compression via PCA and embed channel information (for reconstruction).
        pcaProjection, principal_components = self.encodeSignals.pcaCompression(compressedData, self.numEncodedSignals, False)
        # Reconstruct the signals
        reconstructed = torch.matmul(principal_components, pcaProjection)
        # reconstructed = (reconstructed * compressedData.std(dim=-1, keepdim=True)) + compressedData.mean(dim=-1, keepdim=True)
        print("PCA:", (reconstructed - compressedData).pow(2).mean(dim=-1).mean(dim=1).mean())
        print("")
        
        # ------------------------ Data Augmentation ----------------------- #  

        numEncodedSignals = self.numEncodedSignals
                
        if trainingFlag: 
            # numEncodedSignals = numEncodedSignals + torch.randint(-int(numEncodedSignals/2), int(numEncodedSignals/2), (1,)).item()
            if self.numEncodedSignals == numSignals:
                numEncodedSignals = numEncodedSignals + torch.randint(-int(numEncodedSignals/2), int(numEncodedSignals/2), (1,)).item()
            elif  self.numEncodedSignals < numSignals:
                numEncodedSignals = numEncodedSignals + torch.randint(-int(numEncodedSignals/2), 0, (1,)).item()
            else:
                numEncodedSignals = numEncodedSignals + torch.randint(0, numEncodedSignals, (1,)).item()
            
            if numEncodedSignals == numSignals: numEncodedSignals + 2
                
        # Assert the integrity of the incoming data.
        assert compressedLength == self.compressedLength, f"The signals have length {compressedLength}, but the model expected {self.compressedLength} points."

        # Initialize holders for the output of each batch
        encodedData = torch.zeros((batchSize, numEncodedSignals, self.compressedLength), device=compressedData.device)
        reconstructedCompressedData = torch.zeros((batchSize, numSignals, self.compressedLength), device=encodedData.device) if decodeSignals else None
        
        # Calculate the size of each sub-batch
        subBatchSize = maxBatchSignals // numSignals
        numSubBatches = math.ceil(batchSize / subBatchSize)
        
        # Setup the loss calculation.
        # signalEncodingLayerLoss = torch.zeros((batchSize), device=compressedData.device)
        
        # Project the data into the correct subspace
        initialEncodedData = self.encodeSignals.signalEncodingInterface(compressedData, self.encodeSignals.initialTransformation)
        signalEncodingLayerLoss = self.encodeSignals.calculateStandardizationLoss(initialEncodedData, expectedMean = 0, expectedStandardDeviation = 1, dim=-1)
        
        # For each sub-batch of data.
        for subBatchIdx in range(numSubBatches):
            # Calculate start and end indices for the current sub-batch
            startIdx = subBatchIdx * subBatchSize
            endIdx = min((subBatchIdx + 1) * subBatchSize, batchSize)
            
           # --------------------- Signal Compression --------------------- # 
        
            # # Combine all signals into one vector.
            encodedData[startIdx:endIdx], numSignalPath, batchsignalEncodingLayerLoss \
                        = self.encodeSignals(signalData = initialEncodedData[startIdx:endIdx], 
                                            targetNumSignals = numEncodedSignals,
                                            calculateLoss = calculateLoss,
                                            signalEncodingLayerLoss = 0)
            # encodedData dimension: batchSize, numEncodedSignals, compressedLength
            print("Next: ", numSignalPath)
            
            # -------------------------------------------------------------- #  
            
            # -------------------- Signal Reconstruction ------------------- # 
            
            if decodeSignals:                
                # Set the starting point for the path.
                encodedPathData = encodedData[startIdx:endIdx]
                pathSignalDecodingLoss = 0
                
                # Follow the path back to the original signal.
                for pathInd in range(len(numSignalPath)-1, -1, -1):
                    # Reconstruct to the current signal number in the path.
                    encodedPathData, _, pathSignalDecodingLoss \
                            = self.encodeSignals(signalEncodingLayerLoss = pathSignalDecodingLoss,
                                                targetNumSignals = numSignalPath[pathInd], 
                                                signalData = encodedPathData,
                                                calculateLoss = calculateLoss)
                    # encodedPathData dimension: batchSize, numSignals, compressedLength
                # pathSignalDecodingLoss = pathSignalDecodingLoss/(len(numSignalPath) - 1) if 1 < len(numSignalPath) else pathSignalDecodingLoss
                    
                # Update the encoding information.
                reconstructedCompressedData[startIdx:endIdx] = encodedPathData
                signalEncodingLayerLoss[startIdx:endIdx] = signalEncodingLayerLoss[startIdx:endIdx] + batchsignalEncodingLayerLoss + pathSignalDecodingLoss
                # print("\nFinal", batchsignalEncodingLayerLoss, pathSignalDecodingLoss)
                print(batchsignalEncodingLayerLoss.mean().item(), pathSignalDecodingLoss.mean().item())
                
            # Clear the cache memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # -------------------------------------------------------------- # 
            
        # Undo the initial data projection.
        reconstructedCompressedData = self.encodeSignals.signalEncodingInterface(reconstructedCompressedData, self.encodeSignals.finalTransformation)
        
        if calculateLoss:
            reconstructedCompressedData_Hold = self.encodeSignals.signalEncodingInterface(initialEncodedData, self.encodeSignals.finalTransformation)

            initialReconstructionLoss = (compressedData - reconstructedCompressedData_Hold).pow(2).mean(dim=-1).mean(dim=1)
            signalEncodingLayerLoss = signalEncodingLayerLoss + initialReconstructionLoss
            print("initialReconstructionLoss:", initialReconstructionLoss.mean().item())
            
        print("END:", signalEncodingLayerLoss.mean().item())
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
    
    
    