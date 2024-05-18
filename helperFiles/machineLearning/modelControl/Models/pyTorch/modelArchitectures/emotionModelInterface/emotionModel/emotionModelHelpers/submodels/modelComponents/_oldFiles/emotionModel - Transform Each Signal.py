# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
# PyTorch
import torch

# Import helper files.
sys.path.append(os.path.dirname(__file__) + "/modelComponents/")
import emotionPredictions # Feed-Forward architecture for classifying emotions.
import positionEncodings  # Embedding architecture for position indices.
import transformerEncoder # Encoding architecture for highlighting signal information.
import transformerDecoder # Decoding architecture for adding contextual information.
import signalEmbedding    # Embedding architecture for signal information.
import autoencoder        # Autoencoder architecture for signal compression/decompression.
import featureEncoder  # Encoding architecture for highlighting feature information.
import featureReductor

# Import global model
sys.path.append(os.path.dirname(__file__) + "/../")
import _globalPytorchModel

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class modelHead(_globalPytorchModel.globalModel):
    def __init__(self, outputSizes, numSignals, emotionNames, maxSeqLength, contextualLength, metaTraining = False):
        super(modelHead, self).__init__()
        # General model parameters.
        self.contextualLength = contextualLength # The number of contextual information.
        self.numEmotions = len(emotionNames)     # The number of emotions to predict.
        self.metaTraining = metaTraining         # A flag if this is metatraining data.
        self.maxSeqLength = maxSeqLength         # The maximum length of an indivisual signal.
        self.emotionNames = emotionNames         # The names of the final predicted emotions.
        self.outputSizes = outputSizes           # A list of numbers, indicating the number of answers for each survey question.
        self.numSignals = numSignals             # The number of signals going into the model.

        # Indivisual signal: model parameters.
        self.compressedLength = 16   # The final length of the compressed signal after the autoencoder.
        self.embeddedFeatureDim = 4    # The number of final features (embedding space) for each indivisual signal.
        self.numIndicesShift = 2       # The number of indices to shift during signal embedding.

        # ----------------------- Signal Compression ----------------------- #  
        
        # Initialize signal compressor.
        self.compressSignals = autoencoder.encodingLayer(
                    compressedLength = self.compressedLength,
                    maxSeqLength = self.maxSeqLength,
        )
        
        # ---------------------- Signal Reconstruction --------------------- #  
        
        # Initialize procedures for signal reconstruction.
        self.reconstructSignals = autoencoder.decodingLayer(
                    compressedLength = self.compressedLength,
                    maxSeqLength = self.maxSeqLength,
        )
        
        # ----------------------- Model Preprocessing ---------------------- #    
        
        # Embed the signal with whole sequence information.
        self.signalEncodedDim = self.compressedLength + (1 if self.compressedLength % 2 != 0 else 0)  # Dimension must be even for position embedding.
        self.signalEncodedDim = self.signalEncodedDim // self.numIndicesShift
        self.signalEncodedDim += (1 if self.signalEncodedDim % 2 != 0 else 0) 
        
        # Initialize procedures for adding short-term information.
        self.embedIndices = signalEmbedding.signalEmbedding(
            numIndicesShift = self.numIndicesShift,
            embeddingDim = self.signalEncodedDim,
            clampBoundaries = True
        )
        
        # Initialize procedures for adding position information.
        self.embedPositions = positionEncodings.PositionalEncoding(
                    embeddingDim = self.signalEncodedDim, 
                    maxSeqLength = self.compressedLength, 
                    n = 5
        )
                
        # ------------------------- Signal Encoding ------------------------ #  
        
        # Initialize tranformer for self-attention.
        self.transformerEncoding = transformerEncoder.transformerEncoder(
                    embeddingDim = self.signalEncodedDim,
                    num_heads = 1,
                    numLayers = 2,
        )
        
        # Initialize tranformer for self-attention.
        self.transformerDecoding = transformerDecoder.transformerDecoder(
                    contextualLength = self.contextualLength, 
                    signalDim = self.signalEncodedDim, 
                    numContextHeads = 1, 
                    numSignalHeads = 1, 
                    numLayers = 2,
        )
        
        # Initialize feature extraction.
        self.reduceFeatures = featureReductor.featureReductor(
                    inputShape = (self.compressedLength, self.signalEncodedDim),
                    numFeatures = self.embeddedFeatureDim,
        )
        
        # --------------------- Emotion Classification --------------------- #  
        
        # Initialize tranformer for self-attention.
        self.transformFeatures = featureEncoder.featureEncoder(
                    embeddingDim = self.embeddedFeatureDim,
                    num_heads = 1,
                    numLayers = 2,
        )
        
        self.classifyEmotions = emotionPredictions.emotionClassification(
                    allFeatures = self.numSignals*self.embeddedFeatureDim,
                    allNumAnswers = outputSizes,
        )                
        
        # ------------------------------------------------------------------ #  

    def forward(self, inputData, trainingData = False, compileFeatures = False, allTrainingData = False):
        """ The shape of inputData: (batchSize, signalInfoLength, numSignals) """
        
        # ----------------------- Data Preprocessing ----------------------- #  

        # Seperate out indivisual signals.
        batchSize, signalInfoLength, numSignals = inputData.size()
        signalInfo = inputData.transpose(1, 2).reshape(batchSize * numSignals, signalInfoLength)
        # signalInfo dimension: batchSize*numSignals, sequenceLength + contextualLength

        # Seperate the sequence from contextual information.
        signalData = signalInfo[:, 0:self.maxSeqLength].to(torch.float32)
        contextualData = signalInfo[:, self.maxSeqLength:]
        # contextualData dimension = batchSize*numSignals, contextualLength
        # signalData dimension = batchSize*numSignals, sequenceLength
        assert self.contextualLength == contextualData.size(1)

        # ----------------------- Signal Compression ----------------------- #  
                
        # Data reduction: remove unnecessary timepoints from the signals.
        compressedData = self.compressSignals(signalData)  # Apply CNN for feature compression.
        # compressedData dimension: batchSize*numSignals, compressedLength
                        
        # ------------------------ Signal Embedding ------------------------ #  
                
        # Embed the signals with short and long-term information
        embeddedData = self.embedIndices(compressedData)    # Group each timepoint with surrounding points (sliding window).
        embeddedData = self.embedPositions(embeddedData)    # Embed the position information (sinusoidal encoding).
        # embeddedData dimension: batchSize*numSignals, self.compressedLength, self.signalEncodedDim
                
        # ------------------------- Signal Encoding ------------------------ #  
        
        # Apply self-attention to learn what to focus on in the data.
        # encodedData = self.transformerEncoding(embeddedData, allTrainingData) 
        # encodedData = self.transformerDecoding(encodedData, contextualData, allTrainingData) 
        # encodedData dimension: batchSize*numSignals, self.compressedLength, self.signalEncodedDim
        
        # Downsize the number of extracted features.
        encodedData = self.reduceFeatures(embeddedData) 
        # encodedData dimension: batchSize*numSignals, embeddedFeatureDim

        # --------------------- Emotion Classification --------------------- #  

        # Apply self-attention to learn what features to focus on in the data.
        featureData = encodedData.view(batchSize, numSignals, self.embeddedFeatureDim) # Organize the features into the correct signals.
        featureData = self.transformFeatures(featureData, allTrainingData) 
        # featureData dimension: batchSize, numSignals, embeddedFeatureDim
        
        # Compile all the signal features into one vector.
        compiledFeatures = featureData.view(batchSize, numSignals*self.embeddedFeatureDim) # The new dimension: batchSize, numSignals*embeddedFeatureDim 
        # featureData dimension: batchSize, numSignals*embeddedFeatureDim

        # Return the compiled features.
        if compileFeatures: return compiledFeatures
        
        # Predict the final probability distributions for each emotion.
        finalPredictions = self.classifyEmotions(compiledFeatures)
        # The new dimension: numEmotions, batchSize, numEmotionClasses (Not: numEmotionClasses is emotion specific)
        
        # ---------------------- Signal Reconstruction --------------------- #  

        if trainingData:    
            # Reconstruct the initial signals.
            decompressedData = self.reconstructSignals(compressedData)  # decompressedData dimension: batchSize*numSignals, sequenceLength
            reconstructedSignals = decompressedData.view(batchSize, numSignals, self.maxSeqLength).transpose(1, 2)   # Organize the signals into the original batches.
            # reconstructedSignals dimension: batchSize, sequenceLength, numSignals
            
            return finalPredictions, reconstructedSignals
            
        # ------------------------------------------------------------------ #  

        return finalPredictions
    
    
    