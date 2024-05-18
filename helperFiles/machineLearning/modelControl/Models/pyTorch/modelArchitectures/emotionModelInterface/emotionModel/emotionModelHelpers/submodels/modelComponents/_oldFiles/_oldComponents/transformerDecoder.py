# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch.nn as nn
# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class transformerDecoder(nn.Module):
    def __init__(self, signalDim, contextualLength, numContextHeads, numSignalHeads, numLayers):
        super(transformerDecoder, self).__init__()  
        # Contextual model parameters.
        self.numContextHeads = numContextHeads
        self.contextualLength = contextualLength
        # Signal model parameters.
        self.numSignalHeads = numSignalHeads
        self.signalDim = signalDim
        # General model parameters.
        self.numLayers = numLayers
        
        # Define the attention mechanism for contextual information.
        self.contextAttention = nn.MultiheadAttention(
            num_heads = self.numContextHeads,  # Number of parallel attention heads. Note that embed_dim will be split across num_heads (i.e. each head will have dimension embed_dim // num_heads).
            embed_dim = 1,  # The embdedded dimension of the query.
            kdim = 1,       # Total number of features for keys. Default: None (uses kdim=embed_dim).
            vdim = 1,       # Total number of features for values. Default: None (uses vdim=embed_dim).
            batch_first = True,     # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
            add_bias_kv = True,     # If specified, adds bias to the key and value sequences at dim=0. Default: False.
            add_zero_attn = 0,      # Appends zeros to the key and value sequences at dim=1 for size consitency. Default: False.
            dropout = 0.2,          # Dropout probability on attn_output_weights (commonly 0.1-0.3). Default: 0.0 (no dropout).
            bias = True,            # If specified, adds bias to input / output projection layers. Default: True.
        )
        
        # Define the attention mechanism to integrate  contextual information
        self.crossAttention = nn.MultiheadAttention(
            num_heads = self.numSignalHeads,   # Number of parallel attention heads. Note that embed_dim will be split across num_heads (i.e. each head will have dimension embed_dim // num_heads).
            embed_dim = signalDim,  # The embdedded dimension of the query.
            kdim = 1,               # Total number of features for keys. Default: None (uses kdim=embed_dim).
            vdim = signalDim,       # Total number of features for values. Default: None (uses vdim=embed_dim).
            batch_first = True,     # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
            add_bias_kv = True,     # If specified, adds bias to the key and value sequences at dim=0. Default: False.
            add_zero_attn = 0,      # Appends zeros to the key and value sequences at dim=1 for size consitency. Default: False.
            dropout = 0.2,          # Dropout probability on attn_output_weights (commonly 0.1-0.3). Default: 0.0 (no dropout).
            bias = True,            # If specified, adds bias to input / output projection layers. Default: True.
        )
        
        # Add non-linearity to attention.
        self.feedForwardLayers = nn.Sequential(
            # It is common to expand the embeddingDim
                # By a factor of 2-4 (small), 
                # By a factor of 4-8 (medium), 
                # By a factor of 8-16 (large).
                
            # Neural architecture: Layer 1.
            nn.Linear(signalDim, 32, bias = True),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            # Neural architecture: Layer 1.
            nn.Linear(32, 16, bias = True),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            # Neural architecture: Layer 2.
            nn.Linear(16, signalDim, bias = True),
            nn.SiLU(),
        )
        
        self.synthesizeContext = nn.Sequential( 
            # Neural architecture: Layer 1.
            nn.Linear(contextualLength, 18, bias = True),
            nn.BatchNorm1d(18, track_running_stats = False),
        )
        
        # Initialize the layer normalization.

        self.layerNorm_FF = nn.LayerNorm(signalDim, eps = 1E-10)
        self.layerNorm_Final = nn.LayerNorm(signalDim, eps = 1E-10)
        self.layerNorm_CA_signal = nn.LayerNorm(signalDim, eps = 1E-10)
        self.layerNorm_CA_context = nn.LayerNorm(contextualLength, eps = 1E-10)
                
        # Initialize holder parameters.
        self.allAttentionWeights = []  # Dimension: numEpochs, numLayers, attentionWeightDim, where attentionWeightDim = batch_size, numHeads, seq_length, seq_length

    def forward(self, signalData, contextualData, allTrainingData = False):
        """ 
        The shape of signalData = (batchSize, sequenceLength, signalDim)
        The shape of contextualData = (batchSize, contextualLength)
        """
        # if allTrainingData: self.allAttentionWeights.append([]);# print("")
        # Expand the contextual data.
        expandedContext = self.synthesizeContext(contextualData)
        expandedContext = expandedContext.view(expandedContext.size(0), -1, 1)
        # contextualData dimension = batchSize, signalDim, 1
        
        # For each encoding layer.
        for layerInd in range(self.numLayers):
            # Combine the values from the self attentuation block.
            selfAttentionData = self.selfAttentionBlock(expandedContext, allTrainingData)
            expandedContext = expandedContext + selfAttentionData

            # Combine the values from the cross attentuation block.
            crossAttentionData = self.crossAttentionBlock(signalData, expandedContext, allTrainingData)
            signalData = signalData + crossAttentionData
            
            # Combine the values from the feed-forward block.
            feedForwardData = self.feedForwardBlock(signalData) # Apply feed-forward block.
            signalData = signalData + feedForwardData
                        
        # Final normalization after encoding.
        signalData = self.layerNorm_Final(signalData)
                
        return signalData
    
    def selfAttentionBlock(self, inputData, allTrainingData):
        # Apply self attention block to the signals.
        attentionOutput, attentionWeights = self.contextAttention(inputData, inputData, inputData, 
                                                                  average_attn_weights = False, need_weights = True)
        # if allTrainingData: self.allAttentionWeights[-1].append(attentionWeights[0:1])  # I am indexing 0:1 to only take the first batch due to memory constraints.
        
        return attentionOutput
    
    def crossAttentionBlock(self, signalData, contextualData, allTrainingData):
        # Apply self attention block to the signals.
        normSignalData = self.layerNorm_CA_signal(signalData)
        attentionOutput, attentionWeights = self.crossAttention(query = normSignalData, 
                                                                value = normSignalData, 
                                                                key = contextualData, 
                                                                need_weights = True,
                                                                average_attn_weights = False)
        # if allTrainingData: self.allAttentionWeights[-1].append(attentionWeights[0:1])  # I am indexing 0:1 to only take the first batch due to memory constraints.
                
        return attentionOutput
    
    def feedForwardBlock(self, signalData):
        # Apply feed forward block to the signals.
        normSignalData = self.layerNorm_FF(signalData)
        outputANN = self.feedForwardLayers(normSignalData)
        
        return outputANN
    
    def visualizeAttention(self, modelInd, batchInd  = 0):
        if len(self.allAttentionWeights) == 0: return  None
        
        numPlots = self.num_heads*self.numLayers
        # Calculate the number of attention heads and sequence length
        numPlots_perRow = min(2, numPlots)
        num_rows = (numPlots + numPlots_perRow - 1) // numPlots_perRow

        movieTitle = "Visualization of Self-Attention"
        # Initialize Movie Writer for Plots
        metadata = dict(title=movieTitle, artist='Matplotlib', comment='Movie support!')
        writer = manimation.FFMpegWriter(fps=5, metadata=metadata)

        # Create subplots for each attention head
        fig, axes = plt.subplots(nrows=num_rows, ncols=numPlots_perRow, figsize=(15, 5*num_rows), sharex=True, sharey=True)
        flattenedAxes = axes.ravel() if numPlots != 1 else [axes]

        # Initialize image objects for each subplot
        image_objects = []
        for plotInd in range(numPlots):
            ax = flattenedAxes[plotInd]
            layerInd = plotInd//self.num_heads
            headInd = plotInd%self.num_heads
            
            # Get the colorbar boundaries.
            minVal = self.allAttentionWeights[0][0][batchInd].min().item()
            maxVal = self.allAttentionWeights[0][0][batchInd].max().item()
            
            # import torch
            
            # all_attention_weights = torch.cat(self.allAttentionWeights[-1], dim=0)  # Concatenate attention weights for all layers and epochs
            # minVal = all_attention_weights[batchInd, headInd, :, :-1].min().item()
            # maxVal = all_attention_weights[batchInd, headInd, :, :-1].max().item()

            
            # Add a plot for each head.
            im = ax.imshow(self.allAttentionWeights[0][0][batchInd, 0, :, :-1].detach().numpy(), 
                           cmap='RdBu_r', interpolation='nearest', animated=True, origin='lower', 
                           vmin=minVal, vmax=maxVal)
            image_objects.append(im)
            # Set figure information.
            ax.set_xlabel('Target Sequence Length')
            ax.set_ylabel('Source Sequence Length')
            ax.set_title(f'Attention Head {headInd+1} at Layer {layerInd+1}')
            
        # Add colorbar (created only once)
        cbar = fig.colorbar(image_objects[0], ax=flattenedAxes, shrink=0.4, aspect=5)
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        cbar.ax.set_ylabel('Attention Weight')
        
        # Open the movie and add the data.
        with writer.saving(fig, "videos/" + movieTitle + f"_{modelInd}.mp4", 300):
            # For each training epoch.
            for epoch in range(len(self.allAttentionWeights)):
                # For each head at rach layer.
                for plotInd in range(numPlots):
                    layerInd = plotInd//self.num_heads
                    headInd = plotInd%self.num_heads
                    ax = flattenedAxes[plotInd]
                    im = image_objects[plotInd]

                    # Update the data in the image object
                    im.set_array(self.allAttentionWeights[epoch][layerInd][batchInd, headInd, :, :-1].detach().numpy())
                    fig.suptitle(f'Visualization of Self-Attention Weights: Epoch {epoch}')
                        
                writer.grab_frame()
                
        # Clear plots
        plt.clf(); plt.close(fig)   
        
        
        