# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Pytorch
import torch
import torch.nn as nn

# -------------------------------------------------------------------------- #
# --------------------------- Attention Methods --------------------------- #

class attentionMethods(nn.Module):
    def __init__(self, attentionType, embeddingDim, num_heads, numLayers):
        super(attentionMethods, self).__init__()  
        # General model parameters.
        self.attentionType = attentionType
        self.embeddingDim = embeddingDim
        self.numLayers = numLayers
        self.num_heads = num_heads
        
        if self.attentionType == "bilinear":
            self.attentionMechanism = self.bilinear
        elif self.attentionType == "multiLayerPerception":
            self.attentionMechanism = self.multiLayerPerception
        else:
            assert False
        
    def forward(self):
        self.attentionMechanism()
        

    def bilinear(self, decoder_state, encoder_states):
        decoder_state = self.linear(decoder_state)
        scores = torch.matmul(decoder_state, encoder_states.transpose(1, 2))
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, encoder_states)
        return context, attention_weights
        

    def multiLayerPerception(self, decoder_state, encoder_states):
        query = self.W_query(decoder_state).unsqueeze(1)
        encoder_transformed = self.W_encoder(encoder_states)
        scores = self.V(torch.tanh(query + encoder_transformed)).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_states).squeeze(1)
        return context, attention_weights