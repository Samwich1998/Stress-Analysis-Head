#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 12:15:27 2023

@author: samuelsolomon
"""


# General
import numpy as np
import matplotlib.pyplot as plt

# Embedding
import gensim.downloader as api
from transformers import DistilBertTokenizer, DistilBertModel


# Load pre-trained GloVe word embeddings
glove_model = api.load("glove-wiki-gigaword-100")


# Load pre-trained sentiment analysis model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)


staiQuestions = [
  "I feel calm",
  "I feel secure",
  "I feel at ease",
  "I feel satisfied",
  "I feel comfortable",
  "I feel self-confident",
  "I am relaxed",
  "I feel content",
  "I feel steady",
  "I feel pleasant",
  "I am tense",
  "I feel strained",
  "I feel upset",
  "I am presently worrying over possible misfortunes",
  "I feel frightened",
  "I feel nervous",
  "I am jittery",
  "I feel indecisive",
  "I am worried",
  "I feel confused",
]

staiQuestions = [
  "calm",
  "secure",
  "ease",
  "satisfied",
  "comfortable",
  "self-confident",
  "relaxed",
  "content",
  "steady",
  "pleasant",
  "tense",
  "strained",
  "upset",
  "worrying misfortunes",
  "frightened",
  "nervous",
  "jittery",
  "indecisive",
  "worried",
  "confused",
]



finalEmbeddings = []
for sentence in staiQuestions:
    
    
    tokens = sentence.lower().split()
    sentence_vector = np.mean([glove_model[token] for token in tokens], axis=0)
    
    
    # # Tokenize sentence and convert to input tensor
    # inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors="pt")
    # input_ids = inputs["input_ids"]
    # # Pass input tensor through the model to obtain sentiment-aware representations
    # outputs = model(input_ids)
    # sentence_vector = outputs.last_hidden_state.mean(dim=1).squeeze()
    # # Convert PyTorch tensor to NumPy array
    # sentence_vector = sentence_vector.detach().numpy()
    
    finalEmbeddings.append(sentence_vector)

similarity_matrix = np.zeros((len(staiQuestions), len(staiQuestions)))
for i in range(len(staiQuestions)):
    for j in range(i, len(staiQuestions)):
        similarity_matrix[i, j] = np.dot(finalEmbeddings[i], finalEmbeddings[j]) / (np.linalg.norm(finalEmbeddings[i]) * np.linalg.norm(finalEmbeddings[j]))
        similarity_matrix[j, i] = similarity_matrix[i, j]


# Plot the similarity matrix as a heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(similarity_matrix, cmap='hot')

# Add labels to the x and y axis
ax.set_xticks(np.arange(len(staiQuestions)))
ax.set_yticks(np.arange(len(staiQuestions)))
ax.set_xticklabels(staiQuestions)
ax.set_yticklabels(staiQuestions)

# Rotate the x-axis labels to make them easier to read
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Set the title of the plot
ax.set_title("Image similarity matrix")
plt.tight_layout()

# Show the plot
fig.savefig("emotionSimilarity.png", dpi=300)
plt.show()


