# implementation of appendix F.2 
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=64, commitment_cost=0.25): # hyperparameters - vocab size, 'word' dimension, and loss function definition.
        super(VectorQuantizer, self).__init__() # calls constructor of nn.Module
        
        self._embedding = nn.Embedding(num_embeddings, embedding_dim) # dictionary matrix, where rows represent tokens and columns represent meaning expressed as coordinates.
        self._embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings) # initialization - we use the inverse of such in our args as a range to enforce small numbers.
        self._commitment_cost = commitment_cost # used for calculating loss

    def forward(self, inputs):
        pass