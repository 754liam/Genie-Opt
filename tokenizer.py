# implementation of appendix F.2 
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=64, commitment_cost=0.25): # hyperparameters - vocab size, 'word' dimension, and loss function definition.
        super(VectorQuantizer, self).__init__() # calls constructor of nn.Module
        
        self._embedding = nn.Embedding(num_embeddings, embedding_dim) # dictionary matrix, where rows represent tokens and columns represent meaning expressed as coordinates.
        self._embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings) # initialization - we use the inverse of such in our args as a range to enforce small numbers.
        self._commitment_cost = commitment_cost # used for calculating loss - defines encoders snap range.
        self._num_embeddings = num_embeddings  # defines num_embeddings - the size of the dictionary.
        self._embedding_dim = embedding_dim # defines embedding_dim - the dimension of the tokens in the dictionary.

    def forward(self, inputs):
        # inputs will come in from the encoder as a torch tensor of shape [Batch_Size, 64, 16, 16]. each frame from data_collection.py is 64 x 64 and the encoder will shrink this down into a 16 x 16 'image', but the depth will go from 3 (rgb) to 64.
        # inputs must first be reshaped from [batch, channel, height, width] to [batch, height, width, channel]. note that each 'pixel' is a a vector of 64 numbers describing it. THIS IS IMPORTANT.
        inputs = inputs.permute(0,2,3,1).contiguous()
        inputs_shape = inputs.shape
        # turning the shape into [0 * 1 * 2, 3] where 0-3 represent indices of the inputs shape. this converts the stack of batch_size to a list of batch_size. essentially, this creates a list of all the vectors. 
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # the goal is to find the distance from some input vector A to a codebook vector B using the Euclidean Distance formula (simply the abs of A - B squared.) this is used to see which codebook vector is closest to the input vector. 
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self._embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # note that distances is a Tensor. torch.argmin(distances) is w.r.t. distances. this finds the closest codebook vector.
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # we create a vector that is all zeroes except a 1 at the index of the codebook vector found above. 
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # we then multiply this zero vector (with 1 at the proper index) by the codebook matrix, which essentially isolates the proper vector. we then put it back into its proper shape. NOTE: this literally just extracts the row of the codebook matrix to the proper codebook vector we're looking for!!!
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs_shape)
         
        # loss invocation. first line is a encoder loss - where the encoder moves its ouput closer to the codebook vector. the second line is the opposite. 
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # message to be sent to the decoder. this seemingly redundant schema is used to fix the gradient attribute of quantized. we're removing the undefined gradient of quantized and just keeping it as the gradient of inputs itself.
        quantized = inputs + (quantized - inputs).detach()

        #this line is for seeing the percentage frequency of every vector chosen in the codebook.
        avg_probs = torch.mean(encodings, dim=0)
        #this line is for exponential entropy - it gives you a range of how many unique vectors are actually chosen in the codebook.
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # loss for backpropagation on the encoder; quantized returned in decoder safe form; perplexity for codebook analysis; encodings for debugging purposes (just the zero vector with one 1 representing the most closely aligned vectors index on the codebook).
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

