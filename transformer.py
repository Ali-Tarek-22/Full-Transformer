import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
import numpy as np
import pandas as pd
import re

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_indices = torch.arange(0, self.d_model, 2).float() # d_model/2 # Even indices
        denominator = torch.pow(10000, even_indices/self.d_model) # d_model/2 # Denominator (same for even and odd indices)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1) # max_sequence_length x 1 # 0 -> max_sequence_length
        even_PE = torch.sin(position / denominator) # max_sequence_length x d_model/2
        odd_PE = torch.cos(position / denominator) # max_sequence_length x d_model/2
        stacked = torch.stack([even_PE, odd_PE], dim=2) # max_sequence_length x d_model/2 x 2
        PE = torch.flatten(stacked, start_dim=1, end_dim=2) # max_sequence_length x d_model
        return PE.unsqueeze(0) # 1 x max_sequence_length x d_model


def get_device(): # Return cuda (GPU) if possible, else return cpu
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model) # maps token IDs → dense vectors of size d_model
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        # compute positional encoding
        self.positional_encoding = self.position_encoder().to(get_device())


    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token): # Tokenize the sentence
            # Convert tokens into indices
            sentence_indices = [self.language_to_index[token] for token in list(sentence)] # Iterate over letters, if u want to iterate over words, use sentence.split(), if you switch to words, make sure the vocabulary is words not letters, also that max_seq_len is in number of words not letters, finally You may want a <UNK> token for words not in your vocabulary, which wasn’t needed for characters.
            # Insert start token at the begining
            if start_token:
                sentence_indices.insert(0, self.language_to_index[self.START_TOKEN])
            # Append the end token at the end
            if end_token:
                sentence_indices.append(self.language_to_index[self.END_TOKEN])
            # Pad to max_sequence_length
            for _ in range(len(sentence_indices), self.max_sequence_length):
                sentence_indices.append(self.language_to_index[self.PADDING_TOKEN])
            # Truncate if too long
            sentence_indices = sentence_indices[:self.max_sequence_length]
            return torch.tensor(sentence_indices)

        tokenized = []
        for sentence in range(len(batch)): # Iterate through sentences in the batch
           tokenized.append(tokenize(batch[sentence], start_token, end_token)) # Tokenize the sentence
        # tokenized now is a python list of tensor each with length max_seq_len, the list has batch_size tensors
        # We want to convert this list of tensors into a 2d tensor with shape batch_size x max_seq_len
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device()) # get_device() returns the device (GPU, CPU ..) then .to(device) moves the tensor to that device
        # This ensures that your tokenized tensor is on the same device as the model before you feed it into the embedding layer. Without it, you could get a device mismatch error .
    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x) # Convert tokens into embeddings
        x = self.dropout(x + self.positional_encoding) # When you do x + positional_encoding, PyTorch requires both tensors to be on the same device. If x is on GPU but pos is still on CPU, you would get an error, that's why we also moved the positional encoding to gpu.
        return x
        # Now x is on GPU (if possible), any further computation on it, the output will be on the same device. If a tensor is computed from other tensors already on the correct device, it automatically stays on that device.


class Masks(nn.Module):
  def __init__(self, max_seq_len):
    super().__init__()
    self.max_seq_len = max_seq_len
    # Compute the look ahead mask in __init__ only once as it's the same for all batches avoiding recreating it every forward pass — more efficient.
    self.look_ahead_mask = torch.triu(torch.ones((self.max_seq_len, self.max_seq_len), dtype = torch.bool), diagonal = 1) # All elements below diagonal + diagonal are False

  def forward(self, enc_lang_batch, dec_lang_batch,
                  enc_start_token, enc_end_token,
                  dec_start_token, dec_end_token, NEG_INF = -1e9):
          num_sentences = len(enc_lang_batch)
          # Create the masks with boolean values where Trues will be replaced by -inf and Falses with 0
          encoder_padding_mask_selfAttention = torch.zeros((num_sentences, self.max_seq_len, self.max_seq_len), dtype = torch.bool) # the matrix is filled with zeros, then making the type boolean converts them to False
          decoder_padding_mask_selfAttention = torch.zeros((num_sentences, self.max_seq_len, self.max_seq_len), dtype = torch.bool)
          decoder_padding_mask_crossAttention = torch.zeros((num_sentences, self.max_seq_len, self.max_seq_len), dtype = torch.bool)
          
          for idx in range(num_sentences):
            enc_lang_sent_len, dec_lang_sent_len = len(enc_lang_batch[idx]), len(dec_lang_batch[idx])

            enc_padding_start = enc_lang_sent_len + (1 if enc_start_token else 0) + (1 if enc_end_token else 0)
            enc_lang_padding = np.arange(enc_padding_start , self.max_seq_len)

            dec_padding_start = dec_lang_sent_len + (1 if dec_start_token else 0) + (1 if dec_end_token else 0)
            dec_lang_padding = np.arange(dec_padding_start , self.max_seq_len)

            encoder_padding_mask_selfAttention[idx, :, enc_lang_padding] = True
            encoder_padding_mask_selfAttention[idx, enc_lang_padding, :] = True

            decoder_padding_mask_selfAttention[idx, :, dec_lang_padding] = True
            decoder_padding_mask_selfAttention[idx, dec_lang_padding, :] = True

            decoder_padding_mask_crossAttention[idx, :, enc_lang_padding] = True
            # decoder_padding_mask_crossAttention[idx, dec_lang_padding, :] = True # Remove the line masking decoder padding in cross-attention.
            # It is redundant and incorrect; only encoder padding should be masked there.

          encoder_mask_selfAttention = torch.where(encoder_padding_mask_selfAttention, NEG_INF, 0)
          decoder_mask_selfAttention = torch.where(decoder_padding_mask_selfAttention | self.look_ahead_mask, NEG_INF, 0)
          decoder_mask_crossAttention = torch.where(decoder_padding_mask_crossAttention, NEG_INF, 0)

          encoder_mask_selfAttention = encoder_mask_selfAttention.unsqueeze(1)
          decoder_mask_selfAttention = decoder_mask_selfAttention.unsqueeze(1)
          decoder_mask_crossAttention = decoder_mask_crossAttention.unsqueeze(1)

          return encoder_mask_selfAttention.to(get_device()), decoder_mask_selfAttention.to(get_device()), decoder_mask_crossAttention.to(get_device())


# Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # d_model
        self.num_heads = num_heads # num_heads
        self.head_dim = d_model // num_heads # head_dim
        self.qkv_layer = nn.Linear(d_model , 3 * d_model) # d_model x 3*d_model
        self.linear_layer = nn.Linear(d_model, d_model) # d_model

    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size()  # Batch x max_seq_len, d_model
        qkv = self.qkv_layer(x) # Batch x max_seq_len x 3*d_model
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim) # Batch x max_seq_len x num_heads x 3*head_dim
        qkv = qkv.permute(0, 2, 1, 3) # Batch x num_heads x max_seq_len x 3*head_dim
        q, k, v = qkv.chunk(3, dim=-1) # Batch x num_heads x max_seq_len x head_dim each
        values, attention = self.scaled_dot_product(q, k, v, mask) # Batch x num_heads x max_seq_len x head_dim
        values = values.permute(0, 2, 1, 3).reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim) # value = Batch x max_seq_len x d_model, attention = Batch x num_heads x max_seq_len x max_seq_len
        out = self.linear_layer(values) # Batch x max_seq_len x d_model
        return out

    def scaled_dot_product(self, q, k, v, mask=None):
        # q, k, v are Batch x num_heads x max_seq_len x head_dim each
        d_k = q.size()[-1] # head_dim
        scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) # Batch x num_heads x max_seq_len x max_seq_len
        if mask is not None:
            scaled += mask
        attention = F.softmax(scaled, dim=-1) # Batch x num_heads x max_seq_len x max_seq_len
        values = torch.matmul(attention, v) # Batch x num_heads x max_seq_len x head_dim
        return values, attention


# MultiHead Cross Attention
class MultiHeadCrossAttention(nn.Module):
  def __init__(self, d_model, num_heads):
      super().__init__()
      self.d_model = d_model
      self.num_heads = num_heads
      self.head_dim = d_model // num_heads
      self.kv_layer = nn.Linear(d_model , 2 * d_model) # 1024
      self.q_layer = nn.Linear(d_model , d_model)
      self.linear_layer = nn.Linear(d_model, d_model)

  def forward(self, x, y, mask=None):
      batch_size, sequence_length, d_model = x.size() # Batch x max_seq_len x d_model
      kv = self.kv_layer(x) # Batch x max_seq_len x 1024 # Output of Encoder to generate the keys and values
      q = self.q_layer(y) # Batch x max_seq_len x d_model
      kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)  # Batch x max_seq_len x num_heads x 2*head_dim
      q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)  # Batch x max_seq_len x num_heads x head_dim
      kv = kv.permute(0, 2, 1, 3) # Batch x num_heads x max_seq_len x 2*head_dim
      q = q.permute(0, 2, 1, 3) # Batch x num_heads x max_seq_len x head_dim
      k, v = kv.chunk(2, dim=-1) # K: Batch x num_heads x max_seq_len x head_dim, v: Batch x num_heads x max_seq_len x head_dim
      values, attention = scaled_dot_product(q, k, v, mask) #  Batch x num_heads x max_seq_len x head_dim
      values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model) #  Batch x max_seq_len x d_model
      out = self.linear_layer(values)  #  Batch x max_seq_len x d_model
      return out  #  Batch x max_seq_len x d_model


# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape # d_model
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # d_model
        self.beta =  nn.Parameter(torch.zeros(parameters_shape)) # d_model

    def forward(self, inputs): # Batch x max_seq_len x d_model
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] # [-1]
        mean = inputs.mean(dim=dims, keepdim=True) # Batch x max_seq_len x 1 (1 bcs of keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True) # Batch x max_seq_len x 1
        std = (var + self.eps).sqrt() # Batch x max_seq_len x d_model
        y = (inputs - mean) / std # Batch x max_seq_len x d_model
        out = self.gamma * y  + self.beta # Batch x max_seq_len x d_model
        return out


# Postion-wise Feedforward
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__() # After python 3, you can remove the class name and self arguments from super, just write super().__init__() like we did in the classes above
        self.linear1 = nn.Linear(d_model, hidden) # d_model x 2045
        self.linear2 = nn.Linear(hidden, d_model) # 2048 x d_model
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x): # Batch x max_seq_len x d_model
        x = self.linear1(x) # Batch x max_seq_len x 2048
        x = self.relu(x) # Batch x max_seq_len x 2048
        x = self.dropout(x) # Batch x max_seq_len x 2048
        x = self.linear2(x) # Batch x max_seq_len x d_model
        return x

# Single Encoder Layer
class EncoderLayer(nn.Module):
  def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
    super(EncoderLayer, self).__init__() # After python 3, you can remove the class name and self arguments from super, just write super().__init__() like we did in the classes above
    self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    self.norm1 = LayerNormalization(parameters_shape=[d_model])
    self.dropout1 = nn.Dropout(p=drop_prob)
    self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
    self.norm2 = LayerNormalization(parameters_shape=[d_model])
    self.dropout2 = nn.Dropout(p=drop_prob)

  def forward(self, x, mask):
      residual_x = x # Batch x max_seq_len x d_model
      x = self.attention(x, mask=mask) # Batch x max_seq_len x d_model
      x = self.dropout1(x) # Batch x max_seq_len x d_model
      x = self.norm1(x + residual_x) # Batch x max_seq_len x d_model
      residual_x = x # Batch x max_seq_len x d_model
      x = self.ffn(x) # Batch x max_seq_len x d_model
      x = self.dropout2(x) # Batch x max_seq_len x d_model
      x = self.norm2(x + residual_x) # Batch x max_seq_len x d_model
      return x


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,max_sequence_length,language_to_index,START_TOKEN,END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x


# Decoder Layer
class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob,):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.multi_head_cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        residual_y = y # Batch x max_seq_len x d_model
        y = self.self_attention(y, mask=self_attention_mask) # Batch x max_seq_len x d_model
        y = self.dropout1(y) # Batch x max_seq_len x d_model
        y = self.norm1(y + residual_y) # Batch x max_seq_len x d_model

        residual_y = y # Batch x max_seq_len x d_model
        y = self.multi_head_cross_attention(x, y, mask=cross_attention_mask) #Batch x max_seq_len x d_model
        y = self.dropout2(y)
        y = self.norm2(y + residual_y)  #Batch x max_seq_len x d_model

        residual_y = y  #Batch x max_seq_len x d_model
        y = self.ffn(y) #Batch x max_seq_len x d_model
        y = self.dropout3(y) #Batch x max_seq_len x d_model
        y = self.norm3(y + residual_y) #Batch x max_seq_len x d_model
        return y #Batch x max_seq_len x d_model


# Sequential Decoder
class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask) #Batch x max_seq_len x d_model # the output y is passed to the next decoder layer and so on
        return y


# Full Decoder
class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        #x : Input (output of decoder):  Batch x max_seq_len x d_model
        #y : Previous Decoder output(ground-truth for first decoder) :  Batch x max_seq_len x d_model
        #mask : max_seq_len x max_seq_len
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y # Batch x max_seq_len x d_model


class Transformer(nn.Module):
    def __init__(self,
                d_model,
                ffn_hidden,
                num_heads,
                drop_prob,
                num_layers,
                max_sequence_length,
                out_lang_vocab_size,
                input_language_to_index,
                output_language_to_index,
                START_TOKEN,
                END_TOKEN,
                PADDING_TOKEN
                ):
        super().__init__()
        self.create_masks = Masks(max_sequence_length)
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, input_language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, output_language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, out_lang_vocab_size)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self,
                x,
                y,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_masks(x, y, enc_start_token, enc_end_token, dec_start_token, dec_end_token)
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out

