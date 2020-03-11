# Paper: Attention Is All You Need - https://arxiv.org/abs/1706.03762
# Original Tutorial: https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
# Description:
# Continuing with the non-RNN based models, we implement the Transformer model from Attention 
# Is All You Need. This model is based soley on attention mechanisms and introduces Multi-Head 
# Attention. The encoder and decoder are made of multiple layers, with each layer consisting 
# of Multi-Head Attention and Positionwise Feedforward sublayers. This model is currently used 
# in many state-of-the-art sequence-to-sequence and transfer learning tasks.


# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

import spacy
import numpy as np

import random
import math
import time

import argparse

from data import get_data
from helpers import count_parameters, epoch_time, set_seed
from evaluate import evaluate_convolutional as evaluate
from train import train_convolutional as train
from inference import translate_sentence_convolutional as translate_sentence, display_attention_multiheaded
from bleu_score import calculate_bleu

PARSER = argparse.ArgumentParser(description='Lesson 4 - Padding, Masking, Inference')
PARSER.add_argument('--train', action='store_true',
                    help='Train')
PARSER.add_argument('--test', action='store_true',
                    help='Test')
PARSER.add_argument('--inference', action='store_true',
                    help='Inference')
PARSER.add_argument('--bleu', action='store_true',
                    help='BLEU Score')

ARGS = PARSER.parse_args()

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        # Scale supposedly reduces variance in the embeddings and the model is difficult to train 
        # reliably without this scaling factor
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        #src = [batch size, src len]
        #src_mask = [batch size, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Model has no recurrent it has no idea about the order of the tokens within the sequence. 
        # We solve this by using a second embedding layer called a positional embedding layer
        # The position embedding has a "vocabulary" size of `max_length`, which means our model 
        # can accept sentences up to `max_length` tokens long. This can be increased if we want 
        # to handle longer sentences. 
        # Original Transformer implementation from the Attention is All You Need paper does not 
        # learn positional embeddings and instead it uses a fixed static embedding. Modern 
        # Transformer architectures, like BERT, use positional embeddings. 
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim,  dropout, device):
        super().__init__()
        
        # LayerNorm: https://arxiv.org/abs/1607.06450 
        # It normalizes the values of the features, i.e. across the hidden dimension, so each feature 
        # has a mean of 0 and a standard deviation of 1. This allows neural networks with a larger 
        # number of layers, like the Transformer, to be trained easier.
        # More Details: https://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2/
        self.layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        # https://jalammar.github.io/illustrated-transformer/
        # https://www.mihaileric.com/posts/transformers-attention-in-disguise/
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        # Calculate the Query, Key, and Value matrices by packing the embeddings into a matrix X 
        # and multiplying it by the weight matrices we’ve trained (aka pass through linear layer)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
        # Split the hid_dim of the query, key and value into n_heads using .view and correctly 
        # permute them so they can be multiplied together.
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim] 
        # Calculate the energy (the un-normalized attention) by multiplying Q and K together (for each head)
        # and scaling it by the square root of head_dim, which is calculated as hid_dim // n_heads
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, seq len, seq len]
        # Mask the energy so we do not pay attention over any elements of the sequence we shouldn't
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
        # Apply the attention to the value heads, V
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, seq len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, seq len, n heads, head dim]
        # Combine the `n heads` together because feed-forward layer is not expecting `n heads` matrices
        # (removes `n heads` dimension)
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, seq len, hid dim]
        # Final linear layer aka weight matrix
        x = self.fc_o(x)
        
        #x = [batch size, seq len, hid dim]
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    # The purpose of this module is not explined in the paper
    # Each of the layers in our encoder and decoder contains a fully connected feed-forward network, 
    # which is applied to each position separately and identically. 
    # http://nlp.seas.harvard.edu/2018/04/01/attention.html#position-wise-feed-forward-networks
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
        # Combines positional embeddings with the scaled embedded target tokens, followed by dropout
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        #trg = [batch size, trg len, hid dim]
        # Combined embeddings are passed through the N decoder layers, along with the encoded 
        # source, enc_src, and the source and target masks.
        # As well as using the source mask, as we did in the encoder to prevent our model attending 
        # to <pad> tokens, we also use a target mask. Performs a similar operation as the decoder 
        # padding in the convolutional sequence-to-sequence model. As we are processing all of the 
        # target tokens at once in parallel we need a method of stopping the decoder from "cheating" 
        # by simply "looking" at what the next token in the target sequence is and outputting it. 
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg) # Softmax contained within loss function, so do not need to use a softmax layer here
        
        #output = [batch size, trg len, output dim]
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        # Similar to the encoder layer except that it there are two multi-head attention layers: 
        # self_attention and encoder_attention
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
        
        # Performs self-attention, as in the encoder, by using the decoder representation so far 
        # as the query, key and value.
        # This self_attention layer uses the target sequence mask, trg_mask, in order to prevent 
        # the decoder from "cheating" by paying attention to tokens that are "ahead" of the one it 
        # is currently processing as it processes all tokens in the target sentence in parallel.
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
        #encoder attention: how we actually feed the encoded source sentence, enc_src, into our decoder
        # The queries are the decoder representations and the keys and values are the encoder representations.
        # `src_mask` is used to prevent the multi-head attention layer from attending to <pad> tokens 
        # within the source sentence.
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # `src_mask` can view entire sentence since it is used with encoder
        # It is created by checking where the source sequence is not equal to a <pad> token
        #src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]
        return src_mask
    
    def make_trg_mask(self, trg):
        # Create initial mask to remove <pad> tokens
        #trg = [batch size, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        
        #trg_pad_mask = [batch size, 1, trg len, 1]
        trg_len = trg.shape[1]
        
        # Creates a diagonal matrix where the elements above the diagonal will be zero and the elements 
        # below the diagonal will be set to whatever the input tensor is (ones)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        # If first target token in the matrix has a mask of [1, 0, 0, 0, 0] it can only look at the 
        # first target token. If the second target token has a mask of [1, 1, 0, 0, 0] it means 
        # it can look at both the first and second target tokens. 
        
        # Logically and the "subsequent" mask and the padding mask
        #trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

    def forward(self, src, trg):
        #src = [batch size, src len]
        #trg = [batch size, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        return output, attention

set_seed(42)

BATCH_SIZE = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator, SRC, TRG, data = get_data(BATCH_SIZE, device, get_datasets=True, batch_first=True)
train_data, valid_data, test_data = data

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

if ARGS.train:
    model.apply(initialize_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    LEARNING_RATE = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut6-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

if not ARGS.train:
    model.load_state_dict(torch.load('tut6-model.pt'))

if ARGS.test:
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

if ARGS.inference:
    def inference(idx, data, attention_file_name):
        src = vars(data.examples[idx])['src']
        trg = vars(data.examples[idx])['trg']

        print(f'src = {src}')
        print(f'trg = {trg}')

        translation, attention = translate_sentence(src, SRC, TRG, model, device, aiayn=True)
        print(f'predicted trg = {translation}')
        display_attention_multiheaded(src, translation, attention, output_file=attention_file_name)

    inference(8, train_data, "train_attention.png")
    inference(6, valid_data, "valid_attention.png")
    inference(10, test_data, "test_attention.png")

if ARGS.bleu:
    bleu_score = calculate_bleu(test_data, SRC, TRG, model, device, method="aiayn")
    print(f'BLEU Score = {bleu_score*100:.2f}')

# Training & Testing Results: