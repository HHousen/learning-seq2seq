# Paper: Sequence to Sequence Learning with Neural Networks - https://arxiv.org/abs/1409.3215
# Original Tutorial: https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
# Description: 
# This first tutorial covers the workflow of a PyTorch with TorchText seq2seq project. 
# We'll cover the basics of seq2seq networks using encoder-decoder models, how to implement these 
# models in PyTorch, and how to use TorchText to do all of the heavy lifting with regards to text 
# processing. The model itself will be based off an implementation of Sequence to Sequence Learning 
# with Neural Networks, which uses multi-layer LSTMs.


# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import spacy
import numpy as np

import random
import math
import time

from data import get_data
from helpers import count_parameters, epoch_time, set_seed
from evaluate import evaluate
from train import train

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        #src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hidden dim * n directions]
        #hidden = [n layers * n directions, batch size, hidden dim]
        #cell = [n layers * n directions, batch size, hidden dim]
        #outputs are always from the top hidden layer
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs

set_seed(42)

BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator, SRC, TRG = get_data(BATCH_SIZE, device, reverse=True)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
        
model.apply(init_weights)
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS), desc="Epoch"):
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut1-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# Training & Testing Results:
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [01:23<00:00,  2.73it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.67it/s]
# Epoch: 01 | Time: 1m 23s████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.40it/s]
#         Train Loss: 5.148 | Train PPL: 172.148
#          Val. Loss: 4.949 |  Val. PPL: 141.055
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [01:25<00:00,  2.65it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.52it/s]
# Epoch: 02 | Time: 1m 26s████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.24it/s]
#         Train Loss: 4.750 | Train PPL: 115.576
#          Val. Loss: 4.794 |  Val. PPL: 120.768
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [01:25<00:00,  2.65it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.23it/s]
# Epoch: 03 | Time: 1m 26s████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 12.99it/s]
#         Train Loss: 4.371 | Train PPL:  79.130
#          Val. Loss: 4.598 |  Val. PPL:  99.318
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [01:26<00:00,  2.62it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.77it/s]
# Epoch: 04 | Time: 1m 27s████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.52it/s]
#         Train Loss: 4.145 | Train PPL:  63.122
#          Val. Loss: 4.500 |  Val. PPL:  89.973
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [01:25<00:00,  2.65it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.29it/s]
# Epoch: 05 | Time: 1m 26s████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.05it/s]
#         Train Loss: 4.030 | Train PPL:  56.233
#          Val. Loss: 4.448 |  Val. PPL:  85.428
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [01:24<00:00,  2.67it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 15.52it/s]
# Epoch: 06 | Time: 1m 25s███████████████████████████████████████████████████████████████████████████▉            | 7/8 [00:00<00:00, 17.34it/s]
#         Train Loss: 3.922 | Train PPL:  50.485
#          Val. Loss: 4.430 |  Val. PPL:  83.936
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [01:23<00:00,  2.71it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 14.52it/s]
# Epoch: 07 | Time: 1m 24s████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 14.23it/s]
#         Train Loss: 3.834 | Train PPL:  46.231
#          Val. Loss: 4.462 |  Val. PPL:  86.638
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [01:25<00:00,  2.65it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.94it/s]
# Epoch: 08 | Time: 1m 26s████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.70it/s]
#         Train Loss: 3.757 | Train PPL:  42.801
#          Val. Loss: 4.328 |  Val. PPL:  75.812
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [01:21<00:00,  2.80it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 14.45it/s]
# Epoch: 09 | Time: 1m 21s████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 14.16it/s]
#         Train Loss: 3.681 | Train PPL:  39.673
#          Val. Loss: 4.268 |  Val. PPL:  71.349
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [01:27<00:00,  2.61it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.97it/s]
# Epoch: 10 | Time: 1m 27s████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 13.74it/s]
#         Train Loss: 3.597 | Train PPL:  36.476
#          Val. Loss: 4.252 |  Val. PPL:  70.224
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 14.48it/s]
# | Test Loss: 4.296 | Test PPL:  73.438 |