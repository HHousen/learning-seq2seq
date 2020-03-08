# Paper: Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation - https://arxiv.org/abs/1406.1078
# Original Tutorial: https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb
# Description:
# Now we have the basic workflow covered, this tutorial will focus on improving our results. 
# Building on our knowledge of PyTorch and TorchText gained from the previous tutorial, we'll cover 
# a second second model, which helps with the information compression problem faced by encoder-decoder 
# models. This model will be based off an implementation of Learning Phrase Representations using RNN 
# Encoder-Decoder for Statistical Machine Translation, which uses GRUs.


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
        nn.init.normal_(param.data, mean=0, std=0.01)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim) #no dropout as only one layer!
        
        self.rnn = nn.GRU(emb_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded) #no cell state!
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, context):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]
        
        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        emb_con = torch.cat((embedded, context), dim = 2)
            
        #emb_con = [1, batch size, emb dim + hid dim]
            
        output, hidden = self.rnn(emb_con, hidden)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), 
                           dim = 1)
        
        #output = [batch size, emb dim + hid dim * 2]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
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
        
        #last hidden state of the encoder is the context
        context = self.encoder(src)
        
        #context also used as the initial hidden state of the decoder
        hidden = context
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)
            
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

BATCH_SIZE = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator, SRC, TRG = get_data(BATCH_SIZE, device)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

model.apply(init_weights)
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


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
        torch.save(model.state_dict(), 'tut2-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut2-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# Training & Testing Results:
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:02<00:00,  4.64it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 23.68it/s]
# Epoch: 01 | Time: 1m 2s
#         Train Loss: 5.022 | Train PPL: 151.770
#          Val. Loss: 5.170 |  Val. PPL: 175.878
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:01<00:00,  4.69it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 23.49it/s]
# Epoch: 02 | Time: 1m 2s
#         Train Loss: 4.365 | Train PPL:  78.633
#          Val. Loss: 5.047 |  Val. PPL: 155.556
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:02<00:00,  4.63it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 23.40it/s]
# Epoch: 03 | Time: 1m 3s
#         Train Loss: 3.988 | Train PPL:  53.967
#          Val. Loss: 4.624 |  Val. PPL: 101.945
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:03<00:00,  4.56it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 22.34it/s]
# Epoch: 04 | Time: 1m 4s
#         Train Loss: 3.641 | Train PPL:  38.128
#          Val. Loss: 4.329 |  Val. PPL:  75.859
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:04<00:00,  4.49it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 22.48it/s]
# Epoch: 05 | Time: 1m 5s
#         Train Loss: 3.301 | Train PPL:  27.141
#          Val. Loss: 4.135 |  Val. PPL:  62.479
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:04<00:00,  4.47it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 21.73it/s]
# Epoch: 06 | Time: 1m 5s
#         Train Loss: 3.000 | Train PPL:  20.087
#          Val. Loss: 3.903 |  Val. PPL:  49.527
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:03<00:00,  4.53it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 22.42it/s]
# Epoch: 07 | Time: 1m 4s
#         Train Loss: 2.767 | Train PPL:  15.908
#          Val. Loss: 3.759 |  Val. PPL:  42.920
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:03<00:00,  4.54it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 25.45it/s]
# Epoch: 08 | Time: 1m 4s
#         Train Loss: 2.537 | Train PPL:  12.644
#          Val. Loss: 3.808 |  Val. PPL:  45.042
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:02<00:00,  4.62it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 23.62it/s]
# Epoch: 09 | Time: 1m 3s
#         Train Loss: 2.333 | Train PPL:  10.314
#          Val. Loss: 3.705 |  Val. PPL:  40.631
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:03<00:00,  4.55it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 23.58it/s]
# Epoch: 10 | Time: 1m 4s
#         Train Loss: 2.139 | Train PPL:   8.491
#          Val. Loss: 3.746 |  Val. PPL:  42.351
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 20.75it/s]
# | Test Loss: 3.581 | Test PPL:  35.916 |