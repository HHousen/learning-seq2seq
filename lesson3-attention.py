# Paper: Neural Machine Translation by Jointly Learning to Align and Translate - https://arxiv.org/abs/1409.0473
# Original Tutorial: https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
# Description:
# Next, we learn about attention by implementing Neural Machine Translation by Jointly Learning to 
# Align and Translate. This further allievates the information compression problem by allowing the 
# decoder to "look back" at the input sentence by creating context vectors that are weighted sums 
# of the encoder hidden states. The weights for this weighted sum are calculated via an attention 
# mechanism, where the decoder learns to pay attention to the most relevant words in the input sentence.


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

from data import get_data
from helpers import count_parameters, epoch_time, set_seed
from evaluate import evaluate
from train import train


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, bidirectional):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=bidirectional)
        
        self.fc = nn.Linear(enc_hid_dim * 2 if bidirectional else enc_hid_dim, dec_hid_dim) # 2 because bidirectional
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
                
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        combined_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = torch.tanh(self.fc(combined_hidden))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, bidirectional):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2 if bidirectional else enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        #rearrange encoder outputs 
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        #calculates how well each encoder hidden state "matches" the previous decoder hidden state
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]
        #v can be thought of as the weights for a weighted sum of the energy across all encoder hidden states
        #v is not dependent on time, and the same v is used for each time-step of the decoding
        #removes `dec hid dim` because attention should be over the length of the source sentence not the 
        #length of the source sentence with the hidden dimensions
        #also reduces output to single number
        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, bidirectional):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2 if bidirectional else enc_hid_dim) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2 if bidirectional else enc_hid_dim) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        #takes the previous hidden state, all of the encoder hidden states, and returns the attention vector
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        #uses attention vector to create a weighted source vector, which is a weighted sum of the encoder
        #hidden states, using the attention vector as the weights
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)

        #weighted = [1, batch size, enc hid dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs (used to create attention vector and weight the source vector in decoder)
        #is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
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

train_iterator, valid_iterator, test_iterator, SRC, TRG = get_data(BATCH_SIZE, device)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BIDIRECTIONAL = True

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, BIDIRECTIONAL)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT, BIDIRECTIONAL)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn, BIDIRECTIONAL)

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
        torch.save(model.state_dict(), 'tut3-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model.load_state_dict(torch.load('tut3-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# Training & Testing Results:
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [02:07<00:00,  1.78it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00,  9.71it/s]
# Epoch: 01 | Time: 2m 8s
#         Train Loss: 5.025 | Train PPL: 152.211
#          Val. Loss: 4.934 |  Val. PPL: 138.971
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [02:11<00:00,  1.73it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00,  9.06it/s]
# Epoch: 02 | Time: 2m 12s
#         Train Loss: 4.166 | Train PPL:  64.459
#          Val. Loss: 4.437 |  Val. PPL:  84.508
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [02:09<00:00,  1.75it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 10.23it/s]
# Epoch: 03 | Time: 2m 10s
#         Train Loss: 3.516 | Train PPL:  33.649
#          Val. Loss: 3.842 |  Val. PPL:  46.606
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [02:15<00:00,  1.67it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00,  8.62it/s]
# Epoch: 04 | Time: 2m 16s
#         Train Loss: 2.948 | Train PPL:  19.067
#          Val. Loss: 3.416 |  Val. PPL:  30.459
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [02:19<00:00,  1.63it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00,  8.58it/s]
# Epoch: 05 | Time: 2m 20s
#         Train Loss: 2.572 | Train PPL:  13.087
#          Val. Loss: 3.285 |  Val. PPL:  26.696
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [02:17<00:00,  1.66it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 10.31it/s]
# Epoch: 06 | Time: 2m 17s
#         Train Loss: 2.242 | Train PPL:   9.415
#          Val. Loss: 3.263 |  Val. PPL:  26.138
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [02:08<00:00,  1.77it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00,  9.58it/s]
# Epoch: 07 | Time: 2m 9s
#         Train Loss: 1.993 | Train PPL:   7.336
#          Val. Loss: 3.237 |  Val. PPL:  25.446
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [02:08<00:00,  1.76it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00,  9.97it/s]
# Epoch: 08 | Time: 2m 9s
#         Train Loss: 1.775 | Train PPL:   5.898
#          Val. Loss: 3.222 |  Val. PPL:  25.073
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [02:05<00:00,  1.81it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 10.14it/s]
# Epoch: 09 | Time: 2m 6s
#         Train Loss: 1.627 | Train PPL:   5.091
#          Val. Loss: 3.281 |  Val. PPL:  26.601
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 227/227 [02:06<00:00,  1.80it/s]
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 10.35it/s]
# Epoch: 10 | Time: 2m 7s
#         Train Loss: 1.498 | Train PPL:   4.472
#          Val. Loss: 3.334 |  Val. PPL:  28.059
# Validate: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 11.01it/s]
# | Test Loss: 3.235 | Test PPL:  25.406 |