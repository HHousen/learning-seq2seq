# Paper: None
# Original Tutorial: https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb
# Description:
# In this notebook, we will improve the previous model architecture by adding packed padded 
# sequences and masking. These are two methods commonly used in NLP. Packed padded sequences 
# allow us to only process the non-padded elements of our input sentence with our RNN. Masking 
# is used to force the model to ignore certain elements we do not want it to look at, such as 
# attention over padded elements. Together, these give us a small performance boost. We also 
# cover a very basic way of using the model for inference, allowing us to get translations for 
# any sentence we want to give to the model and how we can view the attention values over the 
# source sequence for those translations. Finally, we show how to calculate the BLEU metric from 
# our translations.


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
from evaluate import evaluate
from train import train
from inference import translate_sentence_rnn as translate_sentence, display_attention
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

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        #src = [src len, batch size]
        #src_len = [batch size]
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        # For more info about pack_padded_sequence: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch/56211056#56211056
        # Essentially it is a way of removeing unnecessary calculations from the training process
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
                
        packed_outputs, hidden = self.rnn(packed_embedded)
                                 
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
            
        #outputs is now a non-packed sequence, all hidden states obtained
        #when the input is a pad token are all zeros
            
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #encoder RNNs fed through a linear layer
        combined_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = torch.tanh(self.fc(combined_hidden))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        # `mask` is is a [batch size, source sentence length] tensor that is 1 when the source 
        # sentence token is not a padding token, and 0 when it is a padding token
        # masked_fill fills the tensor at each element where the first argument (mask == 0) is true, 
        # with the value given by the second argument (-1e10)
        # In other words, this takes the un-normalized attention values, and changes the attention 
        # values over padded elements to be -1e10, this number is miniscule and will become zero
        # when passed through softmax
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        a = self.attention(hidden, encoder_outputs, mask)
                
        #a = [batch size, src len]
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
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
        # return the attention tensor because we want to view the values of attention during inference
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        """
        Uses the pad token index to create the mask, by creating a mask tensor that is 1 wherever the 
        source sentence is not equal to the pad token
        """
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        #src = [src len, batch size]
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        mask = self.create_mask(src)

        #mask = [batch size, src len]
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state, all encoder hidden states, 
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            
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

BATCH_SIZE = 90
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator, SRC, TRG, data = get_data(BATCH_SIZE, device, packed_padding=True, get_datasets=True)
train_data, valid_data, test_data = data

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

if ARGS.train:
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, packed_padding=True)
        valid_loss = evaluate(model, valid_iterator, criterion, packed_padding=True)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut4-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

if not ARGS.train:
    model.load_state_dict(torch.load('tut4-model.pt'))

if ARGS.test:
    test_loss = evaluate(model, test_iterator, criterion, packed_padding=True)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

if ARGS.inference:
    def inference(idx, data, attention_file_name):
        src = vars(data.examples[idx])['src']
        trg = vars(data.examples[idx])['trg']

        print(f'src = {src}')
        print(f'trg = {trg}')

        translation, attention = translate_sentence(src, SRC, TRG, model, device)
        print(f'predicted trg = {translation}')
        display_attention(src, translation, attention, attention_file_name)

    inference(12, train_data, "train_attention.png")
    inference(14, valid_data, "valid_attention.png")
    inference(18, test_data, "test_attention.png")

if ARGS.bleu:
    bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)
    print(f'BLEU Score = {bleu_score*100:.2f}')

# Training & Testing Results:
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 323/323 [01:21<00:00,  3.95it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 13.36it/s]
# Epoch: 01 | Time: 1m 22s
#         Train Loss: 4.972 | Train PPL: 144.327
#          Val. Loss: 4.704 |  Val. PPL: 110.386
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 323/323 [01:14<00:00,  4.32it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 13.98it/s]
# Epoch: 02 | Time: 1m 15s
#         Train Loss: 3.789 | Train PPL:  44.224
#          Val. Loss: 3.799 |  Val. PPL:  44.673
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 323/323 [01:14<00:00,  4.31it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 13.75it/s]
# Epoch: 03 | Time: 1m 15s
#         Train Loss: 2.999 | Train PPL:  20.058
#          Val. Loss: 3.443 |  Val. PPL:  31.269
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 323/323 [01:15<00:00,  4.28it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 13.44it/s]
# Epoch: 04 | Time: 1m 16s
#         Train Loss: 2.501 | Train PPL:  12.193
#          Val. Loss: 3.286 |  Val. PPL:  26.737
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 323/323 [01:16<00:00,  4.22it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 13.38it/s]
# Epoch: 05 | Time: 1m 17s
#         Train Loss: 2.172 | Train PPL:   8.780
#          Val. Loss: 3.242 |  Val. PPL:  25.586
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 323/323 [01:17<00:00,  4.19it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 13.35it/s]
# Epoch: 06 | Time: 1m 17s
#         Train Loss: 1.887 | Train PPL:   6.602
#          Val. Loss: 3.279 |  Val. PPL:  26.550
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 323/323 [01:17<00:00,  4.15it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 13.16it/s]
# Epoch: 07 | Time: 1m 18s
#         Train Loss: 1.692 | Train PPL:   5.429
#          Val. Loss: 3.359 |  Val. PPL:  28.758
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 323/323 [01:17<00:00,  4.16it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 13.35it/s]
# Epoch: 08 | Time: 1m 18s
#         Train Loss: 1.529 | Train PPL:   4.614
#          Val. Loss: 3.319 |  Val. PPL:  27.639
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 323/323 [01:17<00:00,  4.17it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 13.41it/s]
# Epoch: 09 | Time: 1m 18s
#         Train Loss: 1.409 | Train PPL:   4.090
#          Val. Loss: 3.374 |  Val. PPL:  29.200
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 323/323 [01:16<00:00,  4.20it/s]
# Validate: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 13.34it/s]
# Epoch: 10 | Time: 1m 17s
#         Train Loss: 1.284 | Train PPL:   3.610
#          Val. Loss: 3.489 |  Val. PPL:  32.760
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 12.94it/s]
# | Test Loss: 3.224 | Test PPL:  25.117 |
# src = ['ein', 'schwarzer', 'hund', 'und', 'ein', 'gefleckter', 'hund', 'kämpfen', '.']
# trg = ['a', 'black', 'dog', 'and', 'a', 'spotted', 'dog', 'are', 'fighting']
# predicted trg = ['a', 'black', 'dog', 'and', 'a', 'spotted', 'dog', 'fighting', 'fighting', '.', '<eos>']
# src = ['eine', 'frau', 'spielt', 'ein', 'lied', 'auf', 'ihrer', 'geige', '.']
# trg = ['a', 'female', 'playing', 'a', 'song', 'on', 'her', 'violin', '.']
# predicted trg = ['a', 'woman', 'plays', 'a', 'song', 'on', 'her', 'violin', '.', '<eos>']
# src = ['die', 'person', 'im', 'gestreiften', 'shirt', 'klettert', 'auf', 'einen', 'berg', '.']
# trg = ['the', 'person', 'in', 'the', 'striped', 'shirt', 'is', 'mountain', 'climbing', '.']
# predicted trg = ['the', 'person', 'in', 'a', 'striped', 'shirt', 'is', 'climbing', 'a', 'mountain', '.', '<eos>']
# BLEU Score = 30.83