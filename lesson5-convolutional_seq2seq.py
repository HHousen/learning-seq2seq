# Paper: Convolutional Sequence to Sequence Learning - https://arxiv.org/abs/1705.03122
# Original Tutorial: https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb
# Description:
# We finally move away from RNN based models and implement a fully convolutional model. One of 
# the downsides of RNNs is that they are sequential. That is, before a word is processed by the 
# RNN, all previous words must also be processed. Convolutional models can be fully parallelized, 
# which allow them to be trained much quicker. We will be implementing the Convolutional Sequence 
# to Sequence model, which uses multiple convolutional layers in both the encoder and decoder, 
# with an attention mechanism between them.


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
from inference import translate_sentence_convolutional as translate_sentence, display_attention
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

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device, max_length=100):
        super().__init__()
        # Without padding, the length of the sequence coming out of a convolutional layer will be 
        # `filter_size - 1` shorter than the sequence entering the convolutional layer. Thus, we pad 
        # the sentence with one padding element on each side. We can calculate the amount of padding 
        # on each side by simply doing (filter_size - 1)/2 for odd sized filters.
        # Odd kernels allow padding to be added equally to both sides of the source sequence.
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.device = device
        
        # The `scale` variable is used by the authors to "ensure that the variance throughout the 
        # network does not change dramatically". The performance of the model seems to vary wildly 
        # using different seeds if this is not used.
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        # The positional embedding is initialized to have a "vocabulary" of `max_length`. This means 
        # it can handle sequences up to `max_length` elements long, indexed from 0 to `max_length-1`. 
        # This can be increased if used on a dataset with longer sequences.
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        # `out_channels` is multipled by 2 because GLU activation function has gating mechanism that 
        # halves the size of the hidden dimension. Thus, it is multipled by 2 so the hidden dimension 
        # size for each token is the same as it was when it entered the convolutional blocks.
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size, 
                                              padding = (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        #src = [batch size, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        #create position tensor
        # No recurrent connections in this model so it has no idea about the order of the tokens 
        # within a sequence. To rectify this we have a second embedding layer, the positional 
        # embedding layer. This is a standard embedding layer where the input is not the token 
        # itself but the position of the token within the sequence - starting with the first token, 
        # the <sos> (start of sequence) token, in position 0.
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [0, 1, 2, 3, ..., src len - 1]
        #pos = [batch size, src len]
        #embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        
        #tok_embedded = pos_embedded = [batch size, src len, emb dim]
        #combine embeddings by elementwise summing, contains information about the token and also its 
        #  position with in the sequence
        embedded = self.dropout(tok_embedded + pos_embedded)
        
        #embedded = [batch size, src len, emb dim]
        #pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)
        
        #conv_input = [batch size, src len, hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1) 
        
        #conv_input = [batch size, hid dim, src len]
        #begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            #conved = [batch size, 2 * hid dim, src len]
            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)

            #conved = [batch size, hid dim, src len]
            #apply residual connection
            conved = (conved + conv_input) * self.scale

            #conved = [batch size, hid dim, src len]
            #set conv_input to conved for next loop iteration
            conv_input = conved
        
        #...end convolutional blocks

        # Conved vector is the result of each token being passed through the convolutional blocks
        #permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))
        
        #conved = [batch size, src len, emb dim]
        #elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale
        
        # Two context vectors for each token in the input sentence: a conved and a combined vector
        #combined = [batch size, src len, emb dim]
        return conved, combined

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, trg_pad_idx, device, max_length=100):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        
        self.fc_out = nn.Linear(emb_dim, output_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
      
    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        #embedded = [batch size, trg len, emb dim]
        #conved = [batch size, hid dim, trg len]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]
        
        #permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        
        # Embedding summed with conved, which has been converted to emb dim, via a residual connection
        #conved_emb = [batch size, trg len, emb dim]
        combined = (conved_emb + embedded) * self.scale
        
        # Apply standard attention calculation to combined `conved_emb` and `embedded` by finding how 
        # much the combination "matches" with the encoded conved
        #combined = [batch size, trg len, emb dim] 
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        
        #energy = [batch size, trg len, src len]
        attention = F.softmax(energy, dim=2)
        
        # Apply attention by getting a weighted sum over the encoded combined
        #attention = [batch size, trg len, src len]
        attended_encoding = torch.matmul(attention, encoder_combined)

        # Why do they calculate attention first with the encoded conved and then use it to calculate 
        # the weighted sum over the encoded combined? The paper argues that the encoded conved is good 
        # for getting a larger context over the encoded sequence, whereas the encoded combined has more 
        # information about the specific token and is thus therefore more useful for making a prediction.
        
        #attended_encoding = [batch size, trg len, emd dim]
        #convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)
        
        #attended_encoding = [batch size, trg len, hid dim]
        #apply residual connection to the initial input to the attention layer
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        
        #attended_combined = [batch size, hid dim, trg len]
        return attention, attended_combined
        
    def forward(self, trg, encoder_conved, encoder_combined):
        #trg = [batch size, trg len]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim] 
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
            
        #create position tensor
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, trg len]
        #embed tokens and positions
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        
        #tok_embedded = [batch size, trg len, emb dim]
        #pos_embedded = [batch size, trg len, emb dim]
        #combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        
        #embedded = [batch size, trg len, emb dim]
        #pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)
        
        #conv_input = [batch size, trg len, hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1) 
        
        #conv_input = [batch size, hid dim, trg len]
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        
        for i, conv in enumerate(self.convs):
            #apply dropout
            conv_input = self.dropout(conv_input)

            # As we are processing all of the targets simultaneously in parallel, and not sequentially, 
            # we need a method of only allowing the filters translating token i to only look at tokens 
            # before word i. If they were allowed to look at token i+1 (the token they should be 
            # outputting), the model will simply learn to output the next word in the sequence by 
            # directly copying it, without actually learning how to translate.
            #need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size, 
                                  hid_dim, 
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
                
            padded_conv_input = torch.cat((padding, conv_input), dim = 2)
        
            #padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]
            #pass through convolutional layer
            conved = conv(padded_conv_input)

            #conved = [batch size, 2 * hid dim, trg len]
            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)

            #conved = [batch size, hid dim, trg len]
            #calculate attention
            attention, conved = self.calculate_attention(embedded, conved, encoder_conved, encoder_combined)
            
            #attention = [batch size, trg len, src len]
            #apply residual connection
            conved = (conved + conv_input) * self.scale
            
            #conved = [batch size, hid dim, trg len]
            #set conv_input to conved for next loop iteration
            conv_input = conved
            
        conved = self.hid2emb(conved.permute(0, 2, 1))
         
        #conved = [batch size, trg len, emb dim]
        output = self.fc_out(self.dropout(conved))
        
        #output = [batch size, trg len, output dim]
        return output, attention

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        # As the decoding is done in parallel we do not need a decoding loop. All of the 
        # target sequence is input into the decoder at once and the padding is used to 
        # ensure each convolutional filter in the decoder can only see the current and 
        # previous tokens in the sequence as it slides across the sentence.

        #src = [batch size, src len]
        #trg = [batch size, trg len - 1] (<eos> token sliced off the end)
           
        #calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        #encoder_conved is output from final encoder conv block
        #encoder_combined is encoder_conved plus (elementwise) src embedding plus positional embeddings 
        encoder_conved, encoder_combined = self.encoder(src)
            
        #encoder_conved = [batch size, src len, emb dim]
        #encoder_combined = [batch size, src len, emb dim]
        
        #calculate predictions of next words
        #output is a batch of predictions for each word in the trg sentence
        #attention a batch of attention scores across the src sentence for 
        #  each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)
        
        #output = [batch size, trg len - 1, output dim]
        #attention = [batch size, trg len - 1, src len]
        return output, attention

set_seed(42)

BATCH_SIZE = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator, SRC, TRG, data = get_data(BATCH_SIZE, device, get_datasets=True, batch_first=True)
train_data, valid_data, test_data = data

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
EMB_DIM = 256
HID_DIM = 512 # each conv. layer has 2 * hid_dim filters
ENC_LAYERS = 10 # number of conv. blocks in encoder
DEC_LAYERS = 10 # number of conv. blocks in decoder
ENC_KERNEL_SIZE = 3 # must be odd!
DEC_KERNEL_SIZE = 3 # can be even or odd
ENC_DROPOUT = 0.25
DEC_DROPOUT = 0.25
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    
enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)

model = Seq2Seq(enc, dec).to(device)

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

if ARGS.train:
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    N_EPOCHS = 10
    CLIP = 0.1
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut5-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

if not ARGS.train:
    model.load_state_dict(torch.load('tut5-model.pt'))

if ARGS.test:
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

if ARGS.inference:
    def inference(idx, data, attention_file_name):
        src = vars(data.examples[idx])['src']
        trg = vars(data.examples[idx])['trg']

        print(f'src = {src}')
        print(f'trg = {trg}')

        translation, attention = translate_sentence(src, SRC, TRG, model, device)
        print(f'predicted trg = {translation}')
        display_attention(src, translation, attention, attention_file_name, dim_to_squeeze=0)

    inference(2, train_data, "train_attention.png")
    inference(2, valid_data, "valid_attention.png")
    inference(9, test_data, "test_attention.png")

if ARGS.bleu:
    bleu_score = calculate_bleu(test_data, SRC, TRG, model, device, method="convolutional")
    print(f'BLEU Score = {bleu_score*100:.2f}')

# Training & Testing Results:
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:33<00:00,  3.10it/s]
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 16.37it/s]
# Epoch: 01 | Time: 1m 34s
#         Train Loss: 4.861 | Train PPL: 129.152
#          Val. Loss: 3.619 |  Val. PPL:  37.303
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:33<00:00,  3.09it/s]
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 15.02it/s]
# Epoch: 02 | Time: 1m 34s
#         Train Loss: 3.402 | Train PPL:  30.027
#          Val. Loss: 2.595 |  Val. PPL:  13.394
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:34<00:00,  3.08it/s]
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 15.70it/s]
# Epoch: 03 | Time: 1m 34s
#         Train Loss: 2.757 | Train PPL:  15.750
#          Val. Loss: 2.292 |  Val. PPL:   9.898
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:35<00:00,  3.05it/s]
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 14.94it/s]
# Epoch: 04 | Time: 1m 35s
#         Train Loss: 2.470 | Train PPL:  11.821
#          Val. Loss: 2.124 |  Val. PPL:   8.368
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:43<00:00,  2.80it/s]
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 14.71it/s]
# Epoch: 05 | Time: 1m 44s
#         Train Loss: 2.293 | Train PPL:   9.901
#          Val. Loss: 2.022 |  Val. PPL:   7.556
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:43<00:00,  2.80it/s]
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 13.99it/s]
# Epoch: 06 | Time: 1m 44s
#         Train Loss: 2.168 | Train PPL:   8.741
#          Val. Loss: 1.984 |  Val. PPL:   7.270
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:47<00:00,  2.70it/s]
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 13.56it/s]
# Epoch: 07 | Time: 1m 48s
#         Train Loss: 2.078 | Train PPL:   7.992
#          Val. Loss: 1.932 |  Val. PPL:   6.904
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:50<00:00,  2.62it/s]
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 14.60it/s]
# Epoch: 08 | Time: 1m 51s
#         Train Loss: 2.006 | Train PPL:   7.435
#          Val. Loss: 1.898 |  Val. PPL:   6.670
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:46<00:00,  2.73it/s]
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 12.91it/s]
# Epoch: 09 | Time: 1m 46s
#         Train Loss: 1.951 | Train PPL:   7.035
#          Val. Loss: 1.883 |  Val. PPL:   6.573
# Train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [01:43<00:00,  2.80it/s]
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 13.82it/s]
# Epoch: 10 | Time: 1m 44s
#         Train Loss: 1.902 | Train PPL:   6.700
#          Val. Loss: 1.852 |  Val. PPL:   6.373
# Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 14.47it/s]
# | Test Loss: 1.823 | Test PPL:   6.191 |
# src = ['ein', 'kleines', 'mädchen', 'klettert', 'in', 'ein', 'spielhaus', 'aus', 'holz', '.']
# trg = ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']
# predicted trg = ['a', 'little', 'girl', 'climbing', 'in', 'a', 'playhouse', 'of', 'wood', '.', '<eos>']
# src = ['ein', 'junge', 'mit', 'kopfhörern', 'sitzt', 'auf', 'den', 'schultern', 'einer', 'frau', '.']
# trg = ['a', 'boy', 'wearing', 'headphones', 'sits', 'on', 'a', 'woman', "'s", 'shoulders', '.']
# predicted trg = ['a', 'young', 'boy', 'in', 'headphones', 'sits', 'on', 'the', 'shoulders', 'of', 'a', 'woman', '.', '<eos>']
# src = ['ein', 'mann', 'in', 'einer', 'weste', 'sitzt', 'auf', 'einem', 'stuhl', 'und', 'hält', 'magazine', '.']
# trg = ['a', 'man', 'in', 'a', 'vest', 'is', 'sitting', 'in', 'a', 'chair', 'and', 'holding', 'magazines', '.']
# predicted trg = ['a', 'man', 'in', 'a', 'vest', 'sits', 'on', 'a', 'chair', 'holding', '<unk>', '.', '<eos>']
# BLEU Score = 34.51