import torch
from tqdm import tqdm

def train(model, iterator, optimizer, criterion, clip, packed_padding=False):
    model.train()
    epoch_loss = 0
    
    for i, batch in tqdm(enumerate(iterator), total=len(iterator), desc="Train"):
        if packed_padding:
            src, src_len = batch.src
        else:
            src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()

        if packed_padding:
            output = model(src, src_len, trg)
        else:
            output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def train_convolutional(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, batch in tqdm(enumerate(iterator), total=len(iterator), desc="Train"):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1]) # remove <eos> token from target
        
        #output = [batch size, trg len - 1, output dim] # -1 because <eos> removed
        #trg = [batch size, trg len] # no -1 because hopefully model will predict <eos>
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1) # remove <sos> token from target
        
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
