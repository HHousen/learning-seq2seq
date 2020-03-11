import torch
from tqdm import tqdm

def evaluate(model, iterator, criterion, packed_padding=False):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), total=len(iterator), desc="Test"):
            if packed_padding:
                src, src_len = batch.src
            else:
                src = batch.src
            
            trg = batch.trg
            
            if packed_padding:
                output = model(src, src_len, trg, 0) #turn off teacher forcing
            else:
                output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate_convolutional(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), total=len(iterator), desc="Test"):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:,:-1])
        
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)

            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)