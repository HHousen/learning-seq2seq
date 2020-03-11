import torch
import spacy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def translate_sentence_rnn(sentence, src_field, trg_field, model, device, max_len=50):
    #ensure our model is in evaluation mode
    model.eval()
    
    #tokenize the source sentence if it has not been tokenized (is a string)
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    #numericalize the source sentence
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    #convert to a tensor and add a batch dimension
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    #get the length of the source sentence and convert to a tensor
    src_len = torch.LongTensor([len(src_indexes)]).to(device)
    
    #feed the source sentence into the encoder
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    #create the mask for the source sentence
    mask = model.create_mask(src_tensor)

    #create a list to hold the output sentence, initialized with an <sos> token
    #called trg_indexes because indexes for TARGET language sentence
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    #create a tensor to hold the attention values
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    
    #while we have not hit a maximum length
    for i in range(max_len):
        #get the input tensor, which should be either <sos> or the last predicted token
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        #feed the input, all encoder outputs, hidden state and mask into the decoder
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        #store attention values
        attentions[i] = attention
        
        #get the predicted next token
        pred_token = output.argmax(1).item()
        
        #add prediction to current output sentence prediction
        trg_indexes.append(pred_token)

        #break if the prediction was an <eos> token
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    #convert the output sentence from indexes to tokens
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    #return the output sentence (with the <sos> token removed) and the attention values over the sequence
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]

def translate_sentence_convolutional(sentence, src_field, trg_field, model, device, max_len=50, aiayn=False):
    # TODO: Comment the aiayn commands
    model.eval()
    
    #tokenize the source sentence if it has not been tokenized (is a string)
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    #append the <sos> and <eos> tokens
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    
    #numericalize the source sentence
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    #convert it to a tensor and add a batch dimension
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    if aiayn:
        src_mask = model.make_src_mask(src_tensor)

    #feed the source sentence into the encoder
    with torch.no_grad():
        if aiayn:
            enc_src = model.encoder(src_tensor, src_mask)
        else:
            encoder_conved, encoder_combined = model.encoder(src_tensor)

    #create a list to hold the output sentence, initialized with an <sos> token
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    #while we have not hit a maximum length
    for i in range(max_len):
        #convert the current output sentence prediction into a tensor with a batch dimension
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        if aiayn:
            trg_mask = model.make_trg_mask(trg_tensor)

        #place the current output and the two encoder outputs into the decoder
        with torch.no_grad():
            if aiayn:
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            else:
                output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)
        
        #get next output token prediction from decoder
        pred_token = output.argmax(2)[:,-1].item()
        
        #add prediction to current output sentence prediction
        trg_indexes.append(pred_token)

        #break if the prediction was an <eos> token
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    #convert the output sentence from indexes to tokens
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    #return the output sentence (with the <sos> token removed) and the attention from the last layer
    return trg_tokens[1:], attention

def display_attention(sentence, translation, attention, output_file="attention.jpg", dim_to_squeeze=1):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    attention = attention.squeeze(dim_to_squeeze).cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='bone')
   
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                       rotation=45)
    ax.set_yticklabels(['']+translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(output_file)
    plt.close()

def display_attention_multiheaded(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2, output_file="attention.jpg"):
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(output_file)
    plt.close()