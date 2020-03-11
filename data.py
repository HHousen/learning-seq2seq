import spacy

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

def get_data(batch_size, device, reverse=False, packed_padding=False, get_datasets=False, batch_first=False):
    """
    Data getting function. 
    By default RNN models in PyTorch require the sequence to be a tensor of shape [sequence length, batch size].
    CNNs expect the batch dimension to be first. So batches should be [batch size, sequence length] which is 
    done by setting batch_first=True.
    """
    #python -m spacy download en
    #python -m spacy download de
    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings (tokens) and reverses it
        """
        tokens = [tok.text for tok in spacy_de.tokenizer(text)]
        if reverse:
            return tokens[::-1]
        else:
            return tokens

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    #`include_lengths`: When using packed padded sequences, we need to tell PyTorch how long the 
    # actual (non-padded) sequences are. Specifying `include_lengths` will cause  batch.src to 
    # be a tuple. The first element of the tuple is a batch of numericalized source sentence as a 
    # tensor and the second element is the non-padded lengths of each source sentence within the batch.
    SRC = Field(tokenize = tokenize_de, 
                init_token='<sos>', 
                eos_token='<eos>', 
                lower=True,
                batch_first=batch_first,
                include_lengths=packed_padding)

    TRG = Field(tokenize = tokenize_en, 
                init_token='<sos>', 
                eos_token='<eos>', 
                lower=True,
                batch_first=batch_first)

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    training_data_head = [vars(train_data.examples[i]) for i in range(5)]
    print(*training_data_head, sep='\n')

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)
    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

    data = (train_data, valid_data, test_data)

    # For packed padded sequences  all elements in the batch need to be sorted by their 
    # non-padded lengths in descending order.
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        data, 
        batch_size=batch_size,
        sort_within_batch=packed_padding,
        sort_key=lambda x : len(x.src),
        device=device)
    
    if get_datasets:
        return train_iterator, valid_iterator, test_iterator, SRC, TRG, data
    else:
        return train_iterator, valid_iterator, test_iterator, SRC, TRG