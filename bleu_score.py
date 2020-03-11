from torchtext.data.metrics import bleu_score
from inference import translate_sentence_rnn, translate_sentence_convolutional

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50, method=None):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        
        if method == "convolutional":
            pred_trg, _ = translate_sentence_convolutional(src, src_field, trg_field, model, device, max_len)
        elif method == "aiayn":
            pred_trg, _ = translate_sentence_convolutional(src, src_field, trg_field, model, device, max_len, aiayn=True)
        else:
            pred_trg, _ = translate_sentence_rnn(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)