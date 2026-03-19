import torch
import torch.nn.functional as F

# Get probability that the word
def get_word_probability(layers_predictions, word):
    probs = []

    for i in range(1, 13):
        # Iterate in possible words
        for word_l, prob in layers_predictions[i]:
            if word_l == f' {word}': probs.append(prob)
            
        if len(probs) != i: probs.append(0.0) # If the word don't exist in the preds, prob = 0
        
    return probs

#
def calculate_kl_divergence(logits_layer, final_logits):
    log_probs_lens = F.log_softmax(logits_layer, dim=-1) # kl_div requires this in log-probabilities
    probs_final = F.softmax(final_logits, dim=-1)

    kl_div = F.kl_div(log_probs_lens, probs_final, reduction='batchmean')

    return kl_div

def calculate_reciprocal_rank(layer_logits, target_token_id):
    sorted_idx = torch.argsort(layer_logits, descending=True)
    
    rank_tensor = (sorted_idx == target_token_id).nonzero(as_tuple=True)[0]
    rank_position = rank_tensor.item() + 1
    
    reciprocal_rank = 1.0 / rank_position
    
    return reciprocal_rank, rank_position