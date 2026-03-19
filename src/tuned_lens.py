import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleLayerTunedLens(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.transform = nn.Linear(hidden_size, hidden_size)

        nn.init.eye_(self.transform.weight) # Identity matrix
        nn.init.zeros_(self.transform.bias)

    def forward(self, h_l, ln_f, W_U):
        h_l_translated = self.transform(h_l)
        logits = ln_f(h_l_translated) @ W_U.T

        return logits
    
# Function to compute KL divergence
def kl_loss(lens_logits, final_logits):
    log_probs_lens = F.log_softmax(lens_logits, dim=-1) # kl_div requires this in log-probabilities
    probs_final = F.softmax(final_logits, dim=-1)

    loss = F.kl_div(log_probs_lens, probs_final, reduction='batchmean')

    return loss

# Function to run Tuned Lens
def run_tuned_lens(model, tokenizer, lenses, text, top_k=3, device='cpu'):
    inputs = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states
    num_layers = model.config.n_layer

    ln_f = model.transformer.ln_f
    W_U = model.lm_head.weight

    layer_prediction = {}

    for layer_idx in range(1, num_layers):
        h_l = hidden_states[layer_idx]

        # Get last token
        h_l_last_token = h_l[0, -1, :]

        # Get actual lens
        tuned_lens_l = lenses[str(layer_idx)]

        # Calculate projection
        logits = tuned_lens_l(h_l_last_token, ln_f, W_U)
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_idx = torch.topk(probs, top_k, dim=-1) 
        
        # Get top tokens
        top_tokens = [tokenizer.decode(idx.item()) for idx in top_idx]
        layer_prediction[layer_idx] = list(zip(top_tokens, top_probs.tolist()))

    # Add last layer
    with torch.no_grad():
        h_final_last_token = hidden_states[-1][0, -1, :]

        logits_finales = h_final_last_token @ W_U.T
        probs_finales = torch.softmax(logits_finales, dim=-1)

        top_probs_fin, top_indices_fin = torch.topk(probs_finales, top_k, dim=-1)

        top_tokens_fin = [tokenizer.decode(idx.item()) for idx in top_indices_fin]
        layer_prediction[num_layers] = list(zip(top_tokens_fin, top_probs_fin.tolist()))
        
    return layer_prediction

def get_all_tuned_lens_logits(model, tokenizer, lenses, text, device='cpu'):
    inputs = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states
    num_layers = model.config.n_layer

    ln_f = model.transformer.ln_f
    W_U = model.lm_head.weight

    layer_logits_dict = {}

    for layer_idx in range(1, num_layers):
        h_l = hidden_states[layer_idx]

        h_l_last_token = h_l[0, -1, :] # Last token
        tuned_lens_l = lenses[str(layer_idx)] # Actual lens
        
        # Calculate logits
        logits = tuned_lens_l(h_l_last_token, ln_f, W_U)
        
        layer_logits_dict[layer_idx] = logits

    # Get last layer
    final_logits = outputs.logits[0, -1, :]
    layer_logits_dict[num_layers] = final_logits

    return layer_logits_dict