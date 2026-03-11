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