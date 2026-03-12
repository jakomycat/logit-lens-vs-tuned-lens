import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from tqdm import tqdm # type: ignore
import os

from src.tuned_lens import SingleLayerTunedLens, kl_loss
from src.model_utils import prepare_dataloader

# Configuration
model_name = 'gpt2'
batch_size = 8
epochs = 2
learning_rate = 1e-3
max_length = 64
device = "cuda" if torch.cuda.is_available() else "cpu" 
print(f'Device: {device}')

# Get model: GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
model.eval() # Only evaluation

# Freeze all LLM's parameters
for param in model.parameters():
    param.requires_grad = False

# Get layer norm and unembedding matrix
ln_f = model.transformer.ln_f
W_U = model.lm_head.weight

#
hidden_size = model.config.n_embd
num_layers = model.config.n_layer

lenses = nn.ModuleDict({
    str(layer_idx): SingleLayerTunedLens(hidden_size).to(device)
    for layer_idx in range(1, num_layers)
})

optimizer = AdamW(lenses.parameters(), lr=learning_rate)
data_loader = prepare_dataloader(tokenizer, batch_size, max_length)

# Make carpet for save weigths
os.makedirs('checkpoints', exist_ok=True)

for epoch in range(epochs):
    total_loss_epoch = 0
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1} / {epochs}')

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Calculate outputs
        with torch.no_grad():
            ouputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = ouputs.hidden_states

        # Calculate logits of last layer
        with torch.no_grad():
            h_final = hidden_states[-1]
            logits_finals = model.lm_head(h_final)

        # Train lens
        optimizer.zero_grad()
        loss_batch_acumulate = 0

        for layer_idx in range(1, num_layers):
            h_l = hidden_states[layer_idx]

            logits_lens_i = lenses[str(layer_idx)](h_l, ln_f, W_U)

            loss_layer = kl_loss(logits_lens_i, logits_finals)
            loss_batch_acumulate += loss_layer

        # Backpropagation
        loss_batch_acumulate.backward()
        optimizer.step()

        #
        avg_loss_batch = loss_batch_acumulate.item() / (num_layers - 1)
        total_loss_epoch += avg_loss_batch

        progress_bar.set_postfix({'Avg_KL_Loss': f'{avg_loss_batch:.4f}'})

    avg_loss_epoch = total_loss_epoch / len(data_loader)
    print(f'Average KL loss global: {avg_loss_epoch:.4f}')

# Save weigth
save_path = 'checkpoints/all_tuned_lenses.pt'

torch.save(lenses.state_dict(), save_path)
print(f'Train complete. Sens\' weigths on {save_path}')