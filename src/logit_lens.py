import torch

def run_logit_lens(model, tokenizer, text, top_k=3):
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    # This is a tuple: (emb, lay_1, lay_2 ,..., lay_L)
    hidden_sates = outputs.hidden_states

    # Specific for GPT-2
    ln_f = model.transformer.ln_f # Final layer norm
    lm_head = model.lm_head # Unembedding matrix
    W_U = lm_head.weight

    layer_predictions = {}

    for layer_idx, h_l in enumerate(hidden_sates[1:], start=1):
        h_l_last_token = h_l[0, -1, :]

        # Apply norm
        if layer_idx < (len(hidden_sates) - 1):
            h_l_norm = ln_f(h_l_last_token)
        else: # No apply two times layer norm to the last layer
            h_l_norm = h_l_last_token

        # Get logits projecting vocabulary matrix and transform to probabilities
        logits = h_l_norm @ W_U.T
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_idx = torch.topk(probs, top_k) # Get top-k tokens

        top_tokens = [tokenizer.decode(idx.item()) for idx in top_idx] # Convert to text

        # Save results
        layer_predictions[layer_idx] = list(zip(top_tokens, top_probs.tolist()))

    return layer_predictions

def get_all_logit_lens_logits(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states

    # Specific for GPT-2
    ln_f = model.transformer.ln_f # Final layer norm
    lm_head = model.lm_head # Unembedding matrix

    layer_logits_dict = {}

    for layer_idx, h_l in enumerate(hidden_states[1:], start=1):
        h_l_last_token = h_l[0, -1, :]

        # Apply norm
        if layer_idx < (len(hidden_states) - 1):
            h_l_norm = ln_f(h_l_last_token)
        else: # No apply two times layer norm to the last layer
            h_l_norm = h_l_last_token

        logits = lm_head(h_l_norm) 
        
        # Save logits
        layer_logits_dict[layer_idx] = logits

    return layer_logits_dict