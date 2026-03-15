# Get probability that the word
def get_word_probability(layers_predictions, word):
    probs = []

    for i in range(1, 13):
        # Iterate in possible words
        for word_l, prob in layers_predictions[i]:
            if word_l == f' {word}': probs.append(prob)
            
        if len(probs) != i: probs.append(0.0) # If the word don't exist in the preds, prob = 0
        
    return probs