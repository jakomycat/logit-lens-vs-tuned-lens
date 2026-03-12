import argparse
import torch
from torch import nn

from src.model_utils import load_model_and_tokenizer
from src.tuned_lens import SingleLayerTunedLens, run_tuned_lens

def main():
    parser = argparse.ArgumentParser()

    # Different arguments
    parser.add_argument(
        '--prompt',
        type=str,
        default='The color of the sky is'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=3
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/all_tuned_lenses.pt'
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')

    model, tokenizer = load_model_and_tokenizer('gpt2')
    model.to(device)
    model.eval()

    hidden_size = model.config.n_embd
    num_layers = model.config.n_layer
    
    # Structure from lenses
    lenses = nn.ModuleDict({
        str(layer_idx): SingleLayerTunedLens(hidden_size)
        for layer_idx in range(1, num_layers)
    })
    
    # Load weights saved
    lenses.load_state_dict(torch.load(args.checkpoint, map_location=device))
    lenses.to(device)

    lenses.eval()

    result = run_tuned_lens(model, tokenizer, lenses, text=args.prompt, top_k=args.top_k, device=device)

    for layer in sorted(result.keys(), key=int):
        predict = result[layer]
        print(f'\nLayer {layer}:')
        for token, prob in predict:
            print(f'{token!r} : {prob:.2%}')

if __name__ == '__main__':
    main()