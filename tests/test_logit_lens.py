import argparse

from src.model_utils import load_model_and_tokenizer
from src.logit_lens import run_logit_lens

def main():
    # Instance argparse
    parser = argparse.ArgumentParser()

    # Prompt arg
    parser.add_argument(
        '--prompt',
        type=str,
        default='The color of the sky is'
    )

    # Number of top-k arg
    parser.add_argument(
        '--top_k',
        type=int,
        default=3
    )

    args = parser.parse_args()

    # Get model and tokenizer
    model, tokenizer = load_model_and_tokenizer('gpt2')

    result = run_logit_lens(model, tokenizer, text=args.prompt, top_k=args.top_k)

    for layer, predict in result.items():
        print(f'\nLayer {layer}:')

        for token, prob in predict:
            print(f'{token!r} : {prob:.2%}')

if __name__ == '__main__':
    main()