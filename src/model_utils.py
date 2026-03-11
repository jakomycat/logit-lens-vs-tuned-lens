from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from datasets import load_dataset  # type: ignore

def load_model_and_tokenizer(model_name='gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

    model.eval()

    return model, tokenizer

# Wikitext-2 - This is for train traductors
def prepare_dataloader(tokenizer, batch_size, max_length):
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)