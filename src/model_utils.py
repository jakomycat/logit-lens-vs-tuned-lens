from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name='gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

    model.eval()

    return model, tokenizer