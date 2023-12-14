from transformers import AutoTokenizer

def tokenize_data(data, tokenizer):
    data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)
    return data
