from datasets import load_dataset

def load_and_process_data():
    data = load_dataset("Abirate/english_quotes")

    def merge_columns(example):
        example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
        return example

    data['train'] = data['train'].map(merge_columns)
    return data
