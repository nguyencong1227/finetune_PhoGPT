from model import prepare_model, print_trainable_parameters
from data_processing import load_and_process_data
from tokenization import tokenize_data
from transformers import AutoTokenizer
from model_setup import setup_trainer
from transformers import AutoTokenizer

def main():
    model = prepare_model("vinai/PhoGPT-7B5-Instruct")
    print_trainable_parameters(model)
    data = load_and_process_data()
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    processed_data = tokenize_data(data, tokenizer)
    print(processed_data['train']["prediction"][:5])

    trainer = setup_trainer(model, processed_data['train'], tokenizer)
    trainer.train()

    # Rest of your experiment code...

if __name__ == "__main__":
    main()
    



