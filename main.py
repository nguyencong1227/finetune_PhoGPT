from model import prepare_model, print_trainable_parameters
from data_processing import load_and_process_data
from tokenization import tokenize_data
from transformers import AutoTokenizer
from model_setup import setup_trainer
from transformers import AutoTokenizer
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, LongformerTokenizer
import logging

def main():
     # Configurations
    DEVICE_TYPE = "cuda" 
    #if torch.cuda.is_available() else "cpu"
    SHOW_SOURCES = True
    logging.info(f"Running on: {DEVICE_TYPE}")
    logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")
    
    # Model configurations
    model_id = "vinai/PhoGPT-7B5-Instruct"
    model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
    model = prepare_model(device_type=DEVICE_TYPE, model_id=model_id, model_basename=None)
    print_trainable_parameters(model)
    data = load_and_process_data()
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    processed_data = tokenize_data(data, tokenizer)
    print(processed_data['train']["prediction"][:5])

    trainer = setup_trainer(model, processed_data['train'], tokenizer)
    trainer.train()

    # Rest of your experiment code...

if __name__ == "__main__":
    main()
    



