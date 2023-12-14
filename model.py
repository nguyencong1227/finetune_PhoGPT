import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def prepare_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce the number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    return model


def create_peft_model(model):
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return get_peft_model(model, config)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print('trainable params: {} || all params: {} || trainable%: {}'.format(trainable_params, all_param, 100 * trainable_params / all_param))