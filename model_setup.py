from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

def setup_trainer(model, train_dataset, tokenizer):
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=200,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir='outputs'
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    return trainer
