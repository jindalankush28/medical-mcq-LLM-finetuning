from huggingface_hub import login
import os
import torch
import transformers
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
import random
import numpy as np
import pandas as pd
from trl import SFTTrainer, SFTConfig
from utils import create_bnb_config, create_peft_config, prepare_dataset, seed_everything

seed_everything(0)

train_df = pd.read_parquet('/projects/florence_echo/ankush_agent_projects/SFT/train-00000-of-00001_new.parquet')
val_df = pd.read_parquet('/projects/florence_echo/ankush_agent_projects/SFT/validation-00000-of-00001_new.parquet')
test_df = pd.read_parquet('/projects/florence_echo/ankush_agent_projects/SFT/test-00000-of-00001_new.parquet')

# Model and training configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "medical_qa_model"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
NUM_EPOCHS = 3

train_dataset = prepare_dataset(train_df)
val_dataset = prepare_dataset(val_df)
test_dataset = prepare_dataset(test_df)

def setup_model_and_tokenizer():
    """Setup the model and tokenizer with quantization and PEFT."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=create_bnb_config(),
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adaptor
    model = get_peft_model(model, create_peft_config(LORA_R=LORA_R, LORA_ALPHA=LORA_ALPHA, LORA_DROPOUT=LORA_DROPOUT))
    
    return model, tokenizer
from trl import DataCollatorForCompletionOnlyLM

def main():
    """Main training function."""

    # Initialize accelerator with DeepSpeed plugin
    accelerator = Accelerator()

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    # Setup training arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.3,
        learning_rate=LEARNING_RATE,
        bf16=True,
        fp16=False,
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,
        max_seq_length=512,
        dataloader_persistent_workers=True,
        remove_unused_columns=True,
        optim="paged_adamw_32bit",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        peft_config=create_peft_config(LORA_R=LORA_R, LORA_ALPHA=LORA_ALPHA, LORA_DROPOUT=LORA_DROPOUT),
        processing_class=tokenizer,
        data_collator=collator,  # Use default collator
    )
    
    # Prepare everything with accelerator
    trainer = accelerator.prepare(trainer)
    
    # Train the model
    trainer.train()
    
    # Save the model (only on main process)
    if accelerator.is_main_process:
        trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()