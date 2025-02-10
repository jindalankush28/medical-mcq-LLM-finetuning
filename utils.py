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
import random
import numpy as np
import pandas as pd
from trl import SFTTrainer
import torch

def create_bnb_config():
    """Create BitsAndBytes configuration for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def create_peft_config(LORA_R, LORA_ALPHA, LORA_DROPOUT):
    """Create PEFT configuration with LoRA."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj"
        ]
    )

def prepare_dataset(df: pd.DataFrame) -> Dataset:
    """Prepare the dataset for training."""
    # Convert DataFrame to dictionary format
    data_dict = df.to_dict('records')
    dataset = Dataset.from_list(data_dict)
    
    def format_prompt(example):
        """Format the input prompt for the model."""
        try:
            # Ensure cop is an integer
            cop = int(example['cop'])
            
            # Get the correct option text
            correct_option = {
                0: example['opa'],
                1: example['opb'],
                2: example['opc'],
                3: example['opd']
            }.get(cop, '')
            
            # Get the letter corresponding to the correct option
            letter = chr(65 + cop)  # A=1, B=2, etc.
            
            formatted_prompt = f"""You are a helpful assistant. Take a deep breath. Read the following question carefully.
Question: {example['question']}
Options:
A) {example['opa']}
B) {example['opb']}
C) {example['opc']}
D) {example['opd']}
First, think about the reason for each option. Then, choose the single best option from the given 4 options only and provide the response as 'A', 'B', 'C', or 'D'. Do not choose any other option.
### Response:\n 
Explanation: {example['exp']}
Hence, the correct answer is {letter} ({correct_option})."""
            
            # Debugging: Print the formatted prompt
            # print(f"Formatted prompt: {formatted_prompt}")
            
            return formatted_prompt
        

        except Exception as e:
            print(f"Error processing example: {e}")
            print(f"Example data: {example}")
            raise
    
    # Process the dataset
    processed_dataset = dataset.map(
        lambda x: {"text": format_prompt(x)},
        remove_columns=dataset.column_names,
        desc="Formatting prompts"
    )
    
    return processed_dataset

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)