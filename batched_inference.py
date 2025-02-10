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
from tqdm import tqdm
import numpy as np
import pandas as pd
from trl import SFTTrainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import time
def format_inference_prompt(row):
    return f"""You are a helpful assistant. Take a deep breath. Read the following question carefully.
Question: {row['question']}
Options:
A) {row['opa']}
B) {row['opb']}
C) {row['opc']}
D) {row['opd']}
First, think about the reason for each option. Then, choose the single best option from the given 4 options only and provide the response as 'A', 'B', 'C', or 'D'. Do not choose any other option.
### Response:\n """

def load_model_and_tokenizer(base_model_name, peft_model_path, device_id):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{device_id}",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(
        model,
        peft_model_path,
        torch_dtype=torch.bfloat16
    )
    
    return model, tokenizer

def generate_responses_batch(model, tokenizer, prompts, max_new_tokens=256):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses

def main():
    WORLD_SIZE = torch.cuda.device_count()
    BATCH_SIZE = 32
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    
    test_df = pd.read_parquet('/projects/florence_echo/ankush_agent_projects/SFT/test-00000-of-00001_new.parquet')
    
    # Shard the dataset
    shard_size = len(test_df) // WORLD_SIZE
    start_idx = LOCAL_RANK * shard_size
    end_idx = start_idx + shard_size if LOCAL_RANK < WORLD_SIZE - 1 else len(test_df)
    process_df = test_df.iloc[start_idx:end_idx]
    #start time
    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "/projects/florence_echo/ankush_agent_projects/SFT/medical_qa_model/checkpoint-1995",
        LOCAL_RANK
    )
    
    results = []
    for i in range(0, len(process_df), BATCH_SIZE):
        batch_df = process_df.iloc[i:i+BATCH_SIZE]
        prompts = [format_inference_prompt(row) for _, row in batch_df.iterrows()]
        responses = generate_responses_batch(model, tokenizer, prompts)
        
        for j, (_, row) in enumerate(batch_df.iterrows()):
            results.append({
                'true_answer': chr(65 + int(row['cop'])),
                'model_response': responses[j]
            })
    
    # Save results for each GPU
    pd.DataFrame(results).to_csv(f'inference_results_rank{LOCAL_RANK}.csv', index=False)
    #end time
    end_time = time.time()
    print("Time taken for batched inference: ", end_time-start_time)
if __name__ == "__main__":
    main()

# Time taken for batched inference:  227.70902180671692
# Time taken for batched inference:  229.11245107650757
# Time taken for batched inference:  231.38528084754944
# Time taken for batched inference:  238.30364203453064