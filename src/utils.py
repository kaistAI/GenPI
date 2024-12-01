from pathlib import Path
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import argparse
import transformers
from typing import Dict

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def load_merged_model(model_name, peft_model_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # peft_model = _load_peft_model(base_model, peft_model_path)
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

    merged_model = peft_model.merge_and_unload()

    # tokenizer.pad_token = tokenizer.eos_token
    return merged_model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--peft_path')
    parser.add_argument('--save_path')
    args    = parser.parse_args()

    # base_path = 'google/gemma-2b'
    base_path = "meta-llama/Meta-Llama-3-8B-Instruct"

    print('merge start...')
    merged_model, tokenizer = load_merged_model(base_path, args.peft_path)
    merged_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print('merge finish')