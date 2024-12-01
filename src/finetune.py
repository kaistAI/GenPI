# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import re
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    # LlamaForCausalLM,
    # LlamaTokenizer,
    # LlamaForSequenceClassification
)
from datasets import load_dataset, Dataset, DatasetDict
# import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# from src.dataset_cls.spider.dataset import SpiderDataset
from src.dataset_cls.agentbench.dataset import AgentBenchDataset

from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import random
import gc

W_PROMPT_RANDOMIZE = False

class MixtureTrainer(Seq2SeqTrainer):
    stage1_loss = []
    stage2_loss = []
    total_loss = []

    def log(self, logs: Dict[str, float]) -> None:
        if len(self.total_loss) > 0:
            print(len(self.stage1_loss), len(self.stage2_loss), len(self.stage1_loss))
            logs['stage1_loss'] = torch.stack(self.stage1_loss).mean().item()
            logs['stage2_loss'] = torch.stack(self.stage2_loss).mean().item()
            logs['total_loss'] = torch.stack(self.total_loss).mean().item()

        super().log(logs)

        self.stage1_loss = []
        self.stage2_loss = []
        self.total_loss = []


    def compute_loss(self, model, inputs, return_outputs=False):
        stage2_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': inputs['labels']
        }
        stage2_rets = super().compute_loss(model, stage2_inputs, return_outputs)

        stage1_inputs = {
            'input_ids': inputs['stage1_input_ids'],
            'attention_mask': inputs['stage1_attention_mask'],
            'labels': inputs['stage1_labels']
        }
        stage1_rets = super().compute_loss(model, stage1_inputs, return_outputs)

        if return_outputs:
            stage1_loss = stage1_rets[0]
            stage1_output = stage1_rets[1]
            stage2_loss = stage2_rets[0]
            stage2_output = stage2_rets[1]
        else:
            stage1_loss = stage1_rets
            stage1_output = None
            stage2_loss = stage2_rets
            stage2_output = None

        loss = (1-self.args.stage2_ratio)*stage1_loss + self.args.stage2_ratio*stage2_loss

        self.stage1_loss.append(self._nested_gather(stage1_loss.detach()))
        self.stage2_loss.append(self._nested_gather(stage2_loss.detach()))
        self.total_loss.append(self._nested_gather(loss.detach()))
        
        return (loss, stage2_output) if return_outputs else loss

def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True
    

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    ####
    adapter_checkpoint_dir: str = field(default=None, metadata={"help": 'Adapter checkpoint dir you want to resume'})
    dpo: bool = field(default=False, metadata={"help": 'DPO or not'})
    dpo_beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    ref_adapter_checkpoint_dir: str = field(default='./output', metadata={"help": 'Reference Adapter checkpoint dir. It will use only dpo is True.'})
    overwrite_ref_log_probs_cache: bool = field(default=True, metadata={"help": 'Overwrite ref precomputed logps cache. It will use only dpo is True.'})
    context_id: int = field(default=-1, metadata={"help": "context id"})
    num_train_epochs: int = field(default=1)
    stage2_ratio: float = field(default=0.5, metadata={"help": "The ratio of stage2 loss in the total loss."})
    stage3_ratio: float = field(default=0.5, metadata={"help": "The ratio of stage3 loss in the total loss."})
    w_prompt_randomize: bool = field(default=False, metadata={"help": 'when training w_prompt mode, randomly remove the context'})
    ####
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def get_accelerate_model(args, checkpoint_dir):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        # load_in_4bit=args.bits == 4,
        # load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        token=args.token,
        attn_implementation="flash_attention_2",
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)
            
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            print(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ### for llama3
        sources = [f"{self.tokenizer.apply_chat_template([{'role': 'user', 'content': ex['input']}], tokenize=False, add_generation_prompt=True)}" for ex in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]

        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        
        # Apply padding
        _pad_token = self.tokenizer.pad_token_id if self.tokenizer._pad_token is not None else self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")  ## for llama3   

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=_pad_token)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(_pad_token),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict
    
@dataclass
class DataCollatorForMultiTurnCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    target_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ### for llama3
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for example in instances:
            input_ids = [self.tokenizer.bos_token_id]
            attention_mask = [1]
            labels = [IGNORE_INDEX]

            # in case of 'with prompt'
            if 'prompt' in example:
                if not W_PROMPT_RANDOMIZE or ( W_PROMPT_RANDOMIZE and (random.sample([0,1], 1)[0] % 2 == 0) ):
                    # print('---------------------------- this is w_prompt mode !!! -------------------------------')
                    output = self.tokenizer.apply_chat_template(example['prompt'], tokenize=True, return_dict=True, add_generation_prompt=False)
                    input_ids += output['input_ids'][1:]
                    attention_mask += output['attention_mask'][1:]
                    labels += [IGNORE_INDEX for _ in range(len(output['input_ids'][1:]))]


            for conv_dict in example['conv']:
                if conv_dict['role']=='user':
                    output = self.tokenizer.apply_chat_template([conv_dict], tokenize=True, return_dict=True, add_generation_prompt=True)
                    input_ids += output['input_ids'][1:]
                    attention_mask += output['attention_mask'][1:]
                    labels += [IGNORE_INDEX for _ in range(len(output['input_ids'][1:]))]
                elif conv_dict['role']=='assistant':
                    output = self.tokenizer(f"{conv_dict['content']}{self.tokenizer.eos_token}", max_length=self.target_max_len, truncation=True, add_special_tokens=False)
                    input_ids += output['input_ids']
                    attention_mask += output['attention_mask']
                    labels += output['input_ids']
                else:
                    raise NotImplementedError()

            batch_input_ids.append(torch.tensor(input_ids))
            batch_attention_mask.append(torch.tensor(attention_mask))
            batch_labels.append(torch.tensor(labels))
        
        # Apply padding
        _pad_token = self.tokenizer.pad_token_id if self.tokenizer._pad_token is not None else self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")  ## for llama3

        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=_pad_token)
        batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
        batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': batch_labels
        }

        return data_dict

    
@dataclass
class DataCollatorForMultiTurnMixtureTraining(object):
    tokenizer: transformers.PreTrainedTokenizer
    target_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        stage2_dict = self._collator(instances, 'stage2_conv')
        
        stage1_dict = self._collator(instances, 'stage1_conv')
        # change the key name of the stage1_dict
        stage1_dict = {f'stage1_{k}': v for k, v in stage1_dict.items()}

        return {**stage1_dict, **stage2_dict}

    def _collator(self, instances: Sequence[Dict], input_column_name) -> Dict[str, torch.Tensor]:
        ### for llama3
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for example in instances:
            input_ids = [self.tokenizer.bos_token_id]
            attention_mask = [1]
            labels = [IGNORE_INDEX]

            assert 'prompt' not in example, f'{example.keys()}'
            # in case of 'with prompt'
            if 'prompt' in example:
                # print('---------------------------- this is w_prompt mode !!! -------------------------------')
                output = self.tokenizer.apply_chat_template(example['prompt'], tokenize=True, return_dict=True, add_generation_prompt=False)
                input_ids += output['input_ids'][1:]
                attention_mask += output['attention_mask'][1:]
                labels += [IGNORE_INDEX for _ in range(len(output['input_ids'][1:]))]


            for conv_dict in example[input_column_name]:
                if conv_dict['role']=='user':
                    output = self.tokenizer.apply_chat_template([conv_dict], tokenize=True, return_dict=True, add_generation_prompt=True)
                    input_ids += output['input_ids'][1:]
                    attention_mask += output['attention_mask'][1:]
                    labels += [IGNORE_INDEX for _ in range(len(output['input_ids'][1:]))]
                elif conv_dict['role']=='assistant':
                    output = self.tokenizer(f"{conv_dict['content']}{self.tokenizer.eos_token}", max_length=self.target_max_len, truncation=True, add_special_tokens=False)
                    input_ids += output['input_ids']
                    attention_mask += output['attention_mask']
                    labels += output['input_ids']
                else:
                    raise NotImplementedError()

            batch_input_ids.append(torch.tensor(input_ids))
            batch_attention_mask.append(torch.tensor(attention_mask))
            batch_labels.append(torch.tensor(labels))
        
        # Apply padding
        _pad_token = self.tokenizer.pad_token_id if self.tokenizer._pad_token is not None else self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")  ## for llama3

        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=_pad_token)
        batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
        batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': batch_labels
        }

        return data_dict

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    args.context_id = int(args.context_id)
    print(args.context_id)
    assert args.context_id != -1, f"context id wrong: {args.context_id}"

    if 'agentbench-' in args.dataset:
        taskname = re.findall(r"agentbench-(.+)", args.dataset)[0]
        print('----------------------------------------------')
        print(taskname)
        print('----------------------------------------------')
        if args.dataset_format == 'cot-stage1_conv_stage2_joint_loss':
            path = f'src/dataset_cls/agentbench/{taskname}/pseudo_input_conv'
            mydataset = AgentBenchDataset(environment_name=f'{taskname}', dataset_mode='pseudo', load_dataset_path=path)
            template_list = [
                {'role': 'user', 'content': None},
                {'role': 'assistant', 'content': None}
            ]

            # stage1
            stage1_conv = []
            for ex in mydataset.ds:
                instance_conv = copy.deepcopy(template_list)
                instance_conv[0]['content'] = mydataset.get_meta_cognition_input_prompt(
                    input=ex['pseudo_input'],
                    teacher_output=ex['teacher_output_single'],
                    student_output=ex['student_output_single']
                )
                instance_conv[1]['content'] = mydataset.get_meta_cognition_output_prompt(
                    context=ex['context'],
                    reason=ex['reason']
                )
                stage1_conv.append(instance_conv)


            # stage2
            END_CONVERSATION_TOKEN = "[END_CONVERSATION]"
            stage2_conv = []
            for ex in mydataset.ds:
                teacher_output = ex['teacher_output_conv']
                if ex['teacher_output_conv'][-1]['role'] == 'user' and END_CONVERSATION_TOKEN in ex['teacher_output_conv'][-1]['content']:
                    teacher_output = ex['teacher_output_conv'][:-1]

                instance_conv = [{
                    'role': 'user', 
                    'content': mydataset.get_student_input_prompt(input=ex['pseudo_input'])
                }] + teacher_output
                stage2_conv.append(instance_conv)
            

            dataset = DatasetDict({
                'train': Dataset.from_dict({
                    'stage1_conv': stage1_conv,
                    'stage2_conv': stage2_conv
                })
            })
            dataset['train'] = dataset['train'].shuffle(seed=42)

        elif args.dataset_format == 'conv-stage2':
            END_CONVERSATION_TOKEN = "[END_CONVERSATION]"
            path = f'src/dataset_cls/agentbench/{taskname}/pseudo_input_conv'
            mydataset = AgentBenchDataset(environment_name=f'{taskname}', dataset_mode='pseudo', load_dataset_path=path)
            
            all_conv = []
            for ex in mydataset.ds:
                teacher_output = ex['teacher_output_conv']
                if ex['teacher_output_conv'][-1]['role'] == 'user' and END_CONVERSATION_TOKEN in ex['teacher_output_conv'][-1]['content']:
                    teacher_output = ex['teacher_output_conv'][:-1]

                instance_conv = [{
                    'role': 'user', 
                    'content': mydataset.get_student_input_prompt(input=ex['pseudo_input'])
                }] + teacher_output
                all_conv.append(instance_conv)

            dataset = DatasetDict({
                'train': Dataset.from_dict({
                    'conv': all_conv
                })
            })
            dataset['train'] = dataset['train'].shuffle(seed=42)
        elif args.dataset_format == 'conv-stage2_w_prompt':
            END_CONVERSATION_TOKEN = "[END_CONVERSATION]"
            path = f'src/dataset_cls/agentbench/{taskname}/pseudo_input_conv'
            mydataset = AgentBenchDataset(environment_name=f'{taskname}', dataset_mode='pseudo', load_dataset_path=path)
            
            all_context_history = []
            all_conv = []
            for ex in mydataset.ds:
                _conv_list = re.split(r"<USER>:|<AGENT>:", ex['context'])
                _conv_list = [turn.strip() for turn in _conv_list]
                _conv_list = list(filter(None, _conv_list))
                context_history = [{"role": "user", "content": turn} if i%2==0 else {"role": "assistant", "content": turn} for i, turn in enumerate(_conv_list)]
                if context_history[-1]['role'] == 'user':
                    context_history = context_history[:-1]
                all_context_history.append(context_history)


                teacher_output = ex['teacher_output_conv']
                if ex['teacher_output_conv'][-1]['role'] == 'user' and END_CONVERSATION_TOKEN in ex['teacher_output_conv'][-1]['content']:
                    teacher_output = ex['teacher_output_conv'][:-1]

                instance_conv = [{
                    'role': 'user', 
                    'content': mydataset.get_student_input_prompt(input=ex['pseudo_input']).strip()
                }] + teacher_output
                all_conv.append(instance_conv)

            dataset = DatasetDict({
                'train': Dataset.from_dict({
                    'prompt': all_context_history,
                    'conv': all_conv
                })
            })
            dataset['train'] = dataset['train'].shuffle(seed=42)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet.")

    # Remove unused columns.
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names['train'] if col not in ['input', 'output', 'conv', 'prompt', 'stage1_conv', 'stage2_conv', 'stage3_conv']]
    )

    # # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            if 'stage2_conv' in train_dataset.features:
                train_dataset = train_dataset.map(lambda x: {'length': len(x['stage2_conv'])})
            elif 'conv' in eval_dataset.features:
                eval_dataset = eval_dataset.map(lambda x: {'length': len(x['conv'])})
            else:
                eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            if 'stage2_conv' in train_dataset.features:
                train_dataset = train_dataset.map(lambda x: {'length': len(x['stage2_conv'])})
            elif 'conv' in train_dataset.features:
                train_dataset = train_dataset.map(lambda x: {'length': len(x['conv'])})
            else:
                train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    if 'joint_loss' in args.dataset_format:
        data_collator = DataCollatorForMultiTurnMixtureTraining(
            tokenizer=tokenizer,
            target_max_len=args.target_max_len,
        )
    else:
        data_collator = DataCollatorForMultiTurnCausalLM(
            tokenizer=tokenizer,
            target_max_len=args.target_max_len,
        )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    global W_PROMPT_RANDOMIZE 
    W_PROMPT_RANDOMIZE = args.w_prompt_randomize
    print(f"w_prompt_randomize: {W_PROMPT_RANDOMIZE}")
    
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    if checkpoint_dir is None and args.adapter_checkpoint_dir:
        checkpoint_dir = args.adapter_checkpoint_dir

    set_seed(args.seed)

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)
    model.config.use_cache = False
    print('loaded model')

    
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    if 'joint_loss' in args.dataset_format:
        trainer = MixtureTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
        )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()