import os
os.environ['CUDA_VISIBLE_DEVICES']="0"

from vllm import LLM, SamplingParams
from src.dataset_cls.agentbench.dataset import AgentBenchDataset
import random
random.seed(42)

import re


if __name__ == '__main__':
    # dataset = 'spider'
    dataset = 'agentbench-webshop'

    if dataset == 'spider':
        mydataset = SpiderDataset(dataset_mode='pseudo')
        ds = mydataset.ds
    elif dataset == 'agentbench-os':
        path = 'src/dataset_cls/agentbench/os/pseudo_input_output'
        mydataset = AgentBenchDataset(environment_name='os', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
    elif dataset == 'agentbench-kg':
        path = 'src/dataset_cls/agentbench/kg/pseudo_input_conv'
        mydataset = AgentBenchDataset(environment_name='kg', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
    elif dataset == 'agentbench-m2w':
        path = 'src/dataset_cls/agentbench/m2w/pseudo_input_conv'
        mydataset = AgentBenchDataset(environment_name='m2w', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
    elif dataset == 'agentbench-webshop':
        path = 'src/dataset_cls/agentbench/webshop/pseudo_input_conv'
        mydataset = AgentBenchDataset(environment_name='webshop', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
    else:
        raise NotImplementedError()

    
    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        download_dir='/mnt/sda/haebin/.cache/huggingface/hub',
        tensor_parallel_size=1
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.,
        max_tokens=2048,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    )

    
    _student_input_list = []
    for ex in ds:
        turn = mydataset.get_student_input_prompt(input=ex['pseudo_input']).strip()
        _student_input_list.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": turn}
                ],
                tokenize=False,
            )
        )
    _student_vllm_outputs = llm.generate(_student_input_list, sampling_params, use_tqdm=True)
    student_output_list = [out_.outputs[0].text.strip() for out_ in _student_vllm_outputs]
    student_output_list = [out_.split("<|end_header_id|>")[-1].strip() for out_ in student_output_list]


    _teacher_input_list = []
    for ex in ds:
        conv_str = mydataset.get_teacher_input_prompt(context=ex['context'], input=ex['pseudo_input']).strip()
        conv_list = re.split(r"<USER>:|<AGENT>:", conv_str)
        conv_list = [turn.strip() for turn in conv_list]
        conv_list = list(filter(None, conv_list))
        _teacher_input_list.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."}] + \
                [{"role": "user", "content": turn} if i%2==0 else {"role": "assistant", "content": turn} for i, turn in enumerate(conv_list)],
                tokenize=False,
            )
        )
    _teacher_vllm_outputs = llm.generate(_teacher_input_list, sampling_params, use_tqdm=True)
    teacher_output_list = [out_.outputs[0].text.strip() for out_ in _teacher_vllm_outputs]
    teacher_output_list = [out_.split("<|end_header_id|>")[-1].strip() for out_ in teacher_output_list]

    mydataset.save_pseudo_dataset(path, ds['context_id'], ds['context'], ds['pseudo_input'], teacher_output_list, student_output_list)
    