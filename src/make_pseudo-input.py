import random
random.seed(42)

from src.dataset_cls.agentbench.dataset import AgentBenchDataset
from tqdm.auto import tqdm
import re
from vllm import LLM, SamplingParams


def get_pseudo_input(dataset_cls, num_gen_per_context, flatten_postprocessing_func, numbered_list_format, num_gen_once=10):
    # num_cutoff = num_gen_per_context
    _num_generation = num_gen_per_context*2  # for removing duplicated 

    context_id_list = list(dict.fromkeys(dataset_cls.ds['context_id']))
    context_list = list(dict.fromkeys(dataset_cls.ds['context']))
    assert len(context_id_list)==len(context_list), "wrong"

    llm = LLM(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        download_dir='/home/haebin/.cache/huggingface/hub',
        tensor_parallel_size=4
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=1.,
        top_p=0.9,
        max_tokens=8192,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    )

    conversations_list = []
    metadata_list = []
    for context_id, context in zip(context_id_list, context_list):
        _task_system_prompt = dataset_cls.get_pseudo_input_system_prompt(context, numbered_list_format)
        for _ in range(int(_num_generation/num_gen_once)):
            conversations_list.append(
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": _task_system_prompt
                        },
                        {
                            "role": "user",
                            "content": dataset_cls.get_pseudo_input_user_prompt(context=context, num_gen_inputs=num_gen_once)
                        }
                    ],
                    tokenize=False,
                )
            )
            metadata_list.append({'context': context, 'context_id': context_id})
            
    vllm_outputs = llm.generate(conversations_list, sampling_params, use_tqdm=True)

    flatten_context_id_list = []
    flatten_context_list = []
    pseudo_input_list = []
    for out_, metadata in tqdm(zip(vllm_outputs, metadata_list), total=len(context_list*int(_num_generation/num_gen_once)), desc="Pseudo..", colour='red'):
        out_ = out_.outputs[0].text.strip()
        response = out_.split("<|end_header_id|>")[-1].strip()

        context = metadata['context']
        context_id = metadata['context_id']

        _pseudo_input_list, _flatten_context_list, _flatten_context_id_list = \
            flatten_postprocessing_func(response, context, context_id)

        pseudo_input_list += _pseudo_input_list
        flatten_context_list += _flatten_context_list
        flatten_context_id_list += _flatten_context_id_list

    unique_pseudo_input_set = dict()
    filtered_pseudo_input_list = []
    filtered_context_id_list = []
    filtered_context_list = []
    for context, context_id, pseudo_input in zip(flatten_context_list, flatten_context_id_list, pseudo_input_list):
        if unique_pseudo_input_set.get(pseudo_input) != None:
            continue
        unique_pseudo_input_set[pseudo_input]=1

        filtered_pseudo_input_list.append(pseudo_input)
        filtered_context_id_list.append(context_id)
        filtered_context_list.append(context)

    return filtered_context_id_list[:num_gen_per_context], filtered_context_list[:num_gen_per_context], filtered_pseudo_input_list[:num_gen_per_context]


if __name__ == '__main__':
    
    dataset = 'agentbench-webshop'
    num_gen_per_context = 1000
    _num_gen_once = 10

    if dataset == 'spider':
        mydataset = SpiderDataset(dataset_mode='original')
        flatten_context_id_list, flatten_context_list, pseudo_input_list = get_pseudo_input(mydataset, num_gen_per_context, _num_gen_once)
        mydataset.save_pseudo_dataset('src/dataset_cls/spider/spider_pseudo_input_output', flatten_context_id_list, flatten_context_list, pseudo_input_list)
    elif dataset == 'agentbench-os':
        mydataset = AgentBenchDataset(environment_name='os', dataset_mode='original')
        numbered_list_format = "{num}. "
        def flatten_postprocessing_func(response, context, context_id):
            flatten_context_id_list = []
            flatten_context_list = []
            pseudo_input_list = []
            for line in response.split('\n'):
                line = line.strip()
                if line=='':
                    continue
                match = re.search(r'^\d+\.(.*)', line) # question
                
                if not match:
                    continue
                line = match.group(1).strip()
                question = line.split('Question:')[-1].strip()

                if question==None or len(question)==0:
                    continue

                pseudo_input_list.append(question)
                flatten_context_list.append(context)
                flatten_context_id_list.append(context_id)
            return pseudo_input_list, flatten_context_list, flatten_context_id_list
                
        flatten_context_id_list, flatten_context_list, pseudo_input_list = get_pseudo_input(mydataset, num_gen_per_context, flatten_postprocessing_func, numbered_list_format, _num_gen_once)
        mydataset.save_pseudo_dataset('src/dataset_cls/agentbench/os/pseudo_input_output', flatten_context_id_list, flatten_context_list, pseudo_input_list)
    elif dataset == 'agentbench-kg':
        mydataset = AgentBenchDataset(environment_name='kg', dataset_mode='original')
        def _kg_postprocessing_func(str):
            if not ('entities:' in str.lower() and 'question:' in str.lower()):
                return ''
            str = str.split('Question:')[-1].strip()
            str = str.replace('<SEP>', '\n')
            return str
        flatten_context_id_list, flatten_context_list, pseudo_input_list = get_pseudo_input(mydataset, num_gen_per_context, _num_gen_once, _kg_postprocessing_func)
        mydataset.save_pseudo_dataset('src/dataset_cls/agentbench/kg/pseudo_input_conv', flatten_context_id_list, flatten_context_list, pseudo_input_list)
    elif dataset == 'agentbench-m2w':
        mydataset = AgentBenchDataset(environment_name='m2w', dataset_mode='original')
        numbered_list_format = "#### Problem {num}:\n"
        def flatten_postprocessing_func(response, context, context_id):
            flatten_context_id_list = []
            flatten_context_list = []
            pseudo_input_list = []
            for chunk in re.split(r'#### Problem \d+:\n', response):
                chunk = chunk.strip()
                if chunk=='':
                    continue
                if not chunk.startswith("'''"):
                    continue
                
                question = chunk
                if question==None or len(question)==0:
                    continue

                pseudo_input_list.append(question)
                flatten_context_list.append(context)
                flatten_context_id_list.append(context_id)
            return pseudo_input_list, flatten_context_list, flatten_context_id_list
        flatten_context_id_list, flatten_context_list, pseudo_input_list = get_pseudo_input(mydataset, num_gen_per_context, flatten_postprocessing_func, numbered_list_format, _num_gen_once)
        mydataset.save_pseudo_dataset('src/dataset_cls/agentbench/m2w/pseudo_input_conv', flatten_context_id_list, flatten_context_list, pseudo_input_list)
    elif dataset == 'agentbench-webshop':
        mydataset = AgentBenchDataset(environment_name='webshop', dataset_mode='original')
        numbered_list_format = "#### Problem {num}:\n"
        def flatten_postprocessing_func(response, context, context_id):
            flatten_context_id_list = []
            flatten_context_list = []
            pseudo_input_list = []
            for chunk in re.split(r'#### Problem \d+:\n', response):
                chunk = chunk.strip()
                if chunk=='':
                    continue
                if 'WebShop [SEP]' not in chunk:
                    continue
                
                question = chunk
                if question==None or len(question)==0:
                    continue

                pseudo_input_list.append(question)
                flatten_context_list.append(context)
                flatten_context_id_list.append(context_id)
            return pseudo_input_list, flatten_context_list, flatten_context_id_list
        flatten_context_id_list, flatten_context_list, pseudo_input_list = get_pseudo_input(mydataset, num_gen_per_context, flatten_postprocessing_func, numbered_list_format, _num_gen_once)
        mydataset.save_pseudo_dataset('src/dataset_cls/agentbench/webshop/pseudo_input_conv', flatten_context_id_list, flatten_context_list, pseudo_input_list)
    else:
        raise NotImplementedError()
