import os
os.environ['CUDA_VISIBLE_DEVICES']="0"

from vllm import LLM, SamplingParams
from src.dataset_cls.agentbench.dataset import AgentBenchDataset

import random
random.seed(42)

import re
from tqdm.auto import tqdm

END_CONVERSATION_TOKEN = "[END_CONVERSATION]"


if __name__ == '__main__':
    # dataset = 'spider'
    dataset = 'agentbench-webshop'

    if dataset == 'spider':
        mydataset = SpiderDataset(dataset_mode='pseudo')
        ds = mydataset.ds
    elif dataset == 'agentbench-os':
        max_turn = 10
        path = 'src/dataset_cls/agentbench/os/pseudo_input_output'
        mydataset = AgentBenchDataset(environment_name='os', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
        ENV_NAME = "Operating System"
        ENV_TURN_START_IDX = 1
        END_CONVERSATION_CONDITION = "you want end this conversation"
    elif dataset == 'agentbench-kg':
        max_turn = 20
        path = 'src/dataset_cls/agentbench/kg/pseudo_input_conv'
        mydataset = AgentBenchDataset(environment_name='kg', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
        ENV_NAME = "Freebase Engine"
        ENV_TURN_START_IDX = 3
    elif dataset == 'agentbench-m2w':
        max_turn = 2
        path = 'src/dataset_cls/agentbench/m2w/pseudo_input_conv'
        mydataset = AgentBenchDataset(environment_name='m2w', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
        ENV_NAME = "HTML Multi-choice QA Problem"
        ENV_TURN_START_IDX = 1
        END_CONVERSATION_CONDITION = "you want end this conversation"
    elif dataset == 'agentbench-webshop':
        max_turn = 10
        path = 'src/dataset_cls/agentbench/webshop/pseudo_input_conv'
        mydataset = AgentBenchDataset(environment_name='webshop', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
        ENV_NAME = "Observation of web shopping page, and Available Actions including 'has_search_bars' 'clickables'"
        ENV_TURN_START_IDX = 1
        END_CONVERSATION_CONDITION = "the click[Buy Now] action is triggered"
        ENV_REPEAT_DESCRIPTION = "\n\n### Requirements:\n - Actions: Do not generate '...' in 'clickable' list even if it's been used in the context above. Generate the actual clickable elements from the observation in the 'clickable' list.\n - Observation: Follow the format as above example like 'Instruction: [SEP] <seach instruction> [SEP] <web info1> [SEP] <web info2> [SEP] ...'. But imagine a new, shorter web shopping page and product information to solve the current search instruction. Do not generate the same web page as the examples above.\n\n"
        ENV_DESCRIPTION = ENV_REPEAT_DESCRIPTION
    else:
        raise NotImplementedError()


    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.,
        max_tokens=2048,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    )

    
    all_teacher_output_conv_list = []
    for ex in tqdm(ds, total=len(ds)):
        conv_str = mydataset.get_teacher_input_prompt(context=ex['context'], input=ex['pseudo_input']).strip()
        conv_list = re.split(r"<USER>:|<AGENT>:", conv_str)
        conv_list = [turn.strip() for turn in conv_list]
        conv_list = list(filter(None, conv_list))

        user_persona_system_prompt = [{"role": "system", "content": f"You have to act like a {ENV_NAME} of context below. {ENV_DESCRIPTION} If {END_CONVERSATION_CONDITION}, you can end the conversation with {END_CONVERSATION_TOKEN} token."}]
        user_persona_history = [{"role": "user", "content": turn} if i%2==0 else {"role": "assistant", "content": turn} for i, turn in enumerate(conv_list[ENV_TURN_START_IDX:])]

        assistant_persona_system_prompt = [{"role": "system", "content": "You have to act like a AGENT of context below."}]
        assistant_persona_history = [{"role": "user", "content": turn} if i%2==0 else {"role": "assistant", "content": turn} for i, turn in enumerate(conv_list)]

        teacher_output_conv_list = []
        for _ in range(max_turn):

            assistant_persona_vllm_input = tokenizer.apply_chat_template(
                assistant_persona_system_prompt + assistant_persona_history, tokenize=False,
            )
            assistant_persona_vllm_outputs = llm.generate([assistant_persona_vllm_input], sampling_params, use_tqdm=False)
            assistant_persona_output = assistant_persona_vllm_outputs[0].outputs[0].text.strip()
            assistant_persona_output = assistant_persona_output.split("<|end_header_id|>")[-1].strip()

            assistant_persona_history.append({"role": "assistant", "content": assistant_persona_output})
            user_persona_history.append({
                "role": "user",
                "content": assistant_persona_output + ENV_REPEAT_DESCRIPTION if dataset == 'agentbench-webshop' else ''
            })

            user_persona_vllm_input = tokenizer.apply_chat_template(
                user_persona_system_prompt + user_persona_history, tokenize=False,
            )
            user_persona_vllm_outputs = llm.generate([user_persona_vllm_input], sampling_params, use_tqdm=False)
            user_persona_output = user_persona_vllm_outputs[0].outputs[0].text.strip()
            user_persona_output = user_persona_output.split("<|end_header_id|>")[-1].strip()

            user_persona_history.append({"role": "assistant", "content": user_persona_output})
            assistant_persona_history.append({"role": "user", "content": user_persona_output})

            teacher_output_conv_list.append({"role": "assistant", "content": assistant_persona_output})
            teacher_output_conv_list.append({"role": "user", "content": user_persona_output})

            if END_CONVERSATION_TOKEN in user_persona_output:
                break

        all_teacher_output_conv_list.append(teacher_output_conv_list)

    mydataset.save_pseudo_dataset(path, ds['context_id'], ds['context'], ds['pseudo_input'], ds['teacher_output_single'], ds['student_output_single'], ds['reason'], all_teacher_output_conv_list)
