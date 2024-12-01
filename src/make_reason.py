import random
random.seed(42)

from src.dataset_cls.agentbench.dataset import AgentBenchDataset
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams


def get_reason_openai(dataset_cls):
    ds = dataset_cls.ds

    _task_system_prompt = dataset_cls.get_reason_system_prompt()
    _aclient = AsyncOpenAI()
    api = OpenAIMultiOrderedClient(_aclient, endpoint="chat.completions", data_template={"model": "gpt-3.5-turbo-0125"})

    def _make_requests():
        for ex in ds:
            api.request(data={
                "messages": [
                    {
                        "role": "system",
                        "content": _task_system_prompt
                    },
                    {
                        "role": "user",
                        "content": dataset_cls.get_reason_user_prompt(
                            context=ex['context'],
                            input=ex['pseudo_input'],
                            student_output=ex['student_output'],
                            teacher_output=ex['teacher_output']
                        )
                    }
                ],
                "temperature": 1.
            })
    api.run_request_function(_make_requests)

    response_list = []
    for result in tqdm(api, total=len(ds), desc="Reason..", colour='red'):
        # context = result.metadata['context']
        response = result.response.choices[0].message.content.strip()
        response_list.append(response)
    return response_list


def get_reason(dataset_cls):
    ds = dataset_cls.ds

    llm = LLM(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        download_dir='/mnt/sda/haebin/.cache/huggingface/hub',
        tensor_parallel_size=4
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    )

    conversations_list = []
    for ex in ds:
        conversations_list.append(
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": dataset_cls.get_reason_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": dataset_cls.get_reason_user_prompt(
                            context=ex['context'],
                            input=ex['pseudo_input'],
                            student_output=ex['student_output_single'],
                            teacher_output=ex['teacher_output_single']
                        )
                    }
                ],
                tokenize=False,
            )
        )
            
    vllm_outputs = llm.generate(conversations_list, sampling_params, use_tqdm=True)

    response_list = []
    for out_ in tqdm(vllm_outputs, total=len(ds), desc="Reason..", colour='red'):
        response = out_.outputs[0].text.strip()
        response = response.split("<|end_header_id|>")[-1].strip()
        response_list.append(response)

    return response_list


if __name__ == '__main__':
    dataset = 'agentbench-webshop'

    if dataset == 'spider':
        mydataset = SpiderDataset(dataset_mode='pseudo')
        ds = mydataset.ds
        response_list = get_reason(mydataset)
        mydataset.save_pseudo_dataset(
            context_id_list=ds['context_id'],
            context_list=ds['context'],
            pseudo_input_list=ds['pseudo_input'],
            teacher_output_list=ds['teacher_output'],
            student_output_list=ds['student_output'],
            reason=response_list
        )
    elif dataset == 'agentbench-os':
        path = 'src/dataset_cls/agentbench/os/pseudo_input_output'
        mydataset = AgentBenchDataset(environment_name='os', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
        response_list = get_reason(mydataset)
        mydataset.save_pseudo_dataset(
            path,
            context_id_list=ds['context_id'],
            context_list=ds['context'],
            pseudo_input_list=ds['pseudo_input'],
            teacher_output_list=ds['teacher_output_single'],
            student_output_list=ds['student_output_single'],
            reason=response_list
        )
    elif dataset == 'agentbench-kg':
        path = 'src/dataset_cls/agentbench/kg/pseudo_input_conv'
        mydataset = AgentBenchDataset(environment_name='kg', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
        response_list = get_reason(mydataset)
        mydataset.save_pseudo_dataset(
            path,
            context_id_list=ds['context_id'],
            context_list=ds['context'],
            pseudo_input_list=ds['pseudo_input'],
            teacher_output_list=ds['teacher_output'],
            student_output_list=ds['student_output'],
            reason=response_list
        )
    elif dataset == 'agentbench-m2w':
        path = 'src/dataset_cls/agentbench/m2w/pseudo_input_conv'
        mydataset = AgentBenchDataset(environment_name='m2w', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
        response_list = get_reason(mydataset)
        mydataset.save_pseudo_dataset(
            path,
            context_id_list=ds['context_id'],
            context_list=ds['context'],
            pseudo_input_list=ds['pseudo_input'],
            teacher_output_list=ds['teacher_output_single'],
            student_output_list=ds['student_output_single'],
            reason=response_list
        )
    elif dataset == 'agentbench-webshop':
        path = 'src/dataset_cls/agentbench/webshop/pseudo_input_conv'
        mydataset = AgentBenchDataset(environment_name='webshop', dataset_mode='pseudo', load_dataset_path=path)
        ds = mydataset.ds
        response_list = get_reason(mydataset)
        mydataset.save_pseudo_dataset(
            path,
            context_id_list=ds['context_id'],
            context_list=ds['context'],
            pseudo_input_list=ds['pseudo_input'],
            teacher_output_list=ds['teacher_output_single'],
            student_output_list=ds['student_output_single'],
            reason=response_list
        )
    else:
        raise NotImplementedError()
