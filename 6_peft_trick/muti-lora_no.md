# muti-lora实现多任务统一推理

## 一、为什么需要 muti-lora实现多任务统一推理？

由于模型较大，训练及部署成本较高，lora作为较优的参数高效微调方法，具有很强的实用性，使用muti-lora可以实现基于一个基础模型+不同的 lora adapter 在一个服务中部署多个任务，根据不同的任务id来启用不同的lora模型，不用每个任务单独部署，有利于节省资源。

## 二、如何实现 muti-lora 多任务统一推理？

### 2.1 如何基于peft 实现 muti-lora 多任务统一推理？

- requirement

```s
    python==3.10.12 
    torch=='2.1.2+cu121'
    transformers==3.9.3 
    peft==0.9.0
```

- 代码实现

```s
'''
refer to https://huggingface.co/docs/transformers/v4.39.3/zh/peft
python==3.10.12 + torch=='2.1.2+cu121' + transformers==3.9.3 + peft==0.9.0 
'''
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# print(model)
model_id = "Qwen/Qwen-1_8B-Chat"
peft_model_id = "lora1"
peft_model_id2 = "lora2"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True).eval()

# model = PeftModel.from_pretrained(model, peft_model_id, adapter_name="adapter_1")
# model = PeftModel.from_pretrained(model, peft_model_id2, adapter_name="adapter_2")
model.load_adapter(peft_model_id, adapter_name="adapter_1") # 联系方识别模型
model.load_adapter(peft_model_id2, adapter_name="adapter_2") # 信息抽取模型

print(model)
t1 = time.time()
# use adapter_1
model.set_adapter("adapter_1")
t2 = time.time()
query = '请你判别下述文本中包含联系方式嘛？ 我的电话是13225535689'
response, _ = model.chat(tokenizer, query, history=None)
t3 = time.time()
print(response, t2-t1, t3-t1)


# use adapter_2
model.set_adapter("adapter_2")
query = '请你抽取出下述文本中的联系方式。 我的电话是13225535689'
response, _ = model.chat(tokenizer, query, history=None)
print(response)


#禁用adapter, 使用base model
model.disable_adapters()
query = '请你判别下述文本中包含联系方式嘛？ 我的电话是13225535689'
response, _ = model.chat(tokenizer, query, history=None)
print(response)


#启用adapters
model.enable_adapters()
model.set_adapter("adapter_1")
query = '请你判别下述文本中包含联系方式嘛？ 我的电话是13225535689'
response, _ = model.chat(tokenizer, query, history=None)
print(response)
```

### 2.2 如何基于 vllm 实现 muti-lora 多任务统一推理？

- requirement

```s
    python==3.10.12
    torch=='2.1.2+cu121'
    transformers==3.9.3
    vllm==0.4.0 
    flash-atten
```

- 代码实现

```s
"""
This example shows how to use the multi-LoRA functionality
for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""
import time
from typing import List, Optional, Tuple
from huggingface_hub import snapshot_download
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def create_test_prompts(
       lora_path: str,
       query: str
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.


    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128), None),
        ("To be or not to be,",
         SamplingParams(temperature=0.8,
                        top_k=5,
                        presence_penalty=0.2,
                        max_tokens=128), None),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0.0,
                           logprobs=1,
                           prompt_logprobs=1,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora", 1, lora_path)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
            SamplingParams(n=3,
                           best_of=3,
                           use_beam_search=True,
                           temperature=0,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora", 1, lora_path)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0.0,
                           logprobs=1,
                           prompt_logprobs=1,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora2", 2, lora_path)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
            SamplingParams(n=3,
                           best_of=3,
                           use_beam_search=True,
                           temperature=0,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora", 1, lora_path)),
        
        (
            "[user] {} [/user] [assistant]".format(query),  # noqa: E501
            SamplingParams(n=1,
                           best_of=3,
                           use_beam_search=True,
                           temperature=0,
                           max_tokens=300),
            LoRARequest("element_code-lora", 1, lora_path)),
    ]


def process_requests_single(engine: LLMEngine,
                            test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]],
                            request_id=0):
    """Continuously process a list of prompts and handle the outputs."""
    res = ''
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            test_prompts = [test_prompts[request_id]]
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            # request_id += 1
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                res = request_output.outputs[0].text
                # print(request_output.outputs[1].text)
    res = request_output.outputs[0].text
    return res
                
def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]],
                      ):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1


        request_outputs: List[RequestOutput] = engine.step()


        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine(model_pth) -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model=model_pth,
                             enable_lora=True,
                             max_loras=1,
                             max_lora_rank=8,
                             max_cpu_loras=2,
                             max_num_seqs=256,
                             gpu_memory_utilization=0.9,
                             max_model_len=1024,
                             trust_remote_code=True)
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""


    pretrained_model_pth = "Qwen1.5/Qwen1.5-7B-Chat"
    engine = initialize_engine(pretrained_model_pth)

    lora_path = "lora1"
    query = "你好， 你是谁？"

    t1 = time.time()
    test_prompts = create_test_prompts(lora_path, query=query) 
    ##单任务单lora调用       
    res = process_requests_single(engine, test_prompts, request_id=6)
    t2 = time.time()
    print('---', res, t2-t1)
    
    ##多任务多lora调用，也可以换不同的lora模型，需更改代码       
    res = process_requests(engine, test_prompts)

if __name__ == '__main__':
    main()
```

### 2.3 如何基于 S-LoRA 实现 muti-lora 多任务统一推理？

> 代码：https://github.com/S-LoRA/S-LoR

- requirement

- 参考
  - S-LoRA：一个GPU运行数千大模型成为可能  https://zhuanlan.zhihu.com/p/666972073
  - S-LoRA：同时应用多个LoRA模块并行推理   https://zhuanlan.zhihu.com/p/681430762

## 致谢

- muti-lora实现多任务统一推理  https://zhuanlan.zhihu.com/p/691710751

