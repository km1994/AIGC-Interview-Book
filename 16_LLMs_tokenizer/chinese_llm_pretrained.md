# 怎么让英文大语言模型支持中文？（二） —— 继续预训练篇

- [怎么让英文大语言模型支持中文？（二） —— 继续预训练篇](#怎么让英文大语言模型支持中文二--继续预训练篇)
  - [一、为什么需要进行继续预训练？](#一为什么需要进行继续预训练)
  - [二、如何对 继续预训练 数据预处理？](#二如何对-继续预训练-数据预处理)
  - [三、如何 构建模型？](#三如何-构建模型)
  - [四、如何 使用模型？](#四如何-使用模型)
  - [致谢](#致谢)

## 一、为什么需要进行继续预训练？

前面我们已经讲过怎么构建中文领域的tokenization：

接下来我们将介绍继续预训练。

我们**新增加了一些中文词汇到词表中，这些词汇是没有得到训练的，因此在进行指令微调之前我们要进行预训练**。预训练的方式一般都是相同的，简单来说，就是根据上一个字预测下一个字是什么。为了方便起见，我们这里直接使用IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese模型，并且tokenizer也是其自带的。

## 二、如何对 继续预训练 数据预处理？

同样的，我们使用的数据还是斗破苍穹小说数据。首先我们看看是怎么处理数据的， 数据位于data下，分别为corpus.txt和test_corpus.txt，每一行为一句或多句话。再看看数据预处理的部分，在test_dataset.py里面：

```s
import os
import logging
import datasets
import transformers

from pprint import pprint
from itertools import chain
from datasets import load_dataset, concatenate_datasets
from transformers.testing_utils import CaptureLogger
from transformers import AutoTokenizer, LlamaTokenizer

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

logger = logging.getLogger(__name__)

lm_datasets = []
files = ["data/test_corpus.txt"]
data_cache_dir = "./cache_data"
preprocessing_num_workers = 1

# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
tokenizer = LlamaTokenizer.from_pretrained("ziqingyang/chinese-llama-lora-7b")
tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese")

def print_dict(adict):
  for k,v in adict.items():
    print(k, v)

def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

block_size = 128

# 将所有文本进行拼接
def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

for idx, file in enumerate(files):
    data_file = file
    filename = ''.join(file.split(".")[:-1])

    cache_path = os.path.join(data_cache_dir, filename)
    os.makedirs(cache_path, exist_ok=True)
    try:
        processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
        print(f'training datasets-{filename} has been loaded from disk')
    except Exception:
        cache_dir = os.path.join(data_cache_dir, filename + "_text")
        os.makedirs(cache_dir, exist_ok=True)

        raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
        print_dict(raw_dataset["train"][0])
        # 直接进行tokenize，需要注意的是只需要在句子开头加上bos_token
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns="text",
            load_from_cache_file=True,
            keep_in_memory=False,
            cache_file_names={k: os.path.join(cache_dir, f'tokenized.arrow') for k in raw_dataset},
            desc="Running tokenizer on dataset",
        )

        print_dict(tokenized_dataset["train"][0])

        grouped_datasets = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=True,
            keep_in_memory=False,
            cache_file_names={k: os.path.join(cache_dir, f'grouped.arrow') for k in tokenized_dataset},
            desc=f"Grouping texts in chunks of {block_size}",
        )
        processed_dataset = grouped_datasets

        print_dict(processed_dataset["train"][0])
        processed_dataset.save_to_disk(cache_path)
    if idx == 0:
        lm_datasets = processed_dataset['train']
    else:
        assert lm_datasets.features.type == processed_dataset["train"].features.type
        lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])

lm_datasets = lm_datasets.train_test_split(test_size=0.1)

print_dict(lm_datasets["train"][0])
```

> 输出
```s
text 又一次上架了，这次比上次还激动，甚至激动到了上传了章节却不知道发出来的地步。
input_ids [21134, 1348, 671, 3613, 677, 3373, 749, 8024, 6821, 3613, 3683, 677, 3613, 6820, 4080, 1220, 8024, 4493, 5635, 4080, 1220, 1168, 749, 677, 837, 749, 4995, 5688, 1316, 679, 4761, 6887, 1355, 1139, 3341, 4638, 1765, 3635, 511, 21133]
token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
attention_mask [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
input_ids [21134, 1348, 671, 3613, 677, 3373, 749, 8024, 6821, 3613, 3683, 677, 3613, 6820, 4080, 1220, 8024, 4493, 5635, 4080, 1220, 1168, 749, 677, 837, 749, 4995, 5688, 1316, 679, 4761, 6887, 1355, 1139, 3341, 4638, 1765, 3635, 511, 21133, 21134, 2219, 2217, 8024, 1068, 754, 3173, 741, 8024, 677, 3373, 1184, 2768, 5327, 1962, 2533, 3300, 763, 1139, 725, 1759, 6486, 4638, 2692, 3160, 8024, 2190, 754, 6821, 819, 1331, 4798, 4638, 2768, 5327, 8024, 1759, 6486, 2552, 7027, 6820, 4696, 3300, 1126, 1146, 2684, 2607, 680, 2558, 2559, 8024, 6006, 6432, 3295, 5307, 3300, 782, 6432, 1759, 6486, 3221, 1170, 1139, 3341, 4638, 3144, 2945, 8024, 2190, 754, 6821, 763, 4522, 6241, 8024, 2769, 738, 2400, 3313, 1922, 6814, 1762, 2692, 8024, 1166, 4638, 2769, 679]
token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
attention_mask [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
labels [21134, 1348, 671, 3613, 677, 3373, 749, 8024, 6821, 3613, 3683, 677, 3613, 6820, 4080, 1220, 8024, 4493, 5635, 4080, 1220, 1168, 749, 677, 837, 749, 4995, 5688, 1316, 679, 4761, 6887, 1355, 1139, 3341, 4638, 1765, 3635, 511, 21133, 21134, 2219, 2217, 8024, 1068, 754, 3173, 741, 8024, 677, 3373, 1184, 2768, 5327, 1962, 2533, 3300, 763, 1139, 725, 1759, 6486, 4638, 2692, 3160, 8024, 2190, 754, 6821, 819, 1331, 4798, 4638, 2768, 5327, 8024, 1759, 6486, 2552, 7027, 6820, 4696, 3300, 1126, 1146, 2684, 2607, 680, 2558, 2559, 8024, 6006, 6432, 3295, 5307, 3300, 782, 6432, 1759, 6486, 3221, 1170, 1139, 3341, 4638, 3144, 2945, 8024, 2190, 754, 6821, 763, 4522, 6241, 8024, 2769, 738, 2400, 3313, 1922, 6814, 1762, 2692, 8024, 1166, 4638, 2769, 679]
input_ids [21134, 1348, 671, 3613, 677, 3373, 749, 8024, 6821, 3613, 3683, 677, 3613, 6820, 4080, 1220, 8024, 4493, 5635, 4080, 1220, 1168, 749, 677, 837, 749, 4995, 5688, 1316, 679, 4761, 6887, 1355, 1139, 3341, 4638, 1765, 3635, 511, 21133, 21134, 2219, 2217, 8024, 1068, 754, 3173, 741, 8024, 677, 3373, 1184, 2768, 5327, 1962, 2533, 3300, 763, 1139, 725, 1759, 6486, 4638, 2692, 3160, 8024, 2190, 754, 6821, 819, 1331, 4798, 4638, 2768, 5327, 8024, 1759, 6486, 2552, 7027, 6820, 4696, 3300, 1126, 1146, 2684, 2607, 680, 2558, 2559, 8024, 6006, 6432, 3295, 5307, 3300, 782, 6432, 1759, 6486, 3221, 1170, 1139, 3341, 4638, 3144, 2945, 8024, 2190, 754, 6821, 763, 4522, 6241, 8024, 2769, 738, 2400, 3313, 1922, 6814, 1762, 2692, 8024, 1166, 4638, 2769, 679]
token_type_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
attention_mask [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
labels [21134, 1348, 671, 3613, 677, 3373, 749, 8024, 6821, 3613, 3683, 677, 3613, 6820, 4080, 1220, 8024, 4493, 5635, 4080, 1220, 1168, 749, 677, 837, 749, 4995, 5688, 1316, 679, 4761, 6887, 1355, 1139, 3341, 4638, 1765, 3635, 511, 21133, 21134, 2219, 2217, 8024, 1068, 754, 3173, 741, 8024, 677, 3373, 1184, 2768, 5327, 1962, 2533, 3300, 763, 1139, 725, 1759, 6486, 4638, 2692, 3160, 8024, 2190, 754, 6821, 819, 1331, 4798, 4638, 2768, 5327, 8024, 1759, 6486, 2552, 7027, 6820, 4696, 3300, 1126, 1146, 2684, 2607, 680, 2558, 2559, 8024, 6006, 6432, 3295, 5307, 3300, 782, 6432, 1759, 6486, 3221, 1170, 1139, 3341, 4638, 3144, 2945, 8024, 2190, 754, 6821, 763, 4522, 6241, 8024, 2769, 738, 2400, 3313, 1922, 6814, 1762, 2692, 8024, 1166, 4638, 2769, 679]
```

具体是：

1. 先使用tokenizer()得到相关的输入，需要注意的是可能会在文本前后添加特殊的标记，比如bos_token_id和eos_token_id，针对于不同模型的tokneizer可能会不太一样。这里在unput_ids前后添加了21134和21133两个标记。
2. 然后将所有文本的input_ids、attention_mask, token_type_ids各自拼接起来（展开后拼接，不是二维数组之间的拼接），再设定一个最大长度block_size，这样得到最终的输入。

## 三、如何 构建模型？

在test_model.py里面我们可以初步使用预训练的模型看看效果：

```s
from transformers import BertTokenizer,GPT2LMHeadModel, AutoModelForCausalLM
hf_model_path = 'IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese'
tokenizer = BertTokenizer.from_pretrained(hf_model_path)
# model = GPT2LMHeadModel.from_pretrained(hf_model_path)
model = AutoModelForCausalLM.from_pretrained(hf_model_path)

def generate_word_level(input_text,n_return=5,max_length=128,top_p=0.9):
    inputs = tokenizer(input_text,return_tensors='pt',add_special_tokens=False).to(model.device)
    gen = model.generate(
                            inputs=inputs['input_ids'],
                            max_length=max_length,
                            do_sample=True,
                            top_p=top_p,
                            eos_token_id=21133,
                            pad_token_id=0,
                            num_return_sequences=n_return)

    sentences = tokenizer.batch_decode(gen)
    for idx,sentence in enumerate(sentences):
        print(f'sentence {idx}: {sentence}')
        print('*'*20)
    return gen
# 西湖的景色
outputs = generate_word_level('西湖的景色',n_return=5,max_length=128)
print(outputs)
```

> 输出
```s
sentence 0: 西 湖 的 景 色 很 美 丽, 古 代 有 个 名 叫 : 西 湖 的 湖 南 和 江 南 的 一 段 。 湖 面 上 有 一 座 小 小 的 湖 泊, 有 一 片 湖 泊 和 一 座 小 岛, 有 一 处 小 的 小 镇 。 在 西 湖 里, 每 个 人 都 是 在 湖 边, 你 可 以 在 小 小 湖 里 畅 游 。 西 湖 上 是 古 代 建 筑, 但 湖 水 不 多 。 西 湖 上 是 一 座 水 库, 古 代 有 个 名 叫 : 西 湖 的 湖 南 和 江 南 的 一 段 。 湖
********************
sentence 1: 西 湖 的 景 色 美 不 胜 数 。 近 日, 位 于 湖 北 省 湖 北 省 石 家 庄 市 的 石 家 庄 旅 游 风 景 区 被 命 名 为 " 湖 北 省 国 家 级 森 林 公 园 " 。 园 内 有 一 座 石 屋, 位 于 石 屋 与 石 屋 的 对 面, 总 面 积 3. 2 平 方 公 里, 其 中 一 座 石 屋, 由 石 屋 和 石 屋 组 成, 一 栋 大 型 石 屋 由 石 屋 组 成, 三 栋 石 屋 由 石 屋 组 成 。 石 屋 主 要 是 一 座 石 屋
********************
sentence 2: 西 湖 的 景 色 在 古 城 、 小 镇 和 城 郊 中, 有 大 片 的 湖 泊, 是 古 典 中 的 佳 肴, 湖 水 清 澈, 湖 中 有 一 大 块 鱼, 在 湖 水 里 散 发 着 浓 郁 的 清 香 。 湖 水 中, 有 各 种 颜 色 的 鱼 、 蟹 、 贝 壳 类 的 水 产 品 。 湖 边 有 的 池 塘 也 有 的 水 果 摊 位, 可 供 上 千 家 店 。 在 湖 中 央 的 湖 中 央 有 三 个 小 水 塘, 水 塘 长 约 三 丈, 两 端 长, 塘 底
********************
sentence 3: 西 湖 的 景 色 也 很 漂 亮, 可 以 说 是 城 市 的 象 征, 而 且 还 有 小 小 的 山 洞, 看 到 了, 我 们 在 西 湖 的 中 心 也 很 近, 所 以 也 没 有 停 止, 西 湖 的 风 景 很 秀 美, 我 们 也 不 愿 意 停 留 在 这 样 的 地 方 。 西 湖 是 世 界 上 最 美 的 湖 泊, 也 是 最 令 人 羡 慕 的 旅 游 区, 西 湖 的 美 丽 不 容 小 视, 是 我 们 心 中 最 美 的 风 景 。 西 湖 在 西 湖
********************
sentence 4: 西 湖 的 景 色 是 如 此 独 特, 那 水 碧 草 如 黛, 池 水 清 新, 一 池 青 湖, 游 人 可 以 品 一 小 池 花 。 " " 好 景 如 画, 山 清 水 秀, 碧 草 如 茵, 池 清 潭 秀 。 " 黄 湖 " 是 西 湖 的 " 绿 色 湖 " 。 西 湖 的 景 色 是 如 此 独 特, 那 水 碧 草 如 黛, 池 水 清 新, 一 池 青 湖, 游 人 可 以 品 一 小 池 花 。 " " 好 景 如 画, 山 清 水 秀, 碧 草 如 茵
********************
```

接下来是使用该模型针对我们自己的数据进行继续预训练了。需要注意的几个地方：

1. 如果是我们自己定义的tokenizer，需要将模型的嵌入层和lm_head层的词表数目进行重新设置：

```s
model_vocab_size = model.get_output_embeddings().weight.size(0)
model.resize_token_embeddings(len(tokenizer))
```

2. 这里我们使用参数有效微调方法lora进行微调，我们需要设置额外保存的参数：transformer.wte,lm_head。这个可以通过find_lora_names.py里面获得。
3. 原始chinsee-llama-alpaca使用lora保存参数有问题，这里进行了修改并只保存一份lora权重。
4. 使用test_pretrained_model.py的时候也要记得先对vocab_size进行重新设置。

```s
    $ torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py --deepspeed ds_zero2_no_offload.json --model_name_or_path IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese --tokenizer_name_or_path IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese --dataset_dir data --data_cache_dir temp_data_cache_dir --validation_split_percentage 0.001 --per_device_train_batch_size 32 --per_device_eval_batch_size 16 --do_train --seed $RANDOM --fp16 --max_steps 2500 --lr_scheduler_type cosine --learning_rate 2e-4 --warmup_ratio 0.05 --weight_decay 0.01 --logging_strategy steps --logging_steps 10 --save_strategy steps --save_total_limit 3 --save_steps 50 --gradient_accumulation_steps 1 --preprocessing_num_workers 8 --block_size 512 --output_dir output_dir --overwrite_output_dir --ddp_timeout 30000 --logging_first_step True --lora_rank 8 --lora_alpha 32 --trainable c_attn --modules_to_save transformer.wte,lm_head --lora_dropout 0.05 --torch_dtype float16 --gradient_checkpointing --ddp_find_unused_parameters False
```

即：

```s
torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
--deepspeed ds_zero2_no_offload.json \
--model_name_or_path IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese \
--tokenizer_name_or_path IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese \
--dataset_dir data \
--data_cache_dir temp_data_cache_dir \
--validation_split_percentage 0.001 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 16 \
--do_train --seed $RANDOM \
--fp16 \
--max_steps 2500 \
--lr_scheduler_type cosine \
--learning_rate 2e-4 \
--warmup_ratio 0.05 \
--weight_decay 0.01 \
--logging_strategy steps \
--logging_steps 10 \
--save_strategy steps \
--save_total_limit 3 \
--save_steps 50 \
--gradient_accumulation_steps 1 \
--preprocessing_num_workers 8 \
--block_size 512 \
--output_dir output_dir \
--overwrite_output_dir \
--ddp_timeout 30000 \
--logging_first_step True \
--lora_rank 8 \
--lora_alpha 32 \
--trainable c_attn \
--modules_to_save transformer.wte,lm_head \
--lora_dropout 0.05 \
--torch_dtype float16 \
--gradient_checkpointing \
--ddp_find_unused_parameters False
```

由于使用了seepspeed中ZeRo，占用的显存会更小。

## 四、如何 使用模型？

最后我们可以这么使用模型，在test_pretrained_model.py中：

```s
import os
import torch
from transformers import BertTokenizer,GPT2LMHeadModel, AutoModelForCausalLM
from peft import PeftModel
hf_model_path = 'IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese'
tokenizer = BertTokenizer.from_pretrained(hf_model_path)
# model = GPT2LMHeadModel.from_pretrained(hf_model_path)
model = AutoModelForCausalLM.from_pretrained(hf_model_path)

model_vocab_size = model.get_output_embeddings().weight.size(0)
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, os.path.join("output_dir", "adapter_model"), torch_dtype=torch.float32)
model.cuda()
model.eval()

def generate_word_level(input_text,n_return=5,max_length=128,top_p=0.9):
    inputs = tokenizer(input_text,return_tensors='pt',add_special_tokens=False).to(model.device)
    gen = model.generate(
                            inputs=inputs['input_ids'],
                            max_length=max_length,
                            do_sample=True,
                            top_p=top_p,
                            eos_token_id=21133,
                            pad_token_id=0,
                            num_return_sequences=n_return)

    sentences = tokenizer.batch_decode(gen)
    for idx,sentence in enumerate(sentences):
        print(f'sentence {idx}: {sentence}')
        print('*'*20)
    return gen

outputs = generate_word_level('眼角斜瞥着柳翎那略微有些阴沉的脸庞。萧炎',n_return=5,max_length=128)
print(outputs)
```

> output
```s
sentence 0: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 淡 淡 的 道 。 <|endoftext|> [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
********************
sentence 1: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 一 怔 。 手 掌 猛 然 一 僵 。 手 指 一 扯 。 旋 即 在 房 门 内 停 留 。 旋 即 一 口 鲜 血 喷 涌 而 出 。 <|endoftext|>
********************
sentence 2: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 顿 时 愣 了 愣 。 他 这 是 何 人 ？ 怎 能 知 道 这 位 灰 袍 老 者 出 手 啊 ？ <|endoftext|> [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
********************
sentence 3: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 心 中 有 着 什 么 感 触 ？ <|endoftext|> [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
********************
sentence 4: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 微 皱 着 眉 头 。 转 过 身 。 轻 声 道 ： “ 柳 翎 。 是 你 的 人 ？ ” <|endoftext|> [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
********************
```

对于没有经过继续预训练的模型结果：

```s
sentence 0: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎, 男, 1964 年 生, 河 北 齐 齐 哈 尔 市 人 。 1979 年 毕 业 于 武 汉 工 学 院 中 文 系, 1988 年 毕 业 于 中 国 人 民 大 学 中 文 系, 历 任 中 国 人 民 大 学 高 级 教 师 、 教 育 部 大 学 文 学 系 主 任, 中 国 语 言 文 学 会 理 事, 中 国 人 民 大 学 历 史 学 会 副 会 长, 中 国 作 家 协 会 员, 中 国 作 家 协 会 会
********************
sentence 1: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 的 脸 庞 在 不 同 时 期 会 发 出 来 ， 这 样 的 眉 目 和 眉 目 能 够 很 容 易 的 在 一 起 ， 能 够 让 人 看 得 见 的 就 是 这 样 的 眉 目 。 那 一 对 情 侣 还 是 非 常 喜 欢 的 ， 不 过 他 们 的 交 往 方 式 也 是 各 种 多 样 的 ， 最 后 的 交 往 方 式 就 是 让 所 有 的 人 都 看 到 了 自 己 的 内 心 。 他 们 俩 是 非 常 相
********************
sentence 2: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 眼 睛 看 向 柳 翎, 眼 眸 里 满 是 伤 痕 。 “ 天 边 来 客 。 ” 柳 翎 那 无 情 的 目 光 中 透 着 几 分 冷 漠 的 微 笑 。 “ 没 有 你 的 名 字, 你 只 是 名 字 。 ” 柳 翎 在 柳 翎 眼 前 一 怔, 无 意 中 却 看 出 了 柳 翎 已 经 在 想 要 离 开 了 。 柳 翎 说 这 些 东 西 有 的 是 一 次 次 的 意 外, 她 还 是 有 意 的,
********************
sentence 3: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 的 脸 上 只 有 几 分 阴 沉, 但 却 能 够 带 着 微 微 的 怜 惜 之 心 。 萧 炎 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 眼 角
********************
sentence 4: 眼 角 斜 瞥 着 柳 翎 那 略 微 有 些 阴 沉 的 脸 庞 。 萧 炎 已 经 是 年 轻 貌 美 的 人, 在 某 处 留 下 的 是 无 尽 的 光 影 。 她 的 微 笑 也 在 耳 畔 闪 烁 着 光 影 。 他 不 断 地 伸 出 手 指, 他 在 他 的 微 笑 中 轻 松 地 走 着, 而 柳 翎 却 始 终 沉 默 。 他 已 经 是 个 女 孩 子, 在 某 处 也 许 你 听 不 见 。 他 轻 轻 地 接 过 他 的 手, 轻 轻 地 说 道 : " 没 有 人 听
********************
```

模型确实得到了有效的训练。


## 致谢

1. 怎么让英文大语言模型支持中文？（二）继续预训练：https://mp.weixin.qq.com/s/iGGJ98i_E2l6kdClMsk1Tg
2. https://github.com/taishan1994/chinese_llm_pretrained


