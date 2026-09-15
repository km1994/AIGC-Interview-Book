# 使用 LoRA 对 大模型进行 高效参数微调

- [使用 LoRA 对 大模型进行 高效参数微调](#使用-lora-对-大模型进行-高效参数微调)
  - [一、前言](#一前言)
  - [step 1：LLMs 加载](#step-1llms-加载)
  - [step 2：配置LoraConfig](#step-2配置loraconfig)
  - [step 3：PEFT 包装](#step-3peft-包装)
    - [3.1 LORA 整体实现思路](#31-lora-整体实现思路)
    - [3.2 LORA \_find\_and\_replace() 实现思路](#32-lora-_find_and_replace-实现思路)
    - [3.3 Lora层的实现 思路](#33-lora层的实现-思路)
      - [3.3.1 基类 LoraLayer 实现](#331-基类-loralayer-实现)
      - [3.3.2 Linear 实现](#332-linear-实现)
  - [step 4：数据加载](#step-4数据加载)
  - [step 5：定义优化器，并开始训练](#step-5定义优化器并开始训练)
  - [step 6：模型保存](#step-6模型保存)
  - [step 7：推理前的Weight合并](#step-7推理前的weight合并)
  - [step 8：加载合并后的模型进行推理](#step-8加载合并后的模型进行推理)
  - [参考](#参考)


## 一、前言

本文章 主要介绍 使用 LoRA 对 大模型进行 高效参数微调，涉及内容：

1. 涉及SFT部分的代码实现记录；
2. 在调试过程中遇到的问题，添加注释进行解释说明；
3. 在推理时如何先进行weight的合并在加载模型进行推理；

涉及框架

```s
  torch
  peft
  transformers
```

## step 1：LLMs 加载

```s
# AutoModel可以使用自定义的模型trust_remote_code要设定为True
# 注意这里对model的half转换，要在加载原始模型时就转换，不能在获取peft_model之后
# 否则可能会因为精度的转换问题出现Loss为nan的情况
# 同时需要注意half转换要在分配gpu之前
# 打印模型参数可以发现chatglm-6B模型参数都是torch.float16类型的，这里要用.half()
model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True).half()
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
```

## step 2：配置LoraConfig

```s
  # 设置超参数及配置
  LORA_R = 8
  LORA_ALPHA = 16
  LORA_DROPOUT = 0.05
  TARGET_MODULES = [
      "q_proj",
      "v_proj",
  ]

  config = LoraConfig(
      r=LORA_R,
      lora_alpha=LORA_ALPHA,
      target_modules=TARGET_MODULES,
      lora_dropout=LORA_DROPOUT,
      bias="none",
      task_type="CAUSAL_LM",
  )
```

- 参数介绍：
  - r：lora的秩，矩阵A和矩阵B相连接的宽度，r<<d；
  - lora_alpha：归一化超参数，lora参数 ΔWx 被以 α/r 归一化，以便减少改变r rr时需要重新训练的计算量；
  - target_modules：lora的目标位置；
  - merge_weights:eval模式中，是否将lora矩阵的值加到原有 W0 的值上;
  - lora_dropout：lora层的dropout比率；
  - fan_in_fan_out：只有应用在Conv1D层时置为True，其他情况False；
  - bias: 是否可训练bias，none：均不可；all：均可；lora_only：只有lora部分的bias可训练；
  - task_type：这是LoraConfig的父类PeftConfig中的参数，设定任务的类型；
  - modules_to_save：除了lora部分之外，还有哪些层可以被训练，并且需要保存；

> 注意：target_modules中的作用目标名在不同模型中的名字是不一样的。query_key_value是在ChatGLM中的名字

## step 3：PEFT 包装

PEFT 包装

```s
  # 加入PEFT策略
  model = get_peft_model(model, config)
  model = model.to(device)
```

### 3.1 LORA 整体实现思路

具体 PEFT 包装 包装，结合PEFT模块的源码，来看一下LORA是如何实现的。

在PEFT模块中，peft_model.py中的PeftModel类是一个总控类，用于模型的读取保存等功能，继承了transformers中的Mixin类，我们主要来看LORA的实现：

> 代码位置：https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py

```s
class LoraModel(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward
```

从构造方法可以看出，这个类在创建的时候主要做了两步：

- 第一步：self._find_and_replace()。找到所有需要加入lora策略的层，例如q_proj，把它们替换成lora模式；
- 第二步：mark_only_lora_as_trainable(self.model, self.peft_config.bias)。保留lora部分的参数可训练，其余参数全都固定下来不动；

### 3.2 LORA _find_and_replace() 实现思路

_find_and_replace() 实现思路：

1. 找到需要的做lora的层：

```s
  # 其中的target_modules在上面的例子中就是"q_proj"，"v_proj"
  # 这一步就是找到模型的各个组件中，名字里带"q_proj"，"v_proj"的
  target_module_found = re.fullmatch(self.peft_config.target_modules, key)
```

2. 对于每一个找到的目标层，创建一个新的lora层：

```s
  # 注意这里的Linear是在该py中新建的类，不是torch的Linear
  new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
```

3. 调用_replace_module方法替换掉原来的linear：

```s
  self._replace_module(parent, target_name, new_module, target)
```

> 注：其中这个replace的方法并不复杂，就是把原来的weight和bias赋给新创建的module，然后再分配到指定的设备上：

```s
    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
```

### 3.3 Lora层的实现 思路

#### 3.3.1 基类 LoraLayer 实现

Lora的基类，可以看出这个类就是用来构造Lora的各种超参数用：

```s
class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False
```

#### 3.3.2 Linear 实现

上文中所提到的Linear类，也就是Lora的具体实现，它同时继承了nn.Linear和LoraLayer：

```s
class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
```

在构造方法中，除了对各个超参数进行配置之外，还对所有参数进行了初始化，定义如下：

```s
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
```

> 其中lora的A矩阵采用了kaiming初始化，是Xavier初始化针对非线性激活函数的一种优化；B矩阵采用了零初始化，以确保在初始状态 ΔW=BA 为零。（值得注意的是在LORA的论文中，A采用的是Gaussian初始化）。

对于train和eval方法，放在一起介绍，它主要是需要对merge状态进行记录：

```s
    def train(self, mode: bool = True):
        # 对于新定义的这个Linear层，其本身继承了torch.nn.Linear，所以需要调用nn.Linear.train(self, mode)来控制一下自身原本参数的状态，并且此外它加入了lora_A和lora_B两部分额外的参数，这两部分本质上也是nn.Linear，也需要控制状态。
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)

        # not mode说明是eval模式
        # self.merge_weights在上文中有介绍，是配置文件中的，意思是评估时是否需要将lora部分的weight加到linear层原本的weight中
        # not self.merged是状态的记录
        if not mode and self.merge_weights and not self.merged:
            # 如果设置了需要融合，而当前状态没有融合的话，就把lora部分的参数scale之后加上去，并且更新self.merged状态
            if self.r > 0:
                self.weight.data += (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged:
            # 为了在训练的过程中，确保linear本身的weights是没有经过融合过的（理论上这一步应该是在eval之后的下一轮train的第一个step触发）
            if self.r > 0:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()
```

> 注：为什么是在train中涉及merge_weights，其实在torch的源码中，nn.Linear.eval()实际上是调用了nn.Linear.train(mode=False)，所以这里train方法中的merge_weigths，实际上是在eval中也发挥作用的。

forward中也是类似的原理，正常情况下训练过程应该是走elif的分支：

```s
    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
                self.merged = False

            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

```

## step 4：数据加载

```s
train_dataset = Seq2SeqDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.prompt_text)
train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=coll_fn,
                                  drop_last=False,
                                  num_workers=0)
```

对于Seq2SeqDataSet的实现，就是包括输入和label的构建这部分有很多细节的地方，在代码中添加了注释：

```s
class Seq2SeqDataSet(Dataset):
    """数据处理函数"""
    def __init__(self, data_path, tokenizer, max_len, max_src_len, prompt_text):
        # prompt_text = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
        # -3是因为需要拼接三个特殊字符[gMASK]、<sop>、<eop>
        max_tgt_len = max_len - max_src_len - 3
        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                # chatglm的token不是中文的字，是词
                # add_special_tokens = True时会在末位添加["[gMASK]", "<sop>"]
                src_tokens = tokenizer.tokenize(sample["text"])
                # print(sample["text"])
                # print(src_tokens)
                prompt_tokens = tokenizer.tokenize(prompt_text)
                # 根据限制的长度对输入进行截断
                if len(src_tokens) > max_src_len - len(prompt_tokens):
                    src_tokens = src_tokens[:max_src_len - len(prompt_tokens)]

                tgt_tokens = tokenizer.tokenize(sample["answer"])
                # 根据限制的长度对输入进行截断
                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                # 问、答之间需要通过特殊字符进行分割，同时需要添加终止符
                # [gMASK]与<sop>作为模型生成结果的起始标记，属于同一个block，
                # 所以这两个token对应的在原始文本中所在的位置是一样的，具体可参考这个issue https://github.com/THUDM/ChatGLM-6B/issues/1313
                # tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]
                tokens = prompt_tokens + src_tokens + [tokenizer.gmask_token, tokenizer.bos_token] + tgt_tokens + [tokenizer.eos_token]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                # prompt和问题部分不参与损失值计算
                labels = [-100] * context_length + input_ids[mask_position + 1:]
                # 根据最大长度进行后填充
                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                # 填充部分不参与损失值计算
                labels = labels + [-100] * pad_len
                # 区分有用的部分和填充的token
                attention_mask = []
                for input_id in input_ids:
                    if input_id != tokenizer.pad_token_id:
                        attention_mask.append(True)
                    else:
                        attention_mask.append(False)
                self.all_data.append(
                    {"text": sample["text"], "answer": sample["answer"], "input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


def coll_fn(batch):
    input_ids_list, labels_list, attention_mask_list = [], [], []
    for instance in batch:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
        attention_mask_list.append(torch.tensor(instance["attention_mask"]))
    # dataset中已经根据设定的长度进行了长度对齐
    # 3是[pad]这个特殊的token对应的token_id,不同版本可能不一样
    sample = {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=3),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100),
            # "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=False)
            }
    # items = {k: torch.tensor(v).to("cuda:1") for k, v in sample.items()}

    return sample
```

## step 5：定义优化器，并开始训练

```s
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_train_epochs),
    )

for i_epoch in range(args.num_train_epochs):
    model.train()
    train_iter = iter(train_dataloader)
    for step, batch in enumerate(train_iter):
        # 数据和模型使用相同gpu
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

## step 6：模型保存

```s
    # 注意这里的模型保存方式，peft重写了model的save_pretrained方法，这里只把lora层的权重进行存储
    # 如果是用trainer进行训练，需要注意对模型保存的方法进行重写只保存lora的参数，具体参考：https://cloud.tencent.com/developer/article/2276508
    model.save_pretrained(args.output_dir)
    copy(os.path.join(args.model_dir, "tokenizer_config.json"), os.path.join(args.output_dir, "tokenizer_config.json"))
    copy(os.path.join(args.model_dir, "tokenization_chatglm.py"),
         os.path.join(args.output_dir, "tokenization_chatglm.py"))
    copy(os.path.join(args.model_dir, "ice_text.model"), os.path.join(args.output_dir, "ice_text.model"))
```

## step 7：推理前的Weight合并

```s
original_model_dir = "models/chatglm-6b/"
lora_model_dir = "ChatGLM6b_finetuning/output_dir_lora/"  # 这是前面训练时的输出路径
model = AutoModel.from_pretrained(original_model_dir, trust_remote_code=True).half()
tokenizer = AutoTokenizer.from_pretrained(original_model_dir, trust_remote_code=True)

## 用来检查权重是否合并成功，合并成功weight会改变
first_weight = model.base_model.layers[0].attention.query_key_value.weight
first_weight_old = first_weight.clone()
lora_model = PeftModel.from_pretrained(model, lora_model_dir)
# # 报错：A*B shape mismatch，大概率是get_peft_model错误修改了peft_config里面的fan_in_fan_out参数，某个peft的revision有这个bug
lora_model = lora_model.merge_and_unload()
lora_model.train(False)
# 验证weight值是否改变了
# # 报错：大概率peft训练有问题，检查adapter.bin大小
assert not torch.allclose(first_weight_old, first_weight), 'Weight Should Change after Lora Merge'
# # lora模型权重把原模型权重加了prefix，这里移除恢复原始key
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model.state_dict().items()
    if "lora" not in k
}
# 保存合并后的模型权重
output_dir = "output_dir_lora_merge"
os.makedirs(output_dir, exist_ok=True)
lora_model.save_pretrained(output_dir, state_dict=deloreanized_sd, max_shard_size="4GB")
```

## step 8：加载合并后的模型进行推理

```s
original_model_dir = "models/chatglm-6b/" # 原始模型路径，主要是加载对应的tokenizer，把相关文件复制到和新模型一个目录也行
lora_model_dir = "output_dir_lora_merge"  # 上一步合并后的模型输出路径
model = AutoModel.from_pretrained(lora_model_dir, trust_remote_code=True).half()
tokenizer = AutoTokenizer.from_pretrained(original_model_dir, trust_remote_code=True)
device = torch.device("cuda:1")
model = model.to(device)
text = "Hello, my name is "
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=30, top_p=0.85)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


## 参考

1. ChatGLM-6b通过PEFT进行Lora的高效参数微调 https://zhuanlan.zhihu.com/p/641889757
2. 大模型训练——PEFT与LORA介绍 https://blog.csdn.net/weixin_44826203/article/details/129733930
3. LLM Parameter-Efficient 训练方案梳理 https://zhuanlan.zhihu.com/p/624935413
4. LORA微调系列(一)：LORA和它的基本原理 https://zhuanlan.zhihu.com/p/646791309
5. 2023年的深度学习入门指南(12) - PEFT与LoRA https://juejin.cn/post/7230415879882850362
6. 如何使用 PETF/LoRA 封装LLM https://zhuanlan.zhihu.com/p/662103844
7. LLM - LoRA 模型合并与保存 https://blog.csdn.net/BIT_666/article/details/132065177
8. 代码位置：https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py


