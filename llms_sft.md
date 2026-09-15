# 大模型从 0 到 1 训练 面试常考题篇

## 常考题

- [RL为什么会训崩？](https://articles.zsxq.com/id_l492uajawq66.html)
- [介绍一下 SFT 的 loss 如何设计？](https://articles.zsxq.com/id_4dwsr5plfj8w.html)
- [RL 为什么不如 SFT稳定？](https://articles.zsxq.com/id_84tm2kmasod5.html)
- [面试官问：为什么RL训练不看loss大小，而SFT就需要?](https://articles.zsxq.com/id_hpqovqvwa6wk.html)
- [如果构建行业垂直大模型，到底是用RAG还是微调？](https://articles.zsxq.com/id_u4w09pxrg76h.html)
- [面试官问：从零到一介绍一下大模型训练流程?](https://articles.zsxq.com/id_41qhc6w2gpnz.html)
- [微调大模型之前，你的数据真的干净吗？](https://articles.zsxq.com/id_jdw7aqunebiz.html)
- [面试官：模型如何在指令微调过程中构造或筛选高质量数据？](https://articles.zsxq.com/id_twtgholu6cls.html)
- [🔥 SFT和RL在后训练中哪个更容易导致灾难性遗忘？](https://articles.zsxq.com/id_brpa2rlo0eka.html)

## 一、大模型（LLMs）预训练篇 :fire:

### [大模型持续预训练（Continued Pre-training）](https://articles.zsxq.com/id_qfce5k3e1gqj.html) :fire:

- 一、什么情况下LLM需要持续预训练？
  - 1.1 领域知识注入 (Domain Adaptation)
  - 1.2 知识更新 (Knowledge Cutoff Update)
  - 1.3 新能力/语言/风格的掌握 (Acquiring New Skills/Languages/Styles)
  - 1.4 模型架构调整后的再适应 (Adapting to Architectural Changes)
- 二、持续预训练 vs. 指令微调 (SFT)：一个核心区别
- 三、LLM持续微调的过程面临什么问题？
  - 1、灾难性遗忘 (Catastrophic Forgetting)
  - 2、数据质量与管理
  - 3、评估的复杂性与“跷跷板效应”
  - 4、高昂的计算与资源成本
  - 5、对齐与安全性的漂移
  - 6、知识能力错配

### [从0到1：揭秘LLM预训练前的海量数据清洗全流程](https://articles.zsxq.com/id_4c43qgnhwujs.html) :fire:

- 前言
- 一、为什么要进行数据清洗？——“垃圾进，垃圾出”的铁律
- 二、llm-from-scratch中的数据炼金术：清洗流程概览
  - 第一站：HTML处理 —— 从网页源码中提取纯净文本
  - 第二站：语言识别 —— 在万国语料中找到“通用语”
  - 第三站：质量过滤 —— 用启发式规则筛掉“坏品味”内容
  - 第四站：去重处理 —— 识别并消除“重复的旋律”
    - 4.1 精确打击：行级别去重
    - 4.2 智能识别：MinHash与LSH算法
- 第五站：PII屏蔽 —— 为个人隐私信息“打上马赛克”
- 第六站：有害内容检测 —— 拦截“精神毒药”
- 终点站：高质量分类器 —— 用AI为内容“画龙点睛”
- 结论：数据决定上限，模型决定逼近上限的程度

### [大模型（LLMs）增量预训练篇](https://articles.zsxq.com/id_jfq8la7g20ww.html)

1. 为什么要增量预训练？
2. 进行 增量预训练 需要做哪些准备工作？
3. 增量预训练 所用 训练框架？
4. 增量预训练 训练流程 是怎么样？
5. 增量预训练 一般需要多大数据量？
6. 增量预训练 过程中，loss 上升正常么？
7. 增量预训练 过程中，lr 如何设置？
8. 增量预训练 过程中，warmup\_ratio 如何设置？
9. warmup 的步数 对 大模型继续预训练 是否有影响？
10. 学习率 大小 对 大模型继续预训练 后 上下游任务影响？
11. 在初始预训练中使用 Rewarmup 对 大模型继续预训练 性能 影响？

- [点击查看答案](https://articles.zsxq.com/id_jfq8la7g20ww.html)

### [增量预训练（Pretrain）样本拼接篇](https://articles.zsxq.com/id_8f35p8piwl4v.html)

- 一、 推理过程 分哪些阶段？
  - 1.1 Prefill（输入理解与初始化）阶段
  - 1.2 Decoding（递归推理与解码输出）阶段
- 二、推理性能的评价指标？
  - 2.1 Throughput（吞吐量）
  - 2.2 First Token Latency（首字延迟）
  - 2.3 Latency（延迟）
  - 2.4 QPS（每秒请求数）
- 三、当前优化模型最主要技术手段有哪些？
  - 3.1 KVCache
  - 3.2 分布式推理
  - 3.3 流水线处理
  - 3.4 动态批处理
  - 3.5 低比特量化
  - 3.6 System Prompt
  - 3.7 预测生成长度
  - 3.8 Flash Attention
  - 3.9 Paged Attention
  - 3.10 精简Attention的MHA\\GQA\\MQA技术
  - 3.11 选择合适的硬件
- 四、推理加速框架有哪一些？都有什么特点？
- 五、vLLM 篇
  - 5.1 vLLM 的 功能有哪些？
  - 5.2 vLLM 的 优点有哪些？
  - 5.3 vLLM 的 缺点有哪些？
  - 5.4 vLLM 离线批量推理？
  - 5.5 vLLM API Server？
- 六、Text generation inference 篇
  - 6.1 介绍一下 Text generation inference？
  - 6.2 Text generation inference 的 功能有哪些？
  - 6.3 Text generation inference 的 优点有哪些？
  - 6.4 Text generation inference 的 缺点有哪些？
  - 6.5 Text generation inference 的 使用docker运行web server？

- [点击查看答案](https://articles.zsxq.com/id_8f35p8piwl4v.html)

### [增量预训练（Pretrain）样本拼接篇](https://articles.zsxq.com/id_enteq22h1nhq.html)

- 一、Pretrain阶段，为什么需要拼接拼接？
- 二、有哪些 拼接方式？
  - 2.1 拼接方式一：Random Concatenate
  - 2.2 拼接方式二：Random Concatenate + NoiseMask
  - 2.3 拼接方式三：Random Concatenate + Cluster
  - 2.4 拼接方式四：IN-CONTEXT PRETRAINING

- [点击查看答案](https://articles.zsxq.com/id_enteq22h1nhq.html)

### [基于lora的llama2二次预训练](https://articles.zsxq.com/id_xo09u14omdjw.html)

- 一、为什么需要 对 llama2 做 基于lora的二次预训练?
- 二、基于lora的llama2二次预训练 的目标是什么？
- 三、基于lora的llama2二次预训练 的思想是什么？
- 四、基于lora的llama2二次预训练 语料构建思路？
- 五、如何 基于lora的llama2二次预训练 ？
  - 5.1 基于lora的llama2二次预训练 参数介绍
  - 5.2 基于lora的llama2二次预训练
- 六、如何 基于lora的llama2 微调 ？
  - 6.1 训练数据介绍
  - 6.2 基于lora的llama2 微调 参数介绍
  - 6.3 基于lora的llama2 微调
- 七、如何 使用 基于lora的llama2 做推理 ？

- [点击查看答案](https://articles.zsxq.com/id_xo09u14omdjw.html)


## 二、大模型（LLMs）训练集面 

### [大模型（LLMs）训练集面](https://articles.zsxq.com/id_axtljtl0bsvw.html)

1. SFT（有监督微调）的数据集格式？
2. RM（奖励模型）的数据格式？
3. PPO（强化学习）的数据格式？
4. 找数据集哪里找？
5. 微调需要多少条数据？
6. 有哪些大模型的训练集？
7. 进行领域大模型预训练应用哪些数据集比较好？
8. 如何选取和构建大模型微调数据？

- [点击查看答案](https://articles.zsxq.com/id_axtljtl0bsvw.html)

### [大模型（LLMs）LLM生成SFT数据方法面](https://articles.zsxq.com/id_1gzdghj84f9f.html)

- 四、大模型微调数据集格式篇
- 一、SFT数据集如何生成？
- 二、Self-Instruct 篇
  - 2.1 什么是 Self-Instruct ？
  - 2.2 Self-Instruct 处理思路？
- 三、Backtranslation 篇
  - 3.1 什么是 Backtranslation？

- [点击查看答案](https://articles.zsxq.com/id_1gzdghj84f9f.html)

## 三、大模型（LLMs）微调面

### [大模型（LLMs）微调面](https://articles.zsxq.com/id_kv7jdah2zw5n.html) 

- 39 大模型 sft 过程中，为什么会出现第二个epoch的时候loss会突然下降问题？
- 1 如果想要在某个模型基础上做全参数微调，究竟需要多少显存？
- 2 为什么SFT之后感觉LLM傻了?
- 3 SFT 指令微调数据 如何构建?
  - 3.1 提升sft的prompt的代表性有什么好的方法？
  - 3.2 提升sft的prompt的数据量有什么好的方法？
- 4 领域模型Continue PreTrain 数据选取？
- 5 领域数据训练后，通用能力往往会有所下降，如何缓解模型遗忘通用能力？
- 6 领域模型Continue PreTrain ，如何 让模型在预训练过程中就学习到更多的知识？
- 7 进行SFT操作的时候，基座模型选用Chat还是Base?
- 8 领域模型微调 指令\&数据输入格式 要求？
- 9 领域模型微调 领域评测集 构建？
- 10 领域模型词表扩增是不是有必要的？
- 11 如何训练自己的大模型？
- 12 训练中文大模型有啥经验？
- 13 指令微调的好处？
- 14 预训练和微调哪个阶段注入知识的？
- 15 想让模型学习某个领域或行业的知识，是应该预训练还是应该微调？
- 16 多轮对话任务如何微调模型？
- 17 微调后的模型出现能力劣化，灾难性遗忘是怎么回事？
- 18 微调模型需要多大显存？
- 19 大模型LLM进行SFT操作的时候在学习什么？
- 20 预训练和SFT操作有什么不同
- 21 样本量规模增大，训练出现OOM错
- 22 大模型LLM进行SFT 如何对样本进行优化？
- 23 模型参数迭代实验
- 24 微调大模型的一些建议
- 25 微调大模型时，如果 batch size 设置太小 会出现什么问题？
- 26 微调大模型时，如果 batch size 设置太大 会出现什么问题？
- 27  微调大模型时, batch size 如何设置问题？
- 28 微调大模型时, 优化器如何？
- 29 哪些因素会影响内存使用？
- 30 进行领域大模型预训练应用哪些数据集比较好？
- 31 用于大模型微调的数据集如何构建？
- 32 大模型训练loss突刺原因和解决办法
  - 32.1 大模型训练loss突刺是什么？
  - 32.2 为什么大模型训练会出现loss突刺？
  - 32.3 大模型训练loss突刺 如何解决？
- 33 什么是Cosine优化器？在大模型中应该怎么设置cosine优化器的周期比较好？
- 34 在预训练阶段，若模型训练文本时不包含标记，而在后续预测阶段却添加了标记；或者相反，训练阶段加入了标记，但在预测时却没有使用，这两种情况下 benchmark 预测可能会遇到哪些问题呢？
- 35 SFT packing是什么？
- 36 SFT packing对SFT训练的影响是什么？
- 37 SFT阶段模型可以学习新知识么？
- 38 建立sft数据主要需要关注什么方面？
- 40 解决显存不够的方法？
- 41 指令策略的选择及其影响
- 42 如何解决prompt泛化性？
- 43 大模型SFT最重要的是什么，分次SFT会发生什么?
- 44 大模型 supervised fine-tuning(SFT)的方法主要分为哪些?

- [点击查看答案](https://articles.zsxq.com/id_kv7jdah2zw5n.html)

### [大模型 SFT Trick 篇](https://articles.zsxq.com/id_srd92pvnjwmu.html)

- 一、常见 SFT的开发流程是如何的？
- 二、训练数据要注重什么？
- 三、大 size 和小 size 模型的选择？
- 四、多任务训练时怎么确保每个任务都优秀？
- 五、SFT真的不能学到知识？
- 六、怎么科学挑选数据集？
- 七、怎么解决幻觉问题
- 八、BERT 开发与 LLM 开发有什么不同之处？
- 九、该选什么微调方法， Full tuning\\P-tuning\\Lora?
- 十、SFT 还有什么方面值得研究？
- 十一、介绍一下训练过程中，显存占用分析？
- 十二、介绍一下训练过程中，显存占用分析？
- 十三、训练数据数据质量评估
- 十四、SFT 调参技巧
  - 14.1 有哪些参数可以调呢?
  - 14.2 Loss function 如何调参？
  - 14.3 Learning rate 和 Batch size 如何调参？
  - 14.4 Epoch number 和 early stopping 如何调参？
  - 14.5 Optimizer 如何调参？
  - 14.6 Activation function 如何调参？
  - 14.7 Weights initialization 如何调参？
  - 14.8 Regularization 如何调参？

- [点击查看答案](https://articles.zsxq.com/id_srd92pvnjwmu.html)

### [大模型（LLMs）训练经验帖](https://articles.zsxq.com/id_06n25d9wjs0e.html)

- 分布式训练框架选择？
- LLMs 训练时 有哪些有用的建议？
- 模型大小如何选择？
- 加速卡如何选择？

- [点击查看答案](https://articles.zsxq.com/id_06n25d9wjs0e.html)

## 四、大模型（LLMs）评测面

### [大模型（LLMs）评测面](https://articles.zsxq.com/id_j9wcj62eovgc.html)

- 1 大模型怎么评测？
- 2 大模型的honest原则是如何实现的？模型如何判断回答的知识是训练过的已知的知识，怎么训练这种能力？
- 3 如何衡量大模型水平？
- 4 大模型评估方法 有哪些？
- 5 大模型评估工具 有哪些？
- 6介绍一下 困惑度？
- 7 大模型训练的性能指标:吞吐率 Throughput 是指什么?
- 8 如何评价大语言模型的训练数据的质量?

- [点击查看答案](https://articles.zsxq.com/id_j9wcj62eovgc.html)

### [大模型自动评估理论和实战](https://articles.zsxq.com/id_b2gh8asr8xqw.html)

- 引言
- 一、LLM评估的方法论
  - 1.1 如何评估一个LLM
  - 1.2 自动评估方法
    - 1.2.1 模型效果评估
    - 1.2.2 Rule-based自动评测
    - 1.2.3 Model-based自动评测
    - 1.2.4 模型性能评估
  - 1.3 问题和挑战
    - 1.3.1 基准失效&数据泄露
    - 1.3.2 裁判员模型的能力上限
- 二、LLM评估实战
  - 2.1 框架特性
  - 2.2 环境安装
  - 2.3 简单评测
  - 2.4 带参数评测
  - 2.5 竞技场模式--Single mode
  - 2.6 竞技场模式--Pairwise mode
  - 2.7 效果评测报告
  - 2.8 模型性能评测（Perf Eval）

## 五、大模型（LLMs）分布式训练面 

### [大模型（LLMs）分布式训练面](https://articles.zsxq.com/id_ah2ibj3z22c7.html)

- 理论篇
  - 1.1 训练 大语言模型 存在问题？
  - 1.2 什么是 点对点通信？
  - 1.3 什么是 集体通信？
  - 1.4 什么是 数据并行？
  - 1.5 数据并行 如何 提升效率？
  - 1.6 什么是 流水线并行？
  - 1.7 什么是 张量并行 (intra-layer)？
  - 1.8 数据并行 vs 张量并行 vs 流水线并行?
  - 1.9 什么是 3D并行？
  - 1.10 想要训练1个LLM，如果只想用1张显卡，那么对显卡的要求是什么？
  - 1.11 如果有N张显存足够大的显卡，怎么加速训练？
  - 1.12 如果显卡的显存不够装下一个完整的模型呢？
  - 1.13 PP推理时，是一个串行的过程，1个GPU计算，其他空闲，有没有其他方式？
  - 1.14 3种并行方式可以叠加吗？
  - 1.15 Colossal-AI 有1D/2D/2.5D/3D，是什么情况？
  - 1.16 除了3D并行有没有其他方式大规模训练？
  - 1.17 有了ZeRO系列，为什么还需要3D并行？
  - 1.18 平民适不适合玩3D并行？
  - 1.19 平民适不适合直接上多机多卡的ZeRO3（万兆网）？
  - 1.20 分布式并行及显存优化技术并行技术有哪一些，都有什么特点？
  - 1.21 显存优化技术有哪一些，都有什么特点？
  - 1.22 常见的分布式训练框架哪一些，都有什么特点？
- 实践篇
  - 2.1 假如有超多的8卡A100节点（DGX A100），如何应用3D并行策略？
  - 2.2 如果想构这样一个大规模并行训练系统，训练框架如何选？
  - 2.3 训练框架如何选？
- 并行化策略选择篇
  - 3.1 如何选择一款分布式训练框架？
  - 3.2 如何选择一款分布式训练框架？
  - 3.3 单GPU
  - 3.4 单节点多卡
  - 3.5 多节点多卡
- 问题篇
  - 4.1 推理速度验证
  - 4.2 并行化训练加速
  - 4.3 deepspeed 训练过程，报找不主机
  - 4.4 为什么 多机训练效率不如单机？
  - 4.5 多机训练不通，DeepSPeed配置问题

- [点击查看答案](https://articles.zsxq.com/id_ah2ibj3z22c7.html)

### [图解分布式训练（一） —— 流水线并行（Pipeline Parallelism）面](https://articles.zsxq.com/id_wre1eni0oq7d.html)

- 为什么需要流水线并行（Pipeline Parallelism）？
- 一、流水线并行（Pipeline Parallelism） 优化目标是什么？
- 二、图解 流水线并行（Pipeline Parallelism）模型并行 必要性？
- 三、流水线并行（Pipeline Parallelism） 图解？
- 四、流水线并行（Pipeline Parallelism）优缺点？

- [点击查看答案](https://articles.zsxq.com/id_wre1eni0oq7d.html)

### [图解分布式训练（二） —— nn.DataParallel面](https://articles.zsxq.com/id_9dfwi0ooio2z.html)

- 为什么需要nn.DataParallel？
- 一、pytorch中的GPU操作默认是什么样？
- 二、介绍一下 nn.DataParallel 函数？
- 三、nn.DataParallel 函数 处理逻辑 介绍一下？
- 四、nn.DataParallel 函数 常见问题及解答 有哪些？
  - 4.1 多GPU计算减少了程序运行的时间？
  - 4.2 如何保存和加载多GPU训练模型呢？
  - 4.3 为什么第一块卡的显存会占用的更多一些？
  - 4.4 直接使用nn.DataParallel的时候，训练采用多卡训练，会出现一个warning？
  - 4.5 device\_ids 0 被占用问题
- 五、nn.DataParallel 函数 参数更新方式 ？
- 六、nn.DataParallel 函数 优点 介绍一下？
- 七、nn.DataParallel 函数 缺点 介绍一下？
- 八、nn.DataParallel 函数 实战？

- [点击查看答案](https://articles.zsxq.com/id_9dfwi0ooio2z.html)

### [图解分布式训练（三） ——  nn.parallel.DistributedDataParallel](https://articles.zsxq.com/id_i4s3ia057rmh.html)

- 为什么需要 nn.parallel.DistributedDataParallel ？
- 一、什么是 DistributedDataParallel 核心 —— Ring-AllReduce？
- 二、nn.parallel.DistributedDataParallel 函数 介绍一下？
- 三、nn.parallel.DistributedDataParallel 函数 如何多卡加速训练？
- 四、nn.parallel.DistributedDataParallel 实现流程介绍一下？
- 五、nn.parallel.DistributedDataParallel 参数更新介绍一下？
- 六、nn.DataParallel(以下简称DP) vs DistributedDataParallel(以下简称DDP)介绍一下？
- 七、DistributedDataParallel(以下简称DDP) 优点有哪些？
- 八、DistributedDataParallel(以下简称DDP) 缺点有哪些？

- [点击查看答案](https://articles.zsxq.com/id_i4s3ia057rmh.html)

### [图解分布式训练（四） ——  torch.multiprocessing 详细解析](https://articles.zsxq.com/id_gu9smpbn510e.html)

- 一、torch.multiprocessing 函数介绍一下？
- 二、torch.multiprocessing 函数如何使用？
- 三、介绍一下 共享CUDA张量？
- 四、介绍一下 共享策略？
- 五、torch.multiprocessing 函数使用

- [点击查看答案](https://articles.zsxq.com/id_gu9smpbn510e.html)

### [图解分布式训练（五） ——  AMP混合精度训练 详细解析](https://articles.zsxq.com/id_0slrgoti6gvb.html)

- 为什么需要 AMP混合精度训练？
- 一、什么是自动混合精度训练(AMP)
- 二、为什么需要自动混合精度？
- 三、混合精度训练的优点是什么？
- 四、混合精度训练的缺点是什么？
- 五、混合精度训练的关键技术是什么？
- 六、介绍一下 混合精度训练 动态损失缩放？
- 七、如何在PyTorch中使用自动混合精度？
- 八、如何使用 AMP混合精度训练 ？

- [点击查看答案](https://articles.zsxq.com/id_0slrgoti6gvb.html)

### [图解分布式训练（六） —— Pytorch的 DeepSpeed 详细解析](https://articles.zsxq.com/id_kmq9rn2vo4kz.html)

- 一、为什么需要 Deepspeed？
- 二、DeepSpeed 基本概念 介绍一下？
  - 2.1 DeepSpeed 介绍
  - 2.2 DeepSpeed 基础的概念
  - 2.3 DeepSpeed 支持的功能
- 三、DeepSpeed 通信策略 介绍一下？
- 四、DeepSpeed 如何使用？
  - 4.1 DeepSpeed 安装
  - 4.2 DeepSpeed 使用
- 五、DeepSpeed 全部代码
- 六、优化器和调度器
  - 6.1 优化器
  - 6.2 调度器
- 七、训练精度
  - 7.1 自动混合精度
  - 7.2 NCCL
  - 7.3 apex
- 八、获取模型参数
  - 8.1 ZeRO-3 and Infinity Nuances
- 填坑笔记
  - 1. ModuleNotFoundError: No module named 'torch._six
  - 2. 为什么单卡的情况，也可以使用deepspeed？
  - 3. 不同 ZeRO 如何配置
  - 4. ZeRO-3 会比 ZeRO-2 慢很多 如何优化？
  - 5. 如何选择不同的Zero stage和offload
  - 6. DeepSpeed 遇到问题，如何 确定 调参步骤？
  - 7. 如何估算需要的显存？
  - 8. 启动时，进程被杀死，并且没有打印出traceback
  - 9. loss是NaN
  - 10. 确保一致性
  - 11. 如何配置 配置ssh？
  - 12. 如何配置 安装pdsh？
  - 12. 如何配置 配置deepspeed文件？

- [点击查看答案](https://articles.zsxq.com/id_kmq9rn2vo4kz.html)

### [图解分布式训练（七）—— accelerate 分布式训练 详细解析](https://articles.zsxq.com/id_o5wkeionnqr7.html)

- 一、为什么需要 accelerate 分布式训练？
- 二、什么是 accelerate 分布式训练?
- 三、accelerate 分布式训练 原理讲解？
- 四、accelerate 分布式训练 如何实践？

- [点击查看答案](https://articles.zsxq.com/id_o5wkeionnqr7.html)

### [图解分布式训练（八）—— ZeRO 学习](https://articles.zsxq.com/id_grv7uddls2g1.html)

- 一、什么是 3D 并行？
- 二、3D 并行 策略有哪些？
- 三、为什么需要 ZeRO？
- 四、ZeRO 的 核心思想是什么？
- 五、ZeRO 显存如何分配？
- 六、ZeRO 优化策略是怎么样？
- 七、ZeRO Offload后的计算流程是怎么样？

- [点击查看答案](https://articles.zsxq.com/id_grv7uddls2g1.html)

### [大模型分布式训练故障恢复篇](https://articles.zsxq.com/id_zspm2q33tckx.html)

- 一、为什么 大模型分布式训练 需要 故障恢复？
- 二、如何获取最优的ckpt存储间隔？
- 三、ckpt存储能否实现异步或者部分掩盖？
- 四、断点续训/临终遗言是否真实可行？

- [点击查看答案](https://articles.zsxq.com/id_zspm2q33tckx.html)

### [图解分布式训练（九）—— Megatron-LM 篇](https://articles.zsxq.com/id_o4qtcspmuwqv.html)

- 1、Activation Recomputation是怎么实现的?
- 2、Megatron中的OverlappedDistributed Optimizer 是如何实现的?
- 3、Megatron-LM 中 Context Parallel 篇
  - 3.1 介绍一下 Megatron-LM 中 Context Parallel 实现原理？
  - 3.2 介绍一下 Megatron-LM 中 Context Parallel 实现步骤？
  - 3.3 cp依然是FA的计算逻辑，为何分块FA计算为何需要修正？
  - 3.4 cp介绍里面说，相比ring-attention增加了负载均衡的处理逻辑，如何实现的？
  - 3.5 cp的性能如何？相比其它序列并行优劣如何？

- [点击查看答案](https://articles.zsxq.com/id_o4qtcspmuwqv.html)

### [分布式训练 Trick 汇总篇](https://articles.zsxq.com/id_fu9065izm2m4.html)

- 一、数据并行 Trick 篇
  - 1.1 数据并行 FSDP
  - 1.2 数据并行 DDP
  - 1.3 数据并行 ZeRO
    - 1.3.1 Model state
    - 1.3.2 Residual state
    - 1.3.3 offload
- 二、模型并行 Trick 篇
  - 2.1 tensor-wise parallelism
  - 2.2 pipeline paralelism
  - 2.3 equence parallelism
  - 2.4 layer-wise parallelism
- 三、MoE Trick 篇

- [点击查看答案](https://articles.zsxq.com/id_fu9065izm2m4.html)

### [pytorch 分布式计算 坑/bug 梳理篇](https://articles.zsxq.com/id_onztfzwdckom.html)

- 一、使用 DistributedDataParallel（分布式并行）时，显存分布不均衡问题
- 二、如果是用pytorch实现同步梯度更新，自研 数据接口，出现 第一个epoch结尾处程序卡死问题
- 三、在微调大模型的时候，单机2卡的时候正常训练，但是采用4卡及其以上，就会卡住，卡在读完数据和开始训练之间？

- [点击查看答案](https://articles.zsxq.com/id_onztfzwdckom.html)

## 六、大模型蒸馏篇 :fire:

### [大模型知识蒸馏技术方法（DeepSeek， Llama 4 \& Gemma 3中使用的技术）](https://articles.zsxq.com/id_w0dgjdlhocee.html)

- 一、前言
- 二、大模型知识蒸馏技术方法有哪些方法?
- 三、大模型知识蒸馏技术方法的核心思想是什么?
- 四、大模型知识蒸馏技术方法一般在哪个阶段?
- 五、大模型常用的蒸馏技术?
  - 5.1 软标签蒸馏（Soft-label Distillation）
    - 5.1.1 软标签蒸馏（Soft-label Distillation）思路
    - 5.1.2 软标签蒸馏（Soft-label Distillation）存在问题
  - 5.2 硬标签蒸馏（Hard-label Distillation）
    - 5.2.1 硬标签蒸馏（Hard-label Distillation）思路
  - 5.3 协同蒸馏（Co-distillation）
    - 5.3.1 协同蒸馏（Co-distillation）思路

### [大模型蒸馏篇](https://articles.zsxq.com/id_jkiw9vhzopgv.html)

- 一、知识蒸馏和无监督样本训练？
- 二、对知识蒸馏知道多少，有哪些改进用到了？
- 三、谈一下对模型量化的了解？
- 四、模型压缩和加速的方法有哪些？
- 五、你了解的知识蒸馏模型有哪些？
- 六、模型蒸馏是怎么做的

- [点击查看答案](https://articles.zsxq.com/id_jkiw9vhzopgv.html)

### [LLMs 浮点数篇](https://articles.zsxq.com/id_vu744g6jklli.html) 

- 一、fp32和fp16的区别，混合精度的原理
- 二、半精度是什么？
- 三、半精度的理论原理是什么？

- [点击查看答案](https://articles.zsxq.com/id_vu744g6jklli.html)

### [自定义 CUDA 函数的轻量级包装器 —— bitsandbytes篇](https://articles.zsxq.com/id_2nwi4napgvlh.html) 

- 一、什么是 bitsandbytes?
- 二、如何才能使用 bitsandbytes？
- 三、如何使用 bitsandbytes？

- [点击查看答案](https://articles.zsxq.com/id_2nwi4napgvlh.html)

## 七、[大模型 Loss 篇](https://articles.zsxq.com/id_x6gs0l0qbxi0.html) :fire:

- 引言
- 一、基础回顾：自回归、右移标签与 NLL
- 二、预训练模式（LM Loss：全监督）
- 三、单轮 SFT（只监督回答区域）
- 四、多轮 SFT（chat template：多段回答的并集）
- 五、对齐要点（防止 off-by-one）
- 六、常见错误与排查
- 七、小结

## 八、[大模型推理、监督微调、GRPO显存消耗 篇](https://articles.zsxq.com/id_9z45vwhwufml.html) :fire:

- 大模型推理、微调，需要消耗多少显存？
  - Qwen3-8B为例
    - Qwen3-8B 推理所需显存
    - Qwen3-8B 训练峰值显存
  - Qwen2.5-1.5B为例
    - Qwen2.5-1.5B 推理所需显存
    - Qwen2.5-1.5B 训练峰值显存
- 针对不同训练过程，显存消耗有什么差异？
  - 1、参数高效微调PEFT (Parameter-Efficient Fine-Tuning)
  - 2、直接偏好优化DPO (Direct Preference Optimization)
  - 3、近端策略优化PPO (Proximal Policy Optimization)
  - 4、PPO算法（RFT版本）
  - 5、广义奖励排序偏好优化GRPO (Generalized Reward-Ranked Preference Optimization)

- [点击查看答案](https://articles.zsxq.com/id_9z45vwhwufml.html)

## 九、LLM 训练 Trick 篇 :fire:

### [LLM 训练 Trick 篇](https://articles.zsxq.com/id_r1kwk4aw9ph1.html)

- 引言
- 一、传统方法的痛点：为何 concat-and-chunk 效率低下？
- 二、解决方案：DD + VSL 的双重创新
  - 2.1 数据集分解（Dataset Decomposition, DD）
  - 2.2 可变序列长度训练（Variable Sequence Length Training, VSL）
- 三、 实验分析与深度洞察
  - 3.1 训练效率与成本
  - 3.2 序列长度对任务的归纳偏置
  - 3.3 课程学习与稳定性
  - 3.4 其他技术细节
- 总结

- [点击查看答案](https://articles.zsxq.com/id_r1kwk4aw9ph1.html)

