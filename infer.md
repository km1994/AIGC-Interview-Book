# 大模型推理加速面试常考题篇

## 参考题

- [如何减少大模型 token 的消耗？](https://articles.zsxq.com/id_hw758oi8s4aa.html)
- [同一个prompt重复多次输入进 LLM，为啥得到的输出都不一样？](https://articles.zsxq.com/id_tq03cusn77vb.html)

## 一、大模型（LLMs）推理面

### [大模型（LLMs）推理加速篇](https://articles.zsxq.com/id_kgzsxgro8cee.html)

- 一、推理过程 分哪些阶段？
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
  - 3.10 精简Attention的MHA\GQA\MQA技术
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

- [点击查看答案](https://articles.zsxq.com/id_kgzsxgro8cee.html)

### [LLMs 推理性能面](https://articles.zsxq.com/id_jwd03u0l7feo.html)

- 一、介绍一下 LLMs 的文本生成过程？
- 二、如何准确衡量模型的推理速度呢？
- 三、如果对整体推理时延有具体目标，有哪些有效的启发式方法来评估模型？
- 四、LLMs 推理存在哪些挑战？

- [点击查看答案](https://articles.zsxq.com/id_jwd03u0l7feo.html)

### [大模型（LLMs）推理面](https://articles.zsxq.com/id_b9eecaoga75i.html)

1. 为什么大模型推理时显存涨的那么多还一直占着？
2. 大模型在gpu和cpu上推理速度如何？
3. 推理速度上，int8和fp16比起来怎么样？
4. 大模型有推理能力吗？
5. 大模型生成时的参数怎么设置？
6. 有哪些省内存的大语言模型训练/微调/推理方法？
7. 如何让大模型输出合规化
8. 应用模式变更
9. 模型输出的分布比较稀疏，怎么处理？

- [点击查看答案](https://articles.zsxq.com/id_b9eecaoga75i.html)

## 二、大模型——推理大模型面 :fire:

### [大模型——推理大模型面](https://articles.zsxq.com/id_ofc8lbfxntit.html) :fire:

- 一、什么是思维链？
- 二、什么是推理大模型？
- 三、什么是推理？
- 四、相比于普通大模型，推理大模型有什么优点？
- 五、什么时候适合使用推理大模型？
- 六、推理大模型的推理过程是否是必须生成？
- 七、如何训练推理大模型？
  - 7.1 什么是 推理时扩展（Inference-time scaling）？
  - 7.2 什么是 纯强化学习（Pure reinforcement learning, RL）？
  - 7.3 什么是 监督微调与强化学习结合（Supervised fine-tuning and reinforcement learning, SFT + RL）？
  - 7.4 什么是 纯监督微调与蒸馏（Pure supervised fine-tuning and distillation）？
- 八、推理大模型和普通大模型的简单对比？
- 九、推理大模型的提示词（prompt）如何写？

[点击查看答案](https://articles.zsxq.com/id_ofc8lbfxntit.html)

## 三、大模型（LLMs）解码策略篇  :fire:

### [LLM输出质变！ 一个简单技巧提升了200%](https://articles.zsxq.com/id_w6w7te7tvvts.html)

- 一、问题：当LLM忽视你告诉它的内容
- 二、什么是激活调整（Activation Steering）？
- 三、如何“调整”词嵌入？
- 四、如何“调整”LLM行为？
- 五、为什么调整如此重要？
- 六、如何使用调整来减少上下文幻觉

- [点击查看答案](https://articles.zsxq.com/id_w6w7te7tvvts.html)

### [大模型-packing](https://articles.zsxq.com/id_cu81jkabtar9.html)

- 一、什么是 packing？
- 二、介绍一下 Packing 的核心构件（需要构造的数据结构）
- 三、介绍一下 基本 packing 策略（常见做法）
- 四、介绍一下 基本 packing 关键实现细节（伪代码 + attention mask）
- 五、介绍一下 位置编码（position embeddings）—— 最常踩坑的地方
- 六、介绍一下 loss & metric 对齐（labels、loss_mask、negative sample）
- 七、介绍一下 模态嵌入 & 分隔符

- [点击查看答案](https://articles.zsxq.com/id_cu81jkabtar9.html)

### [beam search及其变种](https://articles.zsxq.com/id_e5v80404iu5t.html)

- beam search及其变种
  - 前言
  - 一、什么是 Beam Search？
  - 二、Beam Search 重奏什么问题？
  - 三、常见 Beam Search 变种算法
    - 3.1 Diverse Beam Search（DBS）
    - 3.2 Constrained Beam Search（CBS）
    - 3.3 Minimum Bayes Risk Decoding（MBR）
    - 3.4 Typical Decoding（TD）
    - 3.5 Guided Decoding（引导式解码）
    - 3.6 Contrastive Decoding（对比式解码）
    - 3.7 Entropy Sampling（熵驱动采样）
    - 3.8 Beam Re-ranking（重排序）
    - 3.9 MoE-aware Routing Beam（专家感知解码）

- [点击查看答案](https://articles.zsxq.com/id_e5v80404iu5t.html)

### [Continuous Batching 与 Selective Batching 实现](https://articles.zsxq.com/id_cenk2dnof3cx.html)

- Continuous Batching 与 Selective Batching 实现
  - 一、前言
  - 二、Continuous Batching 实现
    - 2.1 算法原理
    - 2.2 具体实现
  - 三、Selective Batching 实现
    - 3.1 算法原理
    - 3.2 具体实现
  - 四、实验结果分析
    - 4.1 实验设置
    - 4.2 Continuous Batching 运行过程
    - 4.3 Selective Batching 运行过程
    - 4.4 性能对比
  - 五、总结与思考

- [点击查看答案](https://articles.zsxq.com/id_cenk2dnof3cx.html)

### [Speculative Decoding（推测解码）](https://articles.zsxq.com/id_8hfss9wes6ww.html)

- 一、什么是 speculative decoding？
- 二、介绍一下 speculative decoding 核心原理？
- 三、介绍一下 speculative decoding 工作流程？
- 四、为什么 speculative decoding 能够加速？
- 五、speculative decoding 实际效果？
- 六、speculative decoding 比 传统解码 优势在哪里？
- 七、speculative decoding 应用场景？
- 八、speculative decoding 限制？
- 九、speculative decoding 总结

- [点击查看答案](https://articles.zsxq.com/id_8hfss9wes6ww.html)

## 四、大模型（LLMs）推理优化篇

### [Linear Attention 推理设计思考](https://articles.zsxq.com/id_28fxlvwk9g6b.html)

- 前言
- 一、GDN 计算量: Linear State 一定比 Softmax KVcache 节省吗？
- 二、KDA Prologue Kernel 并行
- 三、Speculative Decoding 优化
- 四、Prefix Caching 优化

- [点击查看答案](https://articles.zsxq.com/id_28fxlvwk9g6b.html)

### [LLM（大语言模型）部署加速方法——PagedAttention篇](https://articles.zsxq.com/id_p22mjq881n3n.html)

- 一、vLLM 用于大模型并行推理加速 存在什么问题？
- 二、vLLM 如何 优化 大模型并行推理加速？
- 三、什么是 PagedAttention？
- 四、 PagedAttention 如何存储 连续的key和value？
- 五、 PagedAttention 技术细节？
- 六、 PagedAttention 如何 实现安全共享？
- 七、 PagedAttention 源码介绍？

- [点击查看答案](https://articles.zsxq.com/id_p22mjq881n3n.html)

### [LLM（大语言模型）部署加速方法——Faster Transformer篇](https://articles.zsxq.com/id_dd2gowztxtfg.html)

- 一、为什么需要 FasterTransformer？
- 二、FasterTransformer 介绍一下？
- 三、FasterTransformer 核心是什么？
- 四、FasterTransformer 优化？

- [点击查看答案](https://articles.zsxq.com/id_dd2gowztxtfg.html)

## 五、大模型推理加速框架面

### [大模型（LLMs）加速篇](https://articles.zsxq.com/id_w9wewc152eux.html)

- 1 当前优化模型最主要技术手段有哪些？
- 2 推理加速框架有哪一些？都有什么特点？
- 3 vLLM 篇
  - 3.1 vLLM 的 功能有哪些？
  - 3.2 vLLM 的 优点有哪些？
  - 3.3 vLLM 的 缺点有哪些？
  - 3.4 vLLM 离线批量推理？
  - 3.5 vLLM API Server？
- 4 Text generation inference 篇
  - 4.1 介绍一下 Text generation inference？
  - 4.2 Text generation inference 的 功能有哪些？
  - 4.3 Text generation inference 的 优点有哪些？
  - 4.4 Text generation inference 的 缺点有哪些？
  - 4.5 Text generation inference 的 使用docker运行web server？

- [点击查看答案](https://articles.zsxq.com/id_w9wewc152eux.html)

### [大模型推理加速工具 —— vLLM](https://articles.zsxq.com/id_zw5h9ogvac2w.html)

- 一、引言
  - 1.1 前言
  - 1.2 为什么 需要 vLLM ?
  - 1.3 vLLM 具有哪些特点 ?
  - 1.4 vLLM 支持哪些 Huggingface 模型 ?
- 二、vLLM 性能如何？
- 三、vLLM 依赖包
- 四、vLLM 如何安装？
- 五、vLLM 如何使用？
- 六、vLLM 分布式推理与服务

- [点击查看答案](https://articles.zsxq.com/id_zw5h9ogvac2w.html)

### [纯Python超轻量高性能LLM推理框架 —— LightLLM](https://articles.zsxq.com/id_9a643feq2b0b.html)

- 一、引言
  - 1.1 前言
  - 1.2 为什么 需要 LightLLM ?
  - 1.3 目前 LLM推理框架 有 哪些?
- 二、LightLLM 介绍一下？
  - 2.1 什么是 LightLLM ？
  - 2.2 Token Attention 介绍？
  - 2.3 Efficient Router 介绍？
- 三、LightLLM 性能表现 介绍？
- 四、LightLLM 依赖包 有哪些？
- 五、LightLLM  如何安装？
  - 5.1 下载 LightLLM
  - 5.2 安装 LightLLM 依赖
  - 5.3 安装 LightLLM
- 六、LightLLM 如何使用？
  - 6.1 启动 LightLLM 服务
- 填坑笔记
  - LightLLM 支持模型 LLMs 模型？

- [点击查看答案](https://articles.zsxq.com/id_9a643feq2b0b.html)

### [LLM推理技术之StreamingLLM：如何拥有无限长生成能力](https://articles.zsxq.com/id_w1gwi9z7qm5s.html)

- 一、前言
  - 1.1 大型语言模型（LLM）存在什么问题？
  - 1.2 StreamingLLM 背景介绍
  - 1.3 StreamingLLM 核心问题？
  - 1.4 StreamingLLM 存在哪些挑战？
  - 1.5 目前主流地增加输入文本长度的方法有哪些？
- 二、StreamingLLM 的思路是什么？

- [点击查看答案](https://articles.zsxq.com/id_w1gwi9z7qm5s.html)

### [SwiftInfer —— 大模型无限流式输入推理飙升46%，打破多轮对话长度限制](https://articles.zsxq.com/id_0rpua5fejfwc.html) 

- StreamingLLM 篇
  - 一、为什么需要 StreamingLLM？
  - 二、StreamingLLM 思路是什么？
  - 三、StreamingLLM 优点是什么？
- SwiftInfer 篇：基于TensorRT的StreamingLLM实现
  - 一、为什么需要 SwiftInfer？
  - 二、SwiftInfer 思路是什么？
  - 三、SwiftInfer 优点是什么？

- [点击查看答案](https://articles.zsxq.com/id_0rpua5fejfwc.html)

### [大模型(LLM)部署框架对比篇](https://articles.zsxq.com/id_7d31dgh26fcp.html)

- 大模型(LLM)部署框架对比篇
- 一、为什么需要对大模型推理加速？
- 二、大模型(LLM)部署框架对比总览
- 三、大模型(LLM)部署优化策略
  - 3.1 大模型(LLM)部署优化策略——量化
    - 3.1.1 介绍一下 大模型(LLM)量化方法？
    - 3.1.2 一般会对大模型(LLM)中哪些模块进行量化？
    - 3.1.3 大模型(LLM)量化方法
  - 3.2 大模型(LLM)部署优化策略——KV Cache
    - 3.2.1 介绍一下 大模型(LLM) KV Cache 方法？
    - 3.2.2 为什么需要 大模型(LLM) KV Cache 方法？
  - 3.3 大模型(LLM)部署优化策略——Flash Attention
    - 3.3.1 介绍一下 大模型(LLM) Flash Attention 方法？
  - 3.4 大模型(LLM)部署优化策略——Paged Attention
    - 3.4.1 介绍一下 大模型(LLM) Paged Attention 方法？
  - 3.5 大模型(LLM)部署优化策略——Continuous batching
    - 3.5.1 介绍一下 大模型(LLM) Continuous batching 方法？
  - 3.6 大模型(LLM)部署优化策略——Speculative Decoding
    - 3.6.1 介绍一下 大模型(LLM) Speculative Decoding 方法？
  - 3.7 大模型(LLM)部署优化策略——Medusa
    - 3.7.1 介绍一下 大模型(LLM) Medusa 方法？

- [点击查看答案](https://articles.zsxq.com/id_7d31dgh26fcp.html)

