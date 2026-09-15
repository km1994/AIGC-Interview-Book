# 大模型面试常考题篇

## LLMs 对比篇 :fire: 

### [从 DeepSeek-V3 到 Kimi K2：八种现代大语言模型架构设计](https://articles.zsxq.com/id_q6lmgs09rn2i.html):fire: 

- 从 DeepSeek-V3 到 Kimi K2：八种现代大语言模型架构设计
  - 一、前言
  - 二、DeepSeek V3/R1 篇
    - 2.1 Multi-Head Latent Attention (MLA)
      - 2.1.1 Multi-Head Attention (MHA)
      - 2.1.2 Grouped-query Attention (GQA)
        - 2.1.2.1 Grouped-query Attention (GQA) 介绍
        - 2.1.2.2 Grouped-query Attention (GQA) 核心思想
        - 2.1.2.3 Grouped-query Attention (GQA) 优点
        - 2.1.2.4 Grouped-query Attention (GQA) 缺点
      - 2.1.3 Multi-Head Latent Attention (MLA)
    - 2.2 Mixture-of-Experts (MoE)
      - 2.2.1 Mixture-of-Experts (MoE) 核心思想 是什么？
      - 2.2.2 Mixture-of-Experts (MoE) 优点 是什么？
    - 2.3 DeepSeek Summary
      - 2.3.1 为什么 6710B DeepSeek-V3 推理效率 优于 405B Llama 3?
      - 2.3.2 DeepSeek-V3 和 Llama 3 有什么区别?
  - 三、OLMo 2
    - 3.1 Normalization Layer Placement
    - 3.2 QK-Norm
    - 3.3 OLMo 2 Summary
  - 四、Gemma 3
    - 4.1 介绍 Gemma 3
    - 4.2 Gemma 3 还有什么有趣的地方呢？
    - 4.3 Sliding Window Attention
      - 4.3.1 什么是滑动窗口注意力呢？
      - 4.3.2 Normalization Layer Placement in Gemma 3
      - 4.3.3 Gemma 3 Summary
      - 4.3.4 Bonus: Gemma 3n
  - 五、Mistral Small 3.1
  - 六、Llama 4
    - 6.1 介绍一下 Llama 4？
    - 6.2 介绍一下 Llama 4 和 DeepSeek-V3 区别？
  - 七、Qwen3
    - 7.1 Qwen3
      - 7.1.1 介绍一下 Qwen3？
      - 7.1.2 介绍一下 Qwen3 0.6B 和 Llama 3 1B 区别？
      - 7.1.3 介绍一下 Qwen3 0.6B 优点？
      - 7.1.4 介绍一下 Qwen3 0.6B 存在问题？
    - 7.2 Qwen3 (MoE)
      - 7.2.1 Qwen3 (MoE) 包含哪些版本？
      - 7.2.2 为什么像 Qwen3 这样的架构既有常规（密集）又有 MoE（稀疏）变体？
      - 7.2.3  Qwen3 常规（密集）又有 MoE（稀疏）变体 都适用哪些场景？
      - 7.2.4 DeepSeek-V3 和 Qwen3 235B-A22B 的架构比较？
  - 八、SmolLM3
    - 8.1 介绍一下 SmolLM3？
    - 8.2 No Positional Embeddings (NoPE)
  - 九、Kimi 2

- [点击查看答案](https://articles.zsxq.com/id_q6lmgs09rn2i.html)

### [LLMs 对比篇](https://articles.zsxq.com/id_fsq8czgwjxse.html)

- LLMs 对比篇
  - 一、谈谈你对当前出现的各种大模型的见解？
  - 二、目前大模型常见的 base 模型训练和 chat 模型训练 方式 的区别么？
  - 三、llama、baichuan、ChatGLM、Bloom 和 qwen 等开源大模型技术对比篇
    - 3.1 llama 系列篇
      - 3.1.1 llama 篇
        - 3.1.1.1 llama 训练数据 介绍
        - 3.1.1.2 llama 模型参数量 介绍
        - 3.1.1.3 llama 模型结构 介绍
        - 3.1.1.4 llama 训练目标 介绍
        - 3.1.1.5 llama tokenizer 介绍
        - 3.1.1.6 llama 衍生模型 介绍
        - 3.1.1.7 llama 词表扩展: Chinese LLaMA
      - 3.2.1 llama2 篇
        - 3.2.1 llama2 系列 数据预处理方式？
        - 3.2.2 llama2 系列 Tokenizer 处理方式？
        - 3.2.3 llama2 系列 Architectural？
        - 3.2.4 llama2 系列 content长度？
    - 3.2 Mistral 7B 系列篇
      - 3.2.1  Mistral 7B Architectural？
    - 3.3 Qwen 系列篇
      - 3.3.1 Qwen 系列 数据预处理方式？
      - 3.3.2 Qwen 系列 Tokenizer 处理方式？
      - 3.3.3 Qwen 系列 ARCHITECTURE？
    - 3.4 Baichuan 系列篇
      - 3.4.1 Baichuan2 篇
        - 3.4.1.1 Baichuan2 系列 数据预处理方式？
        - 3.4.1.2 Baichuan2 系列 Tokenizer 处理方式？
        - 3.4.1.2 Baichuan2 系列 Architecture ？
    - 3.5 GLM 系列篇
      - 3.5.1 ChatGLM-6B 篇
        - 3.5.1.1 ChatGLM-6B 结构特点？
        - 3.5.1.2 ChatGLM-6B 训练目标？
        - 3.5.1.3 ChatGLM-6B  tokenizer？
    - 3.6 BLOOM 系列篇
      - 3.6.1 BLOOM 篇
        - 3.6.1.1 BLOOM 训练数据构建？
        - 3.6.1.2 BLOOM 模型参数量？
        - 3.6.1.3 BLOOM 模型结构？
        - 3.6.1.4 BLOOM 训练目标？
        - 3.6.1.5 BLOOM tokenizer?
  - 四、分析与总结？
    - 4.1 大模型训练共同点？
    - 4.2 大模型训练不同点？
  - 五、对比
    - 5.1 LLaMA、ChatGLM 和 BLOOM 对比
    - 5.2 LLaMA、ChatGLM 和 BLOOM 的 tokenizer 比较
    - 5.3LLaMA、ChatGLM 和 BLOOM 的 结果 比较

- [点击查看答案](https://articles.zsxq.com/id_fsq8czgwjxse.html)

### [大模型-attention mask 篇](https://articles.zsxq.com/id_0j7k3gxa5hpm.html)

- 大模型-attention mask 篇
  - 1、prefix-tuning的prefix tokens是双向注意力吗？
  - 2、chatglm1和chatglm2的attention mask是怎么样的？
  - 3、llama的attention mask是怎么样的？

- [点击查看答案](https://articles.zsxq.com/id_0j7k3gxa5hpm.html)

## 七、大模型—— Qwen系列 篇 :fire:

### 大模型—— Qwen3 篇 :fire:

#### [Qwen3-RL训练过程详解](https://articles.zsxq.com/id_dtn1elnwgben.html)

- 一、引言
- 阶段一：冷启动训练
  - 2.1 为什么要进行 数据筛选
  - 2.2 Query筛选
  - 2.3 Response筛选
- 阶段二：RL推理能力训练
- 阶段三：思考模式融合
- 阶段四：通用RL
- Strong-to-Weak Distillation
- 后训练效果评估

- [点击查看答案](https://articles.zsxq.com/id_dtn1elnwgben.html)

#### [Qwen3开源大模型源码分析](https://articles.zsxq.com/id_nud427c7ekpm.html)

- 一、Qwen3-Model
- 二、Qwen3Config
- 三、Qwen3Model
- 3.1 Qwen3Model 初始化
- 3.2 Qwen3Model forward
- 3.3 Qwen3DecoderLayer
- 3.4 DecoderLayers 初始化
- 3.5 DecoderLayers forward
- 3.6 Qwen3 Attention
- 3.7 Qwen3Attention forward
- 3.8 Qwen3MLP
- 3.9 Qwen3RMSNorm

- [点击查看答案](https://articles.zsxq.com/id_nud427c7ekpm.html)

## 六、大模型——DeepSeek-R1 百问百搭 篇 :fire:

### [大模型——DeepSeek-R1 百问百搭 篇](https://articles.zsxq.com/id_qixbm7loips9.html) :fire:

- [GRPO（Group Relative Policy Optimization）篇](https://articles.zsxq.com/id_eh0ee91au5t4.html)  :fire:
- [DeepSeek-R1-Zero 篇](https://articles.zsxq.com/id_f8gqnqzxhx0n.html)  :fire:
- [DeepSeek-R1 百问百搭-DeepSeek-R1篇](https://articles.zsxq.com/id_tnkh7vh7jvw1.html) :fire:
- [千面郎君 篇（三十二章）—— DeepSeek-R1 论文解读](https://articles.zsxq.com/id_o4r4cpl8cqzm.html)  :fire:
- [DeepSeek-R1 篇——如何通过强化学习实现复杂推理](https://articles.zsxq.com/id_6w5tdozaf2pn.html)

## 五、大模型——Chat o1 篇 :fire:

### [千面郎君 篇（三十一章）—— OpenAI o1 篇](https://articles.zsxq.com/id_71rmw7acx3cd.html)

- 千面郎君 篇（三十一章）—— OpenAI o1 篇
  - 一、Shortcut learning (捷径学习) vs Journey learning (旅程学习)
    - 1.1 Shortcut learning (捷径学习)
      - 1.1.1 什么是 Shortcut learning (捷径学习)？
      - 1.1.2 Shortcut learning (捷径学习) 包含哪些关键特征？
      - 1.1.3 Shortcut learning (捷径学习) 优点是什么？
      - 1.1.4 Shortcut learning (捷径学习) 缺点是什么？
    - 1.2 Journey learning (旅程学习)
      - 1.2.1 什么是 Journey learning (旅程学习)？
      - 1.2.2 Journey learning (旅程学习) 包含哪些关键特征？
      - 1.2.3 Journey learning (旅程学习) 优点是什么？
    - 1.3 Shortcut learning (捷径学习) vs Journey learning (旅程学习)
  - 二、o1 的长思维链篇
    - 2.1 o1 的长思维链是什么样子？
    - 2.2 长思维 (Long thought) 是如何工作的？
    - 2.3 如何构建长思维？
  - 三、过程奖励模型 (PRM)篇
    - 3.1 为什么需要 过程奖励模型 (PRM)？
    - 3.2 过程奖励模型 (PRM) 作用是什么？
    - 3.3 过程奖励模型 (PRM) 目标值定义？
    - 3.4 过程奖励模型 (PRM) 思路介绍？
    - 3.5 如何训练 过程奖励模型 (PRM) ？
      - 3.5.1 介绍一下 ORM 目标函数？
      - 3.5.2 介绍一下 PRM 目标函数？
      - 3.5.3 如何构建 PRM 训练数据？
      - 3.5.4 PRM 训练细节？
        - 3.5.4.1 将输入数据转化为模型输入（token id）
        - 3.5.4.2 利用 PRM 对每个步骤打分
        - 3.5.4.3 完整代码
  - 四、on-policy 推理树篇
    - 4.1 如何构建 on-policy 推理树？
    - 4.2 如何从推理树中推导出 Long Thought？
  - 五、如何评估尝试方法？
  - 六、如何训练模型？
    - 6.1 第一阶段：监督微调（SFT）
    - 6.2 第二阶段：直接偏好学习（DPO）
  - 七、什么是人类和 AI 协同标注的有效策略？

- [点击查看答案](https://articles.zsxq.com/id_71rmw7acx3cd.html)

### [OpenAI o1 面试篇](https://articles.zsxq.com/id_032nwgcgwhc6.html)

- OpenAI o1 面试篇
  - Q: o1 的训练方法与之前的模型有何主要区别？
  - Q: o1 的"思考"过程与简单的提示有何不同？
  - Q: 为什么 o1 在推理任务上比之前的模型更强大？
  - Q: o1 如何处理安全性问题？
  - Q: o1 在数学和编程任务上有哪些具体的改进？
  - Q: o1 Mini 与完整版 o1 模型相比如何？
  - Q: o1 是否只擅长数学和 STEM 任务？
  - Q: 给予 o1 更多时间如何增强其推理能力？
  - Q: o1 如何决定在给定问题上花费多少时间进行推理？
  - Q: 当前 o1 思考时间的瓶颈是否由上下文长度决定？
  - Q: o1 在更抽象、创造性领域的表现如何？
  - Q: o1 的改进是否仅仅由训练数据的变化导致的？
  - Q: 科学家如何帮助构建用于科学发现的 AGI？
  - Q: o1 是否表现出意识或自我意识的特征？
  - Q: o1 的推理时间和质量之间是否存在线性关系？
  - Q: 在开发 o1 时，研究人员的第一个"啊哈时刻"是什么？
  - Q: o1 如何处理工具使用以进行自我验证或理智检查？
  - Q: o1 如何处理更主观任务中的文化背景？
  - Q: o1 Mini 如何在更小更便宜的同时实现其性能？
  - Q: 改进 o1 和 o1 Mini 的下一步计划是什么？
  - 致谢

- [点击查看答案](https://articles.zsxq.com/id_032nwgcgwhc6.html)

### [Scaling LLM Test-Time：谁说类o1推理一定要用RL?](https://articles.zsxq.com/id_71l9woqohebk.html)

- Scaling LLM Test-Time：谁说类o1推理一定要用RL?
  - 一、Scaling LLM Test-Time 介绍篇
    - 1.1 为什么需要 Scaling LLM Test-Time？
    - 1.2 三种 Scaling LLM Test-Time 类型定义？
    - 1.3 有哪些 Scaling Test-Time的方法？
    - 问题引申
  - 二、方法一：纯 Inference Scaling 篇
    - 2.1 Inferece Test-Time的统一视角：Proposer \& Verifier
    - 2.2 Proposer \& Verifier 实例：Best-of-N
    - 2.3 Process Reward Modeling
      - 2.3.1 结果奖励模型(Outcome Reward Model, ORM) 有什么问题？
      - 2.3.2 为什么要用 过程奖励模型(Process Reward Model, PRM)？
      - 2.3.3 为什么有ORM了还需要PRM？
      - 2.3.4 介绍一下 PRM建模：Let’s Verify Step by Step？
      - 2.3.5 介绍一下 PRM建模：reward-to-go？
      - 2.3.6 OpenAI PRM 建模 和 Math-Shephered PRM 建模 的区别？
      - 2.3.7 介绍一下 PRM使用实例？
      - 2.3.8 介绍 PRM数据格式？
    - 2.4 搜索方法 Process-Searching
      - 2.4.1 介绍一下 Best-of-N Weighted？
      - 2.4.2 介绍一下 Beam Search PRM？
      - 2.4.3 介绍一下 LookAhead Search？
      - 2.4.4 对比 Best-of-N Weighted 和 Beam Search PRM 和 LookAhead Search 效能？
    - 2.5 不同的PRM search在各难度问题的表现
    - 2.6 小结
  - 三、方法二：推理能力增强
    - 3.1 什么是提议分布 ？
    - 3.2 什么是 修改提议分布 ？
    - 3.3 什么是 Self-Improve ？
    - 3.4 什么是 revision 数据收集 ？
    - 3.5 什么是 revision model结果 ？
    - 3.6 什么是 verifier 训练方案 ？
    - 3.7 小结
  - 四、Pretrain Compute V.S. Scaling Test Time
    - 4.1 Exchange Rate度量
    - 4.2 小模型推理token增加14倍就能战胜大模型？
    - 4.3 完整的性能表现
    - 4.4 小结
  - 五、Scaling LLM Test-Time总结

- [点击查看答案](https://articles.zsxq.com/id_71l9woqohebk.html)

## 四、大模型——Kimi1.5 篇 :fire:

- [kimi1.5 论文研读](https://articles.zsxq.com/id_8v9xlujg57pv.html) 

## 三、baichuan篇

### [百川智能baichuan7B、13B、53B、baichuan2 总结篇](https://articles.zsxq.com/id_ma6pw7v2g9pi.html)

- 一、baichuan-7B篇
  - 你了解baichuan-7B解构么？介绍一下？
  - baichuan-7B 如何 收集原始数据并 构建 训练数据？
  - baichuan-7B 如何 提高 训练稳定性和吞吐？
- 二、baichuan-13B篇
  - 相比于 baichuan-7B，baichuan-13B 的 特点体现在哪里？
  - 如何 对 baichuan-13B 进行推理和部署？
  - 如何 对 baichuan-13B 进行微调？
- 三、baichuan-53B篇
  - 3.1 baichuan-53B 相比于 baichuan-7B 和 baichuan-13B 有哪些优势？
  - 3.2 baichuan-53B 如何对 预训练数据 做处理？
  - 3.3 baichuan-53B 如何进行 搜索增强？
- 四、baichuan2篇
  - 4.1 baichuan2 与 其他大模型 对比
- 五、baichuan 数据构建篇
  - 5.1 baichuan 进行微调时，领域数据：通用数据配比？

- [点击查看答案](https://articles.zsxq.com/id_ma6pw7v2g9pi.html)

## 二、LLaMa 篇

### [LLaMa 篇](https://articles.zsxq.com/id_9ba6a72wan2w.html) 

- 一、相比较于llama而言，llama2有哪些改进，对于llama2是应该如何finetune？

- [点击查看答案](https://articles.zsxq.com/id_9ba6a72wan2w.html)

## 一、GPT 经验篇

### [GPT 经验篇](https://articles.zsxq.com/id_r46k6bqu34xh.html) 

- 一、gpt源码past\_key\_value是干啥的？
- 二、gpt onebyone 每一层怎么输入输出？
- 三、bert和gpt有什么区别
- 四、文本生成的几大预训练任务？
- 五、讲讲T5和Bart的区别，讲讲bart的DAE任务？
- 六、讲讲Bart和Bert的区别？
- 七、gpt3和gpt2的区别？

- [点击查看答案](https://articles.zsxq.com/id_r46k6bqu34xh.html)


