# LLMs 千面郎君（更新版）

> 介绍：本项目是作者们根据个人面试和经验总结出的 大模型(LLMs)面试准备的学习笔记与资料，该资料目前包含 大模型(LLMs)各领域的 面试题积累。

## 一、大模型（LLMs）基础面 

### [大模型（LLMs）基础面](https://articles.zsxq.com/id_mw52p1pfbzql.html) 

- 1 目前 主流的开源模型体系 有哪些？
- 2 prefix Decoder 和 causal Decoder 和 Encoder-Decoder 区别是什么？
- 3 大模型LLM的 训练目标 是什么？
- 4 涌现能力是啥原因？
- 5 为何现在的大模型大部分是Decoder only结构？
- 6 简单 介绍一下 大模型【LLMs】？
- 7 大模型【LLMs】后面跟的 175B、60B、540B等 指什么？
- 8 大模型【LLMs】具有什么优点？
- 9 大模型【LLMs】具有什么缺点？
- 10 encoder-only, decoder-only, encoder-decoder的区别?
- 11 BART、llama、gpt、t5、palm等主流模型异同点?
- 12 prefix LM 和 causal LM 区别是什么?

- [点击查看答案](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### [Layer normalization 篇](https://articles.zsxq.com/id_pzcgd4ovk098.html)

- Layer normalization-方法篇
  - Layer Norm 篇
    - Layer Norm 的计算公式写一下？
  - RMS Norm 篇 （均方根 Norm）
    - RMS Norm 的计算公式写一下？
    - RMS Norm 相比于 Layer Norm 有什么特点？
  - Deep Norm 篇
    - Deep Norm 思路？
    - 写一下 Deep Norm 代码实现？
  - Deep Norm 有什么优点？
- Layer normalization-位置篇
  - 1 LN 在 LLMs 中的不同位置 有什么区别么？如果有，能介绍一下区别么？
- Layer normalization 对比篇
  - LLMs 各模型分别用了 哪种 Layer normalization？

- [点击查看答案](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### [LLMs 激活函数篇](https://articles.zsxq.com/id_6xm3wzzice2s.html) 

- 1 介绍一下 FFN 块 计算公式？
- 2 介绍一下 GeLU 计算公式？
- 3 介绍一下 Swish 计算公式？
- 4 介绍一下 使用 GLU 线性门控单元的 FFN 块 计算公式？
- 5 介绍一下 使用 GeLU 的 GLU 块 计算公式？
- 6 介绍一下 使用 Swish 的 GLU 块 计算公式？
- 7 各LLMs 都使用哪种激活函数？
- 8 Adam优化器和SGD的区别？

- [点击查看答案](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### [Attention 升级面](https://articles.zsxq.com/id_u67us9zex93d.html) 

- [Attention 升级面](https://articles.zsxq.com/id_u67us9zex93d.html) 
  - 1 传统 Attention 存在哪些问题？
  - 2 Attention 有哪些 优化方向？
  - 3 Attention 变体有哪些？
  - 4 Multi-Query Attention 篇
    - 4.1 Multi-head Attention 存在什么问题？
    - 4.2 介绍一下 Multi-Query Attention？
    - 4.3 对比一下 Multi-head Attention 和 Multi-Query Attention？
    - 4.4 Multi-Query Attention 这样做的好处是什么？
    - 4.5 有 哪些模型 是 使用 Multi-Query Attention？
  - 5 Grouped-query Attention
    - 5.1 什么是 Grouped-query Attention？
    - 5.2 有哪些大模型使用 Grouped-query Attention？
  - 6 FlashAttention
    - 6.1 为什么需要  FlashAttention？
    - 6.2 简单介绍一下 FlashAttention？
    - 6.3 简单介绍一下 FlashAttention 核心？
    - 6.4 介绍一下 FlashAttention 优点？
    - 6.5 介绍一下 FlashAttention 代表模型？
  - 7 并行 transformer block
  - 8 attention计算复杂度以及如何改进？
  - 9 Paged Attention篇
    - 9.1 简单介绍一下 Paged Attention？
  - 对比篇
    - 1、MHA，GQA，MQA 三种注意力机制是否了解?区别是什么?

- [点击查看答案](https://articles.zsxq.com/id_u67us9zex93d.html)

- [跨注意力机制（Cross-Attention）篇](https://articles.zsxq.com/id_gwn416686pic.html) 
  - 一、为什么需要 跨注意力机制（Cross-Attention）？
  - 二、介绍一些 跨注意力机制（Cross-Attention）？
  - 三、Cross Attention 和 Self Attention 篇
    - 3.1 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么相同点？
    - 3.2 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么不同点？
  - 四、Cross Attention 和 多头注意力（Multi-Head Attention）篇
    - 4.2 Cross Attention 和 多头注意力（Multi-Head Attention） 都是基于注意力机制的，有什么异同点？
  - 五、Cross Attention 代码实现
  - 六、Cross Attention 应用场景
  - 七、Cross Attention 的优势和挑战？

- [点击查看答案](https://articles.zsxq.com/id_gwn416686pic.html)

### [transformers 操作篇](https://articles.zsxq.com/id_rsll7gsd8va5.html) 

- 1. 如何 利用 transformers 加载 Bert 模型？
- 2. 如何 利用 transformers 输出 Bert 指定 hidden\_state？
- 3. BERT 获取最后一层或每一层网络的向量输出

- [点击查看答案](https://articles.zsxq.com/id_rsll7gsd8va5.html)

### [LLMs 损失函数篇](https://articles.zsxq.com/id_q0ajjlbc8493.html) 

- 一、介绍一下 KL 散度？
- 二、交叉熵损失函数写一下，物理意义是什么？
- 三、KL 散度与交叉熵的区别？
- 四、多任务学习各loss差异过大怎样处理？
- 五、分类问题为什么用交叉熵损失函数不用均方误差（MSE）？
- 六、什么是信息增益？
- 七、多分类的分类损失函数(Softmax)？
- 八、softmax和交叉熵损失怎么计算，二值交叉熵呢？
- 九、如果softmax的e次方超过float的值了怎么办？

- [点击查看答案](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### [相似度函数篇](https://articles.zsxq.com/id_wp25j5xr8ocw.html) 

- 一、除了cosin还有哪些算相似度的方法
- 二、了解对比学习嘛？
- 三、对比学习负样本是否重要？负样本构造成本过高应该怎么解决？

- [点击查看答案](https://articles.zsxq.com/id_wp25j5xr8ocw.html)

## [二、大模型（LLMs）进阶面](https://articles.zsxq.com/id_xr65bxpcsnoh.html) 

- 一、什么是生成式大模型？
- 二、大模型是怎么让生成的文本丰富而不单调的呢？
- 三、LLMs 复读机问题
  - 3.1 什么是 LLMs 复读机问题？
  - 3.2 为什么会出现 LLMs 复读机问题？
  - 3.3 如何缓解 LLMs 复读机问题？
- 四、llama 系列问题
  - 4.1 llama 输入句子长度理论上可以无限长吗？
- 五、什么情况用Bert模型，什么情况用LLaMA、ChatGLM类大模型，咋选？
- 六、各个专业领域是否需要各自的大模型来服务？
- 七、如何让大模型处理更长的文本？

- [点击查看答案](https://articles.zsxq.com/id_xr65bxpcsnoh.html)

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

## 四、大模型（LLMs）参数高效微调(PEFT) 面

### [大模型（LLMs）参数高效微调(PEFT) 面](https://articles.zsxq.com/id_ipkod91a939n.html)

- 1. 微调方法是啥？如何微调？
- 2. 为什么需要 PEFT？
- 3. 介绍一下 PEFT？
- 4. PEFT 有什么优点？
- 5. 微调方法批处理大小模式GPU显存速度？
- 6. Peft 和 全量微调区别？
- 7. 多种不同的高效微调方法对比
- 8. 当前高效微调技术存在的一些问题
- 9. 高效微调技术最佳实践
- 10. PEFT 存在问题？
- 11. 能不能总结一下各种参数高效微调方法？

- [点击查看答案](https://articles.zsxq.com/id_ipkod91a939n.html)

### [配器微调（Adapter-tuning）篇](https://articles.zsxq.com/id_0n6pfw0wz3xb.html)

- 一、为什么 需要 适配器微调（Adapter-tuning）？
- 二、适配器微调（Adapter-tuning）思路？
- 三、 适配器微调（Adapter-tuning）特点是什么？
- 四、AdapterFusion 思路 是什么？
- 五、AdapterDrop 思路 是什么？
- 六、AdapterDrop 特点 是什么？
- 七、MAM Adapter 思路 是什么？
- 八、MAM Adapter 特点 是什么？

- [点击查看答案](https://articles.zsxq.com/id_0n6pfw0wz3xb.html)

### [提示学习（Prompting）](https://articles.zsxq.com/id_662wpbw47gtj.html)

- 一、为什么需要 提示学习（Prompting）？
- 二、什么是 提示学习（Prompting）？
- 三、提示学习（Prompting） 有什么优点？
- 四、提示学习（Prompting）有哪些方法，能不能稍微介绍一下它们间？
  - 4.1 前缀微调（Prefix-tining）篇
    - 4.1.1 为什么需要 前缀微调（Prefix-tining）？
    - 4.1.2 前缀微调（Prefix-tining）思路是什么？
    - 4.1.3 前缀微调（Prefix-tining）的优点是什么？
    - 4.1.4 前缀微调（Prefix-tining）的缺点是什么？
  - 4.2 指示微调（Prompt-tuning）篇
    - 4.2.1 为什么需要 指示微调（Prompt-tuning）？
    - 4.2.2 指示微调（Prompt-tuning）思路是什么？
    - 4.2.3 指示微调（Prompt-tuning）优点是什么？
    - 4.2.4 指示微调（Prompt-tuning）缺点是什么？
    - 4.2.5 指示微调（Prompt-tuning）与 Prefix-tuning 区别 是什么？
    - 4.2.6 指示微调（Prompt-tuning）与 fine-tuning 区别 是什么？
  - 4.3 P-tuning 篇
    - 4.3.1 为什么需要 P-tuning？
    - 4.3.2 P-tuning 思路是什么？
    - 4.3.3 P-tuning 优点是什么？
    - 4.3.4 P-tuning 缺点是什么？
  - 4.4 P-tuning v2 篇
    - 4.4.1 为什么需要 P-tuning v2？
    - 4.4.2 P-tuning v2 思路是什么？
    - 4.4.3 P-tuning v2 优点是什么？
    - 4.4.4 P-tuning v2 缺点是什么？

- [点击查看答案](https://articles.zsxq.com/id_662wpbw47gtj.html)

### [LoRA 系列篇](https://articles.zsxq.com/id_gjkhd8xn4pvt.html) 

一、LoRA篇
    - 1.1 什么是 LoRA？
    - 1.2 LoRA 的思路是什么？
    - 1.3 LoRA 的特点是什么？
    - 1.4 简单描述一下 LoRA?
    - 1.5 解释一下 LORA 微调的原理和计算流程？
- 二、LoRA变体篇
    - 2.1 QLoRA篇
        - 2.1.1 QLoRA 的思路是怎么样的？
        - 2.1.2 QLoRA 的特点是什么？
        - 2.1.3 QLORA相比LORA做了哪些改进?
    - 2.2 AdaLoRA篇
    -   .2.1 AdaLoRA 的思路是怎么样的？
    - 2.3 LongLoRA篇
        - 2.3.1 为什么需要 LongLoRA？
        - 2.3.2 LongLoRA 思路是什么？
        - 2.3.3 介绍一下 shift short attention？
- 三、Lora的矩阵怎么初始化？为什么要初始化为全0？
- 四、LoRA权重是否可以合入原模型？
- 五、ChatGLM-6B LoRA后的权重多大？
- 六、LoRA 微调优点是什么？
- 七、LoRA微调方法为啥能加速训练？
- 八、如何在已有LoRA模型上继续训练？
- 九、LoRA 缺点是什么？
- 十、LoRA这种微调方法和全参数比起来有什么劣势吗？
- 十一、LORA应该作用于Transformer的哪个参数矩阵？
- 十二、LoRA 微调参数量怎么确定？
- 十三、Rank 如何选取？
- 十四、alpha参数 如何选取？
- 十五、LoRA 高效微调 如何避免过拟合？
- 十六、微调大模型时, 优化器如何？
- 十七、哪些因素会影响内存使用？
- 十八、LoRA权重是否可以合并？
- 十九、是否可以逐层调整LoRA的最优rank？
- 二十、LORA 微调有哪些超参数需要注意?
- 实践篇
    - 1. LoRA 微调计算可训练参数的比例 如何确定？
    - 2. LoRA 微调结果如何保存？

- [点击查看答案](https://articles.zsxq.com/id_gjkhd8xn4pvt.html)

### [如何使用 PEFT库 中 LoRA？](https://articles.zsxq.com/id_8lx1t1t3w4qf.html) 

- 一、前言
- 二、如何 配置 LoraConfig？
- 三、模型 加入PEFT策略
  - 3.1 模型加载 策略有哪些？
  - 3.2 模型显存占用的部分有哪些？
  - 3.3 模型显存占用 优化策略？
    - 3.3.1 8bit量化 优化策略？
    - 3.3.2 梯度检查 优化策略？
  - 3.4 如何 向 模型 加入PEFT策略？
- 四、PEFT库 中 LoRA 模块 代码介绍
  - 4.1 PEFT库 中 LoRA 模块 整体实现思路
  - 4.2 PEFT库 中 LoRA 模块 \_find\_and\_replace() 实现思路
  - 4.3 PEFT库 中 Lora层的 实现思路
    - 4.3.1 基类 LoraLayer 实现
    - 4.3.2 Linear 实现
- 五、使用 LoRA 对 大模型进行 高效参数微调，如何进行存储？
- 六、使用 LoRA 对 大模型进行 推理，如何进行加载？
- 七、huggingface大模型如何加载多个LoRA并随时切换？

- [点击查看答案](https://articles.zsxq.com/id_8lx1t1t3w4qf.html)

### [大模型 SFT 方式对比篇](https://articles.zsxq.com/id_e2piver2uzei.html) 

- 一、SFT 微调方案如何选择？
- 二、Full Fine Tuning vs Parameter-Efficient Fine-Tuning
- 三、Full Fine Tuning 篇
  - 3.1 介绍一下 Full Fine Tuning？
  - 3.2 介绍一下 Full Fine Tuning 优点？
  - 3.3 介绍一下 Full Fine Tuning 缺点？
- 四、Parameter-Efficient Fine-Tuning 篇
  - 4.1 介绍一下 Parameter-Efficient Fine-Tuning？
- 五、LoRA 篇
  - 5.1 介绍一下 LoRA？
  - 5.2 介绍一下 LoRA 流程？
  - 5.3 介绍一下 LoRA 优点？
  - 5.4 介绍一下 LoRA 缺点？
- 六、QLoRA 篇
  - 6.1 介绍一下 QLoRA？
  - 6.2 介绍一下 QLoRA 流程？
- 七、Adapter Tuning 篇
  - 6.1 介绍一下 Adapter Tuning？
  - 6.2 介绍一下 Adapter Tuning 流程？
- 八、Prefix Tuning 篇
  - 6.1 介绍一下 Prefix Tuning？
  - 6.2 介绍一下 Prefix Tuning 训练示例？
- 九、Prompt Tuning 篇
  - 9.1 介绍一下 Prompt Tuning？
  - 9.2 介绍一下 Prompt Tuning 优点？
  - 9.3 介绍一下 Prompt Tuning 缺点？
  - 9.4 介绍一下 Prompt Tuning 训练示例？
- 十、P-Tuning 篇
  - 10.1 介绍一下 P-Tuning？
  - 10.2 介绍一下 P-Tuning 优点？
  - 10.3 介绍一下 P-Tuning 缺点？
- 十一、P-Tuning V2 篇
  - 11.1 介绍一下 P-Tuning V2？
  - 11.2 介绍一下 P-Tuning V2 优点？

- [点击查看答案](https://articles.zsxq.com/id_e2piver2uzei.html)

## 五、大模型（LLMs）推理面 

### [大模型（LLMs）推理面](https://articles.zsxq.com/id_b9eecaoga75i.html)

- 1. 为什么大模型推理时显存涨的那么多还一直占着？
- 2. 大模型在gpu和cpu上推理速度如何？
- 3. 推理速度上，int8和fp16比起来怎么样？
- 4. 大模型有推理能力吗？
- 5. 大模型生成时的参数怎么设置？
- 6. 有哪些省内存的大语言模型训练/微调/推理方法？
- 7. 如何让大模型输出合规化
- 8. 应用模式变更
- 9. 模型输出的分布比较稀疏，怎么处理？

- [点击查看答案](https://articles.zsxq.com/id_b9eecaoga75i.html)

## 六、大模型（LLMs）增量预训练篇 

### [大模型（LLMs）增量预训练篇](https://articles.zsxq.com/id_jfq8la7g20ww.html)

- 1. 为什么要增量预训练？
- 2. 进行 增量预训练 需要做哪些准备工作？
- 3. 增量预训练 所用 训练框架？
- 4. 增量预训练 训练流程 是怎么样？
- 5. 增量预训练 一般需要多大数据量？
- 6. 增量预训练 过程中，loss 上升正常么？
- 7. 增量预训练 过程中，lr 如何设置？
- 8. 增量预训练 过程中，warmup\_ratio 如何设置？
- 9. warmup 的步数 对 大模型继续预训练 是否有影响？
- 10. 学习率 大小 对 大模型继续预训练 后 上下游任务影响？
- 11. 在初始预训练中使用 Rewarmup 对 大模型继续预训练 性能 影响？

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

## 七、大模型（LLMs）分布式训练面 

### [大模型（LLMs）分布式训练面](https://articles.zsxq.com/id_ah2ibj3z22c7.html)

- 1. 理论篇
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
- 2. 实践篇
  - 2.1 假如有超多的8卡A100节点（DGX A100），如何应用3D并行策略？
  - 2.2 如果想构这样一个大规模并行训练系统，训练框架如何选？
  - 2.3 训练框架如何选？
- 3. 并行化策略选择篇
  - 3.1 如何选择一款分布式训练框架？
  - 3.2 如何选择一款分布式训练框架？
  - 3.3 单GPU
  - 3.4 单节点多卡
  - 3.5 多节点多卡
- 4. 问题篇
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

## 八、LLMs 对比篇 :fire: 

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

### [LLMs 对比篇](https://articles.zsxq.com/id_0j7k3gxa5hpm.html)

- 大模型-attention mask 篇
  - 1、prefix-tuning的prefix tokens是双向注意力吗？
  - 2、chatglm1和chatglm2的attention mask是怎么样的？
  - 3、llama的attention mask是怎么样的？

- [点击查看答案](https://articles.zsxq.com/id_0j7k3gxa5hpm.html)

### [百川智能baichuan7B、13B、53B、baichuan2 总结篇](https://articles.zsxq.com/id_ma6pw7v2g9pi.html)

- 一、baichuan-7B篇
  - 1. 你了解baichuan-7B解构么？介绍一下？
  - 2. baichuan-7B 如何 收集原始数据并 构建 训练数据？
  - 3. baichuan-7B 如何 提高 训练稳定性和吞吐？
- 二、baichuan-13B篇
  - 1. 相比于 baichuan-7B，baichuan-13B 的 特点体现在哪里？
  - 2. 如何 对 baichuan-13B 进行推理和部署？
  - 3. 如何 对 baichuan-13B 进行微调？
- 三、baichuan-53B篇
  - 3.1 baichuan-53B 相比于 baichuan-7B 和 baichuan-13B 有哪些优势？
  - 3.2 baichuan-53B 如何对 预训练数据 做处理？
  - 3.3 baichuan-53B 如何进行 搜索增强？
- 四、baichuan2篇
  - 4.1 baichuan2 与 其他大模型 对比
- 五、baichuan 数据构建篇
  - 5.1 baichuan 进行微调时，领域数据：通用数据配比？

- [点击查看答案](https://articles.zsxq.com/id_ma6pw7v2g9pi.html)

### [LLaMa 篇](https://articles.zsxq.com/id_9ba6a72wan2w.html) 

- 一、相比较于llama而言，llama2有哪些改进，对于llama2是应该如何finetune？

- [点击查看答案](https://articles.zsxq.com/id_9ba6a72wan2w.html)

### [GPT 经验篇](https://articles.zsxq.com/id_r46k6bqu34xh.html) 

- 一、gpt源码past\_key\_value是干啥的？
- 二、gpt onebyone 每一层怎么输入输出？
- 三、bert和gpt有什么区别
- 四、文本生成的几大预训练任务？
- 五、讲讲T5和Bart的区别，讲讲bart的DAE任务？
- 六、讲讲Bart和Bert的区别？
- 七、gpt3和gpt2的区别？

- [点击查看答案](https://articles.zsxq.com/id_r46k6bqu34xh.html)

## 九、大模型（LLMs）加速篇  :fire:

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

### [大模型（LLMs）推理加速篇](https://articles.zsxq.com/id_kgzsxgro8cee.html)

- 一、 推理过程 分哪些阶段？
    - 1.1 Prefill（输入理解与初始化）阶段
    - 1.2 Decoding（递归推理与解码输出）阶段
- 二、 推理性能的评价指标？
    - 2.1 Throughput（吞吐量）
    - 2.2 First Token Latency（首字延迟）
    - 2.3 Latency（延迟）
    - 2.4 QPS（每秒请求数）
- 三、 当前优化模型最主要技术手段有哪些？
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
- 四、 推理加速框架有哪一些？都有什么特点？
- 五、 vLLM 篇
    - 5.1 vLLM 的 功能有哪些？
    - 5.2 vLLM 的 优点有哪些？
    - 5.3 vLLM 的 缺点有哪些？
    - 5.4 vLLM 离线批量推理？
    - 5.5 vLLM API Server？
- 六、 Text generation inference 篇
    - 6.1 介绍一下 Text generation inference？
    - 6.2 Text generation inference 的 功能有哪些？
    - 6.3 Text generation inference 的 优点有哪些？
    - 6.4 Text generation inference 的 缺点有哪些？
    - 6.5 Text generation inference 的 使用docker运行web server？

- [点击查看答案](https://articles.zsxq.com/id_kgzsxgro8cee.html)


### [大模型（LLMs）加速篇](https://articles.zsxq.com/id_w9wewc152eux.html)

- 1. 当前优化模型最主要技术手段有哪些？
- 2. 推理加速框架有哪一些？都有什么特点？
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

### [LLMs 推理性能面](https://articles.zsxq.com/id_jwd03u0l7feo.html) 

- 一、介绍一下 LLMs 的文本生成过程？
- 二、如何准确衡量模型的推理速度呢？
- 三、如果对整体推理时延有具体目标，有哪些有效的启发式方法来评估模型？
- 四、LLMs 推理存在哪些挑战？

- [点击查看答案](https://articles.zsxq.com/id_jwd03u0l7feo.html)

### [LLM（大语言模型）部署加速方法——PagedAttention篇](https://articles.zsxq.com/id_p22mjq881n3n.html)

- 一、vLLM 用于大模型并行推理加速 存在什么问题？
- 二、vLLM 如何 优化 大模型并行推理加速？
- 三、什么是 PagedAttention？
- 四、 PagedAttention 如何存储 连续的key和value？
- 五、 PagedAttention 技术细节？
- 六、 PagedAttention 如何 实现安全共享？
- 七、 PagedAttention 源码介绍？

- [点击查看答案](https://articles.zsxq.com/id_p22mjq881n3n.html)

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

### [LLM（大语言模型）部署加速方法——Faster Transformer篇](https://articles.zsxq.com/id_dd2gowztxtfg.html)

- 一、为什么需要 FasterTransformer？
- 二、FasterTransformer 介绍一下？
- 三、FasterTransformer 核心是什么？
- 四、FasterTransformer 优化？

- [点击查看答案](https://articles.zsxq.com/id_dd2gowztxtfg.html)

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

## 十、大模型推理加速——KV Cache篇 :fire:

- [大模型推理加速——KV Cache篇](https://articles.zsxq.com/id_swmfcls3sp1j.html)
  - 一、介绍一下 KV Cache是啥？
  - 二、为什么要进行 KV Cache？
    - 2.1 不使用 KV Cache 场景
    - 2.2 使用 KV Cache 场景
  - 三、说一下 KV Cache 在 大模型中的应用？
    - 3.1 KV Cache 在 Llama 推理流程中应用？
  - 四、 KV Cache 优点？
  - 五、 KV Cache 缺点？
  - 六、 KV Cache 优化策略？
    - 6.1 PageAttention显存优化
    - 6.2 MHA、GQA、MQA优化技术
    - 6.3 FlashAttention优化技术

- [点击查看答案](https://articles.zsxq.com/id_swmfcls3sp1j.html)

- [KV-Cache 面试参考题篇](https://articles.zsxq.com/id_smwaj8hquckz.html)
  - 一、为什么文本生成如此缓慢?
  - 二、如何解决文本生成缓慢问题？
  - 三、什么是键值缓存？
  - 四、KV缓存如何加速？
  - 五、KV缓存能够降低多少计算成本？
  - 六、KV缓存 存在什么问题？
  - 七、如何改善传统的KV缓存？
    - 7.1 Token 选择和修剪方法（Token Selection and Pruning Approaches）
      - 7.1.1 Heavy-Hitter Oracle (H2O)
      - 7.1.2 StreamLLM
      - 7.1.3 Value-Aware Token Pruning (VATP)
    - 7.2 后处理压缩技术（Post-hoc Compression Techniques）
      - 7.2.1 Adaptive KV Compression (FastGen)
      - 7.2.2 动态内存压缩（DMC）
      - 7.2.3 $L\_2$ 范数基础的压缩
    - 7.3 体系结构重设计
      - 7.3.1 多查询注意力（MQA）
      - 7.3.2 分组查询注意力（GQA）
      - 7.3.3 多头潜在注意力（MLA）
      - 7.3.4 SnapKV
      - 7.3.5 只缓存一次（YOCO）

- [点击查看答案](https://articles.zsxq.com/id_smwaj8hquckz.html)

- [从多头共享到潜变量：MLA在低秩投影与按需解压中重新定义 KV-Cache 存储](https://articles.zsxq.com/id_4lcum6gsda71.html)
  - 前言
  - 一、为什么要减少 KV-Cache？
    - 1.1 长序列推理中显存的“隐形杀手”
    - 1.2 显存与带宽的双重约束
  - 二、多头注意力（MHA）篇
    - 2.1 介绍一下 经典注意力公式？
    - 2.2 介绍一下 经典注意力所带来的 显存压力？
  - 三、MQA 篇：极端共享 K/V
    - 3.1 介绍一下 MQA ？
  - 四、GQA 篇：分组共享
    - 4.1 介绍一下 GQA ？
  - 五、对比：MHA / MQA / GQA
  - 六、MLA 的核心：低秩投影与按需还原（不含 RoPE）
    - 6.1 基本思路：改“存多头 K/V”为“存低维潜变量”
    - 6.2 动态解压：显存怎么省？
    - 6.3 低秩投影如何大幅压缩存储？
  - 七、从智能相册系统看 MLA 的“低秩缩略图”运作
    - 7.1 拍照存储：低秩投影
    - 7.2 浏览照片：实时动态解压
    - 7.3 动态解压的数学对应：按需还原
  - 八、RoPE 的挑战：为何要再加“位置贴纸”？
    - 8.1 RoPE：拍摄时间与 GPS 坐标
    - 8.2 分治策略： $\\boldsymbol{c}\_i$ + RoPE 小维度
  - 九、MLA 的综合优势：存储革命、灵活查询、时空保真
  - 十、工程视角：落地 MLA 时需注意的要点
    - 10.1 显存 VS. 推理速度
    - 10.2 RoPE 维度调参
    - 10.3 数值误差与精度
  - 十一、整体总结与展望
  - 十二、一个最小化 MLA 实现
  - 附录：核心公式与对应场景

- [点击查看答案](https://articles.zsxq.com/id_4lcum6gsda71.html)

## 十一、大模型（LLMs）langchain 面

### [大模型（LLMs）langchain 面](https://articles.zsxq.com/id_ve2dgaiqrjzv.html) 

- 一、什么是 LangChain?
- 二、LangChain 包含哪些 核心概念？
  - 2.1 LangChain 中 Components and Chains 是什么？
  - 2.2 LangChain 中 Prompt Templates and Values 是什么？
  - 2.3 LangChain 中 Example Selectors 是什么？
  - 2.4 LangChain 中 Output Parsers 是什么？
  - 2.5 LangChain 中 Indexes and Retrievers 是什么？
  - 2.6 LangChain 中  Chat Message History 是什么？
  - 2.7 LangChain 中  Agents and Toolkits 是什么？
- 三、什么是 LangChain Agent?
- 四、如何使用 LangChain ?
- 五、LangChain 支持哪些功能?
- 六、什么是 LangChain model?
- 七、LangChain 包含哪些特点?
- 八、LangChain 如何使用?
  - 8.1 LangChain 如何调用 LLMs 生成回复？
  - 8.2 LangChain 如何修改 提示模板？
  - 8.3 LangChain 如何链接多个组件处理一个特定的下游任务？
  - 8.4 LangChain 如何Embedding \& vector store？
- 九、LangChain 存在哪些问题及方法方案？
  - 9.1 LangChain 低效的令牌使用问题
  - 9.2 LangChain 文档的问题
  - 9.3 LangChain 太多概念容易混淆，过多的“辅助”函数问题
  - 9.4 LangChain 行为不一致并且隐藏细节问题
  - 9.5 LangChain 缺乏标准的可互操作数据类型问题
- 十、LangChain 替代方案？

- [点击查看答案](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### [多轮对话中让AI保持长期记忆的8种优化方式篇](https://articles.zsxq.com/id_3qgicwcwzjpi.html) 

- 一、前言
- 二、Agent 如何获取上下文对话信息？
  - 2.1 获取全量历史对话
  - 2.2 滑动窗口获取最近部分对话内容
  - 2.3 获取历史对话中实体信息
  - 2.4 利用知识图谱获取历史对话中的实体及其联系
  - 2.5 对历史对话进行阶段性总结摘要
  - 2.6 需要获取最新对话，又要兼顾较早历史对话
  - 2.7 回溯最近和最关键的对话信息
  - 2.8 基于向量检索对话信息

- [点击查看答案](https://articles.zsxq.com/id_3qgicwcwzjpi.html)

### [基于langchain RAG问答应用实战](https://articles.zsxq.com/id_3kw7snrk2rql.html) 

- [点击查看答案](https://articles.zsxq.com/id_3kw7snrk2rql.html)

## 十二、大模型（LLMs）RAG 检索增强生成面 

### 12.1 大模型（LLMs）RAG 入门篇

#### [基于LLM+向量库的文档对话 经验面](https://articles.zsxq.com/id_xk58m8ok2sob.html)

- 一、基于LLM+向量库的文档对话 基础面
  - 1.1 为什么 大模型 需要 外挂(向量)知识库？
  - 1.2. 基于LLM+向量库的文档对话 思路是怎么样？
  - 1.3. 基于LLM+向量库的文档对话 核心技术是什么？
  - 1.4. 基于LLM+向量库的文档对话 prompt 模板 如何构建？
- 二、基于LLM+向量库的文档对话 存在哪些痛点？
- 三、基于LLM+向量库的文档对话 工程示例面

- [点击查看答案](https://articles.zsxq.com/id_xk58m8ok2sob.html)

#### [RAG（Retrieval-Augmented Generation）面](https://articles.zsxq.com/id_xk58m8ok2sob.html) 

- 一、LLMs 已经具备了较强能力了，存在哪些不足点?
- 二、什么是 RAG?
  - 2.1 R：检索器模块
    - 2.1.1 如何获得准确的语义表示？
    - 2.1.2 如何协调查询和文档的语义空间？
    - 2.1.3 如何对齐检索模型的输出和大语言模型的偏好？
  - 2.2 G：生成器模块
    - 2.2.1 生成器介绍
    - 2.2.2 如何通过后检索处理提升检索结果？
    - 2.2.3 如何优化生成器应对输入数据？
- 三、使用 RAG 的好处?
- 四、RAG V.S. SFT
- 五、介绍一下 RAG 典型实现方法？
  - 5.1 如何 构建 数据索引？
  - 5.2 如何 对数据进行 检索（Retrieval）？
  - 5.3 对于 检索到的文本，如果生成正确回复？
- 六、介绍一下 RAG 典型案例？
- 七、RAG 存在什么问题？

- [点击查看答案](https://articles.zsxq.com/id_xk58m8ok2sob.html)

### 12.2 大模型（LLMs）RAG 版面分析篇

#### [大模型（LLMs）RAG —— pdf解析关键问题](https://articles.zsxq.com/id_2693k55it84w.html)

- 一、为什么需要进行pdf解析？
- 二、为什么需要 对 pdf 进行解析？
- 三、pdf解析 有哪些方法，对应的区别是什么？
- 四、pdf解析 存在哪些问题？
- 五、如何 长文档（书籍）中关键信息？
- 六、为什么要提取标题甚至是多级标题？
- 七、如何提取 文章标题？
- 八、如何区分单栏还是双栏pdf？如何重新排序？
- 九、如何提取表格和图片中的数据？
- 十、基于AI的文档解析有什么优缺点？

- [点击查看答案](https://articles.zsxq.com/id_2693k55it84w.html)

#### [大模型（LLMs）RAG 版面分析——表格识别方法篇](https://articles.zsxq.com/id_7x4qv94hxv8r.html)

- 一、为什么需要识别表格？
- 二、介绍一下 表格识别 任务？
- 三、有哪些 表格识别方法？
  - 3.1 传统方法
  - 3.2 pdfplumber表格抽取
    - 3.2.1 pdfplumber 如何进行 表格抽取？
    - 3.2.2 pdfplumber 常见的表格抽取模式？
  - 3.3 深度学习方法-语义分割
    - 3.3.1 table-ocr/table-detect：票据图片复杂表格框识别(票据单元格切割)
    - 3.3.2 腾讯表格图像识别
    - 3.3.3 TableNet
    - 3.3.4 CascadeTabNet
    - 3.3.5 SPLERGE
    - 3.3.6 DeepDeSRT

- [点击查看答案](https://articles.zsxq.com/id_7x4qv94hxv8r.html)

#### [大模型（LLMs）RAG 版面分析——文本分块面](https://articles.zsxq.com/id_iw7debl8akxh.html)

- 一、为什么需要对文本分块？
- 二、能不能介绍一下常见的文本分块方法？
  - 2.1 一般的文本分块方法
  - 2.2 正则拆分的文本分块方法
  - 2.3 Spacy Text Splitter 方法
  - 2.4 基于 langchain 的 CharacterTextSplitter 方法
  - 2.5 基于 langchain 的 递归字符切分 方法
  - 2.6 HTML 文本拆分 方法
  - 2.7 Mrrkdown 文本拆分 方法
  - 2.8 Python代码拆分 方法
  - 2.9 LaTex 文本拆分 方法

- [点击查看答案](https://articles.zsxq.com/id_iw7debl8akxh.html)

### 12.3 大模型（LLMs）RAG 检索策略篇

#### [大模型外挂知识库优化——如何利用大模型辅助召回？](https://articles.zsxq.com/id_oznm6qixjw61.html)

- 一、为什么需要使用大模型辅助召回？
  - 策略一： HYDE
    - 1. 介绍一下 HYDE 思路？
    - 2. 介绍一下 HYDE 问题？
  - 策略二： FLARE
    - 1. 为什么 需要 FLARE ？
    - 2. FLARE 有哪些召回策略？

- [点击查看答案](https://articles.zsxq.com/id_oznm6qixjw61.html)

#### [大模型外挂知识库优化——负样本样本挖掘篇](https://articles.zsxq.com/id_wa7nl8wsuilh.html)

- 一、为什么需要构建负难样本？
- 二、负难样本构建方法篇
  - 2.1 随机采样策略（Random Sampling）方法
  - 2.2 Top-K负例采样策略（Top-K Hard Negative Sampling）方法
  - 2.3 困惑负样本采样方法SimANS 方法
  - 2.4 利用 对比学习微调 方式构建负例方法
  - 2.5 基于批内负采样的对比学习方法
  - 2.6 相同文章采样方法
  - 2.7 LLM辅助生成软标签及蒸馏
- 辅助知识
  - 附一：梯度计算方法

- [点击查看答案](https://articles.zsxq.com/id_wa7nl8wsuilh.html)

### 12.4 大模型（LLMs）RAG 评测篇

#### [RAG（Retrieval-Augmented Generation）评测面](https://articles.zsxq.com/id_vjwt6uzml13l.html)

- 一、为什么需要 对 RAG 进行评测？
- 二、RAG 有哪些评估方法？
- 三、RAG 有哪些关键指标和能力？
- 四、RAG 有哪些评估框架？

- [点击查看答案](https://articles.zsxq.com/id_vjwt6uzml13l.html)

### 12.5 大模型（LLMs）RAG 优化策略篇

#### [检索增强生成(RAG) 优化策略篇](https://articles.zsxq.com/id_gu4p7gszsh82.html)

- 一、RAG基础功能篇
  - 1.1 RAG 工作流程
- 二、RAG 各模块有哪些优化策略？
- 三、RAG 架构优化有哪些优化策略？
  - 3.1 如何利用 知识图谱（KG）进行上下文增强？
    - 3.1.1 典型RAG架构中，向量数据库进行上下文增强 存在哪些问题？
    - 3.1.2 如何利用 知识图谱（KG）进行上下文增强？
  - 3.2 Self-RAG：如何让 大模型 对 召回结果 进行筛选？
    - 3.2.1 典型RAG架构中，向量数据库 存在哪些问题？
    - 3.2.2 Self-RAG：如何让 大模型 对 召回结果 进行筛选？
    - 3.2.3 Self-RAG 的 创新点是什么？
    - 3.2.4 Self-RAG 的 训练过程？
    - 3.2.5 Self-RAG 的 推理过程？
    - 3.2.6 Self-RAG 的 代码实战？
  - 3.3 多向量检索器多模态RAG篇
    - 3.3.1 如何让 RAG 支持 多模态数据格式？
      - 3.3.1.1 如何让 RAG 支持 半结构化RAG（文本+表格）?
      - 3.3.1.2 如何让 RAG 支持 多模态RAG（文本+表格+图片）?
      - 3.3.1.3 如何让 RAG 支持 私有化多模态RAG（文本+表格+图片）?
  - 3.4 RAG Fusion 优化策略
  - 3.5 模块化 RAG 优化策略
  - 3.6 RAG 新模式 优化策略
  - 3.7 RAG 结合 SFT
  - 3.8 查询转换（Query Transformations）
  - 3.9 bert在RAG中具体是起到了一个什么作用，我刚搜了下nsp的内容，但有点没法将这几者联系起来
- 四、RAG 索引优化有哪些优化策略？
  - 4.1 嵌入 优化策略
  - 4.2 RAG检索召回率低，一般都有哪些解决方案呀。尝试过不同大小的chunk，和混合检索。效果都不太好，然后优化？
  - 4.3 RAG 如何 优化索引结构?
  - 4.4 如何通过 混合检索 提升 RAG 效果?
  - 4.5 如何通过 重新排名 提升 RAG 效果?
- 五、RAG 索引数据优化有哪些优化策略？
  - 5.1 RAG 如何 提升索引数据的质量?
  - 5.2 如何通过添加元数据 提升 RAG 效果?
  - 5.3 如何通过 输入查询与文档对齐 提升 RAG 效果?
  - 5.4 如何通过 提示压缩 提升 RAG 效果?
  - 5.5 如何通过 查询重写和扩展 提升 RAG 效果?
- RAG 未来发展方向
- Rag 的垂直优化
- RAG 的水平扩展
- RAG 生态系统

- [点击查看答案](https://articles.zsxq.com/id_gu4p7gszsh82.html)

#### [RAG 关键痛点及对应解决方案](https://articles.zsxq.com/id_1bmbedojsj0t.html)

- 前言
- 问题一：内容缺失问题
  - 1.1 介绍一下 内容缺失问题？
  - 1.2 如何 解决 内容缺失问题？
- 问题二：错过排名靠前的文档
  - 2.1 介绍一下 错过排名靠前的文档 问题？
  - 2.2 如何 解决 错过排名靠前的文档 问题？
- 问题三：脱离上下文 — 整合策略的限制
  - 3.1 介绍一下 脱离上下文 — 整合策略的限制 问题？
  - 3.2 如何 解决 脱离上下文 — 整合策略的限制 问题？
- 问题四：未能提取答案
  - 4.1 介绍一下 未能提取答案 问题？
  - 4.2 如何 解决 未能提取答案 问题？
- 问题五：格式错误
  - 5.1 介绍一下 格式错误 问题？
  - 5.2 如何 解决 格式错误 问题？
- 问题六： 特异性错误
  - 6.1 介绍一下 特异性错误 问题？
  - 6.2 如何 解决 特异性错误 问题？
- 问题七： 回答不全面
  - 7.1 介绍一下 回答不全面 问题？
  - 7.2 如何 解决 回答不全面 问题？
- 问题八： 数据处理能力的挑战
  - 8.1 介绍一下 数据处理能力的挑战 问题？
  - 8.2 如何 解决 数据处理能力的挑战 问题？
- 问题九： 结构化数据查询的难题
  - 9.1 介绍一下 结构化数据查询的难题 问题？
  - 9.2 如何 解决 结构化数据查询的难题 问题？
- 问题十： 从复杂PDF文件中提取数据
  - 10.1 介绍一下 从复杂PDF文件中提取数据 问题？
  - 10.2 如何 解决 从复杂PDF文件中提取数据 问题？
- 问题十一： 备用模型
  - 11.1 介绍一下 备用模型 问题？
  - 11.2 如何 解决 备用模型 问题？
- 问题十二： 大语言模型（LLM）的安全挑战
  - 12.1 介绍一下 大语言模型（LLM）的安全挑战 问题？
  - 12.2 如何 解决 大语言模型（LLM）的安全挑战 问题？

- [点击查看答案](https://articles.zsxq.com/id_1bmbedojsj0t.html)

#### [大模型（LLMs）RAG 优化策略 —— RAG-Fusion篇](https://articles.zsxq.com/id_4ce04xwvic1z.html)

- 一、RAG 有哪些优点？
- 二、RAG 存在哪些局限性？
- 三、为什么 需要 RAG-Fusion？
- 四、说一下 RAG-Fusion 核心技术？
- 五、说一下 RAG-Fusion 工作流程？
  - 5.1 多查询生成
  - 5.2 多查询生成 技术实现（提示工程）？
  - 5.3 多查询生成 工作原理？
  - 5.4 逆向排名融合（RRF）
    - 5.4.1 为什么选择RRF？
    - 5.4.2 RRF 技术实现？
    - 5.4.3 生成性输出 用户意图保留
    - 5.4.4 生成性输出 用户意图保留 技术实现
- 六、RAG-Fusion 的优势和不足
  - 6.1 RAG-Fusion 优势
  - 6.2 RAG-Fusion 挑战

- [点击查看答案](https://articles.zsxq.com/id_4ce04xwvic1z.html)

### 12.6 大模型（LLMs）Graph RAG篇

#### [Graph RAG（Retrieval-Augmented Generation） 面 —— 一种 基于知识图谱的大模型检索增强实现策略](https://articles.zsxq.com/id_dwhonmw976n7.html)

- 一、为什么需要 Graph RAG？
- 二、什么是 Graph RAG？
- 三、Graph RAG 思路介绍？
- 四、用代码 介绍 Graph RAG ？
- 五、用 示例 介绍 Graph RAG ？
- 六、Graph RAG 排序优化方式？

- [点击查看答案](https://articles.zsxq.com/id_dwhonmw976n7.html)

## 十三、大模型（LLMs）agent 面 :fire:

### [大模型（LLMs）agent 面](https://articles.zsxq.com/id_le02luntesap.html) 

- 一、什么是 大模型（LLMs）agent？
- 二、大模型（LLMs）agent 有哪些部分组成？
  - 2.1 介绍一下 规划（planning）？
    - 2.1.1 拆解子目标和任务分解
      - 2.1.1.1 如何进行 拆解子目标和任务分解？
      - 2.1.1.2 拆解子目标和任务分解 有哪些方法？
    - 2.1.2 模型自我反省
      - 2.1.2.1 如何进行 模型自我反省？
      - 2.1.2.2 模型自我反省 有哪些方法？
  - 2.2 介绍一下 记忆（Memory）？
  - 2.3 介绍一下 工具使用（tool use）？
- 三、大模型（LLMs）agent 主要 利用了 大模型 哪些能力？
- 四、结合 代码 讲解 大模型（LLMs）agent 思路？
- 4.1 思路介绍
- 4.2 实例一：利用大模型判断做选择
- 4.3 实例二：让大模型通过判断正确选择函数工具并输出
- 4.4 实例三：agent模板和解析
- 4.5 实例四：将 skylark 接入 langchain 中测试 agent
- 五、如何给LLM注入领域知识？
- 六、常见LLM Agent框架或者应用 有哪些？

- [点击查看答案](https://articles.zsxq.com/id_le02luntesap.html)

### 函数调用 Function Call 篇

- [函数调用 Function Call 篇](https://articles.zsxq.com/id_asxg09gtrx89.html)
  - 一、为什么需要 函数调用(function call)？
  - 二、什么是 函数调用(function call)？
  - 三、函数调用(function-call)目的是什么？
  - 四、怎么样使用函数调用？
  - 五、如何使用API完成函数调用？
    - 5.1 如何使用 函数调用(function-call) 构建 新闻机器人？

- [点击查看答案](https://articles.zsxq.com/id_asxg09gtrx89.html)

- [开源模型 Function Call 篇](https://articles.zsxq.com/id_s2ojkzdw83gb.html)
  - 开源模型 Function Call 方案有哪些？
    - Llama 3.1
      - 对话协议（Chat Protocal）
      - Tool Call Template 样式
      - 样式1：Json based tool calling
      - 样式2：Built in Python based tool calling
      - 工具调用相关的训练
        - SFT 相关训练
        - 数据
          - single-step tool use 数据
          - Multi-step tool use 数据
          - File uploads 数据
        - DPO 相关训练
      - 其他
    - Mistral Large 2
      - 关注点：
    - glm-4-9b-chat
      - 效果如下
      - 其他
    - Qwen 2

- [点击查看答案](https://articles.zsxq.com/id_s2ojkzdw83gb.html)

## 十四、大模型——角色扮演大模型篇

### [大模型——角色扮演大模型篇](https://articles.zsxq.com/id_16kl2onmsf8t.html)

- 大模型——角色扮演大模型篇
  - 一、什么是角色扮演大模型？
  - 二、为什么需要角色扮演大模型？
  - 三、角色扮演大模型 相比于 通用大模型 具有哪些区别？
  - 四、能否通俗易懂的介绍 【角色扮演大模型】？
  - 五、有哪些策略可以提升【角色扮演大模型】能力？
  - 六、角色扮演类的产品或模型可以分为哪些方向？
  - 七、为什么超拟人和模型基础能力存在矛盾？具体指哪些？如何解决？
  - 八、为什么真人说话风格难以复刻？有什么好的解决方法
  - 九、长多轮对话能力开盒\&上下文一致性问题
  - 十、如何提升拟人能力？

- [点击查看答案](https://articles.zsxq.com/id_16kl2onmsf8t.html)

## 十五、大模型幻觉（LLM Hallucination）面 

### [大模型幻觉（LLM Hallucination）面](https://articles.zsxq.com/id_schwrdmvmhr7.html)

- 一、什么是大模型幻觉？
- 二、为什么LLM会产生幻觉？
- 三、为什么需要解决LLM的幻觉问题？
- 四、幻觉一定是有害的吗？
- 五、幻觉有哪些不同类型？
- 六、如何度量幻觉？
- 七、如何缓解LLM幻觉？
  - 7.1 通过使用外部知识验证主动检测和减轻幻觉
  - 7.2 事实核心采样
  - 7.3 SelfCheckGPT
- 八、LLMs什么时候最容易产生幻觉？

- [点击查看答案](https://articles.zsxq.com/id_schwrdmvmhr7.html)

### [大模型的幻觉问题篇](https://articles.zsxq.com/id_8mr4mlhe5q1x.html)

- 一、什么是 大模型幻觉问题？
- 二、为什么 会 出现 大模型幻觉问题？
- 三、如何 评估 大模型幻觉问题？
- 四、如何 缓解 大模型幻觉问题？

- [点击查看答案](https://articles.zsxq.com/id_8mr4mlhe5q1x.html)

### [如何缓解大模型幻觉？](https://articles.zsxq.com/id_tbezgzifowzp.html)

- 一、为什么 会 出现 大模型幻觉？
- 二、如何 缓解 大模型幻觉？

- [点击查看答案](https://articles.zsxq.com/id_tbezgzifowzp.html)

## 十六、思维链 Chain-of-Thought（COT）篇 

### [思维链 Chain-of-Thought（COT）篇](https://articles.zsxq.com/id_c0jpjo7q95wg.html)

- 一、什么是思维链提示？
- 二、思维链提示本质是什么？
- 三、思维链提示 与 标准的提示学习方法有什么不同?
- 四、思维链提示 为什么可以提高语言模型的复杂推理能力?它的优势在哪里?
- 五、思维链提示 适用场景 有 哪些？
- 六、思维链提示 目前还存在哪些不足点？
- 七、思维链提示 对推动语言模型复杂推理能力研究有哪些启发和影响?
- 八、思维链提示 对实现真正的通用人工智能仍面临哪些挑战?
- 九、如何通过增加模型规模来获得语言模型强大的思路链推理能力的?这与模型获得的哪些能力有关?
- 十、你认为可以在哪些其他方面应用“思路链提示”这一思路来提升语言模型的能力?
- 十一、如果需要你对 思维链提示 进行改进，你觉得你会改进哪些地方？
- 十二、思维链提示 未来研究方向？
- 十三、常见面试题
  - 13.1 对COT(Chain-of-Thought Prompt)和Instruction Tuning的理解？

- [点击查看答案](https://articles.zsxq.com/id_c0jpjo7q95wg.html)

### [思维链 Chain-of-Thought（COT）变体篇](https://articles.zsxq.com/id_thdljw9vgxt1.html)

- 思维链 Chain-of-Thought（COT）：思维链的启蒙
  - 1. 什么是 思维链 Chain-of-Thought（COT）？
  - 2. 思维链 Chain-of-Thought（COT）是思路是什么？
  - 3. 思维链 Chain-of-Thought（COT）存在问题？
- 思维树 Tree of Thoughts（TOT）：一种用树结构解决复杂问题的方法
  - 1. 为什么需要 思维树 Tree of Thoughts（TOT）？
  - 2. 什么是 思维树 Tree of Thoughts（TOT）？
  - 3. 思维树 Tree of Thoughts（TOT）涉及问题有哪些？
- 思维图 Graph of Thoughts（GOT）：一种把思维链过程建模层图结构的方法
  - 1. 为什么 需要 思维图 Graph of Thoughts（GOT）？
  - 2. 什么是 思维图 Graph of Thoughts（GOT） ？
  - 3. 思维图 Graph of Thoughts（GOT）核心思想是什么 ？
- 思维算法 Algorithm of Thoughts（AOT）：一种用DFS/BFS示例解决问题的方法
  - 1. 为什么 需要 思维算法 Algorithm of Thoughts（AOT）？
  - 2. 思维算法 Algorithm of Thoughts（AOT）思路是什么？
  - 3. 思维算法 Algorithm of Thoughts（AOT） vs 其他 COT 的 区别？
- 思维链 Chain-of-Thought（COT） 有哪些 应用场景？
- 思维链 Chain-of-Thought（COT） 有哪些 局限性？

- [点击查看答案](https://articles.zsxq.com/id_thdljw9vgxt1.html)

### [小样本提示学习篇](https://articles.zsxq.com/id_re6ap2lq88gw.html) 

- 一、什么是Zero-shot提示方法？
- 二、什么是Few-shot提示方法？
- 三、阐述One-shot和Few-shot提示策略及其应用场景？
- 四、什么是逐步Zero-shot
- 五、定义Zero-shot-CoT提示策略并描述其应用方法？
- 六、解释Few-shot-CoT提示策略及其实际使用方式？
- 七、Few-shot-LtM策略包含哪些主要阶段及其职责？

- [点击查看答案](https://articles.zsxq.com/id_re6ap2lq88gw.html)

## [十七、MOE（Mixture-of-Experts）篇](https://articles.zsxq.com/id_w6ebwrhprpj9.html) :fire:

### [MOE（Mixture-of-Experts）篇](https://articles.zsxq.com/id_5anfhj9qoh2v.html)

- 一、为什么需要 MOE（Mixture-of-Experts）？
- 二、MOE（Mixture-of-Experts）的思路是什么样的？
- 三、介绍一下 MOE（Mixture-of-Experts）分布式并行策略？
  - 3.1 MOE + 数据并行?
  - 3.2 MOE + 模型并行?
- 四、MoE大模型具备哪些优势？
- 五、MoE大模型具备哪些缺点？
- 六、MoE为什么可以实现更大模型参数、更低训练成本？
- 七、MoE如何解决训练稳定性问题？
- 八、MoE如何解决Fine-Tuning过程中的过拟合问题？
- 九、MOE算法在训练大语言模型时有哪些应用场景?
- 十、有哪些MoE模型？
- 十一、介绍稀疏 MoE 层
- 十二、介绍门控网络或路由
- 十三、为什么门控网络要引入噪声呢
- 十四、如何均衡专家间的负载
- 十五、“专家”指什么
- 十六、专家的数量对预训练有何影响？
- 十七、什么是topK门控
- 十八、MoE模型的主要特点
- 十九、moe和稠密模型的对比
- 二十、如何微调moe？

- [点击查看答案](https://articles.zsxq.com/id_5anfhj9qoh2v.html)

### [MOE大模型对比篇](https://articles.zsxq.com/id_j51bnu3xfgm9.html)

- DeepSpeed-MoE
- PAI-Megatron-Patch MoE
  
- [点击查看答案](https://articles.zsxq.com/id_j51bnu3xfgm9.html)

### [MoE 大模型负载均衡策略演进的回顾](https://articles.zsxq.com/id_5anfhj9qoh2v.html)

- MoE 大模型负载均衡策略演进的回顾
  - 前言
    - 为什么要用稀疏专家（Sparse MoE）？
    - 这篇博文要讨论些什么？
    - 我注意到的一些关键主题
  - 一、历史脉络：从 GShard 到 Switch
    - 1.1 GShard：先锋之作
      - 1.1.1 介绍一下 GShard？
      - 1.1.2 介绍一下 GShard 痛点？
    - 1.2 Switch Transformer：当“少就是多”
      - 1.2.1 介绍一下 Switch Transformer？
      - 1.2.2 介绍一下 Switch Transformer 利弊？
  - 二、进一步改进与变体：GLaM、DeepSpeed-MoE、ST-MoE、Mixtral
    - 2.1 GLaM：带着效率回归 Top-2
      - 2.1.1 介绍一下 GLaM？
      - 2.1.2 介绍一下 GLaM 坑与经验？
    - 2.2 DeepSpeed-MoE：主打推理效率
      - 2.2.1 介绍一下 DeepSpeed-MoE？
      - 2.2.2 介绍一下 DeepSpeed-MoE 核心思路？
      - 2.2.3 介绍一下 DeepSpeed-MoE 跨 GPU 的负载均衡？
      - 2.2.4 介绍一下 DeepSpeed-MoE 跨 GPU 的痛点与教训？
    - 2.3 ST-MoE：聚焦容量因子与路由器 Z-Loss
      - 2.3.1 介绍一下 ST-MoE？
      - 2.3.2 介绍一下 ST-MoE 亮点？
      - 2.3.3 介绍一下 ST-MoE 容量因子调优？
    - 2.4 Mixtral 8x7B：时间局部性与专门的稀疏 Kernel
      - 2.4.1 介绍一下 Mixtral 8x7B？
      - 2.4.2 介绍一下 Mixtral 8x7B 时间局部性？
      - 2.4.3 介绍一下 Mixtral 8x7B 稀疏 Kernel 优化？
      - 2.4.4 介绍一下 Mixtral 8x7B 经验？
  - 三、新一代方案：OpenMoE、DeepSeekMoE、JetMoE、DeepSeek-V3 等等
    - 3.1 OpenMoE：上下文无关的“专长化”与末端 Token 的“掉队”问题
      - 3.1.1 介绍 OpenMoE？
      - 3.1.2 介绍 OpenMoE 结论？
    - 3.2 DeepSeekMoE：细粒度专家与共享专家
      - 3.2.1 介绍 DeepSeekMoE？
      - 3.2.2 介绍 DeepSeekMoE 的 细粒度专家拆分 (Fine-Grained Expert Segmentation)？
      - 3.2.3 介绍 DeepSeekMoE 的 两级负载均衡损失？
    - 3.3 JetMoE：无 Token 丢弃的 MoE 与流水线并行
      - 3.3.1 介绍一下 JetMoE？
      - 3.3.2 介绍一下 JetMoE 经验教训？
    - 3.4 Skywork-MoE：gating logit 归一化 \& 自适应辅助损失
      - 3.4.1 介绍一下 Skywork-MoE？
      - 3.4.2 介绍一下 Skywork-MoE 痛点？
    - 3.5 DeepSeek-V3：偏置加成与弱化辅助损失
      - 3.5.1 介绍一下 DeepSeek-V3？
      - 3.5.2 介绍一下 DeepSeek-V3 风险与启示？
  - 四、趋势与总结
  - 五、经历的坑和总结的教训

- [点击查看答案](https://articles.zsxq.com/id_1bs9p5c56ku4.html)

## 十八、大模型蒸馏篇 :fire:

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

## 十九、大模型（LLMs）强化学习面 

### [大模型（LLMs）强化学习面](https://articles.zsxq.com/id_20xnfnoprj9s.html) 

- 1 简单介绍强化学习？
- 2 简单介绍一下 RLHF？
- 3 奖励模型需要和基础模型一致吗？
- 4 RLHF 在实践过程中存在哪些不足？
- 5 如何解决 人工产生的偏好数据集成本较高，很难量产问题？
- 6 如何解决三个阶段的训练（SFT-\>RM-\>PPO）过程较长，更新迭代较慢问题？
- 7 如何解决 PPO 的训练过程同时存在4个模型（2训练，2推理），对计算资源的要求较高 问题？
- 8 强化学习跟大语言模型的本质联系是什么？
- 9 大语言模型与强化学习的本质联系？
- 10 GPT 到底是基于概率的还是基于价值?
- 11 强化学习究竟是如何与大语言模型做结合的?
- 12 强化学习与大模型的结合在公式中是如何体现?
- 13 强化学习与大模型的结合的好处?
- 14 reward 模型训练步骤中，为什么这一步骤在标注数据过程中不让人直接打分，而是去标排列序列呢?
- 15 reward 模型的 loss 是怎么计算的?
- 16 直接用训练 reward model 的数据精调模型，而不用强化学习，是否可行?为什么?
  - 16.1 reward模型、强化学习、SFT分别的作用?
  - 16.2 reward 模型、强化学习、SFT之间如何配合?
- 17 假如 reward model 不太准，怎么办?
- 18 chatGPT 强化学习训练阶段还有什么改进的空间和思路吗?
- 19 现阶段LLM的对齐阶段分为sft和rlhf阶段，我们可以跳过sft阶段直接进行rlhf么？

- [点击查看答案](https://articles.zsxq.com/id_20xnfnoprj9s.html)

### [大模型（LLMs）强化学习——RLHF及其变种面](https://articles.zsxq.com/id_3ct6sw0wouna.html) 

- 一、介绍一下 LLM的经典预训练Pipeline？
- 二、预训练（Pre-training）篇
  - 2.1 具体介绍一下 预训练（Pre-training）？
- 三、有监督微调（Supervised Tinetuning）篇
  - 3.1 具体介绍一下 有监督微调（Supervised Tinetuning）？
  - 3.2 有监督微调（Supervised Tinetuning）的训练数据格式是什么样？
  - 3.3 预训练（Pre-training） vs 有监督微调（Supervised Tinetuning）区别？
- 四、对齐（Alignment）篇
  - 4.1 简单介绍一下 对齐（Alignment）？
- 五、Reinforcement Learning with Human Feedback (RLHF)篇
  - 5.1 简单介绍一下 RLHF 流程？
  - 5.2 如何在在预训练好的模型上进行有监督微调？
  - 5.3 如何在有监督微调模型基础上创建一个RM模型？
  - 5.4 如何基于RM模型使用PPO算法微调SFT模型？
  - 5.5 instructGPT的原理，讲讲rlhf和reward？
- 六、LLaMA 2 的 RLHF 篇
  - 6.1 介绍一下 LLaMA 2 的 RLHF？
  - 6.2 LLaMA 2 中 Margin Loss 的 实现逻辑？
  - 6.3 LLaMA 2 中 两个RM模型 的 实现逻辑？
  - 6.4 LLaMA 2 中 拒绝采样 逻辑？
- 七、 RLHF 替代方案篇
  - 7.1 为什么需要 RLHF 替代方案？
  - 7.2 RLHF 有哪些替代方案？
- 八、 RLHF 实践篇
  - 8.1 RLHF 训练过程，怎么选取最优 checkpoint？

- [点击查看答案](https://articles.zsxq.com/id_3ct6sw0wouna.html)

### [大模型（LLMs）强化学习—— PPO 面](https://articles.zsxq.com/id_s8kwqw1gowvh.html) 

  - 一、大语言模型RLHF中的PPO主要分哪些步骤？
  - 二、举例描述一下 大语言模型的RLHF？
  - 三、大语言模型RLHF 采样篇
    - 3.1 什么是 PPO 中 采样过程？
    - 3.2 介绍一下 PPO 中 采样策略？
    - 3.3 PPO 中 采样策略中，如何评估“收益”？
  - 四、在PPO过程中，reward model的效果上会有什么问题？
  - 五、如何解决reward model的OOD的问题？
  - 六、RLHF中PPO有什么问题，为什么大家都设计很多方法去替代它？
  - 七、PPO 思路介绍？
  - 八、介绍在线PPO训练？
  - 九、介绍PPO模块构成
  - 十、在训练过程中，如果奖励曲线出现剧烈抖动，可以考虑以下几个因素对此进行优化？

- [点击查看答案](https://articles.zsxq.com/id_s8kwqw1gowvh.html)

### [RLHF平替算法DPO篇](https://articles.zsxq.com/id_mlq44r1p7nob.html) 

- RLHF平替算法DPO篇
  - 一、DPO vs RLHF？
  - 二、介绍一下 DPO的损失函数？
  - 三、DPO 微调流程 ?
  - 四、说一下 DPO 是如何简化 RLHF 的？
  - 五、DPO的第0步loss是固定的么？如果固定的话，值是多少？
  - 六、DPO是一个on-policy还是off-policy的算法，以及这样的算法有什么优劣？
  - 七、DPO公式是由PPO的objective公式推导过来的，为什么DPO是off-policy算法，而PPO是on-policy算法，到底哪一步推导出了问题？
  - 八、DPO为什么会在学习过程中training positive的概率和training negative的概率都同时下降？
  - 九、在什么情况下DPO exactly 数学上等同于 PPO？
  - 十、DPO的变体有哪些，主要解决DPO的什么问题？
  - 十一、DPO训练后的模型为什么会输出越来越长？
  - 十二、DPO训练可能会出现什么问题？
  - 十三、讲一下DPO和PPO，DPO和PPO有什么区别？
  - 代码解释
    - 1. DPO损失函数 代码实现？
    - 2. DPO 批次训练过程 代码实现？
    - 3. LM的交叉熵计算 代码实现？

- [点击查看答案](https://articles.zsxq.com/id_mlq44r1p7nob.html)

### [reward 篇](https://articles.zsxq.com/id_vblb0j5qnaxg.html) 

  - 1 介绍一下 RM模型？
  - 2 为什么需要 RM模型？
  - 3 RM模型训练数据如何构建？
  - 4 reward 模型训练步骤中，为什么这一步骤在标注数据过程中不让人直接打分，而是去标排列序列呢?
  - 5 reward 模型的 loss 是怎么计算的?
  - 5 直接用训练 reward model 的数据精调模型，而不用强化学习，是否可行?为什么?
    - 5.1 reward模型、强化学习、SFT分别的作用?
    - 5.2 reward 模型、强化学习、SFT之间如何配合?
  - 6 假如 reward model 不太准，怎么办?
  - 7 使用多少数据能够训练好一个RM？
  - 8 RM 模型的大小限制？
  - 9 RM 模型架构是怎么样？
  - 10 奖励模型的损失函数为什么会比较答案的排序，而不是去对每一个答案的具体分数做一个回归？
  - 11 奖励模型中每个问题对应的答案数量即K值为什么选 9 更合适，而不是选择 4 呢？

- [点击查看答案](https://articles.zsxq.com/id_vblb0j5qnaxg.html)

### [强化学习在自然语言处理下的应用篇](https://articles.zsxq.com/id_5tsn84l32eea.html) 

- 一、强化学习基础面
  - 1.1 介绍一下强化学习？
  - 1.2 介绍一下强化学习 的 状态（States） 和 观测（Observations）？
  - 1.3 强化学习 有哪些 动作空间（Action Spaces），他们之间的区别是什么？
  - 1.4 强化学习 有哪些 Policy策略？
  - 1.5 介绍一下 强化学习 的 轨迹？
  - 1.6 介绍一下 强化学习 的 奖赏函数？
  - 1.7 介绍一下 强化学习问题？
- 二、RL发展路径（至PPO）
  - 2.1 介绍一下 强化学习 中 优化方法 Value-based？
  - 2.2 介绍一下 强化学习 中 贝尔曼方程？
  - 2.3 介绍一下 强化学习 中 优势函数Advantage Functions？

- [点击查看答案](https://articles.zsxq.com/id_5tsn84l32eea.html)

## [二十、大模型（LLMs）评测面](https://articles.zsxq.com/id_j9wcj62eovgc.html)

- 1 大模型怎么评测？
- 2 大模型的honest原则是如何实现的？模型如何判断回答的知识是训练过的已知的知识，怎么训练这种能力？
- 3 如何衡量大模型水平？
- 4 大模型评估方法 有哪些？
- 5 大模型评估工具 有哪些？
- 6介绍一下 困惑度？
- 7 大模型训练的性能指标:吞吐率 Throughput 是指什么?
- 8 如何评价大语言模型的训练数据的质量?

- [点击查看答案](https://articles.zsxq.com/id_j9wcj62eovgc.html)

## 二十一、大模型（LLMs）强化学习面 

### [大模型（LLMs）强化学习面](https://articles.zsxq.com/id_20xnfnoprj9s.html) 

- 1 简单介绍强化学习？
- 2 简单介绍一下 RLHF？
- 3 奖励模型需要和基础模型一致吗？
- 4 RLHF 在实践过程中存在哪些不足？
- 5 如何解决 人工产生的偏好数据集成本较高，很难量产问题？
- 6 如何解决三个阶段的训练（SFT-\>RM-\>PPO）过程较长，更新迭代较慢问题？
- 7 如何解决 PPO 的训练过程同时存在4个模型（2训练，2推理），对计算资源的要求较高 问题？
- 8 强化学习跟大语言模型的本质联系是什么？
- 9 大语言模型与强化学习的本质联系？
- 10 GPT 到底是基于概率的还是基于价值?
- 11 强化学习究竟是如何与大语言模型做结合的?
- 12 强化学习与大模型的结合在公式中是如何体现?
- 13 强化学习与大模型的结合的好处?
- 14 reward 模型训练步骤中，为什么这一步骤在标注数据过程中不让人直接打分，而是去标排列序列呢?
- 15 reward 模型的 loss 是怎么计算的?
- 16 直接用训练 reward model 的数据精调模型，而不用强化学习，是否可行?为什么?
  - 16.1 reward模型、强化学习、SFT分别的作用?
  - 16.2 reward 模型、强化学习、SFT之间如何配合?
- 17 假如 reward model 不太准，怎么办?
- 18 chatGPT 强化学习训练阶段还有什么改进的空间和思路吗?
- 19 现阶段LLM的对齐阶段分为sft和rlhf阶段，我们可以跳过sft阶段直接进行rlhf么？

- [点击查看答案](https://articles.zsxq.com/id_20xnfnoprj9s.html)

### [大模型（LLMs）强化学习——RLHF及其变种面](https://articles.zsxq.com/id_3ct6sw0wouna.html) 

- 一、介绍一下 LLM的经典预训练Pipeline？
- 二、预训练（Pre-training）篇
  - 2.1 具体介绍一下 预训练（Pre-training）？
- 三、有监督微调（Supervised Tinetuning）篇
  - 3.1 具体介绍一下 有监督微调（Supervised Tinetuning）？
  - 3.2 有监督微调（Supervised Tinetuning）的训练数据格式是什么样？
  - 3.3 预训练（Pre-training） vs 有监督微调（Supervised Tinetuning）区别？
- 四、对齐（Alignment）篇
  - 4.1 简单介绍一下 对齐（Alignment）？
- 五、Reinforcement Learning with Human Feedback (RLHF)篇
  - 5.1 简单介绍一下 RLHF 流程？
  - 5.2 如何在在预训练好的模型上进行有监督微调？
  - 5.3 如何在有监督微调模型基础上创建一个RM模型？
  - 5.4 如何基于RM模型使用PPO算法微调SFT模型？
  - 5.5 instructGPT的原理，讲讲rlhf和reward？
- 六、LLaMA 2 的 RLHF 篇
  - 6.1 介绍一下 LLaMA 2 的 RLHF？
  - 6.2 LLaMA 2 中 Margin Loss 的 实现逻辑？
  - 6.3 LLaMA 2 中 两个RM模型 的 实现逻辑？
  - 6.4 LLaMA 2 中 拒绝采样 逻辑？
- 七、 RLHF 替代方案篇
  - 7.1 为什么需要 RLHF 替代方案？
  - 7.2 RLHF 有哪些替代方案？
- 八、 RLHF 实践篇
  - 8.1 RLHF 训练过程，怎么选取最优 checkpoint？

- [点击查看答案](https://articles.zsxq.com/id_3ct6sw0wouna.html)

### [大模型（LLMs）强化学习—— PPO 面](https://articles.zsxq.com/id_s8kwqw1gowvh.html) 

  - 一、大语言模型RLHF中的PPO主要分哪些步骤？
  - 二、举例描述一下 大语言模型的RLHF？
  - 三、大语言模型RLHF 采样篇
    - 3.1 什么是 PPO 中 采样过程？
    - 3.2 介绍一下 PPO 中 采样策略？
    - 3.3 PPO 中 采样策略中，如何评估“收益”？
  - 四、在PPO过程中，reward model的效果上会有什么问题？
  - 五、如何解决reward model的OOD的问题？
  - 六、RLHF中PPO有什么问题，为什么大家都设计很多方法去替代它？
  - 七、PPO 思路介绍？
  - 八、介绍在线PPO训练？
  - 九、介绍PPO模块构成
  - 十、在训练过程中，如果奖励曲线出现剧烈抖动，可以考虑以下几个因素对此进行优化？

- [点击查看答案](https://articles.zsxq.com/id_s8kwqw1gowvh.html)

### [RLHF平替算法DPO篇](https://articles.zsxq.com/id_mlq44r1p7nob.html) 

- RLHF平替算法DPO篇
  - 一、DPO vs RLHF？
  - 二、介绍一下 DPO的损失函数？
  - 三、DPO 微调流程 ?
  - 四、说一下 DPO 是如何简化 RLHF 的？
  - 五、DPO的第0步loss是固定的么？如果固定的话，值是多少？
  - 六、DPO是一个on-policy还是off-policy的算法，以及这样的算法有什么优劣？
  - 七、DPO公式是由PPO的objective公式推导过来的，为什么DPO是off-policy算法，而PPO是on-policy算法，到底哪一步推导出了问题？
  - 八、DPO为什么会在学习过程中training positive的概率和training negative的概率都同时下降？
  - 九、在什么情况下DPO exactly 数学上等同于 PPO？
  - 十、DPO的变体有哪些，主要解决DPO的什么问题？
  - 十一、DPO训练后的模型为什么会输出越来越长？
  - 十二、DPO训练可能会出现什么问题？
  - 十三、讲一下DPO和PPO，DPO和PPO有什么区别？
  - 代码解释
    - 1. DPO损失函数 代码实现？
    - 2. DPO 批次训练过程 代码实现？
    - 3. LM的交叉熵计算 代码实现？

- [点击查看答案](https://articles.zsxq.com/id_mlq44r1p7nob.html)

### [reward 篇](https://articles.zsxq.com/id_vblb0j5qnaxg.html) 

  - 1 介绍一下 RM模型？
  - 2 为什么需要 RM模型？
  - 3 RM模型训练数据如何构建？
  - 4 reward 模型训练步骤中，为什么这一步骤在标注数据过程中不让人直接打分，而是去标排列序列呢?
  - 5 reward 模型的 loss 是怎么计算的?
  - 5 直接用训练 reward model 的数据精调模型，而不用强化学习，是否可行?为什么?
    - 5.1 reward模型、强化学习、SFT分别的作用?
    - 5.2 reward 模型、强化学习、SFT之间如何配合?
  - 6 假如 reward model 不太准，怎么办?
  - 7 使用多少数据能够训练好一个RM？
  - 8 RM 模型的大小限制？
  - 9 RM 模型架构是怎么样？
  - 10 奖励模型的损失函数为什么会比较答案的排序，而不是去对每一个答案的具体分数做一个回归？
  - 11 奖励模型中每个问题对应的答案数量即K值为什么选 9 更合适，而不是选择 4 呢？

- [点击查看答案](https://articles.zsxq.com/id_vblb0j5qnaxg.html)

### [强化学习在自然语言处理下的应用篇](https://articles.zsxq.com/id_5tsn84l32eea.html) 

- 一、强化学习基础面
  - 1.1 介绍一下强化学习？
  - 1.2 介绍一下强化学习 的 状态（States） 和 观测（Observations）？
  - 1.3 强化学习 有哪些 动作空间（Action Spaces），他们之间的区别是什么？
  - 1.4 强化学习 有哪些 Policy策略？
  - 1.5 介绍一下 强化学习 的 轨迹？
  - 1.6 介绍一下 强化学习 的 奖赏函数？
  - 1.7 介绍一下 强化学习问题？
- 二、RL发展路径（至PPO）
  - 2.1 介绍一下 强化学习 中 优化方法 Value-based？
  - 2.2 介绍一下 强化学习 中 贝尔曼方程？
  - 2.3 介绍一下 强化学习 中 优势函数Advantage Functions？

- [点击查看答案](https://articles.zsxq.com/id_5tsn84l32eea.html)

## [二十二、LLMs 位置编码篇](https://articles.zsxq.com/id_amt4qkusdcir.html) 

- 一、什么是位置编码？
- 二、为什么需要位置编码？
- 三、什么是绝对位置编码？
  - 3.1 训练式位置编码篇
    - 3.1.1 什么是 训练式位置编码？
    - 3.1.2 如何为每个位置的词向量注入位置信息呢？
    - 3.1.3 训练式位置编码篇 应用场景？
    - 3.1.4 训练式位置编码篇 存在哪些问题？
  - 3.2 Sinusoidal位置编码篇
    - 3.2.1 什么是 Sinusoidal位置编码？
    - 3.2.2 Sinusoidal位置编码 有哪些优点？
- 四、什么是相对位置编码？
- 五、旋转位置编码 RoPE篇
  - 5.1 旋转位置编码 RoPE 思路是什么？
  - 5.2 推导一下 旋转位置编码 RoPE ？
  - 5.3 旋转位置编码 RoPE 有什么优点？
  - 5.4 旋转位置编码 RoPE 被哪些 LLMs 应用？
- 六、长度外推问题篇
  - 6.1 什么是 长度外推问题？
  - 6.2 长度外推问题 的 解决方法 有哪些？
- 七、 ALiBi (Attention with Linear Biases)篇
  - 7.1 ALiBi (Attention with Linear Biases) 思路是什么？
  - 7.2 ALiBi (Attention with Linear Biases) 的偏置矩阵是什么？有什么作用？
  - 7.3 ALiBi (Attention with Linear Biases) 有什么优点？
  - 7.4 ALiBi (Attention with Linear Biases)  被哪些 LLMs 应用？

- [点击查看答案](https://articles.zsxq.com/id_amt4qkusdcir.html)


## 二十三、大模型（LLMs）训练集面 

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

## 二十四、大模型（LLMs）显存问题面 

### [大模型（LLMs）显存问题面](https://articles.zsxq.com/id_jhiocx89p3su.html)

1. 大模型大概有多大，模型文件有多大?
2. 能否用4 * v100 32G训练vicuna 65b？
3. 如果就是想要试试65b模型，但是显存不多怎么办？
4. nB模型推理需要多少显存？
5. nB模型训练需要多少显存？
6. 如何 估算模型所需的RAM？
7. 如何评估你的显卡利用率?
8. 测试你的显卡利用率 实现细节篇
   1. 如何查看多机训练时的网速？
   2. 如何查看服务器上的多卡之间的NVLINK topo？
   3. 如何查看服务器上显卡的具体型号?
   4. 如何查看训练时的flops？（也就是每秒的计算量）
   5. 如何查看对deepspeed的环境配置是否正确？
   6. tf32格式有多长？
   7. 哪里看各类显卡算力比较？
   8. （torch profiler）如何查看自己的训练中通信开销？

- [点击查看答案](https://articles.zsxq.com/id_jhiocx89p3su.html)

### [大模型（LLMs）显存优化策略篇](https://articles.zsxq.com/id_a1l60awgge6q.html)

- 一、介绍一下 gradient accumulation 显存优化方式？
- 二、介绍一下 gradient checkpointing 显存优化方式？

- [点击查看答案](https://articles.zsxq.com/id_a1l60awgge6q.html)

## 二十五、LLMs Tokenizer 篇

### [LLMs Tokenizer 篇](https://articles.zsxq.com/id_6z8ptdwqeid7.html)

- LLMs Tokenizer 篇
  - Byte-Pair Encoding(BPE)篇
    - 1 介绍一下 Byte-Pair Encoding(BPE) ？
    - 2 Byte-Pair Encoding(BPE) 如何构建词典？
    - 3 Byte-Pair Encoding(BPE) 具有什么优点？
    - 4 Byte-Pair Encoding(BPE) 具有什么缺点？
    - 5 手撕 Byte-Pair Encoding(BPE) ？
  - Byte-level BPE 篇
    - 1 介绍一下 Byte-level BPE ？
    - 2 Byte-level BPE 如何构建词典？
    - 3 Byte-level BPE 具有什么优点？
    - 4 Byte-level BPE 具有什么缺点？
  - WordPiece 篇
    - 1 介绍一下 WordPiece ？
    - 2 WordPiece 如何构建词典？
    - 3 WordPiece 具有什么优点？
    - 4 WordPiece 具有什么缺点？
    - 5 手撕 WordPiece ？
    - 6 WordPiece 与 BPE 异同点是什么？
  - SentencePiece 篇
    - 1 介绍一下 SentencePiece ？
    - 2 SentencePiece 如何构建词典？
    - 3 SentencePiece 具有什么优点？
    - 4 SentencePiece 具有什么缺点？
    - 5 手撕 SentencePiece ？
  - Transformer Tokenizer 篇
    - 1 介绍一下 Transformer Tokenizer ？
    - 2 Transformer Tokenizer 如何构建词典？
    - 3 Transformer Tokenizer 具有什么优点？
    - 4 Transformer Tokenizer 具有什么缺点？
    - 5 手撕 Transformer Tokenizer ？
  - Unigram LM 篇
    - 1 介绍一下 Unigram LM ？
    - 2 Unigram LM 如何构建词典？
    - 3 Unigram LM 具有什么优点？
    - 4 Unigram LM 具有什么缺点？
    - 5 手撕 Unigram LM ？
  - 对比篇
    - 1 对比一下 不同 分词器 优缺点？
    - 2 举例 介绍一下 不同 大模型LLMs 的分词效果？
    - 3 介绍一下 不同 大模型LLMs 的分词方式 的区别？

- [点击查看答案](https://articles.zsxq.com/id_6z8ptdwqeid7.html)

### [怎么让英文大语言模型支持中文？（一） —— 构建中文tokenization](https://articles.zsxq.com/id_w0d2q29sueq7.html)

- 一、为什么需要 构建中文tokenization？
- 二、如何对 原始数据预处理？
- 三、如何构建中文的词库？
- 四、如何使用transformers库加载sentencepiece模型？
- 五、如何合并英文词表和中文词表？
- 六、怎么使用修改后的词表？
- 总结一下 构建中文tokenization？

- [点击查看答案](https://articles.zsxq.com/id_w0d2q29sueq7.html)

### [怎么让英文大语言模型支持中文？（二） —— 继续预训练篇](https://articles.zsxq.com/id_jprkwhrvf3tw.html)

- 一、为什么需要进行继续预训练？
- 二、如何对 继续预训练 数据预处理？
- 三、如何 构建模型？
- 四、如何 使用模型？

- [点击查看答案](https://articles.zsxq.com/id_jprkwhrvf3tw.html)

### [怎么让英文大语言模型支持中文？（三） —— 对预训练模型进行指令微调](https://articles.zsxq.com/id_p2wj7zadwxwb.html)

- 一、为什么需要对预训练模型进行指令微调？
- 二、对预训练模型进行指令微调 数据 如何处理？
- 三、对预训练模型进行指令微调 tokenization 如何构建？
- 四、对预训练模型进行指令微调 模型 如何构建？
- 五、是否可以结合 其他库 使用？

- [点击查看答案](https://articles.zsxq.com/id_p2wj7zadwxwb.html)


## [二十六、LLMs 测试集 中 数据泄露 问题篇](https://articles.zsxq.com/id_6e3k0i8x5ggm.html)

- 一、什么是 LLMs 测试集数据泄露 问题？
- 二、如何解决 LLMs 测试集数据泄露 问题？
- 三、是否可以 避开训练集来处理 LLMs 测试集数据泄露 问题？
  - 3.1 如何 判断 网络上是否有原题？
  - 3.2 如何 判断答案是否存在？
  - 3.3 性能差异对比
- 四、常见测试集有多少比例的数据泄露？

- [点击查看答案](https://articles.zsxq.com/id_6e3k0i8x5ggm.html)

## [二十七、大模型（LLMs）软硬件配置面](https://articles.zsxq.com/id_m5q8zk3wo84k.html)

1. 建议的软件环境是什么？

- [点击查看答案](https://articles.zsxq.com/id_m5q8zk3wo84k.html)

## [二十八、Token及模型参数准备篇](https://articles.zsxq.com/id_9oplu4014qx5.html)

1. 预训练数据 Token 重复 是否影响 模型性能？
2. SFT需要训练Token数？

- [点击查看答案](https://articles.zsxq.com/id_9oplu4014qx5.html)

## 二十九、多模态常见面试篇

### [多模态常见面试篇](https://articles.zsxq.com/id_hmoqafrxjumk.html)

- 一、最近关注的论文，多模态视觉大模型(CLIP,DALLE)？
- 二、blip2的架构，优势和之前多模态模型的区别？
- 三、多模态融合后，怎样知道最终结果受哪种模态影响更大？
- 四、多模态中常见的SOTA模型有哪些？
- 五、介绍一下stable diffusion的原理？

- [点击查看答案](https://articles.zsxq.com/id_hmoqafrxjumk.html)

## 三十、NLP常见面试篇

### [NLP Trick 篇](https://articles.zsxq.com/id_bnzc5w57w7ox.html) 

- 一、怎么处理类别不平衡？
- 二、有了解其他模型去尝试解决长度限制的方案吗？

- [点击查看答案](https://articles.zsxq.com/id_bnzc5w57w7ox.html)

### [文本分类常见面试篇](https://articles.zsxq.com/id_fku4xbzkano0.html) 

- 一、文本分类任务有哪些应用场景？
- 二、文本分类的具体流程？
- 三、fastText的分类过程？fastText的优点？
- 四、TextCNN进行文本分类的过程?
- 五、TextCNN可以调整哪些参数？
- 六、文本分类任务使用的评估指标有哪些？

- [点击查看答案](https://articles.zsxq.com/id_fku4xbzkano0.html)

### [文本摘要常见面试篇](https://articles.zsxq.com/id_gw097zgji66q.html) 

- 一、抽取式摘要和生成式摘要存在哪些问题？
- 二、Pointer-generator network解决了什么问题？
- 三、文本摘要有哪些应用场景？
- 四、几种ROUGE指标之间的区别是什么？
- 五、BLEU和ROUGE有什么不同？

- [点击查看答案](https://articles.zsxq.com/id_gw097zgji66q.html)

### [命名实体识别常见面试篇](https://articles.zsxq.com/id_2nueuvwwm7v0.html) 

- 一、CRF 常见面试题
  - 1.1 什么是CRF？CRF的主要思想是什么？
  - 1.2 CRF的三个基本问题是什么？
  - 1.3 线性链条件随机场的参数化形式？
  - 1.4 CRF的优缺点是什么？
  - 1.5 HMM与CRF的区别？
  - 1.6 生成模型与判别模型的区别？
- 二、HMM 常见面试题
  - 2.1 什么是马尔科夫过程？
  - 2.2 马尔科夫过程的核心思想是什么？
  - 2.3 隐马尔可夫算法中的两个假设是什么？
  - 2.4 隐马尔可夫模型三个基本问题是什么？
  - 2.5 隐马尔可夫模型三个基本问题的联系？
  - 2.6 隐马尔可夫算法存在哪些问题？

- [点击查看答案](https://articles.zsxq.com/id_2nueuvwwm7v0.html)

### [向量检索常见面试篇](https://articles.zsxq.com/id_dnq0o4aicjso.html) 

- 一、向量检索库总结
  - 1.1 Annoy
    - 1.1.1 Annoy 介绍
    - 1.1.2 Annoy 使用
  - 1.2 Faiss
    - 1.2.1 Faiss 介绍
    - 1.2.2 Faiss 主要特性
    - 1.2.3 Faiss 使用
  - 1.3 Milvus
  - 1.4 ElasticSearch
    - 1.4.1 ElasticSearch 介绍
    - 1.4.2 什么是倒排索引呢？
    - 1.4.3 ES机制

- [点击查看答案](https://articles.zsxq.com/id_dnq0o4aicjso.html)

## 三十一、其他常见面试篇

### [LLMs 其他 Trick](https://articles.zsxq.com/id_958pher9zdxp.html)

1. huggingface 下载不了模型问题？

- [点击查看答案](https://articles.zsxq.com/id_958pher9zdxp.html)

## 三十二、大模型——Chat o1 篇 :fire:

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

## [三十三、大模型——DeepSeek-R1 百问百搭 篇](https://articles.zsxq.com/id_qixbm7loips9.html) :fire:

- [GRPO（Group Relative Policy Optimization）篇](https://articles.zsxq.com/id_eh0ee91au5t4.html)  :fire:
- [DeepSeek-R1-Zero 篇](https://articles.zsxq.com/id_f8gqnqzxhx0n.html)  :fire:
- [DeepSeek-R1 百问百搭-DeepSeek-R1篇](https://articles.zsxq.com/id_tnkh7vh7jvw1.html) :fire:
- [千面郎君 篇（三十二章）—— DeepSeek-R1 论文解读](https://articles.zsxq.com/id_o4r4cpl8cqzm.html)  :fire:

## 三十四、[大模型——推理大模型面](https://articles.zsxq.com/id_ofc8lbfxntit.html) :fire:

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

## 三十五、大模型——Kimi1.5 篇 :fire:

- [kimi1.5 论文研读](https://articles.zsxq.com/id_8v9xlujg57pv.html) 

