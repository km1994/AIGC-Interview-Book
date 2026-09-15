# LLMs 基础面试题篇

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

- 如何 利用 transformers 加载 Bert 模型？
如何 利用 transformers 输出 Bert 指定 hidden\_state？
- BERT 获取最后一层或每一层网络的向量输出

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

## 三、大模型（LLMs）显存问题面 

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

## 四、大模型幻觉（LLM Hallucination）面 

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

## [五、LLMs 测试集 中 数据泄露 问题篇](https://articles.zsxq.com/id_6e3k0i8x5ggm.html)

- 一、什么是 LLMs 测试集数据泄露 问题？
- 二、如何解决 LLMs 测试集数据泄露 问题？
- 三、是否可以 避开训练集来处理 LLMs 测试集数据泄露 问题？
  - 3.1 如何 判断 网络上是否有原题？
  - 3.2 如何 判断答案是否存在？
  - 3.3 性能差异对比
- 四、常见测试集有多少比例的数据泄露？

- [点击查看答案](https://articles.zsxq.com/id_6e3k0i8x5ggm.html)

## [六、大模型（LLMs）软硬件配置面](https://articles.zsxq.com/id_m5q8zk3wo84k.html)

1. 建议的软件环境是什么？

- [点击查看答案](https://articles.zsxq.com/id_m5q8zk3wo84k.html)

## [七、Token及模型参数准备篇](https://articles.zsxq.com/id_9oplu4014qx5.html)

1. 预训练数据 Token 重复 是否影响 模型性能？
2. SFT需要训练Token数？

- [点击查看答案](https://articles.zsxq.com/id_9oplu4014qx5.html)

## 八、多模态常见面试篇

### [多模态常见面试篇](https://articles.zsxq.com/id_hmoqafrxjumk.html)

- 一、最近关注的论文，多模态视觉大模型(CLIP,DALLE)？
- 二、blip2的架构，优势和之前多模态模型的区别？
- 三、多模态融合后，怎样知道最终结果受哪种模态影响更大？
- 四、多模态中常见的SOTA模型有哪些？
- 五、介绍一下stable diffusion的原理？

- [点击查看答案](https://articles.zsxq.com/id_hmoqafrxjumk.html)

## 九、其他常见面试篇

### [LLMs 其他 Trick](https://articles.zsxq.com/id_958pher9zdxp.html)

1. huggingface 下载不了模型问题？

- [点击查看答案](https://articles.zsxq.com/id_958pher9zdxp.html)

## 十、大模型——角色扮演大模型篇

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

