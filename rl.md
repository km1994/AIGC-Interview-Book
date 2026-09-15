# 大模型（LLM）强化学习面试常考题篇

## 常考题

- [RL为什么会训崩？](https://articles.zsxq.com/id_l492uajawq66.html)
- [介绍一下 SFT 的 loss 如何设计？](https://articles.zsxq.com/id_4dwsr5plfj8w.html)
- [介绍一下 RL 的 loss 如何设计？](https://articles.zsxq.com/id_hum8535jxnpx.html)
- [RL 为什么不如 SFT稳定？](https://articles.zsxq.com/id_84tm2kmasod5.html)
- [RL 有哪些 Trick？](https://articles.zsxq.com/id_s57stkzhss97.html)
- [RL 数据如何构建？](https://articles.zsxq.com/id_398dl328xbv3.html)
- [VLM RL如何优化？](https://articles.zsxq.com/id_owikt7iv031l.html)
- [面试官问：为什么RL训练不看loss大小，而SFT就需要?](https://articles.zsxq.com/id_hpqovqvwa6wk.html)
- [RL为什么训练不稳定？](https://articles.zsxq.com/id_5u16irr0slr7.html)
- [🔥 SFT和RL在后训练中哪个更容易导致灾难性遗忘？](https://articles.zsxq.com/id_brpa2rlo0eka.html)

## [大模型（LLMs）强化学习面](https://articles.zsxq.com/id_20xnfnoprj9s.html) 

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

## [大模型（LLMs）强化学习——RLHF及其变种面](https://articles.zsxq.com/id_3ct6sw0wouna.html)

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

## [大模型（LLMs）强化学习—— PPO 面](https://articles.zsxq.com/id_s8kwqw1gowvh.html)

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

## [RLHF平替算法DPO篇](https://articles.zsxq.com/id_mlq44r1p7nob.html)

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
    - DPO损失函数 代码实现？
    - DPO 批次训练过程 代码实现？
    - LM的交叉熵计算 代码实现？

- [点击查看答案](https://articles.zsxq.com/id_mlq44r1p7nob.html)

## [reward 篇](https://articles.zsxq.com/id_vblb0j5qnaxg.html)

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

## [强化学习在自然语言处理下的应用篇](https://articles.zsxq.com/id_5tsn84l32eea.html)

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
