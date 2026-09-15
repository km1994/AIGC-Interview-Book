# MOE（Mixture-of-Experts）面试常考题篇 :fire:

## [MOE（Mixture-of-Experts）篇](https://articles.zsxq.com/id_5anfhj9qoh2v.html)

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

## [MOE大模型对比篇](https://articles.zsxq.com/id_j51bnu3xfgm9.html)

- DeepSpeed-MoE
- PAI-Megatron-Patch MoE
  
- [点击查看答案](https://articles.zsxq.com/id_j51bnu3xfgm9.html)

## [MoE 大模型负载均衡策略演进的回顾](https://articles.zsxq.com/id_5anfhj9qoh2v.html)

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

