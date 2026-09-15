# 大模型（LLMs）参数高效微调(PEFT) 面试常考题篇

## [大模型（LLMs）参数高效微调(PEFT) 面](https://articles.zsxq.com/id_ipkod91a939n.html)

1. 微调方法是啥？如何微调？
2. 为什么需要 PE
3. 介绍一下 PEFT？
4. PEFT 有什么优点？
5. 微调方法批处理大小模式GPU显存速度？
6. Peft 和 全量微调区别？
7. 多种不同的高效微调方法对比
8. 当前高效微调技术存在的一些问题
9. 高效微调技术最佳实践
10. PEFT 存在问题？
11. 能不能总结一下各种参数高效微调方法？

- [点击查看答案](https://articles.zsxq.com/id_ipkod91a939n.html)

## [配器微调（Adapter-tuning）篇](https://articles.zsxq.com/id_0n6pfw0wz3xb.html)

- 一、为什么 需要 适配器微调（Adapter-tuning）？
- 二、适配器微调（Adapter-tuning）思路？
- 三、 适配器微调（Adapter-tuning）特点是什么？
- 四、AdapterFusion 思路 是什么？
- 五、AdapterDrop 思路 是什么？
- 六、AdapterDrop 特点 是什么？
- 七、MAM Adapter 思路 是什么？
- 八、MAM Adapter 特点 是什么？

- [点击查看答案](https://articles.zsxq.com/id_0n6pfw0wz3xb.html)

## [提示学习（Prompting）](https://articles.zsxq.com/id_662wpbw47gtj.html)

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

## [LoRA 系列篇](https://articles.zsxq.com/id_gjkhd8xn4pvt.html) 

- 一、LoRA篇
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
    - 2.2.1 AdaLoRA 的思路是怎么样的？
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
  - LoRA 微调计算可训练参数的比例 如何确定？
  - LoRA 微调结果如何保存？

- [点击查看答案](https://articles.zsxq.com/id_gjkhd8xn4pvt.html)

## [如何使用 PEFT库 中 LoRA？](https://articles.zsxq.com/id_8lx1t1t3w4qf.html) 

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

## [大模型 SFT 方式对比篇](https://articles.zsxq.com/id_e2piver2uzei.html) 

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
