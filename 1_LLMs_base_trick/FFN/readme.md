# FFN前馈神经网络篇

- [FFN前馈神经网络篇](#ffn前馈神经网络篇)
  - [为什么 需要 FFN前馈神经网络？](#为什么-需要-ffn前馈神经网络)
  - [什么是 FFN前馈神经网络？](#什么是-ffn前馈神经网络)
  - [FFN前馈神经网络 有什么优点？](#ffn前馈神经网络-有什么优点)
  - [FFN前馈神经网络 变种有哪些？](#ffn前馈神经网络-变种有哪些)

## 为什么 需要 FFN前馈神经网络？

- 动机：Attention层 局限性。**Attention机制能够处理词语之间的相互关系，但是它无法进行更为复杂的、非线性的数据处理。**

## 什么是 FFN前馈神经网络？

- 介绍：处理来自Attention层的信息；在原始的Transformer模型中，FFN层通常由两个线性变换和一个非线性激活函数（如ReLU或GELU）组成。

## FFN前馈神经网络 有什么优点？

- 优点：**FFN层可以在每个Transformer模块中增加非线性处理能力，增强模型的整体表达能力**。

## FFN前馈神经网络 变种有哪些？

- FFN层的变种：ReLU、GELU、GLU、GeGLU、Swish、SwiGLU （套娃即视感）

1. ReLU： [Rectified Linear Unit](https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)

![](img/微信截图_20230717204450.png)

2. GELU: [Gaussian Error Linear Unit](https://arxiv.org/abs/1606.08415)

![](img/微信截图_20230717204552.png)

> P为伯努利分布，X为标准正态分布

3. GLU：[Gated Linear Units](https://arxiv.org/pdf/2002.05202.pdf)

![](img/微信截图_20230717204638.png)

4. GEGLU

![](img/微信截图_20230717204702.png)

5. Swish

![](img/微信截图_20230717204726.png)

6. SwiGLU

![](img/微信截图_20230717204745.png)


![](img/微信截图_20230717204834.png)
> 不同激活函数的log-PPL对比

可以看到SwiGLU最强，所以现在大部分都用这个了。

- 参考：
  - 冯良骏：昇腾大模型|结构组件-2——ReLU、GeLU、SwiGLU、GeGLU：https://zhuanlan.zhihu.com/p/621058772
  - 机器之心：谷歌大脑提出新型激活函数Swish惹争议：可直接替换并优于ReLU？（附机器之心测试）：https://zhuanlan.zhihu.com/p/30332306
