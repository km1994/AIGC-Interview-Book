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

$$
\mathrm{FFN}(x) = \mathrm{GELU}(xW_1 + b_1)\,W_2 + b_2
$$

- 其中 $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$、$W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$，原始 Transformer 取 $d_{ff} = 4\,d_{\text{model}}$；FFN 对序列中每个位置（position-wise）独立作用，位置之间不交互。

## FFN前馈神经网络 有什么优点？
- 优点：**FFN层可以在每个Transformer模块中增加非线性处理能力，增强模型的整体表达能力**。同时 FFN 承担了 Transformer 中大部分的参数量（约占 2/3），可以看作按 token 独立检索的键值记忆。

## FFN前馈神经网络 变种有哪些？
- FFN层的变种：ReLU、GELU、GLU、GeGLU、Swish、SwiGLU （套娃即视感）
1. ReLU： [Rectified Linear Unit](https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)

$$
\mathrm{ReLU}(x) = \max(0,\; x)
$$

2. GELU: [Gaussian Error Linear Unit](https://arxiv.org/abs/1606.08415)

$$
\mathrm{GELU}(x) = x \cdot P(X \le x) = x \cdot \Phi(x), \qquad X \sim \mathcal{N}(0, 1)
$$

> P为伯努利分布，X为标准正态分布；$\Phi(\cdot)$ 为标准正态分布的累积分布函数。常用 tanh 近似：

$$
\mathrm{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
$$

3. GLU：[Gated Linear Units](https://arxiv.org/pdf/2002.05202.pdf)

$$
\mathrm{GLU}(x) = (xW + b) \otimes \sigma(xV + c)
$$

4. GEGLU

$$
\mathrm{GEGLU}(x) = \mathrm{GELU}(xW + b) \otimes (xV + c)
$$

5. Swish

$$
\mathrm{Swish}(x) = x \cdot \sigma(\beta x)
$$

> $\beta$ 可取可学习参数，$\beta = 1$ 时即 SiLU：$\mathrm{SiLU}(x) = x \cdot \sigma(x)$

6. SwiGLU

$$
\mathrm{SwiGLU}(x) = \mathrm{Swish}(xW + b) \otimes (xV + c)
$$

> 其中 $\sigma$ 为 sigmoid 函数，$\otimes$ 为逐元素相乘。由于多引入了一个门控矩阵 $V$，实践中通常把 $d_{ff}$ 缩小到约 $\frac{2}{3} \times 4d_{\text{model}}$（如 $8d/3$ 再对齐到 64 的倍数），以保持参数量可比。

7. 不同激活函数的 log-PPL 对比

> 此处原为一张 log-PPL 对比折线图（图片已丢失），结论：SwiGLU 表现最好，其次是 GEGLU / Swish / GELU，ReLU 垫底；因此在 LLaMA、PaLM 等模型中大量采用 SwiGLU。

可以看到SwiGLU最强，所以现在大部分都用这个了。

- 参考：
  - 冯良骏：昇腾大模型|结构组件-2——ReLU、GeLU、SwiGLU、GeGLU：https://zhuanlan.zhihu.com/p/621058772
  - 机器之心：谷歌大脑提出新型激活函数Swish惹争议：可直接替换并优于ReLU？（附机器之心测试）：https://zhuanlan.zhihu.com/p/30332306
