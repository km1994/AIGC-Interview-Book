# 跨注意力机制（Cross-Attention）篇

- [跨注意力机制（Cross-Attention）篇](#跨注意力机制cross-attention篇)
  - [一、为什么需要 跨注意力机制（Cross-Attention）？](#一为什么需要-跨注意力机制cross-attention)
  - [二、介绍一些 跨注意力机制（Cross-Attention）？](#二介绍一些-跨注意力机制cross-attention)
  - [三、Cross Attention 和 Self Attention 篇](#三cross-attention-和-self-attention-篇)
    - [3.1 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么相同点？](#31-cross-attention-和-self-attention-都是基于注意力机制的有什么相同点)
    - [3.2 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么不同点？](#32-cross-attention-和-self-attention-都是基于注意力机制的有什么不同点)
  - [四、Cross Attention 和 多头注意力（Multi-Head Attention）篇](#四cross-attention-和-多头注意力multi-head-attention篇)
    - [4.2 Cross Attention 和 多头注意力（Multi-Head Attention） 都是基于注意力机制的，有什么异同点？](#42-cross-attention-和-多头注意力multi-head-attention-都是基于注意力机制的有什么异同点)
  - [五、Cross Attention 代码实现](#五cross-attention-代码实现)
  - [六、Cross Attention 应用场景](#六cross-attention-应用场景)
  - [七、Cross Attention 的优势和挑战？](#七cross-attention-的优势和挑战)
  - [致谢](#致谢)

## 一、为什么需要 跨注意力机制（Cross-Attention）？

对于一些NLP任务（eg：机器翻译、文本匹配等）不仅需要“关注”自身信息，还需要“关注”另外一个序列信息。

> 注：
> 
> 1. 两个序列可以是不同的模式形态（如：文本、声音、图像）
> 
> 2. 两个序列必须具有相同的维度
> 

## 二、介绍一些 跨注意力机制（Cross-Attention）？

跨注意力机制（Cross-Attention）的思想是使一个序列能够“关注”另一个序列。在许多场景中，这可能很有用，例如在机器翻译中，将输入序列（源语言）的部分与输出序列（目标语言）的部分对齐是有益的。

## 三、Cross Attention 和 Self Attention 篇

### 3.1 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么相同点？

1. **机制**：两者都使用了点积注意力机制（scaled dot-product attention）来计算注意力权重。
2. **参数**：无论是自注意力还是交叉注意力，它们都有查询（Query）、键（Key）和值（Value）的概念。
3. **计算**：两者都使用查询和键之间的点积，然后应用softmax函数来计算注意力权重。
4. **输出**：在计算完注意力权重后，两者都将这些权重应用于值来得到输出。
5. **可变性**：两者都可以通过掩码（masking）来控制某些位置不被其他位置关注。

### 3.2 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么不同点？

- Self Attention: 
  - **查询、键和值都来自同一个输入序列**。这使得模型能够关注输入序列中的其他部分以产生一个位置的输出。**主要目的是捕捉输入序列内部的依赖关系**。在Transformer的编码器（Encoder）和解码器（Decoder）的每一层都有自注意力。它允许输入序列的每个部分关注序列中的其他部分。

- Cross Attention: 
  - **查询来自一个输入序列，而键和值来自另一个输入序列**。这在诸如序列到序列模型（如机器翻译）中很常见，其中一个序列需要“关注”另一个序列。**目的是使一个序列能够关注另一个不同的序列**。主要出现在Transformer的解码器。它允许解码器关注编码器的输出，这在机器翻译等任务中尤为重要。

总的来说，自注意力和交叉注意力都是基于相同的核心机制，但它们的应用和目的有所不同。**自注意力旨在处理单一序列内部的关系，而交叉注意力则旨在处理两个不同序列之间的关系**。

## 四、Cross Attention 和 多头注意力（Multi-Head Attention）篇

### 4.2 Cross Attention 和 多头注意力（Multi-Head Attention） 都是基于注意力机制的，有什么异同点？

- 多头注意力机制

多头注意力(Multi-Head Attention)是一种基于自注意力机制(self-attention)的改进方法。自注意力是一种能够计算出输入序列中每个位置的权重，因此可以很好地处理序列中长距离依赖关系的问题。但在应用中，可能存在多个不同的关注点，因此就需要多个自注意力机制来处理不同的关注点。**多头注意力就是在一个输入序列上使用多个自注意力机制，得到多组注意力结果，然后将这些结果进行拼接和线性投影得到最终输出**。

**多头注意力的优点是能够处理多个关注点的问题，可以较好地处理复杂语义关系**。

- 多头注意力机制

交叉注意力(Cross-Attention)则是在两个不同序列上计算注意力，用于处理两个序列之间的语义关系。

> 例如，在翻译任务中，需要将源语言句子和目标语言句子进行对齐，就需要使用交叉注意力来计算两个句子之间的注意力权重。

## 五、Cross Attention 代码实现

```s
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
       
        assert d_model % self.num_heads == 0
       
        self.depth = d_model // self.num_heads
       
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
       
        self.dense = nn.Linear(d_model, d_model)
       
    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
   
    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]
       
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
       
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q_x, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k_x, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v_x, depth)
       
        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
       
        # Add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Add the mask to the scaled tensor.
           
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (batch_size, num_heads, seq_len_q_x, seq_len_k_x)
       
        output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q_x, depth)
        output = output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len_q_x, num_heads, depth)
        output = output.reshape(batch_size, -1, self.d_model)  # (batch_size, seq_len_q_x, d_model)
       
        output = self.dense(output)  # (batch_size, seq_len_q_x, d_model)
       
        return output, attention_weights

# 示例使用
# 假设v, k, q是形状为(batch_size, seq_len, d_model)的张量
# mask是形状为(batch_size, 1, seq_len_k)的掩码张量
# cross_attention = CrossAttention(d_model=512, num_heads=8)
# output, attention_weights = cross_attention(v, k, q, mask)
```

在这个实现中，CrossAttention类包含了一个线性层用于查询（query）、键（key）和值（value）的转换，以及一个线性层用于最后的输出转换。split_heads函数用于将输入张量分割成多个头（heads），并重新排列其维度以进行多头注意力计算。在forward函数中，我们首先计算了缩放后的注意力对数几率（logits），然后应用了softmax函数来计算注意力权重，最后使用这些权重对值进行加权求和，并返回输出和注意力权重。

> 请注意，在上面的示例中，我们假设mask是一个二维张量，但在实际使用中，它可能是一个三维张量，其形状为(batch_size, 1, seq_len_k)，以便与scaled_attention_logits的维度相匹配。这取决于你的具体实现和任务需求。

## 六、Cross Attention 应用场景

- 机器翻译

跨注意力机制在机器翻译任务中被广泛应用。通过融合源语言和目标语言的信息，模型可以更好地理解两种语言之间的关系，从而提高翻译质量。

- 文本生成

在文本生成任务中，如语言模型和对话系统，跨注意力机制可以帮助模型结合上下文信息和给定的条件，生成连贯且有逻辑的文本。

- 图像字幕生成

跨注意力机制还可用于图像字幕生成任务中，其中图像被视为一个输入序列，而文字描述作为另一个输入序列。通过跨注意力机制，模型能够将图像和文字相关联，生成准确的图像字幕。

## 七、Cross Attention 的优势和挑战？

跨注意力机制相比传统的序列建模方法具有以下优势：能够融合多个来源的信息、处理跨模态数据等。然而，它也面临一些挑战，如计算资源消耗较大和注意力偏置等问题。

## 致谢

- Cross-Attention 学习笔记 https://zhuanlan.zhihu.com/p/648248676
- 多头注意力（Multi-Head Attention）和交叉注意力（Cross-Attention）是两种常用的注意力机制的原理及区别 https://blog.csdn.net/qq_39506862/article/details/133868090
- 深入理解跨注意力机制（Cross-Attention） https://blog.csdn.net/m0_72410588/article/details/131670508
- 【科研】浅学Cross-attention？ https://blog.csdn.net/MengYa_Dream/article/details/126688503








