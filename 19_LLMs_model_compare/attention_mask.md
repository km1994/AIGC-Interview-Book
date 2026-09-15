# 大模型-attention mask 篇

- [大模型-attention mask 篇](#大模型-attention-mask-篇)
  - [1、prefix-tuning的prefix tokens是双向注意力吗？](#1prefix-tuning的prefix-tokens是双向注意力吗)
  - [2、chatglm1和chatglm2的attention mask是怎么样的？](#2chatglm1和chatglm2的attention-mask是怎么样的)
  - [3、llama的attention mask是怎么样的？](#3llama的attention-mask是怎么样的)
  - [致谢](#致谢)

## 1、prefix-tuning的prefix tokens是双向注意力吗？

chatglm1和chatglm2的prefix tokens训练是用的双向注意力机制。

##  2、chatglm1和chatglm2的attention mask是怎么样的？

- chatglm1

1. 无论是训练还是推理，prompt部分用的双向注意力机制。
2. 如果采用prefix-tuning的方式训练，prefix tokens和prompt都是用的双向注意力机制，并且prefix tokens能看见prompt。
3. 在生成的时候prefix_tokens+prompt通过双向注意力机制存入kv cache，在一轮生成的过程中，不会再重新计算注意力。（但是在多轮对话中，会重新计算，因为又有了新的prompt）

- chatglm2

1. 无论是训练还是推理，chatglm2中使用的是causal_mask

## 3、llama的attention mask是怎么样的？

无论是训练还是推理，llama中使用的是causal_mask


## 致谢

- 大模型-attention mask问题 https://zhuanlan.zhihu.com/p/683040184