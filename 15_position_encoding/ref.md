# 旋转式位置编码 (RoPE) 资料收集

## RoPE 核心思路是什么？

- RoPE**通过绝对位置编码的方式实现相对位置编码**，综合了绝对位置编码和相对位置编码的优点。
- 主要就是**对attention中的q, k向量注入了绝对位置信息，然后用更新的q,k向量做attention中的内积就会引入相对位置信息**了。





## 致谢

- 【偏数学推导】
  - 旋转式位置编码 (RoPE) 知识总结 https://zhuanlan.zhihu.com/p/662790439
  - 十分钟读懂旋转编码（RoPE） https://www.zhihu.com/tardis/zm/art/647109286?source_id=1005 
  - 图解RoPE旋转位置编码及其特性  https://zhuanlan.zhihu.com/p/667864459
- 【理论总结】
  - Rotary Position Embedding (RoPE, 旋转式位置编码) | 原理讲解+torch代码实现  https://blog.csdn.net/weixin_43646592/article/details/130924280
  - 大模型系列：快速通俗理解Transformer旋转位置编码RoPE  https://www.jianshu.com/p/e8be3dbfb4c5
  - 大模型基础｜位置编码｜RoPE｜ALiBi https://zhuanlan.zhihu.com/p/650469278 
