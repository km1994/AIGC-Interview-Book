# 大模型推理加速——KV Cache面试常考题篇 :fire:

## [从头开始了解 LLM 中的 KV 缓存并对其进行编码](https://articles.zsxq.com/id_5lgwgwfboixx.html)

- 引言
- 一、什么是 KV 缓存？
- 二、LLM 如何生成文本（有无 KV 缓存）
- 三、如何 从头开始实现 KV 缓存？
  - 3.1 注册缓存缓冲区
  - 3.2 带use\_cache标志的转发
  - 3.3 清除缓存
  - 3.4 在 Full Model 中传播use\_cache
  - 3.5 在生成中使用缓存
- 四、简单的性能比较
- 五、KV 缓存的优点和缺点
  - 5.1 KV 缓存的优点
  - 5.2 KV 缓存的缺点
- 六、优化 KV 缓存实现
  - 6.1 扩展缓存时的常见陷阱
    - 6.1.1 提示 1：预分配内存
    - 6.1.2 提示 2：通过滑动窗口截断缓存
  - 6.2 实践中的优化
- 结论

- [点击查看答案](https://articles.zsxq.com/id_5lgwgwfboixx.html)

## [大模型推理加速——KV Cache篇](https://articles.zsxq.com/id_swmfcls3sp1j.html)

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

## [KV-Cache 面试参考题篇](https://articles.zsxq.com/id_smwaj8hquckz.html)

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

## [从多头共享到潜变量：MLA在低秩投影与按需解压中重新定义 KV-Cache 存储](https://articles.zsxq.com/id_4lcum6gsda71.html)

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
