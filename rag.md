# 大模型（LLMs）RAG 检索增强生成面试常考题篇

## 常考题

- [如果构建行业垂直大模型，到底是用RAG还是微调？](https://articles.zsxq.com/id_u4w09pxrg76h.html)
- [面试官问：RAG 的 Query 理解模块是怎么做的？](https://articles.zsxq.com/id_bdsjh42pr3gn.html)
- [🤔 面试官问：RAG 的知识库是怎么构建的？](https://articles.zsxq.com/id_sj396fd8qfgz.html)
- [🤔 面试官问：RAG系统里的文档解析和Chunk切分怎么做？](https://articles.zsxq.com/id_rokk0cmdt212.html)
- [😎 面试官问我RAG检索模块怎么优化，我反手就是一个混合检索+Rerank！](https://articles.zsxq.com/id_8it3a39rk0te.html)
- [MCP 与 RAG 的区别是什么?](https://articles.zsxq.com/id_ets7l2awhuqx.html)
- [面试官：客户抱怨 RAG 太慢，有什么办法可以降低延迟？](https://articles.zsxq.com/id_w013582q2el1.html)
- [🔥 RAG项目总翻车？90%的失败都栽在数据准备上！](https://articles.zsxq.com/id_3p2kcqm4fzji.html)
- [😎 面试官问：你们的 RAG 系统是怎么评估的？](https://articles.zsxq.com/id_ekj54kxpvqnp.html)
- [RAG有哪些优化手段？](https://articles.zsxq.com/id_tt993wk99x9f.html)
- [query 优化技术综述](https://articles.zsxq.com/id_cw4eiw3s9m8o.html)
- [🤔 面试官问：RAG系统的检索排序要怎么优化？](https://articles.zsxq.com/id_bko8h827lgvl.html)
- [😎 面试官问：RAG 系统的整体架构要怎么优化？](https://articles.zsxq.com/id_mu44xkk6usux.html)
- [😎 面试官问：你们的RAG系统，首字响应时间（TTFT）是怎么优化的？](https://articles.zsxq.com/id_hsdp9oxxs2ww.html)
- [🤖 面试官突然问你：“RAG 的生成模块是怎么优化的？”](https://articles.zsxq.com/id_we5n7mhlejou.html)
- [🤔 面试官问：RAG 的评估体系怎么做？](https://articles.zsxq.com/id_cse0o9h0uwen.html)


## 大模型（LLMs）RAG 入门篇

### [基于LLM+向量库的文档对话 经验面](https://articles.zsxq.com/id_xk58m8ok2sob.html)

- 一、基于LLM+向量库的文档对话 基础面
  - 1.1 为什么 大模型 需要 外挂(向量)知识库？
  - 1.2. 基于LLM+向量库的文档对话 思路是怎么样？
  - 1.3. 基于LLM+向量库的文档对话 核心技术是什么？
  - 1.4. 基于LLM+向量库的文档对话 prompt 模板 如何构建？
- 二、基于LLM+向量库的文档对话 存在哪些痛点？
- 三、基于LLM+向量库的文档对话 工程示例面

- [点击查看答案](https://articles.zsxq.com/id_xk58m8ok2sob.html)

### [RAG（Retrieval-Augmented Generation）面](https://articles.zsxq.com/id_xk58m8ok2sob.html) 

- 一、LLMs 已经具备了较强能力了，存在哪些不足点?
- 二、什么是 RAG?
  - 2.1 R：检索器模块
    - 2.1.1 如何获得准确的语义表示？
    - 2.1.2 如何协调查询和文档的语义空间？
    - 2.1.3 如何对齐检索模型的输出和大语言模型的偏好？
  - 2.2 G：生成器模块
    - 2.2.1 生成器介绍
    - 2.2.2 如何通过后检索处理提升检索结果？
    - 2.2.3 如何优化生成器应对输入数据？
- 三、使用 RAG 的好处?
- 四、RAG V.S. SFT
- 五、介绍一下 RAG 典型实现方法？
  - 5.1 如何 构建 数据索引？
  - 5.2 如何 对数据进行 检索（Retrieval）？
  - 5.3 对于 检索到的文本，如果生成正确回复？
- 六、介绍一下 RAG 典型案例？
- 七、RAG 存在什么问题？

- [点击查看答案](https://articles.zsxq.com/id_xk58m8ok2sob.html)

### [RAG 面试常考题](https://articles.zsxq.com/id_bpa8qm9ckrmp.html)

- 一、RAG与传统微调（Fine-tuning）的本质区别？
- 二、为什么RAG能缓解大模型的“幻觉”问题？
- 三、如何定义RAG中的“相关文档”？
- 四、RAG中主流检索器有哪些？各有什么优势？
- 五、知识库文档分块（Chunking）有哪些策略？
- 六、为什么需要向量归一化（Vector Normalization）？
- 七、生成阶段如何融合检索结果？
- 八、RAG中如何避免生成无关内容？
- 九、为什么需要重排序（Re-Ranking）？列举常见方法？
- 十、RAG与Fine-tuning如何结合？
- 十一、如何评估RAG生成结果的质量？
- 十二、如何优化检索的召回率（Recall）？
- 十三、RAG如何处理多文档冲突信息？
- 十四、如何解决“检索偏好”问题（Retrieval Bias）？
- 十五、如何优化长文档检索效果？
- 十六、解释HyDE（Hypothetical Document Embeddings）原理？
- 十七、什么是迭代检索（Iterative Retrieval）？
- 十八、Self-RAG的核心创新点是什么？
- 十九、RAG如何适配实时更新知识库
- 二十、用户查询“2025年诺贝尔奖获得者”，但知识库只更新到2024年，RAG如何应对？

- [点击查看答案](https://articles.zsxq.com/id_bpa8qm9ckrmp.html)

## 大模型（LLMs）RAG 进阶面试篇

### [RAG 实践中的十大误区](https://articles.zsxq.com/id_axwvag8w53lp.html)

- 一、RAG 实践中的十大误区
  - 误区一：忽视数据质量，盲目堆砌知识库内容
  - 误区二：过度依赖通用检索算法，未做场景化适配
  - 误区三：文本拆分粒度不合理，影响检索与生成连贯性
  - 误区四：忽略知识更新机制，知识库 “一成不变”
  - 误区五：将 “检索召回率” 等同于 “检索效果”，忽视精确率
  - 误区六：未优化嵌入模型（Embedding），直接使用默认模型
  - 误区七：生成阶段过度依赖大模型，未做知识约束
  - 误区八：忽视用户查询意图理解，检索目标与需求错位
  - 误区九：缺乏系统评估体系，无法定位性能瓶颈
  - 误区十：过度追求 “全自动化”，忽视人工干预环节
- 二、有哪些提升 RAG 准确率的核心策略？
  - 2.1 数据治理：构建高质量、动态更新的知识库
  - 2.2 检索优化：精准匹配用户需求与候选知识
  - 2.3 生成约束：确保回答基于检索知识，避免偏离
  - 2.4 系统迭代：建立科学评估与人工干预机制
- 总结

- [点击查看答案](https://articles.zsxq.com/id_axwvag8w53lp.html)

### [大模型应用中的对话压缩方法](https://articles.zsxq.com/id_fv4l2g6w9w8q.html)

- 前言
- 一、为什么需要对话压缩？
- 二、有哪些对话压缩的主要方法？
  - 2.1 基于摘要的压缩 (Summarization-based Compression)
  - 2.2 基于选择的压缩 (Selection-based Compression)
  - 2.3 基于重写的压缩 (Rewriting-based Compression)
  - 2.4 基于模型自适应的压缩 (Model-adaptive Compression)
- 三、对话压缩有哪些研究方向？

- [点击查看答案](https://articles.zsxq.com/id_fv4l2g6w9w8q.html)
  
### [多轮 RAG 应用中如何进行指代消解？](https://articles.zsxq.com/id_fghhrwoou585.html)

- 一、RAG应用中为什么需要进行指代消解？
- 二、如何利用大模型解决指代消解任务？
  - 2.1 方法一：直接使用 大模型+prompt 解决指代消解任务
  - 2.2 方法二：直接使用 Few-shot prompt + CoT 解决指代消解任务

- [点击查看答案](https://articles.zsxq.com/id_fghhrwoou585.html)

### [基于LLM与RAG的AI智能体意图识别深度实践](https://articles.zsxq.com/id_sfr15mlkq84t.html)

- 一、为何需要超越简单的提示词工程？
- 二、RAG如何赋能意图识别
- 三、如何 从零构建RAG意图识别系统？
  - 第一步：构建高质量的意图知识库
    - 定义意图与泛化语料
    - 知识库的构建与向量化
  - 第二步：实现端到端的RAG意图识别流程
  - 第三步：处理多轮对话的挑战
- 总结与展望

- [点击查看答案](https://articles.zsxq.com/id_sfr15mlkq84t.html)

### [深入解析RAG多轮会话优化：从查询重写到高级策略](https://articles.zsxq.com/id_832cg5xmsm0m.html)

- 一、为何多轮会话是RAG的"必修课"？
- 二、核心策略：查询重写（Query Rewriting）
- 三、开源框架实战
  - 3.1 LlamaIndex 的 CondenseQuestionChatEngine
  - 3.2 LangChain 的记忆机制与会话链
    - 3.2.1 记忆（Memory）组件
    - 3.2.2 使用 ConversationalRetrievalChain 实现多轮RAG
- 四、进阶优化策略
  - 4.1 上下文管理（Context Management）
  - 4.2 查询扩展（Query Expansion）
  - 4.3 混合搜索（Hybrid Search）
  - 4.4 意图驱动的RAG（Intent-Driven RAG）
  - 4.5 多智能体系统（Multi-Agent Systems）
- 五、总结与展望

- [点击查看答案](https://articles.zsxq.com/id_832cg5xmsm0m.html)

### [不同段位的RAG选择](https://articles.zsxq.com/id_tntsmb2dz44k.html)

- 一、RAG的核心价值
  - 1.1 解决时效性问题
  - 1.2 引导模型按需回答
- 二、Naive RAG：四大核心问题与解决方案
  - 2.1 什么时候搜？——避免无效搜索
    - 2.1.1 问题本质
    - 2.1.2 解决方案
  - 2.2 搜什么？——优化搜索Query
    - 2.2.1 问题本质
    - 2.2.2 解决方案
  - 2.3 怎么搜？——分块与排序策略
    - 2.3.1 问题本质
    - 2.3.2 分块经验
    - 2.3.3 排序优化
    - 2.3.4 精排方法对比
  - 2.4 怎么搜？——分块与排序策略
    - 2.4.1 问题本质
    - 2.4.2 解决方案
    - 2.4.3 典型风险案例
- 三、Graph RAG vs Naive RAG
  - 3.1 核心区别
  - 3.2 典型案例对比
    - 3.2.1 Naive RAG失败场景
    - 3.2.2 Graph RAG成功场景
  - 3.3 关键结论
- 四、Agentic RAG：让RAG学会自主思考
  - 4.1 传统RAG vs Agentic RAG
  - 4.2 三大核心类型
    - 4.2.1 类型1：工具调度员（Tool Router）
    - 4.2.2 类型2：任务拆解师（Query Planner）
    - 4.2.3 类型3：智能循环机（ReAct Agent）
  - 4.3 设计黄金法则
- 五、DeepSearch全流程解析
  - 5.1 核心框架
  - 5.2 关键模块详解
    - 5.2.1 模块1：Planner任务拆解
    - 5.2.2 模块2：任务澄清
    - 5.2.3 模块3：信息获取
    - 5.2.4 模块4：结果反思
    - 5.2.5 模块5：通用工具库
- 六、总结：RAG技术演进全景

- [点击查看答案](https://articles.zsxq.com/id_tntsmb2dz44k.html)

### [多轮 Rag 中如何对问题进行改写？](https://articles.zsxq.com/id_xo8as5cgts4s.html)

- 一、引言
- 二、信息缺失问题
  - 2.1 历史会话改写
  - 2.2 关键词扩写
  - 2.3 伪答案改写
  - 2.4 缩写词改写
- 三、去燥改写问题
  - 3.1 一般去噪改写
  - 3.2 关键词改写
  - 3.3 子查询改写
- 四、Prompt 设计
- 五、总结

- [点击查看答案](https://articles.zsxq.com/id_xo8as5cgts4s.html)

### 经典面试题

- [LLM大模型联网查询技术方案全解析](https://articles.zsxq.com/id_thus6waw9qjd.html)
- [如果你的RAG系统不好用，怎么办?](https://articles.zsxq.com/id_pcvgt4mczach.html)
- [为什么RAG效果比大模型微调差？](https://articles.zsxq.com/id_txcp4ct20noz.html)
- [新需求来了，是用RAG还是大模型微调？](https://articles.zsxq.com/id_nwgg65wthixb.html)

## RAG 工程优化篇

### RAG 工程实践优化策略

- 一、优化索引结构
  - 1.1 优化被检索的embedding
  - 1.2 优化 query 的 chunk 大小
  - 1.3 结合不同粒度信息进行混合检索
- 二、混合检索及 chunk 检索效果不佳时的优化策略
- 三、通过 rerank 提升 RAG 效果的方案
  - 3.1 rerank 的背景与目标
  - 3.2 rerank 思路与方法

- [参考答案](https://articles.zsxq.com/id_u67laht743ih.html)

## 大模型（LLMs）RAG 版面分析篇

### [大模型（LLMs）RAG —— pdf解析关键问题](https://articles.zsxq.com/id_2693k55it84w.html)

- 一、为什么需要进行pdf解析？
- 二、为什么需要 对 pdf 进行解析？
- 三、pdf解析 有哪些方法，对应的区别是什么？
- 四、pdf解析 存在哪些问题？
- 五、如何 长文档（书籍）中关键信息？
- 六、为什么要提取标题甚至是多级标题？
- 七、如何提取 文章标题？
- 八、如何区分单栏还是双栏pdf？如何重新排序？
- 九、如何提取表格和图片中的数据？
- 十、基于AI的文档解析有什么优缺点？

- [点击查看答案](https://articles.zsxq.com/id_2693k55it84w.html)

### [大模型（LLMs）RAG 版面分析——表格识别方法篇](https://articles.zsxq.com/id_7x4qv94hxv8r.html)

- 一、为什么需要识别表格？
- 二、介绍一下 表格识别 任务？
- 三、有哪些 表格识别方法？
  - 3.1 传统方法
  - 3.2 pdfplumber表格抽取
    - 3.2.1 pdfplumber 如何进行 表格抽取？
    - 3.2.2 pdfplumber 常见的表格抽取模式？
  - 3.3 深度学习方法-语义分割
    - 3.3.1 table-ocr/table-detect：票据图片复杂表格框识别(票据单元格切割)
    - 3.3.2 腾讯表格图像识别
    - 3.3.3 TableNet
    - 3.3.4 CascadeTabNet
    - 3.3.5 SPLERGE
    - 3.3.6 DeepDeSRT

- [点击查看答案](https://articles.zsxq.com/id_7x4qv94hxv8r.html)

## 大模型（LLMs）RAG 文本分块篇

### [RAG检索不准？问题可能出在文本切分上——5大策略优化实战](https://articles.zsxq.com/id_lhdb6kzjtc5w.html)

- 前言
- 一、RAG 工作流程概览
- 二、分块（Chunking）策略
  - 2.1 固定分块
    - 2.1.1 固定分块思路
    - 2.1.2 什么时候用固定分块？
    - 2.1.3 固定分块代码实现
  - 2.2 递归分块
    - 2.2.1 递归分块思路
    - 2.2.2 什么时候用递归分块？
    - 2.2.3 递归分块代码实现
  - 2.3 语义分块
    - 2.3.1 语义分块思路
    - 2.3.2 什么时候用语义分块？
    - 2.3.3 语义分块代码实现
  - 2.4 基于结构的分块
    - 2.4.1 基于结构的分块思路
    - 2.4.2 什么时候用基于结构的分块？
    - 2.4.3 基于结构的分块实现要点
  - 2.5 延迟分块（也叫动态分块或查询时分块）
    - 2.5.1 延迟分块思路
    - 2.5.2 延迟分块实现流程？
    - 2.5.3 延迟分块适用场景？
    - 2.5.4 延迟分块存在问题
- 总结

- [点击查看答案](https://articles.zsxq.com/id_lhdb6kzjtc5w.html)

### [大模型（LLMs）RAG 版面分析——文本分块面](https://articles.zsxq.com/id_iw7debl8akxh.html)

- 一、为什么需要对文本分块？
- 二、能不能介绍一下常见的文本分块方法？
  - 2.1 一般的文本分块方法
  - 2.2 正则拆分的文本分块方法
  - 2.3 Spacy Text Splitter 方法
  - 2.4 基于 langchain 的 CharacterTextSplitter 方法
  - 2.5 基于 langchain 的 递归字符切分 方法
  - 2.6 HTML 文本拆分 方法
  - 2.7 Mrrkdown 文本拆分 方法
  - 2.8 Python代码拆分 方法
  - 2.9 LaTex 文本拆分 方法

- [点击查看答案](https://articles.zsxq.com/id_iw7debl8akxh.html)

## 大模型（LLMs）RAG 检索策略篇

### [大模型外挂知识库优化——如何利用大模型辅助召回？](https://articles.zsxq.com/id_oznm6qixjw61.html)

- 一、为什么需要使用大模型辅助召回？
  - 策略一： HYDE
    - 介绍一下 HYDE 思路？
    - 介绍一下 HYDE 问题？
  - 策略二： FLARE
    - 为什么 需要 FLARE ？
    - FLARE 有哪些召回策略？

- [点击查看答案](https://articles.zsxq.com/id_oznm6qixjw61.html)

### [大模型外挂知识库优化——负样本样本挖掘篇](https://articles.zsxq.com/id_wa7nl8wsuilh.html)

- 一、为什么需要构建负难样本？
- 二、负难样本构建方法篇
  - 2.1 随机采样策略（Random Sampling）方法
  - 2.2 Top-K负例采样策略（Top-K Hard Negative Sampling）方法
  - 2.3 困惑负样本采样方法SimANS 方法
  - 2.4 利用 对比学习微调 方式构建负例方法
  - 2.5 基于批内负采样的对比学习方法
  - 2.6 相同文章采样方法
  - 2.7 LLM辅助生成软标签及蒸馏
- 辅助知识
  - 附一：梯度计算方法

- [点击查看答案](https://articles.zsxq.com/id_wa7nl8wsuilh.html)

## 大模型（LLMs）RAG 评测篇

### [RAG（Retrieval-Augmented Generation）评测面](https://articles.zsxq.com/id_vjwt6uzml13l.html)

- 一、为什么需要 对 RAG 进行评测？
- 二、RAG 有哪些评估方法？
- 三、RAG 有哪些关键指标和能力？
- 四、RAG 有哪些评估框架？

- [点击查看答案](https://articles.zsxq.com/id_vjwt6uzml13l.html)

## 大模型（LLMs）RAG 优化策略篇

### [检索增强生成(RAG) 优化策略篇](https://articles.zsxq.com/id_gu4p7gszsh82.html)

- 一、RAG基础功能篇
  - 1.1 RAG 工作流程
- 二、RAG 各模块有哪些优化策略？
- 三、RAG 架构优化有哪些优化策略？
  - 3.1 如何利用 知识图谱（KG）进行上下文增强？
    - 3.1.1 典型RAG架构中，向量数据库进行上下文增强 存在哪些问题？
    - 3.1.2 如何利用 知识图谱（KG）进行上下文增强？
  - 3.2 Self-RAG：如何让 大模型 对 召回结果 进行筛选？
    - 3.2.1 典型RAG架构中，向量数据库 存在哪些问题？
    - 3.2.2 Self-RAG：如何让 大模型 对 召回结果 进行筛选？
    - 3.2.3 Self-RAG 的 创新点是什么？
    - 3.2.4 Self-RAG 的 训练过程？
    - 3.2.5 Self-RAG 的 推理过程？
    - 3.2.6 Self-RAG 的 代码实战？
  - 3.3 多向量检索器多模态RAG篇
    - 3.3.1 如何让 RAG 支持 多模态数据格式？
      - 3.3.1.1 如何让 RAG 支持 半结构化RAG（文本+表格）?
      - 3.3.1.2 如何让 RAG 支持 多模态RAG（文本+表格+图片）?
      - 3.3.1.3 如何让 RAG 支持 私有化多模态RAG（文本+表格+图片）?
  - 3.4 RAG Fusion 优化策略
  - 3.5 模块化 RAG 优化策略
  - 3.6 RAG 新模式 优化策略
  - 3.7 RAG 结合 SFT
  - 3.8 查询转换（Query Transformations）
  - 3.9 bert在RAG中具体是起到了一个什么作用，我刚搜了下nsp的内容，但有点没法将这几者联系起来
- 四、RAG 索引优化有哪些优化策略？
  - 4.1 嵌入 优化策略
  - 4.2 RAG检索召回率低，一般都有哪些解决方案呀。尝试过不同大小的chunk，和混合检索。效果都不太好，然后优化？
  - 4.3 RAG 如何 优化索引结构?
  - 4.4 如何通过 混合检索 提升 RAG 效果?
  - 4.5 如何通过 重新排名 提升 RAG 效果?
- 五、RAG 索引数据优化有哪些优化策略？
  - 5.1 RAG 如何 提升索引数据的质量?
  - 5.2 如何通过添加元数据 提升 RAG 效果?
  - 5.3 如何通过 输入查询与文档对齐 提升 RAG 效果?
  - 5.4 如何通过 提示压缩 提升 RAG 效果?
  - 5.5 如何通过 查询重写和扩展 提升 RAG 效果?
- RAG 未来发展方向
- Rag 的垂直优化
- RAG 的水平扩展
- RAG 生态系统

- [点击查看答案](https://articles.zsxq.com/id_gu4p7gszsh82.html)

#### [RAG 关键痛点及对应解决方案](https://articles.zsxq.com/id_1bmbedojsj0t.html)

- 前言
- 问题一：内容缺失问题
  - 1.1 介绍一下 内容缺失问题？
  - 1.2 如何 解决 内容缺失问题？
- 问题二：错过排名靠前的文档
  - 2.1 介绍一下 错过排名靠前的文档 问题？
  - 2.2 如何 解决 错过排名靠前的文档 问题？
- 问题三：脱离上下文 — 整合策略的限制
  - 3.1 介绍一下 脱离上下文 — 整合策略的限制 问题？
  - 3.2 如何 解决 脱离上下文 — 整合策略的限制 问题？
- 问题四：未能提取答案
  - 4.1 介绍一下 未能提取答案 问题？
  - 4.2 如何 解决 未能提取答案 问题？
- 问题五：格式错误
  - 5.1 介绍一下 格式错误 问题？
  - 5.2 如何 解决 格式错误 问题？
- 问题六： 特异性错误
  - 6.1 介绍一下 特异性错误 问题？
  - 6.2 如何 解决 特异性错误 问题？
- 问题七： 回答不全面
  - 7.1 介绍一下 回答不全面 问题？
  - 7.2 如何 解决 回答不全面 问题？
- 问题八： 数据处理能力的挑战
  - 8.1 介绍一下 数据处理能力的挑战 问题？
  - 8.2 如何 解决 数据处理能力的挑战 问题？
- 问题九： 结构化数据查询的难题
  - 9.1 介绍一下 结构化数据查询的难题 问题？
  - 9.2 如何 解决 结构化数据查询的难题 问题？
- 问题十： 从复杂PDF文件中提取数据
  - 10.1 介绍一下 从复杂PDF文件中提取数据 问题？
  - 10.2 如何 解决 从复杂PDF文件中提取数据 问题？
- 问题十一： 备用模型
  - 11.1 介绍一下 备用模型 问题？
  - 11.2 如何 解决 备用模型 问题？
- 问题十二： 大语言模型（LLM）的安全挑战
  - 12.1 介绍一下 大语言模型（LLM）的安全挑战 问题？
  - 12.2 如何 解决 大语言模型（LLM）的安全挑战 问题？

- [点击查看答案](https://articles.zsxq.com/id_1bmbedojsj0t.html)

#### [大模型（LLMs）RAG 优化策略 —— RAG-Fusion篇](https://articles.zsxq.com/id_4ce04xwvic1z.html)

- 一、RAG 有哪些优点？
- 二、RAG 存在哪些局限性？
- 三、为什么 需要 RAG-Fusion？
- 四、说一下 RAG-Fusion 核心技术？
- 五、说一下 RAG-Fusion 工作流程？
  - 5.1 多查询生成
  - 5.2 多查询生成 技术实现（提示工程）？
  - 5.3 多查询生成 工作原理？
  - 5.4 逆向排名融合（RRF）
    - 5.4.1 为什么选择RRF？
    - 5.4.2 RRF 技术实现？
    - 5.4.3 生成性输出 用户意图保留
    - 5.4.4 生成性输出 用户意图保留 技术实现
- 六、RAG-Fusion 的优势和不足
  - 6.1 RAG-Fusion 优势
  - 6.2 RAG-Fusion 挑战

- [点击查看答案](https://articles.zsxq.com/id_4ce04xwvic1z.html)

## 大模型（LLMs）Graph RAG篇

### [Graph RAG（Retrieval-Augmented Generation） 面 —— 一种 基于知识图谱的大模型检索增强实现策略](https://articles.zsxq.com/id_dwhonmw976n7.html)

- 一、为什么需要 Graph RAG？
- 二、什么是 Graph RAG？
- 三、Graph RAG 思路介绍？
- 四、用代码 介绍 Graph RAG ？
- 五、用 示例 介绍 Graph RAG ？
- 六、Graph RAG 排序优化方式？

- [点击查看答案](https://articles.zsxq.com/id_dwhonmw976n7.html)

