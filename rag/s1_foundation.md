# 大模型面试RAG篇（一）——RAG核心概念与流程

## 引言

> 面试官问你RAG，别只背“检索增强生成”这六个字。  
> 
> 你得让他觉得：这人真干过。

大模型火了两年，RAG成了标配。简历上写着“熟悉RAG”的候选人排着队，但一问细节就卡壳：

- “Embedding模型怎么选？” —— 呃……用OpenAI的那个？  
- “相似度用余弦还是欧氏？” —— 不都一样吗？  
- “检索失败怎么办？” —— 让LLM再生成一次？那不就是幻觉吗。

如果你也有这些模糊地带，这篇文章就是为你准备的。

**基础篇**一共15个高频面试题，每一题都按“一句话答案 + 深度拆解 + 避坑指南”来写。不堆概念，只说人话，附Mermaid流程图——面试时手绘出来就是加分项。

开始之前，记住一条心法：**RAG的本质不是技术堆砌，而是让LLM学会“承认不知道”**。

## 第一部分：基础篇（1-15题）

### 模块一：RAG核心概念与流程

#### 1. 【⭐】RAG全称是什么？一句话说清楚它解决了什么问题。

**RAG = Retrieval-Augmented Generation**（检索增强生成）。

解决的是：**大模型记不住、记不准、编瞎话**的问题。  

说白了，让它“先查资料，再回答问题”，别光凭肚子里那点训练集硬撑。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#4A90E2', 'primaryBorderColor': '#1E3A8A', 'lineColor': '#F5A623', 'tertiaryColor': '#F0F4F8'}}}%%
flowchart TB
    Start([用户提问]) --> Q[问题 Embedding]
    
    subgraph RAG[🔄 RAG 核心流程]
        direction TB
        Q --> Search[向量检索<br/>从知识库召回 Top-K 文档]
        
        subgraph Knowledge[📚 外部知识库]
            K1[文档块 1]
            K2[文档块 2]
            K3[文档块 ...]
        end
        
        Search <--> Knowledge
        Search --> Merge[拼接: Prompt + 检索结果]
    end
    
    Merge --> Gen[LLM 基于资料生成]
    Gen --> Answer([输出带依据的答案])
    
    subgraph Traditional[❌ 传统 LLM]
        direction TB
        Q2[用户提问] --> Gen2[LLM 凭参数记忆回答]
        Gen2 --> Hallucination[🚫 可能产生幻觉]
    end
    
    style RAG fill:#E8F0FE,stroke:#1E3A8A,stroke-width:2px
    style Knowledge fill:#FFF3E0,stroke:#E65100,stroke-width:1.5px
    style Traditional fill:#FFEBEE,stroke:#C62828,stroke-width:1.5px
    style Gen fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style Hallucination fill:#FFCDD2,stroke:#B71C1C
```

#### 2. 【⭐】RAG和Fine-tuning的根本区别在哪？各适用什么场景？

| 维度 | RAG | Fine-tuning |
|------|-----|--------------|
| 怎么干 | 动态查外部知识库 | 把知识硬训进模型参数 |
| 知识更新 | 换文档就行，秒级生效 | 得重新训练，费钱费时 |
| 推理成本 | 检索+生成，略高 | 纯生成，低 |
| 适合场景 | 知识经常变、需要可溯源 | 固定风格/任务、私有数据不能外传 |

> 用RAG还是微调？**知识型任务无脑RAG**，风格型任务（比如客服话术、代码格式化）微调更香。

#### 3. 【⭐】RAG的标准Pipeline分几步？

一共4步，别背，要能画出来：

```mermaid
flowchart LR
    A[原始文档] --> B[文档切分+Embedding]
    B --> C[(向量数据库)]
    D[用户提问] --> E[Query Embedding]
    E --> C
    C --> F[相似度检索 Top-K]
    F --> G[拼接Prompt + 检索结果]
    G --> H[LLM生成答案]
```

面试官可能会追问：**第一步切分怎么切？** → 分块策略（固定/递归/语义/结构），下一题就讲。

#### 4. 【⭐】为什么RAG能减少幻觉？

大模型瞎编是因为它**不知道边界**——它分不清“自己学过的”和“没见过”的。

RAG强行给它加了个**外挂知识库**。生成之前先问自己：库里有相关材料吗？有，就照着说；没有，就说不知道。

说白了：**把“猜”变成“查”**。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#1E88E5', 'primaryBorderColor': '#0D47A1', 'lineColor': '#FFB74D', 'tertiaryColor': '#f9f9f9'}}}%%
flowchart TB
    subgraph Problem[🎯 问题根源：LLM的幻觉成因]
        direction TB
        A1[LLM训练方式<br/>基于海量文本的统计规律] --> A2[缺乏“认知边界”机制]
        A2 --> A3{分不清什么}
        A3 -->|❌ 误以为见过| A4[虚构不存在的事实]
        A3 -->|❌ 不确定时| A5[强行编造合理答案]
        A4 --> A6[典型表现：<br/>捏造文献、事件、数据]
        A5 --> A6
    end

    subgraph Solution[💡 RAG解决方案：从“猜”到“查”]
        direction TB
        B0[用户提问 Query] --> B1{📚 检索阶段<br/>Retrieval}
        
        B1 --> B2[向量化用户问题]
        B2 --> B3[在外部知识库中<br/>语义相似度检索]
        B3 --> B4{是否存在<br/>相关文档?}
        
        B4 -->|✅ 有相关材料| B5[📄 增强阶段<br/>Augmentation]
        B5 --> B6[将检索到的文档<br/>作为上下文拼接]
        B6 --> B7[构造增强Prompt:<br/>“基于以下资料回答”]
        B7 --> B8[🤖 生成阶段<br/>Generation]
        B8 --> B9[LLM基于提供的事实<br/>生成准确回答]
        B9 --> B10[✅ 输出：<br/>准确、可溯源]
        
        B4 -->|❌ 无相关材料| B11[🛡️ 安全兜底机制]
        B11 --> B12[LLM被强制约束:<br/>“无法回答，库中无相关信息”]
        B12 --> B13[✅ 输出：<br/>诚实的“不知道”]
    end

    subgraph Comparison[⚖️ 对比：有无RAG的本质差异]
        direction LR
        C1[❌ 无RAG<br/>纯参数化记忆] --> C2[自由生成模式<br/>凭“记忆”发挥]
        C2 --> C3[不确定时选择“编造”<br/>幻觉风险 ⬆️⬆️⬆️]
        
        C4[✅ 有RAG<br/>检索增强生成] --> C5[约束生成模式<br/>凭“事实”回答]
        C5 --> C6[不确定时选择“拒答”<br/>幻觉风险 ⬇️⬇️⬇️]
    end

    Problem --> Solution
    Solution --> Comparison

    style Problem fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    style Solution fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px
    style Comparison fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    
    style B4 fill:#FFECB3,stroke:#FFB300,stroke-width:2px
    style B11 fill:#FFF9C4,stroke:#F9A825,stroke-width:2px
    style B10 fill:#C8E6C9,stroke:#2E7D32
    style B13 fill:#C8E6C9,stroke:#2E7D32
    style A6 fill:#FFCDD2,stroke:#D32F2F
    style C3 fill:#FFCDD2,stroke:#D32F2F
    style C6 fill:#C8E6C9,stroke:#2E7D32
```

#### 5. 【⭐】Embedding模型在RAG里具体负责什么？输入输出分别是什么？

**负责把文字变成向量**，让计算机能算相似度。

- **输入**：一段文本（比如一个chunk、一句用户问题）
- **输出**：一个浮点数向量，比如 `[0.123, -0.456, 0.789, …]`，维度通常是384、768、1536。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#1E88E5', 'primaryBorderColor': '#0D47A1', 'lineColor': '#FFB74D', 'tertiaryColor': '#f9f9f9'}}}%%
flowchart LR
    subgraph Input[📥 输入阶段]
        direction TB
        I1[📝 文本输入<br/>Text Input]
        I2[类型1: 知识库文档片段<br/>Chunk / Paragraph]
        I3[类型2: 用户查询<br/>User Query]
        I4[类型3: 历史对话<br/>Conversation History]
        
        I2 --> I1
        I3 --> I1
        I4 --> I1
    end

    subgraph Core[🧠 Embedding模型核心职责]
        direction TB
        C0[🎯 核心任务:<br/>语义→向量映射<br/>Semantic-to-Vector Mapping]
        C0 --> C1[步骤1: 文本分词<br/>Tokenization]
        C1 --> C2[步骤2: 词嵌入<br/>Token → 初始向量]
        C2 --> C3[步骤3: 上下文编码<br/>Transformer/BERT编码]
        C3 --> C4[步骤4: 池化/归一化<br/>Pooling + L2 Normalization]
        C4 --> C5[📐 输出:<br/>固定维度稠密向量<br/>Dense Vector]
    end

    subgraph Output[📤 输出阶段]
        direction TB
        O1[🔢 浮点数向量<br/>Float Vector Array]
        O2[维度示例:<br/>384 / 768 / 1536]
        O3[本质:<br/>高维空间中的坐标点<br/>语义锚点]
        
        O1 --> O2
        O2 --> O3
    end

    subgraph Usage[🔧 核心用途: 相似度计算]
        direction TB
        U1[📊 向量化后的用途]
        U2[方式1: 余弦相似度<br/>Cosine Similarity]
        U3[方式2: 点积<br/>Dot Product]
        U4[方式3: 欧氏距离<br/>Euclidean Distance]
        
        U1 --> U2
        U1 --> U3
        U1 --> U4
        
        U5[💡 意义:<br/>将语义相似性问题<br/>转化为数学距离问题]
        U2 --> U5
        U3 --> U5
        U4 --> U5
    end

    subgraph Example[📋 具体示例]
        direction TB
        E1[输入文本:<br/>“RAG如何减少幻觉？”]
        E2[Embedding模型处理]
        E3[输出向量<br/>768维浮点数数组]
        E4[向量片段:<br/>0.123, -0.456, 0.789, -0.234, ...]
        
        E1 --> E2 --> E3 --> E4
    end

    Input --> Core --> Output
    Output --> Usage
    Output --> Example

    style Input fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px
    style Core fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Output fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style Usage fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style Example fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
    
    style C0 fill:#FFECB3,stroke:#FFC107,stroke-width:2px
    style C5 fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    style O3 fill:#BBDEFB,stroke:#1976D2
    style U5 fill:#E1BEE7,stroke:#8E24AA
    style E4 fill:#CFD8DC,stroke:#546E7A
```

> 选Embedding模型有个坑：**Query和Document最好用同一个模型**，不然“用户问”和“文档里写”的向量空间可能不一致。

#### 6. 【⭐】向量检索和关键词检索（BM25）各自的优缺点？什么时候用哪个？

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 向量检索（语义） | 同义词、近义词也能召回 | 训练数据偏、领域词可能崩 | 口语化提问、内容改述 |
| BM25（关键词） | 精确匹配强、可解释 | 拼写错误、同义词不行 | 专有名词搜索、代码搜索 |

> 别纠结了，**混合检索**（向量+BM25+RRF融合）已经是生产标配。

#### 7. 【⭐】什么是Top-K检索？K=1和K=10各有什么问题？

Top-K就是检索最相似的K个chunk。

- **K=1**：快，但可能漏掉关键上下文。比如问“优缺点”，只看到一个“优点”块就gg了。
- **K=10**：覆盖全，但噪音也进来了。LLM可能被不相关的块带偏。

> 经验值：**K=3~5**。再配合重排序（Re-ranking）把最准的往前放。

#### 8. 【⭐】相似度计算用余弦距离还是欧氏距离？Embedding归一化前后有区别吗？

- 余弦相似度：只看方向，不看长度。**Embedding模型输出通常已经归一化**，这时候余弦≈欧氏（因为 L2 归一化后，余弦距离和欧氏距离单调相关）。
- 欧氏距离：看绝对距离，对向量长度敏感。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#1E88E5', 'primaryBorderColor': '#0D47A1', 'lineColor': '#FFB74D', 'tertiaryColor': '#f9f9f9'}}}%%
flowchart TB
    subgraph Q[❓ 相似度计算：余弦 vs 欧氏]
        Q1[文本Embedding<br/>该用哪个？]
    end

    subgraph Cosine[📐 余弦相似度]
        direction TB
        C1[公式: cosθ = A·B / 双竖线A双竖线·双竖线B双竖线]
        C2[关注: 方向，忽略长度]
        C3[几何: 高维空间夹角]
        C4[范围: -1 到 1]
        C5[✅ 文本场景首选<br/>语义由方向决定]
    end

    subgraph Euclidean[📏 欧氏距离]
        direction TB
        E1[公式: d = 根号下 ΣAi-Bi平方]
        E2[关注: 绝对距离，对长度敏感]
        E3[几何: 空间直线距离]
        E4[范围: 0 到 +∞]
        E5[⚠️ 文本慎用<br/>长文档向量模长大→距离偏大]
    end

    subgraph Norm[⚡ L2归一化前后区别]
        direction TB
        N0[归一化: v_norm = v 除以 双竖线v双竖线]
        N0 --> N1{归一化?}
        
        N1 -->|❌ 归一化前| N2[向量长度未统一]
        N2 --> N3[余弦 ≠ 欧氏<br/>长文档欧氏距离被放大]
        N3 --> N4[比较不公平<br/>长度干扰相似度判断]
        
        N1 -->|✅ 归一化后| N5[所有向量长度=1<br/>落于单位超球面]
        N5 --> N6[📌 关键结论<br/>余弦距离 ∝ 欧氏距离平方]
        N6 --> N7[两者单调等价<br/>选哪个结果一致]
    end

    subgraph Conclusion[🎯 实践结论]
        direction LR
        P1[🏆 推荐: 余弦相似度] --> P2[理由1: Embedding模型<br/>输出通常已L2归一化]
        P2 --> P3[理由2: 文本语义<br/>由方向决定，与长度无关]
        P3 --> P4[理由3: 计算高效<br/>点积即得 cosθ = A·B]
        P4 --> P5[💡 一句话:<br/>方向比长度更有意义]
    end

    subgraph Insight[💡 深度类比]
        I1[🌍 归一化 = 把所有句子<br/>拉到同一语义球面上]
        I2[🎯 球面上的点<br/>只有方向差异，没有长度差异]
        I3[📖 长文档 vs 短文档<br/>归一化后平等比较语义]
        I4[🔑 本质:<br/>放弃冗余的长度信息<br/>保留核心的方向信息]
        
        I1 --> I2 --> I3 --> I4
    end

    Q --> Cosine
    Q --> Euclidean
    Cosine --> Norm
    Euclidean --> Norm
    Norm --> Conclusion
    Conclusion --> Insight

    style Q fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Cosine fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px
    style Euclidean fill:#FFEBEE,stroke:#E53935,stroke-width:2px
    style Norm fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style Conclusion fill:#E8F5E9,stroke:#43A047,stroke-width:2px
    style Insight fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
    
    style C5 fill:#C8E6C9,stroke:#2E7D32
    style E5 fill:#FFCDD2,stroke:#C62828
    style N6 fill:#FFF9C4,stroke:#F9A825,stroke-width:2px
    style P5 fill:#BBDEFB,stroke:#1976D2
    style I4 fill:#D1C4E9,stroke:#5E35B1
```

> 用谁？ 
> 
> **余弦**更稳，因为文本Embedding的方向比长度更有意义。  
> 
> 归一化前后：不归一化的话，长文档的向量模长大，欧氏距离会偏大。

#### 9. 【⭐】“检索-生成”闭环怎么理解？检索失败LLM能自己补救吗？

闭环不是自动的，是**人可以设计成闭环**——比如先检索，生成答案，发现不靠谱再触发二次检索。

LLM自己能补救吗？**部分能**。如果Prompt里写了“检索结果为空就调用联网搜索”，LLM能主动要求工具调用。但靠它自己凭空补？那是幻觉，不是补救。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#1E88E5', 'primaryBorderColor': '#0D47A1', 'lineColor': '#FFB74D', 'tertiaryColor': '#f9f9f9'}}}%%
flowchart TB
    subgraph Loop[🔄 检索-生成闭环设计]
        direction TB
        Start([用户提问]) --> Retrieval[阶段1: 检索<br/>从知识库召回相关片段]
        Retrieval --> Check{检索结果<br/>是否为空?}
        
        Check -->|有结果| Generate[阶段2: 生成<br/>LLM基于检索内容回答]
        Generate --> Eval{答案质量评估<br/>置信度/相关性}
        Eval -->|满意| Output([输出答案])
        Eval -->|不满意<br/>置信度低| Trigger[触发二次检索]
        Trigger --> Refine[优化Query<br/>改写/扩展关键词]
        Refine --> Retrieval2[重新检索]
        Retrieval2 --> Generate2[重新生成]
        Generate2 --> Output
        
        Check -->|无结果| FailHandle[检索失败处理]
    end

    subgraph Remedy[🛠️ LLM补救机制]
        direction TB
        R1[检索失败时LLM能做什么？]
        R1 --> R2{补救方式}
        
        R2 -->|✅ 工具调用<br/>Prompt中预先定义| R3[主动请求外部工具<br/>联网搜索/计算器/API]
        R3 --> R4[获取额外信息后<br/>生成答案]
        
        R2 -->|❌ 无工具·仅靠参数知识| R5[尝试“凭记忆”回答]
        R5 --> R6[结果: 大概率编造信息<br/>这是幻觉，不是补救]
        
        R2 -->|⚠️ 部分补救| R7[告知用户“未找到”<br/>并给出建议<br/>如调整关键词]
        R7 --> R8[诚实拒答也是一种补救]
    end

    subgraph Insight[💡 核心洞察]
        I1[闭环不是自动发生的<br/>需要人在系统设计时预设]
        I2[LLM自身不具备<br/>“主动补救”的元认知]
        I3[补救能力 = 工具调用的工程化<br/>而非模型智能]
        I4[最佳实践: <br/>检索失败 → 工具调用 → 二次检索 → 诚实拒答]
        
        I1 --> I2 --> I3 --> I4
    end

    Loop --> Remedy
    Remedy --> Insight

    style Loop fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px
    style Remedy fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Insight fill:#E8F5E9,stroke:#43A047,stroke-width:2px
    
    style Check fill:#FFECB3,stroke:#FFB300,stroke-width:2px
    style Eval fill:#FFECB3,stroke:#FFB300,stroke-width:2px
    style FailHandle fill:#FFCCBC,stroke:#E64A19
    style R3 fill:#C8E6C9,stroke:#2E7D32
    style R6 fill:#FFCDD2,stroke:#C62828
    style R8 fill:#BBDEFB,stroke:#1976D2
    style I4 fill:#D1C4E9,stroke:#5E35B1
```

#### 10. 【⭐】长上下文LLM（比如Gemini 1.5 Pro 1M token）能取代RAG吗？为什么？

**不能**。理由：

- **成本**：每次请求塞100万token，贵得离谱。
- **延迟**：处理百万token要好几秒。
- **知识更新**：上下文里的信息过时了得重新塞，还不如RAG换文档来得快。
- **可溯源**：RAG能告诉你“答案来自第3页第2段”，长上下文里怎么溯源？

> 长上下文是RAG的**补充**，不是替代。RAG负责精准召回，长上下文负责全量理解。

#### 11. 【⭐】什么是“上下文注入”？Prompt里检索结果放开头还是结尾？

上下文注入就是把检索到的chunk塞进LLM的Prompt里。

**放开头还是结尾？** 做过实验： 

- 放开头（System消息之后）：LLM会先“记住”这些资料，适合需要严格按资料回答的场景。  
- 放结尾（紧挨着用户问题）：LLM会更关注问题，适合多轮对话。

> 我的经验：**放开头 + 加分隔符**，比如 `### 参考资料 ###\n{chunks}\n### 用户问题 ###\n{query}`。

#### 12. 【⭐】RAG的召回率和精确率哪个更重要？在客服场景和科研场景分别怎么取舍？

- **召回率**：真正相关的文档有多少被检索出来。
- **精确率**：检索出来的文档有多少是相关的。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#1E88E5', 'primaryBorderColor': '#0D47A1', 'lineColor': '#FFB74D', 'tertiaryColor': '#f9f9f9'}}}%%
flowchart TB
    subgraph Def[📖 定义]
        direction LR
        R[召回率 Recall<br/>= 检索到的相关文档数 / 总相关文档数<br/>“有没有漏掉”]
        P[精确率 Precision<br/>= 检索到的相关文档数 / 检索到的总文档数<br/>“有没有给错”]
    end

    subgraph Tradeoff[⚖️ 场景取舍]
        direction TB
        S1[客服场景<br/>Customer Service] --> S2[🎯 目标: 快速准确解决用户问题<br/>用户容忍低噪声]
        S2 --> S3[❌ 低精确率后果:<br/>给一堆无关信息 → 体验崩坏]
        S3 --> S4[✅ 取舍: <b>精确率 > 召回率</b><br/>宁可少给，不能给错]
        S4 --> S5[🔧 策略: 提高相似度阈值<br/>限制返回Top-K数量<br/>后置重排序过滤]
        
        S6[科研场景<br/>Scientific Research] --> S7[🎯 目标: 全面覆盖相关文献<br/>用户容忍一定噪声]
        S7 --> S8[❌ 低召回率后果:<br/>漏掉关键论文 → 研究方向偏差]
        S8 --> S9[✅ 取舍: <b>召回率 > 精确率</b><br/>宁可多给，不可漏网]
        S9 --> S10[🔧 策略: 降低相似度阈值<br/>多路检索合并<br/>扩大检索范围]
    end

    subgraph Insight[💡 深度洞察]
        I1[精确率和召回率难以两全<br/>本质是“阈值”的调节]
        I2[客服场景: 噪声代价极高<br/>一次错误就损失信任]
        I3[科研场景: 漏检代价极高<br/>错过突破性论文可能使研究过时]
        I4[实际系统通常允许动态调节<br/>如滑动阈值滑块]
    end

    Def --> Tradeoff
    Tradeoff --> Insight

    style Def fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px
    style Tradeoff fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Insight fill:#E8F5E9,stroke:#43A047,stroke-width:2px
    
    style S4 fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style S9 fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style S3 fill:#FFCDD2,stroke:#C62828
    style S8 fill:#FFCDD2,stroke:#C62828
```

> 场景

- **客服场景**：精确率更重要。用户问你订单号，你给一堆不相关的，体验直接崩。  
- **科研场景**：召回率更重要。宁可多给几篇，不能漏掉关键论文。

> 实际生产中，**先保证召回率≥90%，再用重排序提精确率**。

#### 13. 【⭐】HyDE（Hypothetical Document Embeddings）是什么？举个具体例子说明。

HyDE的做法：**先用LLM对用户问题生成一个“假设文档”**，然后用这个假设文档去向量库里检索。

> 例子：用户问“怎么注册AWS账号”。  
> 
> 普通检索：“register AWS account” → 可能匹配到“account management”这种泛文档。
>   
> HyDE：让LLM先写一篇“假设文档”——《注册AWS账号的步骤：1.打开官网 2.点击注册...》。 
>  
> 这个假设文档里有很多关键词（“注册”、“步骤”、“验证邮箱”），拿去检索反而更准。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#1E88E5', 'primaryBorderColor': '#0D47A1', 'lineColor': '#FFB74D', 'tertiaryColor': '#f9f9f9'}}}%%
flowchart TB
    subgraph Normal[🔍 普通检索]
        direction TB
        N1[用户问题:<br/>“怎么注册AWS账号”] --> N2[直接Embedding<br/>Query向量化]
        N2 --> N3[向量检索]
        N3 --> N4[召回文档]
        N4 --> N5[匹配结果:<br/>“account management”<br/>“AWS overview”<br/>缺乏具体步骤]
    end

    subgraph HyDE[✨ HyDE检索]
        direction TB
        H1[用户问题:<br/>“怎么注册AWS账号”] --> H2[步骤1: LLM生成<br/><b>假设文档</b>]
        H2 --> H3[假设文档内容:<br/>“注册AWS账号的步骤：<br/>1.打开官网 2.点击注册<br/>3.填写信息 4.验证邮箱...”]
        H3 --> H4[步骤2: 用假设文档<br/>进行Embedding]
        H4 --> H5[步骤3: 向量检索<br/>使用假设文档向量]
        H5 --> H6[步骤4: 召回真实文档]
        H6 --> H7[匹配结果:<br/>“AWS注册详细教程”<br/>“如何创建AWS账号”<br/>高相关步骤文档]
    end

    subgraph Compare[⚖️ 核心差异]
        direction LR
        C1[普通检索:<br/>用<b>问题</b>找文档] --> C2[问题短，语义稀疏<br/>易匹配到泛化内容]
        C3[HyDE:<br/>用<b>假设答案</b>找文档] --> C4[假设文档长，关键词丰富<br/>与真实文档分布更接近]
    end

    subgraph Cost[⚠️ 代价]
        D1[额外LLM生成一次<br/>延迟增加 1-3秒]
        D2[Token成本增加<br/>假设文档长度可控]
        D3[适用场景:<br/>对准确率要求高<br/>对延迟不敏感的任务]
    end

    Normal --> HyDE
    HyDE --> Compare
    Compare --> Cost

    style Normal fill:#FFEBEE,stroke:#E53935,stroke-width:2px
    style HyDE fill:#E8F5E9,stroke:#43A047,stroke-width:2px
    style Compare fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Cost fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px
    
    style N5 fill:#FFCDD2,stroke:#C62828
    style H7 fill:#C8E6C9,stroke:#2E7D32
    style H2 fill:#FFF9C4,stroke:#F9A825
    style H3 fill:#BBDEFB,stroke:#1976D2
```

**代价**：多了一次LLM生成，延迟和成本增加。

#### 14. 【⭐】什么是“查询改写”（Query Rewriting）？什么时候需要？

查询改写：把用户的口语问题转成更适合检索的表述。

> 例子：用户问“那个蓝色的按钮在哪？”，改写为“蓝色按钮的位置”。

**什么时候需要？**  

- 用户问题里指代不清（“它”、“那个”）  
- 太口语化（“咋整”）  
- 太长、带无关情绪词（“救命啊，为什么我…”）

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#1E88E5', 'primaryBorderColor': '#0D47A1', 'lineColor': '#FFB74D', 'tertiaryColor': '#f9f9f9'}}}%%
flowchart TB
    subgraph Input[📥 原始用户问题]
        I1[“那个蓝色的按钮在哪？”<br/>指代不清]
        I2[“咋整”<br/>太口语化]
        I3[“救命啊，为什么我…”<br/>带情绪词]
    end

    subgraph Rewrite[✍️ 查询改写 Query Rewriting]
        direction TB
        R0[🎯 目标: 将口语/指代不清<br/>转为适合检索的表述]
        R1[输入: 原始问题]
        R1 --> R2{用LLM改写<br/>Prompt示例:<br/>“请把用户问题改写成<br/>适合搜索引擎的关键词”}
        R2 --> R3[输出: 改写后查询]
        R3 --> R4[例: “那个蓝色的按钮在哪？”<br/>→ “蓝色按钮的位置”]
    end

    subgraph When[🕐 什么时候需要？]
        direction LR
        W1[❓ 指代不清] --> W1E[“它”、“那个”、“这东西”]
        W2[🗣️ 口语化严重] --> W2E[“咋整”、“咋搞”、“这东西咋用”]
        W3[😤 情绪词/废话] --> W3E[“救命”、“疯了”、“无语”]
        W4[📏 问题过短] --> W4E[“价格”、“在哪”<br/>缺乏上下文]
        W5[🔄 多轮对话] --> W5E[需要结合历史<br/>消除指代]
    end

    subgraph Compare[⚖️ 效果对比]
        direction TB
        C1[❌ 不改写直接检索] --> C2[可能匹配到:<br/>“蓝色 UI 设计规范”<br/>“按钮样式大全”]
        C3[✅ 改写后检索] --> C4[匹配到:<br/>“蓝色按钮位置说明”<br/>“某某页面按钮坐标”]
    end

    subgraph Cost[⚠️ 注意事项]
        D1[额外LLM调用一次<br/>增加延迟与成本]
        D2[改写质量依赖Prompt设计<br/>可能过度改写或丢失原意]
        D3[简单场景无需改写<br/>仅当检索效果差时使用]
    end

    Input --> Rewrite
    Rewrite --> When
    When --> Compare
    Compare --> Cost

    style Input fill:#FFEBEE,stroke:#E53935,stroke-width:2px
    style Rewrite fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px
    style When fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Compare fill:#E8F5E9,stroke:#43A047,stroke-width:2px
    style Cost fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    
    style I1 fill:#FFCDD2,stroke:#C62828
    style I2 fill:#FFCDD2,stroke:#C62828
    style I3 fill:#FFCDD2,stroke:#C62828
    style R4 fill:#C8E6C9,stroke:#2E7D32
    style C2 fill:#FFCDD2,stroke:#C62828
    style C4 fill:#C8E6C9,stroke:#2E7D32
```

> 一个简单做法：用LLM自己改写自己，Prompt写“请把用户问题改写成适合搜索引擎的关键词”。

#### 15. 【⭐】RAG里LLM扮演什么角色？只负责生成吗？

不，LLM至少干三件事：

1. **生成最终答案**（这是明面上的）  
2. **改写用户问题**（如果用了查询改写）  
3. **判断检索结果够不够**（高级RAG里，LLM会说“资料不足，我需要再搜一下”）

> 有的框架甚至让LLM决定：先搜A，从A的结果里再搜B，形成**多跳检索**。LLM变成“检索规划员”。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#1E88E5', 'primaryBorderColor': '#0D47A1', 'lineColor': '#FFB74D', 'tertiaryColor': '#f9f9f9'}}}%%
flowchart TB
    subgraph Input[📥 用户输入]
        Q[原始问题]
    end

    subgraph Role1[🧠 角色1: 查询改写]
        direction TB
        LLM1[LLM 改写器] --> RQ[改写后问题<br/>适合检索的表述]
        Q --> LLM1
    end

    subgraph Role2[🔍 角色2: 检索决策器]
        direction TB
        RQ --> Retriever[向量检索<br/>召回若干文档]
        Retriever --> Judge{LLM 判断器<br/>检索结果足够吗？}
        Judge -->|✅ 足够| Gen[进入生成阶段]
        Judge -->|❌ 不足/缺失关键信息| NeedMore[触发补充检索]
    end

    subgraph Role3[🔄 角色3: 检索规划员 多跳检索]
        direction TB
        NeedMore --> Planner[LLM 规划员]
        Planner --> PlanA[“先搜A概念”]
        PlanA --> RetA[检索A]
        RetA --> Extract[从A结果中提取B关键词]
        Extract --> PlanB[“再搜B”]
        PlanB --> RetB[检索B]
        RetB --> Merge[合并多轮结果]
        Merge --> Gen
    end

    subgraph Role4[📝 角色4: 答案生成器]
        direction TB
        Gen --> AnswerGen[LLM 生成器]
        AnswerGen --> Final[最终答案]
        Judge -->|✅ 足够| AnswerGen
    end

    subgraph Insight[💡 深度洞察]
        I1[LLM在RAG中不只是文本生成器<br/>更是检索流程的决策中枢]
        I2[改写+判断+规划+生成<br/>形成闭环智能检索]
        I3[高级框架如 ReAct / Self-RAG<br/>LLM自主决定何时检索、检索什么]
    end

    Input --> Role1
    Role1 --> Role2
    Role2 --> Role4
    Role2 --> Role3
    Role3 --> Role4
    Role4 --> Insight

    style Input fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px
    style Role1 fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Role2 fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style Role3 fill:#FFEBEE,stroke:#E53935,stroke-width:2px
    style Role4 fill:#E8F5E9,stroke:#43A047,stroke-width:2px
    style Insight fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
    
    style Judge fill:#FFECB3,stroke:#FFB300,stroke-width:2px
    style Planner fill:#FFCCBC,stroke:#E64A19
    style AnswerGen fill:#C8E6C9,stroke:#2E7D32
    style I1 fill:#D1C4E9,stroke:#5E35B1
```

## 结尾

写到这里，基础篇的15个问题已经覆盖了RAG面试中80%的考点。

回顾一下，其实就三条线：

1. **怎么存** —— Embedding、分块、向量数据库  
2. **怎么查** —— 相似度、Top-K、混合检索、HyDE、查询改写  
3. **怎么用** —— 上下文注入、闭环设计、LLM的多角色协作

记住那个心法：**RAG不是让LLM变聪明，而是给它装了个“外挂硬盘”和“查资料的习惯”**。

下一篇我们进入 **进阶篇**：分块策略的坑、Embedding微调实战、Re-ranking的几种玩法、以及Self-RAG怎么让LLM自己决定要不要搜。

如果觉得有帮助，**点个赞/在看**，或者转发给那个正在准备大模型面试的朋友。

评论区留下你想深入的问题——下一篇我专门写。