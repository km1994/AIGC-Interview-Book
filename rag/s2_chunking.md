# 大模型面试RAG篇（二）——分块策略 15 问全解析

> 本文是 RAG 面试系列的第二模块。分块策略直接影响检索命中率、大模型回答质量——所谓“RAG 70%的效果由分块决定”，绝非虚言。下面带你逐一拆解这 15 道高频面试题，附代码、图解和实战坑点。

## ✍️ 引言

> “你的 RAG 系统检索不到正确答案，90% 不是 Embedding 模型不够强，也不是向量数据库选错了——**是分块切得稀烂**。”

这是我带过的第 20 个候选人，在纸上画了半天终于承认：一个 200 字的段落被他一刀切在两块里，关键结论和证据从此天各一方。面试官又问：“chunk_overlap 设 50% 会怎样？” 他愣住，然后沉默。

**分块策略**，在 RAG 面试里就是那道“看起来谁都能说两句，一深挖全露馅”的题。但偏偏，它决定了检索的下限——而你，根本逃不掉。

这一篇，我直接把 **模块二：分块策略（16-30题）** 的 15 道高频题，揉碎了喂给你。代码、流程图、阈值经验、AST 解析、双栏 PDF 坑点……还有面试官最爱追问的“延迟分块到底反转了什么顺序”。

看完这一篇，你不仅能答，还能让面试官点头说 **“嗯，这人是真做过 RAG 的”**。

## 16. 【⭐】为什么说“RAG 70%的效果由分块决定”？不服来辩。

**核心观点**：  
RAG 流程中，**检索是下限，生成是上限**。而检索的“原子单位”就是 chunk。切不好块，顶层设计再牛也救不了——就像厨师用烂食材做菜。

**三个硬理由**：

1. **语义完整性**：chunk 太大 → 引入噪声，污染向量表征；chunk 太小 → 丢失上下文，关键信息被割裂。一个段落里前两句是背景，中间是核心结论，最后是数据支撑——切错了位置，结论和数据就分家。
2. **检索匹配粒度**：Embedding 模型对 128~512 token 的文本表现最佳。短文本容易过泛化（什么都像），长文本容易淹没关键信号。
3. **大模型上下文依赖**：传给 LLM 的 chunk 必须**自包含**。例如问题“第二种方案的时间复杂度是多少？”——如果 chunk 里没带上“第二种方案”的定义，LLM 只能瞎猜。

**反例说服面试官**：  
一个 5000 字的技术文档，按 200 字固定长度切块。某段落在 199 字处被截断，丢失了“综上所述，不建议使用该方法”的结论。用户问“这个方法有什么坑？”，检索到前半段全是正面描述，RAG 回答“该方法表现良好”——完全翻车。

```mermaid
flowchart LR
    A[原始文档] --> B{分块策略}
    B -->|过小| C[语义碎片化<br/>关键信息被割裂]
    B -->|过大| D[噪声过多<br/>向量被稀释]
    B -->|合理| E[语义完整+粒度适中]
    C --> F[检索命中率低]
    D --> F
    E --> G[高精度检索]
    F --> H[LLM答案错误]
    G --> I[LLM答案准确]
    
    style C fill:#FFCDD2
    style D fill:#FFCDD2
    style E fill:#C8E6C9
    style H fill:#FFCDD2
    style I fill:#C8E6C9
```

## 17. 【⭐】固定长度分块怎么实现？重叠窗口设多大？给个代码片段。

**实现方式**：按字符数或 token 数切分，通常配合 `chunk_overlap` 让相邻块共享部分内容。重叠能缓解边界信息断裂问题。

**重叠窗口设多大？** 

- 经验值：`overlap = chunk_size * 0.1 ~ 0.25`。  
- 128 token 的块 → overlap 12~32 token；512 token 的块 → overlap 50~128 token。  
- 原则：重叠长度 ≥ 关键句的平均长度（中文约 20~40 字符）。  

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#4A90E2', 'primaryBorderColor': '#1E3A8A', 'lineColor': '#F5A623', 'tertiaryColor': '#F0F4F8'}}}%%
flowchart TB
    subgraph Input["📄 输入文本"]
        A[原始文档/字符串]
    end

    subgraph Preprocess["⚙️ 预处理（可选）"]
        B{是否需要<br/>词边界对齐？}
        B -->|是| C["使用 jieba 分词<br/>按词拼接"]
        B -->|否| D[直接使用原始文本]
        C --> E["调用 tiktoken 编码器<br/>转为 token 序列"]
        D --> E
    end

    subgraph Core["🔄 滑动窗口分块"]
        E --> F["初始化 start = 0"]
        F --> G{"start < 总 token 数？"}
        G -->|是| H["计算 end = min(start + chunk_size, len(tokens))"]
        H --> I["截取 tokens[start:end]"]
        I --> J[解码为文本块]
        J --> K[添加到 chunks 列表]
        K --> L["更新 start += chunk_size - overlap<br/>  ⬅️ 滑动步长"]
        L --> G
        G -->|否| M[返回 chunks 列表]
    end

    subgraph Output["📤 输出结果"]
        M --> N[分块完成]
    end

    subgraph OverlapHint["📐 重叠窗口设计经验"]
        O["overlap 推荐 = chunk_size * 0.1 ~ 0.25"]
        P["例: 512 tokens → overlap 50~128"]
        Q["原则: 重叠长度 ≥ 关键句平均长度<br/>（中文 20~40 字符）"]
    end

    OverlapHint -.-> Core

    style Input fill:#E8F0FE,stroke:#1E3A8A,stroke-width:2px
    style Preprocess fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    style Core fill:#E0F2F1,stroke:#00695C,stroke-width:2px
    style Output fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    style OverlapHint fill:#F3E5F5,stroke:#6A1B9A,stroke-width:1px,stroke-dasharray:5 5
    style B fill:#FFECB3,stroke:#FFB300
    style G fill:#FFCCBC,stroke:#E64A19
    style L fill:#C8E6C9,stroke:#2E7D32
```

**代码片段（Python + tiktoken）**：

```python
import tiktoken

def fixed_size_chunking(text, chunk_size=500, overlap=100, encoding_name="cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += chunk_size - overlap  # 滑动起点
    return chunks

# 示例
doc = "..." * 1000
chunks = fixed_size_chunking(doc, chunk_size=512, overlap=64)
print(f"生成 {len(chunks)} 个块")
```

**坑点提醒**：中文按字符切分容易打乱词边界（“机器/学习”变成“机器/学”）。建议先分词，再按 token 数切分，或者直接用 `jieba` 按词边界对齐。

## 18. 【⭐】递归分块的分隔符优先级怎么设计？中文和英文有什么区别？

**设计原则**：从**粗粒度到细粒度**递归尝试分割，优先保持自然语义边界。

优先级：

```s
段落边界（\n\n） > 换行符（\n） > 句子边界（。！？；） > 分句（，、） > 字符
```

**中文与英文的区别**：

| 维度 | 英文 | 中文 |
|------|------|------|
| 句子分隔符 | `. ! ?` 后跟空格 | `。！？；` 无需空格 |
| 段落分隔符 | `\n\n` 或 `\r\n\r\n` | 同样 `\n\n`，但中文段落首行缩进不是可靠标记 |
| 特殊处理 | 缩写（Mr. Dr.）需要避开 | 无此问题 |
| 推荐分隔符优先级 | `\n\n` > `\n` > `. ` > `, ` > 空格 | `\n\n` > `\n` > `。！？；` > `，、` > 字符 |

**实现思路**（伪代码）：

```python
def recursive_split(text, max_size, separators):
    if len(text) <= max_size:
        return [text]
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = []
            cur_len = 0
            for part in parts:
                if cur_len + len(part) + len(sep) <= max_size:
                    current.append(part)
                    cur_len += len(part) + len(sep)
                else:
                    if current:
                        chunks.append(sep.join(current))
                    current = [part]
                    cur_len = len(part)
            if current:
                chunks.append(sep.join(current))
            # 递归处理超长块
            result = []
            for chunk in chunks:
                if len(chunk) > max_size:
                    result.extend(recursive_split(chunk, max_size, separators[1:]))
                else:
                    result.append(chunk)
            return result
    # 兜底：按字符切分
    return [text[i:i+max_size] for i in range(0, len(text), max_size)]
```

![](img/18_1.png)

**面试加分点**：  提到 LangChain 的 `RecursiveCharacterTextSplitter` 默认分隔符 `["\n\n", "\n", " ", ""]`，但对中文效果一般。实际生产中可自定义添加 `["。", "！", "？", "；", "，", "、"]`。

![](img/18_2.png)

## 19. 【⭐】语义分块怎么判断语义边界？相似度阈值一般设多少？

**核心原理**：利用 Embedding 模型计算相邻句子或窗口的**语义相似度**，相似度曲线发生“骤降”的位置视为语义边界。

**实现步骤**：

1. 将文本分句（用 NLTK、`pkuseg` 或简单正则）。
2. 计算每个句子的向量（或滑动窗口内句子的平均向量）。
3. 计算相邻句子的余弦相似度。
4. 找到相似度低于阈值的点，在此处分块。

```mermaid
flowchart TB
    subgraph Step1["📝 步骤1: 分句"]
        S1["输入文本"] --> S2["分句器: NLTK / pkuseg / 正则"]
        S2 --> Sentences["句子列表: S1, S2, ..., Sn"]
    end

    subgraph Step2["🔢 步骤2: 向量化"]
        Sentences --> Embed["Embedding模型<br/>ada-002 / bge-large-zh"]
        Embed --> Vectors["每个句子的向量 v1...vn"]
    end

    subgraph Step3["📐 步骤3: 计算相似度"]
        Vectors --> SimLoop["for i = 1 to n-1"]
        SimLoop --> Cosine["计算余弦相似度<br/>cos_sim(v_i, v_{i+1})"]
        Cosine --> SimList["相似度列表 sim[1..n-1]"]
    end

    subgraph Step4["⚖️ 步骤4: 边界判定"]
        SimList --> Threshold{"相似度 &lt; 阈值?"}
        Threshold -->|是| Boundary["标记为语义边界<br/>在此处分块"]
        Threshold -->|否| Continue["合并为同一块"]
        Boundary --> NextPair["下一对"]
        Continue --> NextPair
        NextPair --> Threshold
    end

    Step4 --> Chunks["输出语义块"]

    style Step1 fill:#E3F2FD,stroke:#1565C0
    style Step2 fill:#E8F5E9,stroke:#2E7D32
    style Step3 fill:#FFF3E0,stroke:#EF6C00
    style Step4 fill:#F3E5F5,stroke:#6A1B9A
    style Threshold fill:#FFEBEE,stroke:#C62828
```

**相似度阈值设置**：

- 通用范围：**0.5 ~ 0.7**，取决于 Embedding 模型和领域。
- 模型 `text-embedding-ada-002`：建议 **0.65** 左右。
- 模型 `bge-large-zh`：建议 **0.55** 左右（因为 BGE 的相似度分布整体偏低）。
- 动态方法：计算相似度序列的**均值减一个标准差**作为阈值。

```mermaid
flowchart LR
    subgraph ThresholdGuide["🎯 相似度阈值参考"]
        direction TB
        T1["通用范围: 0.5 ~ 0.7"] --> T2["text-embedding-ada-002: 约0.65"]
        T2 --> T3["bge-large-zh: 约0.55<br/>（分布整体偏低）"]
        T3 --> T4["动态阈值: mean(sim) - std(sim)"]
    end

    subgraph Note["📌 注意事项"]
        N1["⚠️ 短文档 &lt;10句 效果差"]
        N2["⚡ 计算开销大，不适合海量实时"]
        N3["🔧 优化方案: 固定长度粗切 + 语义细切"]
    end

    ThresholdGuide ~~~ Note
    style ThresholdGuide fill:#E8F5E9,stroke:#2E7D32
    style Note fill:#FFF8E1,stroke:#F9A825
    style T1 fill:#E3F2FD
    style T2 fill:#E3F2FD
    style T3 fill:#E3F2FD
    style T4 fill:#FFF3E0
```

**注意事项**：
- 短文档（< 10 句）效果差，因为统计不显著。
- 计算开销大，不适合海量文档实时分块。
- 可以先用固定长度粗切，再对每个块做语义细切，平衡效率与效果。

## 20. 【⭐】结构化分块（按标题/段落）的解析库有哪些？遇到长章节怎么办？

**常用解析库**：

| 文档类型 | 推荐库 | 特点 |
|----------|--------|------|
| Markdown | `markdown-it-py`、`mistune` | 提取标题层级（H1~H6） |
| HTML | `BeautifulSoup`、`lxml` | 按 `h1/h2/p/div` 切分 |
| Word (.docx) | `python-docx` | 按段落、表格、样式分割 |
| PDF（非扫描） | `pymupdf`、`pdfplumber` | 按页面/段落，保留标题样式 |
| 学术论文（LaTeX） | `latex2text`、正则匹配 `\section` | 按 section/subsection 切 |

**长章节处理策略**：

1. **层级递归**：如果一个章节正文超过 `max_chunk_size`，递归地用子标题继续分割。
2. **标题锚点保留**：每个 chunk 头部携带父标题链（如 `"## 第三章 ## 3.2 实验设置"`），保证自包含。
3. **滑动窗口与结构混合**：对超长章节，按段落自然边界切分后，相邻块保留重叠的引导句。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#4A90E2', 'primaryBorderColor': '#1E3A8A', 'lineColor': '#F59E0B', 'tertiaryColor': '#F3F4F6'}}}%%
flowchart TB
    subgraph Libs[📚 结构化分块解析库]
        A1["LangChain (RecursiveCharacterTextSplitter)"]
        A2["LlamaIndex (NodeParser + SemanticSplitter)"]
        A3["Unstructured (HTML/PDF分层解析)"]
        A4["HuggingFace TokenTextSplitter"]
        A5["spaCy / NLTK (句子边界感知)"]
    end

    subgraph Input[📄 输入文档]
        B0([开始]) --> B1["选择/配置解析库"]
        B1 --> B2["按标题/段落解析为初步结构"]
    end

    subgraph Decision[⚖️ 长章节检测]
        C0{存在章节正文长度<br/>> max_chunk_size ?}
        C0 -->|否| C1["直接输出语义完整的chunks"]
        C1 --> End([结束])
        C0 -->|是| C2["进入长章节处理流程"]
    end

    subgraph LongChap[🔁 长章节处理策略]
        direction TB
        D1["递归层级分割<br/>若有子标题 → 按子标题递归分割"]
        D2["标题锚点保留<br/>每个chunk头部携带完整父标题链<br/>示例: '## 第三章 ## 3.2 实验设置'"]
        D3{"分割后仍存在超长段落<br/>且无子标题可递归?"}
        D4["滑动窗口 + 自然边界切分<br/>- 按句子/段落边界切分<br/>- 相邻块保留重叠引导句 (如保留前一块的最后1~2句)"]
        D5["组装最终chunks<br/>每个chunk包含: 锚点链 + 正文 + 元数据"]
        
        C2 --> D1
        D1 --> D2
        D2 --> D3
        D3 -->|是| D4
        D4 --> D5
        D3 -->|否| D5
        D5 --> End
    end

    subgraph Output[📤 输出]
        End
    end

    style Libs fill:#E8F0FE,stroke:#4A90E2,stroke-width:2px
    style Input fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px
    style Decision fill:#FFF3E0,stroke:#F59E0B,stroke-width:2px
    style LongChap fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    style Output fill:#E8F5E9,stroke:#43A047,stroke-width:2px
    style C0 fill:#FFECB3,stroke:#FFB300
    style D3 fill:#FFCCBC,stroke:#E64A19
```

**实战代码（Markdown 分块示例）**：

```python
import re

def split_by_headers(md_text, max_chunk_size=1000):
    lines = md_text.split('\n')
    chunks = []
    current_chunk = []
    current_headers = []  # 记录当前标题栈
    
    for line in lines:
        header_match = re.match(r'(#{1,6})\s+(.*)', line)
        if header_match:
            # 遇到新标题，先保存当前块
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            # 更新标题栈
            level = len(header_match.group(1))
            title = header_match.group(2)
            current_headers = current_headers[:level-1] + [title]
            current_chunk = [line]
        else:
            current_chunk.append(line)
            # 如果当前块超出大小，需要强制切割
            if len('\n'.join(current_chunk)) > max_chunk_size:
                # 在段落边界切割，并保留当前标题栈
                chunks.append('\n'.join(current_chunk[: -1]))
                current_chunk = current_chunk[-1:]
                # 在切割块前补上标题链
                if current_headers:
                    current_chunk.insert(0, '## ' + ' > '.join(current_headers))
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks
```

## 21. 【⭐⭐】延迟分块（查询时分块）反转了什么顺序？存什么、什么时候切？

**核心思想**：  

- 传统流程：**先分块 → 存向量**。 
- 延迟分块：**先存更细粒度的单元（比如句子）→ 查询时动态组合成块**。

**反转顺序**：

- 存储粒度：句子或短语（而不是固定块）。
- 检索时：根据 query 检索相关句子，然后以检索到的句子为中心，**动态拉取前后 N 个句子**组成上下文窗口。

**存什么**：

- 每个句子的 embedding 向量 + 位置信息（文档 ID、段落 ID、句索引）。
- 可选：句子之间的相似度图（用于扩展上下文）。

**什么时候切**：

- 不在索引阶段切，而是在**查询时实时切**。  
- 流程：query → 找到相关句子 → 扩展邻居句子 → 合并成最终块 → 传给 LLM。

```mermaid
sequenceDiagram
    participant User
    participant Index as 索引（存句子）
    participant Retriever
    participant Expander as 上下文扩展器
    participant LLM
    
    User->>Retriever: 查询 Q
    Retriever->>Index: 检索相似句子（top-k）
    Index-->>Retriever: 返回句子列表 S
    Retriever->>Expander: 传递 S + 位置信息
    Expander->>Expander: 对每个句子，取前后 L 个邻居句子
    Expander-->>User: 合并后的语义块
    User->>LLM: 块 + 问题
    LLM-->>User: 答案
```

**优缺点**：
- ✅ 避免索引阶段 chunk size 选错导致的永久性信息丢失。
- ✅ 能自适应不同 query 对上下文宽度的需求（有些问题需要更多背景）。
- ❌ 查询延迟增加（需要额外扩展和合并）。
- ❌ 对存储和计算要求更高（需记录细粒度位置关系）。

**典型应用**：  
`Contextual Retrieval`（Anthropic 提出）、`ColBERT` 的后期交互思想。

## 22. 【⭐】chunk size 太小和太大分别会导致什么问题？给个具体例子。

| 问题 | 太小（< 100 token） | 太大（> 1000 token） |
|------|---------------------|----------------------|
| **语义完整性** | 句子被拆散，“虽然…但是…”分成两半 | 多个主题混在一起，向量平均化 |
| **检索精度** | 召回率高但精确率低（单个词匹配） | 精确率可能高，但相关段落被埋没 |
| **LLM 输入** | 上下文不足，缺乏背景 | 超出 LLM 窗口或稀释关键信息 |
| **典型错误** | 问“苹果公司的股价”，chunk 只有“苹果公司”没有“股价” | 问“错误码 404 含义”，chunk 包含整个 HTTP 协议章节，检索得分被其他内容拉低 |

**具体例子**：

- **太小（50 token）**：  
  原文：“RAG 的核心步骤包括：索引、检索、生成。其中索引又分为文档解析、分块、向量化。”  
  切成两块：  
  块1：“RAG 的核心步骤包括：索引、检索、生成。”  
  块2：“其中索引又分为文档解析、分块、向量化。”  
  用户问“RAG 索引阶段有哪些子步骤？” → 块1 不含答案，块2 没有“索引”一词导致检索得分低。

- **太大（1500 token）**：  
  文档包含 A/B/C 三个完全不相关的子话题。用户问“话题 A 的参数是多少？” → 检索时整块与 query 相似度为 0.6（因为包含 B、C 的噪声），而另一个更精确的小块本来可以得到 0.9，但不存在。

## 23. 【⭐】chunk_overlap 太大会有什么副作用？重叠部分会被重复检索吗？

**副作用**：

1. **存储膨胀**：每个块大小 = `chunk_size`，但有效新信息只有 `chunk_size - overlap`。overlap 占 50% 时，存储量翻倍。
2. **检索冗余**：同一个句子出现在相邻两个块中，query 可能同时命中这两个块，造成结果去重负担。
3. **LLM 重复消费**：传给 LLM 的上下文中，大量信息重复，浪费 token，且可能让模型对重复内容过度关注。

**重叠部分会被重复检索吗？**  
**会**。Embedding 模型对不同块内的相同文本生成几乎相同的向量，query 检索时两个块都会进入召回列表。需要在检索后做**去重**或**MMR**（最大边际相关）来减少冗余。

**解决思路**：

- 限制 overlap ≤ 25% chunk_size。
- 检索后对 top-k 结果进行语义去重（例如按 Jaccard 相似度过滤）。
- 使用 `parent-document` 策略：只检索子块，返回父块。

```mermaid
flowchart LR
    A[原文: ABCDEFG] --> B[chunk1: ABCDE<br/>overlap=3]
    A --> C[chunk2: CDEFG]
    B --> D[向量库存储]
    C --> D
    D --> E[Query: 'C D']
    E -->|命中| F[chunk1]
    E -->|命中| G[chunk2]
    F --> H[去重或合并]
    G --> H
    H --> I[LLM]
    
    style F fill:#FFCDD2
    style G fill:#FFCDD2
    style H fill:#C8E6C9
```

## 24. 【⭐⭐】代码文件怎么分块？按函数、按类还是按语义？AST 有用吗？

**最佳实践**：**按语法结构（AST）分块**，每个块对应一个函数、方法或类。

**分块策略对比**：

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 按行固定长度 | 简单 | 破坏语法结构，注释和代码混在一起 | 临时脚本 |
| 按函数/类 | 保持语义完整，调用关系清晰 | 超大函数仍需进一步切分 | 通用 |
| 按语义（docstring + 代码） | 检索时更容易匹配意图 | 需要解析 AST | 高质量代码库 |
| 按 git commit 块 | 保留变更上下文 | 依赖版本历史 | 代码审查 RAG |

**AST 的作用**：

- 精确识别函数边界、参数列表、内部逻辑块（`if/for/while`）。
- 提取函数签名、文档字符串（docstring）作为块的元数据，增强检索。
- 可以按**依赖图**组织：先检索函数定义，再自动拉取其调用的子函数。

**代码片段（Python + `ast` 模块）**：

```python
import ast

def extract_functions(code):
    tree = ast.parse(code)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_code = ast.get_source_segment(code, node)
            functions.append({
                'name': node.name,
                'docstring': ast.get_docstring(node),
                'code': func_code,
                'lineno': node.lineno
            })
    return functions

# 示例
code = """
def add(a, b):
    '''Return sum of two numbers.'''
    return a + b

def multiply(a, b):
    return a * b
"""
print(extract_functions(code))
```

**进阶技巧**：  

- 对类中的方法按类整体索引，同时保留每个方法的独立块。  
- 为代码块增加**调用链**上下文：检索到函数 A，自动附带它调用的 B、C 的定义（通过静态分析）。

## 25. 【⭐】表格在 RAG 里怎么处理？整表 embedding 还是转成文本描述？

**多种策略，取决于表格大小和查询类型**：

| 策略 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| 整表 embedding | 将表格渲染为 markdown/html 字符串，直接向量化 | 简单，不丢失结构 | 大表会超过 embedding 模型上限（通常 512 token），且检索粒度粗 |
| 转文本描述 | 用 LLM 将表格总结为自然语言段落 | 语义紧凑，易于检索 | 丢失精确数值和行列关系 |
| 行/列独立分块 | 每行（或每列）单独存储，检索时返回多行 | 可精确查询某行数据 | 丢失表格上下文（表头需要重复携带） |
| 混合索引 | 表格整体存一次（用于宽泛问题），同时每行存一次（用于精确查询） | 兼顾召回和精度 | 存储和检索复杂度高 |

**最佳实践（面试回答）**：  

> 对于行数 < 50 的小表，整表 embedding + 转文本描述两者都存，query 时先用描述检索，命中后返回原表。 
>  
> 对于大表，将表格按行拆成多个块，每块包含表头 + 该行数据，同时额外存储一份表格摘要。

**代码示例（按行分块）**：

```python
def table_to_chunks(df, table_name):
    header = " | ".join(df.columns)
    chunks = []
    for idx, row in df.iterrows():
        row_str = " | ".join([str(v) for v in row.values])
        chunk = f"表格: {table_name}\n表头: {header}\n行{idx+1}: {row_str}"
        chunks.append(chunk)
    return chunks
```

## 26. 【⭐⭐】图片里的文字怎么处理？OCR 之后再分块？还是用多模态 embedding？

**两种主流路径**：

1. **OCR 预处理**（传统方案）  
   - 用 `PaddleOCR`、`Tesseract` 提取图片中的文字。  
   - 将提取出的文字按阅读顺序拼接到文档原文中，然后统一分块。  
   - 优点：兼容现有 RAG 流程，成本低。  
   - 缺点：丢失图片布局、图表语义（如箭头、颜色）。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#ffffff', 'primaryColor': '#5E35B1', 'primaryBorderColor': '#311B92', 'lineColor': '#FFB74D', 'tertiaryColor': '#F3E5F5'}}}%%
flowchart TB
    Start([📷 输入图片]) --> Split{选择处理路径}

    subgraph OCR_Path [📄 OCR 预处理方案]
        direction TB
        O1[🔍 OCR 提取文字<br/>PaddleOCR / Tesseract] --> O2[🔤 按阅读顺序拼接文本]
        O2 --> O3[✂️ 统一分块策略<br/>按标题/段落递归分割]
        O3 --> O4[📦 输出文本块<br/>保留父标题链]
    end

    subgraph Multimodal_Path [🧠 多模态 Embedding 方案]
        direction TB
        M1[🖼️ 多模态编码器<br/>CLIP / GPT-4V / BLIP-2] --> M2[📐 生成图片级 Embedding]
        M2 --> M3[🔢 向量检索/聚类<br/>不显式提取文字]
        M3 --> M4[📎 基于相似度的分块<br/>视觉语义单元]
    end

    Split -->|直接提取文字| OCR_Path
    Split -->|保留视觉语义| Multimodal_Path

    OCR_Path --> End([✅ 下游任务<br/>问答/摘要/RAG])
    Multimodal_Path --> End

    style Start fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style End fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style Split fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px
    style OCR_Path fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px
    style Multimodal_Path fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px
    style O1 fill:#BBDEFB,stroke:#1976D2
    style M1 fill:#E1BEE7,stroke:#7B1FA2
```

2. **多模态 embedding**（2024+ 趋势）  
   - 直接用多模态模型（如 `CLIP`、`GPT-4V`、`NV-DINOv2`）将图片编码为向量。  
   - 与文本向量放在同一个向量空间。  
   - 查询时，既可输入文本，也可输入图片。  
   - 优点：保留视觉语义，支持图文联合检索。  
   - 缺点：需要多模态模型，推理成本高。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#FFFFFF', 'primaryColor': '#4A90E2', 'primaryBorderColor': '#1E3A8A', 'lineColor': '#F5A623', 'tertiaryColor': '#F0F4F8', 'primaryTextColor': '#1F2937', 'tertiaryTextColor': '#1F2937', 'fontFamily': 'Arial'}}}%%
flowchart TB
    Start([📷 输入图片]) --> Choose{选择处理策略}

    subgraph OCR [📄 OCR 预处理方案]
        direction TB
        O1[🔍 OCR引擎<br/>PaddleOCR / Tesseract] --> O2[📝 提取文字 + 坐标信息]
        O2 --> O3[🧩 按阅读顺序拼接为文本]
        O3 --> O4[✂️ 文本分块<br/>保留标题链/重叠窗口]
        O4 --> O5[📦 输出文本块<br/>用于关键词检索或LLM]
    end

    subgraph MM [🧠 多模态 Embedding 方案<br/>2024+ 趋势]
        direction TB
        M1[🖼️ 多模态编码器<br/>CLIP / GPT-4V / NV-DINOv2] --> M2[📐 生成图片级 Embedding 向量]
        M2 --> M3[🔗 与文本向量置于同一空间<br/>支持图文联合检索]
        M3 --> M4[🔍 查询方式<br/>文本输入 / 图片输入]
        M4 --> M5[📚 向量检索/聚类<br/>保留视觉语义]
    end

    Choose -->|文字提取优先| OCR
    Choose -->|视觉语义优先| MM

    OCR --> Downstream([🎯 下游任务])
    MM --> Downstream

    Downstream --> Example1[📖 文档问答]
    Downstream --> Example2[🔎 以图搜图/搜文]
    Downstream --> Example3[🤖 RAG 增强生成]

    style Start fill:#D1FAE5,stroke:#059669,stroke-width:2px
    style Choose fill:#FEF3C7,stroke:#D97706,stroke-width:2px
    style OCR fill:#DBEAFE,stroke:#2563EB,stroke-width:2px
    style MM fill:#F3E8FF,stroke:#9333EA,stroke-width:2px
    style Downstream fill:#FEF9C3,stroke:#CA8A04,stroke-width:2px
    style O1 fill:#BFDBFE,stroke:#1D4ED8
    style M1 fill:#E9D5FF,stroke:#7E22CE
    style M3 fill:#D8B4FE,stroke:#6B21A8
    style Example1 fill:#FFEDD5
    style Example2 fill:#FFEDD5
    style Example3 fill:#FFEDD5
```

**实际工程选择**：

| 场景 | 推荐方案 |
|------|----------|
| 扫描版 PDF 中的截图 | OCR + 文本分块 |
| 信息图（infographic）、PPT 图表 | 多模态 embedding |
| 手写文字 | OCR（多模态模型对手写效果也一般） |
| 图文混排的网页 | 多模态 embedding 或 同时存两种向量 |

**核心结论（面试考点）**：  
> OCR 是“把图片变成文本”，适用于文字为主的图片；多模态 embedding 是“保持图片本身语义”，适用于布局、颜色、形状等非文本信息至关重要的场景。两者可共存：对同一图片同时生成文本向量和多模态向量，检索时加权融合。

## 27. 【⭐】PDF 文档分块有什么坑？页眉页脚、双列排版怎么处理？

**三大坑点 + 解法**：

| 坑点 | 表现 | 解决方案 |
|------|------|----------|
| **页眉页脚** | 每页顶部/底部重复出现，污染 chunk 语义 | 解析时检测重复模式，过滤页眉页脚（`pdfplumber` 可获取页边距，裁剪掉固定区域） |
| **双列排版** | 正文从左列下半段跳到右列上半段，文本顺序错乱 | 使用 `layout=True` 模式（`pdfplumber` 或 `camelot`），按阅读顺序（从左到右，从上到下）重排文本 |
| **表格跨页** | 同一个表格被切到两页，分块后不完整 | 检测表格特征，跨页合并（`tabula-py` 支持跨页提取） |
| **字体/编码** | 中文字符变成乱码或空格 | 预先用 `pymupdf` 强制 UTF-8，或转成图片后 OCR |

**实战技巧**：  
```python
import pdfplumber

def extract_clean_text(pdf_path):
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 按布局保留顺序
            text = page.extract_text(layout=True)
            # 简单过滤页眉（假设页码在底部，页眉高度 < 30px）
            if page.height:
                # 实际需结合具体文档
                pass
            full_text.append(text)
    return "\n".join(full_text)
```

**双列重排原理图**：

```mermaid
flowchart LR
    A[原始PDF页面] --> B{检测列数}
    B -->|双列| C[按坐标分组为左列和右列]
    C --> D[左列从上到下]
    C --> E[右列从上到下]
    D --> F[合并: 左列第一段 -> 右列第一段 -> 左列第二段...]
    E --> F
    B -->|单列| G[直接按 y 坐标排序]
    F --> H[输出连续文本]
    G --> H
```

## 28. 【⭐⭐】章节标题和正文的关系怎么保持？每个 chunk 里要带父标题吗？

**强烈建议：每个 chunk 携带完整的标题路径**。  

原因：用户提问“在 3.2 节中提到的算法复杂度是多少？”——如果 chunk 里没有 `"3.2 节"` 的信息，检索就无法命中。

**实现方式**：

1. **标题锚点注入**：解析文档时维护一个栈 `current_headers = ["H1", "H2", ...]`。每遇到新的正文段落，在该段落前拼接 `" > ".join(current_headers)`。
2. **元数据分离**： 向量存储时，将标题路径作为 metadata 字段，检索时根据 query 相关性过滤或 boost 权重。
3. **层级分块**：每个标题下的内容作为一个独立块，块内自然包含标题。如果子标题内容过多，再递归分割，但分割后的子块依然携带该标题。

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#FFFFFF', 'primaryColor': '#3B82F6', 'primaryBorderColor': '#1E3A8A', 'lineColor': '#F59E0B', 'tertiaryColor': '#F3F4F6', 'primaryTextColor': '#1F2937', 'tertiaryTextColor': '#1F2937'}}}%%
flowchart TB
    Start([📄 解析文档]) --> Parse[🔍 遍历章节与正文<br/>维护标题栈 current_headers]

    Parse --> Question{每个 chunk 要带父标题吗？}
    Question -->|强烈建议| Yes[✅ 每个 chunk 携带完整标题路径]
    Question -->|也可选择| No[⚠️ 仅文本内容（不推荐）]

    Yes --> Method{三种实现方式}

    subgraph Method1 [🔗 标题锚点注入]
        M1_1[📝 遇到正文段落] --> M1_2[✂️ 拼接标题路径<br/>使用连接符]
        M1_2 --> M1_3[📦 生成 chunk<br/>内容 = 标题路径 + 正文]
    end

    subgraph Method2 [🏷️ 元数据分离]
        M2_1[📊 向量存储时] --> M2_2[📌 标题路径作为 metadata 字段]
        M2_2 --> M2_3[🔍 检索时根据 query<br/>相关性过滤或提升权重]
    end

    subgraph Method3 [📚 层级分块]
        M3_1[📂 每个标题下的内容作为独立块] --> M3_2[🔁 子标题内容过多<br/>递归分割]
        M3_2 --> M3_3[🧬 分割后的子块<br/>仍携带原标题路径]
    end

    Method --> Method1
    Method --> Method2
    Method --> Method3

    Method1 --> End([🎯 支持精准检索<br/>例如在3.2节中提到的算法复杂度])
    Method2 --> End
    Method3 --> End

    No --> Risk[❌ 检索无法命中章节信息]
    Risk --> End

    style Start fill:#D1FAE5,stroke:#059669,stroke-width:2px
    style Question fill:#FEF3C7,stroke:#D97706,stroke-width:2px
    style Yes fill:#DBEAFE,stroke:#2563EB,stroke-width:2px
    style No fill:#FEE2E2,stroke:#DC2626,stroke-width:2px
    style Method fill:#E0E7FF,stroke:#4338CA,stroke-width:2px
    style Method1 fill:#F3E8FF,stroke:#9333EA
    style Method2 fill:#FCE7F3,stroke:#DB2777
    style Method3 fill:#FFEDD5,stroke:#EA580C
    style End fill:#D1FAE5,stroke:#059669,stroke-width:2px
```

**代码示例（标题注入）**：

```python
def parse_with_headers(markdown_text):
    lines = markdown_text.split('\n')
    stack = []  # 存储 (level, title)
    result_chunks = []
    current_para = []
    
    for line in lines:
        header = re.match(r'(#{1,6})\s+(.*)', line)
        if header:
            if current_para:
                # 保存上一个段落，并注入标题链
                prefix = ' > '.join([t for l, t in stack])
                chunk = prefix + '\n' + '\n'.join(current_para) if prefix else '\n'.join(current_para)
                result_chunks.append(chunk)
                current_para = []
            # 更新标题栈
            level = len(header.group(1))
            title = header.group(2)
            stack = stack[:level-1] + [(level, title)]
            # 标题本身也可以作为一个轻量 chunk
            result_chunks.append(' > '.join([t for l, t in stack]))
        else:
            if line.strip():
                current_para.append(line)
    if current_para:
        prefix = ' > '.join([t for l, t in stack])
        chunk = prefix + '\n' + '\n'.join(current_para) if prefix else '\n'.join(current_para)
        result_chunks.append(chunk)
    return result_chunks
```

**效果对比**：  

- 不带父标题：检索“损失函数如何计算？” → 命中一个 chunk 只包含公式，不知道这个公式属于“第三章 训练细节”还是“附录”。  
- 带父标题：chunk 头部有 `"第三章 > 3.2 损失函数设计"`，用户立马获得上下文。

## 29. 【⭐⭐】滑动窗口分块和固定分块有什么不同？什么时候用滑动窗口？

| 维度 | 固定分块 | 滑动窗口分块 |
|------|----------|--------------|
| **切分方式** | 非重叠切分，边界固定 | 每次滑动 step 个 token，生成大量重叠块 |
| **块数量** | N / chunk_size | N / step，约 chunk_size/step 倍 |
| **覆盖度** | 每个 token 只属于一个块 | 每个 token 属于多个块 |
| **冗余度** | 低 | 高 |
| **检索效果** | 边界可能割裂语义 | 几乎保证任何连续 n 个 token 都被某个块完整包含 |
| **典型应用** | 大多数 RAG 场景 | 需要高召回率的小规模文档；模型输入窗口极小的场景 |

**滑动窗口公式**：  

- `chunk_size = L`，`step = S`（S < L）。  
- 生成块 i：`text[i*S : i*S + L]`。

**什么时候用滑动窗口？**  

- 文档非常珍贵（如法律合同），不能丢失任意连续片段。  
- query 可能是任意长度的连续原文引用（比如“请解释第 3 页中间那段话”）。  
- 配合去重算法（如 MMR），抵消冗余带来的噪声。

```mermaid
flowchart LR
    subgraph 固定分块
        A1["[0-199]"] --- A2["[200-399]"] --- A3["[400-599]"]
    end
    subgraph 滑动窗口 
        B1["[0-199]"] --- B2["[100-299]"] --- B3["[200-399]"] --- B4["[300-499]"] --- B5["[400-599]"]
    end
    style A1 fill:#BBDEFB
    style A2 fill:#BBDEFB
    style A3 fill:#BBDEFB
    style B1 fill:#C8E6C9
    style B2 fill:#FFF9C4
    style B3 fill:#C8E6C9
    style B4 fill:#FFF9C4
    style B5 fill:#C8E6C9
```

**面试官追问**：“滑动窗口会不会导致 query 命中很多冗余块？”  
答：会。解决方案是检索后做**最大边际相关（MMR）**重排，在相关性和多样性之间取得平衡。

## 30. 【⭐⭐】怎么评估一种分块策略好不好？人工评估还是自动指标？

**评估框架**：**端到端 + 组件级** 结合。

### 自动指标（推荐先做）

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| **Hit Rate@k** | 相关 chunk 是否在 top-k 召回结果中 | 衡量检索覆盖率 |
| **MRR** | 第一个相关 chunk 的排名倒数 | 衡量排序质量 |
| **NDCG@k** | 考虑排序位置和相关性分数 | 排序+相关性综合 |
| **Context Relevancy** | 检索到的 chunk 中，与 ground truth 答案的 token 重叠率 | 防止噪声 |
| **Faithfulness** | LLM 答案是否忠于检索内容 | 最终质量 |

**自动化流程**：

1. 准备一个问答对测试集（文档 D，问题 Q，标准答案 A）。
2. 用不同分块策略跑一遍 RAG，得到 LLM 答案。
3. 用 `RAGAS`、`ARES` 等框架自动打分。

### 人工评估（最终验收）

- **细粒度人工标注**：随机抽取 50 个 query，人工判断召回块是否包含答案（3 档：完全包含、部分包含、不包含）。
- **A/B 对比**：同时展示两种分块策略的 RAG 输出，让专家盲选哪个更好。

### 实践经验

> 我曾在项目中对比过三种策略：固定长度（512, overlap 64）、递归分割、语义分块。自动指标显示语义分块的 Hit Rate@5 最高（0.87），但耗时增加 3 倍。最终线上采用递归分割（0.83 hit rate），因为性价比最优。  
> **关键结论**：没有“最好”的分块，只有“最适合你的数据和查询分布”的分块。一定要用真实 query 做离线评估。

**评估流程图**：

```mermaid
flowchart TD
    A[原始文档] --> B[分块策略 A/B/C]
    B --> C[构建向量索引]
    C --> D[测试集 Queries]
    D --> E[检索 top-k chunks]
    E --> F[自动指标计算<br/>Hit Rate, MRR, NDCG]
    F --> G{分数达标？}
    G -->|否| H[调整分块参数]
    H --> B
    G -->|是| I[人工抽样验证]
    I --> J[最终选择策略]
    
    style F fill:#C8E6C9
    style I fill:#FFF3E0
```

## 🔚 总结

回到最初那个问题：为什么说“RAG 70% 的效果由分块决定”？

因为 **分块是检索的最后一公里**。块太大，噪声淹没了信号；块太小，关键信息被腰斩；块切歪了，标题和正文失联，表格和文字分家，代码里的 docstring 和函数体天各一方——再强的 LLM 也救不回来。

这 15 道题，每一道都对应一个真实的线上事故：

- chunk_overlap 设 50% 导致存储翻倍、检索结果大量重复；
- PDF 双栏排版没做重排，直接按物理坐标切出“上下句颠倒”的垃圾块；
- 代码按固定长度切，把 `def train()` 和它的 `return loss` 分到了两个时代……

**没有银弹，只有验证**。希望你带走的不只是答案，更是一套评估分块的思路：离线跑 Hit Rate，在线看 A/B 实验，永远用真实 query 说话。

如果这篇面经帮你拿到了 offer，或者帮你修好了那个烂了三个月的 RAG 召回——**欢迎转发给你的战友**，也欢迎在评论区留下你的“分块翻车”故事。

下一篇，我们聊 **Embedding 模型选型与微调（31-45题）** ——从 BGE 到混合检索，再到微调时如何避免“灾难性遗忘”。点个 **在看**，更新不迷路。

**#大模型面试 #RAG #分块策略 #AI面经 #LLM**