# 大模型—— 上下文工程 Context Engineering 面试参考题篇 :fire:

## [上下文工程 Context Engineering：一文读懂重塑大模型智能系统的技术革命](https://articles.zsxq.com/id_cunj1wwoejxe.html)

- 引言
- 一、重新定义 Agent 数据流：Context is All You Need
  - 1.1 Prompt Engineering - the Art of Instructions
    - 1.1.1 什么是 Prompt Engineering？
    - 1.1.2 以提示为中心的方法存在什么局限性？
  - 1.2 Context Engineering 兴起：范式的转移
    - 1.2.1 prompt 告诉模型如何思考，而Context则赋予模型完成工作所需的知识和工具。
  - 1.3 Prompt Engineering vs Context Engineering
- 二、 Context Engineering 的基石：RAG（Retrieval-Augmented Generation）
  - 2.1 Retrieval-Augmented Generation
    - 2.1.1 RAG 解决LLM的核心弱点
    - 2.1.2 介绍一下 RAG工作流
  - 2.2 介绍一下 RAG架构分类
  - 2.3 向量数据库的角色
    - 2.3.1 Context Stack+:一个新兴的abstract layer
    - 2.3.2 选型关键考量因素
- 三、Context 工程化：如何判断和提取哪些内容应该进入上下文？
  - 3.1 从原始数据到相关分块
    - 3.1.1 高级分块策略
    - 3.1.2 通过重排序提升精度
  - 3.2 核心问题 - Lost in the Middle
  - 3.3 优化上下文窗口:压缩与摘要
  - 3.4 智能体系统的上下文管理
    - 3.4.1 从 HITL 到 SITL
    - 3.4.2 智能体上下文管理框架
- 四、超越检索！智能体架构中的数据流与工作流编排
  - 4.1 工作流（Workflow） vs. 智能体（Agent）
  - 4.2 核心架构：预定义数据流的实现
  - 4.3 决策与数据选择机制
    - 4.3.1 ReAct框架
    - 4.3.2 Planning和任务分解
  - 4.4 框架与工具
- 五、Context Engineering 的未来
