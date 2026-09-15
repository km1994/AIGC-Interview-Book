# Agent 面试参考题篇

## 常考题

- [Agent 的规划模块是怎么实现的？](https://articles.zsxq.com/id_vbipqdhwsgkn.html)
- [🤔 面试官问：Agent 的函数调用怎么做到又准又稳？](https://articles.zsxq.com/id_a7w6ihawpm53.html)
- [Agent 的记忆模块是怎么实现的？](https://articles.zsxq.com/id_w69z7w6idhpr.html)
- [🤔 面试官问：Function Call 的训练数据到底该怎么构建？](https://articles.zsxq.com/id_qwg8w0cm2wkn.html)
-[ 别再瞎调了！Agent Function Call 的稳定之道，揭秘大厂都在用的5层优化法](https://articles.zsxq.com/id_thevg31zu5rq.html)


## 大模型（LLMs）agent 基础面 :fire:

### [大模型（LLMs）agent 面](https://articles.zsxq.com/id_le02luntesap.html) 

- 一、什么是 大模型（LLMs）agent？
- 二、大模型（LLMs）agent 有哪些部分组成？
  - 2.1 介绍一下 规划（planning）？
    - 2.1.1 拆解子目标和任务分解
      - 2.1.1.1 如何进行 拆解子目标和任务分解？
      - 2.1.1.2 拆解子目标和任务分解 有哪些方法？
    - 2.1.2 模型自我反省
      - 2.1.2.1 如何进行 模型自我反省？
      - 2.1.2.2 模型自我反省 有哪些方法？
  - 2.2 介绍一下 记忆（Memory）？
  - 2.3 介绍一下 工具使用（tool use）？
- 三、大模型（LLMs）agent 主要 利用了 大模型 哪些能力？
- 四、结合 代码 讲解 大模型（LLMs）agent 思路？
- 4.1 思路介绍
- 4.2 实例一：利用大模型判断做选择
- 4.3 实例二：让大模型通过判断正确选择函数工具并输出
- 4.4 实例三：agent模板和解析
- 4.5 实例四：将 skylark 接入 langchain 中测试 agent
- 五、如何给LLM注入领域知识？
- 六、常见LLM Agent框架或者应用 有哪些？

- [点击查看答案](https://articles.zsxq.com/id_le02luntesap.html)

## 大模型（LLMs）agent 进阶技巧面 :fire:

### [多轮对话中让AI保持长期记忆的8种优化方式篇](https://articles.zsxq.com/id_3qgicwcwzjpi.html) 

- 一、前言
- 二、Agent 如何获取上下文对话信息？
  - 2.1 获取全量历史对话
  - 2.2 滑动窗口获取最近部分对话内容
  - 2.3 获取历史对话中实体信息
  - 2.4 利用知识图谱获取历史对话中的实体及其联系
  - 2.5 对历史对话进行阶段性总结摘要
  - 2.6 需要获取最新对话，又要兼顾较早历史对话
  - 2.7 回溯最近和最关键的对话信息
  - 2.8 基于向量检索对话信息

- [点击查看答案](https://articles.zsxq.com/id_3qgicwcwzjpi.html)

### [AI agent 性能思考](https://articles.zsxq.com/id_nhbis9t6ips6.html) 

- 前言
- 一、Planning慢问题思考与反思
  - 1.1 Planning慢的本质原因是什么？
  - 1.2 如何解决 Planning 慢问题？

- [点击查看答案](https://articles.zsxq.com/id_nhbis9t6ips6.html)




## Agent 记忆（Memory）模块面

### [AI agent 长期记忆：向量数据库存储历史](https://articles.zsxq.com/id_pxej86wfjcpa.html)
  
- 前言
- 一、长期记忆的核心设计要素
- 二、具体设计方案（以 Milvus 为例）
  - 2.1 经验数据结构定义
  - 2.2 向量数据库集合（Collection）设计
  - 2.3 经验存储：将历史经验写入向量数据库
  - 2.4 经验检索：根据当前任务查询相似经验
  - 2.5 记忆演化：更新与维护长期记忆
- 三、设计要点与技术选型建议
- 总结

- [点击查看答案](https://articles.zsxq.com/id_pxej86wfjcpa.html)

## Agent 路由模块面

### [Agent 路由模块的4种设计模式](https://articles.zsxq.com/id_3jaimwvvr8jt.html)

- 引言
- 路由模块的作用
- 路由的常见实现模式有哪些？
  - 模式一：基于规则的路由
  - 模式二：基于小模型的路由
  - 模式三：基于大模型LLM的路由
  - 模式四：基于嵌入Embedding的路由
- 四种模式的比较

- [点击查看答案](https://articles.zsxq.com/id_3jaimwvvr8jt.html)

### [Agent中意图识别怎么做?](https://articles.zsxq.com/id_vpdi5jfjghvl.html)

- 一、什么是意图识别？
- 二、意图识别有什么作用？
- 三、意图识别难点是什么？
- 四、意图识别难点有哪些解决方案？
- 五、单轮意图
  - 5.1. 基于规则的意图识别
  - 5.2. 向量召回意图识别
  - 5.3. 深度学习意图识别
  - 5.4. 大模型意图识别
  - 5.5. 融合方案
- 六、多轮意图
  - 6.1 Pipeline
  - 6.2 大模型End-2-End方案
  - 6.3. 多级意图

- [点击查看答案](https://articles.zsxq.com/id_vpdi5jfjghvl.html)

## 大模型（LLMs）多智能体 进阶技巧面 :fire:

### [单智能体的痛点，是多智能体的起点：详解多智能体的四种架构与实践](https://articles.zsxq.com/id_5ks4r3raurpp.html)

- 前言
- 一、为什么单Agent总卡壳？多智能体如何救场？
  - 1.1 Agent到底算啥？定义还在打架
  - 1.2 单Agent的三大“痛点”，你中招了吗？
  - 1.3 团队协作：像公司分工一样解锁效率
- 二、多智能体系统的“入门课”
  - 2.1 啥是多智能体系统？简单说，就是拆分+协作
  - 2.2 LangGraph里的四种玩法，选一个
- 三、Subgraphs实战——让Agent“对话”起来
  - 3.1 Subgraphs是啥？子图当节点用
  - 3.2 案例：建个评分系统（父子共享状态键）
    - 3.2.1 环境搭好
    - 3.2.2 模型选一个（云or本地）
    - 3.2.3 父图状态
    - 3.2.4 父图节点
    - 3.2.5 子图状态
    - 3.2.6 子图节点
    - 3.2.7 组子图
    - 3.2.8 组父图（嵌子图）
    - 3.2.9 可视化
    - 3.2.10 跑测试
  - 3.3 无共享键？加个“翻译官”
- 四、Network架构实战——BI数据分析系统
  - 4.1 整体设计：用户意图 → DB操作 → 代码生成 → 可视化
  - 4.2 环境+数据库
  - 4.3 建表
  - 4.4 造测试数据
  - 4.5 工具箱：DB CRUD + 报表
- 五、组队！Network多Agent实现
  - 5.1 每个Agent一个子图
  - 5.2 父图Orchestrator：串联Network
  - 5.3 跑起来
- 六、测试调试，别让小Bug毁了大局
  - 6.1 单元测试建议
  - 6.2 本地调试技巧
  - 6.3 常见错误及解决
- 七、上线前 checklist——生产化别马虎
- 八、扩展与演进（后面持续分享）
- 九、如何避免智能体提桶跑路
- 十、部署前10问自查
- 十一、总结

- [点击查看答案](https://articles.zsxq.com/id_5ks4r3raurpp.html)

### [Multi-Agent效果天差地别？Google发现关键在Prompt设计](https://articles.zsxq.com/id_co6w33c5l4wk.html)

- 一、前言
- 二、Mass框架
- 三、Mass 创新点
- 四、Mass 实验结果
- 五、Mass 意义与贡献
- 六、总结

- [点击查看答案](https://articles.zsxq.com/id_co6w33c5l4wk.html)

## Agent 框架对比面 :fire:

### [17个主流 Agent 框架快速对比](https://articles.zsxq.com/id_uhvhdsubhc8s.html)

- 引言
- 一、先给选型结论
- 二、核心理念与技术风格（通俗版）
  - 2.1 “编排/状态机”派：把 Agent 当流程来建
  - 2.2 “多智能体对话/团队”派：把 Agent 当角色来搭
  - 2.3 “极简/实验”派：把 Agent 当可插积木
  - 2.4 “RAG+Agent 合一”
  - 2.5 “持续自主”与平台
- 三、逐一点评（含定位、成熟度、关注度）
  - 3.1 LangGraph（LangChain 团队）
  - 3.2 AutoGen（Microsoft）
  - 3.3 PydanticAI（Pydantic 官方）
  - 3.4 Agno
  - 3.5 CrewAI
  - 3.6 CAMEL（社区/研究）
  - 3.7 Langroid
  - 3.8 LlamaIndex Agents
  - 3.9 Haystack Agents（deepset）
  - 3.10 Semantic Kernel Agent Framework（Microsoft）
  - 3.11 smolagents（Hugging Face）
  - 3.12 OpenAI Swarm / DurableSwarm
  - 3.13 AutoGPT / SuperAGI / AGiXT（持续自主 & 可视化）
  - 3.14 MetaGPT / AgentVerse（仿真/多角色 SOP）
- 四、怎么选：结合你的常见需求
- 五、注意的坑与趋势
- 六、“落地组合”建议（面向产研一体）

- [点击查看答案](https://articles.zsxq.com/id_uhvhdsubhc8s.html)
