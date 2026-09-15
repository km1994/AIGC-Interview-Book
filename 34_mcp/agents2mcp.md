# 8种主流Agent框架如何使用MCP

- [8种主流Agent框架如何使用MCP](#8种主流agent框架如何使用mcp)
  - [前言](#前言)
  - [一、Open AI Agents SDK](#一open-ai-agents-sdk)
    - [1.1 Open AI Agents SDK 框架简介](#11-open-ai-agents-sdk-框架简介)
    - [1.2 Open AI Agents SDK 如何继承 MCP](#12-open-ai-agents-sdk-如何继承-mcp)
    - [1.3 Open AI Agents SDK 特点](#13-open-ai-agents-sdk-特点)
  - [二、LangGraph](#二langgraph)
    - [2.1 LangGraph 框架简介](#21-langgraph-框架简介)
    - [2.2 LangGraph 如何继承 MCP](#22-langgraph-如何继承-mcp)
    - [2.3 LangGraph 特点](#23-langgraph-特点)
  - [三、LlamaIndex](#三llamaindex)
    - [3.1 LlamaIndex 框架简介](#31-llamaindex-框架简介)
    - [3.2 LlamaIndex 如何继承 MCP](#32-llamaindex-如何继承-mcp)
    - [3.3 LlamaIndex 特点](#33-llamaindex-特点)
  - [四、AutoGen 0.4+](#四autogen-04)
    - [4.1 AutoGen 0.4+ 框架简介](#41-autogen-04-框架简介)
    - [4.2 AutoGen 0.4+ 如何继承 MCP](#42-autogen-04-如何继承-mcp)
    - [4.3 AutoGen 0.4+ 特点](#43-autogen-04-特点)
  - [五、Pydantic AI](#五pydantic-ai)
    - [5.1 Pydantic AI 框架简介](#51-pydantic-ai-框架简介)
    - [5.2 Pydantic AI 如何继承 MCP](#52-pydantic-ai-如何继承-mcp)
    - [5.3 Pydantic AI 特点](#53-pydantic-ai-特点)
  - [六、SmolAgents](#六smolagents)
    - [6.1 SmolAgents 框架简介](#61-smolagents-框架简介)
    - [6.2 SmolAgents 如何继承 MCP](#62-smolagents-如何继承-mcp)
    - [6.3 SmolAgents 特点](#63-smolagents-特点)
  - [七、Camel](#七camel)
    - [7.1 Camel 框架简介](#71-camel-框架简介)
    - [7.2 Camel 如何继承 MCP](#72-camel-如何继承-mcp)
    - [7.3 Camel 特点](#73-camel-特点)
  - [八、CrewAI](#八crewai)
    - [8.1 CrewAI 框架简介](#81-crewai-框架简介)
    - [8.2 CrewAI 如何继承 MCP](#82-crewai-如何继承-mcp)
    - [8.3 CrewAI 特点](#83-crewai-特点)
  - [致谢](#致谢)

## 前言

大模型Agents开发框架如今已经百花齐放，层出不穷。本文将盘点8种主流LLM Agents开发框架，并介绍如何在每种框架中集成当下备受关注的MCP Server，让Agents系统更方便的接入外部工具。包括：

- OpenAI Agents SDK
- LangGraph
- LlamaIndex
- AutoGen 0.4+
- Pydantic AI
- SmolAgents
- Camel
- CrewAI

## 一、Open AI Agents SDK

### 1.1 Open AI Agents SDK 框架简介

OpenAI Agents SDK是OpenAI官方推出的轻量级Agent开发框架，旨在方便开发者构建多Agent协作的智能体系统。该SDK源于OpenAI内部实验项目Swarm，并在近期正式推出生产版本。

### 1.2 Open AI Agents SDK 如何继承 MCP

以下代码演示了如何将OpenAI Agent实例连接到一个搜索的MCP Server，并将其中的工具集成Agent中：

```s
import asyncio, os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from agents.mcp import MCPServerStdio

asyncdefmain():
    # 1. 创建MCP Server实例
    search_server = MCPServerStdio(
        params={
            "command": "npx",
            "args": ["-y", "@mcptools/mcp-tavily"],
            "env": {**os.environ}
        }
    )
    await search_server.connect()

    # 2. 创建Agent并集成MCP Server
    agent = Agent(
        name="助手Agent",
        instructions="你是一个具有网页搜索能力的助手，必要时使用搜索工具获取信息。",
        mcp_servers=[search_server], # 将MCP Server列表传入Agent
    )

    # 3. 运行Agent，让其自动决定何时调用搜索工具
    result = await Runner.run(agent, "Llama4.0发布了吗？",run_config=RunConfig(tracing_disabled=True))
    print(result.final_output)

    await search_server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

> 注：在使用远程MCP Server时，Agents SDK提供了自动缓存工具列表的选项（通过设置cache_tools_list=True）。如果需要手动使缓存失效，可以调用MCP Server实例上的invalidate_tools_cache()方法 。

### 1.3 Open AI Agents SDK 特点

**OpenAI Agents SDK的特点是：简单易用、轻量级、专注在最小集功能，并支持转交（Handoffs）、护栏（Guardrails）等很有特点的功能**。

## 二、LangGraph

### 2.1 LangGraph 框架简介

LangGraph来自著名的LangChain，是一个用于构建Agentic Workflow的强大框架，它将任务过程建模为有状态的Graph结构，从而可以实现更复杂和结构化的交互。在该框架内集成MCP Server可以在工作流程的各个阶段更精确地控制何时以及如何调用外部工具，从而实现复杂的Agentic系统。

### 2.2 LangGraph 如何继承 MCP

```s
import asyncio, os
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

# 加载环境变量
load_dotenv()

# 定义大语言模型
model = ChatOpenAI(model="gpt-4o-mini")

# 定义并运行agent
asyncdefrun_agent():
    # 定义MCP服务器，用于访问Tavily搜索工具
    asyncwith MultiServerMCPClient(
        {
            "tavily": {
            "command": "npx",
            "args": ["-y", "@mcptools/mcp-tavily"],
            "env": {**os.environ} # 传递环境变量给MCP工具
            }
        }
    ) as client:

        # 创建ReAct风格的agent
        agent = create_react_agent(model, client.get_tools())

        # 定义系统消息，指导如何使用工具
        system_message = SystemMessage(content=(
                "你是一个具有网页搜索能力的助手，必要时使用搜索工具获取信息。"
        ))

        # 处理查询
        agent_response = await agent.ainvoke({"messages": [system_message, HumanMessage(content="Llama4.0发布了吗？")]})

        # 返回agent的回答
        return agent_response["messages"][-1].content

# 运行agent
if __name__ == "__main__":
    response = asyncio.run(run_agent())
    print("\n最终回答:", response)
```

> 注：这里使用MultiServerMCPClient可以灵活的支持多个MCP Server的同时连接，对于单个Server场景，你也可以借助load_mcp_tools方法直接从MCP SDK的session中导入Tools（无需MultiServerMCPClient）。

### 2.3 LangGraph 特点

LangGraph的特点是**功能强大，你可以使用Prebuilt的接口快速创建Agent，也可以使用Graph定义复杂的Agentic工作流与多Agent系统；缺点是略显复杂**。

## 三、LlamaIndex

### 3.1 LlamaIndex 框架简介

LlamaIndex最初是一个专注于构建基于外部数据的LLM应用程序的框架，其独特之处在于构建以数据为中心的LLM应用的能力，特别是复杂的企业级RAG应用。但随着LlamaIndex Workflows与AgentWorkflow功能的推出，LlamaIndex也发展为一个更全能的专注于企业级RAG+Agent系统的开发框架。

### 3.2 LlamaIndex 如何继承 MCP

LlamaIndex目前也支持与MCP Server集成，快速导入Tools使用：

```s
from llama_index.tools.mcp import McpToolSpec,BasicMCPClient
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
import os

llm = OpenAI(model="gpt-4o-mini")

asyncdefmain():

    mcp_client = BasicMCPClient("npx", ["-y", "@mcptools/mcp-tavily"], env={**os.environ})
    mcp_tool = McpToolSpec(client=mcp_client)
    tools = await mcp_tool.to_tool_list_async()

    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        system_prompt="你是一个具有网页搜索能力的助手，必要时使用搜索工具获取信息。"
    )

    response = await agent.aquery("Llama4.0发布了吗？")
    print(response) 

if __name__ == "__main__":
    asyncio.run(main())
```

### 3.3 LlamaIndex 特点

特点是**功能强大、预置大量RAG应用优化模块；事件驱动的Workflows在Agent开发上比LangGraph更简单**。

## 四、AutoGen 0.4+

### 4.1 AutoGen 0.4+ 框架简介

AutoGen是微软开发的一个框架，用于构建具有多Agent对话的下一代企业级AI应用。其独特之处在于专注于通过多个Agent之间的协调交互来实现协作和解决复杂任务，在最新的AutoGen0.4中，微软进行了颠覆性的架构修改，特别是开放了AutoGen-Core这一更底层的API层，可用于构建更底层与细粒度控制的分布式多Agent系统。

### 4.2 AutoGen 0.4+ 如何继承 MCP

在Autogen 0.4的扩展中提供了MCP集成的组件，演示如下（代码有省略）：

```s
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
...

async def get_mcp_tools():
    server_params = StdioServerParams(
        command="npx", 
        args = [
        "-y",
        "@mcptools/mcp-tavily",
      ],env={**os.environ}
    )
    tools = await mcp_server_tools(server_params)
    return tools
...

classToolUseAgent(RoutedAgent):
...

async defmain():
    """主函数，设置并运行agent系统"""
    # 创建单线程agent运行时
    runtime = SingleThreadedAgentRuntime()

    mcp_tools = await get_mcp_tools()
    tools = [*mcp_tools]

    # 注册agent类型
    await ToolUseAgent.register(runtime, "my_agent", lambda: ToolUseAgent(tools))
...

    message = Message('Llama4.0发布了吗？)
    response = await runtime.send_message(message, AgentId("my_agent", "default"))
```

> 注：如果需连接远程MCP Server，请使用SseServerParams组件，并使用url参数初始化。

### 4.3 AutoGen 0.4+ 特点

- 特点：功能强大，支持分布式多Agent，可根据需要选择不同层次的API使用；
- 缺点是较复杂。

## 五、Pydantic AI

### 5.1 Pydantic AI 框架简介

Pydantic AI来自于著名的Pydantic库开发者，是一个将Pydantic与LLM集成的Agents开发框架。其独特之处在于专注于在AI应用中利用Pydantic的类型验证、序列化与结构化输出等功能。

### 5.2 Pydantic AI 如何继承 MCP

```s
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
import os

server = MCPServerStdio( 
    'npx',
    ["-y", "@mcptools/mcp-tavily"],
    env={**os.environ} 
)

agent = Agent(
        name="助手Agent",
        system_prompt="你是一个具有网页搜索能力的助手，必要时使用搜索工具获取信息。",
        model='openai:gpt-4o-mini', 
        mcp_servers=[server])

async def main():
    asyncwith agent.run_mcp_servers():
        result = await agent.run('"Llama4.0发布了吗?')
    print(result.data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

> 注：如果需要使用SSE远程MCP，将Server组件更改为MCPServerHTTP即可。

### 5.3 Pydantic AI 特点

Pydantic AI的特点是**天然的结构化输出与强类型验证，且简洁易用，与其他框架也有良好的集成，可以结合使用**。

## 六、SmolAgents

### 6.1 SmolAgents 框架简介

Smloagents是大名鼎鼎的Hugging Face开发的一个轻量级Agent开发框架。

### 6.2 SmolAgents 如何继承 MCP

以下代码演示了如何初始化一个Smloagent并将其连接到MCP Server：

```s
from smolagents import ToolCollection, CodeAgent
from smolagents.agents import ToolCallingAgent
from smolagents import tool, LiteLLMModel
from mcp import StdioServerParameters
import os

model = LiteLLMModel(model_id="gpt-4o-mini")

server_parameters = StdioServerParameters(
    command="npx",
    args=["-y", "@mcptools/mcp-tavily"],
    env={**os.environ},
)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = ToolCallingAgent(tools=[*tool_collection.tools], model=model)
    response = agent.run("llama4.0发布了吗？")
    print(response)
```

### 6.3 SmolAgents 特点

特点在于**简洁易用、基于生成代码的工具调用（核心抽象叫CodeAgent）以及与Hugging Face生态系统的集成**。

Smloagents与MCP的集成提供了一种直接的方式，可以为Agent添加复杂的功能，而无需为每个工具进行自定义编码。

> 注: 如果你需要使用SSE模式的MCP Server，只需要替换服务器配置参数为url即可。

## 七、Camel

### 7.1 Camel 框架简介

Camel是一个专注于创建能够进行复杂对话以解决任务的强大的多智能体构建框架 。

### 7.2 Camel 如何继承 MCP

参考如下方式将基于Camel的Agent与MCP Server做集成：

```s
import asyncio
from mcp.types import CallToolResult
from camel.toolkits.mcp_toolkit import MCPToolkit, MCPClient
import os
from camel.agents import ChatAgent

async def run_example():
    
    mcp_client = MCPClient(
        command_or_url="npx",
        args=["-y", "@mcptools/mcp-tavily"],
        env={**os.environ}
    )
    await mcp_client.connect()
    mcp_toolkit = MCPToolkit(servers=[mcp_client])
    tools = mcp_toolkit.get_tools()

    try:
        agent = ChatAgent(system_message='根据任务描述，使用网页搜索工具获取信息。',
                          tools=tools)
        response = await agent.astep("llama4.0发布了吗？")
        print("Response:", response.msgs[0].content)
    except Exception as e:
        print(f"Error during agent execution: {e}")
    finally:
        # 确保在任何情况下都会断开连接
        await mcp_client.disconnect()

if __name__ == "__main__":
    asyncio.run(run_example())
```

> 注：如果需要连接SSE的远程Server，替换这里的MCPClient中的输入参数为url即可。

### 7.3 Camel 特点

其独特之处**在于使用AI Agent之间的角色扮演和交互协作来完成任务，并内置了多种角色的Agent抽象及大量组件，Camel也可以用来开发RAG应用**。现在这些Agent也可以通过MCP Server得到增强。

## 八、CrewAI

### 8.1 CrewAI 框架简介

CrewAI是一个用于编排自主AI智能体像团队一样协作完成复杂任务的多智能系统开发框架。

### 8.2 CrewAI 如何继承 MCP

```s
import os
from crewai import Agent, Crew, Task # type: ignore 
from mcp import StdioServerParameters
from mcpadapt.core import MCPAdapt
from mcpadapt.crewai_adapter import CrewAIAdapter

with MCPAdapt(
    StdioServerParameters(
        command="npx",
        args=["-y", "@mcptools/mcp-tavily"],
        env={**os.environ}
    ),
    CrewAIAdapter(),
) as tools:
    print(f"Tools: {tools}")
    agent = Agent(
        role="MyAgent",goal="根据任务描述，使用网页搜索工具获取信息。",backstory="你是一个中文搜索助手",
        tools=tools,llm='gpt-4o-mini',
    )

    # Create a task
    task = Task(
        description="llama4.0的最新消息",agent=agent,expected_output="消息列表")

    task.execute_sync()
```

### 8.3 CrewAI 特点

其独特之处在于**其“角色扮演”的设计，专注于创建具有特定角色和职责的结构化Agent团队（称为Crew）;最新的Flow功能可用于创建更可靠的Agentic Workflow**。

## 致谢

- 一文全览：8种主流Agent框架与MCP的集成  https://mp.weixin.qq.com/s/WGOnLFLAZhdvxu1qUs8Aww



