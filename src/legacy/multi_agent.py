from typing import List, Annotated, TypedDict, Literal, cast
from pydantic import BaseModel, Field
import operator
import warnings

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import MessagesState

from langgraph.types import Command, Send
from langgraph.graph import START, END, StateGraph

from legacy.configuration import MultiAgentConfiguration
from legacy.utils import (
    get_config_value,
    tavily_search,
    duckduckgo_search,
    get_today_str,
)

from legacy.prompts import SUPERVISOR_INSTRUCTIONS, RESEARCH_INSTRUCTIONS

## 工具工厂 - 将根据配置进行初始化
def get_search_tool(config: RunnableConfig):
    """根据配置获取相应的搜索工具"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)

    # 如果未请求任何搜索工具,则返回 None
    if search_api.lower() == "none":
        return None

    # TODO: 将其他搜索函数配置为工具
    if search_api.lower() == "tavily":
        search_tool = tavily_search
    elif search_api.lower() == "duckduckgo":
        search_tool = duckduckgo_search
    else:
        raise NotImplementedError(
            f"多智能体实现尚未支持搜索 API '{search_api}'。"
            f"目前仅支持 Tavily/DuckDuckGo/None。其他搜索 API 请使用 "
            f"src/open_deep_research/graph.py 中基于图的实现,或将 search_api 设置为 'tavily'、'duckduckgo' 或 'none'。"
        )

    tool_metadata = {**(search_tool.metadata or {}), "type": "search"}
    search_tool.metadata = tool_metadata
    return search_tool

class Section(BaseModel):
    """报告的一个章节。"""
    name: str = Field(
        description="报告中该章节的名称。",
    )
    description: str = Field(
        description="报告中该章节的研究范围。",
    )
    content: str = Field(
        description="章节的内容。"
    )

class Sections(BaseModel):
    """报告的章节标题列表。"""
    sections: List[str] = Field(
        description="报告的各个章节。",
    )

class Introduction(BaseModel):
    """报告的引言。"""
    name: str = Field(
        description="报告的名称。",
    )
    content: str = Field(
        description="引言的内容,对整份报告作概览性介绍。"
    )

class Conclusion(BaseModel):
    """报告的结论。"""
    name: str = Field(
        description="报告结论的名称。",
    )
    content: str = Field(
        description="结论的内容,对整份报告进行总结。"
    )

class Question(BaseModel):
    """提出后续问题以明确报告的范围。"""
    question: str = Field(
        description="向用户提出的一个具体问题,用于明确报告的范围、侧重点或要求。"
    )

# 空操作工具,用于指示研究已完成
class FinishResearch(BaseModel):
    """完成研究。"""

# 空操作工具,用于指示报告撰写已完成
class FinishReport(BaseModel):
    """完成报告。"""

## 状态
class ReportStateOutput(MessagesState):
    final_report: str # 最终报告
    # 仅用于评估
    # 仅当 configurable.include_source_str 为 True 时才包含此字段
    source_str: str # 来自网页搜索的格式化来源内容字符串

class ReportState(MessagesState):
    sections: list[str] # 报告章节列表 
    completed_sections: Annotated[list[Section], operator.add] # Send() API 所需的键
    final_report: str # 最终报告
    # 仅用于评估
    # 仅当 configurable.include_source_str 为 True 时才包含此字段
    source_str: Annotated[str, operator.add] # 来自网页搜索的格式化来源内容字符串

class SectionState(MessagesState):
    section: str # 报告章节  
    completed_sections: list[Section] # 为 Send() API 在外部状态中复制的最终键
    # 仅用于评估
    # 仅当 configurable.include_source_str 为 True 时才包含此字段
    source_str: str # 来自网页搜索的格式化来源内容字符串

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # 为 Send() API 在外部状态中复制的最终键
    # 仅用于评估
    # 仅当 configurable.include_source_str 为 True 时才包含此字段
    source_str: str # 来自网页搜索的格式化来源内容字符串


async def _load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    if not configurable.mcp_server_config:
        return []

    mcp_server_config = configurable.mcp_server_config
    client = MultiServerMCPClient(mcp_server_config)
    mcp_tools = await client.get_tools()
    filtered_mcp_tools: list[BaseTool] = []
    for tool in mcp_tools:
        # TODO:这种方式可能难以管理
        # 尤其是在不受开发者控制的远程服务器上
        # 最佳方案是让 MultiServerMCPClient 支持工具名前缀
        if tool.name in existing_tool_names:
            warnings.warn(
                f"尝试添加名为 {tool.name} 的 MCP 工具,但该名称已被占用 - 此工具将被忽略。"
            )
            continue

        if configurable.mcp_tools_to_include and tool.name not in configurable.mcp_tools_to_include:
            continue

        filtered_mcp_tools.append(tool)

    return filtered_mcp_tools


# 工具列表将根据配置动态构建
async def get_supervisor_tools(config: RunnableConfig) -> list[BaseTool]:
    """根据配置获取监督者工具"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    search_tool = get_search_tool(config)
    tools = [tool(Sections), tool(Introduction), tool(Conclusion), tool(FinishReport)]
    if configurable.ask_for_clarification:
        tools.append(tool(Question))
    if search_tool is not None:
        tools.append(search_tool)  # 若有可用的搜索工具则添加
    existing_tool_names = {cast(BaseTool, tool).name for tool in tools}
    mcp_tools = await _load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools


async def get_research_tools(config: RunnableConfig) -> list[BaseTool]:
    """根据配置获取研究工具"""
    search_tool = get_search_tool(config)
    tools = [tool(Section), tool(FinishResearch)]
    if search_tool is not None:
        tools.append(search_tool)  # 若有可用的搜索工具则添加
    existing_tool_names = {cast(BaseTool, tool).name for tool in tools}
    mcp_tools = await _load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools


async def supervisor(state: ReportState, config: RunnableConfig):
    """由 LLM 决定是否调用工具"""

    # 消息
    messages = state["messages"]

    # 获取配置
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    supervisor_model = get_config_value(configurable.supervisor_model)

    # 初始化模型
    llm = init_chat_model(model=supervisor_model)
    
    # 如果各章节已完成但还没有最终报告,则需要开始撰写引言和结论
    if state.get("completed_sections") and not state.get("final_report"):
        research_complete_message = {"role": "user", "content": "研究已完成。现在请为报告撰写引言和结论。以下是已完成的主要正文章节:\n\n" + "\n\n".join([s.content for s in state["completed_sections"]])}
        messages = messages + [research_complete_message]

    # 根据配置获取工具
    supervisor_tool_list = await get_supervisor_tools(config)
    
    
    llm_with_tools = (
        llm
        .bind_tools(
            supervisor_tool_list,
            parallel_tool_calls=False,
            # 强制至少调用一次工具
            tool_choice="any"
        )
    )

    # 获取系统提示词
    system_prompt = SUPERVISOR_INSTRUCTIONS.format(today=get_today_str())
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # 调用
    return {
        "messages": [
            await llm_with_tools.ainvoke(
                [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
                + messages
            )
        ]
    }

async def supervisor_tools(state: ReportState, config: RunnableConfig)  -> Command[Literal["supervisor", "research_team", "__end__"]]:
    """执行工具调用并发送给研究智能体"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None
    source_str = ""

    # 根据配置获取工具
    supervisor_tool_list = await get_supervisor_tools(config)
    supervisor_tools_by_name = {tool.name: tool for tool in supervisor_tool_list}
    search_tool_names = {
        tool.name
        for tool in supervisor_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }

    # 首先处理所有工具调用,确保对每一个都做出响应(OpenAI 要求)
    for tool_call in state["messages"][-1].tool_calls:
        # 获取工具
        tool = supervisor_tools_by_name[tool_call["name"]]
        # 执行工具调用 - 异步工具使用 ainvoke
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)

        # 追加到消息 
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_call["name"], 
                       "tool_call_id": tool_call["id"]})
        
        # 暂存特殊工具结果,待所有工具调用完毕后再处理
        if tool_call["name"] == "Question":
            # 调用了 Question 工具 - 返回监督者以向用户提问
            question_obj = cast(Question, observation)
            result.append({"role": "assistant", "content": question_obj.question})
            return Command(goto=END, update={"messages": result})
        elif tool_call["name"] == "FinishReport":
            result.append({"role": "user", "content": "报告已完成"})
            return Command(goto=END, update={"messages": result})
        elif tool_call["name"] == "Sections":
            sections_list = cast(Sections, observation).sections
        elif tool_call["name"] == "Introduction":
            # 若引言尚未格式化,则为其加上规范的 H1 标题
            observation = cast(Introduction, observation)
            if not observation.content.startswith("# "):
                intro_content = f"# {observation.name}\n\n{observation.content}"
            else:
                intro_content = observation.content
        elif tool_call["name"] == "Conclusion":
            # 若结论尚未格式化,则为其加上规范的 H2 标题
            observation = cast(Conclusion, observation)
            if not observation.content.startswith("## "):
                conclusion_content = f"## {observation.name}\n\n{observation.content}"
            else:
                conclusion_content = observation.content
        elif tool_call["name"] in search_tool_names and configurable.include_source_str:
            source_str += cast(str, observation)

    # 处理完所有工具调用后,决定下一步做什么
    if sections_list:
        # 将各章节发送给研究智能体
        return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], update={"messages": result})
    elif intro_content:
        # 暂存引言,等待结论
        # 向消息追加内容,引导 LLM 接下来撰写结论
        result.append({"role": "user", "content": "引言已写好。现在请撰写结论章节。"})
        state_update = {
            "final_report": intro_content,
            "messages": result,
        }
    elif conclusion_content:
        # 获取所有章节并按正确顺序组合:引言、正文章节、结论
        intro = state.get("final_report", "")
        body_sections = "\n\n".join([s.content for s in state["completed_sections"]])
        
        # 按正确顺序组装最终报告
        complete_report = f"{intro}\n\n{body_sections}\n\n{conclusion_content}"
        
        # 向消息追加内容以表示已完成
        result.append({"role": "user", "content": "报告现已完成,包含引言、正文章节和结论。"})

        state_update = {
            "final_report": complete_report,
            "messages": result,
        }
    else:
        # 默认情况(如搜索工具等)
        state_update = {"messages": result}

    # 包含用于评估的来源字符串
    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str

    return Command(goto="supervisor", update=state_update)

async def supervisor_should_continue(state: ReportState) -> str:
    """根据 LLM 是否发起了工具调用来决定继续循环还是停止"""

    messages = state["messages"]
    last_message = messages[-1]
    # 监督者已提问或已完成,结束流程
    if not last_message.tool_calls:
        # 退出图
        return END

    # 如果 LLM 发起了工具调用,则执行相应操作
    return "supervisor_tools"

async def research_agent(state: SectionState, config: RunnableConfig):
    """由 LLM 决定是否调用工具"""
    
    # 获取配置
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    researcher_model = get_config_value(configurable.researcher_model)
    
    # 初始化模型
    llm = init_chat_model(model=researcher_model)

    # 根据配置获取工具
    research_tool_list = await get_research_tools(config)
    system_prompt = RESEARCH_INSTRUCTIONS.format(
        section_description=state["section"],
        number_of_queries=configurable.number_of_queries,
        today=get_today_str(),
    )
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # 确保至少有一条用户消息(Anthropic 要求)
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": f"请研究并撰写以下章节:{state['section']}"}]

    return {
        "messages": [
            # 强制调用工具:要么继续搜索,要么调用 Section 工具撰写章节
            await llm.bind_tools(research_tool_list,             
                                 parallel_tool_calls=False,
                                 # 强制至少调用一次工具
                                 tool_choice="any").ainvoke(
                [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
                + messages
            )
        ]
    }

async def research_agent_tools(state: SectionState, config: RunnableConfig):
    """执行工具调用,并路由到监督者或继续研究循环"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result = []
    completed_section = None
    source_str = ""
    
    # 根据配置获取工具
    research_tool_list = await get_research_tools(config)
    research_tools_by_name = {tool.name: tool for tool in research_tool_list}
    search_tool_names = {
        tool.name
        for tool in research_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }
    
    # 先处理所有工具调用(OpenAI 要求)
    for tool_call in state["messages"][-1].tool_calls:
        # 获取工具
        tool = research_tools_by_name[tool_call["name"]]
        # 执行工具调用 - 异步工具使用 ainvoke
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)

        # 追加到消息 
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_call["name"], 
                       "tool_call_id": tool_call["id"]})
        
        # 若调用了 Section 工具,则保存该章节结果
        if tool_call["name"] == "Section":
            completed_section = cast(Section, observation)

        # 若调用了搜索工具,则保存来源字符串
        if tool_call["name"] in search_tool_names and configurable.include_source_str:
            source_str += cast(str, observation)
    
    # 处理完所有工具后,决定下一步做什么
    state_update = {"messages": result}
    if completed_section:
        # 将已完成的章节写入状态并返回监督者
        state_update["completed_sections"] = [completed_section]
    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str

    return state_update

async def research_agent_should_continue(state: SectionState) -> str:
    """根据 LLM 是否发起了工具调用来决定继续循环还是停止"""

    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls[0]["name"] == "FinishResearch":
        # 研究完成 - 返回监督者
        return END
    else:
        return "research_agent_tools"
    
"""构建多智能体工作流"""

# 研究智能体工作流
research_builder = StateGraph(SectionState, output=SectionOutputState, config_schema=MultiAgentConfiguration)
research_builder.add_node("research_agent", research_agent)
research_builder.add_node("research_agent_tools", research_agent_tools)
research_builder.add_edge(START, "research_agent") 
research_builder.add_conditional_edges(
    "research_agent",
    research_agent_should_continue,
    ["research_agent_tools", END]
)
research_builder.add_edge("research_agent_tools", "research_agent")

# 监督者工作流
supervisor_builder = StateGraph(ReportState, input=MessagesState, output=ReportStateOutput, config_schema=MultiAgentConfiguration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("research_team", research_builder.compile())

# 监督者智能体的流程
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_conditional_edges(
    "supervisor",
    supervisor_should_continue,
    ["supervisor_tools", END]
)
supervisor_builder.add_edge("research_team", "supervisor")

graph = supervisor_builder.compile()