"""深度研究智能体的 LangGraph 主实现。"""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_extra_body,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# 初始化一个可在整个智能体中使用的可配置模型
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "extra_body"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """分析用户消息，并在研究范围不明确时提出澄清性问题。
    
    此函数判断用户的请求在开始研究之前是否需要澄清。
    如果澄清功能被禁用或不需要澄清，则直接进入研究。
    
    参数：
        state: 当前智能体状态，包含用户消息
        config: 运行时配置，包含模型设置与偏好
        
    返回：
        Command：以澄清性问题结束，或进入研究简报编写
    """
    # 步骤 1：检查配置中是否启用了澄清功能
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # 跳过澄清步骤，直接进入研究
        return Command(goto="write_research_brief")
    
    # 步骤 2：为结构化澄清分析准备模型
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "extra_body": get_model_extra_body(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # 为模型配置结构化输出与重试逻辑
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # 步骤 3：分析是否需要澄清
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 步骤 4：根据澄清分析结果路由
    if response.need_clarification:
        # 以澄清性问题结束，等待用户回答
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # 带确认消息继续研究
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将用户消息转化为结构化的研究简报，并初始化监督者。
    
    此函数分析用户消息并生成聚焦的研究简报，
    用于引导研究监督者。同时以适当的提示词和指令
    初始化监督者的上下文。
    
    参数：
        state: 当前智能体状态，包含用户消息
        config: 运行时配置，包含模型设置
        
    返回：
        Command：带着初始化好的上下文进入研究监督者
    """
    # 步骤 1：为结构化输出设置研究模型
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "extra_body": get_model_extra_body(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # 为生成结构化研究问题配置模型
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 步骤 2：从用户消息生成结构化研究简报
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 步骤 3：用研究简报和指令初始化监督者
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """研究监督者的主导节点，负责规划研究策略并委派给研究员。
    
    监督者分析研究简报，决定如何将研究拆分为可控的任务。
    它可以使用 think_tool 进行战略规划，使用 ConductResearch
    将任务委派给子研究员，或在满意时使用 ResearchComplete。
    
    参数：
        state: 当前监督者状态，包含消息与研究上下文
        config: 运行时配置，包含模型设置
        
    返回：
        Command：进入 supervisor_tools 执行工具
    """
    # 步骤 1：为监督者模型配置可用工具
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "extra_body": get_model_extra_body(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # 可用工具：研究委派、完成信号、战略思考
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # 为模型配置工具、重试逻辑与模型设置
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 步骤 2：基于当前上下文生成监督者响应
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # 步骤 3：更新状态并进入工具执行
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """执行监督者调用的工具，包括研究委派与战略思考。
    
    此函数处理三类监督者工具调用：
    1. think_tool - 战略性反思，延续对话
    2. ConductResearch - 将研究任务委派给子研究员
    3. ResearchComplete - 标志研究阶段完成
    
    参数：
        state: 当前监督者状态，包含消息与迭代计数
        config: 运行时配置，包含研究限制与模型设置
        
    返回：
        Command：继续监督循环，或结束研究阶段
    """
    # 步骤 1：提取当前状态并检查退出条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # 定义研究阶段的退出条件
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # 满足任一终止条件即退出
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # 步骤 2：一并处理所有工具调用（think_tool 与 ConductResearch）
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # 处理 think_tool 调用（战略性反思）
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"反思已记录：{reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # 处理 ConductResearch 调用（研究委派）
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # 限制并发研究单元数量，防止资源耗尽
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # 并行执行研究任务
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # 用研究结果创建工具消息
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "合成研究报告时出错：重试次数已达上限"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # 以错误消息处理超出上限的研究调用
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"错误：由于你已超出并发研究单元的最大数量，本次研究未执行。请将研究单元数量缩减到 {configurable.max_concurrent_research_units} 个或更少后重试。",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # 从所有研究结果中汇总原始笔记
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # 处理研究执行错误
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # 超出 token 限制或其他错误 - 结束研究阶段
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # 步骤 3：返回包含所有工具结果的 Command
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# 监督者子图构建
# 构建管理研究委派与协调的监督者工作流
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# 添加用于研究管理的监督者节点
supervisor_builder.add_node("supervisor", supervisor)           # 主监督者逻辑
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # 工具执行处理器

# 定义监督者工作流的边
supervisor_builder.add_edge(START, "supervisor")  # 监督者的入口

# 编译监督者子图，供主工作流使用
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """针对特定主题开展聚焦研究的独立研究员。
    
    该研究员由监督者指定一个具体研究主题，并使用可用工具
    （搜索、think_tool、MCP 工具）收集全面的信息。
    它可以在搜索之间使用 think_tool 进行战略规划。
    
    参数：
        state: 当前研究员状态，包含消息与主题上下文
        config: 运行时配置，包含模型设置与可用工具
        
    返回：
        Command：进入 researcher_tools 执行工具
    """
    # 步骤 1：加载配置并校验工具可用性
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # 获取所有可用的研究工具（搜索、MCP、think_tool）
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "未找到可用于开展研究的工具：请配置你的搜索 API，"
            "或在配置中添加 MCP 工具。"
        )
    
    # 步骤 2：为研究员模型配置工具
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "extra_body": get_model_extra_body(configurable.research_model),
        "tags": ["langsmith:nostream"]
    }
    
    # 如有 MCP 上下文则将其并入系统提示词
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # 为模型配置工具、重试逻辑与设置
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # 步骤 3：带系统上下文生成研究员响应
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # 步骤 4：更新状态并进入工具执行
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# 工具执行辅助函数
async def execute_tool_safely(tool, args, config):
    """带错误处理地安全执行工具。"""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"执行工具时出错：{str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """执行研究员调用的工具，包括搜索工具与战略思考。
    
    此函数处理多种类型的研究员工具调用：
    1. think_tool - 战略性反思，延续研究对话
    2. 搜索工具（tavily_search、web_search）- 信息收集
    3. MCP 工具 - 外部工具集成
    4. ResearchComplete - 标志单个研究任务完成
    
    参数：
        state: 当前研究员状态，包含消息与迭代计数
        config: 运行时配置，包含研究限制与工具设置
        
    返回：
        Command：继续研究循环，或进入压缩阶段
    """
    # 步骤 1：提取当前状态并检查提前退出条件
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # 若未发生任何工具调用（包括原生网页搜索）则提前退出
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # 步骤 2：处理其他工具调用（搜索、MCP 工具等）
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # 并行执行所有工具调用
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # 根据执行结果创建工具消息
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # 步骤 3：检查延迟退出条件（在处理工具之后）
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # 结束研究并进入压缩
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # 带着工具结果继续研究循环
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """将研究发现压缩并合成为简洁、结构化的摘要。
    
    此函数提取研究员工作中全部的研究发现、工具输出与 AI 消息，
    并将其提炼为干净、全面的摘要，
    同时保留所有重要信息与发现。
    
    参数：
        state: 当前研究员状态，包含累积的研究消息
        config: 运行时配置，包含压缩模型设置
        
    返回：
        包含压缩后研究摘要与原始笔记的字典
    """
    # 步骤 1：配置压缩模型
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "extra_body": get_model_extra_body(configurable.compression_model),
        "tags": ["langsmith:nostream"]
    })
    
    # 步骤 2：为压缩准备消息
    researcher_messages = state.get("researcher_messages", [])
    
    # 添加指令，从研究模式切换到压缩模式
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # 步骤 3：尝试压缩，并针对 token 超限问题配备重试逻辑
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # 创建聚焦于压缩任务的系统提示词
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # 执行压缩
            response = await synthesizer_model.ainvoke(messages)
            
            # 从所有工具消息与 AI 消息中提取原始笔记
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # 返回压缩成功的结果
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # 通过移除较早的消息来处理 token 超限
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # 其他错误则继续重试
            continue
    
    # 步骤 4：所有尝试均失败时返回错误结果
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "合成研究报告时出错：重试次数已达上限",
        "raw_notes": [raw_notes_content]
    }

# 研究员子图构建
# 构建独立研究员工作流，用于针对特定主题开展聚焦研究
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# 添加用于研究执行与压缩的研究员节点
researcher_builder.add_node("researcher", researcher)                 # 主研究员逻辑
researcher_builder.add_node("researcher_tools", researcher_tools)     # 工具执行处理器
researcher_builder.add_node("compress_research", compress_research)   # 研究压缩

# 定义研究员工作流的边
researcher_builder.add_edge(START, "researcher")           # 研究员的入口
researcher_builder.add_edge("compress_research", END)      # 压缩完成后的出口

# 编译研究员子图，供监督者并行执行
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """生成全面的最终研究报告，并针对 token 超限配备重试逻辑。
    
    此函数提取已收集的全部研究发现，并使用配置的报告生成模型 
    将其合成为结构良好、全面的最终报告。
    
    参数：
        state: 智能体状态，包含研究发现与上下文
        config: 运行时配置，包含模型设置与 API 密钥
        
    返回：
        包含最终报告与已清空状态的字典
    """
    # 步骤 1：提取研究发现并准备状态清理
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # 步骤 2：配置最终报告生成模型
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "extra_body": get_model_extra_body(configurable.final_report_model),
        "tags": ["langsmith:nostream"]
    }
    
    # 步骤 3：尝试生成报告，并针对 token 超限配备重试逻辑
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # 用全部研究上下文创建完整提示词
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # 生成最终报告
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # 返回报告生成成功的结果
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # 通过渐进式截断处理 token 超限错误
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # 第一次重试：确定初始截断上限
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"生成最终报告时出错：已超出 token 限制，且无法确定该模型的最大上下文长度。请在 deep_researcher/utils.py 的模型映射表中补充此信息。{e}",
                            "messages": [AIMessage(content="因 token 限制，报告生成失败")],
                            **cleared_state
                        }
                    # 用 4 倍 token 上限近似为字符数，作为截断阈值
                    findings_token_limit = model_token_limit * 4
                else:
                    # 后续重试：每次缩减 10%
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # 截断研究发现后重试
                findings = findings[:findings_token_limit]
                continue
            else:
                # 非 token 超限错误：立即返回错误
                return {
                    "final_report": f"生成最终报告时出错：{e}",
                    "messages": [AIMessage(content="因错误，报告生成失败")],
                    **cleared_state
                }
    
    # 步骤 4：所有重试用尽后返回失败结果
    return {
        "final_report": "生成最终报告时出错：重试次数已达上限",
        "messages": [AIMessage(content="重试次数用尽后报告生成失败")],
        **cleared_state
    }

# 深度研究主图构建
# 构建从用户输入到最终报告的完整深度研究工作流
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# 为完整研究流程添加主工作流节点
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # 用户澄清阶段
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # 研究规划阶段
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # 研究执行阶段
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # 报告生成阶段

# 定义主工作流的边，用于顺序执行
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # 入口
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # 从研究到报告
deep_researcher_builder.add_edge("final_report_generation", END)                   # 最终出口

# 编译完整的深度研究工作流
deep_researcher = deep_researcher_builder.compile()
