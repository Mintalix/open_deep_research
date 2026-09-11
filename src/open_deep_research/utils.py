"""深度研究智能体的工具函数与辅助函数。"""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

##########################
# Tavily 搜索工具辅助函数
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "一个为全面、准确、可信结果而优化的搜索引擎。"
    "适用于需要回答时事相关问题的场景。"
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """从 Tavily 搜索 API 获取并摘要搜索结果。

    参数：
        queries: 要执行的搜索查询列表
        max_results: 每个查询返回的最大结果数
        topic: 搜索结果的主题过滤条件（general、news 或 finance）
        config: 用于 API 密钥和模型设置的运行时配置

    返回：
        包含摘要后搜索结果的格式化字符串
    """
    # 步骤 1：异步执行搜索查询
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    
    # 步骤 2：按 URL 对结果去重，避免重复处理相同内容
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    
    # 步骤 3：根据配置设置摘要模型
    configurable = Configuration.from_runnable_config(config)
    
    # 字符数上限，用于保持在模型 token 限制内（可配置）
    max_char_to_include = configurable.max_content_length
    
    # 初始化带重试逻辑的摘要模型
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    
    # 步骤 4：创建摘要任务（跳过空内容）
    async def noop():
        """针对没有原始内容的结果的空操作函数。"""
        return None
    
    summarization_tasks = [
        noop() if not result.get("raw_content") 
        else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]
    
    # 步骤 5：并行执行所有摘要任务
    summaries = await asyncio.gather(*summarization_tasks)
    
    # 步骤 6：将结果与其摘要合并
    summarized_results = {
        url: {
            'title': result['title'], 
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(), 
            unique_results.values(), 
            summaries
        )
    }
    
    # 步骤 7：格式化最终输出
    if not summarized_results:
        return "未找到有效的搜索结果。请尝试其他搜索查询，或改用其他搜索 API。"
    
    formatted_output = "搜索结果：\n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- 来源 {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"摘要：\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    config: RunnableConfig = None
):
    """异步执行多个 Tavily 搜索查询。

    参数：
        search_queries: 要执行的搜索查询字符串列表
        max_results: 每个查询的最大结果数
        topic: 用于过滤结果的主题类别
        include_raw_content: 是否包含网页完整内容
        config: 用于获取 API 密钥的运行时配置

    返回：
        来自 Tavily API 的搜索结果字典列表
    """
    # 使用配置中的 API 密钥初始化 Tavily 客户端
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # 创建搜索任务以并行执行
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]
    
    # 并行执行所有搜索查询并返回结果
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """使用 AI 模型对网页内容进行摘要，并带超时保护。

    参数：
        model: 为摘要任务配置的聊天模型
        webpage_content: 待摘要的原始网页内容

    返回：
        包含关键摘录的格式化摘要；若摘要失败则返回原始内容
    """
    try:
        # 创建带当前日期上下文的提示词
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        )
        
        # 执行摘要并设置超时，防止任务挂起
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 摘要任务的超时时间为 60 秒
        )
        
        # 以结构化分节格式化摘要
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except asyncio.TimeoutError:
        # 摘要过程中超时 - 返回原始内容
        logging.warning("摘要任务超过 60 秒超时，返回原始内容")
        return webpage_content
    except Exception as e:
        # 摘要过程中的其他错误 - 记录日志并返回原始内容
        logging.warning(f"摘要失败，错误：{str(e)}，返回原始内容")
        return webpage_content

##########################
# 反思工具辅助函数
##########################

@tool(description="用于研究规划的战略性反思工具")
def think_tool(reflection: str) -> str:
    """用于对研究进展和决策进行战略性反思的工具。

    在每次搜索后使用此工具，系统地分析结果并规划下一步。
    它会在研究工作流中设置一个刻意的停顿，以便做出高质量的决策。

    何时使用：
    - 收到搜索结果后：我发现了哪些关键信息？
    - 决定下一步之前：我掌握的信息是否足以全面作答？
    - 评估研究缺口时：我还缺少哪些具体信息？
    - 结束研究之前：我现在能否给出完整的回答？

    反思应涵盖：
    1. 当前发现的分析 - 我已经收集了哪些具体信息？
    2. 缺口评估 - 还缺少哪些关键信息？
    3. 质量评估 - 我是否有足够的证据/示例来给出好的回答？
    4. 战略决策 - 我应该继续搜索，还是给出我的回答？

    参数：
        reflection: 你对研究进展、发现、缺口和下一步的详细反思

    返回：
        确认反思已记录，供决策使用
    """
    return f"反思已记录：{reflection}"

##########################
# MCP 辅助函数
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """通过 OAuth 令牌交换，将 Supabase 令牌换取为 MCP 访问令牌。

    参数：
        supabase_token: 有效的 Supabase 身份验证令牌
        base_mcp_url: MCP 服务器的基础 URL

    返回：
        成功时返回令牌数据字典，失败时返回 None
    """
    try:
        # 准备 OAuth 令牌交换请求数据
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # 执行令牌交换请求
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # 成功获取令牌
                    token_data = await response.json()
                    return token_data
                else:
                    # 记录错误详情以便调试
                    response_text = await response.text()
                    logging.error(f"令牌交换失败：{response_text}")
                    
    except Exception as e:
        logging.error(f"令牌交换过程中出错：{e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """获取已存储的身份验证令牌，并校验其有效期。

    参数：
        config: 包含线程和用户标识符的运行时配置

    返回：
        令牌有效且未过期时返回令牌字典，否则返回 None
    """
    store = get_store()
    
    # 从配置中提取所需标识符
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # 获取已存储的令牌
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # 检查令牌是否过期
    expires_in = tokens.value.get("expires_in")  # 距过期剩余的秒数
    created_at = tokens.created_at  # 令牌创建时间（datetime）
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # 令牌已过期，清理后返回 None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """将身份验证令牌存储到配置存储中。

    参数：
        config: 包含线程和用户标识符的运行时配置
        tokens: 要存储的令牌字典
    """
    store = get_store()
    
    # 从配置中提取所需标识符
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # 存储令牌
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """获取并刷新 MCP 令牌，必要时获取新令牌。

    参数：
        config: 带有身份验证信息的运行时配置

    返回：
        有效的令牌字典；若无法获取令牌则返回 None
    """
    # 先尝试获取现有的有效令牌
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # 提取 Supabase 令牌以进行新的令牌交换
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # 提取 MCP 配置
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # 将 Supabase 令牌交换为 MCP 令牌
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # 存储新令牌并返回
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """为 MCP 工具包装完整的身份验证与错误处理。

    参数：
        tool: 要包装的 MCP 结构化工具

    返回：
        带身份验证错误处理的增强工具
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """带 MCP 错误处理和友好提示信息的增强协程。"""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """在异常链中递归查找 MCP 错误。"""
            if isinstance(exc, McpError):
                return exc
            
            # 通过检查属性来处理 ExceptionGroup（Python 3.11+）
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # 执行原始工具功能
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # 在异常链中查找 MCP 特有错误
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # 不是 MCP 错误，重新抛出原始异常
                raise original_error
            
            # 处理 MCP 特有的错误情形
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # 检查是否为需要身份验证/交互的错误
            if error_code == -32003:  # 表示需要交互的错误码
                message_payload = error_data.get("message", {})
                error_message = "需要交互"
                
                # 如有面向用户的友好提示则提取
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # 如提供了 URL 则附加，供用户参考
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # 对于其他 MCP 错误，重新抛出原始异常
            raise original_error
    
    # 用增强版协程替换工具的原协程
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """加载并配置带身份验证的 MCP（Model Context Protocol）工具。

    参数：
        config: 包含 MCP 服务器详情的运行时配置
        existing_tool_names: 已在使用的工具名称集合，用于避免冲突

    返回：
        配置完成、可供使用的 MCP 工具列表
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 步骤 1：如需身份验证则先行处理
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # 步骤 2：校验配置要求
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        return []
    
    # 步骤 3：建立 MCP 服务器连接
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # 如有令牌则配置身份验证请求头
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO：待 OAP 合并多 MCP 服务器支持后，更新此代码
    
    # 步骤 4：从 MCP 服务器加载工具
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # 若 MCP 服务器连接失败，返回空列表
        return []
    
    # 步骤 5：筛选并配置工具
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # 跳过名称冲突的工具
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP 工具 '{mcp_tool.name}' 与现有工具名称冲突 - 跳过"
            )
            continue
        
        # 仅纳入配置中指定的工具
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # 为工具包装身份验证处理并加入列表
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools


##########################
# 工具辅助函数
##########################

async def get_search_tool(search_api: SearchAPI):
    """根据指定的 API 提供商配置并返回搜索工具。

    参数：
        search_api: 要使用的搜索 API 提供商（Anthropic、OpenAI、Tavily 或 None）

    返回：
        为指定提供商配置好的搜索工具对象列表
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic 原生网络搜索（带使用次数限制）
        return [{
            "type": "web_search_20250305", 
            "name": "web_search", 
            "max_uses": 5
        }]
        
    elif search_api == SearchAPI.OPENAI:
        # OpenAI 的网络搜索预览功能
        return [{"type": "web_search_preview"}]
        
    elif search_api == SearchAPI.TAVILY:
        # 为 Tavily 搜索工具配置元数据
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]
        
    elif search_api == SearchAPI.NONE:
        # 未配置搜索功能
        return []
        
    # 针对未知搜索 API 类型的默认回退处理
    return []
    
async def get_all_tools(config: RunnableConfig):
    """组装完整的工具集，包括研究、搜索和 MCP 工具。

    参数：
        config: 指定搜索 API 和 MCP 设置的运行时配置

    返回：
        研究操作所需的全部已配置且可用的工具列表
    """
    # 从核心研究工具开始
    tools = [tool(ResearchComplete), think_tool]
    
    # 添加已配置的搜索工具
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    # 记录现有工具名称以避免冲突
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # 如已配置则添加 MCP 工具
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """从工具调用消息中提取笔记。"""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# 模型提供商原生网络搜索辅助函数
##########################

def anthropic_websearch_called(response):
    """检测响应中是否使用了 Anthropic 原生网络搜索。

    参数：
        response: 来自 Anthropic API 的响应对象

    返回：
        如调用了网络搜索则返回 True，否则返回 False
    """
    try:
        # 逐层查看响应元数据结构
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # 检查服务端工具使用信息
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # 查找网络搜索请求次数
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # 只要发生过网络搜索请求即返回 True
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # 处理响应结构不符合预期的情况
        return False

def openai_websearch_called(response):
    """检测响应中是否使用了 OpenAI 的网络搜索功能。

    参数：
        response: 来自 OpenAI API 的响应对象

    返回：
        如调用了网络搜索则返回 True，否则返回 False
    """
    # 检查响应元数据中的工具输出
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # 在工具输出中查找网络搜索调用
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False


##########################
# Token 超限辅助函数
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """判断某个异常是否表明超出了 token/上下文限制。

    参数：
        exception: 要分析的异常
        model_name: 可选的模型名称，用于优化提供商检测

    返回：
        若异常表明超出了 token 限制则返回 True，否则返回 False
    """
    error_str = str(exception).lower()
    
    # 步骤 1：若提供了模型名称则据此判断提供商
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    # 步骤 2：检查各提供商特有的 token 超限模式
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    # 步骤 3：若提供商未知，则逐一检查所有提供商
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表明 OpenAI token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 检查这是否为 OpenAI 异常
    is_openai_exception = (
        'openai' in exception_type.lower() or 
        'openai' in module_name.lower()
    )
    
    # 检查典型的 OpenAI token 超限错误类型
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_request_error:
        # 在错误信息中查找与 token 相关的关键词
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # 检查特定的 OpenAI 错误码
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')
        
        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True
    
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表明 Anthropic token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 检查这是否为 Anthropic 异常
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or 
        'anthropic' in module_name.lower()
    )
    
    # 检查 Anthropic 特有的错误模式
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        # Anthropic 使用特定的错误信息表示 token 超限
        if 'prompt is too long' in error_str:
            return True
    
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表明 Google/Gemini token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 检查这是否为 Google/Gemini 异常
    is_google_exception = (
        'google' in exception_type.lower() or 
        'google' in module_name.lower()
    )
    
    # 检查 Google 特有的资源耗尽错误
    is_resource_exhausted = class_name in [
        'ResourceExhausted', 
        'GoogleGenerativeAIFetchError'
    ]
    
    if is_google_exception and is_resource_exhausted:
        return True
    
    # 检查特定的 Google API 资源耗尽模式
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# 注意：此表可能已过时或不适用于你的模型。请按需更新。
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """查找特定模型的 token 限制。

    参数：
        model_string: 要查找的模型标识字符串

    返回：
        找到时返回整数形式的 token 限制；若模型不在查找表中则返回 None
    """
    # 在已知模型 token 限制中查找
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # 未在查找表中找到该模型
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """通过移除从最后一条 AI 消息开始的所有消息来截断消息历史。

    这在处理 token 超限错误时很有用，可以借此移除较近的上下文。

    参数：
        messages: 要截断的消息对象列表

    返回：
        截断后的消息列表，至（但不包含）最后一条 AI 消息
    """
    # 从后向前遍历消息，查找最后一条 AI 消息
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # 返回最后一条 AI 消息之前的所有消息（不含该消息）
            return messages[:i]
    
    # 未找到 AI 消息，返回原始列表
    return messages

##########################
# 其他辅助函数
##########################

def get_today_str() -> str:
    """获取适合在提示词和输出中展示的当前日期。

    返回：
        人类可读的日期字符串，格式形如 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """从配置中提取值，处理枚举与 None 值。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """从环境变量或配置中获取特定模型的 API 密钥。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    """从环境变量或配置中获取 Tavily API 密钥。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")
