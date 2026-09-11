import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict, Literal

from langchain_core.runnables import RunnableConfig

DEFAULT_REPORT_STRUCTURE = """使用以下结构围绕用户提供的主题撰写报告：

1. 引言（无需研究）
   - 简要概述该主题领域

2. 主体部分：
   - 每个章节应聚焦于用户所提供主题的一个子主题

3. 结论
   - 力求包含 1 个结构化元素（列表或表格），提炼主体各章节的要点
   - 提供报告的简明摘要"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"
    NONE = "none"

@dataclass(kw_only=True)
class Configuration:
    """基于工作流/图的实现（graph.py）的配置。"""
    # 常用配置
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str = "openai"
    summarization_model: str = "gpt-4.1"
    max_structured_output_retries: int = 3
    include_source_str: bool = False
    
    # 工作流专用配置
    number_of_queries: int = 2 # 每次迭代要生成的搜索查询数量
    max_search_depth: int = 2 # 反思 + 搜索的最大迭代次数
    planner_provider: str = "anthropic"
    planner_model: str = "claude-3-7-sonnet-latest"
    planner_model_kwargs: Optional[Dict[str, Any]] = None
    writer_provider: str = "openai"
    writer_model: str = "gpt-4.1"
    writer_model_kwargs: Optional[Dict[str, Any]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 创建 Configuration 实例。"""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

@dataclass(kw_only=True)
class MultiAgentConfiguration:
    """多智能体实现（multi_agent.py）的配置。"""
    # 常用配置
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str = "openai"
    summarization_model: str = "gpt-4.1"
    include_source_str: bool = False
    
    # 多智能体专用配置
    number_of_queries: int = 2 # 每个章节要生成的搜索查询数量
    supervisor_model: str = "anthropic:claude-sonnet-4-20250514"
    researcher_model: str = "anthropic:claude-sonnet-4-20250514"
    ask_for_clarification: bool = False # 是否向用户请求澄清
    # MCP 服务器配置
    mcp_server_config: Optional[Dict[str, Any]] = None
    mcp_prompt: Optional[str] = None
    mcp_tools_to_include: Optional[list[str]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "MultiAgentConfiguration":
        """从 RunnableConfig 创建 MultiAgentConfiguration 实例。"""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

# 保留旧的 Configuration 类以保持向后兼容
Configuration = Configuration
