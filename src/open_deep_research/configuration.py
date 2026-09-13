"""Open Deep Research 系统的配置管理。"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """可用搜索 API 提供商的枚举。"""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """模型上下文协议（MCP）服务器的配置。"""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """MCP 服务器的 URL"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """向 LLM 提供的可用工具"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """该 MCP 服务器是否需要身份验证"""

class Configuration(BaseModel):
    """深度研究智能体的主配置类。"""
    
    # 通用配置
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "模型结构化输出调用的最大重试次数"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否允许研究员在开始研究前向用户提出澄清性问题"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=2,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 2,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "并发运行的研究单元最大数量。这将允许研究员使用多个子智能体开展研究。注意：并发越高，越容易触发速率限制。"
            }
        }
    )
    # 研究配置
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "研究使用的搜索 API。注意：请确保你的研究员模型支持所选的搜索 API。",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI 原生网页搜索", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic 原生网页搜索", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "无", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "研究监督者的最大研究迭代次数，即研究监督者对已有研究进行反思并提出后续问题的次数。"
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "单个研究员步骤中工具调用的最大迭代次数。"
            }
        }
    )
    # 模型配置
    summarization_model: str = Field(
        default="openai:deepseek-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek-flash",
                "description": "用于对 Tavily 搜索结果进行摘要的模型"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "摘要模型的最大输出 token 数"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "网页内容在进行摘要前的最大字符长度"
            }
        }
    )
    research_model: str = Field(
        default="openai:deepseek-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek-flash",
                "description": "执行研究的模型。注意：请确保你的研究员模型支持所选的搜索 API。"
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "研究模型的最大输出 token 数"
            }
        }
    )
    compression_model: str = Field(
        default="openai:deepseek-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek-flash",
                "description": "用于压缩子智能体研究发现的模型。注意：请确保你的压缩模型支持所选的搜索 API。"
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "压缩模型的最大输出 token 数"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:deepseek-flash",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek-flash",
                "description": "根据全部研究发现撰写最终报告的模型"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "最终报告模型的最大输出 token 数"
            }
        }
    )
    # MCP 服务器配置
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP 服务器配置"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "需要传递给智能体的、关于其可用 MCP 工具的任何附加说明。"
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 创建 Configuration 实例。"""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic 配置。"""
        
        arbitrary_types_allowed = True
