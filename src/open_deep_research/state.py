"""深度研究智能体的图状态定义与数据结构。"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 结构化输出
###################
class ConductResearch(BaseModel):
    """调用此工具以针对特定主题开展研究。"""
    research_topic: str = Field(
        description="要研究的主题。应为单一主题，并需详细描述（至少一段话）。",
    )

class ResearchComplete(BaseModel):
    """调用此工具以表明研究已完成。"""

class Summary(BaseModel):
    """包含关键发现的研究摘要。"""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """用于向用户请求澄清的模型。"""
    
    need_clarification: bool = Field(
        description="是否需要向用户提出澄清性问题。",
    )
    question: str = Field(
        description="用于向用户澄清报告范围的问题",
    )
    verification: str = Field(
        description="确认信息：在用户提供必要信息后，我们将开始研究。",
    )

class ResearchQuestion(BaseModel):
    """用于引导研究的研究问题与简报。"""
    
    research_brief: str = Field(
        description="将用于引导研究的研究问题。",
    )


###################
# 状态定义
###################

def override_reducer(current_value, new_value):
    """允许覆盖状态值的归约器函数。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """输入状态仅包含 messages 字段。"""

class AgentState(MessagesState):
    """主智能体状态，包含消息与研究数据。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """管理研究任务的监督者状态。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """执行研究的各研究员的状态。"""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """各研究员的输出状态。"""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []