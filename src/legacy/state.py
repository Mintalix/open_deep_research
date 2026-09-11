from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

class Section(BaseModel):
    name: str = Field(
        description="报告该章节的名称。",
    )
    description: str = Field(
        description="本章节将涵盖的主要主题与概念的简要概述。",
    )
    research: bool = Field(
        description="是否针对报告的该章节进行网络研究。"
    )
    content: str = Field(
        description="章节的内容。"
    )   

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="报告的各个章节。",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="用于网络搜索的查询。")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="搜索查询列表。",
    )

class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="评估结果，表明回答是否满足要求（'pass'）或需要修改（'fail'）。"
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="后续搜索查询列表。",
    )

class ReportStateInput(TypedDict):
    topic: str # 报告主题
    
class ReportStateOutput(TypedDict):
    final_report: str # 最终报告
    # 仅供评估使用
    # 仅当 configurable.include_source_str 为 True 时才包含此字段
    source_str: str # 网络搜索来源内容格式化后的字符串

class ReportState(TypedDict):
    topic: str # 报告主题    
    feedback_on_report_plan: Annotated[list[str], operator.add] # 关于报告计划的反馈列表
    sections: list[Section] # 报告章节列表 
    completed_sections: Annotated[list, operator.add] # Send() API 所需的键
    report_sections_from_research: str # 由研究中已完成的章节组成的字符串，用于撰写最终章节
    final_report: str # 最终报告
    # 仅供评估使用
    # 仅当 configurable.include_source_str 为 True 时才包含此字段
    source_str: Annotated[str, operator.add] # 网络搜索来源内容格式化后的字符串

class SectionState(TypedDict):
    topic: str # 报告主题
    section: Section # 报告章节  
    search_iterations: int # 已完成的搜索迭代次数
    search_queries: list[SearchQuery] # 搜索查询列表
    source_str: str # 网络搜索来源内容格式化后的字符串
    report_sections_from_research: str # 由研究中已完成的章节组成的字符串，用于撰写最终章节
    completed_sections: list[Section] # 为 Send() API 在外层状态中复制的最终键

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # 为 Send() API 在外层状态中复制的最终键
    # 仅供评估使用
    # 仅当 configurable.include_source_str 为 True 时才包含此字段
    source_str: str # 网络搜索来源内容格式化后的字符串
