#!/usr/bin/env python

import os
import uuid
import pytest
import asyncio
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langsmith import testing as t
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# 导入报告生成智能体
from legacy.graph import builder
from legacy.multi_agent import supervisor_builder

# 初始化 rich 控制台，设置 force_terminal 以确保即使 pytest 捕获 stdout 时也有输出
console = Console(force_terminal=True, width=120)

class CriteriaGrade(BaseModel):
    """依据特定标准对回答进行评分。"""
    grade: bool = Field(description="回答是否满足所给标准？")
    justification: str = Field(description="评分与得分的理由，包括回答中的具体示例。")

# 在测试时创建评估 LLM 的函数
def get_evaluation_llm(eval_model=None):
    """创建并返回一个评估 LLM。
    
    Args:
        eval_model: 用于评估的模型标识符
                    格式："provider:model_name"（例如 "anthropic:claude-3-7-sonnet-latest"）
                    如果为 None，则使用环境变量或默认值
    
    Returns:
        用于生成评估评分的结构化 LLM
    """
    # 依次使用：传入的模型 → 环境变量 → 默认值
    model_to_use = eval_model or os.environ.get("EVAL_MODEL", "anthropic:claude-3-7-sonnet-latest")
    
    criteria_eval_llm = init_chat_model(model_to_use)
    return criteria_eval_llm.with_structured_output(CriteriaGrade)

RESPONSE_CRITERIA_SYSTEM_PROMPT = """
你正在评估一份研究报告的质量。请依据以下标准评估该报告，并对章节相关性格外严格。

1. 主题相关性（总体）：报告是否直接且充分地回应用户输入的主题？

2. 章节相关性（关键）：请仔细评估每个章节与主题的相关性：
   - 通过 ## 标题识别每个章节
   - 针对每个章节，判断其是否与核心主题直接相关
   - 标记任何看似偏离正题、跑题或仅与主题松散相关的章节
   - 高质量的报告不应包含任何不相关的章节

3. 结构与连贯性：各章节之间的推进是否合乎逻辑，构成连贯的叙述？

4. 引言质量：引言是否有效地提供了背景信息并确立了报告的范围？

5. 结论质量：结论是否切实总结了报告的关键发现与洞察？

6. 结构化元素：报告是否有效运用结构化元素（如表格、列表）来传递信息？

7. 章节标题：章节标题是否正确使用了 Markdown 格式（# 用于标题、## 用于章节、### 用于小节）？

8. 引用来源：报告是否在每个正文章节中正确引用了来源？

9. 总体质量：报告是否研究充分、内容准确且行文专业？

评估说明：
- 对章节相关性必须严格——所有章节都必须与核心主题明确相关
- 只要存在哪怕一个不相关的章节，就应视为报告存在缺陷
- 你必须逐一点名每个章节并评估其相关性
- 针对每项标准，引用报告中的具体示例来支撑你的评估
- 只要存在与主题不相关的章节，无论其他方面如何，报告均判定为不通过
""" 

# 为测试配置定义 fixture
@pytest.fixture
def research_agent(request):
    """从命令行或环境变量获取研究智能体类型。"""
    return request.config.getoption("--research-agent") or os.environ.get("RESEARCH_AGENT", "multi_agent")

@pytest.fixture
def search_api(request):
    """从命令行或环境变量获取搜索 API。"""
    return request.config.getoption("--search-api") or os.environ.get("SEARCH_API", "tavily")

@pytest.fixture
def eval_model(request):
    """从命令行或环境变量获取评估模型。"""
    return request.config.getoption("--eval-model") or os.environ.get("EVAL_MODEL", "anthropic:claude-3-7-sonnet-latest")

@pytest.fixture
def models(request, research_agent):
    """根据智能体类型获取模型配置。"""
    if research_agent == "multi_agent":
        return {
            "supervisor_model": (
                request.config.getoption("--supervisor-model") or 
                os.environ.get("SUPERVISOR_MODEL", "anthropic:claude-3-7-sonnet-latest")
            ),
            "researcher_model": (
                request.config.getoption("--researcher-model") or 
                os.environ.get("RESEARCHER_MODEL", "anthropic:claude-3-5-sonnet-latest")
            ),
        }
    else:  # graph 智能体
        return {
            "planner_provider": (
                request.config.getoption("--planner-provider") or 
                os.environ.get("PLANNER_PROVIDER", "anthropic")
            ),
            "planner_model": (
                request.config.getoption("--planner-model") or 
                os.environ.get("PLANNER_MODEL", "claude-3-7-sonnet-latest")
            ),
            "writer_provider": (
                request.config.getoption("--writer-provider") or 
                os.environ.get("WRITER_PROVIDER", "anthropic")
            ),
            "writer_model": (
                request.config.getoption("--writer-model") or 
                os.environ.get("WRITER_MODEL", "claude-3-5-sonnet-latest")
            ),
            "max_search_depth": int(
                request.config.getoption("--max-search-depth") or 
                os.environ.get("MAX_SEARCH_DEPTH", "2")
            ),
        }

# 注意：命令行选项定义在 conftest.py 中
# 这些 fixture 仍可使用在其中定义的选项

@pytest.mark.langsmith
def test_response_criteria_evaluation(research_agent, search_api, models, eval_model):
    """测试报告是否满足指定的质量标准。"""
    console.print(Panel.fit(
        f"[bold blue]正在以 {search_api} 搜索测试 {research_agent} 的报告生成[/bold blue]",
        title="测试配置"
    ))
    
    # 创建模型配置表格
    models_table = Table(title="模型配置")
    models_table.add_column("参数", style="cyan")
    models_table.add_column("值", style="green")
    
    for key, value in models.items():
        models_table.add_row(key, str(value))
    models_table.add_row("eval_model", eval_model)
    
    console.print(models_table)
    
    # 将输入记录到 LangSmith
    t.log_inputs({
        "agent_type": research_agent, 
        "search_api": search_api,
        "models": models,
        "eval_model": eval_model,
        "test": "report_quality_evaluation",
        "description": f"测试 {research_agent} 智能体在 {search_api} 下的报告质量"
    })
 
    # 根据参数运行相应的智能体
    if research_agent == "multi_agent":

        # 初始消息
        initial_msg = [{"role": "user", "content": "请给我一份关于 MCP（model context protocol）的高层次概览。报告保持为 3 个正文章节：一节介绍 MCP 的起源，一节介绍 MCP 服务器的有趣示例，还有一节介绍 MCP 的未来路线图。报告应面向开发者读者撰写。"}]

        # 多智能体方案的 checkpointer
        checkpointer = MemorySaver()
        graph = supervisor_builder.compile(checkpointer=checkpointer)

        # 使用提供的参数创建配置
        config = {
            "thread_id": str(uuid.uuid4()),
            "search_api": search_api,
            "supervisor_model": models.get("supervisor_model"),
            "researcher_model": models.get("researcher_model"),
            "ask_for_clarification": False, # 不向用户征询澄清，直接继续撰写报告
            "process_search_results": "summarize", # 可选：对搜索结果进行摘要 
        }
        
        thread_config = {"configurable": config}

        # 使用 asyncio 运行工作流
        asyncio.run(graph.ainvoke({"messages": initial_msg}, config=thread_config))
        
        # 两次调用都完成后获取最终状态
        final_state = graph.get_state(thread_config)
        report = final_state.values.get('final_report', "未生成报告")
        console.print(f"[bold green]报告已生成，长度：{len(report)} 个字符[/bold green]")

    elif research_agent == "graph":
        
        # 主题查询 
        topic_query = "请给我一份关于 MCP（model context protocol）的高层次概览。报告保持为 3 个正文章节：一节介绍 MCP 的起源，一节介绍 MCP 服务器的有趣示例，还有一节介绍 MCP 的未来路线图。报告应面向开发者读者撰写。"
   
        # graph 方案的 checkpointer
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)
        
        # 使用提供的参数为 graph 智能体创建配置
        thread = {"configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": search_api,
            "planner_provider": models.get("planner_provider", "anthropic"),
            "planner_model": models.get("planner_model", "claude-3-7-sonnet-latest"),
            "writer_provider": models.get("writer_provider", "anthropic"),
            "writer_model": models.get("writer_model", "claude-3-5-sonnet-latest"),
            "max_search_depth": models.get("max_search_depth", 2),
        }}
        
        async def run_graph_agent(thread):    
            # 运行 graph 直到中断
            async for event in graph.astream({"topic":topic_query}, thread, stream_mode="updates"):
                if '__interrupt__' in event:
                    interrupt_value = event['__interrupt__'][0].value

            # 传入 True 以批准报告计划并继续撰写报告
            async for event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
                # console.print(f"[dim]{event}[/dim]")
                # console.print()
                None
            
            final_state = graph.get_state(thread)
            report = final_state.values.get('final_report', "未生成报告")
            return report
    
        report = asyncio.run(run_graph_agent(thread))

    # 使用指定模型获取评估 LLM
    criteria_eval_structured_llm = get_evaluation_llm(eval_model)
    
    # 依据我们的质量标准评估报告
    eval_result = criteria_eval_structured_llm.invoke([
        {"role": "system", "content": RESPONSE_CRITERIA_SYSTEM_PROMPT},
        {"role": "user", "content": f"""\n\n 报告： \n\n{report}\n\n请评估该报告是否满足上述标准，并为你的评估提供详细的理由。"""}
    ])

    # 提取章节标题用于分析
    import re
    section_headers = re.findall(r'##\s+([^\n]+)', report)
    
    # 展示生成的报告
    console.print(Panel(
        Markdown(report),
        title="生成的报告",
        border_style="blue"
    ))
    
    # 创建评估结果展示
    result_color = "green" if eval_result.grade else "red"
    result_text = "通过" if eval_result.grade else "失败"
    
    console.print(Panel.fit(
        f"[bold {result_color}]{result_text}[/bold {result_color}]",
        title="评估结果"
    ))
    
    # 创建章节表格
    sections_table = Table(title="报告结构分析")
    sections_table.add_column("章节", style="cyan")
    sections_table.add_column("标题", style="yellow")
    
    for i, header in enumerate(section_headers, 1):
        sections_table.add_row(f"章节 {i}", header)
    
    console.print(sections_table)
    console.print(f"[bold]共找到 {len(section_headers)} 个章节[/bold]")
    
    # 在面板中展示评分理由
    console.print(Panel(
        eval_result.justification,
        title="评估理由",
        border_style="yellow"
    ))
    
    # 将输出记录到 LangSmith
    t.log_outputs({
        "report": report,
        "evaluation_result": eval_result.grade,
        "justification": eval_result.justification,
        "report_length": len(report),
        "section_count": len(section_headers),
        "section_headers": section_headers,
    })
    
    # 满足评估标准则测试通过
    assert eval_result.grade