from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from legacy.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)

from legacy.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from legacy.configuration import Configuration
from legacy.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search,
    get_today_str
)

## 节点 -- 

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """生成包含各章节的初始报告计划。

    该节点：
    1. 获取报告结构和搜索参数的配置
    2. 生成搜索查询以收集规划所需的上下文
    3. 使用这些查询执行网络搜索
    4. 使用 LLM 生成包含各章节的结构化计划

    参数：
        state: 当前包含报告主题的图状态
        config: 模型、搜索 API 等的配置

    返回：
        包含所生成章节的 Dict
    """

    # 输入
    topic = state["topic"]

    # 获取关于报告计划的反馈列表
    feedback_list = state.get("feedback_on_report_plan", [])

    # 将关于报告计划的反馈拼接为单个字符串
    feedback = " /// ".join(feedback_list) if feedback_list else ""

    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # 获取配置字典，默认为空
    params_to_pass = get_search_params(search_api, search_api_config)  # 过滤参数

    # 如有必要，将 JSON 对象转换为字符串
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # 设置撰写者模型（用于生成查询的模型）
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Queries)

    # 格式化系统指令
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries,
        today=get_today_str()
    )

    # 生成查询  
    results = await structured_llm.ainvoke([SystemMessage(content=system_instructions_query),
                                     HumanMessage(content="生成有助于规划报告各章节的搜索查询。")])

    # 网络搜索
    query_list = [query.search_query for query in results.queries]

    # 使用给定参数进行网络搜索
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    # 格式化系统指令
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str, feedback=feedback)

    # 设置规划器
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})

    # 报告规划器指令
    planner_message = """生成报告的各个章节。你的回答必须包含一个 'sections' 字段，其中为章节列表。
                        每个章节必须包含：name、description、research 和 content 字段。"""

    # 运行规划器
    if planner_model == "claude-3-7-sonnet-latest":
        # 为作为规划器的 claude-3-7-sonnet-latest 分配思考预算
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider, 
                                      max_tokens=20_000, 
                                      thinking={"type": "enabled", "budget_tokens": 16_000})

    else:
        # 对于其他模型，不专门分配思考 token
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider,
                                      model_kwargs=planner_model_kwargs)
    
    # 生成报告的各个章节
    structured_llm = planner_llm.with_structured_output(Sections)
    report_sections = await structured_llm.ainvoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])

    # 获取章节
    sections = report_sections.sections

    return {"sections": sections}

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """获取关于报告计划的人工反馈，并路由到后续步骤。

    该节点：
    1. 格式化当前报告计划以供人工审阅
    2. 通过中断获取反馈
    3. 路由到以下二者之一：
       - 若计划获批则进入章节撰写
       - 若提供了反馈则重新生成计划

    参数：
        state: 包含待审阅章节的当前图状态
        config: 工作流配置

    返回：
        用于重新生成计划或开始章节撰写的 Command
    """

    # 获取章节
    topic = state["topic"]
    sections = state['sections']
    sections_str = "\n\n".join(
        f"章节：{section.name}\n"
        f"描述：{section.description}\n"
        f"是否需要研究：{'是' if section.research else '否'}\n"
        for section in sections
    )

    # 通过中断获取关于报告计划的反馈
    interrupt_message = f"""请对以下报告计划提供反馈。
                        \n\n{sections_str}\n
                        \n该报告计划是否满足你的需求？\n传入 'true' 以批准该报告计划。\n或者，提供反馈以重新生成该报告计划："""
    
    feedback = interrupt(interrupt_message)

    # 若用户批准报告计划，则启动章节撰写
    if isinstance(feedback, bool) and feedback is True:
        # 将其视为批准并启动章节撰写
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ])
    
    # 若用户提供了反馈，则重新生成报告计划 
    elif isinstance(feedback, str):
        # 将其视为反馈并追加到现有列表
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": [feedback]})
    else:
        raise TypeError(f"不支持类型为 {type(feedback)} 的中断值。")
    
async def generate_queries(state: SectionState, config: RunnableConfig):
    """为研究特定章节而生成搜索查询。

    该节点使用 LLM，根据章节主题和描述
    生成有针对性的搜索查询。

    参数：
        state: 包含章节详情的当前状态
        config: 包含待生成查询数量等信息的配置

    返回：
        包含所生成搜索查询的 Dict
    """

    # 获取状态 
    topic = state["topic"]
    section = state["section"]

    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # 生成查询 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Queries)

    # 格式化系统指令
    system_instructions = query_writer_instructions.format(topic=topic, 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries,
                                                           today=get_today_str())

    # 生成查询  
    queries = await structured_llm.ainvoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="就所提供的主题生成搜索查询。")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """针对该章节的查询执行网络搜索。

    该节点：
    1. 取用已生成的查询
    2. 使用配置的搜索 API 执行搜索
    3. 将结果格式化为可用的上下文

    参数：
        state: 包含搜索查询的当前状态
        config: 搜索 API 配置

    返回：
        包含搜索结果及更新后迭代次数的 Dict
    """

    # 获取状态
    search_queries = state["search_queries"]

    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # 获取配置字典，默认为空
    params_to_pass = get_search_params(search_api, search_api_config)  # 过滤参数

    # 网络搜索
    query_list = [query.search_query for query in search_queries]

    # 使用给定参数进行网络搜索
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

async def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """撰写报告的一个章节，并评估是否需要更多研究。

    该节点：
    1. 使用搜索结果撰写章节内容
    2. 评估章节质量
    3. 二者择一：
       - 质量通过则完成该章节
       - 质量不通过则触发更多研究

    参数：
        state: 包含搜索结果和章节信息的当前状态
        config: 用于撰写和评估的配置

    返回：
        用于完成章节或进行更多研究的 Command
    """

    # 获取状态 
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    # 获取配置
    configurable = Configuration.from_runnable_config(config)

    # 格式化系统指令
    section_writer_inputs_formatted = section_writer_inputs.format(topic=topic, 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)

    # 生成章节  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 

    section_content = await writer_model.ainvoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # 将内容写入章节对象  
    section.content = section_content.content

    # 评分提示词 
    section_grader_message = ("对该报告进行评分，并针对缺失的信息考虑后续问题。 "
                              "如果评分为 'pass'，则所有后续查询都返回空字符串。 "
                              "如果评分为 'fail'，请提供具体的搜索查询以收集缺失的信息。")
    
    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    # 使用规划器模型进行反思
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})

    if planner_model == "claude-3-7-sonnet-latest":
        # 为作为规划器的 claude-3-7-sonnet-latest 分配思考预算
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, 
                                           max_tokens=20_000, 
                                           thinking={"type": "enabled", "budget_tokens": 16_000}).with_structured_output(Feedback)
    else:
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, model_kwargs=planner_model_kwargs).with_structured_output(Feedback)
    # 生成反馈
    feedback = await reflection_model.ainvoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    # 若该章节通过评分或已达到最大搜索深度，则将该章节发布到已完成章节 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # 将该章节发布到已完成章节 
        update = {"completed_sections": [section]}
        if configurable.include_source_str:
            update["source_str"] = source_str
        return Command(update=update, goto=END)

    # 用新内容更新现有章节，并更新搜索查询
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="search_web"
        )
    
async def write_final_sections(state: SectionState, config: RunnableConfig):
    """以已完成的章节为上下文，撰写无需研究的章节。

    该节点处理诸如结论或总结之类的章节，
    它们建立在已研究章节之上，而不需要直接开展研究。

    参数：
        state: 以已完成章节为上下文的当前状态
        config: 撰写模型的配置

    返回：
        包含新撰写章节的 Dict
    """

    # 获取配置
    configurable = Configuration.from_runnable_config(config)

    # 获取状态 
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # 格式化系统指令
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    # 生成章节  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    
    section_content = await writer_model.ainvoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="根据所提供的来源生成一个报告章节。")])
    
    # 将内容写入章节 
    section.content = section_content.content

    # 将更新后的章节写入已完成章节
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """将已完成的章节格式化，作为撰写最终章节的上下文。

    该节点获取所有已完成的研究章节，并将其格式化为
    单个上下文字符串，用于撰写总结性章节。

    参数：
        state: 包含已完成章节的当前状态

    返回：
        以格式化章节为上下文的 Dict
    """

    # 已完成章节的列表
    completed_sections = state["completed_sections"]

    # 将已完成的章节格式化为字符串，用作最终章节的上下文
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState, config: RunnableConfig):
    """将所有章节汇编为最终报告。

    该节点：
    1. 获取所有已完成的章节
    2. 按原始计划对它们排序
    3. 将它们组合成最终报告

    参数：
        state: 包含所有已完成章节的当前状态

    返回：
        包含完整报告的 Dict
    """

    # 获取配置
    configurable = Configuration.from_runnable_config(config)

    # 获取章节
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # 在保持原始顺序的同时，用已完成的内容更新各章节
    for section in sections:
        section.content = completed_sections[section.name]

    # 汇编最终报告
    all_sections = "\n\n".join([s.content for s in sections])

    if configurable.include_source_str:
        return {"final_report": all_sections, "source_str": state["source_str"]}
    else:
        return {"final_report": all_sections}

def initiate_final_section_writing(state: ReportState):
    """为撰写无需研究的章节创建并行任务。

    该边函数识别无需研究的章节，
    并为每个章节创建并行撰写任务。

    参数：
        state: 包含所有章节及研究上下文的当前状态

    返回：
        用于并行撰写章节的 Send 命令列表
    """

    # 对无需研究的章节，通过 Send() API 并行启动章节撰写
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]

# 报告章节子图 -- 

# 添加节点 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# 添加边
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# 用于初始报告计划、汇总各章节结果的外层图 -- 

# 添加节点
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# 添加边
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
