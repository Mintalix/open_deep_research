from typing import cast

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from open_deep_research.utils import get_today_str
from tests.prompts import (
    COMPLETENESS_PROMPT,
    CORRECTNESS_PROMPT,
    GROUNDEDNESS_PROMPT,
    OVERALL_QUALITY_PROMPT,
    RELEVANCE_PROMPT,
    STRUCTURE_PROMPT,
)

eval_model = ChatOpenAI(
    model="gpt-4.1",
)

def _format_input_query(inputs: dict) -> str:
    messages = inputs["messages"]
    if len(messages) == 1:
        return messages[0]["content"]

    role_to_string_format_map = {
        "user": "<user_input>\n{content}\n</user_input>",
        "assistant": "<assistant_follow_up>\n{content}\n</assistant_follow_up>",
    }

    return "\n\n".join([role_to_string_format_map[message["role"]].format(content=message["content"]) for message in messages])


class OverallQualityScore(BaseModel):
    """根据具体评分标准为报告的总体质量打分。"""
    research_depth: int = Field(description="1-5 的整数评分,表示报告满足所提供评分标准的程度(1 = 完全不满足,5 = 满足全部标准)。")
    source_quality: int = Field(description="1-5 的整数评分,表示报告满足所提供评分标准的程度(1 = 完全不满足,5 = 满足全部标准)。")
    analytical_rigor: int = Field(description="1-5 的整数评分,表示报告满足所提供评分标准的程度(1 = 完全不满足,5 = 满足全部标准)。")
    practical_value: int = Field(description="1-5 的整数评分,表示报告满足所提供评分标准的程度(1 = 完全不满足,5 = 满足全部标准)。")
    balance_and_objectivity: int = Field(description="1-5 的整数评分,表示报告满足所提供评分标准的程度(1 = 完全不满足,5 = 满足全部标准)。")
    writing_quality: int = Field(description="1-5 的整数评分,表示报告满足所提供评分标准的程度(1 = 完全不满足,5 = 满足全部标准)。")

def eval_overall_quality(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = f"""用户输入:{query}\n\n报告:\n\n{final_report}\n\n请评估报告是否满足上述标准,并为你的评估提供详细理由。"""
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]
    eval_result = cast(OverallQualityScore, eval_model.with_structured_output(OverallQualityScore).invoke([
        {"role": "system", "content": OVERALL_QUALITY_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content}
    ]))
    return [
        {"key": "research_depth_score", "score": eval_result.research_depth / 5},
        {"key": "source_quality_score", "score": eval_result.source_quality / 5},
        {"key": "analytical_rigor_score", "score": eval_result.analytical_rigor / 5},
        {"key": "practical_value_score", "score": eval_result.practical_value / 5},
        {"key": "balance_and_objectivity_score", "score": eval_result.balance_and_objectivity / 5},
        {"key": "writing_quality_score", "score": eval_result.writing_quality / 5},
    ]


class RelevanceScore(BaseModel):
    """根据具体评分标准为报告的相关性打分。"""
    reasoning: str = Field(description="评分理由,需引用报告中的具体示例。")
    score: int = Field(description="1-5 的整数评分,表示报告满足相关性评分标准的程度(1 = 完全不满足,5 = 满足全部标准)。")

def eval_relevance(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = f"""用户输入:{query}\n\n报告:\n\n{final_report}\n\n请评估报告是否满足上述标准,并为你的评估提供详细理由。"""
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(RelevanceScore, eval_model.with_structured_output(RelevanceScore).invoke([
        {"role": "system", "content": RELEVANCE_PROMPT.format(today=get_today_str())},
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "relevance_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


class StructureScore(BaseModel):
    """根据具体评分标准为报告的结构打分。"""
    reasoning: str = Field(description="评分理由,需引用报告中的具体示例。")
    score: int = Field(description="1-5 的整数评分,表示报告满足结构与行文流畅度评分标准的程度(1 = 完全不满足,5 = 满足全部标准)。")

def eval_structure(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    user_input_content = STRUCTURE_PROMPT.format(user_question=query, report=final_report, today=get_today_str())
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(StructureScore, eval_model.with_structured_output(StructureScore).invoke([
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "structure_and_cohesiveness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}


class CorrectnessScore(BaseModel):
    """根据具体评分标准为报告的正确性打分。"""
    reasoning: str = Field(description="评分理由,需引用报告中的具体示例。")
    score: int = Field(description="1-5 的整数评分,表示报告满足正确性评分标准的程度(1 = 完全不满足,5 = 满足全部标准)。")

def eval_correctness(inputs: dict, outputs: dict, reference_outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    answer = reference_outputs["answer"]
    user_input_content = CORRECTNESS_PROMPT.format(user_question=query, report=final_report, answer=answer, today=get_today_str())
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(CorrectnessScore, eval_model.with_structured_output(CorrectnessScore).invoke([
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "correctness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}

class GroundednessClaim(BaseModel):
    """报告中的一条论断,以及它是否有上下文依据"""
    claim: str = Field(description="从报告中提取的论断。")
    grounded: bool = Field(description="该论断是否有上下文依据。")

class GroundednessScore(BaseModel):
    """提取各条论断,并判断它们是否有上下文依据"""
    claims: list[GroundednessClaim] = Field(description="从报告中提取的全部论断,以及它们是否有上下文依据。")

def eval_groundedness(inputs: dict, outputs: dict):
    final_report = outputs["final_report"]
    context = str(outputs["raw_notes"])

    user_input_content = GROUNDEDNESS_PROMPT.format(context=context, report=final_report, today=get_today_str())
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(GroundednessScore, eval_model.with_structured_output(GroundednessScore).with_retry(stop_after_attempt=3).invoke([
        {"role": "user", "content": user_input_content},
    ]))
    # 归一化到 0-1
    grounded_claims = [claim for claim in eval_result.claims if claim.grounded]
    return {"key": "groundedness_score", "score": len(grounded_claims) / len(eval_result.claims), "comment": str(eval_result.claims)}


class CompletenessScore(BaseModel):
    """根据具体评分标准为报告的完整性打分。"""
    reasoning: str = Field(description="评分理由,需引用报告中的具体示例。")
    score: int = Field(description="1-5 的整数评分,表示报告满足完整性评分标准的程度(1 = 完全不满足,5 = 满足全部标准)。")

def eval_completeness(inputs: dict, outputs: dict):
    query = _format_input_query(inputs)
    final_report = outputs["final_report"]
    research_brief = outputs["research_brief"]
    user_input_content = COMPLETENESS_PROMPT.format(user_question=query, research_brief=research_brief, report=final_report, today=get_today_str())
    if isinstance(eval_model, ChatAnthropic):
        user_input_content = [{
            "type": "text",
            "text": user_input_content,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    eval_result = cast(CompletenessScore, eval_model.with_structured_output(CompletenessScore).invoke([
        {"role": "user", "content": user_input_content}
    ]))
    return {"key": "completeness_score", "score": eval_result.score / 5, "comment": eval_result.reasoning}
