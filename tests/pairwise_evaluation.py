from langchain_anthropic import ChatAnthropic
from langsmith.evaluation import evaluate_comparative
from pydantic import BaseModel, Field

HEAD_TO_HEAD_PROMPT = """
我们正在测试深度研究智能体的两种不同实现。该研究智能体旨在针对给定问题开展深度研究。

问题是:
{question}

第一个实现的回答:
{answer_a}

第二个实现的回答:
{answer_b}

在评估这些智能体时,请牢记以下标准:
- 一个好的研究智能体应当研究足够的来源来回答问题。这些来源应当多样且高质量。来源并非越多越好,但必须拥有足够多的来源,才能让人对其提出的论断有信心。
- 一个好的研究智能体应当完整、全面地回答用户的问题。
- 深度研究智能体的成本很高。用户期望得到非常好且非常详尽的回答。他们应当能从这一次回答中获取所需的全部信息,通常无需再追问。
- 所有论断都应提供引用,且引用的格式应便于阅读和理解。

重要提示:
这两个实现的研究方式不同,因此它们可能拥有不同的信息和不同的来源。这是你需要评估的关键点。
哪个智能体找到了更好的来源并更好地回答了问题?我们最看重的是回答的质量与全面性。

请基于上述标准,选出你更偏好的回答,并说明理由!
"""

class HeadToHeadRanking(BaseModel):
    reasoning: str = Field(description="你选择该偏好回答的理由。这应当是一份详细的说明!")
    preferred_answer: int = Field(description="1 与 2 之中偏好的回答,其中 1 表示第一个回答,2 表示第二个回答。")


def head_to_head_evaluator(inputs: dict, outputs: list[dict]) -> list:
    grader_llm = ChatAnthropic(
        model="claude-opus-4-20250514",
        max_tokens=20000,
        thinking={"type": "enabled", "budget_tokens": 16000},
    )

    response = grader_llm.with_structured_output(HeadToHeadRanking).invoke(HEAD_TO_HEAD_PROMPT.format(
        question=inputs["messages"][0]["content"],
        answer_a=outputs[0].get("final_report", "N/A"),
        answer_b=outputs[1].get("final_report", "N/A"),
    ))

    if response.preferred_answer == 1:
        scores = [1, 0]
    elif response.preferred_answer == 2:
        scores = [0, 1]
    else:
        scores = [0, 0]
    return scores


ALL_THREE_PROMPT = """
我们正在测试深度研究智能体的三种不同实现。该研究智能体旨在针对给定问题开展深度研究。

问题是:
{question}

第一个实现的回答:
{answer_a}

第二个实现的回答:
{answer_b}

第三个实现的回答:
{answer_c}

在评估这些智能体时,请牢记以下标准:
- 一个好的研究智能体应当研究足够的来源来回答问题。这些来源应当多样且高质量。来源并非越多越好,但必须拥有足够多的来源,才能让人对其提出的论断有信心。
- 一个好的研究智能体应当完整、全面地回答用户的问题。
- 深度研究智能体的成本很高。用户期望得到非常好且非常详尽的回答。他们应当能从这一次回答中获取所需的全部信息,通常无需再追问。
- 所有论断都应提供引用,且引用的格式应便于阅读和理解。

重要提示:
这三种实现的研究方式不同,因此它们可能拥有不同的信息和不同的来源。这是你需要评估的关键点。
哪个智能体找到了更好的来源并更好地回答了问题?我们最看重的是回答的质量与全面性。

请基于上述标准,将三个回答排定 1 到 3 的名次,其中 1 是最佳回答,2 是第二好的回答,3 是最差的回答。
另外,请说明你为何做出这样的排名!
"""

class Rankings(BaseModel):
    reasoning: str = Field(description="你选择该偏好回答的理由。这应当是一份详细的说明!")
    preferred_answer: int = Field(description="1 与 3 之中偏好的回答,其中 1 表示第一个回答,2 表示第二个回答,3 表示第三个回答。")
    second_best_answer: int = Field(description="1 与 3 之中第二好的回答,其中 1 表示第一个回答,2 表示第二个回答,3 表示第三个回答。")
    worst_answer: int = Field(description="1 与 3 之中最差的回答,其中 1 表示第一个回答,2 表示第二个回答,3 表示第三个回答。")

def free_for_all_evaluator(inputs: dict, outputs: list[dict]) -> list:
    grader_llm = ChatAnthropic(
        model="claude-opus-4-20250514",
        max_tokens=20000,
        thinking={"type": "enabled", "budget_tokens": 16000},
    )

    response = grader_llm.with_structured_output(Rankings).invoke(ALL_THREE_PROMPT.format(
        question=inputs["messages"][0]["content"],
        answer_a=outputs[0].get("final_report", "N/A"),
        answer_b=outputs[1].get("final_report", "N/A"),
        answer_c=outputs[2].get("final_report", "N/A"),
    ))

    scores = [0, 0, 0]
    scores[response.preferred_answer - 1] = 1
    scores[response.second_best_answer - 1] = .5
    scores[response.worst_answer - 1] = 0
    return scores

single_agent = "DR Single Agent - Tavily #-87e8a6c0"
multi_agent_supervisor = "DR Supervisor: Multi Agent - Tavily #-cd25e7e3"
multi_agent_supervisor_v2 = "DR Supervisor: Multi Agent - Tavily (v2) #-40967f53"
multi_agent_workflow = "DR MAW - Tavily #-c6818a83"


# evaluate_comparative(
#     (single_agent_experiment_name, multi_agent_supervisor_experiment_name, multi_agent_workflow_experiment_name),  # 请替换为你的实验名称/ID
#     evaluators=[free_for_all_evaluator],
#     randomize_order=True,
# )

evaluate_comparative(
    (single_agent, multi_agent_supervisor_v2),  # 请替换为你的实验名称/ID
    evaluators=[head_to_head_evaluator],
    randomize_order=True,
)