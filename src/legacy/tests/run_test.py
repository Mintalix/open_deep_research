#!/usr/bin/env python
import os
import subprocess
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

"""
Open Deep Research 的简化测试运行器，带 rich 控制台输出。

示例用法：
python tests/run_test.py --all  # 以 rich 输出运行所有智能体的测试
python tests/run_test.py --agent multi_agent --supervisor-model "anthropic:claude-3-7-sonnet-latest"
python tests/run_test.py --agent graph --search-api tavily
"""

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="以 rich 控制台输出运行 Open Deep Research 测试")
    parser.add_argument("--rich-output", action="store_true", default=True, help="在终端显示 rich 输出（默认：True）")
    parser.add_argument("--experiment-name", help="LangSmith 实验的名称")
    parser.add_argument("--agent", choices=["multi_agent", "graph"], help="为指定智能体运行测试")
    parser.add_argument("--all", action="store_true", help="为所有智能体运行测试")
    
    # 模型配置选项
    parser.add_argument("--supervisor-model", help="监督者智能体的模型（例如 'anthropic:claude-3-7-sonnet-latest'）")
    parser.add_argument("--researcher-model", help="研究员智能体的模型（例如 'anthropic:claude-3-5-sonnet-latest'）")
    parser.add_argument("--planner-provider", help="规划者模型的提供商（例如 'anthropic'）")
    parser.add_argument("--planner-model", help="基于 graph 的智能体中规划者的模型（例如 'claude-3-7-sonnet-latest'）")
    parser.add_argument("--writer-provider", help="撰写者模型的提供商（例如 'anthropic'）")
    parser.add_argument("--writer-model", help="基于 graph 的智能体中撰写者的模型（例如 'claude-3-5-sonnet-latest'）")
    parser.add_argument("--eval-model", help="用于评估报告质量的模型（默认：openai:claude-3-7-sonnet-latest）")
    parser.add_argument("--max-search-depth", help="graph 智能体的最大搜索深度")
    
    # 搜索 API 配置
    parser.add_argument("--search-api", choices=["tavily", "duckduckgo"], 
                        help="用于内容检索的搜索 API")
    
    args = parser.parse_args()
    
    # 定义可用的智能体及其测试配置
    agents = {
        "multi_agent": {
            "test": "tests/test_report_quality.py::test_response_criteria_evaluation",
            "topic": "Model Context Protocol",
            "description": "使用完整的 MCP 报告测试 multi_agent",
            "needs_research_agent_param": True,
        },
        "graph": {
            "test": "tests/test_report_quality.py::test_response_criteria_evaluation",
            "topic": "Model Context Protocol", 
            "description": "使用完整的 MCP 报告测试 graph 智能体",
            "needs_research_agent_param": True,
        }
    }
    
    # 确定要测试哪些智能体
    if args.agent:
        if args.agent in agents:
            agents_to_test = [args.agent]
        else:
            console.print(f"[red]错误：未知的智能体 '{args.agent}'[/red]")
            console.print(f"可用的智能体：{', '.join(agents.keys())}")
            return 1
    elif args.all:
        agents_to_test = list(agents.keys())
    else:
        # 默认测试所有智能体
        agents_to_test = list(agents.keys())
    
    # 为每个智能体运行测试
    for agent in agents_to_test:
        console.print(Rule(f"[bold blue]正在测试 {agent.upper()} 智能体[/bold blue]"))
        
        agent_config = agents[agent]
        
        # 为该智能体设置 LangSmith 环境
        project_name = f"ODR: Pytest"
        os.environ["LANGSMITH_PROJECT"] = project_name
        os.environ["LANGSMITH_TEST_SUITE"] = project_name
        
        # 确保已启用追踪
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        # 设置实验名称
        experiment_name = args.experiment_name if args.experiment_name else f"{agent_config['topic']}"
        os.environ["LANGSMITH_EXPERIMENT"] = experiment_name
        
        console.print(f"[dim]项目：{project_name}[/dim]")
        console.print(f"[dim]测试：{agent_config['description']}[/dim]")
        console.print(f"[dim]实验：{experiment_name}[/dim]")
        
        # 运行测试
        console.print(f"\n[green]正在为 {agent} 智能体运行测试...[/green]")
        run_test(agent, agent_config, args)
    
    console.print(Rule("[bold green]所有测试已完成[/bold green]"))

def run_test(agent, agent_config, args):
    """以 rich 控制台格式运行 pytest。"""
    # pytest 基础选项（添加 -s 以禁用输出捕获）
    base_pytest_options = ["-v", "-s", "--disable-warnings", "--langsmith-output"]
    
    # 构建命令
    cmd = ["python", "-m", "pytest", agent_config["test"]] + base_pytest_options
    
    # 如有需要，添加研究智能体参数
    if agent_config["needs_research_agent_param"]:
        cmd.append(f"--research-agent={agent}")
    
    # 如提供了模型配置则添加
    add_model_configs(cmd, args)
    
    # 在面板中美观地展示命令
    console.print(Panel(
        f"[bold]正在运行命令：[/bold]\n[dim]{' '.join(cmd)}[/dim]",
        style="blue",
        title="pytest 执行"
    ))
    
    # 运行命令并实时输出（不捕获输出）
    console.print(f"\n[yellow]开始执行测试...[/yellow]\n")
    result = subprocess.run(cmd)
    
    # 以 rich 格式展示结果
    console.print(f"\n[yellow]测试执行完成。[/yellow]")
    if result.returncode == 0:
        console.print(Panel(
            f"[bold green]✅ {agent} 智能体的测试已通过[/bold green]",
            style="green",
            title="测试结果"
        ))
    else:
        console.print(Panel(
            f"[bold red]❌ {agent} 智能体的测试失败[/bold red]\n[red]返回码：{result.returncode}[/red]",
            style="red",
            title="测试结果"
        ))

def add_model_configs(cmd, args):
    """向命令添加模型配置参数。"""
    if args.supervisor_model:
        cmd.append(f"--supervisor-model={args.supervisor_model}")
    if args.researcher_model:
        cmd.append(f"--researcher-model={args.researcher_model}")
    if args.planner_provider:
        cmd.append(f"--planner-provider={args.planner_provider}")
    if args.planner_model:
        cmd.append(f"--planner-model={args.planner_model}")
    if args.writer_provider:
        cmd.append(f"--writer-provider={args.writer_provider}")
    if args.writer_model:
        cmd.append(f"--writer-model={args.writer_model}")
    if args.eval_model:
        cmd.append(f"--eval-model={args.eval_model}")
    if args.search_api:
        cmd.append(f"--search-api={args.search_api}")
    if args.max_search_depth:
        cmd.append(f"--max-search-depth={args.max_search_depth}")

if __name__ == "__main__":
    sys.exit(main() or 0)