"""
open_deep_research 测试的 Pytest 配置。
"""

import pytest

def pytest_addoption(parser):
    """为 pytest 添加命令行选项。"""
    parser.addoption("--research-agent", action="store", help="智能体类型：multi_agent 或 graph")
    parser.addoption("--search-api", action="store", help="要使用的搜索 API")
    parser.addoption("--eval-model", action="store", help="用于评估的模型")
    parser.addoption("--supervisor-model", action="store", help="监督者智能体的模型")
    parser.addoption("--researcher-model", action="store", help="研究员智能体的模型")
    parser.addoption("--planner-provider", action="store", help="规划者模型的提供商")
    parser.addoption("--planner-model", action="store", help="用于规划的模型")
    parser.addoption("--writer-provider", action="store", help="撰写者模型的提供商")
    parser.addoption("--writer-model", action="store", help="用于撰写的模型")
    parser.addoption("--max-search-depth", action="store", help="最大搜索深度")