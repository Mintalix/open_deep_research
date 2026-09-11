# Open Deep Research 仓库概览

## 项目描述
Open Deep Research 是一个可配置、完全开源的深度研究智能体，可在多种模型提供商、搜索工具和 MCP（Model Context Protocol）服务器上运行。它支持并行处理的自动化研究，并能生成全面的报告。

## 仓库结构

### 根目录
- `README.md` - 包含快速开始指南的全面项目文档
- `pyproject.toml` - Python 项目配置与依赖
- `langgraph.json` - LangGraph 配置，定义主图入口
- `uv.lock` - UV 包管理器锁文件
- `LICENSE` - MIT 许可证
- `.env.example` - 环境变量模板（未被版本跟踪）

### 核心实现（`src/open_deep_research/`）
- `deep_researcher.py` - 主 LangGraph 实现（入口点：`deep_researcher`）
- `configuration.py` - 配置管理与设置
- `state.py` - 图状态定义与数据结构
- `prompts.py` - 系统提示词与提示词模板
- `utils.py` - 工具函数与辅助函数
- `files/` - 研究输出与示例文件

### 旧版实现（`src/legacy/`）
包含两个较早的研究实现：
- `graph.py` - 带人工介入的规划与执行工作流
- `multi_agent.py` - 监督者-研究者多智能体架构
- `legacy.md` - 旧版实现文档
- `CLAUDE.md` - 旧版专用的 Claude 说明
- `tests/` - 旧版专用测试

### 安全（`src/security/`）
- `auth.py` - 用于 LangGraph 部署的认证处理器

### 测试（`tests/`）
- `run_evaluate.py` - 配置为在 deep research bench 上运行的主评估脚本
- `evaluators.py` - 专用评估函数
- `prompts.py` - 评估提示词与评估标准
- `pairwise_evaluation.py` - 对比评估工具
- `supervisor_parallel_evaluation.py` - 多线程评估

### 示例（`examples/`）
- `arxiv.md` - ArXiv 研究示例
- `pubmed.md` - PubMed 研究示例
- `inference-market.md` - 推理市场分析示例

## 关键技术
- **LangGraph** - 工作流编排与图执行
- **LangChain** - LLM 集成与工具调用
- **多家 LLM 提供商** - 支持 OpenAI、Anthropic、Google、Groq、DeepSeek
- **搜索 API** - Tavily、OpenAI/Anthropic 原生搜索、DuckDuckGo、Exa
- **MCP 服务器** - 用于扩展能力的模型上下文协议

## 开发命令
- `uvx langgraph dev` - 启动带 LangGraph Studio 的开发服务器
- `python tests/run_evaluate.py` - 运行综合评估
- `ruff check` - 代码检查
- `mypy` - 类型检查

## 配置
所有设置均可通过以下方式配置：
- 环境变量（`.env` 文件）
- LangGraph Studio 中的 Web UI
- 直接修改配置

关键设置包括模型选择、搜索 API 选择、并发限制以及 MCP 服务器配置。
