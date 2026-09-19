# 🔬 Open Deep Research

<img alt="完整架构图" src="https://img.mintalix.com/2026/09/55050167206456d6.png" />

深度研究（deep research）已成为最流行的智能体应用之一。本项目是一个简单、可配置、完全开源的深度研究智能体，可在多种模型提供商、搜索工具和 MCP 服务器上运行。它的性能与许多流行的深度研究智能体不相上下（[参见 Deep Research Bench 排行榜](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)）。

> 本仓库基于 LangChain 的 [open_deep_research](https://github.com/langchain-ai/open_deep_research) 进行中文本地化与兼容性增强。感谢原项目作者及所有贡献者。

📖 **源码教学**：[从读懂源码到独立实现](docs/代码实现教学.md)——逐函数解释状态、模型调用、工具调度、并发研究和报告生成，附完整离线实现、复现练习及源码边界分析。

<img alt="界面截图 2025-07-13" src="https://img.mintalix.com/2026/09/3c069a2f3cbae331.png" />

### 🔥 近期更新

**2025 年 8 月 14 日**：我们推出了关于构建开源深度研究的免费课程，请见[这里](https://academy.langchain.com/courses/deep-research-with-langgraph)（课程仓库见[这里](https://github.com/langchain-ai/deep_research_from_scratch)）。

**2025 年 8 月 7 日**：新增 GPT-5，并更新了包含 GPT-5 结果的 Deep Research Bench 评估。

**2025 年 8 月 2 日**：以 0.4344 的总分在 [Deep Research Bench 排行榜](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)上取得第 6 名。

**2025 年 7 月 30 日**：欢迎阅读我们的[博文](https://rlancemartin.github.io/2025/07/30/bitter_lesson/)，了解从最初的实现到当前版本的演进历程。

**2025 年 7 月 16 日**：在我们的[博客](https://blog.langchain.com/open-deep-research/)中了解更多内容，并观看我们的[视频](https://www.youtube.com/watch?v=agGiWUpxkhg)快速概览。

### 🚀 快速开始

1. 克隆仓库并激活虚拟环境：
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
uv venv
source .venv/bin/activate  # Windows 下：.venv\Scripts\activate
```

2. 安装依赖：
```bash
uv sync
# 或
uv pip install -r pyproject.toml
```

3. 创建 `.env` 文件以自定义环境变量（用于模型选择、搜索工具及其他配置项）：
```bash
cp .env.example .env
```

4. 在本地通过 LangGraph 服务器启动智能体：

```bash
# 安装依赖并启动 LangGraph 服务器
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

这将在浏览器中打开 LangGraph Studio UI。

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

在 `messages` 输入框中提出问题，然后点击 `Submit`。可在 "Manage Assistants"（管理助手）标签页中选择不同的配置。

### ⚙️ 配置项

#### LLM :brain:

Open Deep Research 通过 [init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/) 支持众多 LLM 提供商。它在若干不同任务中使用 LLM。有关更多细节，请参阅 [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) 文件中的下列模型字段。这些配置可通过 LangGraph Studio UI 进行访问。

- **摘要**（默认：`openai:gpt-4.1-mini`）：对搜索 API 的结果进行摘要
- **研究**（默认：`openai:gpt-4.1`）：驱动搜索智能体
- **压缩**（默认：`openai:gpt-4.1`）：压缩研究发现
- **最终报告模型**（默认：`openai:gpt-4.1`）：撰写最终报告

> 注意：所选模型需要支持[结构化输出](https://python.langchain.com/docs/integrations/chat/)和[工具调用](https://python.langchain.com/docs/how_to/tool_calling/)。

> 注意：使用 OpenRouter 请遵循[此指南](https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408)；通过 Ollama 使用本地模型请参见[安装说明](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318)。

##### OpenAI 兼容第三方 API

DeepSeek、通义千问、OpenRouter 等提供 OpenAI 兼容接口时，可将
`OPENAI_API_KEY` 设置为第三方密钥，并将 `OPENAI_BASE_URL` 设置为其
文档指定的 base URL。四个模型配置字段使用
`openai:<第三方模型名>`，例如：

```dotenv
OPENAI_API_KEY=your-provider-key
OPENAI_BASE_URL=https://api.deepseek.com
```

然后将 `research_model`、`compression_model`、`final_report_model` 等配置为
`openai:deepseek-chat`（具体模型名以服务商文档为准）。不要把 API 密钥提交到
Git；若启用 `GET_API_KEYS_FROM_CONFIG=true`，请在 OAP 的 `apiKeys` 配置中填写
`OPENAI_API_KEY`。

要永久修改本地默认模型，请编辑 `src/open_deep_research/configuration.py` 中的
`summarization_model`、`research_model`、`compression_model` 和
`final_report_model` 字段，并将其默认值改为目标模型，例如
`openai:deepseek-chat`。也可以在 LangGraph Studio 的 "Manage Assistants"
中针对单个助手覆盖这些配置。

#### 搜索 API :mag:

Open Deep Research 支持众多搜索工具。默认情况下它使用 [Tavily](https://www.tavily.com/) 搜索 API。它具备完整的 MCP 兼容性，并支持 Anthropic 和 OpenAI 的原生网络搜索。有关更多细节，请参阅 [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) 文件中的 `search_api` 和 `mcp_config` 字段。这些配置可通过 LangGraph Studio UI 进行访问。

#### 其他配置

请参阅 [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) 中的各个字段，了解可用于自定义 Open Deep Research 行为的其他设置。

### 📊 评估

Open Deep Research 已配置为使用 [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) 进行评估。该基准包含 100 个博士级研究任务（50 个英文、50 个中文），由 22 个领域（如科学与技术、商业与金融）的领域专家精心设计，以贴近真实的深度研究需求。它有 2 个评估指标，但排行榜基于 RACE 分数。该指标使用 LLM 评审（Gemini），依据一组指标将研究报告与专家编制的黄金标准报告集进行对比评估。

#### 用法

> 警告：在全部 100 个示例上运行可能花费约 20 至 100 美元，具体取决于所选模型。

数据集可通过[此链接](https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d)在 LangSmith 上获取。要启动评估，请运行以下命令：

```bash
# 在 LangSmith 数据集上运行综合评估
python tests/run_evaluate.py
```

这会生成一个 LangSmith 实验的链接，实验名称为 `YOUR_EXPERIMENT_NAME`。完成后，将结果提取为 JSONL 文件，以便提交给 Deep Research Bench。

```bash
python tests/extract_langsmith_data.py --project-name "YOUR_EXPERIMENT_NAME" --model-name "you-model-name" --dataset-name "deep_research_bench"
```

这会创建符合所需格式的 `tests/expt_results/deep_research_bench_model-name.jsonl` 文件。将生成的 JSONL 文件移动到 Deep Research Bench 仓库的本地克隆中，并按照其[快速入门指南](https://github.com/Ayanami0730/deep_research_bench?tab=readme-ov-file#quick-start)提交评估。

#### 结果

| 名称 | 提交 | 摘要 | 研究 | 压缩 | 总成本 | 总 token 数 | RACE 分数 | 实验 |
|------|--------|---------------|----------|-------------|------------|--------------|------------|------------|
| GPT-5 | [ca3951d](https://github.com/langchain-ai/open_deep_research/pull/168/commits) | openai:gpt-4.1-mini | openai:gpt-5 | openai:gpt-4.1 |  | 204,640,896 | 0.4943 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-613c-4bda-8bde-f64f0422bbf3/compare?selectedSessions=4d5941c8-69ce-4f3d-8b3e-e3c99dfbd4cc&baseline=undefined) |
| 默认配置 | [6532a41](https://github.com/langchain-ai/open_deep_research/commit/6532a4176a93cc9bb2102b3d825dcefa560c85d9) | openai:gpt-4.1-mini | openai:gpt-4.1 | openai:gpt-4.1 | $45.98 | 58,015,332 | 0.4309 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=cf4355d7-6347-47e2-a774-484f290e79bc&baseline=undefined) |
| Claude Sonnet 4 | [f877ea9](https://github.com/langchain-ai/open_deep_research/pull/163/commits/f877ea93641680879c420ea991e998b47aab9bcc) | openai:gpt-4.1-mini | anthropic:claude-sonnet-4-20250514 | openai:gpt-4.1 | $187.09 | 138,917,050 | 0.4401 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=04f6002d-6080-4759-bcf5-9a52e57449ea&baseline=undefined) |
| Deep Research Bench 提交 | [c0a160b](https://github.com/langchain-ai/open_deep_research/commit/c0a160b57a9b5ecd4b8217c3811a14d8eff97f72) | openai:gpt-4.1-nano | openai:gpt-4.1 | openai:gpt-4.1 | $87.83 | 207,005,549 | 0.4344 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=e6647f74-ad2f-4cb9-887e-acb38b5f73c0&baseline=undefined) |

### 🚀 部署与使用

#### LangGraph Studio

按照[快速开始](#-快速开始)在本地启动 LangGraph 服务器，并在 LangGraph Studio 上试用该智能体。

#### 托管部署

您可以轻松部署到 [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options)。

#### Open Agent Platform

Open Agent Platform（OAP）是一个面向非技术用户的 UI，让他们可以构建并配置自己的智能体。OAP 非常适合让用户使用最适合自身需求与待解决问题的 MCP 工具和搜索 API 来配置 Deep Researcher。

我们已将 Open Deep Research 部署到我们的 OAP 公开演示实例。您只需添加自己的 API 密钥，即可亲自试用 Deep Researcher！请到[这里](https://oap.langchain.com)体验。

您也可以部署自己的 OAP 实例，并在其上向您的用户提供您自己的自定义智能体（如 Deep Researcher）：
1. [部署 Open Agent Platform](https://docs.oap.langchain.com/quickstart)
2. [将 Deep Researcher 添加到 OAP](https://docs.oap.langchain.com/setup/agents)

### 📚 原项目与开源协议

本项目基于 [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research) 修改而来，主要进行了中文本地化、第三方模型兼容及其他适配。原项目采用 [MIT License](./LICENSE) 发布，本项目继续遵循 MIT License。使用、复制、修改或再分发本项目时，请保留原项目及本项目的版权声明和许可证文本。代码修改以当前仓库内容为准，原项目的最新功能和文档请以前述官方仓库为准。

### 🔗 相关链接

- [原项目仓库（LangChain）](https://github.com/langchain-ai/open_deep_research)
- [原项目文档与使用说明](https://github.com/langchain-ai/open_deep_research#readme)
- [本项目问题反馈](https://github.com/Mintalix/open_deep_research/issues)

### 🔗 友情链接

- [Linux.do](https://linux.do/) — 一个面向 Linux、开源与技术交流的社区



`src/legacy/` 文件夹包含两个较早的实现，提供了自动化研究的替代方案。它们的性能不及当前实现，但为理解深度研究的不同方法提供了另类思路。

#### 1. 工作流实现（`legacy/graph.py`）
- **规划与执行（Plan-and-Execute）**：带人工介入规划的结构化工作流
- **顺序处理**：逐一创建章节并伴随反思
- **交互式控制**：允许对报告规划进行反馈与审批
- **注重质量**：通过迭代完善来强调准确性

#### 2. 多智能体实现（`legacy/multi_agent.py`）
- **监督者-研究者架构**：协同工作的多智能体系统
- **并行处理**：多个研究者同时工作
- **速度优化**：通过并发加速报告生成
- **MCP 支持**：广泛的 Model Context Protocol 集成
