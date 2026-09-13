# Open Deep Research

Open Deep Research 是一个实验性的、完全开源的研究助手，可自动化执行深度研究并就任意主题生成全面的报告。它包含两种实现——一种[工作流](https://langchain-ai.github.io/langgraph/tutorials/workflows/)和一种多智能体架构。你可以通过指定模型、提示词、报告结构和搜索工具，来定制整个研究与写作流程。

#### 工作流

![Open Deep Research 总览](https://img.mintalix.com/2026/09/f1534ece45587240.png)

#### 多智能体

![多智能体研究员](https://img.mintalix.com/2026/09/f4cb52ff27a9c7ce.png)


### 🚀 快速开始

克隆仓库：
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
```

然后编辑 `.env` 文件以自定义环境变量（用于模型选择、搜索工具及其他配置项）：
```bash
cp .env.example .env
```

在本地通过 LangGraph 服务器启动该助手，它会自动在浏览器中打开：

#### Mac

```bash
# 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖并启动 LangGraph 服务器
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

#### Windows / Linux

```powershell
# 安装依赖
pip install -e .
pip install -U "langgraph-cli[inmem]" 

# 启动 LangGraph 服务器
langgraph dev
```

使用以下地址打开 Studio UI：
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

#### 多智能体

(1) 与智能体聊一聊你感兴趣的主题，它就会启动报告生成：

<img alt="输入" src="https://img.mintalix.com/2026/09/68d31c178041d174.png" />

(2) 报告以 markdown 格式生成。

#### 工作流

(1) 提供一个 `Topic`（主题）：

<img alt="输入" src="https://img.mintalix.com/2026/09/b3e4ba9fef44f99c.png" />

(2) 系统会生成报告计划并呈现给用户审阅。

(3) 我们可以传入一个包含反馈内容的字符串（`"..."`），根据反馈重新生成计划。

<img alt="反馈" src="https://img.mintalix.com/2026/09/7de31e9f005858fa.png" />

(4) 或者，我们也可以直接在 Studio 的 JSON 输入框中传入 `true` 来接受该计划。

<img alt="接受" src="https://img.mintalix.com/2026/09/2445913ca0d259aa.png" />

(5) 计划被接受后，系统将开始生成报告的各个章节。

<img alt="报告生成" src="https://img.mintalix.com/2026/09/274d5d1cedcbd624.png" />

报告以 markdown 格式生成。

<img alt="报告" src="https://img.mintalix.com/2026/09/94f26376f56277b6.png" />

### 搜索工具

可用的搜索工具：

* [Tavily API](https://tavily.com/) - 通用网页搜索
* [Perplexity API](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) - 通用网页搜索
* [Exa API](https://exa.ai/) - 面向网页内容的强大神经搜索
* [ArXiv](https://arxiv.org/) - 物理、数学、计算机科学等领域的学术论文
* [PubMed](https://pubmed.ncbi.nlm.nih.gov/) - 来自 MEDLINE、生命科学期刊和在线图书的生物医学文献
* [Linkup API](https://www.linkup.so/) - 通用网页搜索
* [DuckDuckGo API](https://duckduckgo.com/) - 通用网页搜索
* [Google Search API/Scrapper](https://google.com/) - 在[此处](https://programmablesearchengine.google.com/controlpanel/all)创建自定义搜索引擎，并在[此处](https://developers.google.com/custom-search/v1/introduction)获取 API 密钥
* [Microsoft Azure AI Search](https://azure.microsoft.com/en-us/products/ai-services/ai-search) - 基于云的向量数据库解决方案

Open Deep Research 兼容多种不同的 LLM：

* 你可以选择任何[通过 `init_chat_model()` API 集成](https://python.langchain.com/docs/how_to/chat_models_universal_init/)的模型
* 支持的集成完整列表请见[此处](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)

### 使用该软件包

```bash
pip install open-deep-research
```

有关在 Jupyter notebook 中的用法示例，请参见 [src/legacy/graph.ipynb](src/legacy/graph.ipynb) 和 [src/legacy/multi_agent.ipynb](src/legacy/multi_agent.ipynb)：

## Open Deep Research 的各种实现

Open Deep Research 提供三种不同的实现方式，各有其优势：

## 1. 基于图的工作流实现（`src/legacy/graph.py`）

基于图的实现遵循一种结构化的"先规划后执行"（plan-and-execute）工作流：

- **规划阶段**：使用规划器模型分析主题并生成结构化的报告计划
- **人工介入**：在继续执行之前，允许人工对报告计划进行反馈和审批
- **顺序研究流程**：逐个创建章节，并在多轮搜索之间进行反思
- **按章节定制研究**：每个章节都有专属的搜索查询和内容检索
- **支持多种搜索工具**：兼容所有搜索服务提供商（Tavily、Perplexity、Exa、ArXiv、PubMed、Linkup 等）

这种实现提供了更具交互性的体验，并且对报告结构的控制力更强，非常适合对报告质量和准确性要求极高的场景。

你可以通过以下若干参数定制研究助手工作流：

- `report_structure`: 为报告定义自定义结构（默认采用标准研究报告格式）
- `number_of_queries`: 每个章节要生成的搜索查询数量（默认：2）
- `max_search_depth`: 反思与搜索迭代的最大轮数（默认：2）
- `planner_provider`: 规划阶段使用的模型提供商（默认："anthropic"，也可以是[此处](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)所列、通过 `init_chat_model` 支持的集成中的任意提供商）
- `planner_model`: 用于规划的具体模型（默认："claude-3-7-sonnet-latest"）
- `planner_model_kwargs`: 传给 planner_model 的附加参数
- `writer_provider`: 写作阶段使用的模型提供商（默认："anthropic"，也可以是[此处](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)所列、通过 `init_chat_model` 支持的集成中的任意提供商）
- `writer_model`: 用于撰写报告的模型（默认："claude-3-5-sonnet-latest"）
- `writer_model_kwargs`: 传给 writer_model 的附加参数
- `search_api`: 用于网页搜索的 API（默认："tavily"，可选项包括 "perplexity"、"exa"、"arxiv"、"pubmed"、"linkup"）

## 2. 多智能体实现（`src/legacy/multi_agent.py`）

多智能体实现采用"监督者-研究员"架构：

- **监督者智能体**：管理整体研究流程、规划章节并汇总最终报告
- **研究员智能体**：多个相互独立的智能体并行工作，各自负责研究并撰写特定章节
- **并行处理**：所有章节同时开展研究，大幅缩短报告生成时间
- **专用工具设计**：每个智能体都可使用与其角色对应的专属工具（研究员用搜索，监督者用章节规划）
- **搜索与 MCP 支持**：可使用 Tavily/DuckDuckGo 进行网页搜索，使用 MCP 服务器访问本地/外部数据，也可以不使用搜索工具、仅用 MCP 工具运行

这种实现侧重于效率与并行化，非常适合希望更快生成报告、且不需要太多用户直接参与的场景。

你可以通过以下若干参数定制多智能体实现：

- `supervisor_model`: 监督者智能体使用的模型（默认："anthropic:claude-3-5-sonnet-latest"）
- `researcher_model`: 研究员智能体使用的模型（默认："anthropic:claude-3-5-sonnet-latest"）
- `number_of_queries`: 每个章节要生成的搜索查询数量（默认：2）
- `search_api`: 用于网页搜索的 API（默认："tavily"，可选项包括 "duckduckgo"、"none"）
- `ask_for_clarification`: 监督者在开始研究前是否应提出澄清性问题（默认：false）- **重要**：设为 `true` 可为监督者智能体启用 Question（提问）工具
- `mcp_server_config`: MCP 服务器的配置（可选）
- `mcp_prompt`: 关于如何使用 MCP 工具的附加说明（可选）
- `mcp_tools_to_include`: 要包含的特定 MCP 工具（可选）

## MCP（模型上下文协议）支持

多智能体实现（`src/legacy/multi_agent.py`）支持 MCP 服务器，以便将研究能力扩展到网页搜索之外。研究智能体可以在传统搜索工具之外（或干脆取而代之）使用 MCP 工具，从而访问本地文件、数据库、API 及其他数据源。

**注意**：目前 MCP 支持仅存在于多智能体实现（`src/legacy/multi_agent.py`）中，基于工作流的实现（`src/legacy/graph.py`）暂不支持。

### 主要特性

- **工具集成**：MCP 工具可与现有的搜索及章节撰写工具无缝集成
- **研究智能体专用**：只有研究智能体（而非监督者）可以访问 MCP 工具
- **灵活配置**：可单独使用 MCP 工具，也可与网页搜索结合使用
- **禁用默认搜索**：设置 `search_api: "none"` 可完全关闭网页搜索工具
- **自定义提示词**：可添加关于使用 MCP 工具的专门指令

### 文件系统服务器示例

#### SKK

```python
config = {
    "configurable": {
        "search_api": "none",  # 使用 "tavily" 或 "duckduckgo" 可与网页搜索结合使用
        "mcp_server_config": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/path/to/your/files"
                ],
                "transport": "stdio"
            }
        },
        "mcp_prompt": "Step 1: Use the `list_allowed_directories` tool to get the list of allowed directories. Step 2: Use the `read_file` tool to read files in the allowed directory.",
        "mcp_tools_to_include": ["list_allowed_directories", "list_directory", "read_file"]  # 可选：指定要包含哪些工具
    }
}
```

#### Studio

MCP 服务器配置：
```
{
  "filesystem": {
    "command": "npx",
    "args": [
      "-y",
      "@modelcontextprotocol/server-filesystem",
      "/Users/rlm/Desktop/Code/open_deep_research/src/legacy/files"
    ],
    "transport": "stdio"
  }
}
```

MCP 提示词：
```
CRITICAL: You MUST follow this EXACT sequence when using filesystem tools:

1. FIRST: Call `list_allowed_directories` tool to discover allowed directories
2. SECOND: Call `list_directory` tool on a specific directory from step 1 to see available files  
3. THIRD: Call `read_file` tool to read specific files found in step 2

DO NOT call `list_directory` or `read_file` until you have first called `list_allowed_directories`. You must discover the allowed directories before attempting to browse or read files.
```

MCP 工具：
```
list_allowed_directories
list_directory 
read_file
```

下面是一个示例测试主题和后续反馈，你可以提供它们来引用所包含的文件：

主题：
```
I want an overview of vibe coding
```

对研究智能体所提问题的后续回复：

```
I just want a single section report on vibe coding that highlights an interesting / fun example
```

生成的追踪记录：

https://smith.langchain.com/public/d871311a-f288-4885-8f70-440ab557c3cf/r

### 配置选项

- **`mcp_server_config`**: 定义 MCP 服务器配置的字典（参见 [langchain-mcp-adapters 示例](https://github.com/langchain-ai/langchain-mcp-adapters#client-1)）
- **`mcp_prompt`**: 可选的附加指令，会加入研究智能体的提示词中，用于指导 MCP 工具的使用
- **`mcp_tools_to_include`**: 可选的列表，指定要包含的 MCP 工具名称（若未设置，则包含来自所有服务器的全部工具）
- **`search_api`**: 设为 `"none"` 表示仅使用 MCP 工具；保留现有搜索 API 则可将两者结合使用

### 常见用例

- **本地文档**：访问项目文档、代码文件或知识库
- **数据库查询**：连接数据库以检索特定数据
- **API 集成**：访问外部 API 与服务
- **文件分析**：在研究过程中读取并分析本地文件

借助 MCP 集成，研究智能体可以把本地知识和外部数据源纳入研究流程，从而生成更全面、更贴合上下文的报告。

## 搜索 API 配置

并非所有搜索 API 都支持额外的配置参数。以下是支持的：

- **Exa**: `max_characters`, `num_results`, `include_domains`, `exclude_domains`, `subpages`
  - 注意：`include_domains` 与 `exclude_domains` 不能同时使用
  - 当你需要将研究范围缩小到特定的可信来源、确保信息准确性，或研究要求使用指定域名（如学术期刊、政府网站）时尤为实用
  - 提供针对具体查询定制的 AI 生成摘要，便于从搜索结果中提取相关信息
- **ArXiv**: `load_max_docs`, `get_full_documents`, `load_all_available_meta`
- **PubMed**: `top_k_results`, `email`, `api_key`, `doc_content_chars_max`
- **Linkup**: `depth`

Exa 配置示例：
```python
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "exa",
                           "search_api_config": {
                               "num_results": 5,
                               "include_domains": ["nature.com", "sciencedirect.com"]
                           },
                           # 其他配置……
                           }}
```

## 模型使用注意事项

(1) 你可以使用[通过 `init_chat_model()` API 支持](https://python.langchain.com/docs/how_to/chat_models_universal_init/)的模型。支持的集成完整列表请见[此处](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)。

(2) ***工作流的规划器模型和写作模型必须支持结构化输出***：请在[此处](https://python.langchain.com/docs/integrations/chat/)查看你使用的模型是否支持结构化输出。

(3) ***智能体模型必须支持工具调用：*** 请确保模型对工具调用有良好支持；已用 Claude 3.7、o3、o3-mini 和 gpt4.1 完成过测试。参见[此处](https://smith.langchain.com/public/adc5d60c-97ee-4aa0-8b2c-c776fb0d7bd6/d)。

(4) 使用 Groq 时，如果你处于 `on_demand` 服务层级，则存在每分钟 token（TPM）限制：
- `on_demand` 服务层级的限制为 `6000 TPM`
- 若要用 Groq 模型撰写章节，你需要[付费套餐](https://github.com/cline/cline/issues/47#issuecomment-2640992272)

(5) `deepseek-R1` [并不擅长函数调用](https://api-docs.deepseek.com/guides/reasoning_model)，而本助手正是通过函数调用来为报告章节生成结构化输出并进行章节评分的。示例追踪记录见[此处](https://smith.langchain.com/public/07d53997-4a6d-4ea8-9a1f-064a85cd6072/r)。
- 请考虑擅长函数调用的提供商，例如 OpenAI、Anthropic，以及 Groq 的 `llama-3.3-70b-versatile` 等部分开源模型。
- 如果你看到如下错误，很可能是模型无法生成结构化输出所致（参见[追踪记录](https://smith.langchain.com/public/8a6da065-3b8b-4a92-8df7-5468da336cbe/r)）：
```
groq.APIError: Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.
```

(6) 若要配合 OpenRouter 使用，请参照[此处](https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408)。

(7) 关于通过 Ollama 使用本地模型，请参见[此处](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318)。

## 评估系统

Open Deep Research 内置了两套完整的评估系统，用于评估报告的质量与性能：

### 1. 基于 Pytest 的评估系统

一套对开发者友好的测试框架，可在开发与测试周期中提供即时反馈。

#### **特性：**
- **丰富的控制台输出**：格式化表格、进度指示器和颜色标识的结果
- **二元通过/失败测试**：明确的成败判定标准，便于集成到 CI/CD
- **LangSmith 集成**：自动进行实验追踪与日志记录
- **灵活配置**：提供大量 CLI 选项以适配不同测试场景
- **实时反馈**：测试执行过程中实时输出

#### **评估标准：**
系统会从 9 个全面的质量维度对报告进行评估：
- 主题相关性（整体与章节层面）
- 结构与逻辑流畅性
- 引言与结论质量
- 结构元素的规范使用（标题、引用）
- Markdown 格式合规性
- 引用质量与来源标注
- 整体研究深度与准确性

#### **用法：**
```bash
# 以默认设置运行所有智能体
python tests/run_test.py --all

# 使用自定义模型测试特定智能体
python tests/run_test.py --agent multi_agent \
  --supervisor-model "anthropic:claude-3-7-sonnet-latest" \
  --search-api tavily

# 使用 OpenAI o3 模型进行测试
python tests/run_test.py --all \
  --supervisor-model "openai:o3" \
  --researcher-model "openai:o3" \
  --planner-provider "openai" \
  --planner-model "o3" \
  --writer-provider "openai" \
  --writer-model "o3" \
  --eval-model "openai:o3" \
  --search-api "tavily"
```

#### **关键文件：**
- `tests/run_test.py`: 主测试运行器，带功能丰富的 CLI 界面
- `tests/test_report_quality.py`: 核心测试实现
- `tests/conftest.py`: Pytest 配置与 CLI 选项

### 2. LangSmith Evaluate API 系统

一套完整的批量评估系统，专为深入分析与对比研究而设计。

#### **特性：**
- **多维度评分**：四个专用评估器，按 1-5 分制打分
- **加权标准**：细粒度评分，可为不同质量方面自定义权重
- **数据集驱动的评估**：跨多个测试用例批量处理
- **性能优化**：对评估器提示词进行长 TTL 缓存
- **专业化报告**：结构化分析并附带改进建议

#### **评估维度：**

1. **整体质量**（7 项加权标准）：
   - 研究深度与来源质量（20%）
   - 分析严谨性与批判性思维（15%）
   - 结构与组织（20%）
   - 实用价值与可操作性（10%）
   - 平衡性与客观性（15%）
   - 写作质量与清晰度（10%）
   - 专业化呈现（10%）

2. **相关性**：按严格标准逐章分析主题相关性

3. **结构**：评估逻辑流畅性、格式规范与引用实践

4. **事实依据性（Groundedness）**：评估报告与检索到的上下文及来源的契合程度

#### **用法：**
```bash
# 在 LangSmith 数据集上运行综合评估
python tests/evals/run_evaluate.py
```

#### **关键文件：**
- `tests/evals/run_evaluate.py`: 主评估脚本
- `tests/evals/evaluators.py`: 四个专用评估器函数
- `tests/evals/prompts.py`: 针对各维度的详细评估提示词
- `tests/evals/target.py`: 报告生成工作流

### 何时使用哪套系统

**在以下场景使用 Pytest 系统：**
- 开发与调试周期
- CI/CD 流水线集成
- 快速的模型对比实验
- 需要即时反馈的交互式测试
- 生产部署前的质量把关

**在以下场景使用 LangSmith 系统：**
- 跨数据集的全面模型评估
- 对系统性能的研究与分析
- 详细的性能剖析与基准测试
- 不同配置之间的对比研究
- 生产环境监控与质量保障

两套评估系统互为补充，可为不同的使用场景和开发阶段提供全面覆盖。

## 用户体验

### 本地部署

按照[快速开始](#-quickstart)的说明，在本地启动 LangGraph 服务器。

### 托管部署

你可以轻松部署到 [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options)。
