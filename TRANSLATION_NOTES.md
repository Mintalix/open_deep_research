# 翻译说明（TRANSLATION NOTES）

> 本文档记录本次"英文→简体中文"翻译的全部情况，供查阅。
> 翻译原则：**只翻译，不改逻辑**。所有代码标识符、字典键、LangGraph 节点名、
> 模型 ID、配置取值、占位符（如 `{date}`）、JSON 键名、XML 标签等一律保留英文原样。

## 一、代码文件（就地翻译：注释、docstring、自然语言字符串）

| 文件 | 说明 |
|---|---|
| `src/open_deep_research/deep_researcher.py` | 主图实现（718 行）：全部注释、函数 docstring、报错消息 |
| `src/open_deep_research/prompts.py` | 全部 8 个提示词模板（澄清、研究简报、监督者、研究员、压缩、最终报告、网页摘要等） |
| `src/open_deep_research/configuration.py` | 配置项的 UI 描述文字（`x_oap_ui_config` 的 description 与 label） |
| `src/open_deep_research/state.py` | 状态定义与结构化输出模型的描述 |
| `src/open_deep_research/utils.py` | 工具函数、@tool 描述、日志与报错（925 行） |
| `src/security/auth.py` | 认证处理器的注释、docstring、报错消息 |
| `src/legacy/`（graph.py、multi_agent.py、prompts.py、configuration.py、state.py、utils.py、`__init__.py`） | 旧版实现的全部注释、docstring、提示词与展示文字 |
| `src/legacy/tests/`（conftest.py、run_test.py、test_report_quality.py） | 旧版测试脚本 |
| `tests/`（run_evaluate.py、evaluators.py、prompts.py、pairwise_evaluation.py、supervisor_parallel_evaluation.py、extract_langsmith_data.py） | 评估脚本：评估提示词、字段描述、注释、输出文字 |
| `.env.example` | 注释（环境变量名不动） |

**已通过机器验证**：对全部 21 个改动的 Python 文件做了 AST（抽象语法树）结构比对——
除字符串内容与注释外，代码结构与 HEAD 版本完全一致；全部文件通过 `py_compile` 语法检查。
仅有的 4 处 f-string 内变量语序调整（如控制台打印 `{a}...{b}` 译为中文后自然语序变为
`{b}...{a}`）均为纯显示文字，无功能影响。

**关于 ruff 检查**：项目原版即有 194 个 ruff 提示（print 用法、docstring 风格等，并非零错误
状态）。翻译后新增约 97 个 D415（"docstring 缺结尾标点"）**均为误报**——ruff 不识别中文
全角句号"。"，而中文 docstring 实际都以规范句号结尾；另有 14 个 D401（非祈使语气）因
docstring 变为中文而消失。无需处理。

## 二、文档文件（新建 `*.zh-CN.md` 中文副本，原件未动）

| 中文副本 | 对应原文 |
|---|---|
| `README.zh-CN.md` | `README.md`（项目主文档） |
| `CLAUDE.zh-CN.md` | `CLAUDE.md`（仓库概览） |
| `examples/arxiv.zh-CN.md` | `examples/arxiv.md`（ArXiv 研究示例报告） |
| `examples/pubmed.zh-CN.md` | `examples/pubmed.md`（PubMed 研究示例报告） |
| `examples/inference-market.zh-CN.md` | `examples/inference-market.md`（推理市场分析示例） |
| `examples/inference-market-gpt45.zh-CN.md` | `examples/inference-market-gpt45.md` |
| `src/legacy/legacy.zh-CN.md` | `src/legacy/legacy.md`（旧版实现文档，440 行） |
| `src/legacy/CLAUDE.zh-CN.md` | `src/legacy/CLAUDE.md` |
| `src/legacy/files/vibe_code.zh-CN.md` | `src/legacy/files/vibe_code.md`（研究报告输出样例） |

代码块中的命令/变量/路径原样保留，仅代码块内注释译为中文；所有链接与图片 URL 原样保留。

## 三、有意跳过、未翻译的内容

| 内容 | 原因 |
|---|---|
| `pyproject.toml` | 包元数据（description 等为英文属于发布约定，改动会影响打包元数据） |
| `langgraph.json`、`.github/`（dependabot 与 CI workflow）、`uv.lock` | 纯机器配置，无阅读价值，改动有风险 |
| `LICENSE` | 法律文本，以英文原件为准 |
| 代码中参与 `==` 比较 / `in` 判断 / 字典查找的功能性字符串（如 `token_keywords`、`'prompt is too long'`、LangSmith 实验名、数据集名 `"Deep Research Bench"` 等） | 翻译会破坏运行逻辑，各文件报告中已逐项列明 |
| 结构化输出标签（`<summary>`、`<think>` 等）与 JSON 键名 | 供代码解析 |
| `get_today_str` 的英文日期输出（如 `Thu Sep 11, 2026`） | 由 `%a/%b` 格式符产生，docstring 中的示例须与真实输出一致 |

## 四、跳过的图片（保留原样，待后续处理）

以下图片均为 GitHub 上的外链 URL，在中文副本中已原样保留：

**README.md（2 张）**
1. 完整架构图（1388×298）：`https://github.com/user-attachments/assets/12a2371b-8be2-4219-9b48-90503eb43c69`
2. 界面截图（817×666）：`https://github.com/user-attachments/assets/052f2ed3-c664-4a4f-8ec2-074349dcaa3f`

**src/legacy/legacy.md（8 张）**
1. 工作流总览图：`https://github.com/user-attachments/assets/a171660d-b735-4587-ab2f-cd771f773756`
2. 多智能体架构图：`https://github.com/user-attachments/assets/3c734c3c-57aa-4bc0-85dd-74e2ec2c0880`
3. 输入界面（width=1326）：`https://github.com/user-attachments/assets/dc8f59dd-14b3-4a62-ac18-d2f99c8bbe83`
4. 输入界面（width=1326）：`https://github.com/user-attachments/assets/de264b1b-8ea5-4090-8e72-e1ef1230262f`
5. 反馈界面（width=1326）：`https://github.com/user-attachments/assets/c308e888-4642-4c74-bc78-76576a2da919`
6. 确认界面（width=1480）：`https://github.com/user-attachments/assets/ddeeb33b-fdce-494f-af8b-bd2acc1cef06`
7. 报告生成界面（width=1326）：`https://github.com/user-attachments/assets/74ff01cc-e7ed-47b8-bd0c-4ef615253c46`
8. 最终报告界面（width=1326）：`https://github.com/user-attachments/assets/92d9f7b7-3aea-4025-be99-7fb0d4b47289`

另：`src/legacy/files/vibe_code.md` 中有一行纯文本 "Screenshot 2025-04-26 at 1"（疑似图片占位残留，无 URL），已按原样保留。

### 图片翻译方案（Qwen-Image-Edit 图生图）

已搭建基于 Gitee AI（模力方舟）Qwen-Image-Edit 的图片翻译流程（`scripts/translate_images.py`）：
- **原理**：下载原图 → 连同中文翻译提示词一起发给 Qwen-Image-Edit（纯图生图：仅要求把图内英文
  文字原位替换为中文，版式/配色/图标/构图逐像素保持不变，专有名词保留英文）→ 保存到
  `images/zh-CN/` → 自动把各 `*.zh-CN.md` 中的原图链接替换为本地中文图片相对路径。
- **配置**：`.env`（已被 git 忽略）中的 `BASE_URL`、`API_KEY`、`QWEN_IMAGE_EDIT_MODEL`。
- **运行**：`python scripts/translate_images.py`（支持 `--only <id>`、`--force`、`--list`、
  `--no-md-update`）。脚本内置多种网关参数形态的自动降级重试，仅用标准库、无额外依赖。
- **网络提示**：本机直连 GitHub 图床可能超时。脚本自动重试 3 次，并支持 `https_proxy`
  环境变量走代理；仍不行时，可把原图手动下载保存为 `images/_originals/<id>.png`（该目录已被
  git 忽略），脚本会优先使用本地原图。

## 五、需要知晓的两点运行时影响

1. **提示词已中文化**：`src/open_deep_research/prompts.py`、`src/legacy/prompts.py`、
   `tests/` 下评估提示词等发给大模型的文字均已译为中文。运行智能体时，模型将以中文
   理解指令；最终报告语言仍由提示词中"与用户消息相同语言"的规则决定（用户用中文提问
   则输出中文报告，行为不变）。若需恢复英文原版，执行
   `git checkout -- src/open_deep_research/prompts.py` 即可（所有改动均可通过 git 还原）。
2. **评估脚本中的研究查询已中文化**（如 "Give me a high-level overview of MCP..." → 中文），
   实际运行评估时智能体会以中文查询开展研究；相关断言均不依赖这些字符串，逻辑不受影响。

## 六、术语表（全项目统一）

agent→智能体；supervisor→监督者；deep research→深度研究；query→查询；final report→最终
报告；summary→摘要；sources→来源；prompt→提示词；structured output→结构化输出；crawl→
爬取；human-in-the-loop→人工介入；interrupt→中断；thread→线程；state→状态；node→节点；
background investigation→背景调查；feedback→反馈；evaluation→评估；rubric→评分标准；
judge→评审。专有名词（LangGraph、LangChain、OpenAI、Anthropic、Claude、Tavily、MCP、
token、LLM、API 等）保留英文。
