import os
import asyncio
import json
import datetime
import requests
import random 
import concurrent
import hashlib
import aiohttp
import httpx
import time
from typing import List, Optional, Dict, Any, Union, Literal, Annotated, cast
from urllib.parse import unquote
from collections import defaultdict
import itertools

from exa_py import Exa
from linkup import LinkupClient
from tavily import AsyncTavilyClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient as AsyncAzureAISearchClient
from duckduckgo_search import DDGS 
from bs4 import BeautifulSoup
from markdownify import markdownify
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from legacy.configuration import Configuration
from legacy.state import Section
from legacy.prompts import SUMMARIZATION_PROMPT


def get_config_value(value):
    """
    辅助函数,用于处理配置值为字符串、字典或枚举的情况
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    过滤 search_api_config 字典,仅保留指定搜索 API 接受的参数。

    Args:
        search_api (str): 搜索 API 的标识符(例如 "exa"、"tavily")。
        search_api_config (Optional[Dict[str, Any]]): 该搜索 API 的配置字典。

    Returns:
        Dict[str, Any]: 要传给搜索函数的参数字典。
    """
    # 定义各搜索 API 接受的参数
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": ["max_results", "topic"],
        "perplexity": [],  # Perplexity 不接受额外参数
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "linkup": ["depth"],
        "googlesearch": ["max_results"],
    }

    # 获取给定搜索 API 接受的参数列表
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # 如果未提供配置,则返回空字典
    if not search_api_config:
        return {}

    # 过滤配置,仅保留接受的参数
    return {k: v for k, v in search_api_config.items() if k in accepted_params}

def deduplicate_and_format_sources(
    search_response,
    max_tokens_per_source=5000,
    include_raw_content=True,
    deduplication_strategy: Literal["keep_first", "keep_last"] = "keep_first"
):
    """
    接收一组搜索响应并将其格式化为可读字符串。
    将 raw_content 限制在约 max_tokens_per_source 个 token 以内。
 
    Args:
        search_responses: 搜索响应字典的列表,每个字典包含:
            - query: str
            - results: 字典列表,包含以下字段:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool
        deduplication_strategy: 对每个唯一 URL 保留第一条还是最后一条搜索结果
    Returns:
        str: 包含去重后来源的格式化字符串
    """
     # 收集所有结果
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])

    # 按 URL 去重
    if deduplication_strategy == "keep_first":
        unique_sources = {}
        for source in sources_list:
            if source['url'] not in unique_sources:
                unique_sources[source['url']] = source
    elif deduplication_strategy == "keep_last":
        unique_sources = {source['url']: source for source in sources_list}
    else:
        raise ValueError(f"无效的去重策略: {deduplication_strategy}")

    # 格式化输出
    formatted_text = "来自以下来源的内容:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*80}\n"  # 清晰的章节分隔符
        formatted_text += f"来源: {source['title']}\n"
        formatted_text += f"{'-'*80}\n"  # 子章节分隔符
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"来源中最相关的内容: {source['content']}\n===\n"
        if include_raw_content:
            # 按每个 token 约 4 个字符粗略估算
            char_limit = max_tokens_per_source * 4
            # 处理 raw_content 为 None 的情况
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"警告: 来源 {source['url']} 未找到 raw_content")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [已截断]"
            formatted_text += f"来源全文(限制为 {max_tokens_per_source} 个 token): {raw_content}\n\n"
        formatted_text += f"{'='*80}\n\n" # 章节结束分隔符
                
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """ 将章节列表格式化为字符串 """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
章节 {idx}: {section.name}
{'='*60}
描述:
{section.description}
是否需要研究:
{section.research}

内容:
{section.content if section.content else '[尚未撰写]'}

"""
    return formatted_str

@traceable
async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True):
    """
    使用 Tavily API 并发执行网页搜索

    Args:
        search_queries (List[str]): 要处理的搜索查询列表
        max_results (int): 返回结果的最大数量
        topic (Literal["general", "news", "finance"]): 用于筛选结果的主题
        include_raw_content (bool): 结果中是否包含原始内容

    Returns:
            List[dict]: 来自 Tavily API 的搜索响应列表:
                {
                    'query': str,
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # 搜索结果列表
                        {
                            'title': str,            # 网页标题
                            'url': str,              # 结果的 URL
                            'content': str,          # 内容摘要/片段
                            'score': float,          # 相关性得分
                            'raw_content': str|None  # 完整页面内容(如有)
                        },
                        ...
                    ]
                }
    """
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )

    # 并发执行所有搜索
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

@traceable
async def azureaisearch_search_async(search_queries: list[str], max_results: int = 5, topic: str = "general", include_raw_content: bool = True) -> list[dict]:
    """
    使用 Azure AI Search API 并发执行网页搜索。

    Args:
        search_queries (List[str]): 要处理的搜索查询列表
        max_results (int): 每个查询返回结果的最大数量
        topic (str): 搜索的语义主题过滤器。
        include_raw_content (bool)

    Returns:
        List[dict]: 来自 Azure AI Search API 的搜索响应列表,每个查询一条。
    """
    # 配置并创建 Azure Search 客户端
    # 确保所有环境变量已设置
    if not all(var in os.environ for var in ["AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_INDEX_NAME", "AZURE_AI_SEARCH_API_KEY"]):
        raise ValueError("缺少 Azure Search API 所需的环境变量:AZURE_AI_SEARCH_ENDPOINT、AZURE_AI_SEARCH_INDEX_NAME、AZURE_AI_SEARCH_API_KEY")
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    credential = AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY"))

    reranker_key = '@search.reranker_score'

    async with AsyncAzureAISearchClient(endpoint, index_name, credential) as client:
        async def do_search(query: str) -> dict:
            # 搜索查询
            paged = await client.search(
                search_text=query,
                vector_queries=[{
                    "fields": "vector",
                    "kind": "text",
                    "text": query,
                    "exhaustive": True
                }],
                semantic_configuration_name="fraunhofer-rag-semantic-config",
                query_type="semantic",
                select=["url", "title", "chunk", "creationTime", "lastModifiedTime"],
                top=max_results,
            )
            # 异步迭代器,获取所有结果
            items = [doc async for doc in paged]
            # 转换为简单的 Dict 格式
            results = [
                {
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "content": doc.get("chunk"),
                    "score": doc.get(reranker_key),
                    "raw_content": doc.get("chunk") if include_raw_content else None
                }
                for doc in items
            ]
            return {"query": query, "results": results}

        # 并行执行各搜索查询
        tasks = [do_search(q) for q in search_queries]
        return await asyncio.gather(*tasks)


@traceable
def perplexity_search(search_queries):
    """使用 Perplexity API 搜索网页。
    
    Args:
        search_queries (List[SearchQuery]): 要处理的搜索查询列表
  
    Returns:
        List[dict]: 来自 Perplexity API 的搜索响应列表,每个查询一条。每个响应的格式如下:
            {
                'query': str,                    # 原始搜索查询
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # 搜索结果列表
                    {
                        'title': str,            # 搜索结果的标题
                        'url': str,              # 结果的 URL
                        'content': str,          # 内容摘要/片段
                        'score': float,          # 相关性得分
                        'raw_content': str|None  # 完整内容;次级引用时为 None
                    },
                    ...
                ]
            }
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    search_docs = []
    for query in search_queries:

        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "搜索网页并提供带来源的事实性信息。"
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # 状态码异常时抛出异常
        
        # 解析响应
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", ["https://perplexity.ai"])
        
        # 为该查询创建结果列表
        results = []
        
        # 第一条引用获得完整内容
        results.append({
            "title": f"Perplexity 搜索,来源 1",
            "url": citations[0],
            "content": content,
            "raw_content": content,
            "score": 1.0  # 添加 score 以匹配 Tavily 格式
        })
        
        # 添加其余引用,且不重复内容
        for i, citation in enumerate(citations[1:], start=2):
            results.append({
                "title": f"Perplexity 搜索,来源 {i}",
                "url": citation,
                "content": "完整内容请参见主要来源",
                "raw_content": None,
                "score": 0.5  # 次级来源使用较低分数
            })
        
        # 将响应格式化为与 Tavily 一致的结构
        search_docs.append({
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": results
        })
    
    return search_docs

@traceable
async def exa_search(search_queries, max_characters: Optional[int] = None, num_results=5, 
                     include_domains: Optional[List[str]] = None, 
                     exclude_domains: Optional[List[str]] = None,
                     subpages: Optional[int] = None):
    """使用 Exa API 搜索网页。
    
    Args:
        search_queries (List[SearchQuery]): 要处理的搜索查询列表
        max_characters (int, optional): 每个结果的 raw_content 可获取的最大字符数。
                                       如果为 None,text 参数将被设为 True 而非对象。
        num_results (int): 每个查询的搜索结果数量。默认为 5。
        include_domains (List[str], optional): 搜索结果中要包含的域名列表。 
            指定后,仅返回来自这些域名的结果。
        exclude_domains (List[str], optional): 搜索结果中要排除的域名列表。
            不能与 include_domains 同时使用。
        subpages (int, optional): 每个结果要获取的子页面数量。如果为 None,则不获取子页面。
        
    Returns:
        List[dict]: 来自 Exa API 的搜索响应列表,每个查询一条。每个响应的格式如下:
            {
                'query': str,                    # 原始搜索查询
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # 搜索结果列表
                    {
                        'title': str,            # 搜索结果的标题
                        'url': str,              # 结果的 URL
                        'content': str,          # 内容摘要/片段
                        'score': float,          # 相关性得分
                        'raw_content': str|None  # 完整内容;次级引用时为 None
                    },
                    ...
                ]
            }
    """
    # 检查 include_domains 与 exclude_domains 是否未被同时指定
    if include_domains and exclude_domains:
        raise ValueError("不能同时指定 include_domains 和 exclude_domains")
    
    # 初始化 Exa 客户端(API key 应配置在 .env 文件中)
    exa = Exa(api_key = f"{os.getenv('EXA_API_KEY')}")
    
    # 定义处理单个查询的函数
    async def process_query(query):
        # 使用 run_in_executor 以非阻塞方式执行同步的 exa 调用
        loop = asyncio.get_event_loop()
        
        # 为执行器定义包含所有参数的函数
        def exa_search_fn():
            # 构建参数字典
            kwargs = {
                # 若 max_characters 为 None 则将 text 设为 True,否则使用包含 max_characters 的对象
                "text": True if max_characters is None else {"max_characters": max_characters},
                "summary": True,  # 这是 Exa 的一项出色功能,会基于查询为内容生成 AI 摘要
                "num_results": num_results
            }
            
            # 仅在提供可选参数时才添加
            if subpages is not None:
                kwargs["subpages"] = subpages
                
            if include_domains:
                kwargs["include_domains"] = include_domains
            elif exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
                
            return exa.search_and_contents(query, **kwargs)
        
        response = await loop.run_in_executor(None, exa_search_fn)
        
        # 将响应格式化为预期的输出结构
        formatted_results = []
        seen_urls = set()  # 跟踪 URL 以避免重复
        
        # 辅助函数:无论项是 dict 还是对象,都能安全取值
        def get_value(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default) if hasattr(item, key) else default
        
        # 从 SearchResponse 对象中获取结果
        results_list = get_value(response, 'results', [])
        
        # 先处理所有主结果
        for result in results_list:
            # 获取 score,若为 None 或不存在则默认为 0.0
            score = get_value(result, 'score', 0.0)
            
            # 若摘要与文本均可用,则合并为 content
            text_content = get_value(result, 'text', '')
            summary_content = get_value(result, 'summary', '')
            
            content = text_content
            if summary_content:
                if content:
                    content = f"{summary_content}\n\n{content}"
                else:
                    content = summary_content
            
            title = get_value(result, 'title', '')
            url = get_value(result, 'url', '')
            
            # 若该 URL 已出现过则跳过(去除重复条目)
            if url in seen_urls:
                continue
                
            seen_urls.add(url)
            
            # 主结果条目
            result_entry = {
                "title": title,
                "url": url,
                "content": content,
                "score": score,
                "raw_content": text_content
            }
            
            # 将主结果添加到格式化结果中
            formatted_results.append(result_entry)
        
        # 仅在提供了 subpages 参数时才处理子页面
        if subpages is not None:
            for result in results_list:
                subpages_list = get_value(result, 'subpages', [])
                for subpage in subpages_list:
                    # 获取子页面分数
                    subpage_score = get_value(subpage, 'score', 0.0)
                    
                    # 合并摘要与文本作为子页面内容
                    subpage_text = get_value(subpage, 'text', '')
                    subpage_summary = get_value(subpage, 'summary', '')
                    
                    subpage_content = subpage_text
                    if subpage_summary:
                        if subpage_content:
                            subpage_content = f"{subpage_summary}\n\n{subpage_content}"
                        else:
                            subpage_content = subpage_summary
                    
                    subpage_url = get_value(subpage, 'url', '')
                    
                    # 若该 URL 已出现过则跳过
                    if subpage_url in seen_urls:
                        continue
                        
                    seen_urls.add(subpage_url)
                    
                    formatted_results.append({
                        "title": get_value(subpage, 'title', ''),
                        "url": subpage_url,
                        "content": subpage_content,
                        "score": subpage_score,
                        "raw_content": subpage_text
                    })
        
        # 收集可用的图片(仅取自主结果以避免重复)
        images = []
        for result in results_list:
            image = get_value(result, 'image')
            if image and image not in images:  # 避免重复图片
                images.append(image)
                
        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": images,
            "results": formatted_results
        }
    
    # 顺序处理所有查询并加延迟,以遵守速率限制
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # 请求间添加延迟(0.25 秒 = 每秒 4 次请求,远低于每秒 5 次的限制)
            if i > 0:  # 第一次请求不加延迟
                await asyncio.sleep(0.25)
            
            result = await process_query(query)
            search_docs.append(result)
        except Exception as e:
            # 妥善处理异常
            print(f"处理查询 '{query}' 时出错: {str(e)}")
            # 为失败的查询添加占位结果,以保持索引对齐
            search_docs.append({
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e)
            })
            
            # 若遇到速率限制错误,则增加额外延迟
            if "429" in str(e):
                print("已超出速率限制,正在增加额外延迟...")
                await asyncio.sleep(1.0)  # 遇到速率限制时改用更长延迟
    
    return search_docs

@traceable
async def arxiv_search_async(search_queries, load_max_docs=5, get_full_documents=True, load_all_available_meta=True):
    """
    使用 ArxivRetriever 在 arXiv 上并发执行搜索。

    Args:
        search_queries (List[str]): 搜索查询或文章 ID 列表
        load_max_docs (int, optional): 每个查询返回文档的最大数量。默认为 5。
        get_full_documents (bool, optional): 是否获取文档全文。默认为 True。
        load_all_available_meta (bool, optional): 是否加载所有可用元数据。默认为 True。

    Returns:
        List[dict]: 来自 arXiv 的搜索响应列表,每个查询一条。每个响应的格式如下:
            {
                'query': str,                    # 原始搜索查询
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # 搜索结果列表
                    {
                        'title': str,            # 论文标题
                        'url': str,              # 论文的 URL(Entry ID)
                        'content': str,          # 带元数据的格式化摘要
                        'score': float,          # 相关性得分(近似值)
                        'raw_content': str|None  # 论文完整内容(如有)
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # 为每个查询创建检索器
            retriever = ArxivRetriever(
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta
            )
            
            # 在线程池中运行同步检索器
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))
            
            results = []
            # 按顺序分配递减的分数
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # 提取元数据
                metadata = doc.metadata
                
                # 使用 entry_id 作为 URL(即实际的 arxiv 链接)
                url = metadata.get('entry_id', '')
                
                # 用所有有用的元数据格式化内容
                content_parts = []

                # 基本信息
                if 'Summary' in metadata:
                    content_parts.append(f"摘要: {metadata['Summary']}")

                if 'Authors' in metadata:
                    content_parts.append(f"作者: {metadata['Authors']}")

                # 添加发表信息
                published = metadata.get('Published')
                published_str = published.isoformat() if hasattr(published, 'isoformat') else str(published) if published else ''
                if published_str:
                    content_parts.append(f"发表时间: {published_str}")

                # 如有可用则添加额外元数据
                if 'primary_category' in metadata:
                    content_parts.append(f"主分类: {metadata['primary_category']}")

                if 'categories' in metadata and metadata['categories']:
                    content_parts.append(f"分类: {', '.join(metadata['categories'])}")

                if 'comment' in metadata and metadata['comment']:
                    content_parts.append(f"备注: {metadata['comment']}")

                if 'journal_ref' in metadata and metadata['journal_ref']:
                    content_parts.append(f"期刊引用: {metadata['journal_ref']}")

                if 'doi' in metadata and metadata['doi']:
                    content_parts.append(f"DOI: {metadata['doi']}")

                # 从 links 中获取 PDF 链接(如有)
                pdf_link = ""
                if 'links' in metadata and metadata['links']:
                    for link in metadata['links']:
                        if 'pdf' in link:
                            pdf_link = link
                            content_parts.append(f"PDF: {pdf_link}")
                            break

                # 用换行符拼接所有内容部分 
                content = "\n".join(content_parts)
                
                result = {
                    'title': metadata.get('Title', ''),
                    'url': url,  # 使用 entry_id 作为 URL
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.page_content if get_full_documents else None
                }
                results.append(result)
                
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # 妥善处理异常
            print(f"处理 arXiv 查询 '{query}' 时出错: {str(e)}")
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # 顺序处理查询并加延迟,以遵守 arXiv 的速率限制(每 3 秒 1 次请求)
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # 请求间添加延迟(按 ArXiv 的速率限制为 3 秒)
            if i > 0:  # 第一次请求不加延迟
                await asyncio.sleep(3.0)
            
            result = await process_single_query(query)
            search_docs.append(result)
        except Exception as e:
            # 妥善处理异常
            print(f"处理 arXiv 查询 '{query}' 时出错: {str(e)}")
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # 若遇到速率限制错误,则增加额外延迟
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("已超出 ArXiv 速率限制,正在增加额外延迟...")
                await asyncio.sleep(5.0)  # 遇到速率限制时改用更长延迟
    
    return search_docs

@traceable
async def pubmed_search_async(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    """
    使用 PubMedAPIWrapper 在 PubMed 上并发执行搜索。

    Args:
        search_queries (List[str]): 搜索查询列表
        top_k_results (int, optional): 每个查询返回文档的最大数量。默认为 5。
        email (str, optional): PubMed API 使用的邮箱地址。NCBI 要求提供。
        api_key (str, optional): PubMed API 的密钥,用于获得更高的速率限制。
        doc_content_chars_max (int, optional): 文档内容的最大字符数。默认为 4000。

    Returns:
        List[dict]: 来自 PubMed 的搜索响应列表,每个查询一条。每个响应的格式如下:
            {
                'query': str,                    # 原始搜索查询
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # 搜索结果列表
                    {
                        'title': str,            # 论文标题
                        'url': str,              # 论文在 PubMed 上的 URL
                        'content': str,          # 带元数据的格式化摘要
                        'score': float,          # 相关性得分(近似值)
                        'raw_content': str       # 完整摘要内容
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # print(f"处理 PubMed 查询: '{query}'")
            
            # 为该查询创建 PubMed 包装器
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "your_email@example.com",
                api_key=api_key if api_key else ""
            )
            
            # 在线程池中运行同步包装器
            loop = asyncio.get_event_loop()
            
            # 使用 wrapper.lazy_load 而非 load,以获得更好的可见性
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))
            
            print(f"查询 '{query}' 返回了 {len(docs)} 条结果")
            
            results = []
            # 按顺序分配递减的分数
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # 用元数据格式化内容
                content_parts = []
                
                if doc.get('Published'):
                    content_parts.append(f"发表时间: {doc['Published']}")
                
                if doc.get('Copyright Information'):
                    content_parts.append(f"版权信息: {doc['Copyright Information']}")
                
                if doc.get('Summary'):
                    content_parts.append(f"摘要: {doc['Summary']}")
                
                # 根据文章 UID 生成 PubMed URL
                uid = doc.get('uid', '')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""
                
                # 用换行符拼接所有内容部分
                content = "\n".join(content_parts)
                
                result = {
                    'title': doc.get('Title', ''),
                    'url': url,
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('Summary', '')
                }
                results.append(result)
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # 处理异常并提供更详细的信息
            error_msg = f"处理 PubMed 查询 '{query}' 时出错: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())  # 打印完整堆栈以便调试
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # 处理所有查询,并在其间加入合理延迟
    search_docs = []
    
    # 从较小延迟开始,遇到限流则逐步增加
    delay = 1.0  # 从较保守的延迟开始
    
    for i, query in enumerate(search_queries):
        try:
            # 请求间添加延迟
            if i > 0:  # 第一次请求不加延迟
                # print(f"等待 {delay} 秒后进行下一个查询...")
                await asyncio.sleep(delay)
            
            result = await process_single_query(query)
            search_docs.append(result)
            
            # 如果查询成功且返回了结果,可略微缩短延迟(但不低于下限)
            if result.get('results') and len(result['results']) > 0:
                delay = max(0.5, delay * 0.9)  # 不低于 0.5 秒
            
        except Exception as e:
            # 妥善处理异常
            error_msg = f"主循环处理 PubMed 查询 '{query}' 时出错: {str(e)}"
            print(error_msg)
            
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # 若发生异常,则增加下一次查询的延迟
            delay = min(5.0, delay * 1.5)  # 不超过 5 秒
    
    return search_docs

@traceable
async def linkup_search(search_queries, depth: Optional[str] = "standard"):
    """
    使用 Linkup API 并发执行网页搜索。

    Args:
        search_queries (List[SearchQuery]): 要处理的搜索查询列表
        depth (str, optional): "standard"(默认)或 "deep"。更多细节见 https://docs.linkup.so/pages/documentation/get-started/concepts

    Returns:
        List[dict]: 来自 Linkup API 的搜索响应列表,每个查询一条。每个响应的格式如下:
            {
                'results': [            # 搜索结果列表
                    {
                        'title': str,   # 搜索结果的标题
                        'url': str,     # 结果的 URL
                        'content': str, # 内容摘要/片段
                    },
                    ...
                ]
            }
    """
    client = LinkupClient()
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
                client.async_search(
                    query,
                    depth,
                    output_type="searchResults",
                )
            )

    search_results = []
    for response in await asyncio.gather(*search_tasks):
        search_results.append(
            {
                "results": [
                    {"title": result.name, "url": result.url, "content": result.content}
                    for result in response.results
                ],
            }
        )

    return search_results

@traceable
async def google_search_async(search_queries: Union[str, List[str]], max_results: int = 5, include_raw_content: bool = True):
    """
    使用 Google 并发执行网页搜索。
    若设置了相应环境变量则使用 Google Custom Search API,否则回退到网页爬取。

    Args:
        search_queries (List[str]): 要处理的搜索查询列表
        max_results (int): 每个查询返回结果的最大数量
        include_raw_content (bool): 是否获取完整页面内容

    Returns:
        List[dict]: 来自 Google 的搜索响应列表,每个查询一条
    """


    # 从环境变量检查 API 凭据
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    use_api = bool(api_key and cx)
    
    # 处理 search_queries 为单个字符串的情况
    if isinstance(search_queries, str):
        search_queries = [search_queries]
    
    # 定义 user agent 生成器
    def get_useragent():
        """生成随机的 user agent 字符串。"""
        lynx_version = f"Lynx/{random.randint(2, 3)}.{random.randint(8, 9)}.{random.randint(0, 2)}"
        libwww_version = f"libwww-FM/{random.randint(2, 3)}.{random.randint(13, 15)}"
        ssl_mm_version = f"SSL-MM/{random.randint(1, 2)}.{random.randint(3, 5)}"
        openssl_version = f"OpenSSL/{random.randint(1, 3)}.{random.randint(0, 4)}.{random.randint(0, 9)}"
        return f"{lynx_version} {libwww_version} {ssl_mm_version} {openssl_version}"
    
    # 创建用于运行同步操作的执行器
    executor = None if use_api else concurrent.futures.ThreadPoolExecutor(max_workers=5)
    
    # 使用信号量限制并发请求数
    semaphore = asyncio.Semaphore(5 if use_api else 2)
    
    async def search_single_query(query):
        async with semaphore:
            try:
                results = []
                
                # 基于 API 的搜索
                if use_api:
                    # 该 API 每次请求最多返回 10 条结果
                    for start_index in range(1, max_results + 1, 10):
                        # 计算本批次请求的结果数量
                        num = min(10, max_results - (start_index - 1))
                        
                        # 向 Google Custom Search API 发起请求
                        params = {
                            'q': query,
                            'key': api_key,
                            'cx': cx,
                            'start': start_index,
                            'num': num
                        }
                        print(f"正在向 Google API 请求 '{query}' 的 {num} 条结果...")

                        async with aiohttp.ClientSession() as session:
                            async with session.get('https://www.googleapis.com/customsearch/v1', params=params) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    print(f"API 错误: {response.status},{error_text}")
                                    break
                                    
                                data = await response.json()
                                
                                # 处理搜索结果
                                for item in data.get('items', []):
                                    result = {
                                        "title": item.get('title', ''),
                                        "url": item.get('link', ''),
                                        "content": item.get('snippet', ''),
                                        "score": None,
                                        "raw_content": item.get('snippet', '')
                                    }
                                    results.append(result)
                        
                        # 加短暂延迟以遵守 API 配额
                        await asyncio.sleep(0.2)
                        
                        # 如果未取满一页结果,则无需继续请求
                        if not data.get('items') or len(data.get('items', [])) < num:
                            break
                
                # 基于网页爬取的搜索
                else:
                    # 请求间添加延迟
                    await asyncio.sleep(0.5 + random.random() * 1.5)
                    print(f"正在爬取 Google 搜索 '{query}'...")

                    # 定义爬取函数
                    def google_search(query, max_results):
                        try:
                            lang = "en"
                            safe = "active"
                            start = 0
                            fetched_results = 0
                            fetched_links = set()
                            search_results = []
                            
                            while fetched_results < max_results:
                                # 向 Google 发送请求
                                resp = requests.get(
                                    url="https://www.google.com/search",
                                    headers={
                                        "User-Agent": get_useragent(),
                                        "Accept": "*/*"
                                    },
                                    params={
                                        "q": query,
                                        "num": max_results + 2,
                                        "hl": lang,
                                        "start": start,
                                        "safe": safe,
                                    },
                                    cookies = {
                                        'CONSENT': 'PENDING+987',  # 用于绕过同意页面
                                        'SOCS': 'CAESHAgBEhIaAB',
                                    }
                                )
                                resp.raise_for_status()
                                
                                # 解析结果
                                soup = BeautifulSoup(resp.text, "html.parser")
                                result_block = soup.find_all("div", class_="ezO2md")
                                new_results = 0
                                
                                for result in result_block:
                                    link_tag = result.find("a", href=True)
                                    title_tag = link_tag.find("span", class_="CVA68e") if link_tag else None
                                    description_tag = result.find("span", class_="FrIlee")
                                    
                                    if link_tag and title_tag and description_tag:
                                        link = unquote(link_tag["href"].split("&")[0].replace("/url?q=", ""))
                                        
                                        if link in fetched_links:
                                            continue
                                        
                                        fetched_links.add(link)
                                        title = title_tag.text
                                        description = description_tag.text
                                        
                                        # 以与 API 结果相同的格式保存结果
                                        search_results.append({
                                            "title": title,
                                            "url": link,
                                            "content": description,
                                            "score": None,
                                            "raw_content": description
                                        })
                                        
                                        fetched_results += 1
                                        new_results += 1
                                        
                                        if fetched_results >= max_results:
                                            break
                                
                                if new_results == 0:
                                    break
                                    
                                start += 10
                                time.sleep(1)  # 页面间延迟
                            
                            return search_results
                                
                        except Exception as e:
                            print(f"Google 搜索 '{query}' 时出错: {str(e)}")
                            return []
                    
                    # 在线程池中执行搜索
                    loop = asyncio.get_running_loop()
                    search_results = await loop.run_in_executor(
                        executor, 
                        lambda: google_search(query, max_results)
                    )
                    
                    # 处理结果
                    results = search_results
                
                # 如有要求,异步获取完整页面内容(API 与网页爬取均适用)
                if include_raw_content and results:
                    content_semaphore = asyncio.Semaphore(3)
                    
                    async with aiohttp.ClientSession() as session:
                        fetch_tasks = []
                        
                        async def fetch_full_content(result):
                            async with content_semaphore:
                                url = result['url']
                                headers = {
                                    'User-Agent': get_useragent(),
                                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                                }
                                
                                try:
                                    await asyncio.sleep(0.2 + random.random() * 0.6)
                                    async with session.get(url, headers=headers, timeout=10) as response:
                                        if response.status == 200:
                                            # 检查内容类型以处理二进制文件
                                            content_type = response.headers.get('Content-Type', '').lower()
                                            
                                            # 处理 PDF 及其他二进制文件
                                            if 'application/pdf' in content_type or 'application/octet-stream' in content_type:
                                                # 对 PDF,标明内容为二进制且未解析
                                                result['raw_content'] = f"[二进制内容: {content_type}。不支持对此类文件提取内容。]"
                                            else:
                                                try:
                                                    # 尝试以 UTF-8 解码,并用替换符处理非 UTF-8 字符
                                                    html = await response.text(errors='replace')
                                                    soup = BeautifulSoup(html, 'html.parser')
                                                    result['raw_content'] = soup.get_text()
                                                except UnicodeDecodeError as ude:
                                                    # 若仍有解码问题则回退
                                                    result['raw_content'] = f"[无法解码内容: {str(ude)}]"
                                except Exception as e:
                                    print(f"警告: 获取 {url} 内容失败: {str(e)}")
                                    result['raw_content'] = f"[获取内容时出错: {str(e)}]"
                                return result
                        
                        for result in results:
                            fetch_tasks.append(fetch_full_content(result))
                        
                        updated_results = await asyncio.gather(*fetch_tasks)
                        results = updated_results
                        print(f"已获取 {len(results)} 条结果的完整内容")
                
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": results
                }
            except Exception as e:
                print(f"Google 搜索查询 '{query}' 时出错: {str(e)}")
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": []
                }
    
    try:
        # 为所有搜索查询创建任务
        search_tasks = [search_single_query(query) for query in search_queries]
        
        # 并发执行所有搜索
        search_results = await asyncio.gather(*search_tasks)
        
        return search_results
    finally:
        # 仅在执行器已创建时才将其关闭
        if executor:
            executor.shutdown(wait=False)

async def scrape_pages(titles: List[str], urls: List[str]) -> str:
    """
    爬取一组 URL 的内容,并格式化为可读的 markdown 文档。
    
    此函数:
    1. 接收页面标题和 URL 列表
    2. 对每个 URL 发起异步 HTTP 请求
    3. 将 HTML 内容转换为 markdown
    4. 格式化所有内容并附上清晰的来源标注
    
    Args:
        titles (List[str]): 与每个 URL 对应的页面标题列表
        urls (List[str]): 要爬取内容的 URL 列表
        
    Returns:
        str: 格式化字符串,以 markdown 格式包含每个页面的完整内容,
             带有清晰的章节分隔符和来源标注
    """
    
    # 创建异步 HTTP 客户端
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        pages = []
        
        # 获取每个 URL 并转换为 markdown
        for url in urls:
            try:
                # 获取内容
                response = await client.get(url)
                response.raise_for_status()
                
                # 若成功则将 HTML 转换为 markdown
                if response.status_code == 200:
                    # 处理不同的内容类型
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        # 将 HTML 转换为 markdown
                        markdown_content = markdownify(response.text)
                        pages.append(markdown_content)
                    else:
                        # 非 HTML 内容仅说明其内容类型
                        pages.append(f"内容类型: {content_type}(未转换为 markdown)")
                else:
                    pages.append(f"错误: 收到状态码 {response.status_code}")
        
            except Exception as e:
                # 处理获取过程中的任何异常
                pages.append(f"获取 URL 时出错: {str(e)}")
        
        # 创建格式化输出
        formatted_output = f"搜索结果: \n\n"
        
        for i, (title, url, page) in enumerate(zip(titles, urls, pages)):
            formatted_output += f"\n\n--- 来源 {i+1}: {title} ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"完整内容:\n {page}"
            formatted_output += "\n\n" + "-" * 80 + "\n"
        
    return formatted_output

@tool
async def duckduckgo_search(search_queries: List[str]):
    """使用 DuckDuckGo 执行搜索,带重试逻辑以应对速率限制
    
    Args:
        search_queries (List[str]): 要处理的搜索查询列表
        
    Returns:
        str: 格式化的搜索结果字符串
    """
    
    async def process_single_query(query):
        # 在事件循环的线程池中执行同步搜索
        loop = asyncio.get_event_loop()
        
        def perform_search():
            max_retries = 3
            retry_count = 0
            backoff_factor = 2.0
            last_exception = None
            
            while retry_count <= max_retries:
                try:
                    results = []
                    with DDGS() as ddgs:
                        # 稍微更改查询,并在重试间添加延迟
                        if retry_count > 0:
                            # 指数退避的随机延迟
                            delay = backoff_factor ** retry_count + random.random()
                            print(f"查询 '{query}' 第 {retry_count}/{max_retries} 次重试,延迟 {delay:.2f} 秒后执行")
                            time.sleep(delay)
                            
                            # 向查询添加随机元素以绕过缓存/速率限制
                            modifiers = ['about', 'info', 'guide', 'overview', 'details', 'explained']
                            modified_query = f"{query} {random.choice(modifiers)}"
                        else:
                            modified_query = query
                        
                        # 执行搜索
                        ddg_results = list(ddgs.text(modified_query, max_results=5))
                        
                        # 格式化结果
                        for i, result in enumerate(ddg_results):
                            results.append({
                                'title': result.get('title', ''),
                                'url': result.get('href', ''),
                                'content': result.get('body', ''),
                                'score': 1.0 - (i * 0.1),  # 简单的打分机制
                                'raw_content': result.get('body', '')
                            })
                        
                        # 返回成功的结果
                        return {
                            'query': query,
                            'follow_up_questions': None,
                            'answer': None,
                            'images': [],
                            'results': results
                        }
                except Exception as e:
                    # 保存异常并重试
                    last_exception = e
                    retry_count += 1
                    print(f"DuckDuckGo 搜索错误: {str(e)}。正在重试 {retry_count}/{max_retries}")
                    
                    # 若非速率限制错误,则不再重试
                    if "Ratelimit" not in str(e) and retry_count >= 1:
                        print(f"非速率限制错误,停止重试: {str(e)}")
                        break
            
            # 若执行到这里,说明所有重试均已失败
            print(f"查询 '{query}' 的所有重试均已失败: {str(last_exception)}")
            # 返回空结果,但保留查询信息
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(last_exception)
            }
            
        return await loop.run_in_executor(None, perform_search)

    # 处理各查询并在其间加延迟,以降低触发限流的概率
    search_docs = []
    urls = []
    titles = []
    for i, query in enumerate(search_queries):
        # 查询间添加延迟(第一条除外)
        if i > 0:
            delay = 2.0 + random.random() * 2.0  # 2-4 秒的随机延迟
            await asyncio.sleep(delay)
        
        # 处理该查询
        result = await process_single_query(query)
        search_docs.append(result)
        
        # 从结果中安全提取 URL 和标题,并处理结果为空的情况
        if result['results'] and len(result['results']) > 0:
            for res in result['results']:
                if 'url' in res and 'title' in res:
                    urls.append(res['url'])
                    titles.append(res['title'])
    
    # 如果得到了任何有效 URL,则爬取相应页面
    if urls:
        return await scrape_pages(titles, urls)
    else:
        return "未找到有效的搜索结果。请尝试其他搜索查询或使用其他搜索 API。"

TAVILY_SEARCH_DESCRIPTION = (
    "一款为全面、准确、可信的结果而优化的搜索引擎。"
    "适合在需要回答时事相关问题时使用。"
)

@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """
    从 Tavily 搜索 API 获取结果。

    Args:
        queries (List[str]): 搜索查询列表
        max_results (int): 返回结果的最大数量
        topic (Literal['general', 'news', 'finance']): 用于筛选结果的主题

    Returns:
        str: 格式化的搜索结果字符串
    """
    # 使用 tavily_search_async 并设置 include_raw_content=True 以直接获取内容
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True
    )

    # 直接使用已提供的 raw_content 格式化搜索结果
    formatted_output = f"搜索结果: \n\n"
    
    # 按 URL 对结果去重
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    async def noop():
        return None

    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = 30_000
    # TODO: 在所有搜索实现/工具间共享此行为
    if configurable.process_search_results == "summarize":
        if configurable.summarization_model_provider == "anthropic":
            extra_kwargs = {"betas": ["extended-cache-ttl-2025-04-11"]}
        else:
            extra_kwargs = {}

        summarization_model = init_chat_model(
            model=configurable.summarization_model,
            model_provider=configurable.summarization_model_provider,
            max_retries=configurable.max_structured_output_retries,
            **extra_kwargs
        )
        summarization_tasks = [
            noop() if not result.get("raw_content") else summarize_webpage(summarization_model, result['raw_content'][:max_char_to_include])
            for result in unique_results.values()
        ]
        summaries = await asyncio.gather(*summarization_tasks)
        unique_results = {
            url: {'title': result['title'], 'content': result['content'] if summary is None else summary}
            for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
        }
    elif configurable.process_search_results == "split_and_rerank":
        embeddings = init_embeddings("openai:text-embedding-3-small")
        results_by_query = itertools.groupby(unique_results.values(), key=lambda x: x['query'])
        all_retrieved_docs = []
        for query, query_results in results_by_query:
            retrieved_docs = split_and_rerank_search_results(embeddings, query, query_results)
            all_retrieved_docs.extend(retrieved_docs)

        stitched_docs = stitch_documents_by_url(all_retrieved_docs)
        unique_results = {
            doc.metadata['url']: {'title': doc.metadata['title'], 'content': doc.page_content}
            for doc in stitched_docs
        }

    # 格式化去重后的结果
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- 来源 {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"摘要:\n{result['content']}\n\n"
        if result.get('raw_content'):
            formatted_output += f"完整内容:\n{result['raw_content'][:max_char_to_include]}"  # 限制内容大小
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    if unique_results:
        return formatted_output
    else:
        return "未找到有效的搜索结果。请尝试其他搜索查询或使用其他搜索 API。"


@tool
async def azureaisearch_search(queries: List[str], max_results: int = 5, topic: str = "general") -> str:
    """
    从 Azure AI 搜索 API 获取结果。
    
    Args:
        queries (List[str]): 搜索查询列表
        
    Returns:
        str: 格式化的搜索结果字符串
    """
    # 使用 azureaisearch_search_async 并设置 include_raw_content=True 以直接获取内容
    search_results = await azureaisearch_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True
    )

    # 直接使用已提供的 raw_content 格式化搜索结果
    formatted_output = f"搜索结果: \n\n"
    
    # 按 URL 对结果去重
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    
    # 格式化去重后的结果
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- 来源 {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"摘要:\n{result['content']}\n\n"
        if result.get('raw_content'):
            formatted_output += f"完整内容:\n{result['raw_content'][:30000]}"  # 限制内容大小
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    if unique_results:
        return formatted_output
    else:
        return "未找到有效的搜索结果。请尝试其他搜索查询或使用其他搜索 API。"


async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict) -> str:
    """选择并执行相应的搜索 API。
    
    Args:
        search_api: 要使用的搜索 API 名称
        query_list: 要执行的搜索查询列表
        params_to_pass: 传给搜索 API 的参数
        
    Returns:
        包含搜索结果的格式化字符串
        
    Raises:
        ValueError: 如果指定了不支持的搜索 API
    """
    if search_api == "tavily":
        # Tavily 搜索工具,同时用于工作流和智能体
        # 并返回格式化的来源字符串
        return await tavily_search.ainvoke({'queries': query_list, **params_to_pass})
    elif search_api == "duckduckgo":
        # DuckDuckGo 搜索工具,同时用于工作流和智能体
        return await duckduckgo_search.ainvoke({'search_queries': query_list})
    elif search_api == "perplexity":
        search_results = perplexity_search(query_list, **params_to_pass)
    elif search_api == "exa":
        search_results = await exa_search(query_list, **params_to_pass)
    elif search_api == "arxiv":
        search_results = await arxiv_search_async(query_list, **params_to_pass)
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(query_list, **params_to_pass)
    elif search_api == "linkup":
        search_results = await linkup_search(query_list, **params_to_pass)
    elif search_api == "googlesearch":
        search_results = await google_search_async(query_list, **params_to_pass)
    elif search_api == "azureaisearch":
        search_results = await azureaisearch_search_async(query_list, **params_to_pass)
    else:
        raise ValueError(f"不支持的搜索 API: {search_api}")

    return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000, deduplication_strategy="keep_first")


class Summary(BaseModel):
    summary: str
    key_excerpts: list[str]


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """总结网页内容。"""
    try:
        user_input_content = "请总结这篇文章"
        if isinstance(model, ChatAnthropic):
            user_input_content = [{
                "type": "text",
                "text": user_input_content,
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }]

        summary = await model.with_structured_output(Summary).with_retry(stop_after_attempt=2).ainvoke([
            {"role": "system", "content": SUMMARIZATION_PROMPT.format(webpage_content=webpage_content)},
            {"role": "user", "content": user_input_content},
        ])
    except:
        # 回退到原始内容
        return webpage_content

    def format_summary(summary: Summary):
        excerpts_str = "\n".join(f'- {e}' for e in summary.key_excerpts)
        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{excerpts_str}\n</key_excerpts>"""

    return format_summary(summary)


def split_and_rerank_search_results(embeddings: Embeddings, query: str, search_results: list[dict], max_chunks: int = 5):
    # 将网页内容切分为块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, add_start_index=True
    )
    documents = [
        Document(
            page_content=result.get('raw_content') or result['content'],
            metadata={"url": result['url'], "title": result['title']}
        )
        for result in search_results
    ]
    all_splits = text_splitter.split_documents(documents)

    # 索引各块
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)

    # 检索相关块
    retrieved_docs = vector_store.similarity_search(query, k=max_chunks)
    return retrieved_docs


def stitch_documents_by_url(documents: list[Document]) -> list[Document]:
    url_to_docs: defaultdict[str, list[Document]] = defaultdict(list)
    url_to_snippet_hashes: defaultdict[str, set[str]] = defaultdict(set)
    for doc in documents:
        snippet_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
        url = doc.metadata['url']
        # 按内容对片段去重
        if snippet_hash in url_to_snippet_hashes[url]:
            continue

        url_to_docs[url].append(doc)
        url_to_snippet_hashes[url].add(snippet_hash)

    # 将检索到的块按 URL 拼接为单个文档
    stitched_docs = []
    for docs in url_to_docs.values():
        stitched_doc = Document(
            page_content="\n\n".join([f"...{doc.page_content}..." for doc in docs]),
            metadata=cast(Document, docs[0]).metadata
        )
        stitched_docs.append(stitched_doc)

    return stitched_docs


def get_today_str() -> str:
    """以人类可读的格式获取当前日期。"""
    return datetime.datetime.now().strftime("%a %b %-d, %Y")


async def load_mcp_server_config(path: str) -> dict:
    """从文件加载 MCP 服务器配置。"""

    def _load():
        with open(path, "r") as f:
            config = json.load(f)
        return config

    config = await asyncio.to_thread(_load)
    return config