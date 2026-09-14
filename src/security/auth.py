"""LangGraph Studio 的 Supabase 身份验证处理器。"""

import asyncio
import os
from typing import Any, Optional

from langgraph_sdk import Auth
from langgraph_sdk.auth.types import StudioUser
from supabase import Client, create_client

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Optional[Client] = None

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)

# "Auth" 对象是一个容器，LangGraph 会用它来标记我们的身份验证函数
auth = Auth()


# `authenticate` 装饰器告诉 LangGraph 将此函数作为中间件，
# 对每个请求调用。由它决定请求是否被允许
@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """使用 Supabase 检查用户的 JWT 令牌是否有效。"""
    # 确保提供了 Authorization 请求头
    if not authorization:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="缺少 Authorization 请求头"
        )

    # 解析 Authorization 请求头
    try:
        scheme, token = authorization.split()
        assert scheme.lower() == "bearer"
    except (ValueError, AssertionError):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Authorization 请求头格式无效"
        )

    # 确保 Supabase 客户端已初始化
    if not supabase:
        raise Auth.exceptions.HTTPException(
            status_code=500, detail="Supabase 客户端未初始化"
        )

    try:
        # 使用 Supabase 验证 JWT 令牌，借助 asyncio.to_thread 避免阻塞
        # 这将在单独的线程中解码并验证 JWT 令牌
        async def verify_token() -> dict[str, Any]:
            response = await asyncio.to_thread(supabase.auth.get_user, token)
            return response

        response = await verify_token()
        user = response.user

        if not user:
            raise Auth.exceptions.HTTPException(
                status_code=401, detail="令牌无效或未找到用户"
            )

        # 若有效则返回用户信息
        return {
            "identity": user.id,
        }
    except Exception as e:
        # 处理来自 Supabase 的任何错误
        raise Auth.exceptions.HTTPException(
            status_code=401, detail=f"身份验证错误：{str(e)}"
        )


@auth.on.threads.create
@auth.on.threads.create_run
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create.value,
):
    """创建线程时添加所有者。

    此处理器在创建新线程时运行，做两件事：
    1. 在正在创建的线程上设置元数据以跟踪所有权
    2. 返回一个过滤器，确保只有创建者才能访问该线程
    """
    if isinstance(ctx.user, StudioUser):
        return

    # 为正在创建的线程添加所有者元数据
    # 该元数据随线程存储并持久保留
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.threads.read
@auth.on.threads.delete
@auth.on.threads.update
@auth.on.threads.search
async def on_thread_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.read.value,
):
    """仅允许用户读取自己的线程。

    此处理器在读取操作时运行。由于线程已存在，我们无需设置
    元数据 - 只需返回一个过滤器，
    确保用户只能看到自己的线程。
    """
    if isinstance(ctx.user, StudioUser):
        return

    return {"owner": ctx.user.identity}


@auth.on.assistants.create
async def on_assistants_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.create.value,
):
    """创建助手时添加所有者。"""
    if isinstance(ctx.user, StudioUser):
        return

    # 为正在创建的助手添加所有者元数据
    # 该元数据随助手存储并持久保留
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity


@auth.on.assistants.read
@auth.on.assistants.delete
@auth.on.assistants.update
@auth.on.assistants.search
async def on_assistants_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.read.value,
):
    """仅允许用户读取自己的助手。

    此处理器在读取操作时运行。由于助手已存在，我们无需设置
    元数据 - 只需返回一个过滤器，
    确保用户只能看到自己的助手。
    """
    if isinstance(ctx.user, StudioUser):
        return

    return {"owner": ctx.user.identity}


@auth.on.store()
async def authorize_store(ctx: Auth.types.AuthContext, value: dict):
    """仅允许用户访问自己的存储项。"""
    if isinstance(ctx.user, StudioUser):
        return

    # 每个存储项的 "namespace" 字段是一个元组，可以将其理解为该项所在的目录。
    namespace: tuple = value["namespace"]
    assert namespace[0] == ctx.user.identity, "未授权"
