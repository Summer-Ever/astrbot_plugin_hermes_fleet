"""管理台的读写语义。**不 import astrbot，可脱离本体测试。**

hermes_core 的 WebUI 直接对着那张 `memories` 表写 SQL；fleet 不能这么干——
一 bot 一库，`SELECT * FROM memories` 在 100 个库里是 100 个不同的问题。
这一层把「管理员点开一个账号之后能做什么」收成几个函数，入参是 `bot_id`
和**已经打开的那个 db**，出参是 dataclass / dict。

三条约束写在这里，因为它们是这一层存在的理由：

1. **不跨库。** 所有方法只接受一个 `db`，没有任何「遍历所有账号」的入口。
   跨账号的批量动作交给调用方显式循环，因此也必须在那一层遵守
   `service.probe` 定下的上限（一次最多 20 个，用完立即关库）。
2. **管理视角不做 visibility 过滤。** 管理员要能看到该账号的全部活跃记忆，
   包括 `private` 的。授权是账号级的事（当前还没实现，见 docs/07），
   不是这一层的过滤条件。
   但搜索走存储的 `search_candidates`，那里强制 `can_read`——所以搜索只在
   「某个作用域内、以该 owner 视角」成立。这不是缺陷：它恰好堵住了
   「用搜索绕开归属看别人记忆」这条路。
3. **删除一律软删。** 复用 `merged_into` 指向自身，与 `MemoryService.forget`
   同一套语义（`merged_into != 0` 即不活跃）。物理删除只有 `purge_owner`
   一个入口。模型判断「这条该忘」会判错，管理台手滑也会，必须可回滚。
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from hermes_contracts import (
    EMPTY,
    MAX_MEMORY_CHARS,
    BotId,
    MemoryId,
    MemoryKey,
    MemoryScope,
    ResidentBlock,
    ResidentEntry,
    ResidentScope,
    Visibility,
    clip,
)

from .resident import clean_text

# 一次读多少条未总结消息的上限。消息正文可能很长，管理台只需要看「攒了什么」，
# 不需要全量搬运；真正的总结由 worker 分会话做。
MSG_PER_SESSION = 20


def _now(now: float | None) -> float:
    return float(now) if now is not None else time.time()


# --------------------------------------------------------------------- 作用域


async def list_scopes(db: Any, bot_id: BotId, *, min_active: int = 1) -> list[dict]:
    """该账号下「有活跃记忆」的全部作用域，按条数降序。

    这一个调用就是下钻页左栏的全部数据来源。它存在的意义在存储层的注释里
    写过：没有它，调用方只能自己维护 owner 清单，重启即丢。
    """
    rows = await db.memories.list_scopes(bot_id, min_active=min_active)
    return [
        {
            "scope": str(key.scope),
            "owner_id": key.owner_id or "",
            "group_id": key.group_id or "",
            "count": int(count),
        }
        for key, count in rows
    ]


def key_of(bot_id: BotId, *, scope: str, owner_id: str = "", group_id: str = "") -> MemoryKey:
    """把前端传来的三元组拼成 MemoryKey。

    `MemoryKey.__post_init__` 会挡住「scope=user 但没 owner」这种组合，
    所以这里不做额外校验——让唯一实现去判，别在这里再造一套。
    """
    return MemoryKey(
        bot_id=bot_id,
        scope=MemoryScope(str(scope or "user")),
        owner_id=str(owner_id or EMPTY),
        group_id=str(group_id or EMPTY),
    )


# --------------------------------------------------------------------- 记忆


def to_row(memory: Any) -> dict:
    """给前端的展示结构。字段名与 hermes_core 的列表保持同一套，
    这样两边的肌肉记忆可以复用，页面上也不用两套叫法。"""
    return {
        "id": int(memory.id),
        "scope": str(memory.scope),
        "visibility": str(memory.visibility),
        "owner_id": memory.owner_id or "",
        "group_id": memory.group_id or "",
        "kind": memory.kind or "fact",
        "content": memory.content,
        "ts": float(memory.ts or 0.0),
        "updated_ts": float(memory.updated_ts or 0.0),
        "merged_into": int(memory.merged_into or 0),
        "hits": int(memory.hits or 0),
        "pinned": bool(memory.pinned),
        "chars": len(memory.content or ""),
    }


async def page_memories(
    db: Any,
    catalog: MemoryKey,
    *,
    limit: int = 50,
    after_id: int = 0,
) -> list[dict]:
    """按作用域翻页。游标是 `after_id` 而不是 offset：
    整理任务随时可能合并/新增，offset 分页会漏行或重行。"""
    items = await db.memories.list_active(
        catalog, limit=max(1, min(200, int(limit))), after_id=int(after_id or 0)
    )
    return [to_row(m) for m in items]


async def search_memories(
    db: Any,
    catalog: MemoryKey,
    *,
    query: str,
    limit: int = 50,
) -> list[dict]:
    """在**当前作用域内**做全文检索。

    为什么限定在作用域内：`search_candidates` 强制走 `Memory.can_read`，
    而管理台没有「上帝视角」的读法（那正是待补的鉴权缺口）。
    以 owner 自己的视角查自己的作用域，`can_read` 恒真，结果完整且不会越权。
    """
    q = str(query or "").strip()
    if not q:
        return await page_memories(db, catalog, limit=limit)
    items = await db.memories.search_candidates(
        catalog,
        query=q,
        viewer=catalog.owner_id,
        viewer_group=catalog.group_id,
        limit=max(1, min(200, int(limit))),
    )
    return [to_row(m) for m in items]


async def edit_memory(
    db: Any,
    bot_id: BotId,
    memory_id: int,
    *,
    content: str | None = None,
    kind: str | None = None,
    visibility: str | None = None,
    now: float | None = None,
) -> dict:
    """整条更新。契约层故意不提供「局部 patch」——
    hermes_core 的任意字段 patch 出过「只改 content 却把 owner 清空」的事故。
    这里读出来、改指定字段、整条写回，语义是「后写覆盖」。"""
    existing = await db.memories.get(bot_id, MemoryId(int(memory_id)))
    if existing is None:
        raise KeyError("记忆不存在")
    if not existing.is_active:
        raise ValueError("这条记忆已被合并或删除，先恢复再编辑")

    changes: dict[str, Any] = {"updated_ts": _now(now)}
    if content is not None:
        body = clip(str(content).strip(), MAX_MEMORY_CHARS)
        if not body:
            raise ValueError("记忆内容不能为空")
        changes["content"] = body
    if kind is not None:
        changes["kind"] = str(kind or "fact")
    if visibility is not None:
        changes["visibility"] = Visibility(str(visibility))

    updated = replace(existing, **changes)
    await db.memories.update(updated)
    return to_row(updated)


async def forget_memories(
    db: Any,
    bot_id: BotId,
    memory_ids: list[int],
    *,
    now: float | None = None,
) -> int:
    """软删若干条，返回实际生效条数。

    跳过「已经是墓碑」的 id 而不是报错：管理台一次勾十条，其中一条刚被
    整理任务合并掉是常态，为此让整批失败体验很差。
    """
    at = _now(now)
    done = 0
    for raw in memory_ids or []:
        try:
            mid = int(raw)
        except (TypeError, ValueError):
            continue
        existing = await db.memories.get(bot_id, MemoryId(mid))
        if existing is None or not existing.is_active:
            continue
        await db.memories.update(existing.merged(MemoryId(mid), at=at))
        done += 1
    return done


async def merge_memories(
    db: Any,
    bot_id: BotId,
    source_ids: list[int],
    into_id: int,
    *,
    now: float | None = None,
) -> int:
    """把多条并进一条（软删来源，保留合并产物）。

    与自动整理用的是同一个存储方法，所以「整理只写 0」这个问题在这里
    也能被验证：手动合并一次，`merged_into` 就该出现非零值。
    """
    target_id = int(into_id)
    target = await db.memories.get(bot_id, MemoryId(target_id))
    if target is None:
        raise KeyError("合并目标不存在")
    if not target.is_active:
        raise ValueError("合并目标已被废弃，不能作为合并产物")

    sources = []
    for raw in source_ids or []:
        try:
            mid = int(raw)
        except (TypeError, ValueError):
            continue
        if mid != target_id:
            sources.append(MemoryId(mid))
    if not sources:
        raise ValueError("没有可合并的来源（至少选两条，或去掉目标自身）")
    return await db.memories.merge(sources, MemoryId(target_id), at=_now(now))


# --------------------------------------------------------------------- 常驻


def _scope_of(key: str) -> ResidentScope:
    return ResidentScope.parse(str(key or "private"))


async def resident_of(
    db: Any, bot_id: BotId, owner_id: str, scope_key: str
) -> dict[str, str]:
    """六个固定槽位，缺的补空串。

    存储层的 `get_all` 会跳过空词条（那是为了「注入成本有界」），
    但管理台必须看到六个框——否则「清空过」和「从没设置」在界面上无法区分，
    而这两件事的运维含义完全不同。
    """
    entries = await db.resident.get_all(bot_id, owner_id, _scope_of(scope_key))
    return {b.value: (entries[b].text if b in entries else "") for b in ResidentBlock.ordered()}


async def save_resident(
    db: Any,
    bot_id: BotId,
    owner_id: str,
    scope_key: str,
    block: str,
    text: str,
    *,
    now: float | None = None,
) -> dict:
    """整段覆盖一个词条。空串 = 清空该槽位。

    这里只净化、不做指令腔拒绝：管理台是管理员本人手写，知情且有意，
    「规则：本群禁止发广告」本来就该能写进 rules 词条。拒绝是给模型那条
    路径准备的——它可能把群友的话原样抄进词条。
    """
    try:
        target = ResidentBlock(str(block))
    except ValueError as exc:
        allowed = ", ".join(b.value for b in ResidentBlock.ordered())
        raise ValueError(f"未知词条 {block!r}，可用：{allowed}") from exc
    entry = await db.resident.upsert(ResidentEntry(
        bot_id=bot_id,
        owner_id=owner_id,
        scope=_scope_of(scope_key),
        block=target,
        text=clean_text(text),
        updated_ts=_now(now),
    ))
    return {"block": entry.block.value, "text": entry.text, "updated_ts": entry.updated_ts}


# --------------------------------------------------------------------- 消息


async def pending_messages(
    db: Any, bot_id: BotId, *, limit: int = 50
) -> dict:
    """还没被总结的消息，按会话分组返回。

    这一块的用途很具体：账号「静默」时，管理员第一个要查的问题是
    「消息进来了但没被总结，还是根本没消息」。`stats()` 只给未总结条数，
    给不了「卡在哪个会话」。
    """
    sessions = await db.messages.list_pending_sessions(bot_id, limit=max(1, int(limit)))
    out = []
    total = 0
    for session_id, count, last_ts in sessions:
        total += int(count)
        rows = await db.messages.fetch_unsummarized(
            bot_id, session_id, limit=MSG_PER_SESSION
        )
        out.append({
            "session_id": str(session_id),
            "count": int(count),
            "last_ts": float(last_ts or 0.0),
            "items": [
                {
                    "id": int(m.id or 0),
                    "role": str(m.role),
                    "content": m.content,
                    "ts": float(m.ts or 0.0),
                    "sender_id": m.sender_id or "",
                    "group_id": m.group_id or "",
                }
                for m in rows
            ],
        })
    return {"total": total, "sessions": out}
