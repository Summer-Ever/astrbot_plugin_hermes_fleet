"""`MemoryStore` 的内存实现。

它的用途不是「生产环境的轻量替代」，而是**让并行开发真正可能**：

- `memory` 模块的 agent 不用等 Postgres 迁移脚本写完就能开发；
- `api` 模块的 agent 不用起数据库就能测路由；
- `workers` 模块可以在 10ms 内构造「已有 500 条记忆」的场景；
- 一致性套件在这里先跑一遍，等 Postgres 实现就绪时，同一套用例再跑一遍，
  两边行为若不一致，测试会立刻指出是哪一条语义对不上。

这不是玩具：它必须**完整通过** `MemoryStoreConformance`。曾经有过「fake 实现
偷懒，只实现被用到的几个方法」的方案，结果是上层代码在 fake 上全绿、上真库
全红——fake 的价值完全取决于它有多认真。

代价与边界，写清楚免得误用：
- 无持久化，进程退出即丢；
- 检索用字符重叠打分，**不是**语义检索（`search_candidates` 只承诺「粗排」，
  这一点它与 Postgres 实现一致，但排序质量不同）；
- 全量线性扫描，只适合测试与单机小规模，不要拿它跑生产。
"""

from __future__ import annotations

import threading
from typing import Iterable, Sequence

from hermes_contracts import (
    BotId,
    Clock,
    InvalidArgument,
    Memory,
    MemoryId,
    MemoryKey,
    MemoryScope,
    NotFound,
    OwnerId,
    clip,
    MAX_MEMORY_CHARS,
)


def _grams(text: str, n: int = 2) -> Iterable[str]:
    """字符 n-gram。中文没有空格，按字切比按词切可靠（不引分词库）。"""
    text = text.strip()
    if len(text) < n:
        if text:
            yield text
        return
    for i in range(len(text) - n + 1):
        yield text[i : i + n]


def _overlap_score(query: str, content: str) -> float:
    """查询被覆盖的比例，而不是 Jaccard。

    短查询匹配长文本时，Jaccard 会被长度差严重惩罚（「乌龙茶」对一句 40 字的
    记忆，Jaccard 只有 0.05），而实际语义完全命中。用「query 的 gram 有多少
    在 content 里出现」更贴近召回需求。这个取舍和 `recall.py` 的思路一致。
    """
    q = set(_grams(query))
    if not q:
        return 0.0
    c = set(_grams(content))
    if not c:
        return 0.0
    return len(q & c) / len(q)


class InMemoryMemoryStore:
    """线程安全的内存实现。

    用 `threading.Lock` 而不是 `asyncio.Lock`：这些方法体内没有 `await`，
    加锁区间是纯内存操作，同步锁更简单，而且能同时服务多线程调用方
    （并发用例会用线程跑）。
    """

    __slots__ = ("_clock", "_rows", "_seq", "_lock")

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._rows: dict[int, Memory] = {}
        self._seq = 0
        self._lock = threading.Lock()

    # -- 内部 --------------------------------------------------------------

    def _next_id(self) -> int:
        self._seq += 1
        return self._seq

    @staticmethod
    def _in_key(m: Memory, key: MemoryKey) -> bool:
        """精确归属匹配。用于 list_active / count_active。

        检索不走这里——检索的范围是「viewer 能看见什么」，比单个坐标宽。
        """
        if m.bot_id != key.bot_id:
            return False
        if m.scope != key.scope:
            return False
        if key.scope is MemoryScope.USER:
            return m.owner_id == key.owner_id
        if key.scope is MemoryScope.GROUP:
            return m.group_id == key.group_id
        return True

    # -- MemoryStore -------------------------------------------------------

    async def add(self, memory: Memory) -> Memory:
        with self._lock:
            content = clip(memory.content, MAX_MEMORY_CHARS)
            ts = memory.ts or self._clock.now()
            stored = Memory(
                id=MemoryId(self._next_id()),
                bot_id=memory.bot_id,
                content=content,
                scope=memory.scope,
                visibility=memory.visibility,
                owner_id=memory.owner_id,
                group_id=memory.group_id,
                kind=memory.kind,
                source=memory.source,
                ts=ts,
                updated_ts=memory.updated_ts or ts,
                merged_into=memory.merged_into,
                hits=memory.hits,
                importance=memory.importance,
                pinned=memory.pinned,
            )
            self._rows[int(stored.id)] = stored
            return stored

    async def get(self, bot_id: BotId, memory_id: MemoryId) -> Memory | None:
        with self._lock:
            row = self._rows.get(int(memory_id))
        if row is None or row.bot_id != bot_id:
            # bot 不匹配也当「不存在」，不能返回别人 bot 的数据
            return None
        return row

    async def update(self, memory: Memory) -> None:
        with self._lock:
            if int(memory.id) not in self._rows:
                raise NotFound(f"memory {memory.id} 不存在")
            self._rows[int(memory.id)] = memory

    async def merge(
        self,
        sources: Sequence[MemoryId],
        into: MemoryId,
        *,
        at: float,
    ) -> int:
        target = int(into)
        src_ids = [int(s) for s in sources]
        if target in src_ids:
            raise InvalidArgument("合并源不能包含合并目标（会形成自环，回滚时死循环）")

        with self._lock:
            if target not in self._rows:
                raise NotFound(f"合并目标 {target} 不存在")
            affected = 0
            for sid in src_ids:
                row = self._rows.get(sid)
                if row is None:
                    raise NotFound(f"合并源 {sid} 不存在")
                if row.merged_into:  # 已经合并过，幂等跳过
                    continue
                self._rows[sid] = row.merged(target, at=at)
                affected += 1
            return affected

    async def list_active(
        self,
        key: MemoryKey,
        *,
        limit: int,
        after_id: int = 0,
    ) -> Sequence[Memory]:
        with self._lock:
            rows = [
                m
                for m in self._rows.values()
                if m.is_active and int(m.id) > after_id and self._in_key(m, key)
            ]
        rows.sort(key=lambda m: int(m.id))
        return tuple(rows[:limit])

    async def search_candidates(
        self,
        key: MemoryKey,
        *,
        query: str,
        viewer: OwnerId,
        viewer_group: str,
        limit: int,
    ) -> Sequence[Memory]:
        """粗排 top-k。

        两道过滤的顺序很重要：**先按 bot 划边界**，再按 `can_read` 判可见性。
        如果反过来（或漏掉 bot 过滤），就会把别的机器人的记忆混进来——
        hermes_core 的 schema 里没有 bot 维度，这个错误在它上面根本无法表达。
        """
        with self._lock:
            candidates = [
                m
                for m in self._rows.values()
                if m.bot_id == key.bot_id and m.can_read(viewer, viewer_group=viewer_group)
            ]
        scored = [(_overlap_score(query, m.content), m) for m in candidates]
        scored = [(s, m) for s, m in scored if s > 0]
        # 分数相同时用 id 兜底，保证排序稳定（否则同一个查询两次结果不同）
        scored.sort(key=lambda pair: (-pair[0], int(pair[1].id)))
        return tuple(m for _, m in scored[:limit])

    async def count_active(
        self,
        bot_id: BotId,
        *,
        scope: str | None = None,
        owner_id: str | None = None,
    ) -> int:
        with self._lock:
            rows = list(self._rows.values())
        n = 0
        for m in rows:
            if not m.is_active or m.bot_id != bot_id:
                continue
            if scope is not None and m.scope != scope:
                continue
            if owner_id is not None and m.owner_id != owner_id:
                continue
            n += 1
        return n

    async def list_scopes(
        self,
        bot_id: BotId,
        *,
        min_active: int = 1,
    ) -> Sequence[tuple[MemoryKey, int]]:
        with self._lock:
            rows = [m for m in self._rows.values()
                    if m.is_active and m.bot_id == bot_id]
        counts: dict[tuple, list] = {}
        for m in rows:
            k = (str(m.scope), m.owner_id, m.group_id)
            if k not in counts:
                counts[k] = [MemoryKey(
                    bot_id=bot_id, scope=m.scope,
                    owner_id=m.owner_id, group_id=m.group_id,
                ), 0]
            counts[k][1] += 1
        out = [(k, n) for k, n in counts.values() if n >= min_active]
        out.sort(key=lambda kv: (-kv[1], str(kv[0].scope), kv[0].owner_id, kv[0].group_id))
        return tuple(out)

    async def purge_owner(self, bot_id: BotId, owner_id: OwnerId) -> int:
        """物理删除。合规删除要求真删，不是软删除。"""
        with self._lock:
            doomed = [
                mid
                for mid, m in self._rows.items()
                if m.bot_id == bot_id and m.owner_id == owner_id
            ]
            for mid in doomed:
                del self._rows[mid]
            return len(doomed)
