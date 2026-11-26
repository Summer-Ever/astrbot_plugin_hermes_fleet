"""整理 worker：把重复的记忆合并成一条。

## 分工

本地配对（`pairing.py`）只负责**召回候选**，模型负责**判断**是否等价，
`MemoryStore.merge` 负责**软删除**。三者谁都替代不了谁：

- 纯本地阈值决定合并 → 会把「喜欢乌龙茶」和「喜欢龙井」当成重复，永久丢信息；
- 纯模型扫描整个库 → 100 个账号下成本不可控，且每次都在赌上下文长度；
- 物理删除 → 判错了无法回滚，而合并不可能次次判对。

## 为什么不用持久化状态

判断「这个作用域该不该整理」需要记住上次整理时的条数。这里把它放在内存
（`_last`），重启即归零。代价是重启后可能对同一批记忆再问一次模型；
换来的收益是不需要在 `memories` 里塞一张「整理进度」表——那张表一旦和实际
数据不一致（合并失败、回滚、手工改库），就会永久性地跳过某些作用域，
而且没人会发现。

重启成本是「多花一次模型调用」，状态表成本是「静默漏整理」，选前者。

## 三重刹车

整理会写库，必须有上限，否则一次模型抽风能把整库并成十条：

1. 每个作用域每轮最多合并 `max_merges` 条；
2. 每个作用域每轮最多处理 `max_groups` 组；
3. 条数没有明显增长（< `growth`）且未到冷却期时直接跳过——
   模型说「保持分开」时条数不变，没有这道闸就会每轮重复烧钱。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

from hermes_contracts import (
    BotId,
    LlmProvider,
    Memory,
    MemoryId,
    MemoryKey,
    MemoryStore,
    clip,
    MAX_MEMORY_CHARS,
)
from hermes_contracts.models import ChatMessage

from .pairing import DEFAULT_THRESHOLD, MAX_GROUP_SIZE, build_groups
from .prompt import (
    CONSOLIDATE_PROMPT,
    build_listing,
    looks_like_complete_array,
    parse_decisions,
)

# 一个作用域有多少条记忆才值得整理。太低的阈值会让整理频繁打扰模型，
# 而两三条记忆之间的「重复」往往只是表述不同，合并没有收益。
DEFAULT_MIN_ACTIVE = 30


@dataclass
class ConsolidateOutcome:
    scopes_seen: int = 0
    scopes_done: int = 0
    groups: int = 0
    merged: int = 0
    failures: int = 0
    skipped: int = 0
    last_error: str = ""

    def add(self, other: "ConsolidateOutcome") -> "ConsolidateOutcome":
        self.scopes_seen += other.scopes_seen
        self.scopes_done += other.scopes_done
        self.groups += other.groups
        self.merged += other.merged
        self.failures += other.failures
        self.skipped += other.skipped
        if other.last_error:
            self.last_error = other.last_error
        return self

    def __str__(self) -> str:
        if self.scopes_seen == 0:
            return "没有达到整理阈值的作用域"
        s = (f"检查 {self.scopes_done}/{self.scopes_seen} 个作用域，"
             f"合并 {self.merged} 条记忆")
        if self.groups:
            s += f"（{self.groups} 组）"
        if self.skipped:
            s += f"，跳过 {self.skipped} 个（条数未增长）"
        if self.failures:
            s += f"，失败 {self.failures} 个"
        return s


class MemoryConsolidator:
    """按作用域合并重复记忆。只依赖端口，不碰具体存储实现。"""

    def __init__(
        self,
        llm: LlmProvider,
        *,
        now=time.time,
        min_active: int = DEFAULT_MIN_ACTIVE,
        growth: int = 8,
        cooldown: float = 6 * 3600,
        scan_limit: int = 200,
        threshold: float = DEFAULT_THRESHOLD,
        max_groups: int = 8,
        max_group_size: int = MAX_GROUP_SIZE,
        max_merges: int = 20,
        max_tokens: int = 800,
        max_scopes: int = 4,
    ) -> None:
        self.llm = llm
        self._now = now
        self.min_active = max(2, int(min_active))
        self.growth = max(1, int(growth))
        self.cooldown = max(60.0, float(cooldown))
        # 一次只看最近 scan_limit 条：整理的目标是「近期反复总结出的重复」，
        # 把三年前的老记忆一起拉进来只会让配对结果变得无法解释。
        self.scan_limit = max(10, int(scan_limit))
        self.threshold = float(threshold)
        self.max_groups = max(1, int(max_groups))
        self.max_group_size = max(2, int(max_group_size))
        self.max_merges = max(1, int(max_merges))
        self.max_tokens = max(200, int(max_tokens))
        self.max_scopes = max(1, int(max_scopes))
        # (bot, scope, owner, group) -> (上次看到的条数, 上次时间)
        self._last: dict[tuple, tuple[int, float]] = {}

    # ------------------------------------------------------------ 入口

    async def run_bot(
        self,
        bot_id: BotId,
        *,
        memories: MemoryStore,
        force: bool = False,
    ) -> ConsolidateOutcome:
        out = ConsolidateOutcome()
        try:
            scopes = await memories.list_scopes(bot_id, min_active=self.min_active)
        except Exception as exc:
            out.failures += 1
            out.last_error = f"枚举作用域失败：{exc}"
            return out

        out.scopes_seen = len(scopes)
        for key, count in list(scopes)[: self.max_scopes]:
            if not force and not self._should_run(key, count):
                out.skipped += 1
                continue
            try:
                merged, groups = await self.consolidate_scope(key, count=count, memories=memories)
            except Exception as exc:
                # 单个作用域失败不影响其他作用域：某个人的库里有格式异常的
                # 记忆，不该让其余上千人永远得不到整理。
                out.failures += 1
                out.last_error = f"{key.scope}/{key.owner_id or key.group_id}: {exc}"
                # 失败也要落状态，否则下一轮会立刻重试同一个失败点，
                # 变成死循环烧模型。
                self._mark(key, count)
                continue
            out.scopes_done += 1
            out.groups += groups
            out.merged += merged
        return out

    async def consolidate_scope(
        self,
        key: MemoryKey,
        *,
        count: int,
        memories: MemoryStore,
    ) -> tuple[int, int]:
        """整理一个作用域，返回 `(合并条数, 处理的组数)`。"""
        decided = await self._decide(key, count=count, memories=memories)
        if decided is None:
            self._mark(key, count)
            return 0, 0
        by_id, decisions, allowed = decided
        merged = await self._apply(
            decisions, by_id=by_id, allowed=allowed, memories=memories)
        self._mark(key, count)
        return merged, len(decisions)

    async def preview_bot(self, bot_id: BotId, *, memories: MemoryStore) -> dict:
        """只读预览：调模型判断该并哪些，但一条都不写。

        与 `run_bot` 共用 `_decide()`，保证「预览看到的组」＝「执行时会处理
        的组」。不看条数增长与冷却：预览是用户当下主动要求的。
        """
        scopes = await memories.list_scopes(bot_id, min_active=self.min_active)
        plans: list[dict] = []
        for key, count in list(scopes)[: self.max_scopes]:
            raw_scope = getattr(key, "scope", "")
            plan = {
                # 枚举在 3.11+ 的 str() 是「MemoryScope.USER」，页面要原始值。
                "scope": str(getattr(raw_scope, "value", raw_scope)),
                "owner_id": str(key.owner_id or ""),
                "group_id": str(key.group_id or ""),
                "active": int(count),
                "groups": [],
                "error": "",
            }
            try:
                plan["groups"] = await self.preview_scope(
                    key, count=count, memories=memories)
            except Exception as exc:
                plan["error"] = str(exc)
            plans.append(plan)
        return {"scopes": len(scopes), "plans": plans}

    async def preview_scope(
        self,
        key: MemoryKey,
        *,
        count: int,
        memories: MemoryStore,
    ) -> list[dict]:
        """预览一个作用域会怎么合并；每一项是一组「保留谁、并掉谁」。"""
        decided = await self._decide(key, count=count, memories=memories)
        if decided is None:
            return []
        by_id, decisions, allowed = decided
        active = set(allowed) & set(by_id)
        plan: list[dict] = []
        for decision in decisions:
            if len(plan) >= self.max_merges:
                break
            keep = int(decision["keep"])
            if keep not in active:
                continue
            sources = [x for x in decision["merge"] if x in active and x != keep]
            if not sources:
                continue
            plan.append({
                "keep": {"id": keep, "content": by_id[keep].content},
                "merge": [
                    {"id": int(x), "content": by_id[int(x)].content}
                    for x in sources
                ],
                "content": str(decision.get("content") or "").strip(),
            })
            active.difference_update(sources)
        return plan

    async def _decide(
        self,
        key: MemoryKey,
        *,
        count: int,
        memories: MemoryStore,
    ) -> tuple[dict[int, Memory], list[dict], set[int]] | None:
        """共同前半段：拉活跃记忆 → 本地配对 → 问模型 → 校验 id 白名单。

        返回 `None` 表示这个作用域配不出组（记忆太少或都不够相似）。
        真实整理与预览共用这一段，避免两条路径的判断规则漂移。
        """
        items = [m for m in await memories.list_active(key, limit=self.scan_limit) if m.is_active]
        if len(items) < 2:
            return None

        groups = build_groups(
            [m.content for m in items],
            threshold=self.threshold,
            max_group_size=self.max_group_size,
        )
        if not groups:
            return None

        by_id = {int(m.id): m for m in items}
        picked = [
            [(int(items[i].id), clip(items[i].content, 120)) for i in g]
            for g in groups[: self.max_groups]
        ]
        # 允许模型操作的 id 只有「真的列进提示词的那几张」。
        # 用整个作用域当白名单是错的：模型只要写出一个扫描到但没列给它的
        # 编号，就能把一条无关记忆并掉——实测确实发生过（用例抓到）。
        allowed = {mid for group in picked for mid, _ in group}
        result = await self.llm.complete(
            [
                ChatMessage(role="system", content=CONSOLIDATE_PROMPT),
                ChatMessage(role="user", content=build_listing(picked)),
            ],
            purpose="consolidate",
            max_tokens=self.max_tokens,
            temperature=0.1,
        )
        if getattr(result, "is_truncated", False):
            raise RuntimeError("整理输出被截断，本轮放弃（下轮重试）")

        raw = getattr(result, "text", "") or ""
        decisions = parse_decisions(raw)
        if not decisions and raw.strip() and not looks_like_complete_array(raw):
            raise RuntimeError("整理输出不完整（缺少闭合的 JSON 数组），本轮放弃")
        return by_id, decisions, allowed

    # ------------------------------------------------------------ 落库

    async def _apply(
        self,
        decisions: list[dict],
        *,
        by_id: dict[int, Memory],
        allowed: set[int],
        memories: MemoryStore,
    ) -> int:
        """把模型的判断落成合并 + 改写。

        模型写的 id 一律视为不可信：它可能写出没列给它的编号、已经并掉的编号、
        或者同一个编号既 keep 又 merge。这里逐条校验，宁可少合并。
        """
        active = set(allowed) & set(by_id)
        now = self._now()
        merged = 0

        for decision in decisions:
            if merged >= self.max_merges:
                break
            keep = int(decision["keep"])
            if keep not in active:
                continue
            sources = [
                x for x in decision["merge"]
                if x in active and x != keep
            ]
            if not sources:
                continue

            keeper = by_id[keep]
            new_content = str(decision.get("content") or "").strip()
            if new_content and new_content != keeper.content:
                # 只有在合并确实发生时改写表述：改写本身是「压缩多条为一条」，
                # 单独改写一条没有收益，却会丢掉原始措辞。
                try:
                    await memories.update(replace(
                        keeper,
                        content=clip(new_content, MAX_MEMORY_CHARS),
                        updated_ts=now,
                    ))
                except Exception:
                    # 改写失败就放弃这一组：如果继续 merge，被并掉的条目
                    # 已经不活跃，而 keeper 还是旧表述，信息就真的少了。
                    continue

            try:
                done = await memories.merge(
                    [MemoryId(x) for x in sources], MemoryId(keep), at=now)
            except ValueError:
                # 自环或成环：实现层拒绝的正是最危险的两类错，直接跳过。
                continue
            merged += int(done or 0)
            # 只把被并掉的移出活动集；keep 仍然活跃，否则后续决策会误以为
            # 「这条已经不存在」而全部跳过（同一个 keep 被写两次时尤其明显）。
            active.difference_update(sources)
        return merged

    # ------------------------------------------------------------ 状态

    def _sig(self, key: MemoryKey) -> tuple:
        return (str(key.bot_id), str(key.scope), key.owner_id, key.group_id)

    def _mark(self, key: MemoryKey, count: int) -> None:
        self._last[self._sig(key)] = (int(count), self._now())

    def _should_run(self, key: MemoryKey, count: int) -> bool:
        last = self._last.get(self._sig(key))
        if last is None:
            return True
        last_count, last_ts = last
        if self._now() - last_ts >= self.cooldown:
            return True
        return int(count) >= last_count + self.growth
