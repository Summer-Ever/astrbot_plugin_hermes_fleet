"""会话总结 worker：把会话原文压成记忆条目。

## 三个刻意的决定

**1. 失败不标记。** 模型调用失败、输出被截断时，消息保持 `summarized=0`，
下一轮会重试。标记了就等于永久丢弃——而这批消息是唯一的原始素材。

**2. 解析出空数组也要标记。** 否则「这段对话确实没什么可记」会变成
每轮都重跑一次模型调用，永远烧钱，且永远不收敛。

**3. 不做并发，串行跑。** 100 个账号 × 上千会话的场景下，把并发开到
10 看起来更划算，但代价是撞限流：provider 侧的 429 会被误判成「模型坏了」，
进而触发降级链，最后所有账号都在用最差的模型。串行 + 每轮限量，
慢但可预测。要提速应该加 worker 实例（各自的 provider 配额），
而不是在单进程里堆并发。

## 归属判定是这里最容易出错的地方

总结模型会输出一个 `owner` 字段，而它会写错：写昵称、写成人名、
或者在群里把 A 的话归给 B。所以规则是：

- 私聊（本批只有一个发言人）→ **强制归给那个人**，忽略模型写的 owner。
- 群聊 → 模型写的 owner 必须在**本批发言人集合**里，否则降级为群公共记忆。
- 既没有可归属的人、也没有群 id → 丢弃。宁可少记一条，也不要写出一行
  谁都读不到的孤儿记忆。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from hermes_contracts import (
    BotId,
    IdentityStore,
    LlmProvider,
    Memory,
    MemoryId,
    MemoryKey,
    MemoryScope,
    MemoryStore,
    MessageStore,
    OwnerId,
    SessionId,
    Visibility,
    clip,
    MAX_MEMORY_CHARS,
)
from hermes_contracts.models import ChatMessage

from .prompt import (
    BATCH_LIMIT,
    SUMMARY_PROMPT,
    build_transcript,
    looks_like_complete_array,
    parse_memories,
)


@dataclass
class SummaryOutcome:
    """一轮总结的结果。字段刻意做成可累加的，方便跨账号汇总。"""

    sessions_seen: int = 0
    sessions_done: int = 0
    memories: int = 0
    messages: int = 0
    failures: int = 0
    skipped: int = 0
    last_error: str = ""

    def add(self, other: "SummaryOutcome") -> "SummaryOutcome":
        self.sessions_seen += other.sessions_seen
        self.sessions_done += other.sessions_done
        self.memories += other.memories
        self.messages += other.messages
        self.failures += other.failures
        self.skipped += other.skipped
        if other.last_error:
            self.last_error = other.last_error
        return self

    def __str__(self) -> str:
        if self.sessions_seen == 0:
            return "没有待总结的会话"
        s = (f"处理 {self.sessions_done}/{self.sessions_seen} 个会话，"
             f"新增记忆 {self.memories} 条，标记 {self.messages} 条消息")
        if self.skipped:
            s += f"，跳过 {self.skipped} 个（条数不够且未超时）"
        if self.failures:
            s += f"，失败 {self.failures} 个（下轮重试）"
        return s


class SessionSummarizer:
    """按会话总结。只依赖端口，不碰具体存储实现。"""

    def __init__(
        self,
        llm: LlmProvider,
        *,
        now=time.time,
        batch_limit: int = BATCH_LIMIT,
        min_messages: int = 8,
        idle_seconds: float = 6 * 3600,
        max_sessions: int = 8,
        max_tokens: int = 900,
    ) -> None:
        self.llm = llm
        self._now = now
        self.batch_limit = max(2, int(batch_limit))
        # 够条数才总结：一两句话就调一次模型，成本随消息量线性涨，
        # 而产出几乎全是噪声。
        self.min_messages = max(1, int(min_messages))
        # 兜底：说完最后一句就没人再说话了，那几句恰恰最需要记住。
        # 只按条数触发的话，这类会话永远等不到总结。
        self.idle_seconds = max(60.0, float(idle_seconds))
        # 每轮最多处理几个会话。一轮跑太久会让「本轮已处理」的进度
        # 迟迟不落盘，重启后重复劳动。
        self.max_sessions = max(1, int(max_sessions))
        self.max_tokens = max(200, int(max_tokens))

    # ------------------------------------------------------------ 入口

    async def run_bot(
        self,
        bot_id: BotId,
        *,
        messages: MessageStore,
        memories: MemoryStore,
        identities: IdentityStore,
        force: bool = False,
    ) -> SummaryOutcome:
        """处理一个账号下的一批待总结会话。

        `force=True` 是「立刻总结」用的：用户已经明确要求，就不再看条数与
        静默时间，有一条算一条（仍然只碰未总结消息，也仍受每轮会话数、
        单批条数上限约束——那是防跑飞，不是触发门槛）。
        """
        out = SummaryOutcome()
        try:
            pending = await messages.list_pending_sessions(
                bot_id, limit=self.max_sessions)
        except Exception as exc:  # 存储层故障不该让整个后台循环挂掉
            out.failures += 1
            out.last_error = f"列出待总结会话失败：{exc}"
            return out

        out.sessions_seen = len(pending)
        for session_id, count, last_ts in pending:
            if not force and not await self._should_summarize(count, last_ts):
                out.skipped += 1
                continue
            try:
                made, marked = await self.summarize_session(
                    bot_id, session_id,
                    messages=messages, memories=memories, identities=identities,
                )
            except Exception as exc:
                # 单个会话失败不能影响其他会话：一个格式异常的会话
                # 不该让 99 个正常会话永远不总结。
                out.failures += 1
                out.last_error = f"{session_id}: {exc}"
                continue
            out.sessions_done += 1
            out.memories += made
            out.messages += marked
        return out

    async def _should_summarize(self, count: int, last_ts: float) -> bool:
        if count >= self.min_messages:
            return True
        return (self._now() - float(last_ts or 0)) >= self.idle_seconds

    # ------------------------------------------------------------ 单会话

    async def summarize_session(
        self,
        bot_id: BotId,
        session_id: SessionId,
        *,
        messages: MessageStore,
        memories: MemoryStore,
        identities: IdentityStore,
    ) -> tuple[int, int]:
        """总结一个会话，返回 `(新增记忆数, 标记消息数)`。"""
        batch = await messages.fetch_unsummarized(
            bot_id, session_id, limit=self.batch_limit)
        if not batch:
            return 0, 0

        transcript, senders, only_sender = build_transcript(batch)
        if not transcript.strip():
            # 全是空消息（纯图片/表情）：没有可提炼的文本，直接标记，
            # 否则这些消息会永远留在队列里。
            return 0, await messages.mark_summarized([m.id for m in batch])

        group_id = next((m.group_id for m in batch if m.group_id), "")
        # solo 只在**私聊**成立：私聊只有一个发言人，归属无歧义，模型漏写或
        # 写错 owner 都必须归给本人。群里这一批也可能只有一个人说话，
        # 但那时模型写的 owner 仍要照常校验，不能被强行改写。
        solo = only_sender if not group_id else ""

        result = await self.llm.complete(
            [
                ChatMessage(role="system", content=SUMMARY_PROMPT),
                ChatMessage(role="user", content=transcript),
            ],
            purpose="summarize",
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        if getattr(result, "is_truncated", False):
            # 输出被 max_tokens 截断时 JSON 一定是半个。硬解析只会得到
            # 半截内容，甚至把下一条记忆的片段当成内容写进库。
            raise RuntimeError("总结输出被截断，本轮放弃（消息保留待重试）")

        raw_text = getattr(result, "text", "") or ""
        entries = parse_memories(raw_text)
        if not entries and raw_text.strip() and not looks_like_complete_array(raw_text):
            # provider 没暴露 finish_reason 时，截断只能从这里暴露出来：
            # 有内容但连右括号都没有。这时**不能标记**，否则素材永久丢失。
            raise RuntimeError("总结输出不完整（缺少闭合的 JSON 数组），本轮放弃")
        saved = 0
        for entry in entries:
            if await self._persist(entry, bot_id=bot_id, senders=senders, solo=solo,
                                   group_id=group_id, memories=memories,
                                   identities=identities):
                saved += 1

        # 解析出空数组也要标记：见模块 docstring 第 2 条。
        marked = await messages.mark_summarized([m.id for m in batch])
        return saved, marked

    # ------------------------------------------------------------ 落库

    async def _persist(
        self,
        entry: dict,
        *,
        bot_id: BotId,
        senders: set[str],
        solo: str,
        group_id: str,
        memories: MemoryStore,
        identities: IdentityStore,
    ) -> bool:
        content = clip(str(entry.get("content") or "").strip(), MAX_MEMORY_CHARS)
        if not content:
            return False

        raw_owner = str(entry.get("owner") or "").strip()
        if solo:
            # 私聊只有一个发言人，归属无歧义——模型写的 owner 一律不采信。
            raw_owner = solo
        elif raw_owner not in senders:
            # 群里模型写了个不在本批的人：无法确认归属，降级为群公共记忆。
            raw_owner = ""

        owner = OwnerId("")
        if raw_owner:
            owner = await identities.resolve(bot_id, raw_owner)

        if not owner and not group_id:
            # 既没有归属人也没有群 → 这行谁都读不到，不写。
            return False

        scope = MemoryScope.USER if owner else MemoryScope.GROUP
        visibility = Visibility.GROUP if group_id else Visibility.PRIVATE
        if await self._already_saved(memories, bot_id, content, owner, group_id):
            return False

        await memories.add(Memory(
            id=MemoryId(0),
            bot_id=bot_id,
            content=content,
            scope=scope,
            visibility=visibility,
            owner_id=str(owner),
            group_id=group_id,
            kind=str(entry.get("kind") or "fact"),
            source="auto_summary",
            ts=self._now(),
        ))
        return True

    async def _already_saved(
        self,
        memories: MemoryStore,
        bot_id: BotId,
        content: str,
        owner: OwnerId,
        group_id: str,
    ) -> bool:
        """内容完全相同的活跃记忆已存在则跳过。

        为什么需要：进程在 `add` 之后、`mark_summarized` 之前被 kill，
        下一轮会把同一批消息再总结一遍。没有这道检查，用户会看到两条
        一模一样的记忆，且它们会被当成两个不同的"证据"参与排序。

        用检索而不是新增一个「按内容查重」的接口：内容检索本来就要走
        FTS，多一个接口只是一份新的、可能写错的 SQL。
        """
        key = MemoryKey(
            bot_id=bot_id,
            scope=MemoryScope.USER if owner else MemoryScope.GROUP,
            owner_id=str(owner),
            group_id="" if owner else group_id,
        )
        try:
            hits = await memories.search_candidates(
                key,
                query=content[:60],
                viewer=owner or OwnerId(""),
                viewer_group=group_id,
                limit=5,
            )
        except Exception:
            return False  # 查重失败不阻塞写入，重复的代价小于丢失
        return any(h.content.strip() == content for h in hits)
