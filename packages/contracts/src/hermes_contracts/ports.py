"""端口契约：模块之间**唯一**允许的通信方式。

这个文件是整个并行开发方案的地基。规则只有一条：

    领域模块之间不互相 import 实现，只 import 这里的 Protocol。
    谁来满足 Protocol，由组装层（composition root）在启动时决定。

为什么必须这样：
    假设 `memory` 模块直接 `from hermes_identity import IdentityService`。
    那么 `memory` 的开发和测试就都卡在 `identity` 上——identity 没好，
    memory 连 import 都会失败，两个 agent 只能串行。加了这层 Protocol 之后，
    memory 的测试注入一个 8 行的假实现就能跑，两边真正并行。

所有方法都是 `async def`，这是硬性决策，理由见下方「为什么全异步」。

为什么全异步：
    hermes_core 最严重的问题之一，是所有 SQL 都在事件循环线程上同步执行
    （单连接 + 一把 RLock）。一次 200ms 的检索会阻塞同进程的**所有**机器人的
    消息处理。用 async 签名不是审美，而是让「这段代码会不会阻塞事件循环」
    在函数签名上就可见——同步的 IO 实现没法伪装成 async Protocol。

    例外：`Clock.now()` 和 `recall` 系列是同步的。它们是纯计算，微秒级，
    且不涉及 IO。硬做成 async 只会让调用点到处 `await` 一个不等待的东西。

返回值约定：
- 查不到返回 `None`，**不抛** `NotFound`。让调用方决定「查不到」是不是异常。
  这是 hermes_core 踩过的坑：`get_memory` 抛异常导致批量场景里要包 try。
- 写操作返回落库后的实体（带 id / 时间戳），调用方不需要再查一次。
- 任何失败抛 `hermes_contracts.errors.HermesError` 子类，不抛实现细节异常。
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from .clock import Clock
from .ids import (
    BotId,
    MemoryId,
    MemoryKey,
    OwnerId,
    ResidentBlock,
    ResidentScope,
    SessionId,
    SkillId,
)
from .models import (
    ChatMessage,
    Job,
    LlmResult,
    Memory,
    Message,
    ResidentEntry,
    Skill,
)

# ---------------------------------------------------------------------------
# 存储端口
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """记忆的持久化边界。

    这个接口的**形状**本身就是一个设计结论：`search_candidates` 只返回
    有限候选，不接受「把所有记忆拿出来我打分」的调用方式。

    hermes_core 的做法是分三层 `LIMIT` 最多捞出 9000 行，在 Python 里逐行
    建 gram_set 再打分（`db.py:1557` + `recall.py:622`）。100 个 bot 下，
    单次耗时 100-300ms、内存峰值极高，而且每次都在事件循环线程上跑。
    在这里把方法签名定死，新实现就无法复用那个病根。
    """

    async def add(self, memory: Memory) -> Memory:
        """插入一条记忆，返回带 `id` 的副本。

        实现必须保证：写入前用 `models.clip` 截断 `content`；
        `ts` 为 0 时用注入的 Clock 补当前时间（调用方可以显式指定，用于回填历史）。
        """
        ...

    async def get(self, bot_id: BotId, memory_id: MemoryId) -> Memory | None:
        """按 id 取。已被合并（`merged_into != 0`）的记录也要返回——
        回滚合并链需要读到它；是否过滤由调用方用 `is_active` 决定。"""
        ...

    async def update(self, memory: Memory) -> None:
        """整条更新。

        故意不提供「局部更新若干字段」的方法：hermes_core 的 `update_memory`
        支持任意字段 patch，结果调用方各自拼 kwargs，出现了「只改 content 却把
        owner 清空」的事故。整条更新让并发写变成「后写覆盖」，语义明确。
        """
        ...

    async def merge(
        self,
        sources: Sequence[MemoryId],
        into: MemoryId,
        *,
        at: float,
    ) -> int:
        """把 `sources` 合并进 `into`：软删除（写 `merged_into=into`），返回受影响条数。

        必须是软删除。物理删除后无法回滚，而合并是模型判断，一定有判错的时候。
        实现还要保证：`into` 不能在 `sources` 里（自环），且不产生链的环。
        """
        ...

    async def list_active(
        self,
        key: MemoryKey,
        *,
        limit: int,
        after_id: int = 0,
    ) -> Sequence[Memory]:
        """按归属列出活跃记忆，按 id 升序，游标分页。

        给管理界面和整理任务用，**不要**用于注入路径（那里走 `search_candidates`）。
        `after_id` 是游标而不是 offset：offset 分页在并发写入时会漏行。
        """
        ...

    async def search_candidates(
        self,
        key: MemoryKey,
        *,
        query: str,
        viewer: OwnerId,
        viewer_group: str,
        limit: int,
    ) -> Sequence[Memory]:
        """在**存储内部**完成粗排 top-k，返回 ≤limit 条候选。

        实现要求：
        - 向量召回 top-k + 字面召回 top-k，在库内完成（pgvector / trigram）；
        - 结果必须已按 `Memory.can_read(viewer, viewer_group=viewer_group)` 过滤；
        - **不要**在这里做精排。精排是 `hermes_recall` 的职责，它只接收
          ≤50 条候选。把这一步也塞进存储，会让存储层实现方被迫理解打分数学。
        """
        ...

    async def count_active(
        self,
        bot_id: BotId,
        *,
        scope: str | None = None,
        owner_id: str | None = None,
    ) -> int:
        """计数。整理任务的触发阈值靠它判断，不要用「查出来再 len()」。"""
        ...

    async def purge_owner(self, bot_id: BotId, owner_id: OwnerId) -> int:
        """物理删除某人的全部记忆，返回删除条数。

        多租户下的合规必需项。hermes_core 只有单条删除和按 owner 删除部分表，
        缺级联——用户的 messages / memories / skills / resident / 向量
        需要一次清干净，否则「删号」只是删了个入口。
        """
        ...


@runtime_checkable
class ResidentStore(Protocol):
    """常驻词条的持久化边界。

    与 `MemoryStore` 物理分离（独立表、独立接口），这是有意的结构性隔离：
    后台总结和整理只操作 memories 表，**物理上碰不到** resident。
    hermes_core 靠「SQL 里记得加 WHERE」来保证，漏一次就污染常驻。
    """

    async def get_all(
        self,
        bot_id: BotId,
        owner_id: OwnerId,
        scope: ResidentScope,
    ) -> dict[ResidentBlock, ResidentEntry]:
        """取该 (bot, owner, scope) 下所有非空词条。

        返回 dict 而不是 list：调用方要按 `ResidentBlock.ordered()` 拼注入块，
        用 dict 查一次是 O(1)，用 list 每次都要线性找。
        """
        ...

    async def upsert(self, entry: ResidentEntry) -> ResidentEntry:
        """整段覆盖一个词条。空字符串表示清空该槽位。

        **故意不提供 append**：整段覆盖让「累计膨胀」不可能发生，
        而 append 语义下模型会不断追加，六席最后变成一大段无法维护的文本。
        """
        ...

    async def clear(self, bot_id: BotId, owner_id: OwnerId, scope: ResidentScope) -> int:
        """清空某作用域的全部词条（用户说「忘掉关于我的设定」时用）。"""
        ...


@runtime_checkable
class IdentityStore(Protocol):
    """身份别名的持久化边界。"""

    async def resolve(self, bot_id: BotId, raw_id: str) -> OwnerId:
        """把平台原始 id 解析成规范 id。

        实现要求（照搬 hermes_core 的防环策略）：
        - 最多 `Identity.MAX_HOPS` 跳，超过抛 `InvalidArgument`；
        - 遇到环要能终止，不能死循环；
        - 没有映射时**返回 raw_id 本身**（而不是抛异常或返回空），
          这是「未注册用户也能正常聊天」的关键，且避免每条消息都要先写一次映射。
        """
        ...

    async def link(self, bot_id: BotId, raw_id: str, canonical_id: OwnerId) -> None:
        """建立映射。重复 link 到相同 canonical_id 必须幂等。"""
        ...

    async def unlink(self, bot_id: BotId, raw_id: str) -> bool:
        """删除映射，返回是否存在。"""
        ...


@runtime_checkable
class SkillStore(Protocol):
    """技能存储边界。个人域与系统域的合并视图在这里收敛。

    系统域来自 AstrBot 本体 `data/skills/` 的只读文件，个人域来自库表。
    合并规则（shadow 覆盖）：同名时个人域覆盖系统域，且个人域那条要带
    `origin='hermes'`，让前端能区分「用户改过的」和「厂商自带的」。
    """

    async def list_for(
        self,
        bot_id: BotId,
        *,
        owner_id: OwnerId,
        group_id: str = "",
        limit: int = 50,
    ) -> Sequence[Skill]:
        """按可见性列出技能，个人域优先。注入路径只取前 12 个名字，
        所以实现要保证排序稳定（按 `last_used_ts` 降序、再按 name）。"""
        ...

    async def get_by_name(self, bot_id: BotId, owner_id: OwnerId, name: str) -> Skill | None:
        """按唯一键取。用于 `update_skill` 的「读-改-写」流程。"""
        ...

    async def upsert(self, skill: Skill) -> Skill:
        """按 `(bot_id, owner_id, name)` 唯一键写入。

        唯一索引冲突要翻译成 `Conflict`，不能外泄驱动异常。
        """
        ...

    async def archive(self, bot_id: BotId, skill_id: SkillId) -> bool:
        """归档（软删除）。技能不提供物理删除：模型可能刚刚引用过它。"""
        ...


@runtime_checkable
class MessageStore(Protocol):
    """消息存储边界。

    消息量是记忆量的 100 倍以上（540 万 vs 几万），所以这个接口刻意只有
    四个方法。任何「按条件扫消息」的需求都应该变成后台任务 + 批量游标，
    而不是给这里加查询方法。
    """

    async def append_batch(self, messages: Sequence[Message]) -> int:
        """批量写入，返回写入条数。

        必须是批量：单条 commit 在 20-50 条/秒的峰值下会打满 IO。
        实现要保证幂等（同一批重投不产生重复行，靠 `(bot, session, ts, role, hash)` 去重）。
        """
        ...

    async def fetch_unsummarized(
        self,
        bot_id: BotId,
        session_id: SessionId,
        *,
        limit: int,
        min_ts: float = 0.0,
    ) -> Sequence[Message]:
        """取未总结的消息，按 id 升序。

        `min_ts` 用于「6 小时兜底」：水位之后太久没动静也要总结一次，
        否则一个话说到一半的会话永远不会被记住。
        """
        ...

    async def mark_summarized(self, message_ids: Sequence[int]) -> int:
        """标记已总结。返回实际更新条数（已标记的不重复计数）。

        用「按行标记」而不是「会话水位」：多 worker 并发时，会话级水位会被
        两个 worker 各自推进一次，导致中间的消息永远不总结。
        """
        ...

    async def delete_session(self, bot_id: BotId, session_id: SessionId) -> int:
        """删除一个会话的全部消息（清屏 / 合规删除）。"""
        ...

    async def list_pending_sessions(
        self,
        bot_id: BotId,
        *,
        limit: int = 50,
    ) -> Sequence[tuple[SessionId, int, float]]:
        """列出该 bot 下还有未总结消息的会话：`(会话, 待总结条数, 最后一条的时间)`。

        为什么必须有这个方法：`fetch_unsummarized` 要求调用方**先知道**
        session_id，而后台 worker 的任务恰恰是「找出哪些会话该总结」。
        没有它，worker 只能靠内存里记过的 session 列表——重启即丢，
        多实例下更不成立。

        为什么把 `last_ts` 一起返回：总结的触发条件是「够条数**或**够久没动静」。
        后者需要最后一条消息的时间，否则会话在说完最后一句后再没人说话时，
        那几条消息永远等不到总结（正是最需要记住的那几句）。

        排序：按 `last_ts` 升序。**最久没被处理的先处理**，而不是按会话 id
        或随机顺序——否则活跃会话会一直挤占队列，冷会话永远轮不到。
        """
        ...


@runtime_checkable
class JobQueue(Protocol):
    """任务队列。表即队列，不引入 Redis / RabbitMQ。

    为什么不引入外部 MQ：这套系统的任务量级是每秒个位数到几十，而 MQ 带来的
    运维成本（另一个有状态服务、另一套备份、另一种故障模式）远高于收益。
    Postgres 的 `SELECT ... FOR UPDATE SKIP LOCKED` 在这量级下完全够用，
    而且任务和业务数据在同一个事务边界内——这一点 MQ 做不到。
    """

    async def enqueue(
        self,
        kind: str,
        bot_id: BotId,
        scope_key: str,
        *,
        payload: dict[str, Any] | None = None,
        delay_s: float = 0.0,
    ) -> Job | None:
        """入队。**返回 None 表示被去重吸收**（同 kind + bot + scope 已有在途任务）。

        这是幂等的核心：调用方不需要先查再写，也不需要自己维护水位。
        hermes_core 用「队列 JSON blob + 水位重算」来实现同样的效果，
        代价是每次都要全量重算，且并发时互相覆盖。
        """
        ...

    async def claim(
        self,
        worker_id: str,
        kinds: Sequence[str],
        *,
        visibility_s: float,
        limit: int = 1,
    ) -> Sequence[Job]:
        """领取任务，把 `visible_at` 推后 `visibility_s`。

        实现必须用 `FOR UPDATE SKIP LOCKED`，否则两个 worker 会领到同一条。
        这是一个**必须用并发测试验证**的行为，见 conformance 套件。
        """
        ...

    async def heartbeat(self, job: Job, *, visibility_s: float) -> Job:
        """续期。长任务（整理跑 6 次模型调用）必须定期续期，
        否则会被别的 worker 当成「已崩溃」重复领取。"""
        ...

    async def complete(self, job: Job) -> None:
        """标记完成。幂等：重复调用不报错（worker 崩溃后重试可能重复完成）。"""
        ...

    async def fail(self, job: Job, error: str, *, max_attempts: int) -> Job:
        """标记失败。未超上限 → `failed`（可再领取），超上限 → `dead`（死信）。"""
        ...

    async def stats(self, bot_id: BotId | None = None) -> dict[str, int]:
        """按 state 计数，供 `queue_lag` 指标与后台面板使用。"""
        ...


# ---------------------------------------------------------------------------
# 能力端口（外部依赖的抽象）
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """向量化。单独成端口而不是塞进 LlmProvider，因为它们的失败模式、
    限流配额、成本结构完全不同：embedding 便宜且可批量，chat 贵且串行。"""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化。空输入返回空列表（不要发一次空请求）。"""
        ...

    @property
    def dimensions(self) -> int:
        """维度。建表时用来校验 `vector(N)` 是否匹配，不匹配要启动即失败。"""
        ...


@runtime_checkable
class LlmProvider(Protocol):
    """模型调用。只暴露「一次补全」，不暴露流式 / 工具调用——
    后台总结与整理都是非流式的单轮请求，多暴露一种能力就多一种要测的组合。"""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LlmResult:
        """返回 `LlmResult`（定义在 `models`）。

        `purpose` 是必填的（`summarize` / `consolidate` / `judge`），
        用于分项计量成本与限流：整理任务不该挤占总结的配额。
        失败必须翻译成 `Unavailable` / `RateLimited`，并带上 `retry_after_s`。
        """
        ...


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------


class Deps(Protocol):
    """组装层交给各模块的东西。

    不用全局单例 / 模块级变量：并行开发时最怕「某个模块 import 时就初始化了
    数据库连接」。所有依赖从构造函数进，测试就能换掉其中任意一个。

    这个 Protocol 只是给 `Deps` 形状一个名字，实际组装由各入口自己实现
    （见 `apps/*/wiring.py`）。
    """

    clock: Clock

    def memory(self) -> MemoryStore: ...
    def resident(self) -> ResidentStore: ...
    def identity(self) -> IdentityStore: ...
    def skill(self) -> SkillStore: ...
    def messages(self) -> MessageStore: ...
    def jobs(self) -> JobQueue: ...
    def embedder(self) -> Embedder | None: ...
    def llm(self) -> LlmProvider | None: ...


__all__ = [
    "MemoryStore",
    "ResidentStore",
    "IdentityStore",
    "SkillStore",
    "MessageStore",
    "JobQueue",
    "Embedder",
    "LlmProvider",
    "Deps",
]
