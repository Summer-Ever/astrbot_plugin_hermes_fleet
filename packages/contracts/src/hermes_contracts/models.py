"""跨模块传递的实体。字段与 `docs/01-开发指南.md` §5.1 的 DDL 一一对应。

为什么实体放在契约层、而不是「每个模块定义自己的 model」：
    多 agent 并行时，`memory` 模块定义一份 `Memory`、`api` 模块又定义一份、
    `workers` 模块再定义一份，三份字段迟早错位（少一个 `visibility`，或者把
    `merged_into` 写成 `merged_to`），而且**拼接时不会报错**——只会表现为
    「某些记忆莫名其妙查不到」。这种 bug 的定位成本极高。

    所以规则是：**实体只有一份定义**，在 `hermes_contracts.models`。
    存储实现负责 <-> 行/文档的映射，业务模块只消费这些不可变对象。

全部 frozen：跨模块传可变对象，等于把「谁改了它」变成不可追的问题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .ids import (
    EMPTY,
    BotId,
    GroupId,
    JobId,
    MemoryId,
    MemoryScope,
    OwnerId,
    ResidentBlock,
    ResidentScope,
    SessionId,
    SkillId,
    Visibility,
)

# ---------------------------------------------------------------------------
# 文本上限
# ---------------------------------------------------------------------------
# 单点定义，所有写入路径共用。hermes_core 把 8000 / 500 散在多个函数里，
# 改一处漏一处，最后某条超长消息把注入块撑爆。
MAX_MESSAGE_CHARS = 8_000
MAX_MEMORY_CHARS = 500
MAX_RESIDENT_BLOCK_CHARS = 2_000
# 「亲密关系与情感表达」是用户自己写的长文本（关系史、称呼、相处方式），
# 默认放宽到 3000；其余词条保持 2000。两者都可以在管理台按「常驻」页
# 里改，所以这里只是**默认值**。
MAX_INTIMACY_BLOCK_CHARS = 3_000
# 契约层的硬上限：配置可以调，但不能调到把一轮上下文塞爆。真正的写入
# 限制由 service 按配置校验；这一层只挡明显失控的输入。
MAX_RESIDENT_HARD_CAP = 10_000


def max_chars_for(block) -> int:
    """词条的默认字数上限：intimacy 3000，其余 2000。"""
    return (MAX_INTIMACY_BLOCK_CHARS if block is ResidentBlock.INTIMACY
            else MAX_RESIDENT_BLOCK_CHARS)
MAX_SKILL_CONTENT_CHARS = 8_000


def clip(text: str, limit: int) -> str:
    """截断而不是拒绝：聊天场景里丢整条消息比丢尾部更糟。

    保留头部，并在尾部标记截断，让下游（模型、用户）知道信息不完整，
    而不是误以为原文就这么短。

    边界：`limit` 小于标记本身长度时直接硬切。否则
    `text[:0] + "…[已截断]"` 会返回一个**比 limit 更长**的字符串，
    而调用方（写库前的长度校验）会因此真的超限。这类「截断函数自己
    没截住」的 bug 在日志里极难看出来。
    """
    if text is None:
        return EMPTY
    if limit <= 0:
        return EMPTY
    if len(text) <= limit:
        return text
    marker = "…[已截断]"
    if limit <= len(marker):
        return text[:limit]
    return text[: limit - len(marker)] + marker


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bot:
    """一个机器人租户。

    `id` 必须是稳定标识（比如部署时写死的 `qq_main`），**不要用平台账号名**：
    账号名会改，改一次全部历史数据就断了关联。
    """

    id: BotId
    platform: str
    display_name: str = EMPTY
    status: str = "active"
    created_at: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True, slots=True)
class Identity:
    """身份别名映射：raw_id -> canonical_id。

    为什么需要：同一个人在不同平台/不同账号下 openid 不同，但用户认为是同一个人。
    解析链最多 5 跳（沿用 hermes_core 的防环策略），超过即视为配置错误。
    """

    bot_id: BotId
    raw_id: str
    canonical_id: OwnerId
    platform: str = EMPTY
    created_at: float = 0.0

    MAX_HOPS = 5


# ---------------------------------------------------------------------------
# 记忆
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Memory:
    """一条记忆。`merged_into` 非 0 表示已被合并（软删除）。

    软删除而不是物理删除的原因：合并错了要能回滚。hermes_core 的 `unmerge`
    靠这个字段把链条反向拆开，是它少数几个「事后可救」的设计。
    """

    id: MemoryId
    bot_id: BotId
    content: str
    scope: MemoryScope = MemoryScope.USER
    visibility: Visibility = Visibility.PRIVATE
    owner_id: str = EMPTY
    group_id: str = EMPTY
    kind: str = "fact"
    source: str = EMPTY
    ts: float = 0.0
    updated_ts: float = 0.0
    merged_into: int = 0
    hits: int = 0
    importance: float = 0.0
    pinned: bool = False

    @property
    def is_active(self) -> bool:
        """所有检索都必须带这个条件。写成属性而不是各处手写 `merged_into == 0`，
        是为了让它能被 grep 到、也被类型检查覆盖。"""
        return self.merged_into == 0

    def can_read(self, viewer: OwnerId, *, viewer_group: str = EMPTY) -> bool:
        """完整的可见性判定：归属 + visibility 两个维度。

        这是**唯一**实现。API 层、worker 层、注入层都必须调它，
        不准各自拼 SQL 条件——拼错一次就是隐私泄露。
        """
        if not self.is_active:
            return False
        if self.visibility is Visibility.PUBLIC:
            return True
        if self.scope is MemoryScope.GLOBAL:
            return True
        if self.owner_id and self.owner_id == viewer:
            return True
        if self.visibility is Visibility.GROUP:
            # 群可见只对「同一群」的读者成立；viewer_group 为空表示私聊上下文
            return bool(viewer_group) and self.group_id == viewer_group
        return False

    def merged(self, into: MemoryId, *, at: float) -> "Memory":
        return replace(self, merged_into=int(into), updated_ts=at)


@dataclass(frozen=True, slots=True)
class RecallHit:
    """检索结果。`score` 已经过最终归一，可直接比较与展示。"""

    memory: Memory
    score: float
    # 各阶段得分，保留下来是为了调参和线上归因；不参与排序
    debug: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 常驻
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResidentEntry:
    """一个常驻词条。按 (bot, owner, scope, block) 唯一定位，整段覆盖。

    `text` 为空表示该槽位未设置，注入时跳过——这是「注入成本有界」的实现方式。
    """

    bot_id: BotId
    owner_id: OwnerId
    scope: ResidentScope
    block: ResidentBlock
    text: str = EMPTY
    updated_ts: float = 0.0

    def __post_init__(self) -> None:
        if len(self.text) > MAX_RESIDENT_HARD_CAP:
            raise ValueError(
                f"常驻词条超长：{self.block}={len(self.text)} > {MAX_RESIDENT_HARD_CAP}"
            )


@dataclass(frozen=True, slots=True)
class InjectPayload:
    """一次注入的成品。

    为什么不直接返回 str：`inject_chars` 是必须采集的指标（它涨了就说明
    常驻在膨胀，会挤占对话预算），而 str 本身没法告诉你成本从哪来。
    """

    text: str
    char_count: int = 0
    resident_blocks: tuple[str, ...] = ()
    skill_names: tuple[str, ...] = ()
    built_at: float = 0.0

    def __post_init__(self) -> None:
        if self.char_count == 0 and self.text:
            # frozen 对象要在 __post_init__ 里用 object.__setattr__ 赋值
            object.__setattr__(self, "char_count", len(self.text))


# ---------------------------------------------------------------------------
# 技能
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Skill:
    """技能。个人域与系统域分开：个人域在库里可写，系统域是本体 `data/skills/`
    下的只读文件。

    改系统技能时落一份同名个人副本（shadow），而不是原地改文件——
    这样本体升级不会覆盖用户的修改，也不会让用户误改坏本体。
    """

    id: SkillId
    bot_id: BotId
    name: str
    content: str
    owner_id: str = EMPTY
    scope: str = "private"
    group_id: str = EMPTY
    trigger: str = EMPTY
    created_ts: float = 0.0
    updated_ts: float = 0.0
    last_used_ts: float = 0.0
    archived: bool = False
    merged_into: int = 0
    origin: str = "hermes"  # hermes | system

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("技能名不能为空")
        if not re.match(r"^[\w\u4e00-\u9fff.-]{1,64}$", self.name):
            # 技能名会变成文件名（系统域扫描）和注入文本，放开字符集会产生
            # 路径穿越和注入文本歧义两个问题
            raise ValueError(f"技能名含非法字符：{self.name!r}")


# ---------------------------------------------------------------------------
# 消息
# ---------------------------------------------------------------------------

MessageRole = Literal["user", "assistant", "system"]


@dataclass(frozen=True, slots=True)
class Message:
    """一条对话消息。`summarized` 是水位标记——注意它按「行」而不是「会话」记，
    这是为了避免「会话级水位」在多 worker 并发时被推进两次。"""

    bot_id: BotId
    session_id: SessionId
    role: MessageRole
    content: str
    ts: float
    id: int = 0
    platform: str = EMPTY
    sender_id: str = EMPTY
    sender_name: str = EMPTY
    group_id: str = EMPTY
    summarized: bool = False

    def clipped(self) -> "Message":
        return replace(self, content=clip(self.content, MAX_MESSAGE_CHARS))


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------


class JobKind:
    """任务类型。用「常量命名空间」而不是 Enum：

    kind 直接进 DB 的 text 列，未来一定会加新类型。用 Enum 时，任何一处
    对 kind 做穷举的代码都会在新类型上线时变成运行期分支——而这类代码
    恰恰是消费端（worker 注册表），出问题表现为「任务入队了但没人处理」。
    """

    __slots__ = ()

    SUMMARIZE = "summarize"
    CONSOLIDATE = "consolidate"
    ARCHIVE = "archive"
    REINDEX = "reindex"

    ALL: tuple[str, ...] = (SUMMARIZE, CONSOLIDATE, ARCHIVE, REINDEX)


class JobState:
    """任务状态。状态机见 `Job.claimed_by` / `Job.failed`。"""

    __slots__ = ()

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"  # 可重试
    DEAD = "dead"      # 超过 attempts 上限，进死信

    ALL: tuple[str, ...] = (PENDING, RUNNING, DONE, FAILED, DEAD)


@dataclass(frozen=True, slots=True)
class Job:
    """一个任务。表即队列。

    关键字段是 `visible_at`：worker 领取任务时把它推后 N 秒（可见性超时），
    崩溃后不必释放锁，等超时自然重新可见。这比「加锁 + 依赖进程退出清理」
    可靠得多——进程被 kill -9 时没有机会清理任何东西。

    `attempts` 由领取时递增，避免「同一个任务被无限重试」。
    """

    id: JobId
    kind: str
    bot_id: BotId
    scope_key: str
    state: str = JobState.PENDING
    attempts: int = 0
    visible_at: float = 0.0
    locked_by: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    last_error: str = EMPTY
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state in (JobState.DONE, JobState.DEAD)

    def claimed_by(self, worker_id: str, *, at: float, visibility_s: float) -> "Job":
        return replace(
            self,
            state=JobState.RUNNING,
            attempts=self.attempts + 1,
            locked_by=worker_id,
            visible_at=at + visibility_s,
            updated_at=at,
        )

    def failed(self, error: str, *, at: float, max_attempts: int) -> "Job":
        """失败后的状态转移。

        超过上限就进死信，**不再重试**。hermes_core 没有死信概念，
        失败的任务只会被下一次水位重算再扫一遍，无限循环。
        """
        state = JobState.DEAD if self.attempts >= max_attempts else JobState.FAILED
        return replace(
            self,
            state=state,
            last_error=clip(error, 1_000),
            locked_by=None,
            updated_at=at,
        )


# ---------------------------------------------------------------------------
# 模型调用
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """送给模型的一条消息。

    不用 dict 或 openai 的 SDK 类型：契约层不能依赖任何 provider SDK，
    否则换 provider 时全平台都要跟着改 import。
    """

    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class LlmResult:
    """一次补全的结果。

    `usage` 保留原样（provider 字段不统一），由调用方取它认识的部分计量。
    `latency_ms` 由 provider 实现填，用于「模型变慢」的早期预警——
    hermes_core 没有这个观测，退避逻辑在 provider 侧变慢时完全失效。
    """

    text: str
    model: str = EMPTY
    latency_ms: float = 0.0
    finish_reason: str = EMPTY
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def is_truncated(self) -> bool:
        """输出被 max_tokens 截断。总结任务遇到这个必须降级重试，
        否则会解析出半个 JSON 并写进库。"""
        return self.finish_reason in ("length", "max_tokens")


__all__ = [
    "MAX_MESSAGE_CHARS",
    "MAX_MEMORY_CHARS",
    "MAX_RESIDENT_BLOCK_CHARS",
    "MAX_INTIMACY_BLOCK_CHARS",
    "MAX_RESIDENT_HARD_CAP",
    "max_chars_for",
    "MAX_SKILL_CONTENT_CHARS",
    "clip",
    "Bot",
    "Identity",
    "Memory",
    "RecallHit",
    "ResidentEntry",
    "InjectPayload",
    "Skill",
    "Message",
    "MessageRole",
    "Job",
    "JobKind",
    "JobState",
    "ChatMessage",
    "LlmResult",
]
