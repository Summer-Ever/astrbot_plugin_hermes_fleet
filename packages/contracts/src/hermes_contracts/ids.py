"""标识符与作用域：全平台唯一一处定义「同一件事叫什么」。

为什么这些类型必须集中在一个包里：
    多 agent 并行开发时，最容易出的不是逻辑错，而是**同一个概念在不同模块里有
    两种表示**。hermes_core 里就同时存在两套作用域表示——`memories` 表用
    `scope ∈ {user, group, global}` + `owner_id` + `group_id` 三列，`resident`
    表却用 `scope ∈ {private, group:<gid>}` 一个字符串。两套都在用、都能跑，
    但每次跨表读写都要在脑子里换算一次，于是必然有人写错一次。

这个文件的作用是把「换算」变成有测试覆盖的唯一点，其余模块只准用这里的类型。

设计约束（改之前先想清楚）：
- 零第三方依赖：只用标准库，任何 agent 不装任何包就能 import 并跑测试。
- 全部不可变：frozen / slots。并行开发时共享可变对象是冲突与竞态的第一来源。
- 不做 IO、不读配置：这里只定义形状，不定义来源。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import NewType

# ---------------------------------------------------------------------------
# 标量 ID
# ---------------------------------------------------------------------------
# 用 NewType 而不是裸 str 的原因：函数签名说谎太多。`def get(owner_id, group_id)`
# 两个参数顺序颠倒时类型检查器一声不吭，线上表现为「私聊记忆串到群里」。
# NewType 几乎零运行时成本，但能让 mypy 抓住这一类错。

BotId = NewType("BotId", str)
OwnerId = NewType("OwnerId", str)
GroupId = NewType("GroupId", str)
SessionId = NewType("SessionId", str)
MemoryId = NewType("MemoryId", int)
SkillId = NewType("SkillId", int)
JobId = NewType("JobId", int)

# 空字符串是「不属于任何群 / 没有 owner」的哨兵值，这是沿用 hermes_core 的约定
# （那里用 '' 而不是 NULL，好处是所有比较都不用处理 NULL 三值逻辑）。
# 代价是必须防住「空 owner 被当成合法值写进私聊记忆」，见 validate_owner。
EMPTY = ""

_BOT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def validate_bot_id(value: str) -> BotId:
    """bot_id 会进分区键、文件名、URL path，字符集必须收紧。

    不允许 `/` 和空格不是洁癖：hermes_core 的 per-bot 二进制备份、以及
    `safe_join` 之外的日志路径拼装，都会因为一个斜杠写到目录外面去。
    """
    if not isinstance(value, str) or not _BOT_ID_RE.match(value):
        raise ValueError(
            f"invalid bot_id {value!r}: 只允许 [A-Za-z0-9_.:-]，长度 1-64"
        )
    return BotId(value)


def require_owner(owner_id: str, *, context: str = "") -> OwnerId:
    """私聊/个人域记忆的 owner 不允许为空。

    hermes_core 的 `_repair_orphan_private` 就是在修这条约束被破坏后的脏数据：
    `visibility='private'` 但 `owner_id=''` 的行，谁都能匹配上，等于泄露。
    与其事后修，不如在写入口拒绝。群公共记忆用 `visibility='group'`，
    不走这个函数。
    """
    if not owner_id:
        raise ValueError(f"owner_id 不能为空{('（' + context + '）') if context else ''}")
    return OwnerId(owner_id)


# ---------------------------------------------------------------------------
# 记忆作用域
# ---------------------------------------------------------------------------


class MemoryScope(str, Enum):
    """`memories.scope`。描述这条记忆**属于谁**。"""

    USER = "user"      # 属于某个人
    GROUP = "group"    # 属于某个群
    GLOBAL = "global"  # 全 bot 公共（管理员维护）

    def __str__(self) -> str:  # 让 f-string 直接给出 DB 里的值
        return self.value


class Visibility(str, Enum):
    """`memories.visibility`。描述这条记忆**能被谁读到**。

    与 scope 正交，这是 hermes_core 设计得最对的一处：scope 说归属，
    visibility 说授权。二者混成一个字段，会导致「群记忆但只能本人可见」
    这种合法需求无处表达。
    """

    PRIVATE = "private"  # 仅 owner 本人
    GROUP = "group"      # 本群成员
    PUBLIC = "public"    # 本 bot 所有人（需管理员写入）

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class MemoryKey:
    """一条记忆的归属坐标，对应 `memories` 的 (bot, scope, owner, group) 四列。

    存在的意义是让「可见性判断」只有一处实现。hermes_core 把这个逻辑散在
    `db.py` 的多段 SQL 字符串里，改一处漏一处是常态。
    """

    bot_id: BotId
    scope: MemoryScope
    owner_id: str = EMPTY
    group_id: str = EMPTY

    def __post_init__(self) -> None:
        if self.scope is MemoryScope.USER and not self.owner_id:
            raise ValueError("scope=user 必须带 owner_id")
        if self.scope is MemoryScope.GROUP and not self.group_id:
            raise ValueError("scope=group 必须带 group_id")
        if self.scope is MemoryScope.GLOBAL and (self.owner_id or self.group_id):
            raise ValueError("scope=global 不允许带 owner_id / group_id")

    def readable_by(self, viewer: OwnerId, *, viewer_group: str = EMPTY) -> bool:
        """默认可见性判定（不看 visibility 字段，只看归属）。

        规则：本人写的本人可读；群归属的本群成员可读；global 全可读。
        注意这里**故意**不处理 `visibility=private` 的群记忆——那是
        `Memory` 实体上的字段，由上层 `can_read()` 组合判断。
        """
        if self.scope is MemoryScope.GLOBAL:
            return True
        if self.scope is MemoryScope.USER:
            return self.owner_id == viewer
        return bool(viewer_group) and self.group_id == viewer_group


# ---------------------------------------------------------------------------
# 常驻作用域（与记忆作用域是两套，故意不合并）
# ---------------------------------------------------------------------------


class ResidentBlock(str, Enum):
    """常驻的七个封闭词条（其中 `intimacy` 只由用户指令维护，不进 AI 工具）。

    为什么是封闭槽位而不是动态条目：hermes_core 的 README 记了一次真实教训——
    按「条数」分配预算时，6 个槽位里 2 个被同一句「叫哥哥」占满，用户信息、
    偏好、避雷全被挤掉。改成「固定槽位、整段覆盖」后规模天然有界
    （最多 7 × 2000 字），且空槽不注入，注入成本可预测。

    顺序即注入顺序，不要随手调换：模型对靠前的内容注意力更高，
    `address`（称呼）必须在最前面，否则会出现叫错人的体感问题。
    """

    ADDRESS = "address"        # 用户希望被怎么称呼
    ASSISTANT = "assistant"    # 助手在这个场景里的自我设定
    USER = "user"              # 用户的基本信息
    PREFERENCE = "preference"  # 偏好与忌讳
    # 亲密关系与情感表达。**只由用户用指令维护**，不进 AI 的 update_resident
    # 白名单：模型最容易把「聊得热络」误判成「关系变了」，自我发挥改写它。
    INTIMACY = "intimacy"
    STYLE = "style"            # 回复风格
    RULES = "rules"            # 规则与避雷

    @classmethod
    def ordered(cls) -> tuple["ResidentBlock", ...]:
        """固定的注入顺序，全平台唯一来源。"""
        return (
            cls.ADDRESS,
            cls.ASSISTANT,
            cls.USER,
            cls.PREFERENCE,
            cls.INTIMACY,
            cls.STYLE,
            cls.RULES,
        )


@dataclass(frozen=True, slots=True)
class ResidentScope:
    """常驻的作用域：`private` 或 `group:<gid>`。

    这里是第二套 scope 表示。保留它（而不是统一成 MemoryScope）的理由：
    常驻没有 `global` 概念，也不需要 owner 参与主键之外的计算，用两态枚举
    更贴合语义。**但转换必须走这个类**，别在业务代码里手拼字符串——
    `f"group:{gid}"` 散落五处，就会出现一处写成了 `group/{gid}`。
    """

    kind: str  # 'private' | 'group'
    group_id: str = EMPTY

    def __post_init__(self) -> None:
        if self.kind not in ("private", "group"):
            raise ValueError(f"invalid resident scope kind: {self.kind!r}")
        if self.kind == "group" and not self.group_id:
            raise ValueError("group 作用域必须带 group_id")
        if self.kind == "private" and self.group_id:
            raise ValueError("private 作用域不允许带 group_id")

    def to_key(self) -> str:
        """序列化成 DB 里那一列的值。"""
        return "private" if self.kind == "private" else f"group:{self.group_id}"

    @classmethod
    def parse(cls, key: str) -> "ResidentScope":
        if key == "private":
            return cls("private")
        if key.startswith("group:") and len(key) > 6:
            return cls("group", key[6:])
        raise ValueError(f"invalid resident scope key: {key!r}")

    @classmethod
    def for_private(cls) -> "ResidentScope":
        return cls("private")

    @classmethod
    def for_group(cls, group_id: str) -> "ResidentScope":
        return cls("group", group_id)


# ---------------------------------------------------------------------------
# 会话键
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionKey:
    """一个会话 = (bot, session_id)。

    bot 是必须的：100 个 bot 下，两个 bot 撞上同一个 session_id 完全可能
    （都是 QQ 群号）。hermes_core 没有 bot 维度，A bot 的群记忆会注入到
    B bot，这是产品事故而不是性能问题。
    """

    bot_id: BotId
    session_id: SessionId

    def to_key(self) -> str:
        """稳定编码，用于 `jobs.scope_key` 这类 text 列。

        bot_id 的字符集已被 `validate_bot_id` 收紧到不含 `|`，
        所以用 `|` 分隔可以无歧义还原。
        """
        return f"{self.bot_id}|{self.session_id}"

    @classmethod
    def parse(cls, key: str) -> "SessionKey":
        bot, sep, sid = key.partition("|")
        if not sep or not bot or not sid:
            raise ValueError(f"invalid session key: {key!r}")
        return cls(BotId(bot), SessionId(sid))
