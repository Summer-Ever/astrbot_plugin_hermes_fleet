"""构造测试数据的工厂，供一致性套件与各模块自测共用。

为什么要放在契约包里：
    每个实现方（内存版、Postgres 版、SQLite 版）如果各写一套 fixture，那么
    「同一个用例」在不同实现上跑的其实是**不同的数据**——这样的一致性测试
    是假的。fixtures 必须是契约的一部分，跟着契约一起冻结。
"""

from __future__ import annotations

from ..clock import Clock
from ..ids import BotId, MemoryId, MemoryKey, MemoryScope, OwnerId, ResidentBlock, ResidentScope, SkillId, Visibility
from ..models import Memory, ResidentEntry, Skill

# 测试里反复出现的固定值，集中一处，改一处全改
BOT = BotId("bot_alpha")
BOT_B = BotId("bot_beta")
ALICE = OwnerId("alice")
BOB = OwnerId("bob")
GROUP = "group_100"


def key(
    *,
    bot: BotId = BOT,
    scope: MemoryScope = MemoryScope.USER,
    owner: str = ALICE,
    group_id: str = "",
) -> MemoryKey:
    """记忆归属坐标。默认是「alice 在 bot_alpha 的私聊」。"""
    return MemoryKey(bot_id=bot, scope=scope, owner_id=owner, group_id=group_id)


def memory(
    clock: Clock,
    *,
    content: str = "用户喜欢喝乌龙茶",
    owner: str = ALICE,
    bot: BotId = BOT,
    scope: MemoryScope = MemoryScope.USER,
    visibility: Visibility = Visibility.PRIVATE,
    group_id: str = "",
    kind: str = "fact",
    memory_id: int = 0,
    ts: float | None = None,
) -> Memory:
    """造一条记忆。`id=0` 表示「还没入库」，由实现的 `add` 分配。"""
    now = clock.now() if ts is None else ts
    return Memory(
        id=MemoryId(memory_id),
        bot_id=bot,
        content=content,
        scope=scope,
        visibility=visibility,
        owner_id=owner,
        group_id=group_id,
        kind=kind,
        ts=now,
        updated_ts=now,
    )


def resident(
    *,
    block: ResidentBlock = ResidentBlock.ADDRESS,
    text: str = "叫我哥哥",
    owner: str = ALICE,
    bot: BotId = BOT,
    scope: ResidentScope | None = None,
) -> ResidentEntry:
    return ResidentEntry(
        bot_id=bot,
        owner_id=OwnerId(owner),
        scope=scope or ResidentScope.for_private(),
        block=block,
        text=text,
    )


def skill(
    *,
    name: str = "日报生成",
    content: str = "1. 拉取今日数据\n2. 汇总成 5 条",
    owner: str = ALICE,
    bot: BotId = BOT,
    origin: str = "hermes",
) -> Skill:
    return Skill(
        id=SkillId(0),
        bot_id=bot,
        name=name,
        content=content,
        owner_id=owner,
        origin=origin,
    )
