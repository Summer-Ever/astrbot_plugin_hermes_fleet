"""记忆的业务语义。**这个模块不 import astrbot。**

## 为什么要把业务和框架隔开

AstrBot 的插件代码只能在 AstrBot 运行环境里测。宿主的 `Star`、`Context`、
`AstrMessageEvent` 都不是 pip 包，本机没装本体就一行都跑不起来——
hermes_core 的 5 套 smoke 里有 3 套正是卡在这里（`tests/_bootstrap.py`
用替身撑不住）。

这层把「拿一个事件、解析出归属、拼注入块、落库」做成纯函数式的
`MemoryService`，输入是普通字符串和 dataclass，输出是字符串和 id。
于是 100 个账号的隔离逻辑可以在没有 AstrBot 的机器上被完整测到，
`main.py` 只剩「把 event 的字段搬过来」这一件事。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from hermes_contracts import (
    EMPTY,
    BotId,
    Memory,
    MemoryId,
    MemoryKey,
    MemoryScope,
    OwnerId,
    ResidentBlock,
    ResidentEntry,
    ResidentScope,
    SessionId,
    Visibility,
    clip,
    MAX_MEMORY_CHARS,
)
from hermes_recall import rerank, render

from .registry import BotRegistry, UNKNOWN_BOT, bot_id_of

# 六词条的中文标签。放在这里而不是契约层：它是「给模型看的措辞」，
# 属于产品文案，会随运营口味调，不该让契约包跟着改版本。
BLOCK_LABELS: dict[ResidentBlock, str] = {
    ResidentBlock.ADDRESS: "称呼",
    ResidentBlock.ASSISTANT: "助手设定",
    ResidentBlock.USER: "用户信息",
    ResidentBlock.PREFERENCE: "偏好与忌讳",
    ResidentBlock.STYLE: "回复风格",
    ResidentBlock.RULES: "规则与避雷",
}

INJECT_HEADER = "[长期稳定信息（每轮必带）]"

# 这两段每次注入都在，但很短。它们的作用是「告诉模型该动哪个工具」，
# 不复述工具用法——schema 每轮都在上下文里，复述只是白烧 token。
MAINT_NOTES = (
    "[记忆使用约定] 需要回忆过去的具体细节时，先调用 search_memory 检索，"
    "不要凭印象编造；检索不到就如实说没有记录，不要说「我的记忆被清空了」。",
    "[常驻维护] 用户要求记住、修改或忘掉稳定信息时，本轮调用 update_resident 处理；"
    "后台任务不会替你维护常驻。",
)

# 注入块总长上限。六词条满配是 6 × 2000 字符 = 12000，全塞进去会挤掉真正的
# 对话。这里用字符预算硬截断，宁可丢掉尾部词条也不让注入失控——
# hermes_core 早期就是没有这个上限，一次误写把整轮上下文占满。
MAX_INJECT_CHARS = 4_000


@dataclass(frozen=True, slots=True)
class Scene:
    """一次 LLM 请求的归属坐标。

    这就是「多租户」在代码里的样子：一条消息必须能回答出
    「哪个账号 / 哪个人 / 哪个群 / 是不是群聊」四个问题，
    任何一个缺失，隔离就只能靠猜。
    """

    bot_id: BotId
    owner_id: OwnerId
    group_id: str = EMPTY
    is_group: bool = False

    @property
    def resident_scope(self) -> ResidentScope:
        return (
            ResidentScope.for_group(self.group_id)
            if self.is_group and self.group_id
            else ResidentScope.for_private()
        )

    @property
    def viewer_group(self) -> str:
        """检索时的「观察者所在群」。私聊必须为空——
        传了群号就等于允许召回该群的记忆，是最典型的越权写法。"""
        return self.group_id if self.is_group else EMPTY


class MemoryService:
    """把存储、检索、注入策略粘在一起。无状态，可安全并发调用。"""

    def __init__(
        self,
        registry: BotRegistry,
        *,
        group_private_memory: bool = False,
        recall_k: int = 5,
        clock_now=None,
    ) -> None:
        self.registry = registry
        self.group_private_memory = bool(group_private_memory)
        self.recall_k = max(1, int(recall_k))
        self._now = clock_now or time.time

    # ------------------------------------------------------------ 归属

    def bot_of(self, platform_id: str) -> BotId:
        return bot_id_of(platform_id or UNKNOWN_BOT)

    async def scene(self, platform_id: str, sender_id: str, group_id: str,
                    *, is_group: bool) -> Scene:
        bot = self.bot_of(platform_id)
        db = self.registry.get(str(bot))
        owner = await db.identities.resolve(bot, str(sender_id or ""))
        return Scene(
            bot_id=bot,
            owner_id=owner,
            group_id=str(group_id or EMPTY),
            is_group=bool(is_group) and bool(group_id),
        )

    def _db(self, scene: Scene):
        return self.registry.get(str(scene.bot_id))

    # ------------------------------------------------------------ 注入

    async def inject_block(self, scene: Scene, query: str) -> str:
        """构建每轮注入块。只含常驻六词条 + 维护约定，不含检索结果。

        归档记忆**故意不自动注入**。每轮把检索结果塞进上下文，代价是固定的
        （几千字符），收益却随查询波动——大多数轮次捞回来的是噪声，还会把
        真正的用户发言挤到后面。召回改成模型主动调 `search_memory`。
        """
        db = self._db(scene)
        entries = await db.resident.get_all(
            scene.bot_id, scene.owner_id, scene.resident_scope
        )

        lines: list[str] = []
        if entries:
            lines.append(INJECT_HEADER)
            for block in ResidentBlock.ordered():
                entry = entries.get(block)
                if entry is None or not entry.text.strip():
                    continue  # 空槽不注入，注入成本才有界
                label = BLOCK_LABELS.get(block, block.value)
                lines.append(f"- {label}：{entry.text.strip()}")
        lines.extend(MAINT_NOTES)

        text = "\n".join(lines)
        if len(text) > MAX_INJECT_CHARS:
            text = clip(text, MAX_INJECT_CHARS)
        return text

    # ------------------------------------------------------------ 落库

    async def capture_turn(
        self,
        scene: Scene,
        session_id: str,
        user_text: str,
        reply_text: str,
    ) -> int:
        """把一轮对话写进消息表。返回实际新增条数（重投时为 0）。

        用 `append_batch` 一次提交两条，而不是两条 `append`：
        单条提交在突发场景下会打满 IO（实测 batch 500 条比逐条快 36 倍）。
        """
        from hermes_contracts import Message

        db = self._db(scene)
        now = self._now()
        rows = []
        if user_text:
            rows.append(Message(
                bot_id=scene.bot_id, session_id=SessionId(session_id),
                role="user", content=user_text, ts=now,
                sender_id=str(scene.owner_id),
                group_id=scene.group_id,
            ))
        if reply_text:
            rows.append(Message(
                bot_id=scene.bot_id, session_id=SessionId(session_id),
                role="assistant", content=reply_text, ts=now + 1e-3,
                group_id=scene.group_id,
            ))
        return await db.messages.append_batch(rows)

    # ------------------------------------------------------------ 检索

    def _visible(self, memory: Memory, scene: Scene) -> bool:
        """群聊里额外加一道闸：不引用 1:1 私聊说过的事。

        `Memory.can_read` 里「owner 相同即可读」在私聊上下文是对的，
        但群聊里把私聊内容念出来就是当众公开。这条策略来自 hermes_core
        的实测教训，保留为默认关闭的开关而不是直接删掉。
        """
        if not memory.can_read(scene.owner_id, viewer_group=scene.viewer_group):
            return False
        if scene.is_group and not self.group_private_memory:
            if memory.visibility is Visibility.PRIVATE:
                return False
            if memory.scope is MemoryScope.USER and memory.group_id and (
                memory.group_id != scene.group_id
            ):
                return False
        return True

    async def search(self, scene: Scene, query: str, *, limit: int | None = None) -> str:
        """检索并返回**已渲染成文本**的结果，直接可回给模型。

        候选池只取 `recall_k * 8` 条：精排只接受 ≤50 条（`hermes_recall.MAX_RERANK`），
        把全库捞进 Python 打分是 hermes_core 的病根，这里从接口上就不留入口。
        """
        db = self._db(scene)
        key = MemoryKey(
            bot_id=scene.bot_id,
            scope=MemoryScope.USER,
            owner_id=str(scene.owner_id),
            group_id=EMPTY,
        )
        pool = max(32, (limit or self.recall_k) * 8)
        candidates = await db.memories.search_candidates(
            key,
            query=query,
            viewer=scene.owner_id,
            viewer_group=scene.viewer_group,
            limit=pool,
        )
        usable = [m for m in candidates if self._visible(m, scene)]
        hits = rerank(query, usable, k=limit or self.recall_k, now=self._now())
        return render(hits)

    # ------------------------------------------------------------ 写记忆

    async def save(self, scene: Scene, content: str, *, kind: str = "fact") -> Memory:
        """存一条记忆。群聊写入的记忆默认「本群可见」，不外溢到别的群。"""
        db = self._db(scene)
        body = clip(str(content or EMPTY).strip(), MAX_MEMORY_CHARS)
        if not body:
            raise ValueError("记忆内容为空")
        memory = Memory(
            id=MemoryId(0),
            bot_id=scene.bot_id,
            content=body,
            scope=MemoryScope.USER,
            visibility=Visibility.GROUP if scene.is_group else Visibility.PRIVATE,
            owner_id=str(scene.owner_id),
            group_id=scene.group_id if scene.is_group else EMPTY,
            kind=str(kind or "fact"),
            source="tool",
            ts=self._now(),
        )
        return await db.memories.add(memory)

    async def forget(self, scene: Scene, memory_id: int) -> bool:
        """软删除一条记忆（`merged_into` 指向自身表示已废弃，可回滚）。

        不做物理删除：模型判断「这条该忘」会有判错的时候，物理删掉无法补救。
        """
        db = self._db(scene)
        existing = await db.memories.get(scene.bot_id, MemoryId(int(memory_id)))
        if existing is None or not self._visible(existing, scene):
            return False
        await db.memories.update(existing.merged(int(memory_id), at=self._now()))
        return True

    # ------------------------------------------------------------ 常驻

    async def set_resident(self, scene: Scene, block: str, text: str) -> ResidentEntry:
        """整段覆盖一个常驻词条。空字符串 = 清空该槽位。"""
        try:
            target = ResidentBlock(str(block))
        except ValueError as exc:
            allowed = ", ".join(b.value for b in ResidentBlock.ordered())
            raise ValueError(f"未知词条 {block!r}，可用：{allowed}") from exc
        db = self._db(scene)
        return await db.resident.upsert(ResidentEntry(
            bot_id=scene.bot_id,
            owner_id=scene.owner_id,
            scope=scene.resident_scope,
            block=target,
            text=str(text or EMPTY).strip(),
            updated_ts=self._now(),
        ))

    async def resident_snapshot(self, scene: Scene) -> dict[str, str]:
        db = self._db(scene)
        entries = await db.resident.get_all(
            scene.bot_id, scene.owner_id, scene.resident_scope
        )
        return {b.value: e.text for b, e in entries.items()}

    # ------------------------------------------------------------ 运维

    async def bot_stats(self, platform_id: str) -> dict[str, int]:
        db = self.registry.get(str(self.bot_of(platform_id)))
        return db.stats()
