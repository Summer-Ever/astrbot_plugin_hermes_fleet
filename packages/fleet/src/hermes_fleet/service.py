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
    MAX_MEMORY_CHARS,
    MAX_RESIDENT_BLOCK_CHARS,
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
)
from hermes_recall import rerank, render

from .registry import BotRegistry, UNKNOWN_BOT, bot_id_of

# 文案与词条解析都在 resident.py：那一层是「模型怎么被引导去改常驻」，
# 和本模块的存储/检索职责分开，测试也不用起数据库。
# 这里 re-export 是为了 `hermes_fleet.BLOCK_LABELS` 这类旧引用不失效。
from .resident import (  # noqa: F401
    BLOCK_LABELS,
    INJECT_HEADER,
    MAINT_NOTES,
    MAX_INJECT_CHARS,
    clean_text,
    is_injection,
    render_inject,
    resolve_block,
)


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

        `query` 只用于「用户是不是在说自己的情况变了」这一个判断：命中就
        多一行明确指令，让模型去核对词条，而不是顺着话回一句「恭喜」。
        """
        db = self._db(scene)
        entries = await db.resident.get_all(
            scene.bot_id, scene.owner_id, scene.resident_scope
        )
        blocks = {
            block: entry.text
            for block, entry in entries.items()
            if entry is not None and entry.text.strip()
        }
        return render_inject(
            blocks,
            is_group=scene.is_group,
            query=query,
            max_chars=MAX_INJECT_CHARS,
        )

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
        """整段覆盖一个常驻词条。空字符串 = 清空该槽位。

        净化、别名解析、指令腔拒绝都放在这里，而不是工具层：工具、WebUI、
        运维命令走的是同一个入口，放在上层就得各写一遍、各漏一处。

        拒绝走异常而不是返回 None：调用方（工具）需要把具体原因念给模型听，
        模型才知道下一次该改什么——静默失败会让它以为已经写进去了。
        """
        target = resolve_block(block)
        if target is None:
            allowed = "、".join(
                f"{b.value}（{BLOCK_LABELS[b]}）" for b in ResidentBlock.ordered()
            )
            raise ValueError(f"未知词条 {block!r}，可用：{allowed}")

        label = BLOCK_LABELS.get(target, target.value)
        body = clean_text(text)
        if body and is_injection(body):
            raise ValueError(
                f"「{label}」词条文本带指令腔（试图改写规则或身份），已拒绝。"
                "词条记录的是事实与偏好，不是指令。"
            )
        if len(body) > MAX_RESIDENT_BLOCK_CHARS:
            raise ValueError(
                f"「{label}」词条超长：{len(body)} 字 > 上限 "
                f"{MAX_RESIDENT_BLOCK_CHARS} 字，请精简后整段重写。"
            )

        db = self._db(scene)
        return await db.resident.upsert(ResidentEntry(
            bot_id=scene.bot_id,
            owner_id=scene.owner_id,
            scope=scene.resident_scope,
            block=target,
            text=body,
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

    async def probe(self, platform_ids: list[str], *, limit: int = 20) -> list[dict]:
        """逐个打开指定账号取真实统计，**用完立刻关掉**。

        为什么必须设上限并主动关闭：`stats()` 不算便宜（要数四张表），而
        一次打开 100 个库会把 LRU 整个冲掉——正在被注入路径使用的连接被
        驱逐，下一个用户发言就得多付一次冷启动。所以这里宁可变慢、宁可
        分页，也不批量常驻。

        为什么关掉而不是留着：这是用户手动点「刷新」才走一次的低频路径，
        留着只会白占 LRU 名额。关掉后下次注入重开一次约 7.6ms，可以接受。

        为什么整个循环里不 `await`：AstrBot 是单事件循环，`stats()` 是同步
        调用，只要中间不让出控制权，就不可能有关掉别人正在用的连接这类交错。
        一旦这里插入 `await`，这个前提就不成立了——改动时要注意。
        """
        out: list[dict] = []
        for pid in list(platform_ids or [])[:max(0, int(limit))]:
            pid = str(pid or "").strip()
            if not pid:
                continue
            if not self.registry.has_account(pid):
                # 不认识的 id 绝不能 get()：那会凭空新建一个空库文件。
                out.append({"platform_id": pid, "error": "not_indexed"})
                continue
            was_open = self.registry.is_open(pid)
            try:
                st = self.registry.get(pid).stats()
            except Exception as exc:  # noqa: BLE001 — 单个库坏不能拖垮整页
                out.append({"platform_id": pid, "error": f"{type(exc).__name__}: {exc}"})
                continue
            finally:
                if not was_open:
                    try:
                        self.registry.close_bot(pid)
                    except Exception:
                        pass
            out.append({"platform_id": pid, "stats": st})
        return out
