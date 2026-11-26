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
    max_chars_for,
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
from hermes_recall import MAX_RERANK, rerank, render

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
        overrides_of=None,
    ) -> None:
        self.registry = registry
        self.group_private_memory = bool(group_private_memory)
        self.recall_k = max(1, int(recall_k))
        self._now = clock_now or time.time
        # 按账号覆盖的解析器（通常是 `registry.overrides_of`）。留成注入
        # 而不是直接调 registry，是为了这层仍然能脱离磁盘测：用例可以塞
        # 一个纯 dict 进来验证「这个号的开关只影响这个号」。
        self._overrides_of = overrides_of

    def _override(self, scene: Scene, key: str):
        if self._overrides_of is None:
            return None
        try:
            return self._overrides_of(str(scene.bot_id)).get(key)
        except Exception:
            return None

    def _group_private_for(self, scene: Scene) -> bool:
        value = self._override(scene, "group_private_memory")
        return self.group_private_memory if value is None else bool(value)

    def _recall_for(self, scene: Scene) -> int:
        value = self._override(scene, "recall_k")
        try:
            return max(1, int(value)) if value is not None else self.recall_k
        except (TypeError, ValueError):
            return self.recall_k

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
        if scene.is_group and not self._group_private_for(scene):
            if memory.visibility is Visibility.PRIVATE:
                return False
            if memory.scope is MemoryScope.USER and memory.group_id and (
                memory.group_id != scene.group_id
            ):
                return False
        return True

    async def search(self, scene: Scene, query: str, *, limit: int | None = None) -> str:
        """检索并返回**已渲染成文本**的结果，直接可回给模型。

        候选池只取 `recall_k * 8` 条，并夹在 `MAX_RERANK` 以内：精排只接受
        ≤50 条（`hermes_recall.MAX_RERANK`）。上限必须在**取候选之前**夹住，
        不能指望 rerank 兜底——它超过 50 是直接抛 ValueError 的，而配置项
        `recall_k` 没有硬上限，k=7 就会把整条检索链路打断。
        """
        db = self._db(scene)
        key = MemoryKey(
            bot_id=scene.bot_id,
            scope=MemoryScope.USER,
            owner_id=str(scene.owner_id),
            group_id=EMPTY,
        )
        take = limit or self._recall_for(scene)
        pool = min(MAX_RERANK, max(32, take * 8))
        candidates = await db.memories.search_candidates(
            key,
            query=query,
            viewer=scene.owner_id,
            viewer_group=scene.viewer_group,
            limit=pool,
        )
        usable = [m for m in candidates if self._visible(m, scene)]
        hits = rerank(query, usable, k=take, now=self._now())
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
        limit = max_chars_for(target)
        if len(body) > limit:
            raise ValueError(
                f"「{label}」词条超长：{len(body)} 字 > 上限 "
                f"{limit} 字，请精简后整段重写。"
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

    async def export_owner(self, *, platform_id: str, owner_id: str,
                           max_messages: int = 0,
                           max_memories: int = 0) -> dict:
        """收集「某人 + 某机器人」的全部记忆数据，供导出文档渲染。

        取数放在服务层而不是命令里，是因为这是唯一一条把原始对话写到
        文件上的路径：过滤条件必须只有一处实现，将来改了才不会漏。

        返回的是**已过滤**的数据——`messages` 里已经没有别人的发言，
        调用方可以直接渲染，不必（也不应该）再做一遍隐私判断。

        `max_messages` / `max_memories` 传 0 表示**全量**。用户明确要求
        原文有几千万字也照发；条数不再截断，唯一兜底是文件大小上限
        （在命令层按 UTF-8 字节算，超 100MB 就拒绝发送）。
        """
        from .export import pair_replies

        pid = str(platform_id or "")
        owner = str(owner_id or "")
        empty = {
            "bot_id": pid, "owner_id": owner, "resident": [], "memories": [],
            "messages": [], "memory_total": 0, "message_total": 0,
        }
        if not pid or not owner or not self.registry.has_account(pid):
            return empty

        db = self.registry.get(pid)
        bot = self.bot_of(pid)
        resident = await db.resident.list_by_owner(bot, owner)
        cap_mem = int(max_memories)
        memories = await db.memories.list_by_owner(
            bot, owner, limit=cap_mem if cap_mem > 0 else 0
        )
        total_mem = await db.memories.count_active(bot, owner_id=owner)
        total_msg = await db.messages.count_transcript(bot, owner)
        # 全量取原始行：群会话里别人说的话要参与配对（判断「这条机器人
        # 回复是回谁的」），不能省；用户要的是全部原文，也不再截条数。
        raw = await db.messages.transcript(bot, owner, limit=0)
        # 过滤必须在取到完整会话上下文之后做：SQL 里提前滤掉群友的
        # 发言，会让跟在后面的机器人回复被误算成回复本人。
        kept = pair_replies(raw, owner)
        cap_msg = int(max_messages)
        if cap_msg > 0:
            kept = kept[-cap_msg:]
        return {
            "bot_id": pid,
            "owner_id": owner,
            "resident": resident,
            "memories": memories,
            "messages": kept,
            "memory_total": max(int(total_mem), len(memories)),
            # 全量取数下 kept 就是准确条数（SQL 的 count 会把回群友的话
            # 也算进去，反而偏高），不再用估算值兜底。
            "message_total": len(kept) if cap_msg <= 0 else max(int(total_msg), len(kept)),
        }

    async def import_owner(self, *, platform_id: str, owner_id: str,
                           payload: dict) -> dict:
        """把记忆配置文档解析出的数据导入到**当前**账号/用户下。

        冲突策略明确写在语义里：常驻按 (owner, scope, block) 整段覆盖，
        归档记忆**追加**为新条目（不按 id 去重——导回的是另一台机器的 id，
        直接对号入座会串到本机别的记忆上）。所以重复导入同一份文件会
        得到两份归档条目，这是刻意的：追加语义可预期，去重语义要猜。

        `payload` 来自 `parse_memory_config()`，已经过字段校验。这里再做
        一次范围校验，防止调用方跳过解析层直接塞脏数据。
        """
        pid = str(platform_id or "")
        owner = str(owner_id or "")
        if not pid or not owner or not self.registry.has_account(pid):
            raise ValueError("目标机器人账号不存在")
        db = self.registry.get(pid)
        bot = self.bot_of(pid)
        now = self._now()

        resident = payload.get("resident", [])
        memories = payload.get("memories", [])
        if not isinstance(resident, list) or not isinstance(memories, list):
            raise ValueError("导入数据格式错误")

        # 先全部校验再落库：第 N 条被拒时不能留下前 N-1 条已写入的
        # 「导了一半」的账号。
        entries: list[ResidentEntry] = []
        for item in resident:
            try:
                scope = ResidentScope.parse(str(item.get("scope", "")))
                block = ResidentBlock(str(item.get("block", "")))
            except ValueError as exc:
                raise ValueError(f"常驻词条字段非法：{exc}") from exc
            # 导入路径必须和 set_resident 走同一套净化：否则用户改一份
            # JSON 文档就能把块边界标记或指令腔写进常驻词条，绕过写入侧的
            # 防护。clean_text 幂等，正常导出文件不受影响。
            text = clean_text(str(item.get("text", "") or ""))
            if text and is_injection(text):
                label = BLOCK_LABELS.get(block, block.value)
                raise ValueError(
                    f"「{label}」词条文本带指令腔（试图改写规则或身份），"
                    "已拒绝导入。"
                )
            entries.append(ResidentEntry(
                bot_id=bot, owner_id=owner, scope=scope, block=block,
                text=text, updated_ts=now,
            ))
        for entry in entries:
            await db.resident.upsert(entry)  # 空文本 = 清空该槽位
        n_res = len(entries)

        to_add: list[Memory] = []
        for item in memories:
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            try:
                scope = MemoryScope(str(item.get("scope", "user") or "user"))
                vis = Visibility(str(item.get("visibility", "private") or "private"))
            except ValueError as exc:
                raise ValueError(f"归档记忆字段非法：{exc}") from exc
            group = str(item.get("group_id", "") or "")
            if scope is MemoryScope.GLOBAL or vis is Visibility.PUBLIC:
                # 全局/公开记忆需要管理员权限，导入路径一律降级到本人私有：
                # 否则用户发一份文件就能往所有人的库里写东西。
                scope = MemoryScope.USER
                vis = Visibility.PRIVATE
            to_add.append(Memory(
                id=MemoryId(0), bot_id=bot, content=content,
                scope=scope, visibility=vis, owner_id=owner, group_id=group,
                kind=str(item.get("kind", "fact") or "fact"),
                source="import",
                ts=float(item.get("ts", 0.0) or 0.0) or now,
                updated_ts=now,
                importance=float(item.get("importance", 0.0) or 0.0),
            ))
        if to_add:
            await db.memories.add_many(to_add)

        return {
            "bot_id": pid,
            "owner_id": owner,
            "resident_imported": n_res,
            "memories_imported": len(to_add),
        }

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
