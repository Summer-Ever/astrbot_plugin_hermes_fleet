"""astrbot_plugin_hermes_fleet

为上百个机器人账号提供**互相隔离**的长期记忆。每个账号一个独立记忆库
（SQLite 文件），串号在物理上不可能发生。

## 与 hermes_core 的关系

沿用同一套接入姿势（`on_llm_request` 注入 + `on_llm_response` 落库 +
`llm_tool` 工具面 + 命令组运维），因为这套姿势在 AstrBot 上被验证过：
注入走临时 content part，不落会话历史、不破坏提示词缓存；工具描述负责
教模型「什么时候用哪个」，注入块只留每轮必带的最小基线。

差异在**租户维度**：hermes_core 的隔离键是 `(owner_id, group_id)`，没有
「哪个账号」这一维；本插件把 `event.get_platform_id()` 作为第一维，
并把它落成**独立库文件**而不是 SQL 条件。

## 这一层只做翻译

所有业务逻辑在 `hermes_fleet`（不 import astrbot，可脱离本体测试）。
本文件负责：取事件字段 → 调 service → 把结果塞回请求/回复。
这样做的直接收益是本机没有 AstrBot 也能跑 20 个隔离用例。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.join(_HERE, "packages")


def _bootstrap_paths() -> list[str]:
    """把 `packages/*/src` 加进 sys.path（等价于 editable install）。

    不做 `pip install -e`：插件装在宿主的 Python 里，往里面塞 .pth 或
    egg-info 会污染本体环境，而且卸载插件时不一定清得干净。sys.path 注入
    是进程内、可逆、零副作用的。

    也解释了为什么不把这些包放到 `data/plugins/` 之外：插件自包含才能
    被 AstrBot 的安装器一把拷走。
    """
    added: list[str] = []
    if not os.path.isdir(_PKG_ROOT):
        return added
    for name in sorted(os.listdir(_PKG_ROOT)):
        src = os.path.join(_PKG_ROOT, name, "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)
            added.append(src)
    return added


def _purge_stale_packages() -> list[str]:
    """把上一次加载残留的 `hermes_*` 模块从 `sys.modules` 里摘掉。

    **为什么必须自己动手**：AstrBot 的「重载插件」只清理 `data.plugins.<插件名>`
    前缀的模块。而本插件的辅助包是通过 sys.path 以**顶层名**（`hermes_fleet`、
    `hermes_storage_sqlite`…）导入的，不匹配那个前缀，于是重载后：

        main.py 是新的，packages/ 下的辅助包还是旧的。

    症状极其隐蔽：新加的接口在第一次调用时 `AttributeError`，而代码看起来
    完全正确，只在「热更新过、没重启」的机器上复现。v0.7 给 `BotRegistry`
    加 `set_note` 时就踩了一次（备注接口 500，其它接口都正常）。

    只摘**文件位于本插件目录内**的模块：万一将来与别的插件同名，也不会
    把人家的模块从缓存里删掉——那会让对方拿到两份互不相同的类。
    """
    purged: list[str] = []
    for name, mod in list(sys.modules.items()):
        if not name.split(".", 1)[0].startswith("hermes_"):
            continue
        path = str(getattr(mod, "__file__", "") or "")
        if path.startswith(_HERE):
            del sys.modules[name]
            purged.append(name)
    return purged


_bootstrap_paths()
_purge_stale_packages()

from astrbot.api import logger  # noqa: E402
from astrbot.api.event import filter, AstrMessageEvent  # noqa: E402
from astrbot.api.star import Context, Star, register  # noqa: E402

try:  # 兼容旧版本导入路径
    from astrbot.api.provider import ProviderRequest, LLMResponse  # noqa: F401
except ImportError:  # pragma: no cover
    from astrbot.core.provider.entities import ProviderRequest, LLMResponse  # noqa: F401

try:
    from astrbot.api.star import StarTools

    def _default_data_dir() -> str:
        return str(StarTools.get_data_dir("astrbot_plugin_hermes_fleet"))
except ImportError:  # pragma: no cover

    def _default_data_dir() -> str:
        return os.path.join("data", "plugin_data", "astrbot_plugin_hermes_fleet")


# 动态注入优先走临时 content part：它不进会话历史，也就不会每轮重发
# 一遍、把提示词缓存打穿。老版本没有这个能力才退回 system_prompt。
try:
    from astrbot.core.agent.message import TextPart
except ImportError:  # pragma: no cover
    TextPart = None

from hermes_contracts import (  # noqa: E402
    MAX_RESIDENT_BLOCK_CHARS,
    ChatMessage,
    LlmResult,
    ResidentBlock,
)
from hermes_contracts.errors import Unavailable  # noqa: E402
from hermes_fleet import BLOCK_LABELS, BotRegistry, MemoryService  # noqa: E402
from hermes_workers import (  # noqa: E402
    ConsolidateOutcome,
    FallbackProvider,
    MemoryConsolidator,
    SessionSummarizer,
    SummaryOutcome,
)

PLUGIN_VERSION = "0.7.1"

# 同一句话的注入去重窗口（秒）。一次用户发言可能触发多轮 LLM 请求
# （工具调用循环），每轮都注入等于把同一段记忆反复塞进上下文。
# 45 秒覆盖一次工具循环，又不会让「换个说法再问一遍」持续失忆。
INJECT_TTL = 45


class AstrBotProvider:
    """把 AstrBot 的对话模型包成 `hermes_contracts.LlmProvider`。

    为什么要包一层，而不是让 worker 直接调 AstrBot：
    worker 在 `packages/workers` 里，一旦 import astrbot 就再也无法脱离
    运行环境测试，而它恰恰是最需要测试的一块（输出解析、归属判定）。
    包一层的代价是这 40 行，收益是那 14 个用例能在任意机器上跑。

    模型选择：配了 `llm_provider_id` 就用指定的那个，否则跟随 AstrBot
    当前对话模型。**推荐显式指定**——长期记忆的归纳风格不该随主体换模型
    而静默漂移，事后也没法归因是哪次换模型让记忆变味的。

    失败一律翻译成契约错误（`Unavailable` / `RateLimited` / …），**不抛裸
    RuntimeError**：外层 `FallbackProvider` 要靠错误码判断「重试这个」还是
    「立刻换下一个」，抛裸异常会让它只能猜。`contracts.errors` 的文档里
    批评过 hermes_core「所有异常都当可重试」，这里不能重犯。
    """

    def __init__(self, plugin: "HermesFleet", provider_id: str = "") -> None:
        self._plugin = plugin
        self.provider_id = str(provider_id or "").strip()

    @property
    def name(self) -> str:
        return self.provider_id or "当前对话模型"

    def _instance(self):
        ctx = getattr(self._plugin, "context", None)
        pid = self.provider_id
        if pid:
            try:
                pm = getattr(ctx, "provider_manager", None)
                insts = getattr(pm, "inst_map", None) or {}
                if pid in insts:
                    return insts[pid]
            except Exception:
                pass
            # 指定的模型还没加载好（热加载/刚改配置）：返回 None 让调用方报错。
            # 单模型配置下这仍然等于「不干活」——但配了降级链时，外层会把它
            # 当成「这个暂时不可用」换下一个，这是用户显式声明过的顺序，
            # 不属于「静默换模型」。
            return None
        try:
            return ctx.get_using_provider()
        except Exception:
            return None

    async def complete(self, messages, *, purpose: str,
                       max_tokens: int | None = None,
                       temperature: float | None = None) -> LlmResult:
        inst = self._instance()
        if inst is None:
            raise Unavailable(
                "没有可用的对话模型"
                + (f"（配置的 {self.provider_id} 未加载）" if self.provider_id else ""),
                retry_after_s=30.0,
                detail={"provider": self.provider_id, "reason": "not_loaded"},
            )
        # contracts.ChatMessage 只有 role/content，这里拍平成单段 prompt：
        # AstrBot 的 text_chat 接受一整段文本，我们自己把 system 段放在最前，
        # 语义等价，且不依赖它的多轮消息格式。
        prompt = "\n\n".join(
            f"[{m.role}]\n{m.content}" if m.role != "user" else m.content
            for m in messages
        )
        t0 = time.perf_counter()
        try:
            resp = await inst.text_chat(
                prompt=prompt, session_id=None, request_max_retries=1
            )
        except TypeError as exc:
            # 老版本 provider 不认 request_max_retries。必须压到 1 次重试：
            # 默认 5 次指数退避会让一次 429 变成一两分钟的静默等待，
            # 外层只看到「超时」，既认不出是限流也没法换模型。
            if "request_max_retries" not in str(exc):
                raise
            t0 = time.perf_counter()
            resp = await inst.text_chat(prompt=prompt, session_id=None)

        # 记下耗时：`LlmResult.latency_ms` 是「模型变慢」的早期预警，
        # hermes_core 缺这个观测，退避逻辑在 provider 侧变慢时完全失效。
        return LlmResult(
            text=str(getattr(resp, "completion_text", "") or ""),
            model=self._model_name(inst),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            finish_reason=_finish_reason(resp),
            usage={},
        )

    def _model_name(self, inst) -> str:
        try:
            return str(inst.meta().model or "")
        except Exception:
            return self.provider_id


def _finish_reason(resp) -> str:
    """尽量读出「为什么停下」。

    只有拿到 `length` 才能确定输出被截断。AstrBot 各 provider 对外的字段
    不统一，所以这里三级尝试；全拿不到就返回空——**不猜**。
    拿不到的后果是截断检测退化，由 `looks_like_complete_array` 兜底。
    """
    for attr in ("finish_reason", "stop_reason"):
        v = getattr(resp, attr, None)
        if v:
            return str(v)
    try:
        return str(resp.raw_completion.choices[0].finish_reason or "")
    except Exception:
        return ""


@register(
    "astrbot_plugin_hermes_fleet",
    "Summer-Ever",
    "百账号隔离的长期记忆系统",
    PLUGIN_VERSION,
)
class HermesFleet(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config

        root = str(self._get_cfg("bot_db_dir", "") or "").strip()
        if not root:
            root = os.path.join(_default_data_dir(), "bots")
        self.registry = BotRegistry(
            root, max_open=int(self._get_cfg("max_open_bots", 24))
        )
        self.svc = MemoryService(
            self.registry,
            group_private_memory=bool(self._get_cfg("group_private_memory", False)),
            recall_k=int(self._get_cfg("recall_k", 5)),
        )
        self._inject_guard: dict[str, float] = {}
        self.provider = self._new_fallback()
        self.summarizer = SessionSummarizer(
            self.provider,
            min_messages=int(self._get_cfg("summarize_min_messages", 8)),
            idle_seconds=float(self._get_cfg("summarize_idle_hours", 6)) * 3600,
            max_sessions=int(self._get_cfg("summarize_sessions_per_bot", 4)),
            max_tokens=int(self._get_cfg("summarize_max_tokens", 900)),
        )
        self._bg_task: asyncio.Task | None = None
        self._bot_cursor = 0
        self.consolidator = MemoryConsolidator(
            self.provider,
            min_active=int(self._get_cfg("consolidate_min_active", 30)),
            max_scopes=int(self._get_cfg("consolidate_scopes_per_bot", 2)),
            max_merges=int(self._get_cfg("consolidate_max_merges", 20)),
        )

    # ------------------------------------------------------------ 工具

    def _get_cfg(self, key: str, default):
        if self.config is None:
            return default
        try:
            return self.config.get(key, default)
        except Exception:
            return default

    def _provider_chain(self) -> list[tuple[str, AstrBotProvider]]:
        """按配置拼出降级链：主模型 → 降级 1 → 降级 2 …

        `llm_provider_ids` 是逗号/换行分隔的 provider id 列表，顺序即优先级；
        `__current__` 表示「跟随主应用当前对话模型」。解析细节见
        `hermes_fleet.chain`——那里能在没有 AstrBot 的机器上被单测覆盖。

        为什么**推荐显式配链**：留空时这条链会跟着主应用的默认模型走，
        于是主应用换一次模型，长期记忆的归纳风格就静默漂移一次，事后
        无法归因。上百个账号同时受影响的场景下，这属于「无声改行为」。
        """
        from hermes_fleet.chain import CURRENT, chain_from_config

        ids = chain_from_config(
            self._get_cfg("llm_provider_ids", ""),
            self._get_cfg("llm_provider_id", ""),
        )
        chain: list[tuple[str, AstrBotProvider]] = []
        for pid in ids:
            inst = AstrBotProvider(self, "" if pid == CURRENT else pid)
            chain.append((inst.name, inst))
        return chain

    def _new_fallback(self) -> FallbackProvider:
        """按当前配置造一条新的降级链实例（初始化与热切换共用）。"""
        return FallbackProvider(
            self._provider_chain(),
            retry_times=int(self._get_cfg("llm_retry_times", 1)),
            retry_base_s=float(self._get_cfg("llm_retry_base_seconds", 0.8)),
            cooldown_s=float(self._get_cfg("llm_cooldown_seconds", 180)),
        )

    def rebuild_provider(self) -> None:
        """按当前配置重建降级链，并把它接到后台任务上。

        为什么要能在运行时换：改完模型链最适合立刻验证（发一条消息、看
        日志）。如果必须重载插件才生效，人就会「先保存着，回头再说」，
        而这类配置忘了重启的代价是整晚的后台总结全部落空。

        # 冷却表随旧实例丢弃：那是「上一套链的故障历史」，
        # 继续沿用会让新选的主模型一上来就背着前一个模型的冷却。
        """
        new = self._new_fallback()
        self.provider = new
        # 后台任务持有的是构造时传进来的 llm 引用，必须一起换掉，
        # 否则「保存了但总结还在用旧模型」——这类不一致只能靠人翻代码发现。
        for worker in (getattr(self, "summarizer", None),
                       getattr(self, "consolidator", None)):
            if worker is not None:
                worker.llm = new

    def chain_config(self) -> dict:
        """当前模型链的「意图」与「来源」，供管理台渲染与判断是否告警。"""
        from hermes_fleet.chain import CURRENT, describe, label, parse, source_of

        raw_chain = self._get_cfg("llm_provider_ids", "")
        raw_single = self._get_cfg("llm_provider_id", "")
        source = source_of(raw_chain, raw_single)
        desc = describe(parse(raw_chain) or parse(raw_single) or [CURRENT])
        desc["source"] = source
        desc["stored"] = str(raw_chain or "")
        desc["primary_label"] = label(desc["primary"])
        return desc

    def providers_catalog(self) -> tuple[list[dict], str]:
        """主应用里已经配置好的对话模型清单 + 当前默认模型 id。

        **只读主应用，不改它**：记忆系统要用哪个模型，由本插件的配置决定。
        这里只是把候选摆在页面上让人挑，所以拿不到（AstrBot 还没启动完、
        或版本差异导致 API 不同）时返回空表，让页面显示「无法读取」，
        而不是猜一个列表出来。
        """
        ctx = getattr(self, "context", None)
        rows: list[dict] = []
        seen: set[str] = set()
        try:
            insts = list(ctx.get_all_providers() or [])
        except Exception:
            insts = []
        for inst in insts:
            try:
                meta = inst.meta()
            except Exception:
                continue
            pid = str(getattr(meta, "id", "") or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            rows.append({
                "id": pid,
                "model": str(getattr(meta, "model", "") or ""),
                "type": str(getattr(meta, "type", "") or ""),
                "kind": str(getattr(getattr(meta, "provider_type", ""), "value", "")
                            or getattr(meta, "provider_type", "") or ""),
            })
        current = ""
        try:
            cur = ctx.get_using_provider()
            if cur is not None:
                current = str(getattr(cur.meta(), "id", "") or "")
        except Exception:
            current = ""
        for row in rows:
            row["current"] = bool(current and row["id"] == current)
        return rows, current

    @staticmethod
    def _platform_id(event: AstrMessageEvent) -> str:
        """取「这是哪个机器人账号」。

        这是本插件最重要的一次取字段。依据是 AstrBot 的适配器元数据：
        `PlatformMetadata.id` 来自面板里每个适配器配置的 `id` 字段
        （`qq_official` 适配器同样如此），且 `get_platform_id()` 的注释
        明确写着「同时运行多个同类型适配器时 ID 唯一」。

        三级兜底：`id` 可能是 None（老配置没写），`get_platform_id` 在
        旧版本可能不存在，最后从 umo 的第一段取——umo 形如
        `qq_official:FriendMessage:xxx`，第一段就是 platform id。
        """
        try:
            pid = event.get_platform_id()
            if pid:
                return str(pid)
        except Exception:
            pass
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        head = umo.split(":", 1)[0].strip()
        return head or "default"

    def _scene_args(self, event: AstrMessageEvent) -> dict:
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        gid = str(event.get_group_id() or "")
        # 群聊判定不能只看 get_group_id：部分适配器私聊也带会话 id，
        # 而 umo 里的 message_type 段是稳定的。
        is_group = bool(gid) or "group" in umo.lower()
        return {
            "platform_id": self._platform_id(event),
            "sender_id": str(event.get_sender_id() or ""),
            "group_id": gid,
            "is_group": is_group,
        }

    # ------------------------------------------------------------ 注入

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            if not self._get_cfg("memory_injection", True):
                return
            query = (event.message_str or "").strip()
            if not query:
                return

            key = f"{event.unified_msg_origin}\u0000{hash(query)}"
            now = time.time()
            if now - self._inject_guard.get(key, 0.0) < INJECT_TTL:
                return

            scene = await self.svc.scene(**self._scene_args(event))
            block = await self.svc.inject_block(scene, query)
            if not block:
                return
            # 写入时机放在构建成功之后：先写后构建的话，一次偶发异常
            # 会让同一个问题在整个窗口内都注入不进来——偶发失败变成静默失忆。
            self._inject_guard[key] = now
            if len(self._inject_guard) > 400:
                cutoff = now - INJECT_TTL
                self._inject_guard = {
                    k: v for k, v in self._inject_guard.items() if v > cutoff
                }

            if TextPart is not None and hasattr(req, "extra_user_content_parts"):
                part = TextPart(text=block)
                if hasattr(part, "mark_as_temp"):
                    part.mark_as_temp()
                req.extra_user_content_parts.append(part)
            else:  # pragma: no cover - 仅老版本
                req.system_prompt = (req.system_prompt or "") + "\n" + block
        except Exception:
            # 注入失败绝不能让整轮对话失败：这里只记录，正常继续。
            logger.error(f"[hermes_fleet] 注入失败:\n{traceback.format_exc(limit=6)}")

    # ------------------------------------------------------------ 落库

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        try:
            if not self._get_cfg("capture_messages", True):
                return
            user_text = (event.message_str or "").strip()
            reply = str(getattr(resp, "completion_text", "") or "").strip()
            if not user_text and not reply:
                return
            scene = await self.svc.scene(**self._scene_args(event))
            await self.svc.capture_turn(
                scene, str(event.unified_msg_origin or ""), user_text, reply
            )
        except Exception:
            logger.error(f"[hermes_fleet] 落库失败:\n{traceback.format_exc(limit=6)}")

    # ------------------------------------------------------------ 工具面

    @filter.llm_tool(name="search_memory")
    async def tool_search_memory(self, event: AstrMessageEvent, query: str) -> str:
        """检索这个机器人账号保存的长期记忆。用户问「你记得吗 / 我之前说过什么 / 我的偏好是什么」时，先调用本工具核实再回答。

        检索不到就如实说没有记录，不要据此断言「记忆被清空了」——本工具只覆盖归档记忆，每轮注入的常驻设定不在其中。

        Args:
            query (string): 检索关键词，用一两个区分度高的词。没命中就换词再搜一次，不要把一长串词塞进同一次查询（会稀释相关性）
        """
        scene = await self.svc.scene(**self._scene_args(event))
        text = await self.svc.search(scene, str(query or ""))
        return text or "没有找到相关记忆。"

    @filter.llm_tool(name="save_memory")
    async def tool_save_memory(self, event: AstrMessageEvent, content: str,
                               kind: str = "fact") -> str:
        """把值得长期记住的事存成归档记忆（自动归属到当前发言用户）。

        归档记忆需要时由 search_memory 检索，不会每轮自动带上。
        「每轮都用得上」的稳定信息（怎么称呼他、他是谁、偏好与忌讳、回复风格、规则）不要用本工具，那是常驻词条的事，用 update_resident 维护。

        Args:
            content (string): 记忆内容，一句简洁的话
            kind (string): 类型，fact/preference/profile/summary 之一
        """
        scene = await self.svc.scene(**self._scene_args(event))
        text = str(content or "").strip()
        if not text:
            return "内容为空，未保存。"
        memory = await self.svc.save(scene, text, kind=str(kind or "fact"))
        return f"已保存（编号 #{int(memory.id)}）：{memory.content}"

    @filter.llm_tool(name="update_resident")
    async def tool_update_resident(self, event: AstrMessageEvent, block: str,
                                   text: str) -> str:
        """整段改写一个常驻词条（传空文本 = 清空该词条）。

        常驻是六个词条的纯文本，每轮都注入：address（称呼）、assistant（助手设定）、
        user（用户信息）、preference（偏好与忌讳）、style（回复风格）、rules（规则与避雷）。
        本工具用新文本**整段覆盖**该词条——词条没有编号、没有「一条一条」，
        要改哪句话，就在现文基础上把那句话改掉后整段写回。

        只在用户明确要求记住/修改/遗忘这类稳定信息时调用（改称呼、改偏好、
        情况变了、「别记着我的XX」）：先对照本轮注入里的 [长期稳定信息] 词条现文，
        把变化合并进整段文本再写回；整段都不需要了才传空文本清空。
        用户没提的事不要擅自改写常驻——后台总结碰不到常驻，你也不代替用户做主。

        词条按场景隔离：私聊改的是私聊那套，群里改的是本群那套，互不影响。

        Args:
            block (string): 词条名，六选一——address/assistant/user/preference/style/rules
                （或中文：称呼/助手设定/用户信息/偏好与忌讳/回复风格/规则与避雷）
            text (string): 该词条的完整新文本（整段替换，不是追加）；空文本 = 清空该词条
        """
        scene = await self.svc.scene(**self._scene_args(event))
        try:
            entry = await self.svc.set_resident(scene, str(block or ""), str(text or ""))
        except ValueError as exc:
            return str(exc)
        label = BLOCK_LABELS.get(entry.block, entry.block.value)
        where = "本群" if scene.is_group else "私聊"
        if not entry.text:
            return f"已清空{where}的「{label}」词条。"
        return (
            f"已更新{where}的「{label}」词条"
            f"（{len(entry.text)}/{MAX_RESIDENT_BLOCK_CHARS} 字）：{entry.text}"
        )

    @filter.llm_tool(name="forget_memory")
    async def tool_forget_memory(self, event: AstrMessageEvent, memory_id: int) -> str:
        """删掉一条归档记忆。用户明确要求「忘掉这件事 / 别记着这个」时使用。

        编号来自 search_memory 的结果（形如 #12）。

        Args:
            memory_id (number): 要删除的记忆编号
        """
        scene = await self.svc.scene(**self._scene_args(event))
        if await self.svc.forget(scene, int(memory_id)):
            return f"已删除记忆 #{int(memory_id)}。"
        return f"没有找到编号 #{int(memory_id)} 的可删除记忆。"

    # ------------------------------------------------------------ 运维命令

    @filter.command_group("fleet")
    def fleet(self):
        """多账号记忆运维"""

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """权限检查必须 fail-closed：判断不出来就当「不是」。"""
        try:
            if event.is_admin():
                return True
        except Exception:
            pass
        role = str(getattr(event, "role", "") or "").strip().lower()
        if role in ("admin", "owner", "administrator"):
            return True
        try:
            conf = self.context.get_config() or {}
            admins = conf.get("admins_id") or conf.get("admin_id") or []
            if isinstance(admins, str):
                admins = [admins]
            if str(event.get_sender_id()) in {str(a) for a in admins}:
                return True
        except Exception:
            pass
        return False

    @fleet.command("状态")
    async def fleet_status(self, event: AstrMessageEvent):
        st = self.registry.stats()
        pid = self._platform_id(event)
        scene = await self.svc.scene(**self._scene_args(event))
        mine = await self.svc.bot_stats(pid)
        scope = f"群 {scene.group_id}" if scene.is_group else "私聊"
        note = self.registry.note_of(pid)
        try:
            from hermes_fleet.chain import SOURCE_LABELS

            source = SOURCE_LABELS.get(self.chain_config().get("source"), "")
        except Exception:
            source = ""
        yield event.plain_result(
            f"【本账号】{note + '（' + pid + '）' if note else pid}\n"
            f"当前场景：{scope}\n"
            f"归档记忆 {mine['memories']} 条，常驻词条 {mine['resident']} 条，"
            f"消息 {mine['messages']} 条（未总结 {mine['unsummarized']}）\n"
            f"【模型链】{self.provider.summary()}"
            + (f"（{source}）" if source else "") + "\n"
            f"【全局】库目录 {st['root']}，已打开 {st['opened']}/{st['max_open']}，"
            f"磁盘上 {len(st['files_on_disk'])} 个账号库"
        )

    @fleet.command("我是谁")
    async def fleet_whoami(self, event: AstrMessageEvent):
        scene = await self.svc.scene(**self._scene_args(event))
        yield event.plain_result(
            f"账号（platform id）：{scene.bot_id}\n"
            f"归属（owner_id）：{scene.owner_id}\n"
            f"场景：{'群 ' + scene.group_id if scene.is_group else '私聊'}\n"
            "注：QQ 官方适配器的 sender_id 是 per-bot 的 openid，"
            "同一个人在群里与私聊是两个值。"
        )

    @fleet.command("常驻")
    async def fleet_resident(self, event: AstrMessageEvent):
        """看当前场景（私聊 / 本群）的六个常驻词条。

        存在的意义是「可观测」：用户说「以后叫我阿明」之后，得有个地方能
        确认模型到底写没写进去。没有这条命令时，唯一的检查手段是进面板翻
        数据库，等于「记住了吗」只能靠机器人自己说。
        """
        scene = await self.svc.scene(**self._scene_args(event))
        values = await self.svc.resident_snapshot(scene)
        where = f"本群（{scene.group_id}）" if scene.is_group else "私聊"
        lines = [f"【{where}常驻词条】"]
        filled = 0
        for block in ResidentBlock.ordered():
            label = BLOCK_LABELS.get(block, block.value)
            text = str(values.get(block.value, "") or "").strip()
            if text:
                filled += 1
                lines.append(f"- {label}：{text}")
            else:
                lines.append(f"- {label}：（空）")
        lines.append(
            f"共 {filled}/6 条。改法：直接对我说「以后叫我阿明」"
            "「别记着我不吃辣」，我会自己改这一套。"
        )
        yield event.plain_result("\n".join(lines))

    @fleet.command("总结")
    async def fleet_summarize(self, event: AstrMessageEvent):
        """手动跑一轮总结（管理员）。用来验证模型链通不通，不必等 15 分钟。"""
        if not self._is_admin(event):
            yield event.plain_result("该命令需要管理员权限。")
            return
        yield event.plain_result("开始总结，完成后回报结果……")
        try:
            note = await self.summarize_pass()
        except Exception as exc:
            yield event.plain_result(f"总结失败：{exc}")
            return
        yield event.plain_result(f"总结完成：{note}")

    @fleet.command("整理")
    async def fleet_consolidate(self, event: AstrMessageEvent):
        """手动跑一轮整理：把重复记忆合并（管理员）。"""
        if not self._is_admin(event):
            yield event.plain_result("该命令需要管理员权限。")
            return
        yield event.plain_result("开始整理，完成后回报结果……")
        try:
            note = await self.consolidate_pass()
        except Exception as exc:
            yield event.plain_result(f"整理失败：{exc}")
            return
        yield event.plain_result(f"整理完成：{note}")

    @fleet.command("账号列表")
    async def fleet_bots(self, event: AstrMessageEvent):
        if not self._is_admin(event):
            yield event.plain_result("该命令需要管理员权限。")
            return
        st = self.registry.stats()
        files = st["files_on_disk"] or []
        yield event.plain_result(
            "库目录：" + str(st["root"]) + "\n"
            + ("\n".join(f"- {f}" for f in files) if files else "（还没有任何账号写过数据）")
        )

    @fleet.command("删除账号记忆")
    async def fleet_purge(self, event: AstrMessageEvent):
        if not self._is_admin(event):
            yield event.plain_result("该命令需要管理员权限。")
            return
        parts = (event.message_str or "").split()
        if len(parts) < 2:
            yield event.plain_result(
                "用法：/fleet 删除账号记忆 <适配器id>\n"
                "这会物理删除该账号的整个记忆库（含消息、常驻、归档），不可撤销。"
            )
            return
        target = parts[1].strip()
        ok = self.registry.purge(target)
        yield event.plain_result(
            f"已删除账号 {target} 的记忆库。" if ok else f"账号 {target} 没有可删除的记忆库。"
        )

    # ------------------------------------------------------------ 后台总结

    async def summarize_pass(self, *, max_bots: int = 25) -> str:
        """跑一轮总结：轮流处理各账号，每个账号最多几个会话。

        为什么要「轮流」而不是「每轮都从头扫」：100 个账号时，如果每轮都从
        第一个开始且中途到点退出，后面的账号永远轮不到。游标存在内存里，
        重启归零——可以接受，因为重启本来就是冷启动，重新从第一个开始也合理。

        为什么单账号失败要吞掉：一个账号的库文件损坏（磁盘满、被外部删了）
        不该让其余 99 个账号永远停止总结。
        """
        if not bool(self._get_cfg("summarize_enabled", True)):
            return "自动总结已关闭（配置里打开 summarize_enabled 后生效）"

        bots = self.registry.known_bots()
        if not bots:
            return "还没有任何账号产生过数据"

        start = self._bot_cursor % len(bots)
        order = bots[start:] + bots[:start]
        picked = order[:max(1, max_bots)]
        self._bot_cursor = (start + len(picked)) % max(1, len(bots))

        total = SummaryOutcome()
        cons = ConsolidateOutcome()
        # 整理只跑在「刚产生新记忆」的账号上：新条目才是重复的来源，
        # 没写新记忆的库每轮都扫一遍纯属空转。
        consolidate_on = bool(self._get_cfg("consolidate_enabled", True))
        for pid in picked:
            try:
                db = self.registry.get(pid)
                outcome = await self.summarizer.run_bot(
                    self.svc.bot_of(pid),
                    messages=db.messages,
                    memories=db.memories,
                    identities=db.identities,
                )
            except Exception as exc:
                logger.error(f"[hermes_fleet] 账号 {pid} 总结失败：{exc}")
                continue
            total.add(outcome)
            if consolidate_on and outcome.memories > 0:
                try:
                    cons.add(await self.consolidator.run_bot(
                        self.svc.bot_of(pid), memories=db.memories))
                except Exception as exc:
                    logger.error(f"[hermes_fleet] 账号 {pid} 整理失败：{exc}")
        note = str(total)
        if cons.merged or cons.failures:
            note += "；整理：" + str(cons)
        return note

    async def consolidate_pass(self, *, max_bots: int = 10) -> str:
        """手动跑一轮整理（管理员）。用来验证整理逻辑，不必等后台。

        与总结不同，整理不要求有新记忆：手动命令的用途正是「把现有重复清一遍」，
        所以这里绕开「条数增长」的闸，但仍然受每轮合并上限约束。
        """
        bots = self.registry.known_bots()
        if not bots:
            return "还没有任何账号产生过数据"
        total = ConsolidateOutcome()
        for pid in bots[: max(1, max_bots)]:
            try:
                db = self.registry.get(pid)
                # 重置该账号的节流状态，让手动命令立刻生效
                total.add(await self.consolidator.run_bot(
                    self.svc.bot_of(pid), memories=db.memories, force=True))
            except Exception as exc:
                logger.error(f"[hermes_fleet] 账号 {pid} 整理失败：{exc}")
                total.failures += 1
                total.last_error = str(exc)
        return str(total)

    async def _bg_loop(self):
        """后台循环。间隔可配，默认 15 分钟。

        用 `asyncio.sleep` 而不是 `while True: await work()`：总结是长任务，
        连续跑会把事件循环占满，反过来拖慢所有账号的对话响应。
        每轮之间必须真的让出控制权。
        """
        interval = max(1.0, float(self._get_cfg("summarize_interval_minutes", 15))) * 60
        # 启动后先等一会儿：刚重启时所有适配器都在重连，此时抢 CPU 只会
        # 让连接建立更慢。
        await asyncio.sleep(min(60.0, interval))
        while True:
            try:
                note = await self.summarize_pass()
                if "新增记忆" in note or "失败" in note:
                    logger.info(f"[hermes_fleet] 自动总结：{note}")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(f"[hermes_fleet] 自动总结异常：\n{traceback.format_exc(limit=5)}")
            await asyncio.sleep(interval)

    # ------------------------------------------------------------ 生命周期

    async def initialize(self):
        st = self.registry.stats()
        logger.info(
            f"[hermes_fleet] v{PLUGIN_VERSION} 已加载，库目录 {st['root']}，"
            f"已有 {len(st['files_on_disk'])} 个账号库，"
            f"最多同时打开 {st['max_open']} 个"
        )
        self._register_web_apis()
        if bool(self._get_cfg("summarize_enabled", True)):
            self._bg_task = asyncio.create_task(self._bg_loop())
            logger.info("[hermes_fleet] 自动总结已启动")

    def _register_web_apis(self) -> None:
        """挂载管理台接口（页面见 views/fleet/）。

        导入用三级兜底，理由同 hermes_core：真实部署里插件未必以包的形式
        被加载，`from .webapi import ...` 会失败，需要退化到同名模块或
        直接按文件加载。任何一种都能工作时，不要把它们删掉「简化」——
        删掉之后只在某些部署方式下才炸，而且炸在加载期，很难归因。
        """
        try:
            try:
                from .webapi import FleetWebAPI
            except ImportError:
                try:
                    from webapi import FleetWebAPI
                except ImportError:
                    import importlib.util

                    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "webapi.py")
                    spec = importlib.util.spec_from_file_location("fleet_webapi", path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    FleetWebAPI = mod.FleetWebAPI
            self._web_api = FleetWebAPI(self, PLUGIN_VERSION)
            self._web_api.register()
            logger.info("[hermes_fleet] 管理台接口已注册：/views/fleet/")
        except Exception:
            # 页面注册失败不能影响记忆功能本身：注入、落库、总结都不依赖它。
            logger.error(f"[hermes_fleet] 注册管理台接口失败：\n"
                         f"{traceback.format_exc(limit=5)}")

    async def terminate(self):
        """关停时先停后台任务，再逐库 checkpoint + 关连接。

        顺序不能反：后台任务可能正拿着某个库的连接在写，先关库会让它
        在 `closed database` 上抛异常；虽然不影响数据（写是事务的），
        但会糊住真正的关闭错误。
        """
        if self._bg_task is not None:
            self._bg_task.cancel()
            try:
                await self._bg_task
            except (asyncio.CancelledError, Exception):
                pass
            self._bg_task = None
        try:
            self.registry.close_all()
        except Exception:
            logger.error(f"[hermes_fleet] 关闭记忆库失败:\n{traceback.format_exc(limit=4)}")
