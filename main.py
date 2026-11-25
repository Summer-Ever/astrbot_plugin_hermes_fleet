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


_bootstrap_paths()

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

from hermes_fleet import BotRegistry, MemoryService  # noqa: E402

PLUGIN_VERSION = "0.1.0"

# 同一句话的注入去重窗口（秒）。一次用户发言可能触发多轮 LLM 请求
# （工具调用循环），每轮都注入等于把同一段记忆反复塞进上下文。
# 45 秒覆盖一次工具循环，又不会让「换个说法再问一遍」持续失忆。
INJECT_TTL = 45


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

    # ------------------------------------------------------------ 工具

    def _get_cfg(self, key: str, default):
        if self.config is None:
            return default
        try:
            return self.config.get(key, default)
        except Exception:
            return default

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
        """维护每轮都会注入的常驻词条。整段覆盖，不是追加。

        用户要求记住/修改/忘掉稳定信息，或本轮对话表明某词条已过时，就调用本工具。
        写入前先看注入块里的现有内容，把该保留的部分一起写进去——这是覆盖，不是追加，漏写等于删掉。

        Args:
            block (string): 词条名，address/assistant/user/preference/style/rules 之一，分别对应 称呼/助手设定/用户信息/偏好与忌讳/回复风格/规则与避雷
            text (string): 该词条的完整新文本，传空字符串表示清空这个词条
        """
        scene = await self.svc.scene(**self._scene_args(event))
        try:
            await self.svc.set_resident(scene, str(block or ""), str(text or ""))
        except ValueError as exc:
            return f"{exc}"
        scope = "本群" if scene.is_group else "私聊"
        return f"已更新常驻词条（{scope}）：{block}{'（已清空）' if not str(text or '').strip() else ''}"

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
        yield event.plain_result(
            f"【本账号】{pid}\n"
            f"当前场景：{scope}\n"
            f"归档记忆 {mine['memories']} 条，常驻词条 {mine['resident']} 条，"
            f"消息 {mine['messages']} 条（未总结 {mine['unsummarized']}）\n"
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

    # ------------------------------------------------------------ 生命周期

    async def initialize(self):
        st = self.registry.stats()
        logger.info(
            f"[hermes_fleet] v{PLUGIN_VERSION} 已加载，库目录 {st['root']}，"
            f"已有 {len(st['files_on_disk'])} 个账号库，"
            f"最多同时打开 {st['max_open']} 个"
        )

    async def terminate(self):
        """关停时逐库 checkpoint + 关连接。

        不关的话，最后一次写入可能还留在 `-wal` 里：进程退出后文件句柄
        释放、数据不会丢，但备份时只拷 `.db` 就会缺最近的记录。
        """
        try:
            self.registry.close_all()
        except Exception:
            logger.error(f"[hermes_fleet] 关闭记忆库失败:\n{traceback.format_exc(limit=4)}")
