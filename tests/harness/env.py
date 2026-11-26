"""真跑 `main.py` 的测试环境：临时库目录 + 假模型 + 假事件。

与 `tests/integration/` 的区别：那边只拼 `hermes_fleet` / `hermes_recall`
这些包，`main.py`（1245 行）与 `webapi.py`（937 行）完全没被执行过。
这里用 `spec_from_file_location` 把 `main.py` 当普通模块加载，于是它的
导入、装饰器、字段翻译、注入去重、落库、路由注册全部是**真的**在跑，
只把「AstrBot 本体」换成 `astrbot_stub`。

刻意不做的事：
- 不调 `initialize()`：它会起后台总结任务，测试里要手动控制总结时机。
  需要路由时直接调 `_register_web_apis()`。
- 不模拟会话历史：`on_llm_request` 只看 `message_str` 与 umo。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN_PATH = os.path.join(ROOT, "main.py")

_main_module = None


def _add_src_paths() -> None:
    pkgs = os.path.join(ROOT, "packages")
    if not os.path.isdir(pkgs):
        return
    for name in sorted(os.listdir(pkgs)):
        src = os.path.join(pkgs, name, "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)
    if ROOT not in sys.path:
        # `_register_web_apis` 的二级兜底是 `from webapi import FleetWebAPI`，
        # 靠插件根目录在 sys.path 上才成立。
        sys.path.insert(0, ROOT)


def load_main():
    """加载并缓存 `main.py`。真实 AstrBot 存在时抛 SkipTest。"""
    global _main_module
    if _main_module is not None:
        return _main_module

    from . import astrbot_stub

    if not astrbot_stub.install():
        raise unittest.SkipTest(
            "宿主已安装真实 AstrBot，跳过基于桩的端到端用例（请在干净环境跑）"
        )

    _add_src_paths()
    name = "fleet_e2e_main_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, MAIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _main_module = module
    return module


class FakeMeta:
    def __init__(self, pid: str = "fake-provider", model: str = "fake-model") -> None:
        self.id = pid
        self.model = model
        self.type = "openai_chat_completion"
        self.provider_type = ""


class FakeModel:
    """`ctx.get_using_provider()` 返回的实例。

    `replies` 是脚本化的回复队列，按调用顺序弹出；弹空之后回 `"[]"`——
    对总结 worker 来说这是「这段对话没有可记的内容」，会正常标记消息，
    不会因为缺省回复把用例拖进重试路径。
    """

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.prompts: list[str] = []
        self._meta = FakeMeta()

    async def text_chat(self, prompt: str = "", session_id: str | None = None,
                        request_max_retries: int = 1, **kwargs):
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else "[]"
        return types.SimpleNamespace(completion_text=text, finish_reason="stop")

    def meta(self) -> FakeMeta:
        return self._meta


class FakeEvent:
    """只覆盖 `main.py` 真正读的字段。"""

    def __init__(self, *, platform_id: str = "qq-01", sender_id: str = "u1",
                 group_id: str = "", message: str = "",
                 umo: str | None = None, admin: bool = False) -> None:
        self._platform_id = platform_id
        self._sender_id = sender_id
        self._group_id = group_id
        self.message_str = message
        if umo is None:
            kind = "GroupMessage" if group_id else "FriendMessage"
            tail = group_id or sender_id
            umo = f"{platform_id}:{kind}:{tail}"
        self.unified_msg_origin = umo
        self._admin = admin
        self.role = ""
        self.results: list[object] = []

    def get_platform_id(self) -> str:
        return self._platform_id

    def get_sender_id(self) -> str:
        return self._sender_id

    def get_group_id(self) -> str:
        return self._group_id

    def get_messages(self) -> list[object]:
        return []

    def is_admin(self) -> bool:
        return self._admin

    def plain_result(self, text: object) -> dict[str, object]:
        return {"type": "plain", "text": str(text)}

    def chain_result(self, chain: object) -> dict[str, object]:
        return {"type": "chain", "chain": chain}


class PluginEnv:
    """一个用例一份临时数据目录 + 一个插件实例。"""

    def __init__(self, *, config: dict | None = None,
                 replies: list[str] | None = None,
                 db_root: str | None = None) -> None:
        self.tmp = tempfile.mkdtemp(prefix="hermes_e2e_")
        self.db_root = db_root or os.path.join(self.tmp, "bots")
        os.makedirs(self.db_root, exist_ok=True)
        self.main = load_main()
        self.model = FakeModel(replies)
        self.context = self.main.Context()
        self.context.use_provider(self.model)

        conf = {
            "bot_db_dir": self.db_root,
            "max_open_bots": 8,
            "recall_k": 5,
            "group_private_memory": False,
            "memory_injection": True,
            "capture_messages": True,
            "inject_to_system_prompt": False,
            "summarize_enabled": True,
            # 条数闸调到 1：用例只有一轮对话，不用等 8 条。
            "summarize_min_messages": 1,
            "summarize_idle_hours": 0,
            "summarize_sessions_per_bot": 4,
            "summarize_max_tokens": 900,
            "consolidate_enabled": False,
            "consolidate_min_active": 999,
        }
        conf.update(config or {})
        self.config = conf
        self.plugin = self.main.HermesFleet(self.context, conf)

        # `_export_dir()` 默认落在 `data/plugin_data/...`（相对 CWD）。测试里
        # 换成临时目录，否则备份/导出接口会把文件写进仓库，污染工作区。
        exports = os.path.join(self.tmp, "exports")
        os.makedirs(exports, exist_ok=True)
        self.plugin._export_dir = lambda: exports

    # ---- 便捷构造

    def event(self, **kwargs) -> FakeEvent:
        return FakeEvent(**kwargs)

    async def turn(self, event: FakeEvent, text: str, reply: str = "收到"):
        """跑一轮完整的「请求注入 → 回复落库」，返回本次的 ProviderRequest。"""
        event.message_str = text
        req = self.main.ProviderRequest(system_prompt="你是助手")
        await self.plugin.on_llm_request(event, req)
        resp = self.main.LLMResponse(completion_text=reply)
        await self.plugin.on_llm_response(event, resp)
        return req

    async def stats(self, platform_id: str = "qq-01") -> dict[str, int]:
        return await self.plugin.svc.bot_stats(platform_id)

    def register_web_apis(self) -> dict[str, dict]:
        self.plugin._register_web_apis()
        return self.context.routes

    # ---- 清理

    async def close(self, *, keep: bool = False) -> None:
        """关闭库连接并删临时目录。`keep=True` 留给「重启后复用同一个库」用例。"""
        await self.plugin.terminate()
        if keep:
            return
        for _ in range(5):
            shutil.rmtree(self.tmp, ignore_errors=True)
            if not os.path.exists(self.tmp):
                break
            import time

            time.sleep(0.05)

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
