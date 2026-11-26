"""AstrBot 4.x 的最小运行时桩。

## 为什么需要它

`main.py` 里的 1200 行是「事件字段 → service → 塞回请求/回复」的翻译层，
真正的串号、注入去重、落库、路由注册都发生在这里。但宿主没装 AstrBot 时
这一层**一行都执行不到**（既有测试全部绕过 `main.py`），于是最需要保证的
那条链路反而零覆盖。

这个桩只实现被 `main.py` / `webapi.py` 真正用到的面：装饰器、`Context`、
`ProviderRequest`、`TextPart`、`request/json_response`。它**不模拟** AstrBot
的调度、会话历史、适配器——那些不是本插件该负责的东西，模拟出来反而会
让人误以为「测过了真实宿主」。

## 只在没有真 AstrBot 时安装

`install()` 先看 `astrbot` 能否 import。真实环境里装了 AstrBot 就原样使用，
桩不参与；这时 `env.load_main()` 会抛 SkipTest，避免用假的 `Star` 跑真实
插件（真 `Star.__init__` 会要求更多宿主状态）。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

_INSTALLED = False


class StubLogger:
    """收集日志，不往 stderr 打。失败用例想查原因可以直接读 `records`。"""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _add(self, level: str, msg: object) -> None:
        self.records.append((level, str(msg)))

    def debug(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._add("debug", msg)

    def info(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._add("info", msg)

    def warning(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._add("warning", msg)

    def error(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._add("error", msg)

    def clear(self) -> None:
        self.records.clear()


class CommandGroup:
    """`@filter.command_group("fleet")` 的返回值。

    真实实现是一个带 `command()` 的对象；这里只保留「能继续挂子命令」和
    「子命令可被测试枚举」两件事。
    """

    def __init__(self, name: str, func: Any) -> None:
        self.name = name
        self.func = func
        self.handlers: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def command(self, name: str, *args: Any, alias: Any = None,
                **kwargs: Any):
        def decorator(fn):
            fn.__command__ = name
            fn.__command_group__ = self.name
            # 宿主支持 `alias={"旧名"}`：别名与主名注册同一个处理函数。
            for cmd in [name, *(sorted(alias) if alias else [])]:
                self.handlers[cmd] = fn
            return fn

        return decorator


class Filter:
    """`astrbot.api.event.filter`。装饰器只记元数据，不改变函数行为。"""

    def command(self, name: str, *args: Any, **kwargs: Any):
        def decorator(fn):
            fn.__command__ = name
            return fn

        return decorator

    def command_group(self, name: str, *args: Any, **kwargs: Any):
        def decorator(fn):
            return CommandGroup(name, fn)

        return decorator

    def llm_tool(self, name: str = "", *args: Any, **kwargs: Any):
        def decorator(fn):
            fn.__llm_tool__ = name or getattr(fn, "__name__", "")
            return fn

        return decorator

    def on_llm_request(self, *args: Any, **kwargs: Any):
        def decorator(fn):
            fn.__on_llm_request__ = True
            return fn

        return decorator

    def on_llm_response(self, *args: Any, **kwargs: Any):
        def decorator(fn):
            fn.__on_llm_response__ = True
            return fn

        return decorator


class AstrMessageEvent:
    """基类占位。测试里的假事件不继承它，`main.py` 只做属性访问。"""


class Context:
    """插件与宿主之间唯一的接口面。"""

    def __init__(self) -> None:
        # route -> {handler, methods, desc}，供路由扫雷与注册断言
        self.routes: dict[str, dict[str, Any]] = {}
        self.provider_manager = types.SimpleNamespace(inst_map={})
        self._using_provider: Any = None
        self._config: dict[str, Any] = {}

    # ---- Web API

    def register_web_api(self, route: str, handler: Any,
                         methods: list[str], desc: str) -> None:
        self.routes[str(route)] = {
            "handler": handler,
            "methods": list(methods),
            "desc": desc,
        }

    # ---- 配置与模型

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    def use_provider(self, inst: Any) -> None:
        self._using_provider = inst
        pid = ""
        try:
            pid = str(inst.meta().id or "")
        except Exception:
            pass
        if pid:
            self.provider_manager.inst_map[pid] = inst

    def get_using_provider(self) -> Any:
        return self._using_provider

    def get_all_providers(self) -> list[Any]:
        return [self._using_provider] if self._using_provider is not None else []


class Star:
    def __init__(self, context: Context | None = None) -> None:
        self.context = context


def register(plugin_name: str, author: str, desc: str, version: str):
    def decorator(cls):
        cls.plugin_name = plugin_name
        cls.plugin_author = author
        cls.plugin_desc = desc
        cls.plugin_version = version
        return cls

    return decorator


class TextPart:
    """只实现 `main._place_block` 需要的两个行为：`text` 与 `mark_as_temp()`。"""

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        self.text = text
        self.is_temp = False

    def mark_as_temp(self) -> "TextPart":
        self.is_temp = True
        return self


class ProviderRequest:
    def __init__(self, system_prompt: str = "",
                 extra_user_content_parts: list[Any] | None = None) -> None:
        self.system_prompt = system_prompt
        self.extra_user_content_parts = list(extra_user_content_parts or [])


class LLMResponse:
    def __init__(self, completion_text: str = "") -> None:
        self.completion_text = completion_text


class FakeRequest:
    """`astrbot.api.web.request` 的替身。

    `query` 是普通 dict，`json()` 回放 `payload`。每个用例在调用 handler
    之前重置一次即可——真实 AstrBot 也是「一次请求一个 request 对象」。
    """

    def __init__(self) -> None:
        self.query: dict[str, Any] = {}
        self.payload: dict[str, Any] = {}

    def reset(self, query: dict[str, Any] | None = None,
              payload: dict[str, Any] | None = None) -> None:
        self.query = dict(query or {})
        self.payload = dict(payload or {})

    async def json(self, default: Any = None) -> Any:
        if self.payload:
            return dict(self.payload)
        return {} if default is None else default


request = FakeRequest()


def json_response(data: Any, status_code: int = 200) -> dict[str, Any]:
    return {"__kind__": "json", "status": int(status_code), "data": data}


def error_response(message: str = "", status_code: int = 400) -> dict[str, Any]:
    return {"__kind__": "error", "status": int(status_code), "message": str(message)}


def file_response(path: str, filename: str = "", content_type: str = "",
                  status_code: int = 200) -> dict[str, Any]:
    """`astrbot.api.web.file_response` 的替身（真实宿主返回 FileResponse）。"""
    return {
        "__kind__": "file",
        "status": int(status_code),
        "path": str(path),
        "filename": str(filename or ""),
        "content_type": str(content_type or ""),
    }


logger = StubLogger()


def _new_module(name: str, *, package: bool = False) -> types.ModuleType:
    mod = types.ModuleType(name)
    if package:
        mod.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


def install() -> bool:
    """安装桩。真实 AstrBot 存在时返回 False（调用方应跳过 E2E）。"""
    global _INSTALLED
    if _INSTALLED:
        return True
    if importlib.util.find_spec("astrbot") is not None:
        return False

    astrbot = _new_module("astrbot", package=True)
    api = _new_module("astrbot.api", package=True)
    core = _new_module("astrbot.core", package=True)
    agent = _new_module("astrbot.core.agent", package=True)
    event_mod = _new_module("astrbot.api.event")
    star_mod = _new_module("astrbot.api.star")
    provider_mod = _new_module("astrbot.api.provider")
    web_mod = _new_module("astrbot.api.web")
    message_mod = _new_module("astrbot.core.agent.message")

    api.logger = logger  # type: ignore[attr-defined]
    api.event = event_mod  # type: ignore[attr-defined]
    api.star = star_mod  # type: ignore[attr-defined]
    api.provider = provider_mod  # type: ignore[attr-defined]
    api.web = web_mod  # type: ignore[attr-defined]
    astrbot.api = api  # type: ignore[attr-defined]
    astrbot.core = core  # type: ignore[attr-defined]
    core.agent = agent  # type: ignore[attr-defined]
    agent.message = message_mod  # type: ignore[attr-defined]

    event_mod.filter = Filter()  # type: ignore[attr-defined]
    event_mod.AstrMessageEvent = AstrMessageEvent  # type: ignore[attr-defined]

    star_mod.Context = Context  # type: ignore[attr-defined]
    star_mod.Star = Star  # type: ignore[attr-defined]
    star_mod.register = register  # type: ignore[attr-defined]
    # 不提供 StarTools：main.py 会走「配置里没写 bot_db_dir 时用相对路径」
    # 的兜底分支，而 E2E 永远显式传 bot_db_dir，不需要真去建宿主数据目录。

    provider_mod.ProviderRequest = ProviderRequest  # type: ignore[attr-defined]
    provider_mod.LLMResponse = LLMResponse  # type: ignore[attr-defined]

    web_mod.request = request  # type: ignore[attr-defined]
    web_mod.json_response = json_response  # type: ignore[attr-defined]
    web_mod.error_response = error_response  # type: ignore[attr-defined]
    web_mod.file_response = file_response  # type: ignore[attr-defined]

    message_mod.TextPart = TextPart  # type: ignore[attr-defined]

    _INSTALLED = True
    return True
