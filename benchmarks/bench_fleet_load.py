#!/usr/bin/env python3
"""等价环境下的记忆插件压力测试（不需要 AstrBot）。

## 这个脚本在测什么

`main.py` 只是把 AstrBot 事件翻译成 `hermes_fleet` 的调用，真正的热路径
（身份解析 → 常驻注入 → 落库 → 检索 → 总结 → 向量索引）全在可独立运行的
包里。所以这里**不起 AstrBot**，直接按真实调用顺序跑同一批代码：

    每轮对话：
      scene()          身份解析（identities 表）
      inject_block()   常驻六词条拼注入块
      search()         FTS + BM25 精排（可选向量补齐）
      capture_turn()   用户发言 + 机器人回复批量落库
      偶尔 save() / set_resident()

它解决的是三个具体问题：

1. **批量并发**：N 个机器人 × M 个用户同时说话，私聊与群聊混合，
   观察吞吐、p50/p95/p99、错误率与事件循环卡顿。
2. **真实模型接入**：`--llm auto` 读取本机 AstrBot 配置里的模型链
   （或显式 `--llm-base-url/--llm-model/--llm-api-key`），用真实总结模型跑
   总结阶段；`--embed auto` 探测可用的 OpenAI 兼容 embedding 服务。
   两个都支持 `fake`（默认，完全离线）。
3. **可用性校验**：压测结束后自动做跨账号 / 跨用户 / 群内隔离探针，
   确认高并发没有把归属判断搞坏；而不是只看「没报错」。

## 常用命令

    # 纯离线压测：20 个机器人 × 20 用户，跑 30 秒
    python benchmarks/bench_fleet_load.py --bots 20 --users-per-bot 20 --duration 30

    # 接真实总结模型（自动读 AstrBot 配置）做 2 个账号的总结
    python benchmarks/bench_fleet_load.py --llm auto --summarize-bots 2

    # 接真实 embedding + 向量检索并发
    python benchmarks/bench_fleet_load.py --embed auto --vector-bots 2 --vector-concurrency 20

    # 本机 CPU 跑本地向量模型（transformers + torch，不需要额外服务）
    python benchmarks/bench_fleet_load.py --embed local --vector-bots 2 \\
        --vector-concurrency 8 --embed-threads 6

    # 更重的混合负载：50 个账号 × 40 用户，阶梯加压
    python benchmarks/bench_fleet_load.py --bots 50 --users-per-bot 40 \\
        --scenario ramp --ramp-seconds 20 --duration 120 --think-ms 2

报告写到 `benchmarks/out/`（已在 .gitignore 里），同时打印到终端。

## 环境变量（优先级高于自动探测）

    HERMES_BENCH_LLM_BASE_URL / _MODEL / _KEY
    HERMES_BENCH_EMBED_BASE_URL / _MODEL / _KEY
    HERMES_BENCH_ASTRBOT_CONFIG   默认 ~/.astrbot/data/cmd_config.json
    HERMES_BENCH_FLEET_CONFIG     默认 ~/.astrbot/data/config/astrbot_plugin_hermes_fleet_config.json

密钥只用于请求，**不写进报告、不打印**。
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for src in sorted((ROOT / "packages").glob("*/src")):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from hermes_contracts import LlmResult  # noqa: E402
from hermes_fleet import BotRegistry, MemoryService  # noqa: E402
from hermes_fleet.registry import bot_id_of  # noqa: E402
from hermes_fleet.vector import VectorGlue, VectorSettings  # noqa: E402
from hermes_vectors import EmbeddingConfig, EmbeddingError, normalize  # noqa: E402
from hermes_workers import SessionSummarizer  # noqa: E402

# ---------------------------------------------------------------------------
# 合成语料
# ---------------------------------------------------------------------------

_ZH = (
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家"
    "可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作开美总从无情"
    "己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间知世什两次使身"
)
QUERIES = [
    "我喜欢喝什么", "上次说的那个项目", "我的口味有什么讲究", "工作安排",
    "家里养的宠物", "周报什么时候交", "我住在哪里", "生日是哪天",
    "最近在忙什么", "有没有忌口",
]
TOPICS = [
    "用户喜欢喝乌龙茶，尤其偏爱铁观音",
    "用户对花生过敏，点外卖要避开",
    "用户在做 AstrBot 记忆插件，关注长期记忆",
    "用户周末常去爬山，喜欢人少的路线",
    "用户养了两只布偶猫，名字叫团子和年糕",
    "用户希望回复简洁直接，不要客套",
    "用户在本群负责每周五的周报汇总",
    "用户下个月搬去深圳，工作地点在南山",
]
# 第一人称的「值得记住」的话：真实模型跑总结时，能从这里提炼出稳定信息，
# 而不是面对一串随机汉字只能回 []。压测既能测吞吐，也能看产出是否正常。
UTTERANCES = [
    "我平时只喝乌龙茶，铁观音最喜欢，别给我推荐咖啡",
    "我对花生过敏，点外卖一定要避开花生碎",
    "我在做 AstrBot 的记忆插件，主要关注跨会话的长期记忆",
    "我周末喜欢去爬山，最好是工作日人少的时候去",
    "我家里有两只布偶猫，一只叫团子，一只叫年糕",
    "我希望你回复简洁直接一点，不用客套",
    "我在这个群负责每周五的周报汇总，别忘了提醒我",
    "我下个月要搬去深圳，工作地点在南山科技园",
]


def _sentence(rng: random.Random, n: int = 30) -> str:
    return "".join(rng.choice(_ZH) for _ in range(n))


# ---------------------------------------------------------------------------
# 进程指标
# ---------------------------------------------------------------------------


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_mb() -> float:
    """当前进程常驻内存（MB）。Windows 走 psapi，其它平台尽量走 resource。"""
    try:
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters),
                ctypes.c_uint32,
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return counters.WorkingSetSize / 1024 / 1024
    except Exception:
        pass
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def dir_size_mb(path: str) -> float:
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total / 1024 / 1024


# ---------------------------------------------------------------------------
# 供应商：发现 / 真实 HTTP / 离线假实现
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    base_url: str
    model: str
    api_key: str = ""
    label: str = ""
    proxy: str = ""

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/")

    def describe(self) -> str:
        return f"{self.label or self.model} @ {self.base_url}"


def _post_json(url: str, payload: dict, *, api_key: str = "",
               timeout: float = 60.0, proxy: str = "") -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST",
    )
    # 只覆盖显式配置的代理；没配就沿用 urllib 的环境变量行为。
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    ) if proxy else None
    try:
        if opener is not None:
            resp = opener.open(req, timeout=timeout)
            try:
                raw = resp.read(8 * 1024 * 1024)
            finally:
                resp.close()
        else:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(8 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(600).decode("utf-8", "replace")
        except Exception:
            pass
        raise ProviderError(f"HTTP {exc.code}: {detail.strip() or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"连接失败: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderError(f"超时 {timeout:.0f}s") from exc
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except ValueError as exc:
        raise ProviderError("响应不是 JSON") from exc
    if not isinstance(data, dict):
        raise ProviderError("响应不是 JSON 对象")
    return data


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8-sig") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _astrbot_paths() -> tuple[Path, Path]:
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    cfg = Path(os.environ.get(
        "HERMES_BENCH_ASTRBOT_CONFIG", home / ".astrbot" / "data" / "cmd_config.json"))
    fleet = Path(os.environ.get(
        "HERMES_BENCH_FLEET_CONFIG",
        home / ".astrbot" / "data" / "config"
        / "astrbot_plugin_hermes_fleet_config.json"))
    return cfg, fleet


def _first_key(raw: object) -> str:
    """AstrBot 的 provider source 里 key 可能是单个字符串，也可能是轮换列表。"""
    if isinstance(raw, list):
        for item in raw:
            text = str(item or "").strip()
            if text:
                return text
        return ""
    return str(raw or "").strip()


def discover_llm_targets() -> list[Target]:
    """从本机 AstrBot 配置里拼出候选模型：优先插件配置的总结模型链。

    只读 `provider_sources[].api_base/api_key` 与 `provider[]`，不修改任何文件。
    """
    env_base = os.environ.get("HERMES_BENCH_LLM_BASE_URL", "").strip()
    env_model = os.environ.get("HERMES_BENCH_LLM_MODEL", "").strip()
    if env_base and env_model:
        return [Target(env_base, env_model,
                       os.environ.get("HERMES_BENCH_LLM_KEY", "").strip(), "env",
                       os.environ.get("HERMES_BENCH_LLM_PROXY", "").strip())]

    cfg_path, fleet_path = _astrbot_paths()
    cfg = _read_json(cfg_path)
    if not cfg:
        return []
    sources = {str(s.get("id") or ""): s for s in cfg.get("provider_sources") or []
               if isinstance(s, dict)}
    providers = [p for p in cfg.get("provider") or []
                 if isinstance(p, dict) and p.get("enable", True)]
    by_id = {str(p.get("id") or ""): p for p in providers}

    chain = str(_read_json(fleet_path).get("llm_provider_ids") or "")
    wanted = [x.strip() for x in chain.replace("，", ",").split(",") if x.strip()]
    ordered: list[dict] = []
    for key in wanted:
        if key in by_id:
            ordered.append(by_id[key])
        else:  # 允许写模型名而不是完整 id
            ordered.extend(p for p in providers if p.get("model") == key)
    ordered.extend(p for p in providers if p not in ordered)

    out: list[Target] = []
    for p in ordered:
        src = sources.get(str(p.get("provider_source_id") or "")) or {}
        base = str(src.get("api_base") or "").strip()
        model = str(p.get("model") or "").strip()
        if not base or not model:
            continue
        out.append(Target(base, model, _first_key(src.get("key")),
                          str(p.get("id") or model),
                          str(src.get("proxy") or "").strip()))
    return out


def discover_embed_targets() -> list[Target]:
    """候选 embedding 服务：env → 本机 provider 源 → DashScope → Ollama。"""
    env_base = os.environ.get("HERMES_BENCH_EMBED_BASE_URL", "").strip()
    env_model = os.environ.get("HERMES_BENCH_EMBED_MODEL", "").strip()
    if env_base and env_model:
        return [Target(env_base, env_model,
                       os.environ.get("HERMES_BENCH_EMBED_KEY", "").strip(), "env",
                       os.environ.get("HERMES_BENCH_EMBED_PROXY", "").strip())]

    models = [x for x in os.environ.get(
        "HERMES_BENCH_EMBED_MODELS",
        "text-embedding-v4,text-embedding-v3,text-embedding-3-small,bge-m3,nomic-embed-text",
    ).replace("，", ",").split(",") if x.strip()]

    bases: list[tuple[str, str, str, str]] = []  # (base, key, label, proxy)
    cfg = _read_json(_astrbot_paths()[0])
    for src in cfg.get("provider_sources") or []:
        if not isinstance(src, dict):
            continue
        base = str(src.get("api_base") or "").strip()
        key = _first_key(src.get("key"))
        label = str(src.get("id") or base)
        proxy = str(src.get("proxy") or "").strip()
        if base and all(base != b[0] for b in bases):
            bases.append((base, key, label, proxy))
        # 阿里云的 MaaS 地址与公共 DashScope 共用同一类密钥：前者只挂了
        # 对话模型时，embedding 往往在公共兼容模式端点可用，顺手多试一次。
        if key and ("aliyuncs.com" in base or "dashscope" in base) and all(
                b[0] != "https://dashscope.aliyuncs.com/compatible-mode/v1"
                for b in bases):
            bases.append(("https://dashscope.aliyuncs.com/compatible-mode/v1",
                          key, f"{label}-dashscope", proxy))
    if os.environ.get("DASHSCOPE_API_KEY"):
        bases.append(("https://dashscope.aliyuncs.com/compatible-mode/v1",
                      os.environ["DASHSCOPE_API_KEY"].strip(), "dashscope", ""))
    bases.append(("http://127.0.0.1:11434/v1", "", "ollama", ""))
    bases.append(("http://127.0.0.1:8000/v1", "", "vllm", ""))

    out: list[Target] = []
    for base, key, label, proxy in bases:
        if not key and not base.startswith("http://127.0.0.1"):
            continue  # 没有密钥的公网源不必浪费请求
        for model in models:
            out.append(Target(base, model, key, label, proxy))
    return out


def probe_embedding(targets: list[Target], *, timeout: float = 8.0) -> Target | None:
    """逐个试，返回第一个能返回合法向量的目标。失败原因汇总后打印。"""
    for target in targets:
        try:
            data = _post_json(
                target.endpoint + "/embeddings",
                {"model": target.model, "input": ["ping"]},
                api_key=target.api_key, timeout=timeout, proxy=target.proxy,
            )
            rows = data.get("data") or []
            vec = rows[0].get("embedding") if rows and isinstance(rows[0], dict) else None
            if isinstance(vec, list) and vec:
                print(f"  ✓ embedding 可用：{target.describe()}（维度 {len(vec)}）")
                return target
            print(f"  ✗ {target.describe()}：响应缺少 embedding")
        except ProviderError as exc:
            print(f"  ✗ {target.describe()}：{exc}")
    return None


class OpenAICompatLLM:
    """OpenAI 兼容的对话补全。只在总结阶段用，不参与对话压测热路径。"""

    def __init__(self, target: Target, *, timeout: float = 120.0) -> None:
        self.target = target
        self.timeout = timeout
        self.calls = 0
        self.errors = 0
        self.latencies: list[float] = []

    async def complete(self, messages, *, purpose: str, max_tokens=None,
                       temperature=None) -> LlmResult:
        payload: dict = {
            "model": self.target.model,
            "messages": [{"role": str(m.role), "content": m.content}
                         for m in messages],
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        t0 = time.perf_counter()
        self.calls += 1
        try:
            data = await asyncio.to_thread(
                _post_json, self.target.endpoint + "/chat/completions",
                payload, api_key=self.target.api_key, timeout=self.timeout,
                proxy=self.target.proxy,
            )
        except ProviderError:
            self.errors += 1
            raise
        latency = (time.perf_counter() - t0) * 1000
        self.latencies.append(latency)
        try:
            choice = (data.get("choices") or [{}])[0]
            text = str((choice.get("message") or {}).get("content") or "")
            finish = str(choice.get("finish_reason") or "")
        except Exception:
            text, finish = "", ""
        return LlmResult(text=text, model=self.target.model, latency_ms=latency,
                         finish_reason=finish,
                         usage=data.get("usage") or {})


class FakeLLM:
    """离线假模型：返回合法 JSON 数组，内容唯一，便于观察记忆增长。"""

    def __init__(self, *, delay_ms: float = 2.0) -> None:
        self.delay_ms = max(0.0, delay_ms)
        self.seq = 0
        self.calls = 0
        self.errors = 0
        self.latencies: list[float] = []

    async def complete(self, messages, *, purpose: str, max_tokens=None,
                       temperature=None) -> LlmResult:
        t0 = time.perf_counter()
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)
        self.seq += 1
        self.calls += 1
        text = json.dumps([{
            "kind": "fact",
            "content": f"压测生成的稳定偏好 #{self.seq}：偏好简洁回复，忌花生",
        }], ensure_ascii=False)
        latency = (time.perf_counter() - t0) * 1000
        self.latencies.append(latency)
        return LlmResult(text=text, model="fake-llm", latency_ms=latency,
                         finish_reason="stop")


class FakeEmbedder:
    """离线假向量：字符散列到固定维度。足够让 KNN 有区分度，又不联网。"""

    def __init__(self, dim: int = 256, *, delay_ms: float = 0.0) -> None:
        self.dim = dim
        self.delay_ms = max(0.0, delay_ms)
        self.calls = 0
        self.texts = 0
        self.seconds = 0.0

    def _vec(self, text: str) -> list[float]:
        out = [0.0] * self.dim
        for i, ch in enumerate(str(text)):
            out[(ord(ch) * 31 + i * 7) % self.dim] += 1.0
        return normalize(out)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)
        self.calls += 1
        self.texts += len(texts)
        self.seconds += time.perf_counter() - t0
        return [self._vec(t) for t in texts]

    async def embed_one(self, text: str) -> list[float]:
        rows = await self.embed([text])
        return rows[0] if rows else []


class OpenAICompatEmbedder:
    def __init__(self, target: Target, *, batch_size: int = 32,
                 timeout: float = 30.0) -> None:
        self.target = target
        self.model = target.model
        self.config = EmbeddingConfig(
            base_url=target.base_url, model=target.model,
            api_key=target.api_key, batch_size=batch_size, timeout_s=timeout,
        )
        self.calls = 0
        self.texts = 0
        self.seconds = 0.0
        self.dim = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        size = self.config.batch_size
        for start in range(0, len(texts), size):
            chunk = texts[start:start + size]
            t0 = time.perf_counter()
            try:
                data = await asyncio.to_thread(
                    _post_json, self.config.endpoint,
                    {"model": self.config.model, "input": chunk},
                    api_key=self.config.api_key, timeout=self.config.timeout_s,
                    proxy=self.target.proxy,
                )
            except ProviderError as exc:
                # 索引器的失败语义只认 EmbeddingError（见 hermes_fleet.vector），
                # 这里翻译一下，否则真实服务限流会把整个压测脚本打断。
                raise EmbeddingError(str(exc)) from exc
            self.seconds += time.perf_counter() - t0
            self.calls += 1
            self.texts += len(chunk)
            try:
                rows = sorted(data.get("data") or [],
                              key=lambda x: int(x.get("index", 0)))
            except (TypeError, ValueError, AttributeError) as exc:
                raise EmbeddingError(f"embedding 响应格式异常：{exc}") from exc
            if len(rows) != len(chunk):
                raise EmbeddingError(
                    f"embedding 条数不符：期望 {len(chunk)}，实际 {len(rows)}")
            for row in rows:
                try:
                    vec = normalize([float(x) for x in row.get("embedding") or []])
                except (TypeError, ValueError) as exc:
                    raise EmbeddingError(f"embedding 分量非法：{exc}") from exc
                if not vec:
                    raise EmbeddingError("embedding 返回空向量")
                self.dim = len(vec)
                out.append(vec)
        return out

    async def embed_one(self, text: str) -> list[float]:
        rows = await self.embed([text])
        return rows[0] if rows else []


class LocalEmbedder:
    """进程内本地向量模型（transformers + torch，CPU）。

    为什么不是 fastembed/onnxruntime：本机 Python 3.15 上 onnxruntime
    没有任何轮子，那条路根本装不上。torch 有 cp315 的 Windows CPU 轮子
    （约 120MB），所以本地推理只能走它。

    线程数与批大小都刻意收着：本机可用内存只有几个 GB，把 16 个逻辑核
    全交给 torch 只会抬高峰值、抢走压测要用的 CPU，换不来吞吐。模型实例
    也不并发调用——同一份权重并发跑只会互相争抢，压测要看的是单实例上限。
    """

    def __init__(self, model: str = "BAAI/bge-small-zh-v1.5", *,
                 threads: int = 6, batch_size: int = 10) -> None:
        self.model = model
        self.threads = max(1, int(threads))
        self.batch_size = max(1, int(batch_size))
        self.dim = 0
        self.calls = 0
        self.texts = 0
        self.seconds = 0.0
        self._torch = None
        self._tok = None
        self._net = None
        # 在事件循环上排队，而不是在线程池里等锁：线程池只有 20 个槽位，
        # 让等待者占着槽位会把同一条链路上的 SQLite 查询一并堵死，
        # 表现为「检索 p95 涨到几秒」，实际上模型只花了 5ms。
        self._alock = asyncio.Lock()

    def _load(self) -> None:
        if self._net is not None:
            return
        import torch  # 惰性导入：跑 fake/off 的压测不该付 torch 的启动成本
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(self.threads)
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model)
        self._net = AutoModel.from_pretrained(self.model).eval()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        torch = self._torch
        with torch.inference_mode():
            batch = self._tok(texts, padding=True, truncation=True,
                              max_length=256, return_tensors="pt")
            hidden = self._net(**batch).last_hidden_state[:, 0]
            # BGE 系列是 CLS 池化再归一化；与云端接口返回的归一化向量同构，
            # 否则余弦相似度会被模长干扰。
            mats = torch.nn.functional.normalize(hidden, p=2, dim=1)
        return [[float(x) for x in row] for row in mats.tolist()]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            t0 = time.perf_counter()
            async with self._alock:
                rows = await asyncio.to_thread(self._encode, chunk)
            self.seconds += time.perf_counter() - t0
            self.calls += 1
            self.texts += len(chunk)
            if rows:
                self.dim = len(rows[0])
            out.extend(rows)
        return out

    async def embed_one(self, text: str) -> list[float]:
        rows = await self.embed([text])
        return rows[0] if rows else []


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


class Sample:
    """有界蓄水池采样：高吞吐下也不把内存吃穿。"""

    __slots__ = ("cap", "items", "count", "total", "worst", "_rng")

    def __init__(self, cap: int = 200_000) -> None:
        self.cap = cap
        self.items: list[float] = []
        self.count = 0
        self.total = 0.0
        self.worst = 0.0
        self._rng = random.Random(7)

    def add(self, ms: float) -> None:
        self.count += 1
        self.total += ms
        if ms > self.worst:
            self.worst = ms
        if len(self.items) < self.cap:
            self.items.append(ms)
        else:
            idx = self._rng.randrange(self.count)
            if idx < self.cap:
                self.items[idx] = ms

    def pct(self, p: float) -> float:
        if not self.items:
            return 0.0
        xs = sorted(self.items)
        idx = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
        return xs[idx]


class Metrics:
    def __init__(self) -> None:
        self.ops: dict[str, Sample] = {}
        self.errors: dict[str, int] = {}
        self.error_samples: list[str] = []

    def record(self, op: str, ms: float) -> None:
        self.ops.setdefault(op, Sample()).add(ms)

    def error(self, op: str, exc: BaseException) -> None:
        self.errors[op] = self.errors.get(op, 0) + 1
        if len(self.error_samples) < 20:
            self.error_samples.append(f"{op}: {type(exc).__name__}: {exc}")

    def reset(self) -> None:
        self.ops.clear()
        self.errors.clear()
        self.error_samples.clear()

    @property
    def total(self) -> int:
        return sum(s.count for s in self.ops.values())


class LoopLag:
    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self.worst = 0.0
        self.samples = Sample(20_000)
        self._stop = asyncio.Event()

    async def run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass
            lag = max(0.0, (time.perf_counter() - t0 - self.interval) * 1000)
            self.samples.add(lag)
            self.worst = max(self.worst, lag)

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# 负载
# ---------------------------------------------------------------------------


@dataclass
class VirtualUser:
    bot: str
    owner: str
    group: str
    session: str
    token: str
    group_user: bool = False


@dataclass
class PhaseResult:
    name: str
    elapsed: float
    data: dict = field(default_factory=dict)
    error: str = ""


class FleetLoad:
    def __init__(self, args, llm, embedder) -> None:
        self.args = args
        self.llm = llm
        self.embedder = embedder
        self.tmp = args.workdir or tempfile.mkdtemp(prefix="fleet-load-")
        os.makedirs(self.tmp, exist_ok=True)
        self.root = os.path.join(self.tmp, "bots")
        self.registry = BotRegistry(self.root, max_open=args.max_open)
        settings = VectorSettings(
            provider="api" if embedder is not None else "off",
            base_url="http://bench.invalid/v1",
            model=str(getattr(embedder, "model", "") or "fake-embed"),
            # 真实服务的维度由第一次响应决定；写死配置值会让每轮同步都判定
            # 「维度变了」而整体重建。fake 模型维度固定，可以直接写。
            dim=(args.embed_dim if isinstance(embedder, FakeEmbedder) else 0),
            batch_size=args.embed_batch,
            min_score=args.vector_min_score,
            sync_batch=args.vector_sync_batch,
        )
        self.glue = VectorGlue(self.registry, settings, embedder=embedder)
        self.svc = MemoryService(
            self.registry,
            group_memory_shared=False,
            recall_k=args.recall_k,
            recall_mode="vector" if embedder is not None else "classic",
            vector_recall=self.glue if embedder is not None else None,
            overrides_of=self.registry.overrides_of,
        )
        self.metrics = Metrics()
        self.lag = LoopLag()
        self.wall = 0.0
        self.cpu_seconds = 0.0
        self.users: list[VirtualUser] = []
        self.bot_ids: list[str] = []
        self._seq = 0
        self._rng = random.Random(args.seed)

    # ------------------------------------------------------------ 准备

    def build_users(self) -> None:
        args = self.args
        for b in range(args.bots):
            pid = f"load-bot-{b:03d}"
            self.bot_ids.append(pid)
            for u in range(args.users_per_bot):
                owner = f"u{b:03d}-{u:04d}"
                in_group = (
                    args.groups_per_bot > 0
                    and self._rng.random() < args.group_ratio
                )
                group = ""
                if in_group:
                    group = f"g{b:03d}-{u % args.groups_per_bot}"
                self.users.append(VirtualUser(
                    bot=pid, owner=owner, group=group,
                    session=(f"sess-{pid}-{group}" if group
                             else f"sess-{pid}-{owner}"),
                    token=f"专属标记{pid}-{owner}",
                    group_user=in_group,
                ))

    async def seed(self) -> None:
        """给每个虚拟用户灌少量常驻 + 归档记忆，让检索有东西可召回。"""
        started = time.perf_counter()
        for user in self.users:
            scene = await self.svc.scene(
                user.bot, user.owner, user.group, is_group=bool(user.group))
            await self.svc.set_resident(scene, "address", f"称呼 {user.owner}")
            await self.svc.set_resident(
                scene, "preference", "偏好简洁回复；对花生过敏（压测量）")
            texts = [
                f"{user.token}：{self._rng.choice(TOPICS)}",
                f"用户画像：{_sentence(self._rng, 24)}",
                f"长期偏好：{self._rng.choice(TOPICS)}",
            ]
            for text in texts:
                await self.svc.save(scene, text)
        print(f"  已灌注 {len(self.users)} 个用户 × 3 条记忆，"
              f"用时 {time.perf_counter() - started:.2f}s")

    # ------------------------------------------------------------ 单次操作

    def _text(self, user: VirtualUser, kind: str) -> str:
        self._seq += 1
        if kind == "用户发言":
            body = (f"{self._rng.choice(UTTERANCES)}"
                    f"（顺便说一句：{_sentence(self._rng, 12)}）")
        else:
            body = f"好的，我记下了：{_sentence(self._rng, 16)}"
        return f"{body} #{self._seq}"

    async def _timed(self, op: str, coro) -> None:
        t0 = time.perf_counter()
        try:
            await coro
            self.metrics.record(op, (time.perf_counter() - t0) * 1000)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — 压测要收集所有错误
            self.metrics.error(op, exc)

    async def user_loop(self, user: VirtualUser, deadline: float) -> None:
        rng = self._rng
        while time.monotonic() < deadline:
            t0 = time.perf_counter()
            try:
                scene = await self.svc.scene(
                    user.bot, user.owner, user.group, is_group=bool(user.group))
                self.metrics.record("scene", (time.perf_counter() - t0) * 1000)
            except Exception as exc:  # noqa: BLE001
                self.metrics.error("scene", exc)
                break
            query = rng.choice(QUERIES)
            if rng.random() < 0.15:
                query = user.token

            await self._timed("inject", self.svc.inject_block(scene, query))
            await self._timed("search", self.svc.search(scene, query))
            await self._timed(
                "capture",
                self.svc.capture_turn(
                    scene, user.session,
                    self._text(user, "用户发言"),
                    self._text(user, "助手回复"),
                ),
            )
            if rng.random() < 0.10:
                await self._timed("save", self.svc.save(
                    scene, f"{user.token}｜{self._rng.choice(TOPICS)}"))
            if rng.random() < 0.05:
                await self._timed("resident", self.svc.set_resident(
                    scene, "preference", f"偏好简洁回复 {self._seq}"))
            # 必然让出一次事件循环：`asyncio.sleep(0)` 的语义是「排到队尾」，
            # 不额外等待。没有它，用户任务会在纯同步的 DB 调用里一路跑到底，
            # 既不公平也不像宿主（AstrBot 每条消息之间都有真实 IO）。
            # 等待时间带 ±30% 抖动：真实用户不会在同一毫秒齐刷刷醒来，
            # 否则 400 个定时器会一起触发，把「卡顿」测成同步闹钟的假象。
            think = max(0.0, self.args.think_ms) / 1000
            if think > 0:
                think *= 0.7 + 0.6 * rng.random()
            await asyncio.sleep(think)

    async def run_load(self) -> None:
        args = self.args
        started = time.monotonic()
        deadline = started + args.warmup + args.duration
        tasks: list[asyncio.Task] = []
        total = max(1, len(self.users))
        for i, user in enumerate(self.users):
            delay = 0.0
            if args.scenario == "ramp" and args.ramp_seconds > 0:
                delay = args.ramp_seconds * i / total

            async def starter(u=user, d=delay):
                if d:
                    await asyncio.sleep(d)
                await self.user_loop(u, deadline)

            tasks.append(asyncio.create_task(starter()))
        lag_task = asyncio.create_task(self.lag.run())
        window_start = started if args.warmup <= 0 else None
        next_progress = args.progress_seconds
        try:
            while True:
                now = time.monotonic()
                if now >= deadline:
                    break
                # 0.2s 的检查粒度：短跑（几秒）也能准确切在预热结束点，
                # 不会因为进度打印间隔是 5 秒而把整段指标清零。
                await asyncio.sleep(min(0.2, max(0.01, deadline - now)))
                elapsed = time.monotonic() - started
                if window_start is None and elapsed >= args.warmup:
                    window_start = time.monotonic()
                    self.metrics.reset()
                    self.lag.samples = Sample(20_000)
                    self.lag.worst = 0.0
                    print(f"  [warmup 结束] 指标清零，开始计时 {args.duration}s")
                if args.progress and elapsed >= next_progress:
                    next_progress += max(0.2, args.progress_seconds)
                    rate = self.metrics.total / max(0.001, elapsed)
                    print(f"  [{elapsed:6.1f}s] 操作 {self.metrics.total:>8} "
                          f"({rate:,.0f}/s) · 事件循环最大卡顿 {self.lag.worst:.1f}ms")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.lag.stop()
            await asyncio.gather(lag_task, return_exceptions=True)
        self.wall = max(0.001, time.monotonic() - (window_start or started))

    # ------------------------------------------------------------ 校验

    async def isolation_checks(self) -> dict:
        """高并发之后重新验一遍归属：隔离坏了，性能再好也不能用。"""
        out: dict = {}
        if len(self.bot_ids) < 2:
            # 单账号没有「跨账号」可比：跳过，别让压测最后一步崩在这里。
            return {"跨账号": "SKIP（只有 1 个账号）"}
        bot_a, bot_b = self.bot_ids[0], self.bot_ids[1]
        probe = f"隔离探针{int(time.time())}"

        scene_a = await self.svc.scene(bot_a, "probe-user-a", "", is_group=False)
        scene_b = await self.svc.scene(bot_b, "probe-user-b", "", is_group=False)
        await self.svc.save(scene_b, f"{probe} 只属于 B 账号")
        a_text = await self.svc.search(scene_a, probe)
        b_text = await self.svc.search(scene_b, probe)
        out["跨账号"] = ("PASS" if probe not in a_text and probe in b_text else "FAIL")

        gid = f"probe-group-{int(time.time())}"
        u1 = await self.svc.scene(bot_a, "probe-g1", gid, is_group=True)
        u2 = await self.svc.scene(bot_a, "probe-g2", gid, is_group=True)
        group_probe = f"群探针{int(time.time())}"
        await self.svc.save(u1, f"{group_probe} 由 u1 说")
        u2_text = await self.svc.search(u2, group_probe)
        out["群内默认隔离"] = ("PASS" if group_probe not in u2_text else "FAIL")
        u1_text = await self.svc.search(u1, group_probe)
        out["本人可召回"] = ("PASS" if group_probe in u1_text else "FAIL")

        own_private = await self.svc.scene(bot_a, "probe-g1", "", is_group=False)
        leaked = await self.svc.search(own_private, group_probe)
        out["群记忆不漏到私聊"] = ("PASS" if group_probe not in leaked else "FAIL")
        return out

    # ------------------------------------------------------------ 总结阶段

    async def summarize(self, count: int) -> PhaseResult:
        started = time.perf_counter()
        if self.llm is None or count <= 0:
            return PhaseResult("总结", 0.0, {"skipped": True})
        summarizer = SessionSummarizer(
            self.llm, min_messages=1, idle_seconds=1.0,
            group_min_messages=1, group_idle_seconds=1.0,
            max_sessions=self.args.summarize_sessions,
            max_tokens=self.args.summarize_max_tokens,
        )
        sem = asyncio.Semaphore(max(1, self.args.summarize_concurrency))
        lat: list[float] = []
        memories = 0
        failures = 0
        sessions = 0
        errors: list[str] = []

        async def one(pid: str) -> None:
            nonlocal memories, failures, sessions
            async with sem:
                db = self.registry.get(pid)
                t0 = time.perf_counter()
                try:
                    res = await summarizer.run_bot(
                        bot_id_of(pid), messages=db.messages,
                        memories=db.memories, identities=db.identities,
                        force=True,
                    )
                    memories += res.memories
                    failures += res.failures
                    sessions += res.sessions_done
                    if res.last_error:
                        errors.append(res.last_error)
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    errors.append(f"{pid}: {type(exc).__name__}: {exc}")
                finally:
                    lat.append((time.perf_counter() - t0) * 1000)

        await asyncio.gather(*(one(pid) for pid in self.bot_ids[:count]))
        data = {
            "bots": min(count, len(self.bot_ids)),
            "sessions": sessions,
            "memories": memories,
            "failures": failures,
            "p50_ms": _pct(lat, 50),
            "p95_ms": _pct(lat, 95),
            "max_ms": max(lat) if lat else 0.0,
            "errors": errors[:5],
        }
        return PhaseResult("总结", time.perf_counter() - started, data)

    # ------------------------------------------------------------ 向量阶段

    async def vector_phase(self, count: int) -> PhaseResult:
        started = time.perf_counter()
        if self.embedder is None or count <= 0:
            return PhaseResult("向量", 0.0, {"skipped": True})

        sync: list[float] = []
        indexed = 0
        sync_errors: list[str] = []
        for pid in self.bot_ids[:count]:
            t0 = time.perf_counter()
            res = await self.glue.sync_bot(pid)
            sync.append((time.perf_counter() - t0) * 1000)
            indexed += int(res.get("indexed") or 0)
            if res.get("error"):
                sync_errors.append(f"{pid}: {res['error']}")

        queries = [self.users[i].token if i % 3 == 0 else QUERIES[i % len(QUERIES)]
                   for i in range(self.args.vector_concurrency)]
        lat: list[float] = []
        hits = 0
        sem = asyncio.Semaphore(max(1, self.args.vector_concurrency))

        async def one(i: int) -> None:
            nonlocal hits
            user = self.users[i % len(self.users)]
            scene = await self.svc.scene(
                user.bot, user.owner, user.group, is_group=bool(user.group))
            async with sem:
                t0 = time.perf_counter()
                try:
                    rows = await self.glue.semantic_hits(
                        scene, queries[i], limit=self.args.recall_k)
                    hits += len(rows)
                finally:
                    lat.append((time.perf_counter() - t0) * 1000)

        await asyncio.gather(*(one(i) for i in range(self.args.vector_concurrency)))
        calls = getattr(self.embedder, "calls", 0)
        texts = getattr(self.embedder, "texts", 0)
        seconds = getattr(self.embedder, "seconds", 0.0)
        data = {
            "bots": min(count, len(self.bot_ids)),
            "indexed": indexed,
            "sync_p95_ms": _pct(sync, 95),
            "query_p50_ms": _pct(lat, 50),
            "query_p95_ms": _pct(lat, 95),
            "query_max_ms": max(lat) if lat else 0.0,
            "hits": hits,
            "embed_calls": calls,
            "embed_texts": texts,
            "embed_texts_per_s": (texts / seconds) if seconds > 0.001 else 0.0,
            "dim": getattr(self.embedder, "dim", 0),
            "sync_errors": sync_errors[:3],
        }
        return PhaseResult("向量", time.perf_counter() - started, data)

    # ------------------------------------------------------------ 清理

    def close(self) -> None:
        try:
            self.glue.close()
        except Exception:
            pass
        try:
            self.registry.close_all()
        except Exception:
            pass


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _ms(v: float) -> str:
    return "—" if not v else f"{v:.2f}ms"


def build_report(load: FleetLoad, *, started_at: float, wall: float,
                 llm_desc: str, embed_desc: str,
                 summary: PhaseResult, vector: PhaseResult,
                 checks: dict, rss_start: float) -> tuple[str, dict]:
    args = load.args
    m = load.metrics
    ops = sorted(m.ops)
    total = sum(s.count for s in m.ops.values())
    rps = total / wall if wall > 0 else 0
    rss_end = rss_mb()
    dbs = dir_size_mb(load.root)

    lines = [
        "# 记忆插件压力测试报告",
        "",
        f"- 时间：{datetime.fromtimestamp(started_at):%Y-%m-%d %H:%M:%S}",
        f"- 负载：{args.bots} 个机器人 × {args.users_per_bot} 用户"
        f"（{len(load.users)} 个会话），{args.scenario}，"
        f"预热 {args.warmup:.0f}s + 计时 {args.duration:.0f}s",
        f"- 并发思考时间：{args.think_ms}ms · recall_k={args.recall_k} · "
        f"max_open_bots={args.max_open}",
        f"- 总结模型：{llm_desc}",
        f"- 向量模型：{embed_desc}",
        "",
        "## 吞吐与延迟",
        "",
        "| 操作 | 次数 | 占比 | p50 | p95 | p99 | max | 错误 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for op in ops:
        s = m.ops[op]
        lines.append(
            f"| {op} | {s.count} | {s.count / total * 100:.1f}% | "
            f"{_ms(s.pct(50))} | {_ms(s.pct(95))} | {_ms(s.pct(99))} | "
            f"{_ms(s.worst)} | {m.errors.get(op, 0)} |"
        )
    lines += [
        "",
        f"**总操作 {total} · {rps:,.1f} ops/s · 错误 "
        f"{sum(m.errors.values())} · 用时 {wall:.1f}s**",
        "",
        "## 事件循环与资源",
        "",
        f"- 单核 CPU 占用（仅压测窗口）："
        f"{load.cpu_seconds / wall * 100:.0f}%",
        f"- 事件循环卡顿：p95 {load.lag.samples.pct(95):.2f}ms · "
        f"p99 {load.lag.samples.pct(99):.2f}ms · 最大 {load.lag.worst:.2f}ms",
        f"- 常驻内存：{rss_start:.1f}MB → {rss_end:.1f}MB",
        f"- 记忆库体积：{dbs:.1f}MB（{args.bots} 个账号）",
        "",
        "## 隔离校验（压测后复核）",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in checks.items()]
    lines += ["", "## 真实模型阶段", ""]

    if summary.data.get("skipped"):
        lines.append("- 总结：未启用（`--summarize-bots 0`）")
    else:
        d = summary.data
        lines += [
            f"- 总结：{d['bots']} 个账号 / {d['sessions']} 个会话 → "
            f"新增记忆 {d['memories']} 条，失败 {d['failures']} 次",
            f"- 单账号耗时：p50 {_ms(d['p50_ms'])} · p95 {_ms(d['p95_ms'])} · "
            f"max {_ms(d['max_ms'])}",
        ]
        if d.get("errors"):
            lines.append(f"- 最近错误：{'; '.join(d['errors'])}")
    if vector.data.get("skipped"):
        lines.append("- 向量：未启用（`--vector-bots 0` 或 embedding 不可用）")
    else:
        d = vector.data
        rate = d["embed_texts_per_s"]
        rate_text = f"{rate:.1f} 条/秒" if rate else "未计时（离线模型）"
        lines += [
            f"- 向量：{d['bots']} 个账号，索引 {d['indexed']} 条，"
            f"同步 p95 {_ms(d['sync_p95_ms'])}",
            f"- 语义检索：p50 {_ms(d['query_p50_ms'])} · "
            f"p95 {_ms(d['query_p95_ms'])} · max {_ms(d['query_max_ms'])}"
            f"（{d['hits']} 条命中）",
            f"- embedding：{d['embed_calls']} 次调用 / {d['embed_texts']} 条"
            f"（{rate_text}，维度 {d['dim']}）",
        ]
        if d.get("sync_errors"):
            lines.append(f"- 索引错误：{'; '.join(d['sync_errors'])}")
    if m.error_samples:
        lines += ["", "## 错误样本", "", "```"]
        lines += m.error_samples
        lines += ["```"]
    lines += [
        "",
        "## 口径说明",
        "",
        "- 本脚本直接调用 `hermes_fleet` 的业务层（与 `main.py` 同一条热路径），"
        "不含 AstrBot 的适配器收发与模型对话；"
        "这样测到的是记忆插件本身，而不是宿主的网络栈。",
        "- 数据库调用是同步的（与生产一致），因此高并发下的表现是"
        "「在事件循环里排队」的真实结果，不是把 SQLite 丢线程池造出来的假象。",
        "- 事件循环卡顿在 `--think-ms 0`（零间隔压满）时会包含「任务排队」时间："
        "此时 CPU 已饱和，卡顿反映的是总负载，不是某条 SQL 阻塞；"
        "要区分「单条操作慢」请看上面的 p95 与 CPU 占比。",
        "- 真实模型阶段只跑少量账号；大规模并发应通过多 worker / 多进程扩展，"
        "而不是在单进程里堆总结协程（与 workers 的设计说明一致）。",
        "",
    ]
    text = "\n".join(lines)
    safe_config = {
        k: ("***" if "api_key" in k.lower() or k.lower().endswith("_key") else v)
        for k, v in vars(args).items()
    }
    payload = {
        "config": safe_config,
        "wall_seconds": wall,
        "ops": {
            op: {
                "count": s.count, "p50": s.pct(50), "p95": s.pct(95),
                "p99": s.pct(99), "max": s.worst,
                "errors": m.errors.get(op, 0),
            }
            for op, s in m.ops.items()
        },
        "total_ops": total,
        "ops_per_second": rps,
        "loop_lag": {
            "p95": load.lag.samples.pct(95),
            "p99": load.lag.samples.pct(99),
            "max": load.lag.worst,
        },
        "rss_mb": {"start": rss_start, "end": rss_end},
        "cpu_percent": (load.cpu_seconds / wall * 100) if wall > 0 else 0.0,
        "db_mb": dbs,
        "isolation": checks,
        "summary": summary.data,
        "vector": vector.data,
        "llm": llm_desc,
        "embedding": embed_desc,
        "errors": m.errors,
        "error_samples": m.error_samples,
    }
    return text, payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="记忆插件等价环境压力测试（直接跑 hermes_fleet 业务层）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bots", type=int, default=20, help="机器人账号数")
    p.add_argument("--users-per-bot", type=int, default=20, help="每个账号的虚拟用户数")
    p.add_argument("--groups-per-bot", type=int, default=2, help="每个账号模拟几个群")
    p.add_argument("--group-ratio", type=float, default=0.4,
                   help="多大比例的用户在群里说话（其余私聊）")
    p.add_argument("--scenario", choices=["steady", "ramp"], default="steady")
    p.add_argument("--ramp-seconds", type=float, default=15.0, help="ramp 模式下逐步启动的时长")
    p.add_argument("--warmup", type=float, default=3.0, help="预热秒数（指标清零）")
    p.add_argument("--duration", type=float, default=30.0, help="计时秒数")
    p.add_argument("--think-ms", type=float, default=1500.0,
                   help="每个虚拟用户两轮之间的等待；0=压满（CPU 会饱和）")
    p.add_argument("--recall-k", type=int, default=5)
    p.add_argument("--max-open", type=int, default=24)
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--progress", action="store_true", default=True,
                   help="每 5 秒打印进度")
    p.add_argument("--progress-seconds", type=float, default=5.0)
    p.add_argument("--no-progress", dest="progress", action="store_false")

    p.add_argument("--llm", choices=["fake", "auto", "off"], default="fake",
                   help="总结模型：fake 离线 / auto 读 AstrBot 配置 / off 不跑总结")
    p.add_argument("--llm-base-url", default="")
    p.add_argument("--llm-model", default="")
    p.add_argument("--llm-api-key", default="")
    p.add_argument("--summarize-bots", type=int, default=0,
                   help="压测后对多少个账号跑真实总结（0=不跑）")
    p.add_argument("--summarize-concurrency", type=int, default=1,
                   help="总结并发；生产 worker 是串行的，>1 只用于观察供应商限流")
    p.add_argument("--summarize-sessions", type=int, default=2)
    p.add_argument("--summarize-max-tokens", type=int, default=600)

    p.add_argument("--embed", choices=["fake", "auto", "off", "local"],
                   default="fake",
                   help="向量模型：fake 离线 / local 本机 transformers / "
                        "auto 探测兼容服务 / off")
    p.add_argument("--embed-base-url", default="")
    p.add_argument("--embed-model", default="")
    p.add_argument("--embed-api-key", default="")
    p.add_argument("--embed-batch", type=int, default=10,
                   help="embedding 单批条数（DashScope 上限 10）")
    p.add_argument("--embed-dim", type=int, default=256, help="fake 模型的维度")
    p.add_argument("--embed-local-model", default="BAAI/bge-small-zh-v1.5",
                   help="--embed local 用的 HF 模型（默认 512 维中文小模型）")
    p.add_argument("--embed-threads", type=int, default=6,
                   help="本地模型的 torch 线程数（别占满，留 CPU 给压测）")
    p.add_argument("--vector-bots", type=int, default=0,
                   help="压测后对多少个账号建索引并做语义检索（0=不跑）")
    p.add_argument("--vector-concurrency", type=int, default=16)
    p.add_argument("--vector-sync-batch", type=int, default=256)
    p.add_argument("--vector-min-score", type=float, default=0.35)

    p.add_argument("--workdir", default="", help="数据目录（默认临时目录，结束删除）")
    p.add_argument("--keep", action="store_true", help="保留数据目录")
    p.add_argument("--report-dir", default=str(ROOT / "benchmarks" / "out"))
    return p.parse_args(argv)


def make_llm(args) -> tuple[object, str]:
    if args.llm == "off":
        return None, "off"
    if args.llm == "fake":
        return FakeLLM(), "fake（离线）"
    if args.llm_base_url and args.llm_model:
        target = Target(args.llm_base_url, args.llm_model, args.llm_api_key, "cli")
        return OpenAICompatLLM(target), target.describe()
    targets = discover_llm_targets()
    if not targets:
        print("  ! 没找到可用的总结模型配置，退回 fake")
        return FakeLLM(), "fake（未发现配置）"
    if args.llm_model:
        matched = [t for t in targets
                   if args.llm_model in (t.model, t.label)]
        if not matched:
            print(f"  ! 配置里没有匹配的模型 {args.llm_model!r}，使用链首")
        else:
            targets = matched + targets
    target = targets[0]
    print(f"  总结模型：{target.describe()}")
    return OpenAICompatLLM(target), target.describe()


def make_embedder(args) -> tuple[object | None, str]:
    if args.embed == "off":
        return None, "off"
    if args.embed == "fake":
        return FakeEmbedder(args.embed_dim), f"fake（{args.embed_dim} 维）"
    if args.embed == "local":
        if args.vector_bots <= 0:
            # 没开向量阶段就别加载模型：95MB 权重 + torch 运行时在这个
            # 内存本就紧张的本机上不值得白占。
            return None, "local（未启用向量阶段，未加载）"
        emb = LocalEmbedder(args.embed_local_model, threads=args.embed_threads,
                            batch_size=args.embed_batch)
        return emb, f"local {args.embed_local_model}（CPU {args.embed_threads} 线程）"
    if args.vector_bots <= 0:
        # 不做向量阶段就不探测：探测会真的发请求，不该为「只是压测对话」付钱。
        return None, "auto（未启用向量阶段，未探测）"
    if args.embed_base_url and args.embed_model:
        target = Target(args.embed_base_url, args.embed_model,
                        args.embed_api_key, "cli")
        print(f"  探测 embedding：{target.describe()}")
        ok = probe_embedding([target])
        if ok is None:
            return None, "auto 失败"
        return OpenAICompatEmbedder(ok, batch_size=args.embed_batch), ok.describe()
    print("  探测 embedding 服务（最多几秒钟）…")
    target = probe_embedding(discover_embed_targets())
    if target is None:
        print("  ! 没有可用的 embedding 服务；向量阶段将跳过")
        return None, "auto 失败"
    return (OpenAICompatEmbedder(target, batch_size=args.embed_batch),
            target.describe())


async def amain(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=" * 74)
    print("记忆插件等价压力测试")
    print(f"账号 {args.bots} × 用户 {args.users_per_bot} · "
          f"{args.scenario} · 预热 {args.warmup:.0f}s + 计时 {args.duration:.0f}s")
    print("=" * 74)

    llm, llm_desc = make_llm(args)
    embedder, embed_desc = make_embedder(args)
    if isinstance(embedder, LocalEmbedder):
        # 首次调用要 import torch 并加载权重（这台机器约 5s）。不预热的话，
        # 首轮并发检索会全部堵在这一次加载后面，被误读成「p95 七秒」。
        t0 = time.perf_counter()
        await embedder.embed(["预热：我平时只喝乌龙茶"])
        print(f"  本地模型预热：{time.perf_counter() - t0:.1f}s · "
              f"{embedder.dim} 维 · RSS {rss_mb():.0f}MB")

    load = FleetLoad(args, llm, embedder)
    started_at = time.time()
    rss_start = rss_mb()
    summary = PhaseResult("总结", 0.0, {"skipped": True})
    vector = PhaseResult("向量", 0.0, {"skipped": True})
    checks: dict = {}
    wall = 0.0
    try:
        try:
            print("\n[1/5] 生成账号与用户")
            load.build_users()
            print("\n[2/5] 灌注基础数据")
            await load.seed()
            print("\n[3/5] 开始压测")
            cpu0 = time.process_time()
            await load.run_load()
            load.cpu_seconds = time.process_time() - cpu0
            wall = load.wall
            print(f"  完成：{load.metrics.total} 次操作 / {wall:.1f}s")

            print("\n[4/5] 隔离校验")
            checks = await load.isolation_checks()
            for name, verdict in checks.items():
                print(f"  {name}: {verdict}")

            if args.summarize_bots > 0 and llm is not None:
                print(f"\n[5/5] 真实总结（{args.summarize_bots} 个账号）")
                summary = await load.summarize(args.summarize_bots)
                print(f"  {summary.data}")
            if args.vector_bots > 0 and embedder is not None:
                print(f"\n[5/5] 向量索引 + 语义检索（{args.vector_bots} 个账号）")
                vector = await load.vector_phase(args.vector_bots)
                print(f"  {vector.data}")
        finally:
            load.close()

        text, payload = build_report(
            load, started_at=started_at, wall=max(0.001, wall),
            llm_desc=llm_desc, embed_desc=embed_desc,
            summary=summary, vector=vector, checks=checks, rss_start=rss_start,
        )
        os.makedirs(args.report_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        md_path = os.path.join(args.report_dir, f"fleet-load-{stamp}.md")
        json_path = os.path.join(args.report_dir, f"fleet-load-{stamp}.json")
        with open(md_path, "w", encoding="utf-8") as fp:
            fp.write(text)
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print("\n" + text)
        print(f"报告：{md_path}\n      {json_path}")

        failed = [k for k, v in checks.items() if v != "PASS"]
        if failed:
            print(f"!! 隔离校验未通过：{failed}")
            return 1
        if load.metrics.errors:
            print(f"!! 压测期间有错误：{load.metrics.errors}")
            return 1
        return 0
    finally:
        if not args.keep and not args.workdir:
            shutil.rmtree(load.tmp, ignore_errors=True)
        else:
            print(f"数据目录：{load.tmp}")


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
