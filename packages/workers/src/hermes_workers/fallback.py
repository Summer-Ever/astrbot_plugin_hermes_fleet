"""模型降级链：主模型 → 降级 1 → 降级 2 …，带错误分类、单模型重试与冷却。

为什么必须有这一层
------------------
后台总结与整理是**用户看不见的**。hermes_core 的 router.py 开篇记的就是这件事：
一旦那个模型笨、上下文短、或者临时抽气，总结就整批失败，而用户看不到报错，
只会觉得「记忆怎么不涨」。

fleet 的 v0.3.0 只包了一个 provider，配了 `llm_provider_id` 就锁死那一个——
它挂掉时，落库的消息永远不再总结，而这件事没有任何界面会告诉你。这与
「100 个账号长期跑」的前提直接冲突：单点故障在 100 个账号上被放大 100 倍。

与 hermes_core 的关系
---------------------
**只沿用分类口径，代码全部重写。** 分类函数的关键洞察值得继承：不看具体
SDK 类型、只看类名与消息文本，这样 OpenAI / Anthropic / Gemini / 各家
国产适配器都能覆盖。但 core 的实现把「选模型、重试、裁剪 prompt、写日志」
揉在一个 900 行的类里；这里只做两件事——**挑一个能用的 provider** 和
**把失败翻译成契约错误码**。prompt 裁剪归 worker，日志归这里。

契约约束
--------
`LlmProvider.complete` 的 docstring 明确要求：失败必须翻译成
`Unavailable` / `RateLimited` 并带上 `retry_after_s`。这条以前是空的——
`AstrBotProvider` 直接抛裸 `RuntimeError`，worker 侧只能 `except Exception`
再猜，正是 errors.py 里批评的那种退避。本模块负责把它补上。
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from hermes_contracts.errors import ErrorCode, HermesError, RateLimited, Unavailable
from hermes_contracts.models import ChatMessage, LlmResult
from hermes_contracts.ports import LlmProvider

# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------

# 「换个时间可能成功」→ 值得在同一模型上重试。
#
# 用**字符串**集合而不是 `ErrorCode` 枚举成员：`kind` 来自 `classify()`，是
# 字符串码。`class ErrorCode(str, Enum)` 的实例 hash 在 3.12 前后行为不同
# （早期是 hash(name)，后期才对齐 str），用枚举做成员检查在旧解释器上会
# 静默返回 False——表现为「重试逻辑突然不生效」，且不报任何错。
# AstrBot 可能跑在 3.10/3.11 上，不能赌这个。
RETRYABLE = frozenset({
    str(ErrorCode.UNAVAILABLE), str(ErrorCode.RATE_LIMITED), str(ErrorCode.INTERNAL),
})

# 「这个模型现在不能用」→ 进冷却，短期内不再优先尝试。
# 两条不变量：
#   - NOT_FOUND 不算：那是配置写错了 provider id，重试和冷却都掩盖不了问题，
#     应当直接报错让人去改配置。
#   - CTX（上下文超长）也不算：模型没坏，只是这批 prompt 太长。冷却它会误伤
#     后续正常长度的请求。
COOLDOWN_CODES = frozenset({
    str(ErrorCode.UNAVAILABLE), str(ErrorCode.RATE_LIMITED), str(ErrorCode.INTERNAL),
})

# 上下文超长单独归一类：在同一模型上重试毫无意义，但换上下文更大的模型有用。
# 用 `ctx` 伪码表示，不塞进 ErrorCode（那是跨模块契约，不能为内部实现加取值）。
CTX = "ctx"


def classify(exc: BaseException) -> str:
    """把各家 SDK 的异常归成有限几类，只返回字符串码，不吞异常。

    判定优先级：业务错误（上下文超长 / 鉴权 / 限流）→ 传输层（超时 / 连接）
    → 兜底。只看类名与消息文本，不依赖具体 SDK 类型。
    """
    if isinstance(exc, HermesError):
        # 已经是契约错误码，不再二次猜测——下游翻译过一次就够了。
        return str(exc.code)

    name = type(exc).__name__.lower()
    low = (str(exc) or "").lower()

    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):
        code = None

    # 上下文超长：各家文案不同，先单独识别。
    if ("context_length_exceeded" in low or "maximum context" in low
            or "too many tokens" in low or "prompt is too long" in low
            or "reduce the length" in low):
        return CTX
    if "context" in low and ("length" in low or "window" in low or "too long" in low):
        return CTX

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in name \
            or "timed out" in low:
        return str(ErrorCode.UNAVAILABLE)

    if code == 429 or "ratelimit" in name or "rate limit" in low \
            or "quota" in low or "too many requests" in low:
        # "too many requests" 是 RFC 6585 给 429 定的标准 reason phrase，
        # 也是 OpenAI 兼容接口最常见的报错文案（`Error code: 429 - Too Many
        # Requests`）。少这一条，最常见的一次限流会被归成 internal。
        return str(ErrorCode.RATE_LIMITED)

    if code in (401, 403) or "unauthorized" in low or "forbidden" in low \
            or "invalid api key" in low or "authentication" in low:
        return str(ErrorCode.FORBIDDEN)

    if code == 404 or "not found" in low or "no such model" in low \
            or "model_not_found" in low:
        return str(ErrorCode.NOT_FOUND)

    if "connection" in low or "connect" in name or "network" in low \
            or "unreachable" in low or "reset by peer" in low \
            or "service unavailable" in low or "overloaded" in low:
        return str(ErrorCode.UNAVAILABLE)

    if code is not None and 500 <= code < 600:
        return str(ErrorCode.UNAVAILABLE)

    if isinstance(exc, (ValueError, TypeError, KeyError)):
        # 参数/解析错：重试与换模型都是浪费，必须让它冒出去。
        return str(ErrorCode.INVALID_ARGUMENT)

    return str(ErrorCode.INTERNAL)


def as_contract_error(exc: BaseException, *, provider: str) -> HermesError:
    """把任意异常翻译成契约错误（已翻译的原样返回）。

    注意 `CTX` 不是 `ErrorCode` 的取值——它是本模块内部为了「换大模型」这个
    动作单独拆出来的类别。对外一律翻译成 `INVALID_ARGUMENT`（确实是输入太长）
    并在消息里保留 `context too long` 字样，好让上层日志能看出区别。
    """
    if isinstance(exc, HermesError):
        return exc
    kind = classify(exc)
    msg = f"{provider}: {type(exc).__name__}: {exc}"
    if kind == str(ErrorCode.RATE_LIMITED):
        return RateLimited(msg, retry_after_s=_retry_after(exc, default=5.0))
    if kind == CTX:
        return HermesError(msg, code=ErrorCode.INVALID_ARGUMENT,
                           detail={"kind": CTX})
    if kind in (str(ErrorCode.FORBIDDEN), str(ErrorCode.NOT_FOUND),
                str(ErrorCode.INVALID_ARGUMENT)):
        # 这三类不可重试，用基类承载具体 code，避免被上层当成可重试。
        return HermesError(msg, code=ErrorCode(kind))
    return Unavailable(msg, retry_after_s=_retry_after(exc, default=2.0))


def _retry_after(exc: BaseException, *, default: float) -> float:
    """读 provider 给的 Retry-After，读不到就用默认值。**不猜**。"""
    for attr in ("retry_after", "retry_after_s"):
        v = getattr(exc, attr, None)
        try:
            if v is not None:
                return max(0.0, float(v))
        except (TypeError, ValueError):
            pass
    hdr = getattr(getattr(exc, "response", None), "headers", None)
    try:
        if hdr:
            v = hdr.get("retry-after") or hdr.get("Retry-After")
            if v is not None:
                return max(0.0, float(v))
    except (TypeError, ValueError, AttributeError):
        pass
    return default


# ---------------------------------------------------------------------------
# 观测：环形记录
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Attempt:
    """一次尝试的结果。给 `/fleet status` 和将来的管理页看。"""

    ts: float
    provider: str
    purpose: str
    ok: bool
    kind: str = ""          # 失败类别；成功时为空
    latency_ms: float = 0.0
    model: str = ""
    detail: str = ""


@dataclass
class ChainStats:
    """累计计数。刻意不按 bot 拆分——模型是全局资源，按 bot 统计只会淹掉信号。"""

    calls: int = 0
    failures: int = 0
    by_provider: dict[str, int] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)

    def hit(self, provider: str, kind: str) -> None:
        self.calls += 1
        self.by_provider[provider] = self.by_provider.get(provider, 0) + 1
        if kind:
            self.failures += 1
            self.by_kind[kind] = self.by_kind.get(kind, 0) + 1


# ---------------------------------------------------------------------------
# 降级链
# ---------------------------------------------------------------------------


class FallbackProvider:
    """按顺序尝试多个 provider，实现 `LlmProvider` 协议。

    透明性：worker 只认 Protocol，所以套上这一层之后，`SessionSummarizer`
    和 `Consolidator` 一行都不用改——这是当初把 provider 抽出契约的直接回报。

    重试策略分三层，别混在一起：
    1. **单模型内重试**：只重试 `RETRYABLE` 的错误（抖动/限流/5xx），
       指数退避 + 抖动，次数由 `retry_times` 控制。
    2. **换模型**：不可重试错误（鉴权/404/参数错）立刻换下一个；
       `ctx` 超长也换（大上下文模型能扛）；单模型重试耗尽也换。
    3. **冷却**：连续失败的模型进冷却期，期间排在队尾。全部冷却时仍然
       强行试一次（`strict_cooldown=False`）——后台任务宁可慢一点，
       也不该因为冷却表算错而整晚不干活。
    """

    def __init__(
        self,
        providers: Sequence[tuple[str, LlmProvider]],
        *,
        retry_times: int = 1,
        retry_base_s: float = 0.8,
        cooldown_s: float = 180.0,
        strict_cooldown: bool = False,
        log_size: int = 64,
    ) -> None:
        if not providers:
            raise ValueError("降级链至少要有一个 provider")
        seen: set[str] = set()
        uniq: list[tuple[str, LlmProvider]] = []
        for name, impl in providers:
            if name in seen:
                # 同名会让冷却表互相污染：一个模型被冷却，另一个假装没事。
                # 这是配置错误（同一个 provider id 配了两遍），直接拒掉。
                raise ValueError(f"降级链里 provider 名重复：{name}")
            seen.add(name)
            uniq.append((name, impl))
        self._providers = uniq
        self.retry_times = max(0, int(retry_times))
        self.retry_base_s = max(0.0, float(retry_base_s))
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.strict_cooldown = strict_cooldown
        self.stats = ChainStats()
        self._cooldown_until: dict[str, float] = {}
        self.log: deque[Attempt] = deque(maxlen=max(8, int(log_size)))

    # -- 只读视图，给状态命令用 ------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self._providers]

    def cooldown_left(self, name: str) -> float:
        return max(0.0, self._cooldown_until.get(name, 0.0) - time.time())

    def chain(self) -> list[dict]:
        """当前实际尝试顺序（冷却中的排后面），含剩余冷却秒数。"""
        out = []
        for name, _ in self._providers:
            left = self.cooldown_left(name)
            out.append({"name": name, "cooling": left > 0, "cooldown_left": round(left, 1)})
        # 稳定排序：未冷却的保持原顺序（主模型优先），冷却的按剩余时间升序垫后
        out.sort(key=lambda x: (x["cooling"], x["cooldown_left"] if x["cooling"] else 0.0))
        return out

    # -- 主流程 ----------------------------------------------------------------

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LlmResult:
        order = self._ordered()
        if not order:
            raise Unavailable(
                "全部模型都在冷却，且已配置严格冷却；本轮放弃",
                retry_after_s=min(self.cooldown_s, 60.0),
            )

        last: HermesError | None = None
        for name, impl in order:
            for attempt in range(self.retry_times + 1):
                t0 = time.perf_counter()
                try:
                    res = await impl.complete(
                        messages, purpose=purpose,
                        max_tokens=max_tokens, temperature=temperature,
                    )
                except BaseException as exc:  # noqa: BLE001 — 必须分类后再决定吞不吞
                    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                        raise
                    # 先拿原始类别再翻译：CTX 翻译后会变成 INVALID_ARGUMENT，
                    # 丢掉「该换大模型」这个语义，所以要提前记下来。
                    raw = classify(exc)
                    err = as_contract_error(exc, provider=name)
                    # kind 直接用 classify 的结果，不用 err.code：
                    # classify 返回的就是契约码或 CTX，一致性有保障；而
                    # as_contract_error 会把 internal 兜底成 Unavailable，
                    # 用 err.code 会让日志里丢掉「内部错误」这个真类。
                    kind = raw
                    self.stats.hit(name, kind)
                    self._note(name, purpose, ok=False, kind=kind,
                               latency_ms=(time.perf_counter() - t0) * 1000,
                               detail=str(exc)[:200])
                    last = err

                    if kind in COOLDOWN_CODES:
                        self._cool(name)

                    # 决定：同一模型重试，还是立刻换下一个
                    can_retry = (kind in RETRYABLE and attempt < self.retry_times)
                    if not can_retry:
                        break
                    # 限流时优先尊重 provider 给的 Retry-After，但仍设上限：
                    # 后台任务不能为了等一个模型睡上十分钟。
                    await asyncio.sleep(self._backoff(attempt, err))
                    continue

                self.stats.hit(name, "")
                self._note(name, purpose, ok=True,
                           latency_ms=(time.perf_counter() - t0) * 1000,
                           model=res.model)
                self._cooldown_until.pop(name, None)
                return res

        # 全链失败：抛出**最后一个**错误，并标注整条链都试过了。
        # 保留原始 code 很重要——若最后是 RateLimited，上层该退避而不是立刻死信。
        if last is None:  # pragma: no cover — order 非空时不可能走到
            raise Unavailable("降级链无可用模型")
        last.detail = {**(last.detail or {}), "chain": self.names, "tried": len(order)}
        raise last

    # -- 内部 ----------------------------------------------------------------

    def _ordered(self) -> list[tuple[str, LlmProvider]]:
        """未冷却的按原顺序在前，冷却中的按剩余时间垫后。

        全部冷却时：`strict_cooldown=False`（默认）仍返回全链——后台静默
        不干活比多撞一次坏模型糟糕得多。
        """
        warm: list[tuple[str, LlmProvider]] = []
        cold: list[tuple[float, tuple[str, LlmProvider]]] = []
        for name, impl in self._providers:
            left = self.cooldown_left(name)
            if left > 0:
                cold.append((left, (name, impl)))
            else:
                warm.append((name, impl))
        cold.sort(key=lambda x: x[0])
        out = warm + [p for _, p in cold]
        if not warm and self.strict_cooldown:
            return []
        return out

    def _cool(self, name: str) -> None:
        if self.cooldown_s <= 0:
            return
        self._cooldown_until[name] = time.time() + self.cooldown_s

    def _backoff(self, attempt: int, err: HermesError) -> float:
        exp = self.retry_base_s * (2 ** attempt)
        jitter = random.uniform(0, self.retry_base_s * 0.5)
        delay = exp + jitter
        if getattr(err, "retry_after_s", None):
            delay = max(delay, min(float(err.retry_after_s), 10.0))
        return min(delay, 15.0)   # 单次等待硬上限，避免一轮任务卡几分钟

    def _note(self, name: str, purpose: str, *, ok: bool, kind: str = "",
              latency_ms: float = 0.0, model: str = "", detail: str = "") -> None:
        self.log.append(Attempt(
            ts=time.time(), provider=name, purpose=purpose, ok=ok, kind=kind,
            latency_ms=round(latency_ms, 1), model=model, detail=detail,
        ))

    def summary(self) -> str:  # 给 `/fleet status` 用的一行摘要
        chain = self.chain()
        parts = [
            f"{c['name']}{'（冷却 ' + str(int(c['cooldown_left'])) + 's）' if c['cooling'] else ''}"
            for c in chain
        ]
        s = " → ".join(parts)
        if self.stats.failures:
            kinds = ", ".join(f"{k}×{v}" for k, v in sorted(self.stats.by_kind.items()))
            s += f"；失败 {self.stats.failures}/{self.stats.calls}（{kinds}）"
        return s
