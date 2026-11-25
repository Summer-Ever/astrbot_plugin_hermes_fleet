"""降级链测试。

这一层最容易被漏测，因为它是**后台静默**的：模型挂了没有界面会报错，
只表现为「记忆怎么不涨」。所以每个分支都必须钉死——
尤其是「哪些错误该重试、哪些该立刻换模型」，猜错一次就是几百次无效重试。
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for p in ("contracts", "workers"):
    sys.path.insert(0, str(_ROOT / "packages" / p / "src"))

from hermes_contracts.errors import (  # noqa: E402
    ErrorCode,
    Forbidden,
    NotFound,
    RateLimited,
    Unavailable,
)
from hermes_contracts.models import ChatMessage, LlmResult  # noqa: E402
from hermes_contracts.ports import LlmProvider  # noqa: E402
from hermes_workers.fallback import (  # noqa: E402
    CTX,
    FallbackProvider,
    as_contract_error,
    classify,
)

MSGS = [ChatMessage(role="user", content="总结一下")]


def run(coro):
    return asyncio.run(coro)


class FakeProvider:
    """按脚本决定第 n 次调用抛什么（None 表示成功）。"""

    def __init__(self, name: str = "p", script: list | None = None) -> None:
        self.name = name
        self.script = list(script or [])
        self.calls = 0

    async def complete(self, messages, *, purpose, max_tokens=None, temperature=None):
        self.calls += 1
        item = self.script.pop(0) if self.script else None
        if item is not None:
            raise item
        return LlmResult(text=f"ok:{self.name}", model=self.name, latency_ms=1.0)


class Boom(Exception):
    """无法分类的异常，用来看兜底行为。"""


class HTTPish(Exception):
    def __init__(self, msg: str, status_code: int) -> None:
        super().__init__(msg)
        self.status_code = status_code


class TestClassify(unittest.TestCase):
    def test_rate_limit_by_status_code(self) -> None:
        self.assertEqual(classify(HTTPish("slow down", 429)), str(ErrorCode.RATE_LIMITED))

    def test_rate_limit_by_message(self) -> None:
        self.assertEqual(classify(Exception("Rate limit reached")),
                         str(ErrorCode.RATE_LIMITED))

    def test_context_too_long_is_its_own_kind(self) -> None:
        """上下文超长必须单独一类：重试无意义，但换大模型有用。"""
        self.assertEqual(classify(Exception("This model's maximum context length is 8192")),
                         CTX)
        self.assertEqual(classify(Exception("context_length_exceeded")), CTX)

    def test_auth_error(self) -> None:
        self.assertEqual(classify(HTTPish("bad key", 401)), str(ErrorCode.FORBIDDEN))
        self.assertEqual(classify(Exception("invalid api key")), str(ErrorCode.FORBIDDEN))

    def test_model_not_found(self) -> None:
        self.assertEqual(classify(Exception("model_not_found")), str(ErrorCode.NOT_FOUND))

    def test_timeout_is_unavailable(self) -> None:
        self.assertEqual(classify(asyncio.TimeoutError()), str(ErrorCode.UNAVAILABLE))

    def test_server_error(self) -> None:
        self.assertEqual(classify(HTTPish("oops", 502)), str(ErrorCode.UNAVAILABLE))

    def test_programming_errors_are_not_retryable(self) -> None:
        self.assertEqual(classify(TypeError("wrong args")),
                         str(ErrorCode.INVALID_ARGUMENT))

    def test_already_translated_is_not_guessed(self) -> None:
        """契约错误不再二次猜测——下游翻译过一次就够了。"""
        self.assertEqual(classify(Forbidden("nope")), str(ErrorCode.FORBIDDEN))
        self.assertEqual(classify(RateLimited("slow")), str(ErrorCode.RATE_LIMITED))

    def test_unknown_falls_back_to_internal(self) -> None:
        self.assertEqual(classify(Boom("???")), str(ErrorCode.INTERNAL))

    def test_openai_style_429_is_rate_limited(self) -> None:
        """「429 Too Many Requests」是 OpenAI 兼容接口最常见的限流文案。

        少了这条，最常见的一次限流会被归成 internal——两者都在冷却和重试
        集合里，所以行为看着没错，但 `/fleet 状态` 会报错类别，排障时误判。
        """
        self.assertEqual(classify(Exception("429 Too Many Requests")),
                         str(ErrorCode.RATE_LIMITED))
        self.assertEqual(classify(Exception("Error code: 429 - Too Many Requests")),
                         str(ErrorCode.RATE_LIMITED))

    def test_service_unavailable_message(self) -> None:
        self.assertEqual(classify(Exception("service unavailable")),
                         str(ErrorCode.UNAVAILABLE))
        self.assertEqual(classify(Exception("server overloaded")),
                         str(ErrorCode.UNAVAILABLE))


class TestTranslate(unittest.TestCase):
    def test_rate_limited_keeps_retry_after(self) -> None:
        err = as_contract_error(HTTPish("429", 429), provider="a")
        self.assertIsInstance(err, RateLimited)
        self.assertTrue(err.code.retryable)

    def test_forbidden_is_not_retryable(self) -> None:
        err = as_contract_error(HTTPish("401", 401), provider="a")
        self.assertEqual(err.code, ErrorCode.FORBIDDEN)
        self.assertFalse(err.code.retryable, "鉴权错重试只会浪费配额")

    def test_ctx_becomes_invalid_argument_but_keeps_marker(self) -> None:
        """CTX 不是契约取值，对外翻译成 INVALID_ARGUMENT，但要留痕。"""
        err = as_contract_error(Exception("maximum context length"), provider="a")
        self.assertEqual(err.code, ErrorCode.INVALID_ARGUMENT)
        self.assertEqual(err.detail.get("kind"), CTX)
        self.assertFalse(err.code.retryable)

    def test_translated_error_passes_through(self) -> None:
        original = NotFound("model gone")
        self.assertIs(as_contract_error(original, provider="a"), original)

    def test_message_carries_provider_name(self) -> None:
        err = as_contract_error(Boom("x"), provider="deepseek")
        self.assertIn("deepseek", str(err))


class TestChain(unittest.TestCase):
    def test_primary_wins_without_touching_fallback(self) -> None:
        a, b = FakeProvider("a"), FakeProvider("b")
        chain = FallbackProvider([("a", a), ("b", b)])
        res = run(chain.complete(MSGS, purpose="summarize"))
        self.assertEqual(res.text, "ok:a")
        self.assertEqual(b.calls, 0)

    def test_falls_back_on_unavailable(self) -> None:
        a = FakeProvider("a", [Unavailable("down")])
        b = FakeProvider("b")
        chain = FallbackProvider([("a", a), ("b", b)], retry_times=0)
        res = run(chain.complete(MSGS, purpose="summarize"))
        self.assertEqual(res.text, "ok:b")
        self.assertEqual(a.calls, 1, "retry_times=0 时不该重试")
        self.assertEqual(b.calls, 1)

    def test_auth_error_switches_immediately_without_retry(self) -> None:
        """鉴权错重试毫无意义——核心断言就是这个 calls==1。"""
        a = FakeProvider("a", [Forbidden("bad key")])
        b = FakeProvider("b")
        chain = FallbackProvider([("a", a), ("b", b)], retry_times=3)
        run(chain.complete(MSGS, purpose="summarize"))
        self.assertEqual(a.calls, 1, "不可重试错误必须立刻换模型")

    def test_retries_within_model_before_switching(self) -> None:
        a = FakeProvider("a", [Unavailable("blip"), Unavailable("blip2")])
        b = FakeProvider("b")
        chain = FallbackProvider([("a", a), ("b", b)], retry_times=1,
                                 retry_base_s=0.0)
        res = run(chain.complete(MSGS, purpose="summarize"))
        self.assertEqual(a.calls, 2, "首次 + 1 次重试")
        self.assertEqual(res.text, "ok:b")

    def test_ctx_switches_model_but_does_not_cool_down(self) -> None:
        """上下文超长说明 prompt 太长，不是模型坏了——冷却它会误伤正常请求。"""
        a = FakeProvider("a", [Exception("maximum context length is 4096")])
        b = FakeProvider("b")
        chain = FallbackProvider([("a", a), ("b", b)], cooldown_s=999)
        run(chain.complete(MSGS, purpose="summarize"))
        self.assertEqual(chain.cooldown_left("a"), 0.0)

    def test_failure_cools_down_and_reorders(self) -> None:
        a = FakeProvider("a", [Unavailable("down")])
        b = FakeProvider("b")
        chain = FallbackProvider([("a", a), ("b", b)], retry_times=0, cooldown_s=60)
        run(chain.complete(MSGS, purpose="summarize"))
        self.assertGreater(chain.cooldown_left("a"), 0)
        self.assertEqual(chain.chain()[0]["name"], "b", "冷却的模型应排到队尾")

    def test_success_clears_cooldown(self) -> None:
        a = FakeProvider("a", [Unavailable("down")])
        chain = FallbackProvider([("a", a)], retry_times=0, cooldown_s=60)
        with self.assertRaises(Exception):
            run(chain.complete(MSGS, purpose="summarize"))
        self.assertGreater(chain.cooldown_left("a"), 0)
        run(chain.complete(MSGS, purpose="summarize"))   # 这次成功
        self.assertEqual(chain.cooldown_left("a"), 0.0)

    def test_all_cooling_still_tries_by_default(self) -> None:
        """后台静默不干活，比多撞一次坏模型糟糕得多。"""
        a = FakeProvider("a", [Unavailable("down")])
        chain = FallbackProvider([("a", a)], retry_times=0, cooldown_s=999)
        with self.assertRaises(Exception):
            run(chain.complete(MSGS, purpose="summarize"))
        a.script = []          # 恢复
        res = run(chain.complete(MSGS, purpose="summarize"))
        self.assertEqual(res.text, "ok:a")

    def test_strict_cooldown_gives_up(self) -> None:
        a = FakeProvider("a", [Unavailable("down")])
        chain = FallbackProvider([("a", a)], retry_times=0, cooldown_s=999,
                                 strict_cooldown=True)
        with self.assertRaises(Exception):
            run(chain.complete(MSGS, purpose="summarize"))
        with self.assertRaises(Unavailable):
            run(chain.complete(MSGS, purpose="summarize"))

    def test_all_fail_raises_last_error(self) -> None:
        """保留最后一个错误的 code：若最后是限流，上层该退避而不是死信。"""
        a = FakeProvider("a", [Unavailable("a down")])
        b = FakeProvider("b", [Forbidden("b bad key")])
        chain = FallbackProvider([("a", a), ("b", b)], retry_times=0)
        with self.assertRaises(Forbidden):
            run(chain.complete(MSGS, purpose="summarize"))

    def test_error_carries_chain_info(self) -> None:
        a = FakeProvider("a", [Unavailable("x")])
        chain = FallbackProvider([("a", a)], retry_times=0)
        try:
            run(chain.complete(MSGS, purpose="summarize"))
        except Unavailable as exc:
            self.assertEqual(exc.detail.get("chain"), ["a"])
            self.assertEqual(exc.detail.get("tried"), 1)

    def test_duplicate_names_rejected(self) -> None:
        """重名会让冷却表互相污染：一个被冷却，另一个假装没事。"""
        with self.assertRaises(ValueError):
            FallbackProvider([("a", FakeProvider()), ("a", FakeProvider())])

    def test_empty_chain_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FallbackProvider([])

    def test_satisfies_protocol(self) -> None:
        """透明性：套上这一层，worker 不用改一行。"""
        chain = FallbackProvider([("a", FakeProvider())])
        self.assertIsInstance(chain, LlmProvider)

    def test_stats_and_log_recorded(self) -> None:
        a = FakeProvider("a", [Unavailable("x")])
        b = FakeProvider("b")
        chain = FallbackProvider([("a", a), ("b", b)], retry_times=0)
        run(chain.complete(MSGS, purpose="consolidate"))
        self.assertEqual(chain.stats.calls, 2)
        self.assertEqual(chain.stats.failures, 1)
        self.assertEqual([x.ok for x in chain.log], [False, True])
        self.assertEqual(chain.log[0].purpose, "consolidate")

    def test_log_keeps_true_kind_not_translated_code(self) -> None:
        """internal 在翻译时兜底成 Unavailable，但日志必须记 internal。

        否则排障时看到「unavailable」却不知道是 provider 真的不可用，
        还是插件自己抛了个未分类异常。
        """
        chain = FallbackProvider([("a", FakeProvider("a", [Boom("???")]))],
                                 retry_times=0)
        with self.assertRaises(Exception):
            run(chain.complete(MSGS, purpose="summarize"))
        self.assertEqual(chain.log[0].kind, str(ErrorCode.INTERNAL))
        self.assertEqual(chain.stats.by_kind, {str(ErrorCode.INTERNAL): 1})

    def test_summary_is_human_readable(self) -> None:
        chain = FallbackProvider([("a", FakeProvider("a"))])
        self.assertIn("a", chain.summary())

    def test_purpose_is_forwarded(self) -> None:
        seen = {}

        class Spy:
            async def complete(self, messages, *, purpose, max_tokens=None,
                               temperature=None):
                seen["purpose"] = purpose
                seen["max_tokens"] = max_tokens
                return LlmResult(text="x")

        chain = FallbackProvider([("s", Spy())])
        run(chain.complete(MSGS, purpose="judge", max_tokens=128))
        self.assertEqual(seen["purpose"], "judge")
        self.assertEqual(seen["max_tokens"], 128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
