"""契约本身的纯逻辑测试（L1）。

这类测试的价值在于**改契约时会立刻红**。多 agent 并行时，契约是唯一被共享
的东西，所以它必须比任何实现都有更密的测试覆盖。
"""

from __future__ import annotations

import unittest

from hermes_contracts import (
    BotId,
    ErrorCode,
    Forbidden,
    Job,
    JobKind,
    JobState,
    ManualClock,
    Memory,
    MemoryId,
    MemoryKey,
    MemoryScope,
    OwnerId,
    ResidentBlock,
    ResidentScope,
    SessionKey,
    SystemClock,
    Visibility,
    clip,
    validate_bot_id,
)
from hermes_contracts.models import MAX_MEMORY_CHARS


class TestResidentScope(unittest.TestCase):
    def test_roundtrip_private(self) -> None:
        s = ResidentScope.for_private()
        self.assertEqual(s.to_key(), "private")
        self.assertEqual(ResidentScope.parse("private"), s)

    def test_roundtrip_group(self) -> None:
        s = ResidentScope.for_group("123456")
        self.assertEqual(s.to_key(), "group:123456")
        self.assertEqual(ResidentScope.parse("group:123456"), s)

    def test_rejects_malformed(self) -> None:
        for bad in ("", "group:", "GROUP:1", "public", "group"):
            with self.assertRaises(ValueError, msg=f"{bad!r} 应该被拒绝"):
                ResidentScope.parse(bad)

    def test_group_requires_id(self) -> None:
        with self.assertRaises(ValueError):
            ResidentScope("group")

    def test_private_rejects_group_id(self) -> None:
        with self.assertRaises(ValueError):
            ResidentScope("private", "123")


class TestMemoryKey(unittest.TestCase):
    def test_user_scope_requires_owner(self) -> None:
        with self.assertRaises(ValueError):
            MemoryKey(BotId("b"), MemoryScope.USER, owner_id="")

    def test_group_scope_requires_group(self) -> None:
        with self.assertRaises(ValueError):
            MemoryKey(BotId("b"), MemoryScope.GROUP, group_id="")

    def test_global_rejects_owner_and_group(self) -> None:
        with self.assertRaises(ValueError):
            MemoryKey(BotId("b"), MemoryScope.GLOBAL, owner_id="alice")

    def test_readable_by(self) -> None:
        key = MemoryKey(BotId("b"), MemoryScope.USER, owner_id="alice")
        self.assertTrue(key.readable_by(OwnerId("alice")))
        self.assertFalse(key.readable_by(OwnerId("bob")))

        gkey = MemoryKey(BotId("b"), MemoryScope.GROUP, group_id="g1")
        self.assertTrue(gkey.readable_by(OwnerId("bob"), viewer_group="g1"))
        self.assertFalse(gkey.readable_by(OwnerId("bob"), viewer_group="g2"))
        self.assertFalse(gkey.readable_by(OwnerId("bob")))  # 私聊上下文读不到群记忆

        self.assertTrue(MemoryKey(BotId("b"), MemoryScope.GLOBAL).readable_by(OwnerId("anyone")))


class TestMemoryVisibility(unittest.TestCase):
    def _mem(self, **kw) -> Memory:
        base = dict(
            id=MemoryId(1),
            bot_id=BotId("b"),
            content="x",
            scope=MemoryScope.USER,
            visibility=Visibility.PRIVATE,
            owner_id="alice",
        )
        base.update(kw)
        return Memory(**base)  # type: ignore[arg-type]

    def test_private_only_owner(self) -> None:
        m = self._mem()
        self.assertTrue(m.can_read(OwnerId("alice")))
        self.assertFalse(m.can_read(OwnerId("bob")))

    def test_group_visibility_requires_same_group(self) -> None:
        m = self._mem(scope=MemoryScope.GROUP, visibility=Visibility.GROUP, owner_id="", group_id="g1")
        self.assertTrue(m.can_read(OwnerId("bob"), viewer_group="g1"))
        self.assertFalse(m.can_read(OwnerId("bob"), viewer_group="g2"))

    def test_public_readable_by_anyone(self) -> None:
        m = self._mem(visibility=Visibility.PUBLIC)
        self.assertTrue(m.can_read(OwnerId("stranger")))

    def test_merged_is_never_readable(self) -> None:
        """软删除的记录对任何人都不可读——这一条比 visibility 优先级更高。"""
        m = self._mem().merged(MemoryId(9), at=1.0)
        self.assertFalse(m.can_read(OwnerId("alice")))


class TestClip(unittest.TestCase):
    def test_short_text_untouched(self) -> None:
        self.assertEqual(clip("短", 100), "短")

    def test_none_becomes_empty(self) -> None:
        self.assertEqual(clip(None, 10), "")

    def test_truncates_and_marks(self) -> None:
        # 输入必须超过上限才会触发截断；写成 100 字符配 500 上限是无效用例
        out = clip("啊" * (MAX_MEMORY_CHARS + 500), MAX_MEMORY_CHARS)
        self.assertEqual(len(out), MAX_MEMORY_CHARS)
        self.assertTrue(out.endswith("[已截断]"))

    def test_never_exceeds_limit_even_when_tiny(self) -> None:
        """上限小于标记长度时，结果也不能超限。

        这是真实踩过的坑：截断函数自己没截住，写库时的长度校验就形同虚设。
        """
        for limit in (1, 2, 5, 6):
            self.assertLessEqual(len(clip("啊" * 50, limit)), limit, f"limit={limit}")

    def test_zero_limit(self) -> None:
        self.assertEqual(clip("啊" * 10, 0), "")


class TestIdentifiers(unittest.TestCase):
    def test_validate_bot_id_accepts_safe_chars(self) -> None:
        self.assertEqual(validate_bot_id("qq_main:1"), BotId("qq_main:1"))

    def test_validate_bot_id_rejects_path_separators(self) -> None:
        """bot_id 会进文件名与 URL，放行斜杠等于给自己开路径穿越。"""
        for bad in ("a/b", "../x", "a b", "", "a" * 65):
            with self.assertRaises(ValueError, msg=f"{bad!r} 应该被拒绝"):
                validate_bot_id(bad)

    def test_session_key_roundtrip(self) -> None:
        k = SessionKey(BotId("bot_alpha"), "group_100")
        self.assertEqual(SessionKey.parse(k.to_key()), k)

    def test_session_key_rejects_garbage(self) -> None:
        for bad in ("", "nosep", "|onlyid", "bot|"):
            with self.assertRaises(ValueError):
                SessionKey.parse(bad)


class TestErrors(unittest.TestCase):
    def test_retryable_set(self) -> None:
        self.assertTrue(ErrorCode.UNAVAILABLE.retryable)
        self.assertTrue(ErrorCode.RATE_LIMITED.retryable)
        self.assertFalse(ErrorCode.NOT_FOUND.retryable)
        self.assertFalse(ErrorCode.INVALID_ARGUMENT.retryable)
        self.assertFalse(ErrorCode.FORBIDDEN.retryable)

    def test_http_mapping_is_total(self) -> None:
        """每个错误码都必须有 HTTP 映射，漏一个就是 500 泄露内部语义。"""
        for code in ErrorCode:
            self.assertIsInstance(code.http_status, int)
            self.assertGreaterEqual(code.http_status, 400)

    def test_str_is_readable(self) -> None:
        err = Forbidden("不是你的记忆")
        self.assertIn("forbidden", str(err))
        self.assertIn("不是你的记忆", str(err))


class TestJobStateMachine(unittest.TestCase):
    def _job(self) -> Job:
        return Job(id=1, kind=JobKind.SUMMARIZE, bot_id=BotId("b"), scope_key="b|s")

    def test_claim_extends_visibility_and_counts_attempt(self) -> None:
        j = self._job().claimed_by("w1", at=100.0, visibility_s=60)
        self.assertEqual(j.state, JobState.RUNNING)
        self.assertEqual(j.attempts, 1)
        self.assertEqual(j.visible_at, 160.0)

    def test_fail_under_limit_is_retriable(self) -> None:
        j = self._job().claimed_by("w1", at=100.0, visibility_s=60)
        j = j.failed("boom", at=110.0, max_attempts=3)
        self.assertEqual(j.state, JobState.FAILED)
        self.assertIsNone(j.locked_by, "失败要释放 worker 归属，否则面板上看不出来")

    def test_fail_over_limit_dead(self) -> None:
        j = self._job().claimed_by("w1", at=100.0, visibility_s=60)
        j = j.failed("boom", at=110.0, max_attempts=1)
        self.assertEqual(j.state, JobState.DEAD)
        self.assertTrue(j.is_terminal)

    def test_error_text_is_clipped(self) -> None:
        j = self._job().claimed_by("w1", at=100.0, visibility_s=60)
        j = j.failed("x" * 5000, at=110.0, max_attempts=3)
        self.assertLessEqual(len(j.last_error), 1000)

    def test_frozen(self) -> None:
        j = self._job()
        with self.assertRaises(Exception):
            j.state = JobState.DONE  # type: ignore[misc]


class TestClock(unittest.TestCase):
    def test_manual_clock_advances(self) -> None:
        c = ManualClock(1000.0)
        self.assertEqual(c.now(), 1000.0)
        self.assertEqual(c.advance(60), 1060.0)
        self.assertEqual(c.now(), 1060.0)

    def test_manual_clock_refuses_rewind(self) -> None:
        """不允许回拨：回拨会让已过期的任务重新可见，通常是测试写错了。"""
        with self.assertRaises(ValueError):
            ManualClock().advance(-1)

    def test_system_clock_is_monotonic_enough(self) -> None:
        c = SystemClock()
        self.assertGreater(c.now(), 1_600_000_000)


class TestResidentBlocks(unittest.TestCase):
    def test_seven_blocks(self) -> None:
        self.assertEqual(
            len(ResidentBlock), 7,
            "常驻是封闭七席（新增 intimacy），加成员要改注入预算")

    def test_address_is_first(self) -> None:
        """称呼必须排第一：模型对靠前内容注意力更高，排后面会叫错人。"""
        self.assertIs(ResidentBlock.ordered()[0], ResidentBlock.ADDRESS)

    def test_ordered_covers_all(self) -> None:
        self.assertEqual(set(ResidentBlock.ordered()), set(ResidentBlock))


if __name__ == "__main__":
    unittest.main()
