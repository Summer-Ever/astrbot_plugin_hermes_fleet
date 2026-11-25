"""`JobQueue` 的一致性套件。

队列是这个系统里最容易「看起来对、实际上会重复执行」的部分。重复执行总结
不是小事：会让同一条记忆被写两次、水位被推两次、模型成本翻倍。

所以这套用例的重心是状态机与超时，而不是「能不能存取」。

关于并发：`test_concurrent_claim_does_not_double_assign` 在内存实现下走的是
asyncio 串行等价路径，**不能**证明 `SKIP LOCKED` 真的生效。Postgres 实现
应当 override 这个用例，用真多连接（threading + 独立连接）跑同一场景。
这一点是刻意标注的——用一个假通过换心安，是这类系统最常见的自欺。
"""

from __future__ import annotations

import asyncio
import unittest

from ..clock import Clock, ManualClock
from ..ids import BotId
from ..models import JobKind, JobState
from ..ports import JobQueue
from .fixtures import BOT, BOT_B

SCOPE = "bot_alpha|session_1"


class JobQueueConformance(unittest.IsolatedAsyncioTestCase):
    __test__ = False

    async def make_queue(self, clock: Clock) -> JobQueue:
        raise unittest.SkipTest("abstract conformance base: override make_queue()")

    async def asyncSetUp(self) -> None:
        self.clock = ManualClock()
        self.queue = await self.make_queue(self.clock)

    async def _enqueue(self, *, kind: str = JobKind.SUMMARIZE, scope: str = SCOPE, bot: BotId = BOT):
        return await self.queue.enqueue(kind, bot, scope, payload={"cursor": 0})

    # -- 入队与去重 --------------------------------------------------------

    async def test_enqueue_creates_pending_job(self) -> None:
        job = await self._enqueue()
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.state, JobState.PENDING)
        self.assertEqual(job.attempts, 0)
        self.assertEqual(job.payload, {"cursor": 0})

    async def test_enqueue_is_deduplicated_while_in_flight(self) -> None:
        """同一 (kind, bot, scope) 在途时只允许一条。

        这是替代 hermes_core「水位重算」的核心机制：调用方不需要先查再写，
        也不需要知道有没有在跑，重复入队自然被吸收。
        """
        first = await self._enqueue()
        second = await self._enqueue()
        self.assertIsNotNone(first)
        self.assertIsNone(second, "重复入队必须返回 None，而不是新建一条")

    async def test_dedup_scoped_per_bot(self) -> None:
        self.assertIsNotNone(await self._enqueue(bot=BOT))
        self.assertIsNotNone(
            await self._enqueue(bot=BOT_B),
            "不同 bot 的同名会话必须各自入队，不能互相吞掉",
        )

    async def test_enqueue_after_done_is_allowed(self) -> None:
        """完成后可以再次入队——去重只覆盖「在途」，不是永久锁。

        如果这里写错（把 done 也算在 dedup 范围内），表现是「一个会话只被
        总结一次，之后再也不总结」，且日志里一切正常。
        """
        job = await self._enqueue()
        assert job is not None
        claimed = (await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60))[0]
        await self.queue.complete(claimed)
        again = await self._enqueue()
        self.assertIsNotNone(again, "done 之后必须允许重新入队")

    # -- 领取 --------------------------------------------------------------

    async def test_claim_marks_running_and_bumps_attempts(self) -> None:
        job = await self._enqueue()
        assert job is not None
        before = self.clock.now()
        claimed = await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60, limit=1)
        self.assertEqual(len(claimed), 1)
        c = claimed[0]
        self.assertEqual(int(c.id), int(job.id))
        self.assertEqual(c.state, JobState.RUNNING)
        self.assertEqual(c.attempts, 1)
        self.assertEqual(c.locked_by, "w1")
        self.assertGreater(c.visible_at, before, "领取时必须把 visible_at 推后")

    async def test_claim_respects_kind_filter(self) -> None:
        await self._enqueue(kind=JobKind.CONSOLIDATE)
        got = await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60)
        self.assertEqual(len(got), 0, "只领指定 kind，否则总结 worker 会去跑整理")

    async def test_claim_is_empty_when_nothing_pending(self) -> None:
        self.assertEqual(await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60), ())

    async def test_invisible_job_cannot_be_reclaimed_before_timeout(self) -> None:
        await self._enqueue()
        await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60)
        second = await self.queue.claim("w2", [JobKind.SUMMARIZE], visibility_s=60)
        self.assertEqual(len(second), 0, "可见性超时之前不允许重复领取")

    async def test_job_becomes_claimable_after_visibility_timeout(self) -> None:
        """worker 被 kill -9 后，任务靠超时自然回到队列。

        这是不用「释放锁」的原因：进程没有机会执行任何清理代码。
        """
        await self._enqueue()
        await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60)
        self.clock.advance(61)
        again = await self.queue.claim("w2", [JobKind.SUMMARIZE], visibility_s=60)
        self.assertEqual(len(again), 1, "超时后必须能被重新领取")
        self.assertEqual(again[0].attempts, 2, "重新领取要累计 attempts，否则会无限重试")

    async def test_concurrent_claim_does_not_double_assign(self) -> None:
        """同一时刻多个 worker 抢一条任务，只能有一个成功。

        注意：内存实现下 asyncio.gather 是协作式串行，这条用例只能验证
        「实现没有明显的先读后写竞态」。真正的并发验证由 Postgres 实现
        override 本用例、用多个独立连接完成。
        """
        await self._enqueue()
        results = await asyncio.gather(
            self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60, limit=1),
            self.queue.claim("w2", [JobKind.SUMMARIZE], visibility_s=60, limit=1),
        )
        total = sum(len(r) for r in results)
        self.assertEqual(total, 1, f"一条任务被领取了 {total} 次")

    async def test_heartbeat_extends_visibility(self) -> None:
        """长任务（整理要跑 6 次模型调用）必须能续期，否则会被误判为崩溃。"""
        await self._enqueue()
        job = (await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=10))[0]
        self.clock.advance(5)
        renewed = await self.queue.heartbeat(job, visibility_s=60)
        self.assertGreater(renewed.visible_at, job.visible_at)
        self.clock.advance(30)  # 已过原超时，但未过新超时
        self.assertEqual(
            len(await self.queue.claim("w2", [JobKind.SUMMARIZE], visibility_s=60)),
            0,
            "续期后不应被重复领取",
        )

    # -- 完成与失败 --------------------------------------------------------

    async def test_completed_job_is_not_claimable(self) -> None:
        await self._enqueue()
        job = (await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60))[0]
        await self.queue.complete(job)
        self.clock.advance(120)
        self.assertEqual(len(await self.queue.claim("w2", [JobKind.SUMMARIZE], visibility_s=60)), 0)

    async def test_complete_is_idempotent(self) -> None:
        """worker 崩溃前完成了、但没来得及 ack，重试会再 complete 一次。
        这里必须不报错，否则重试逻辑要把「已完成」当异常处理。"""
        await self._enqueue()
        job = (await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60))[0]
        await self.queue.complete(job)
        await self.queue.complete(job)

    async def test_fail_under_limit_is_retriable(self) -> None:
        await self._enqueue()
        job = (await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60))[0]
        failed = await self.queue.fail(job, "provider 503", max_attempts=3)
        self.assertEqual(failed.state, JobState.FAILED)
        self.assertIn("503", failed.last_error)
        reclaimed = await self.queue.claim("w2", [JobKind.SUMMARIZE], visibility_s=60)
        self.assertEqual(len(reclaimed), 1, "未超上限必须能被再次领取")

    async def test_fail_over_limit_goes_dead_and_stops_retrying(self) -> None:
        """超过上限进死信，**不再重试**。

        hermes_core 没有死信：失败任务会被下一轮水位重算再扫一遍，无限循环。
        这条用例就是防止那种行为被继承过来。
        """
        await self._enqueue()
        job = (await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60))[0]
        dead = await self.queue.fail(job, "boom", max_attempts=1)
        self.assertEqual(dead.state, JobState.DEAD)
        self.clock.advance(3600)
        self.assertEqual(
            len(await self.queue.claim("w2", [JobKind.SUMMARIZE], visibility_s=60)),
            0,
            "dead 状态的任务不允许被再次领取",
        )

    async def test_stats_counts_by_state(self) -> None:
        await self._enqueue(scope="s1")
        await self._enqueue(scope="s2")
        job = (await self.queue.claim("w1", [JobKind.SUMMARIZE], visibility_s=60))[0]
        await self.queue.complete(job)
        stats = await self.queue.stats(BOT)
        self.assertGreaterEqual(stats.get(JobState.DONE, 0), 1)
        self.assertGreaterEqual(stats.get(JobState.PENDING, 0), 1)
