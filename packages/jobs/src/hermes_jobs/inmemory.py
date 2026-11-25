"""`JobQueue` 的内存实现。

它存在的理由和 `InMemoryMemoryStore` 一样：让 worker / api 模块的 agent
不必等 Postgres 就绪就能开发，并且先在这里把 `JobQueueConformance` 跑绿。

但队列比存储更容易写错，错的方式还特别隐蔽：
- 「领取」写成先查后改，并发下两个 worker 拿到同一条（重复总结）；
- `failed` 不进可领取集合，失败任务永远卡住（积压但不报错）；
- `dead` 忘了排除，死信被反复重试（配额烧光）。

所以这个实现是**严格照着 conformance 的语义**写的，而不是「看起来能跑就行」。

关于状态比较：`JobState` 是普通常量类（不是 Enum），所以一律用 `==`，
绝不能用 `is`——同一个字符串在两处引用可能不是同一个对象，`is` 会在
「内存版通过、经过一次 JSON 往返的 Postgres 版失败」这种地方爆掉。
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, Sequence

from hermes_contracts import (
    BotId,
    Clock,
    Conflict,
    Job,
    JobId,
    JobState,
)

# 可被领取的状态：
# - pending：新任务
# - failed ：上次失败但未超上限，立即可重试
# - running：正在跑但可见性已超时，视为 worker 崩溃，允许接管
_CLAIMABLE = (JobState.PENDING, JobState.FAILED, JobState.RUNNING)

# 计入去重的状态：只有在途的算，done/dead 不算
_IN_FLIGHT = (JobState.PENDING, JobState.RUNNING)


class InMemoryJobQueue:
    __slots__ = ("_clock", "_rows", "_seq", "_lock")

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._rows: dict[int, Job] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def _next_id(self) -> int:
        self._seq += 1
        return self._seq

    async def enqueue(
        self,
        kind: str,
        bot_id: BotId,
        scope_key: str,
        *,
        payload: dict[str, Any] | None = None,
        delay_s: float = 0.0,
    ) -> Job | None:
        with self._lock:
            now = self._clock.now()
            for job in self._rows.values():
                if (
                    job.kind == kind
                    and job.bot_id == bot_id
                    and job.scope_key == scope_key
                    and job.state in _IN_FLIGHT
                ):
                    return None  # 去重命中：在途已有同键任务
            job = Job(
                id=JobId(self._next_id()),
                kind=kind,
                bot_id=bot_id,
                scope_key=scope_key,
                state=JobState.PENDING,
                attempts=0,
                visible_at=now + delay_s,
                payload=dict(payload or {}),
                created_at=now,
                updated_at=now,
            )
            self._rows[int(job.id)] = job
            return job

    async def claim(
        self,
        worker_id: str,
        kinds: Sequence[str],
        *,
        visibility_s: float,
        limit: int = 1,
    ) -> Sequence[Job]:
        now = self._clock.now()
        with self._lock:
            ready = [
                j
                for j in self._rows.values()
                if j.kind in kinds and j.state in _CLAIMABLE and j.visible_at <= now
            ]
            # 按 id 升序保证「先入队先处理」。优先级用 `delay_s` 表达，
            # 不在队列里再引入一个 priority 字段——两个维度会让排序规则无法解释。
            ready.sort(key=lambda j: int(j.id))
            picked: list[Job] = []
            for job in ready[: max(0, limit)]:
                claimed = job.claimed_by(worker_id, at=now, visibility_s=visibility_s)
                self._rows[int(job.id)] = claimed
                picked.append(claimed)
            return tuple(picked)

    async def heartbeat(self, job: Job, *, visibility_s: float) -> Job:
        with self._lock:
            current = self._rows.get(int(job.id))
            if current is None:
                raise Conflict(f"job {job.id} 不存在")
            now = self._clock.now()
            renewed = replace(current, visible_at=now + visibility_s, updated_at=now)
            self._rows[int(job.id)] = renewed
            return renewed

    async def complete(self, job: Job) -> None:
        with self._lock:
            current = self._rows.get(int(job.id))
            if current is None:
                raise Conflict(f"job {job.id} 不存在")
            if current.state == JobState.DONE:
                return  # 幂等：worker 崩溃前已完成、重试时再 ack 一次，不该报错
            self._rows[int(job.id)] = replace(
                current, state=JobState.DONE, locked_by=None, updated_at=self._clock.now()
            )

    async def fail(self, job: Job, error: str, *, max_attempts: int) -> Job:
        with self._lock:
            current = self._rows.get(int(job.id))
            if current is None:
                raise Conflict(f"job {job.id} 不存在")
            now = self._clock.now()
            nxt = current.failed(error, at=now, max_attempts=max_attempts)
            # 失败后立即可重试：把 visible_at 拉回当前，不在这里做退避。
            # 退避是 worker 的策略（按错误语义选 delay_s 重新入队）——队列实现
            # 不理解错误语义，让它做决策等于把「退避多久」藏进存储层。
            nxt = replace(nxt, visible_at=now)
            self._rows[int(job.id)] = nxt
            return nxt

    async def stats(self, bot_id: BotId | None = None) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for j in self._rows.values():
                if bot_id is not None and j.bot_id != bot_id:
                    continue
                out[j.state] = out.get(j.state, 0) + 1
            return out
