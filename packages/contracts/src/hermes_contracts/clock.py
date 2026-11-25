"""时钟：所有时间必须从外部注入，任何模块不准直接调 `time.time()`。

为什么这一条要放进契约层：
    hermes_core 的 `recall.py` 里直接 `time.time()` 做时间衰减，`db.py` 里
    直接 `time.time()` 写 `ts`，`main.py` 里直接 `time.time()` 算水位。后果是
    这些逻辑**没法被测试**——你无法断言「23 小时后这条记忆应该降权」，
    只能真的等 23 小时，所以实际上没人测。

    队列的可见性超时同理：要验证「worker 崩溃后 60 秒任务被重新领取」，
    必须能把时钟拨快。靠 `sleep(60)` 的测试最终都会被删掉。

这个文件同时提供三个实现，前两个生产用，第三个测试用：

- `Clock`：协议，只要 `now()`。
- `SystemClock`：生产默认。
- `ManualClock`：可手动推进，供所有模块的测试复用。
  **放在契约包里而不是各模块的测试目录**，是因为它被每个模块都要用，
  散着放就会长出五份行为不一致的实现。
"""

from __future__ import annotations

import threading
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """单调递增的秒级时间源。

    用 epoch 秒（float）而不是 datetime，理由：
    - `memories.ts` / `resident.updated_ts` / `jobs.visible_at` 在 DDL 里都是
      real / timestamptz，前者本来就要 float；
    - 时间衰减、TTL 比较全在秒域做，避免时区与 `timedelta` 转换；
    - 序列化进 JSON payload 时不需要额外编码。

    取整点的坑要注意：`float` 在 1e9 量级下精度约 0.1μs，够用；
    但**不要**用它做 id 或幂等键的组成部分（同毫秒内会撞）。
    """

    def now(self) -> float:  # pragma: no cover - 协议定义
        ...


class SystemClock:
    """生产实现的唯一入口。"""

    __slots__ = ()

    def now(self) -> float:
        return time.time()


class ManualClock:
    """手动时钟。线程安全，因为并发测试（`SKIP LOCKED`）会在多线程里拨表。"""

    __slots__ = ("_t", "_lock")

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        # 默认值取一个固定的过去时间，而不是 0：
        # `now() == 0` 会和「未设置」哨兵值混淆，DDL 里 `updated_ts default 0`
        # 就是靠 0 表示「从未更新」。测试数据一旦从 0 起，就分不清了。
        self._t = float(start)
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def advance(self, seconds: float) -> float:
        """向前拨。不接受负值——回拨会让「已过期」的任务重新可见，
        测试里出现这种操作，通常说明用例本身写错了。"""
        if seconds < 0:
            raise ValueError("ManualClock 只能前进")
        with self._lock:
            self._t += float(seconds)
            return self._t

    def set(self, epoch_s: float) -> None:
        with self._lock:
            self._t = float(epoch_s)


__all__ = ["Clock", "SystemClock", "ManualClock"]
