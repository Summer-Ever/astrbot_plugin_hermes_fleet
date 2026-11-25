"""`JobQueue` 一致性套件在内存实现上的执行。

和 memory 的对应文件一样，这里只有一行是「本模块的代码」。
Postgres 实现接进来时，也只多一行。
"""

from __future__ import annotations

import unittest

from hermes_contracts.clock import Clock
from hermes_contracts.conformance import JobQueueConformance
from hermes_contracts.ports import JobQueue
from hermes_jobs import InMemoryJobQueue


class TestInMemoryJobQueue(JobQueueConformance):
    async def make_queue(self, clock: Clock) -> JobQueue:
        return InMemoryJobQueue(clock=clock)


if __name__ == "__main__":
    unittest.main()
