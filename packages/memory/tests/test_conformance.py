"""一致性套件在内存实现上的执行结果。

这个文件只有 5 行是「本模块的代码」，其余全部来自契约包。这是刻意设计的：

    多 agent 并行开发时，**测试代码本身也是冲突源**。如果 memory 和 api 各自
    写一套「记忆可见性」的测试，两边会慢慢长出不同的假设（一个测了跨 bot 隔离、
    一个没测），合并之后没人知道实际保证是什么。

    把用例集放进契约包，实现方只写一行「声明自己是哪种实现」，就自动继承了
    全部语义保证。新增一种实现（Postgres）也只是再写一行。
"""

from __future__ import annotations

import unittest

from hermes_contracts.clock import Clock
from hermes_contracts.conformance import MemoryStoreConformance
from hermes_contracts.ports import MemoryStore
from hermes_memory import InMemoryMemoryStore


class TestInMemoryMemoryStore(MemoryStoreConformance):
    async def make_store(self, clock: Clock) -> MemoryStore:
        return InMemoryMemoryStore(clock=clock)


if __name__ == "__main__":
    unittest.main()
