"""SQLite 实现跑同一套一致性测试。

这是「SQLite 够用」这个结论的可执行证据：如果实现有语义偏差（跨 bot 泄露、
合并不可回滚、游标重复），这里会红。通过之后，把 `MemoryStore` 换成 Postgres
只需要再写一份同样只声明「我是哪种实现」的文件。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest

from hermes_contracts.clock import Clock
from hermes_contracts.conformance import MemoryStoreConformance
from hermes_contracts.ports import MemoryStore
from hermes_storage_sqlite import SqliteMemoryStore


class TestSqliteMemoryStore(MemoryStoreConformance):
    async def make_store(self, clock: Clock) -> MemoryStore:
        d = tempfile.mkdtemp(prefix="hss_")
        self.addCleanup(shutil.rmtree, d, True)
        store = SqliteMemoryStore(f"{d}/mem.db", clock=clock)
        self.addCleanup(store.close)
        return store


if __name__ == "__main__":
    unittest.main()
