"""SQLite 实现跑同一套一致性测试。

这是「SQLite 够用」这个结论的可执行证据：如果实现有语义偏差（跨 bot 泄露、
合并不可回滚、游标重复），这里会红。通过之后，把 `MemoryStore` 换成 Postgres
只需要再写一份同样只声明「我是哪种实现」的文件。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest

from hermes_contracts.clock import Clock, ManualClock
from hermes_contracts.conformance import MemoryStoreConformance
from hermes_contracts.conformance.fixtures import memory
from hermes_contracts.ids import MemoryId
from hermes_contracts.ports import MemoryStore
from hermes_storage_sqlite import SqliteMemoryStore


class TestSqliteMemoryStore(MemoryStoreConformance):
    async def make_store(self, clock: Clock) -> MemoryStore:
        d = tempfile.mkdtemp(prefix="hss_")
        self.addCleanup(shutil.rmtree, d, True)
        store = SqliteMemoryStore(f"{d}/mem.db", clock=clock)
        self.addCleanup(store.close)
        return store


class TestMergeTargetGuard(unittest.IsolatedAsyncioTestCase):
    """合并到不存在的目标必须拒绝。

    否则 sources 的 `merged_into` 会指向一个不存在的 id：这些记忆从此对
    所有查询不可见，管理台也翻不出来，等于静默丢数据。内存实现这里是
    抛 NotFound 的，SQLite 实现不能更宽松。
    """

    def setUp(self) -> None:
        self.clock = ManualClock()
        d = tempfile.mkdtemp(prefix="hss_merge_")
        self.addCleanup(shutil.rmtree, d, True)
        self.store = SqliteMemoryStore(f"{d}/mem.db", clock=self.clock)
        self.addCleanup(self.store.close)

    async def test_merge_into_missing_target_is_rejected(self) -> None:
        src = await self.store.add(memory(self.clock))
        with self.assertRaises(ValueError):
            await self.store.merge([src.id], MemoryId(999_999), at=self.clock.now())
        still = await self.store.get(src.bot_id, src.id)
        self.assertIsNotNone(still)
        self.assertTrue(still.is_active, "拒绝合并后源记忆必须仍然可见")


if __name__ == "__main__":
    unittest.main()
