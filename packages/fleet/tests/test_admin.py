"""管理台读写语义的测试。

用的是**真实**的 `SqliteBotDb`（临时文件），不是替身：这一层最容易出的错
就是「SQL 条件写漏一个」，替身测不出来。代价是每个用例建一个库，可接受
（单库建表实测 <5ms）。

覆盖的都是会出真实事故的点：
- 管理视角能看全部记忆，但搜索不能越出作用域
- 编辑之后 owner 不能丢（hermes_core 出过这个事故）
- 删除必须是软删且可回滚
- 合并目标不能是墓碑，来源不能含目标自身
- 常驻六槽位恒在（区分「清空过」与「没设置」）
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for p in ("contracts", "storage_sqlite", "recall", "fleet"):
    sys.path.insert(0, str(_ROOT / "packages" / p / "src"))

from hermes_contracts import (  # noqa: E402
    BotId,
    Memory,
    MemoryId,
    MemoryScope,
    Visibility,
)
from hermes_fleet import admin  # noqa: E402
from hermes_storage_sqlite import SqliteBotDb  # noqa: E402

BOT = BotId("bot-a")
NOW = 1_800_000_000.0


class AdminBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.db = SqliteBotDb(str(Path(self._dir.name) / "bot-a.db"))
        self.addCleanup(self._dir.cleanup)
        self.addCleanup(self.db.close)

    async def put(self, content: str, *, owner: str = "u1", scope: MemoryScope = MemoryScope.USER,
                  group: str = "", visibility: Visibility = Visibility.PRIVATE,
                  kind: str = "fact") -> Memory:
        return await self.db.memories.add(Memory(
            id=MemoryId(0), bot_id=BOT, content=content, scope=scope,
            visibility=visibility, owner_id=owner, group_id=group, kind=kind,
            ts=NOW,
        ))


class TestScopes(AdminBase):
    async def test_groups_by_owner_and_counts_active_only(self) -> None:
        a = await self.put("a1", owner="u1")
        await self.put("a2", owner="u1")
        await self.put("b1", owner="u2")
        # 墓碑不算：作用域条数必须只数活跃的
        await self.db.memories.update(a.merged(MemoryId(int(a.id)), at=NOW + 1))

        rows = await admin.list_scopes(self.db, BOT)
        by_owner = {r["owner_id"]: r["count"] for r in rows}
        self.assertEqual(by_owner, {"u1": 1, "u2": 1})
        self.assertEqual(rows[0]["scope"], "user")

    async def test_removes_scope_that_became_empty(self) -> None:
        only = await self.put("唯一一条", owner="u9")
        await admin.forget_memories(self.db, BOT, [int(only.id)])
        self.assertEqual(await admin.list_scopes(self.db, BOT), [])


class TestPagingAndSearch(AdminBase):
    async def test_cursor_paging_does_not_skip_or_repeat(self) -> None:
        for i in range(5):
            await self.put(f"第 {i} 条", owner="u1")
        key = admin.key_of(BOT, scope="user", owner_id="u1")

        page1 = await admin.page_memories(self.db, key, limit=2)
        page2 = await admin.page_memories(self.db, key, limit=2, after_id=page1[-1]["id"])
        page3 = await admin.page_memories(self.db, key, limit=2, after_id=page2[-1]["id"])
        ids = [r["id"] for r in page1 + page2 + page3]
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(ids)), 5, "游标分页不能重复")

    async def test_search_stays_inside_the_scope(self) -> None:
        """搜索不能把别人的 owner 记忆捞出来——这是越权面。"""
        await self.put("花生过敏", owner="u1")
        await self.put("花生过敏", owner="u2")
        key = admin.key_of(BOT, scope="user", owner_id="u1")
        hits = await admin.search_memories(self.db, key, query="花生过敏")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["owner_id"], "u1")

    async def test_empty_query_falls_back_to_paging(self) -> None:
        await self.put("随便", owner="u1")
        key = admin.key_of(BOT, scope="user", owner_id="u1")
        self.assertEqual(len(await admin.search_memories(self.db, key, query="  ")), 1)


class TestEditDeleteMerge(AdminBase):
    async def test_edit_keeps_owner_and_visibility(self) -> None:
        """只改 content 却把 owner 清空——hermes_core 出过的真实事故。"""
        m = await self.put("旧内容", owner="u1", kind="fact")
        row = await admin.edit_memory(self.db, BOT, int(m.id), content="新内容")
        self.assertEqual(row["content"], "新内容")
        self.assertEqual(row["owner_id"], "u1")
        self.assertEqual(row["scope"], "user")
        self.assertEqual(row["visibility"], "private")

        again = await self.db.memories.get(BOT, MemoryId(int(m.id)))
        self.assertIsNotNone(again)
        self.assertEqual(again.owner_id, "u1")

    async def test_edit_rejects_empty_content(self) -> None:
        m = await self.put("有内容")
        with self.assertRaises(ValueError):
            await admin.edit_memory(self.db, BOT, int(m.id), content="   ")

    async def test_forget_is_soft_and_reversible(self) -> None:
        m = await self.put("要忘掉的")
        self.assertEqual(await admin.forget_memories(self.db, BOT, [int(m.id)]), 1)

        row = await self.db.memories.get(BOT, MemoryId(int(m.id)))
        self.assertIsNotNone(row, "软删不能物理消失，否则回滚不了")
        self.assertEqual(row.merged_into, int(m.id), "墓碑标记应指向自身")
        key = admin.key_of(BOT, scope="user", owner_id="u1")
        self.assertEqual(await admin.page_memories(self.db, key), [])

        # 回滚：把 merged_into 抹回 0（管理台没有这个入口，验证的是「数据可救」）
        await self.db.memories.update(replace(row, merged_into=0))
        restored = await admin.page_memories(self.db, key)
        self.assertEqual(len(restored), 1)

    async def test_forget_is_idempotent(self) -> None:
        m = await self.put("重复点删除")
        first = await admin.forget_memories(self.db, BOT, [int(m.id)])
        second = await admin.forget_memories(self.db, BOT, [int(m.id), 99999])
        self.assertEqual((first, second), (1, 0))

    async def test_merge_soft_deletes_sources(self) -> None:
        a = await self.put("喜欢喝美式")
        b = await self.put("不喝拿铁")
        target = await self.put("喝美式，不喝拿铁")
        n = await admin.merge_memories(self.db, BOT, [int(a.id), int(b.id)], int(target.id))
        self.assertEqual(n, 2)

        for mid in (int(a.id), int(b.id)):
            row = await self.db.memories.get(BOT, MemoryId(mid))
            self.assertEqual(row.merged_into, int(target.id), "来源应指向合并产物")
        kept = await self.db.memories.get(BOT, MemoryId(int(target.id)))
        self.assertTrue(kept.is_active)

    async def test_merge_excludes_target_from_sources(self) -> None:
        a = await self.put("一")
        b = await self.put("二")
        # 目标混在来源里：应当被自动剔除，剩 1 条来源
        n = await admin.merge_memories(self.db, BOT, [int(a.id), int(b.id)], int(b.id))
        self.assertEqual(n, 1)

    async def test_merge_rejects_tombstone_target(self) -> None:
        a = await self.put("一")
        b = await self.put("二")
        await admin.forget_memories(self.db, BOT, [int(b.id)])
        with self.assertRaises(ValueError):
            await admin.merge_memories(self.db, BOT, [int(a.id)], int(b.id))

    async def test_merge_requires_sources(self) -> None:
        target = await self.put("只有自己")
        with self.assertRaises(ValueError):
            await admin.merge_memories(self.db, BOT, [int(target.id)], int(target.id))


class TestResident(AdminBase):
    async def test_six_slots_always_present(self) -> None:
        """没设置过的槽位也要出现——否则界面上「清空过」和「没设置」长得一样。"""
        slots = await admin.resident_of(self.db, BOT, "u1", "private")
        self.assertEqual(len(slots), 6)
        self.assertEqual(slots["address"], "")

    async def test_save_then_clear(self) -> None:
        await admin.save_resident(self.db, BOT, "u1", "private", "address", "叫哥哥")
        slots = await admin.resident_of(self.db, BOT, "u1", "private")
        self.assertEqual(slots["address"], "叫哥哥")

        await admin.save_resident(self.db, BOT, "u1", "private", "address", "")
        slots = await admin.resident_of(self.db, BOT, "u1", "private")
        self.assertEqual(slots["address"], "")

    async def test_group_scope_is_isolated_from_private(self) -> None:
        await admin.save_resident(self.db, BOT, "u1", "private", "style", "简短")
        await admin.save_resident(self.db, BOT, "u1", "group:123", "style", "详细")
        self.assertEqual(
            (await admin.resident_of(self.db, BOT, "u1", "private"))["style"], "简短")
        self.assertEqual(
            (await admin.resident_of(self.db, BOT, "u1", "group:123"))["style"], "详细")

    async def test_rejects_unknown_block(self) -> None:
        with self.assertRaises(ValueError):
            await admin.save_resident(self.db, BOT, "u1", "private", "nickname", "x")


class TestMessages(AdminBase):
    async def test_groups_unsummarized_by_session(self) -> None:
        from hermes_contracts import Message, SessionId

        await self.db.messages.append_batch([
            Message(bot_id=BOT, session_id=SessionId("s1"), role="user",
                    content="早", ts=NOW, sender_id="u1"),
            Message(bot_id=BOT, session_id=SessionId("s1"), role="assistant",
                    content="早呀", ts=NOW + 0.1),
            Message(bot_id=BOT, session_id=SessionId("s2"), role="user",
                    content="在吗", ts=NOW + 1, sender_id="u2"),
        ])
        view = await admin.pending_messages(self.db, BOT)
        self.assertEqual(view["total"], 3)
        self.assertEqual(len(view["sessions"]), 2)
        first = view["sessions"][0]
        self.assertEqual(first["count"], 2)
        self.assertEqual(first["items"][0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
