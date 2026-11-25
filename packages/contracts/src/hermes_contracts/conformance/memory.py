"""`MemoryStore` 的一致性套件。

用法（任何实现都必须这样验一遍）：

    class TestMemoryStoreInMemory(MemoryStoreConformance):
        async def make_store(self, clock):
            return InMemoryMemoryStore(clock=clock)

    class TestMemoryStorePg(MemoryStoreConformance):
        async def make_store(self, clock):
            return await PgMemoryStore.connect(DSN, clock=clock)

这套用例只断言两件事：
1. **硬语义**：可见性、软删除、游标、级联删除。这些错了就是数据事故。
2. **弱语义**：明显该命中的检索要命中。这里刻意不绑死打分算法——
   "乌龙茶" 对 "用户喜欢喝乌龙茶" 无论是 trigram 还是向量都该召回，
   但具体排序分数不进断言。排序质量由 `hermes_recall` 自己的基准测。

基类本身不可执行（`make_store` 抛 SkipTest），子类 override 后才跑。
两个框架的行为都正确：pytest 靠 `__test__ = False` 跳过基类，
unittest 靠 SkipTest。
"""

from __future__ import annotations

import unittest

from ..clock import Clock, ManualClock
from ..ids import MemoryId, MemoryScope, OwnerId, Visibility
from ..models import MAX_MEMORY_CHARS, Memory, clip
from ..ports import MemoryStore
from .fixtures import ALICE, BOB, BOT, BOT_B, GROUP, key, memory


class MemoryStoreConformance(unittest.IsolatedAsyncioTestCase):
    """实现方只需 override `make_store`。"""

    __test__ = False  # pytest: 基类不收集

    async def make_store(self, clock: Clock) -> MemoryStore:
        raise unittest.SkipTest("abstract conformance base: override make_store()")

    async def asyncSetUp(self) -> None:
        self.clock = ManualClock()
        self.store = await self.make_store(self.clock)

    # -- add / get ---------------------------------------------------------

    async def test_add_returns_assigned_id_and_roundtrips(self) -> None:
        created = await self.store.add(memory(self.clock, content="用户住在杭州"))
        self.assertNotEqual(int(created.id), 0, "add 必须返回已分配 id 的副本")
        got = await self.store.get(BOT, created.id)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.content, "用户住在杭州")
        self.assertEqual(got.owner_id, ALICE)
        self.assertEqual(got.scope, MemoryScope.USER)

    async def test_add_clips_overlong_content(self) -> None:
        """超长内容必须截断而不是拒收，也不允许原样入库。"""
        long_text = "啊" * (MAX_MEMORY_CHARS + 200)
        created = await self.store.add(memory(self.clock, content=long_text))
        self.assertLessEqual(len(created.content), MAX_MEMORY_CHARS)
        got = await self.store.get(BOT, created.id)
        assert got is not None
        self.assertEqual(got.content, clip(long_text, MAX_MEMORY_CHARS))

    async def test_add_fills_ts_from_clock_when_zero(self) -> None:
        """`ts=0` 是「未指定」的哨兵，实现必须补 Clock 的时间。

        这条看着琐碎，但它是「时间可测」的地基：如果实现偷偷用 `time.time()`，
        这里断言 0 会被替换成 2026 年的真实时间，用例立刻红。
        """
        now = self.clock.now()
        created = await self.store.add(memory(self.clock, ts=0.0))
        self.assertAlmostEqual(created.ts, now, places=3)

    async def test_get_missing_returns_none(self) -> None:
        """查不到返回 None，不抛异常。批量场景里抛异常会逼调用方到处包 try。"""
        self.assertIsNone(await self.store.get(BOT, MemoryId(999_999)))

    # -- 合并 --------------------------------------------------------------

    async def test_merge_is_soft_and_reversible(self) -> None:
        a = await self.store.add(memory(self.clock, content="喜欢乌龙茶"))
        b = await self.store.add(memory(self.clock, content="爱喝乌龙茶"))
        affected = await self.store.merge([b.id], a.id, at=self.clock.now())
        self.assertEqual(affected, 1)

        merged = await self.store.get(BOT, b.id)
        assert merged is not None
        self.assertFalse(merged.is_active, "合并必须是软删除")
        self.assertEqual(merged.merged_into, int(a.id), "必须指向合并产物，否则回滚不了")

        # 回滚：把 merged_into 抹掉即可恢复
        await self.store.update(merged.merged(0, at=self.clock.now()))
        restored = await self.store.get(BOT, b.id)
        assert restored is not None
        self.assertTrue(restored.is_active, "解除合并后必须恢复可见")

    async def test_merge_rejects_self_reference(self) -> None:
        a = await self.store.add(memory(self.clock, content="只在杭州住过"))
        with self.assertRaises(Exception):
            await self.store.merge([a.id], a.id, at=self.clock.now())

    async def test_list_active_excludes_merged(self) -> None:
        keep = await self.store.add(memory(self.clock, content="保留"))
        gone = await self.store.add(memory(self.clock, content="被合并"))
        await self.store.merge([gone.id], keep.id, at=self.clock.now())
        rows = await self.store.list_active(key(), limit=50)
        ids = {int(m.id) for m in rows}
        self.assertIn(int(keep.id), ids)
        self.assertNotIn(int(gone.id), ids)

    # -- 可见性（最高优先级的回归） ----------------------------------------

    async def test_search_never_returns_other_users_private_memory(self) -> None:
        """私聊记忆不能被别人搜到。

        这是整个系统最不能错的一条。hermes_core 靠 `db.py` 里手拼的 SQL 条件
        保证，新实现只要漏一处 WHERE 就是隐私事故。
        """
        secret = await self.store.add(
            memory(self.clock, content="alice 的银行卡密码提示是生日", owner=ALICE)
        )
        rows = await self.store.search_candidates(
            key(owner=BOB), query="银行卡密码", viewer=BOB, viewer_group="", limit=20
        )
        self.assertNotIn(int(secret.id), {int(m.id) for m in rows})

    async def test_search_never_crosses_bot_boundary(self) -> None:
        """bot A 的记忆绝不能被 bot B 搜到。

        hermes_core 的 schema 里没有 bot 维度，这一条在它上面**无法通过**。
        新实现必须过。
        """
        mine = await self.store.add(memory(self.clock, content="alice 在 bot_alpha 说过喜欢猫"))
        rows = await self.store.search_candidates(
            key(bot=BOT_B), query="喜欢猫", viewer=ALICE, viewer_group="", limit=20
        )
        self.assertNotIn(int(mine.id), {int(m.id) for m in rows})

    async def test_search_returns_group_memory_only_to_same_group(self) -> None:
        g = await self.store.add(
            memory(
                self.clock,
                content="这个群约定每周五发周报",
                scope=MemoryScope.GROUP,
                visibility=Visibility.GROUP,
                group_id=GROUP,
            )
        )
        inside = await self.store.search_candidates(
            key(scope=MemoryScope.GROUP, owner="", group_id=GROUP),
            query="周报",
            viewer=BOB,
            viewer_group=GROUP,
            limit=20,
        )
        self.assertIn(int(g.id), {int(m.id) for m in inside}, "本群成员应读得到群记忆")

        other = await self.store.search_candidates(
            key(scope=MemoryScope.GROUP, owner="", group_id="group_999"),
            query="周报",
            viewer=BOB,
            viewer_group="group_999",
            limit=20,
        )
        self.assertNotIn(int(g.id), {int(m.id) for m in other}, "群记忆不能跨群泄露")

    async def test_search_excludes_merged(self) -> None:
        keep = await self.store.add(memory(self.clock, content="用户喜欢喝乌龙茶"))
        dup = await self.store.add(memory(self.clock, content="用户爱喝乌龙茶"))
        await self.store.merge([dup.id], keep.id, at=self.clock.now())
        rows = await self.store.search_candidates(
            key(), query="乌龙茶", viewer=ALICE, viewer_group="", limit=20
        )
        ids = {int(m.id) for m in rows}
        self.assertIn(int(keep.id), ids)
        self.assertNotIn(int(dup.id), ids)

    async def test_search_finds_obvious_match(self) -> None:
        m = await self.store.add(memory(self.clock, content="用户喜欢喝乌龙茶"))
        rows = await self.store.search_candidates(
            key(), query="乌龙茶", viewer=ALICE, viewer_group="", limit=20
        )
        self.assertIn(
            int(m.id),
            {int(r.id) for r in rows},
            "字面重叠的查询必须召回——否则模型永远拿不到记忆",
        )

    async def test_search_respects_limit(self) -> None:
        for i in range(12):
            await self.store.add(memory(self.clock, content=f"用户提到过第{i}种茶的偏好"))
        rows = await self.store.search_candidates(
            key(), query="茶", viewer=ALICE, viewer_group="", limit=5
        )
        self.assertLessEqual(len(rows), 5, "search_candidates 必须在库内完成 top-k")

    # -- 计数 / 游标 / 级联删除 -------------------------------------------

    async def test_count_active_ignores_merged_and_other_owners(self) -> None:
        a = await self.store.add(memory(self.clock, owner=ALICE))
        b = await self.store.add(memory(self.clock, owner=ALICE))
        await self.store.add(memory(self.clock, owner=BOB))
        await self.store.merge([b.id], a.id, at=self.clock.now())
        self.assertEqual(await self.store.count_active(BOT, owner_id=ALICE), 1)
        self.assertEqual(await self.store.count_active(BOT, owner_id=BOB), 1)
        self.assertEqual(await self.store.count_active(BOT), 2)

    async def test_list_scopes_groups_by_owner_and_group(self) -> None:
        """整理任务的入口：必须能自己数出「哪些作用域值得整理」。

        少了这个方法，调用方只能靠「本次启动后碰过的会话」来枚举作用域，
        重启后昨天攒的记忆就永远等不到整理——而且不会报错。
        """
        for i in range(3):
            await self.store.add(memory(self.clock, content=f"alice 私聊 {i}", owner=ALICE))
        await self.store.add(memory(self.clock, content="bob 私聊", owner=BOB))
        for i in range(2):
            await self.store.add(memory(
                self.clock, content=f"群里第 {i} 条", owner="",
                scope=MemoryScope.GROUP, visibility=Visibility.GROUP, group_id=GROUP,
            ))
        # 另一个 bot 的库不能被数进来
        await self.store.add(memory(self.clock, content="beta 的记忆", bot=BOT_B))

        scopes = await self.store.list_scopes(BOT)
        got = {(str(k.scope), k.owner_id, k.group_id): n for k, n in scopes}
        self.assertEqual(got.get(("user", str(ALICE), "")), 3)
        self.assertEqual(got.get(("user", str(BOB), "")), 1)
        self.assertEqual(got.get(("group", "", GROUP)), 2)
        self.assertFalse(
            any(k.owner_id == str(ALICE) and n == 4 for k, n in scopes),
            "跨 bot 的同一 owner 不能合并计数",
        )
        self.assertEqual(
            [n for _, n in scopes], sorted((n for _, n in scopes), reverse=True),
            "按条数降序返回，调用方才能先处理最该整理的",
        )

    async def test_list_scopes_honours_min_active(self) -> None:
        await self.store.add(memory(self.clock, content="bob 第1条", owner=BOB))
        await self.store.add(memory(self.clock, content="bob 第2条", owner=BOB))
        await self.store.add(memory(self.clock, content="alice 第1条", owner=ALICE))

        few = await self.store.list_scopes(BOT, min_active=2)
        self.assertEqual([n for _, n in few], [2], "低于阈值的应被存储层过滤掉")
        self.assertEqual(str(few[0][0].owner_id), str(BOB))

    async def test_list_scopes_ignores_merged(self) -> None:
        a = await self.store.add(memory(self.clock, content="保留"))
        b = await self.store.add(memory(self.clock, content="会被并掉"))
        await self.store.merge([b.id], a.id, at=self.clock.now())
        scopes = await self.store.list_scopes(BOT)
        self.assertEqual([n for _, n in scopes], [1], "已合并的不能算进活跃条数")

    async def test_list_active_cursor_has_no_duplicates(self) -> None:
        for i in range(6):
            await self.store.add(memory(self.clock, content=f"第{i}条"))
        first = await self.store.list_active(key(), limit=3)
        self.assertEqual(len(first), 3)
        second = await self.store.list_active(key(), limit=3, after_id=int(first[-1].id))
        self.assertFalse(
            {int(m.id) for m in first} & {int(m.id) for m in second},
            "游标分页不允许重叠（offset 分页在并发写入时会漏行，所以用 id 游标）",
        )

    async def test_purge_owner_removes_all_traces_of_that_owner_only(self) -> None:
        """删号必须清干净，且不能误伤别人。多租户合规的硬要求。"""
        for i in range(3):
            await self.store.add(memory(self.clock, content=f"alice 的第{i}条", owner=ALICE))
        await self.store.add(memory(self.clock, content="bob 的记忆", owner=BOB))

        deleted = await self.store.purge_owner(BOT, OwnerId(ALICE))
        self.assertGreaterEqual(deleted, 3, "返回删除条数，用于审计")
        self.assertEqual(await self.store.count_active(BOT, owner_id=ALICE), 0)
        self.assertEqual(await self.store.count_active(BOT, owner_id=BOB), 1)

    # -- 跨 bot 隔离（写侧） ----------------------------------------------

    async def test_same_owner_in_two_bots_are_independent(self) -> None:
        a = await self.store.add(memory(self.clock, content="alpha 里的秘密", bot=BOT))
        b = await self.store.add(memory(self.clock, content="beta 里的秘密", bot=BOT_B))
        got_a = await self.store.get(BOT, a.id)
        assert got_a is not None
        self.assertEqual(got_a.content, "alpha 里的秘密")
        self.assertNotEqual(int(a.id), int(b.id))
