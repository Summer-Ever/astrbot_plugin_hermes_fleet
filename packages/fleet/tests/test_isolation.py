"""账号隔离与注入行为的验收测试。

这组用例的每一条都对应一次**真实事故**或一个必然踩的坑，
不是「覆盖率好看」：

- `test_cross_bot_isolation`：A 号的记忆出现在 B 号嘴里。100 个账号下
  这是产品事故，不是 bug。
- `test_group_does_not_leak_private`：群里把用户私聊说的事念出来。
- `test_resident_scope_is_separate`：常驻词条按 `private` / `group:<gid>`
  分开存，私聊设定不该在群里生效。
- `test_bot_filename_never_collides`：`qq-1` 和 `qq_1` 折叠成同一个库文件，
  两个号共用一个记忆——**静默**发生，最难发现。
- `test_capture_is_idempotent`：消息重投不该写两遍。
- `test_purge_removes_wal`：删号只删了 `.db`，留下 `-wal` 把数据带回来。

全部在本机跑，不需要 AstrBot，也不需要网络——这正是把业务逻辑
放在 `hermes_fleet` 而不是 `main.py` 的收益。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from hermes_fleet import BotRegistry, MemoryService, bot_file_stem


class FleetTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="fleet-")
        self.root = os.path.join(self.tmp, "bots")
        self.reg = BotRegistry(self.root, max_open=4)
        self.svc = MemoryService(self.reg)

    def tearDown(self) -> None:
        self.reg.close_all()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def private(self, bot: str, sender: str):
        return await self.svc.scene(bot, sender, "", is_group=False)

    async def group(self, bot: str, sender: str, gid: str):
        return await self.svc.scene(bot, sender, gid, is_group=True)


class TestCrossBotIsolation(FleetTestBase):
    async def test_cross_bot_isolation(self) -> None:
        """A 号的记忆在 B 号完全不可见——连检索都进不去。"""
        a = await self.private("qq-01", "user-1")
        await self.svc.save(a, "用户喜欢喝乌龙茶，尤其偏爱铁观音")

        hit_a = await self.svc.search(a, "乌龙茶")
        self.assertIn("乌龙茶", hit_a, "本号必须能召回自己存的记忆")

        b = await self.private("qq-02", "user-1")
        hit_b = await self.svc.search(b, "乌龙茶")
        self.assertEqual(hit_b.strip(), "", "另一个账号绝不能召回这条记忆")

    async def test_同一个openid在不同bot是不同的人(self) -> None:
        """openid 是 per-bot 的，但即使同一字符串，两个库也互不相通。"""
        a = await self.private("qq-01", "openid-abc")
        b = await self.private("qq-02", "openid-abc")
        await self.svc.save(a, "这个人住在杭州")
        self.assertEqual((await self.svc.search(b, "杭州")).strip(), "")

    async def test_bot_filename_never_collides(self) -> None:
        """两个视觉上相似的 id 不能映射到同一个库文件。"""
        a = bot_file_stem("qq-1")
        b = bot_file_stem("qq_1")
        c = bot_file_stem("qq 1")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(b, c)
        # 可读性：排障时要能从文件名看出这是哪个号
        self.assertTrue(a.startswith("qq-1"))

    async def test_同一个id永远映射到同一个文件(self) -> None:
        self.assertEqual(bot_file_stem("qq-01"), bot_file_stem("qq-01"))


class TestResidentScope(FleetTestBase):
    async def test_resident_scope_is_separate(self) -> None:
        """私聊的常驻设定不能出现在群聊的注入块里。"""
        p = await self.private("qq-01", "u1")
        await self.svc.set_resident(p, "address", "叫我哥哥")

        block_private = await self.svc.inject_block(p, "你好")
        self.assertIn("叫我哥哥", block_private)

        g = await self.group("qq-01", "u1", "g-100")
        block_group = await self.svc.inject_block(g, "大家好")
        self.assertNotIn("叫我哥哥", block_group,
                         "群聊是当众场合，私聊设定注入进来会被念出来")

    async def test_每个群各有一套常驻(self) -> None:
        g1 = await self.group("qq-01", "u1", "g-1")
        g2 = await self.group("qq-01", "u1", "g-2")
        await self.svc.set_resident(g1, "rules", "本群禁止发广告")
        self.assertIn("禁止发广告", await self.svc.inject_block(g1, "hi"))
        self.assertNotIn("禁止发广告", await self.svc.inject_block(g2, "hi"))

    async def test_空槽不注入(self) -> None:
        """注入块只列非空词条，注入成本才有界。"""
        p = await self.private("qq-01", "u1")
        await self.svc.set_resident(p, "address", "阿明")
        block = await self.svc.inject_block(p, "早")
        self.assertIn("称呼", block)
        self.assertNotIn("偏好与忌讳", block, "没设置的词条不该占位置")

    async def test_清空词条后不再注入(self) -> None:
        p = await self.private("qq-01", "u1")
        await self.svc.set_resident(p, "address", "阿明")
        await self.svc.set_resident(p, "address", "")
        self.assertNotIn("阿明", await self.svc.inject_block(p, "早"))

    async def test_注入块始终带维护约定(self) -> None:
        """常驻全空时也必须告诉模型该怎么维护——那正是最需要指引的时刻。"""
        p = await self.private("qq-01", "u1")
        block = await self.svc.inject_block(p, "早")
        self.assertIn("update_resident", block)
        self.assertIn("search_memory", block)


class TestVisibility(FleetTestBase):
    async def test_group_does_not_leak_private(self) -> None:
        """私聊存的记忆，在群里检索不到。"""
        p = await self.private("qq-01", "u1")
        await self.svc.save(p, "用户对花生过敏，点外卖要避开")
        g = await self.group("qq-01", "u1", "g-1")
        self.assertEqual((await self.svc.search(g, "花生")).strip(), "",
                         "群里把私聊内容念出来就是当众公开")

    async def test_group_memory_visible_in_same_group(self) -> None:
        g1 = await self.group("qq-01", "u1", "g-1")
        await self.svc.save(g1, "本群每周五交周报")
        self.assertIn("周报", await self.svc.search(g1, "周报"))

        g2 = await self.group("qq-01", "u1", "g-2")
        self.assertEqual((await self.svc.search(g2, "周报")).strip(), "",
                         "别的群不该看到本群的记忆")

    async def test_forget_是软删除(self) -> None:
        p = await self.private("qq-01", "u1")
        m = await self.svc.save(p, "用户养了两只猫")
        self.assertTrue(await self.svc.forget(p, int(m.id)))
        self.assertEqual((await self.svc.search(p, "两只猫")).strip(), "")
        # 行还在，只是被标记
        db = self.reg.get("qq-01")
        again = await db.memories.get(m.bot_id, m.id)
        self.assertIsNotNone(again)
        self.assertFalse(again.is_active, "必须是软删除，判错了还能回滚")

    async def test_不能忘掉别人的记忆(self) -> None:
        owner = await self.private("qq-01", "u1")
        m = await self.svc.save(owner, "用户的手机号是 13800000000")
        other = await self.private("qq-01", "u2")
        self.assertFalse(await self.svc.forget(other, int(m.id)),
                         "越权删除必须被拒绝，而不是静默成功")


class TestCapture(FleetTestBase):
    async def test_capture_is_idempotent(self) -> None:
        """同一轮重投不产生重复行（AstrBot 重试、插件热重载都会发生）。"""
        p = await self.private("qq-01", "u1")
        n1 = await self.svc.capture_turn(p, "s1", "你好", "你好呀")
        self.assertEqual(n1, 2)
        n2 = await self.svc.capture_turn(p, "s1", "你好", "你好呀")
        self.assertEqual(n2, 0, "重投必须被唯一索引挡住，返回 0")
        db = self.reg.get("qq-01")
        self.assertEqual(db.stats()["messages"], 2)

    async def test_空回复不写库(self) -> None:
        p = await self.private("qq-01", "u1")
        self.assertEqual(await self.svc.capture_turn(p, "s1", "你好", ""), 1)

    async def test_未总结计数(self) -> None:
        p = await self.private("qq-01", "u1")
        await self.svc.capture_turn(p, "s1", "a", "b")
        self.assertEqual(self.reg.get("qq-01").stats()["unsummarized"], 2)
        rows = await self.reg.get("qq-01").messages.fetch_unsummarized(
            p.bot_id, "s1", limit=10)
        self.assertEqual(await self.reg.get("qq-01").messages.mark_summarized(
            [r.id for r in rows]), 2)
        self.assertEqual(self.reg.get("qq-01").stats()["unsummarized"], 0)


class TestLifecycle(FleetTestBase):
    async def test_关库重开数据还在(self) -> None:
        p = await self.private("qq-01", "u1")
        await self.svc.save(p, "用户在做一个 AstrBot 记忆插件")
        self.reg.close_bot("qq-01")

        again = await self.private("qq-01", "u1")
        self.assertIn("AstrBot", await self.svc.search(again, "记忆插件"))

    async def test_lru_淘汰不影响数据(self) -> None:
        """max_open=4，连开 6 个号后最早的会被关掉，但数据不丢。"""
        for i in range(6):
            s = await self.private(f"qq-{i:02d}", "u1")
            await self.svc.save(s, f"第 {i} 号账号的记忆")
        self.assertLessEqual(len(self.reg.stats()["bots_opened"]), 4)

        s0 = await self.private("qq-00", "u1")
        self.assertIn("第 0 号", await self.svc.search(s0, "账号的记忆"))

    async def test_purge_removes_wal(self) -> None:
        p = await self.private("qq-01", "u1")
        await self.svc.save(p, "临时数据")
        path = self.reg.db_path("qq-01")
        self.assertTrue(os.path.exists(path))

        self.assertTrue(self.reg.purge("qq-01"))
        for suffix in ("", "-wal", "-shm"):
            self.assertFalse(os.path.exists(path + suffix),
                             f"${suffix} 残留会把数据带回来")

    def test_purge_未知账号不报错(self) -> None:
        self.assertFalse(self.reg.purge("never-existed"))


class TestRecallPoolGuard(FleetTestBase):
    async def test_recall_k_超过精排上限也不打断检索(self) -> None:
        """`recall_k` 没有硬上限，k=8 时候选池会超过 MAX_RERANK。

        以前这会让 rerank 抛 ValueError，整条检索链路直接失败；
        现在池子在取候选前就夹住，配置写大最多是少召回，不会报错。
        """
        svc = MemoryService(self.reg, recall_k=8)
        s = await svc.scene("qq-01", "u1", "", is_group=False)
        for i in range(60):
            await svc.save(s, f"第 {i} 条记忆：用户喜欢喝乌龙茶")
        # 这里只断言「不抛异常」：60 条同质记忆会让 idf 触底、软门控把结果
        # 全滤掉，返回空串是预期行为；回归点是以前会直接 ValueError。
        self.assertIsInstance(await svc.search(s, "乌龙茶"), str)


if __name__ == "__main__":
    unittest.main()
