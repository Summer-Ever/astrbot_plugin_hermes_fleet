"""向量召回的集成测试：**不联网、不装 numpy 也能跑**。

覆盖三条最容易悄悄坏掉的边界：

1. classic 档必须与「没有向量」时逐字一致（向量只能是增益）；
2. 向量服务故障时必须退回 classic，而不是把检索整条打断；
3. 索引器要对新增 / 改写 / 合并做增量，模型一变必须整体重建。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import zipfile
from dataclasses import replace

from hermes_contracts import RecallHit
from hermes_fleet import BotRegistry, MemoryService
from hermes_fleet.vector import VectorGlue, VectorSettings
from hermes_vectors import normalize


class FakeEmbedder:
    """三维假模型：猫/宠物、狗、爬山各占一轴，其余居中等分。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @staticmethod
    def vec(text: str) -> list[float]:
        if "猫" in text or "宠物" in text:
            return normalize([1.0, 0.0, 0.0])
        if "狗" in text:
            return normalize([0.0, 1.0, 0.0])
        if "爬山" in text:
            return normalize([0.0, 0.0, 1.0])
        return normalize([0.2, 0.2, 0.2])

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [self.vec(t) for t in texts]

    async def embed_one(self, text):
        return self.vec(text)


class FakeVector:
    """直接返回预设命中的语义召回，用于验证 service 的补齐与过滤。"""

    def __init__(self, hits=None, error: Exception | None = None) -> None:
        self.hits = list(hits or [])
        self.error = error
        self.calls = 0
        self.last_limit = 0

    async def semantic_hits(self, scene, query, *, exclude=None, limit=0):
        self.calls += 1
        self.last_limit = limit
        if self.error is not None:
            raise self.error
        return [h for h in self.hits if int(h.memory.id) not in (exclude or set())]


class Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="fleet-vector-")
        self.root = os.path.join(self.tmp, "bots")
        self.reg = BotRegistry(self.root, max_open=4)

    def tearDown(self) -> None:
        self.reg.close_all()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def scene(self, bot="qq-01", owner="u1", group="", is_group=False):
        return await MemoryService(self.reg).scene(
            bot, owner, group, is_group=is_group)


class TestSemanticFill(Base):
    async def test_classic_档完全不调向量(self) -> None:
        svc = MemoryService(self.reg)
        scene = await self.scene()
        memory = await svc.save(scene, "用户养了两只布偶猫")
        fake = FakeVector([RecallHit(memory=memory, score=0.9)])
        svc._vector_recall = fake

        text = await svc.search(scene, "宠物")
        self.assertEqual(fake.calls, 0)
        self.assertEqual(text.strip(), "")

    async def test_向量档在字面沉默时补齐(self) -> None:
        svc = MemoryService(self.reg, recall_mode="vector")
        scene = await self.scene()
        memory = await svc.save(scene, "用户养了两只布偶猫")
        svc._vector_recall = FakeVector([RecallHit(memory=memory, score=0.72)])

        text = await svc.search(scene, "宠物")
        self.assertIn("布偶", text, "字面零命中时应由语义路径补上")

    async def test_字面已经够数时不再调向量(self) -> None:
        svc = MemoryService(self.reg, recall_mode="vector")
        scene = await self.scene()
        memory = await svc.save(scene, "用户养了两只布偶猫")
        fake = FakeVector([RecallHit(memory=memory, score=0.9)])
        svc._vector_recall = fake

        # limit=1：字面第一条就够数，语义路径不该被触发。
        text = await svc.search(scene, "布偶猫", limit=1)
        self.assertIn("布偶", text)
        self.assertEqual(fake.calls, 0, "字面够用时不该再付一次向量查询")

    async def test_向量故障与_classic_输出一致(self) -> None:
        normal = MemoryService(self.reg)
        scene = await self.scene()
        await normal.save(scene, "用户养了两只布偶猫")
        expected = await normal.search(scene, "宠物")

        broken = MemoryService(self.reg, recall_mode="vector")
        broken._vector_recall = FakeVector(error=RuntimeError("boom"))
        self.assertEqual(await broken.search(scene, "宠物"), expected)

    async def test_补进来的条目仍要过可见性(self) -> None:
        svc = MemoryService(self.reg, recall_mode="vector")
        mine = await self.scene("qq-01", "u1")
        other = await self.scene("qq-01", "u2")
        theirs = await svc.save(other, "u2 养了两只布偶猫")
        svc._vector_recall = FakeVector([RecallHit(memory=theirs, score=0.99)])

        self.assertEqual((await svc.search(mine, "宠物")).strip(), "",
                         "语义召回不能绕过归属判断")

    async def test_按账号覆盖检索档(self) -> None:
        svc = MemoryService(
            self.reg,
            overrides_of=lambda pid: {"recall_mode": "vector"} if pid == "qq-01" else {},
        )
        a = await self.scene("qq-01")
        b = await self.scene("qq-02")
        memory = await svc.save(a, "用户养了两只布偶猫")
        await svc.save(b, "用户养了两只布偶猫")
        fake = FakeVector([RecallHit(memory=memory, score=0.8)])
        svc._vector_recall = fake

        self.assertIn("布偶", await svc.search(a, "宠物"))
        self.assertNotIn("布偶", await svc.search(b, "宠物"))
        self.assertEqual(fake.calls, 1, "只有开了 vector 的账号才会查向量")


class TestVectorGlue(Base):
    def settings(self, **kw) -> VectorSettings:
        base = dict(provider="api", base_url="http://127.0.0.1:9/v1",
                    model="fake-embed", min_score=0.3, sync_batch=64)
        base.update(kw)
        return VectorSettings(**base)

    async def seed(self, svc, scene, texts):
        out = []
        for text in texts:
            out.append(await svc.save(scene, text))
        return out

    async def test_增量索引_改写_合并(self) -> None:
        svc = MemoryService(self.reg)
        scene = await self.scene()
        first, second = await self.seed(svc, scene, [
            "用户养了两只布偶猫", "用户周末去爬山",
        ])
        glue = VectorGlue(self.reg, self.settings(), embedder=FakeEmbedder())

        res = await glue.sync_bot("qq-01")
        self.assertEqual(res.get("error"), None)
        self.assertEqual(res["indexed"], 2)

        status = await glue.status("qq-01")
        self.assertEqual(status["indexed"], 2)
        self.assertEqual(status["active"], 2)
        self.assertTrue(os.path.isfile(
            os.path.join(self.reg.bot_dir("qq-01"), "vectors.db")))

        # 改一条正文：content_hash 变了，只应重算这一条。
        fake = glue._embedder
        before = len(fake.calls)
        changed = replace(first, content="用户养了三只布偶猫",
                          updated_ts=first.ts + 10)
        await self.reg.get("qq-01").memories.update(changed)
        await glue.sync_bot("qq-01")
        self.assertGreater(len(fake.calls), before)
        self.assertEqual(sum(len(c) for c in fake.calls[before:]), 1)

        # 合并（软删）过后，活跃条数下降，索引也必须跟着删。
        db = self.reg.get("qq-01")
        await db.memories.merge([second.id], into=first.id, at=first.ts + 20)
        res = await glue.sync_bot("qq-01")
        self.assertEqual(res["indexed"], 1)

    async def test_模型变更会清空并重建(self) -> None:
        svc = MemoryService(self.reg)
        scene = await self.scene()
        await self.seed(svc, scene, ["用户养了两只布偶猫", "用户周末去爬山"])
        glue = VectorGlue(self.reg, self.settings(), embedder=FakeEmbedder())
        await glue.sync_bot("qq-01")

        glue.settings = self.settings(model="fake-embed-v2")
        res = await glue.sync_bot("qq-01")
        self.assertEqual(res["indexed"], 2, "换模型后必须重新索引全部活跃记忆")
        status = await glue.status("qq-01")
        self.assertEqual(status["model"], "fake-embed-v2")

    async def test_语义检索命中(self) -> None:
        svc = MemoryService(self.reg)
        scene = await self.scene()
        cat = await svc.save(scene, "用户养了两只布偶猫")
        await svc.save(scene, "用户周末去爬山")
        glue = VectorGlue(self.reg, self.settings(), embedder=FakeEmbedder())
        await glue.sync_bot("qq-01")

        hits = await glue.semantic_hits(scene, "宠物", limit=2)
        self.assertTrue(hits)
        self.assertEqual(int(hits[0].memory.id), int(cat.id))

    async def test_重建与清除(self) -> None:
        svc = MemoryService(self.reg)
        scene = await self.scene()
        await self.seed(svc, scene, ["用户养了两只布偶猫", "用户周末去爬山"])
        glue = VectorGlue(self.reg, self.settings(), embedder=FakeEmbedder())
        await glue.sync_bot("qq-01")

        cleared = await glue.clear("qq-01")
        self.assertTrue(cleared["cleared"])
        self.assertEqual((await glue.status("qq-01"))["indexed"], 0)

        rebuilt = await glue.rebuild("qq-01")
        self.assertTrue(rebuilt["enabled"])
        self.assertEqual(rebuilt["indexed"], 2)
        self.assertEqual(rebuilt["pending"], 0)

    async def test_未配置模型时不建库也不报错(self) -> None:
        svc = MemoryService(self.reg)
        scene = await self.scene()
        await svc.save(scene, "用户养了两只布偶猫")
        glue = VectorGlue(self.reg, VectorSettings(), embedder=FakeEmbedder())

        self.assertFalse(glue.enabled)
        self.assertEqual(await glue.semantic_hits(scene, "宠物", limit=2), [])
        status = await glue.status("qq-01")
        self.assertFalse(status["enabled"])
        self.assertFalse(os.path.exists(
            os.path.join(self.reg.bot_dir("qq-01"), "vectors.db")))
        rebuilt = await glue.rebuild("qq-01")
        self.assertFalse(rebuilt["enabled"])

    async def test_损坏的旁路库会被隔离并重建(self) -> None:
        """旁路库是派生数据：坏了改名留档，不能让状态页永远 500。"""
        svc = MemoryService(self.reg)
        scene = await self.scene()
        await svc.save(scene, "用户养了两只布偶猫")
        path = os.path.join(self.reg.bot_dir("qq-01"), "vectors.db")
        with open(path, "wb") as fp:
            fp.write(b"this is not a sqlite database")

        glue = VectorGlue(self.reg, self.settings(), embedder=FakeEmbedder())
        status = await glue.status("qq-01")
        self.assertEqual(status["indexed"], 0)
        self.assertTrue(os.path.exists(path + ".corrupt"))
        # 新库可用：重建后能正常索引。
        rebuilt = await glue.rebuild("qq-01")
        self.assertEqual(rebuilt["indexed"], 1)

    async def test_备份包不含向量库(self) -> None:
        from hermes_fleet.backup import backup_bots

        svc = MemoryService(self.reg)
        scene = await self.scene()
        await svc.save(scene, "用户养了两只布偶猫")
        glue = VectorGlue(self.reg, self.settings(), embedder=FakeEmbedder())
        await glue.sync_bot("qq-01")
        self.reg.close_all()

        out = os.path.join(self.tmp, "backups")
        os.makedirs(out, exist_ok=True)
        path = backup_bots(root=self.root, out_dir=out, version="test")["path"]
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        self.assertTrue(any(n.endswith("memory.db") for n in names))
        self.assertFalse(any("vectors.db" in n for n in names),
                         "向量是旁路数据，备份不该带上它")


if __name__ == "__main__":
    unittest.main()
