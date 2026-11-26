"""`hermes_vectors` 的自测：纯数学、旁路库、embedding 响应解析。

全部零网络、零第三方依赖（numpy 有就一起覆盖，没有就跳过对应断言）。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
import unittest

from hermes_vectors import (
    BatchingEmbedder,
    EmbeddingClient,
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingQueueFull,
    EmbeddingTimeout,
    SqliteVectorStore,
    VectorRow,
    content_hash,
    cosine,
    dim_of,
    normalize,
    pack,
    parse_embeddings,
    top_k,
    unpack,
)


class TestCore(unittest.TestCase):
    def test_normalize_单位向量(self) -> None:
        vec = normalize([3.0, 4.0])
        self.assertAlmostEqual(vec[0], 0.6)
        self.assertAlmostEqual(vec[1], 0.8)
        self.assertAlmostEqual(sum(x * x for x in vec), 1.0)

    def test_全零向量不除零(self) -> None:
        self.assertEqual(normalize([0.0, 0.0]), [0.0, 0.0])

    def test_pack_unpack_往返(self) -> None:
        vec = [0.25, -0.5, 1.0 / 3.0]
        blob = pack(vec)
        self.assertEqual(dim_of(blob), 3)
        back = unpack(blob)
        for a, b in zip(vec, back):
            self.assertAlmostEqual(a, float(b), places=6)

    def test_残缺的_blob_不越界(self) -> None:
        """文件被截断时返回可读部分，而不是让 struct 抛到检索路径上。"""
        blob = pack([1.0, 0.0, 0.0])[:-2]
        self.assertEqual(len(unpack(blob)), 2)

    def test_content_hash_稳定性(self) -> None:
        self.assertEqual(content_hash("用户喜欢乌龙茶"), content_hash("用户喜欢乌龙茶"))
        self.assertNotEqual(content_hash("用户喜欢乌龙茶"), content_hash("用户喜欢普洱"))

    def test_cosine_对未归一化输入也正确(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 0.0], [10.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(cosine([1.0], [1.0, 2.0]), 0.0, "维度不同按 0 处理")

    def test_top_k_排序与截断(self) -> None:
        rows = [
            (1, pack(normalize([1.0, 0.0])), 2),
            (2, pack(normalize([0.0, 1.0])), 2),
            (3, pack(normalize([0.8, 0.2])), 2),
        ]
        hits = top_k([1.0, 0.0], rows, k=2)
        self.assertEqual([i for i, _ in hits], [1, 3])
        self.assertAlmostEqual(hits[0][1], 1.0, places=5)

    def test_top_k_跳过维度不符的旧向量(self) -> None:
        """换过模型后旧向量还在库里时，不能被算出假分数。"""
        rows = [
            (1, pack(normalize([1.0, 0.0, 0.0])), 3),
            (2, pack(normalize([1.0, 0.0])), 2),
        ]
        hits = top_k([1.0, 0.0], rows, k=5)
        self.assertEqual([i for i, _ in hits], [2])

    def test_min_score_过滤(self) -> None:
        rows = [(1, pack(normalize([1.0, 0.0])), 2),
                (2, pack(normalize([0.0, 1.0])), 2)]
        self.assertEqual([i for i, _ in top_k([1.0, 0.0], rows, k=5, min_score=0.5)], [1])


class TestStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="vectors-")
        self.store = SqliteVectorStore(os.path.join(self.tmp, "vectors.db"))

    def tearDown(self) -> None:
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _row(self, memory_id: int, text: str, vec, *, owner="u1", group="",
             visibility="private", scope="user", ts=0.0) -> VectorRow:
        return VectorRow(
            memory_id=memory_id, owner_id=owner, group_id=group, scope=scope,
            visibility=visibility, content_hash=content_hash(text),
            updated_ts=ts, dim=len(vec), vec=pack(normalize(vec)),
        )

    def test_元数据与计数(self) -> None:
        self.store.set_meta(model="m1", dim="2")
        self.assertEqual(self.store.meta()["model"], "m1")
        self.assertEqual(self.store.count(), 0)
        self.store.upsert([self._row(1, "a", [1.0, 0.0])])
        self.store.upsert([self._row(1, "b", [1.0, 0.0])])
        self.assertEqual(self.store.count(), 1, "主键相同必须覆盖，不是追加")
        self.assertEqual(list(self.store.hashes()), [1])
        self.assertEqual(self.store.delete([1]), 1)
        self.assertEqual(self.store.count(), 0)

    def test_clear_保留模型但重置同步状态(self) -> None:
        self.store.set_meta(model="m1", dim="2", fingerprint="f", indexed="9")
        self.store.upsert([self._row(1, "a", [1.0, 0.0])])
        self.store.clear()
        meta = self.store.meta()
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(meta["model"], "m1")
        self.assertEqual(meta["fingerprint"], "")
        self.assertEqual(meta["indexed"], "0")

    def test_search_按可见性预过滤(self) -> None:
        self.store.upsert([
            self._row(1, "u1 私聊", [1.0, 0.0], owner="u1"),
            self._row(2, "u1 的群记忆", [1.0, 0.0], owner="u1", group="g1",
                      visibility="group"),
            self._row(3, "u2 的群记忆", [1.0, 0.0], owner="u2", group="g1",
                      visibility="group"),
            self._row(4, "公开", [1.0, 0.0], owner="", visibility="public",
                      scope="global"),
            self._row(5, "u2 私聊", [1.0, 0.0], owner="u2"),
        ])
        ids = [i for i, _ in self.store.search([1.0, 0.0], viewer="u2",
                                               viewer_group="g1", limit=10)]
        # 2 是 u1 在本群的「群可见」记忆：预过滤必须放它进来，是否真的共享
        # 由 service 层的 group_memory_shared 开关决定；1 是 u1 的私聊记忆，
        # 在 SQL 这一层就不该出现。
        self.assertEqual(sorted(ids), [2, 3, 4, 5],
                         "不能把 u1 的私聊记忆预筛进来")
        self.assertNotIn(1, ids)
        ids = [i for i, _ in self.store.search([1.0, 0.0], viewer="u2",
                                               viewer_group="", limit=10)]
        # 私聊上下文：自己的群记忆（3）仍属于本人，可见；u1 的群记忆（2）
        # 与 u1 的私聊记忆（1）都不该出现。
        self.assertEqual(sorted(ids), [3, 4, 5])

    def test_search_scan_cap_按最近更新截断(self) -> None:
        self.store.upsert([
            self._row(1, "旧", [1.0, 0.0], ts=1.0),
            self._row(2, "新", [1.0, 0.0], ts=2.0),
        ])
        ids = [i for i, _ in self.store.search([1.0, 0.0], viewer="u1",
                                               limit=5, scan_cap=1)]
        self.assertEqual(ids, [2])

    def test_search_维度不符直接返回空(self) -> None:
        self.store.upsert([self._row(1, "a", [1.0, 0.0])])
        self.assertEqual(self.store.search([1.0, 0.0, 0.0], viewer="u1"), [])

    def test_search_min_score(self) -> None:
        self.store.upsert([
            self._row(1, "同向", [1.0, 0.0]),
            self._row(2, "正交", [0.0, 1.0]),
        ])
        hits = self.store.search([1.0, 0.0], viewer="u1", limit=5, min_score=0.5)
        self.assertEqual([i for i, _ in hits], [1])


class TestEmbeddingParse(unittest.TestCase):
    def test_正常响应并归一化(self) -> None:
        out = parse_embeddings(
            {"data": [{"index": 0, "embedding": [3.0, 4.0]}]}, expected=1
        )
        self.assertAlmostEqual(sum(x * x for x in out[0]), 1.0)

    def test_index_乱序按序号还原(self) -> None:
        out = parse_embeddings({"data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]}, expected=2)
        self.assertAlmostEqual(out[0][0], 1.0)

    def test_条数不符报错(self) -> None:
        with self.assertRaises(EmbeddingError):
            parse_embeddings({"data": [{"embedding": [1.0]}]}, expected=2)

    def test_维度不一致报错(self) -> None:
        with self.assertRaises(EmbeddingError):
            parse_embeddings({"data": [{"embedding": [1.0, 0.0]},
                                       {"embedding": [1.0]}]}, expected=2)

    def test_缺字段报错(self) -> None:
        for payload in ({}, {"data": []}, {"data": [{}]},
                        {"data": [{"embedding": "x"}]}):
            with self.subTest(payload=payload):
                with self.assertRaises(EmbeddingError):
                    parse_embeddings(payload, expected=1)


class TestEmbeddingClient(unittest.IsolatedAsyncioTestCase):
    async def test_按批大小分片且顺序不变(self) -> None:
        client = EmbeddingClient(EmbeddingConfig(
            base_url="https://example.invalid/v1", model="m", batch_size=2))
        calls: list[list[str]] = []

        def fake(chunk):
            calls.append(list(chunk))
            return [[1.0, 0.0] for _ in chunk]

        client._embed_sync = fake  # type: ignore[method-assign]
        rows = await client.embed(["a", "b", "c"])
        self.assertEqual(calls, [["a", "b"], ["c"]])
        self.assertEqual(len(rows), 3)

    async def test_未配置模型时报错(self) -> None:
        client = EmbeddingClient(EmbeddingConfig(base_url="", model=""))
        with self.assertRaises(EmbeddingError):
            await client.embed(["a"])

    def test_endpoint_不重复拼_embeddings(self) -> None:
        self.assertEqual(
            EmbeddingConfig(base_url="https://x/v1/").endpoint,
            "https://x/v1/embeddings",
        )
        self.assertEqual(
            EmbeddingConfig(base_url="https://x/v1/embeddings").endpoint,
            "https://x/v1/embeddings",
        )
        self.assertFalse(EmbeddingConfig(base_url="https://x/v1").enabled)


class _RecordingClient:
    """记录每次收到的批量，可选注入延迟 / 故障。"""

    def __init__(self, *, delay: float = 0.0,
                 error: Exception | None = None) -> None:
        self.delay = delay
        self.error = error
        self.batches: list[list[str]] = []

    async def embed(self, texts):
        self.batches.append(list(texts))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return [normalize([float(len(t) or 1), 1.0, 0.0]) for t in texts]

    async def embed_one(self, text):
        return (await self.embed([text]))[0]


class TestBatchingEmbedder(unittest.IsolatedAsyncioTestCase):
    """排队合批的三条边界：合批、有界、有预算。"""

    async def test_并发查询合并成一批(self) -> None:
        client = _RecordingClient()
        emb = BatchingEmbedder(client, batch_size=8, window_s=0.05)
        texts = [f"查询{i}" for i in range(6)]
        rows = await asyncio.gather(*(emb.embed_one(t) for t in texts))
        await emb.aclose()
        self.assertEqual(len(client.batches), 1, "同一瞬间的检索应该并成一批")
        self.assertEqual(sorted(client.batches[0]), sorted(texts))
        self.assertEqual([len(r) for r in rows], [3] * 6)

    async def test_同批重复文本只算一次(self) -> None:
        client = _RecordingClient()
        emb = BatchingEmbedder(client, batch_size=8, window_s=0.05)
        rows = await asyncio.gather(*(emb.embed_one("同一句") for _ in range(3)),
                                    emb.embed_one("另一句"))
        await emb.aclose()
        self.assertEqual(sorted(client.batches[0]), ["另一句", "同一句"])
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], rows[2], "重复查询应广播同一份向量")

    async def test_队列满时立刻失败(self) -> None:
        client = _RecordingClient(delay=0.2)
        emb = BatchingEmbedder(client, batch_size=1, max_pending=1, workers=1)
        tasks = [asyncio.create_task(emb.embed_one(f"q{i}")) for i in range(4)]
        await asyncio.sleep(0.01)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await emb.aclose()
        self.assertTrue(any(isinstance(r, EmbeddingQueueFull) for r in results),
                        "队列满必须立刻拒绝，而不是无限积压")

    async def test_超预算立刻降级(self) -> None:
        client = _RecordingClient(delay=0.3)
        emb = BatchingEmbedder(client, batch_size=1, window_s=0.0)
        t0 = time.perf_counter()
        with self.assertRaises(EmbeddingTimeout):
            await emb.embed_one("慢查询", budget_s=0.02)
        elapsed = time.perf_counter() - t0
        await emb.aclose()
        self.assertLess(elapsed, 0.25, "超预算要立刻返回，不能等模型算完")

    async def test_底层失败传给所有等待者(self) -> None:
        client = _RecordingClient(error=EmbeddingError("boom"))
        emb = BatchingEmbedder(client, batch_size=4, window_s=0.05)
        results = await asyncio.gather(*(emb.embed_one(f"q{i}") for i in range(3)),
                                       return_exceptions=True)
        await emb.aclose()
        self.assertTrue(all(isinstance(r, EmbeddingError) for r in results))

    async def test_批量接口直通不排队(self) -> None:
        client = _RecordingClient()
        emb = BatchingEmbedder(client, batch_size=8)
        rows = await emb.embed(["a", "b"])
        self.assertEqual(client.batches, [["a", "b"]])
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
