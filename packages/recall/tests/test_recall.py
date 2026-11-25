"""`hermes_recall` 的自测。从 hermes_core 搬过来的模块，这里补上它原本缺的测试。

原项目里 `recall.py` 是最值得复用的模块，但它自己的测试是零散的、
且大量依赖 `time.time()` 的真实值。搬过来之后有两条改善：

1. `now` 全部显式传入，时间衰减可以断言；
2. `rerank` 多了一道候选上限闸门，这件事本身也要测。

这个文件同时把「bigram 检索的天花板」固化成可执行的证据——
它是新平台必须上向量检索的直接理由，不该只写在 README 的一段话里。
"""

from __future__ import annotations

import unittest

from hermes_contracts import BotId, Memory, MemoryId, MemoryScope, RecallHit
from hermes_contracts.conformance.fixtures import ALICE, BOT
from hermes_recall import MAX_RERANK, rerank, render, safe_join
from hermes_recall.recall import score

NOW = 1_700_000_000.0
YEAR = 365 * 86400


def _mem(mid: int, content: str, ts: float = NOW) -> Memory:
    return Memory(
        id=MemoryId(mid),
        bot_id=BOT,
        content=content,
        scope=MemoryScope.USER,
        owner_id=ALICE,
        ts=ts,
    )


class TestTimeDecay(unittest.TestCase):
    def test_older_memory_scores_lower(self) -> None:
        """同一句话，一年前的分数必须明显低于现在的。

        `score` 的时间衰减参数在 hermes_core 里是 `now or time.time()`，
        **默认值能跑，但无法被测**。这里显式传 now，断言才成立。
        """
        doc = "用户喜欢喝乌龙茶"
        fresh = score("乌龙茶", doc, {}, ts=NOW, now=NOW)
        old = score("乌龙茶", doc, {}, ts=NOW - YEAR, now=NOW)
        self.assertGreater(fresh, old, "旧记忆不该压住新的")
        self.assertGreater(old, 0, "衰减不等于清零：旧记忆仍应可召回")

    def test_future_ts_does_not_break_scoring(self) -> None:
        """时钟偏移（消息 ts 在未来）不能产生负 age 或负数分。"""
        val = score("乌龙茶", "用户喜欢喝乌龙茶", {}, ts=NOW + YEAR, now=NOW)
        self.assertGreaterEqual(val, 0)


class TestRerank(unittest.TestCase):
    def test_empty_candidates(self) -> None:
        self.assertEqual(rerank("乌龙茶", [], k=3, now=NOW), [])

    def test_rejects_oversized_pool(self) -> None:
        """候选超上限要报错，不是默默硬扛。

        这是防「全库捞进 Python」病根的最后一道闸。存储层如果哪天改回
        一次捞 3000 条，本地就会红，而不是上线后 CPU 打满才发现。
        """
        pool = [_mem(i, f"第{i}条关于乌龙茶的记忆") for i in range(1, MAX_RERANK + 2)]
        with self.assertRaises(ValueError):
            rerank("乌龙茶", pool, k=3, now=NOW)

    def test_returns_hits_with_ids_preserved(self) -> None:
        hits = rerank("乌龙茶", [_mem(7, "用户喜欢喝乌龙茶")], k=3, now=NOW)
        self.assertTrue(hits)
        self.assertEqual(int(hits[0].memory.id), 7, "重排不能丢掉原实体引用")
        self.assertGreater(hits[0].score, 0)

    def test_short_query_yields_nothing(self) -> None:
        """单字查询不召回：`MIN_QUERY_CHARS=2` 是刻意的，
        「在」「嗯」这类查询召回一堆无关记忆只会污染 prompt。"""
        self.assertEqual(rerank("茶", [_mem(1, "用户喜欢喝乌龙茶")], k=3, now=NOW), [])


class TestBigramCeiling(unittest.TestCase):
    """把检索天花板写成证据，而不是注释。

    这些用例断言的是**当前能力的边界**。等 M4 接上向量召回后，
    `test_pure_semantic_query_cannot_be_recalled` 会变红——那时应该改写它，
    而不是删掉：这条红本身就是「能力升级完成了」的信号。
    """

    def test_contiguous_overlap_is_recalled(self) -> None:
        self.assertGreater(score("乌龙茶", "用户喜欢喝乌龙茶", {}, ts=NOW, now=NOW), 0)

    def test_pure_semantic_query_cannot_be_recalled(self) -> None:
        """「喝茶」召不回「喜欢喝乌龙茶」——没有连续二字重叠。

        这就是 hermes_core README 里承认的天花板：用户问「我喝茶的偏好吗」，
        助手的记忆里明明有「喜欢喝乌龙茶」，却召回不到。换向量检索后此条应改写。
        """
        self.assertEqual(
            score("喝茶", "用户喜欢喝乌龙茶", {}, ts=NOW, now=NOW),
            0.0,
            "如果这条不再为 0，说明检索能力升级了，请更新本用例的意图",
        )


class TestRender(unittest.TestCase):
    def test_ids_survive_rendering(self) -> None:
        """注入文本必须带编号：模型靠它调 replaces= / forget(id=)。

        编号在行尾而不是行首——放行首会被 markdown 当标题吞掉，
        hermes_core 的截图里 `#127` 就变成了 `127`。
        """
        hits = [RecallHit(memory=_mem(127, "用户喜欢喝乌龙茶"), score=1.0)]
        text = render(hits)
        self.assertIn("127", text)
        self.assertIn("乌龙茶", text)

    def test_empty_renders_empty(self) -> None:
        self.assertEqual(render([]), "")

    def test_max_lines_truncates(self) -> None:
        hits = [RecallHit(memory=_mem(i, f"第{i}条关于乌龙的记忆"), score=1.0) for i in range(1, 30)]
        text = render(hits, max_lines=2)
        self.assertNotIn("第20条", text)


class TestSafeJoin(unittest.TestCase):
    """`paths.py` 也是整份搬过来的复用件，它防的是路径穿越。"""

    def test_allows_normal_name(self) -> None:
        self.assertTrue(str(safe_join("/tmp/base", "a.txt")).endswith("a.txt"))

    def test_blocks_escape(self) -> None:
        for bad in ("../etc/passwd", "..\\..\\win.ini", "/abs/path"):
            with self.assertRaises(Exception, msg=f"{bad} 必须被拒绝"):
                safe_join("/tmp/base", bad)


if __name__ == "__main__":
    unittest.main()
