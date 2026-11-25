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
from hermes_recall.recall import _query_chars, bm25_scores, rank_fused, score

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
    """把 A 路（bigram）的检索天花板写成证据，而不是注释。

    **v0.3.0 起这条只描述 A 路的能力，不再等于系统上限。**
    `rank_fused` 的默认档换成混合粒度 BM25 之后，「喝茶」已经能召回
    「喜欢喝乌龙茶」（靠单字「茶」），见下面的 TestMixedGranularity。
    这里的用例保留，因为它仍然精确描述了 `score()` 这个函数的边界——
    而 `score()` 仍在给历史消息召回用。

    以后再接向量检索时，这两组用例要一起重新审视：A 路的天花板不会变，
    变的是有没有别的路把它补上。
    """

    def test_contiguous_overlap_is_recalled(self) -> None:
        self.assertGreater(score("乌龙茶", "用户喜欢喝乌龙茶", {}, ts=NOW, now=NOW), 0)

    def test_pure_semantic_query_cannot_be_recalled(self) -> None:
        """「喝茶」召不回「喜欢喝乌龙茶」——A 路没有连续二字重叠。

        这就是 hermes_core README 里承认的天花板。注意它**只对 A 路成立**：
        B 路（`bm25_scores`）默认开启混合粒度后能召回这条，见
        TestMixedGranularity.test_single_char_bridges_vocabulary_gap。
        """
        self.assertEqual(
            score("喝茶", "用户喜欢喝乌龙茶", {}, ts=NOW, now=NOW),
            0.0,
            "如果这条不再为 0，说明 A 路自己升级了，请更新本用例的意图",
        )


class TestMixedGranularity(unittest.TestCase):
    """v0.3.0 的能力升级：混合粒度 BM25 突破 bigram 的「严格相邻」限制。

    这三条对应用户真实提问里最高频的三种失败，全部来自 evals/ 的失败样本，
    不是构造出来的假数据。
    """

    def test_single_char_bridges_vocabulary_gap(self) -> None:
        """「喝茶」→「喜欢喝乌龙茶」：bigram 零交集，单字「茶」架桥。"""
        docs = [{"id": 1, "content": "用户喜欢喝乌龙茶", "ts": NOW}]
        self.assertGreater(bm25_scores("喝茶", docs)[0], 0,
                           "混合粒度必须让这条能被召回")

    def test_word_order_shift_no_longer_kills(self) -> None:
        """「我家猫叫什么」→「名字叫团子」：`猫叫` 与 `名字叫` 差一个字的
        位置，这是 bigram 最典型的失效方式，单字「猫」「叫」能救回来。"""
        docs = [{"id": 1, "content": "用户养了一只橘猫，名字叫团子", "ts": NOW}]
        self.assertGreater(bm25_scores("我家猫叫什么", docs)[0], 0)

    def test_latin_letters_are_not_split(self) -> None:
        """`python` 不能被拆成 p/y/t/h/o/n 当单字 term。

        否则中文记忆里随便出现一个 `p` 就能拿到高分——这些字母在中文语料里
        df 极低、idf 极高，比命中整个 `python` 还值钱。
        """
        docs = [{"id": 1, "content": "用户在配置 OpenRouter 的密钥", "ts": NOW}]
        self.assertEqual(bm25_scores("python", docs)[0], 0.0,
                         "拉丁词必须整词匹配，不能靠单字母蒙中")

    def test_stop_chars_filtered_from_unigrams(self) -> None:
        """虚词单字必须过滤：它们几乎每条都有，累加起来会淹掉真实信号。

        只测单字层（`_query_chars`）。二元组层不过滤虚词是有意的——
        「我的」这类组合偶尔有实义，而双字的 df 高、idf 低，噪声自己会被压住；
        单字不一样，它数量多、单个 idf 又低，靠 idf 压不住，必须直接过滤。
        """
        self.assertEqual(_query_chars("我的了吗在是有"), [])
        self.assertEqual(set(_query_chars("咖啡团子")), {"咖", "啡", "团", "子"})


class TestSoftGate(unittest.TestCase):
    """软闸门：用 BM25 绝对分判断该不该注入，而不是覆盖率。

    覆盖率闸门的分母是**问句长度**，同一个问题多打几个字就会被判「不相关」。
    实测这是 v0.2.x 最大的漏召回来源：32.9% 该答的题一条都注不进去。
    """

    def _docs(self, rows: list[str]) -> list[dict]:
        return [{"id": i, "content": t, "ts": NOW} for i, t in enumerate(rows, 1)]

    def test_relevant_query_is_injected(self) -> None:
        docs = self._docs(["用户喝咖啡只喝美式，不加糖",
                           "用户养了一只橘猫叫团子",
                           "用户在做跨境电商"])
        out = rank_fused("我对咖啡有什么讲究", docs, k=3, now=NOW)
        self.assertEqual([int(r["id"]) for r in out], [1],
                         "相关记忆必须能注入，且不该带上无关的")

    def test_long_query_does_not_kill_recall(self) -> None:
        """同一个问题后面多打六个字，召回结果不能变。

        这是覆盖率闸门最典型的误杀：分母从 8 个实词涨到 10 个，
        一条明明相关的短记忆直接被砍。
        """
        docs = self._docs(["用户喝咖啡只喝美式，不加糖",
                           "用户养了一只橘猫叫团子"])
        short = rank_fused("我喝咖啡有什么讲究", docs, k=3, now=NOW)
        long = rank_fused("我喝咖啡有什么讲究，为什么这么问", docs, k=3, now=NOW)
        self.assertEqual([int(r["id"]) for r in short],
                         [int(r["id"]) for r in long])

    def test_unrelated_query_injects_nothing(self) -> None:
        """库里没有答案时不能硬塞——否则模型会顺着无关记忆编故事。"""
        docs = self._docs(["用户在做跨境电商，主营家居小件",
                           "用户有长期失眠，需要白噪音"])
        out = rank_fused("我妹妹今年多大", docs, k=3, now=NOW)
        self.assertEqual(out, [], "无关查询必须返回空，而不是凑几条")

    def test_gate_off_returns_candidates(self) -> None:
        """gate=False 用于诊断，必须能看到被闸门挡下的候选。"""
        docs = self._docs(["用户在做跨境电商，主营家居小件"])
        self.assertEqual(rank_fused("我妹妹今年多大", docs, k=3, now=NOW, gate=False),
                         [])
        self.assertTrue(rank_fused("跨境电商 家居", docs, k=3, now=NOW, gate=False))


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
