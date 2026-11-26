"""压测脚本的契约测试：不联网，只保证它不会在改动中悄悄坏掉。"""

from __future__ import annotations

import asyncio
import importlib.util
import unittest
from pathlib import Path

from hermes_workers.prompt import parse_memories

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "benchmarks" / "bench_fleet_load.py"


def _load():
    spec = importlib.util.spec_from_file_location("fleet_bench_load_tool", PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCH = _load()


class TestBenchHarness(unittest.TestCase):
    def test_默认参数可解析且离线(self) -> None:
        args = BENCH.parse_args([])
        self.assertEqual(args.llm, "fake")
        self.assertEqual(args.embed, "fake")
        self.assertEqual(args.summarize_bots, 0)
        self.assertEqual(args.vector_bots, 0)
        self.assertGreater(args.think_ms, 0, "默认应有人类节奏，避免静默压满 CPU")

    def test_key_列表取第一个(self) -> None:
        self.assertEqual(BENCH._first_key(["sk-a", "sk-b"]), "sk-a")
        self.assertEqual(BENCH._first_key("sk-c"), "sk-c")
        self.assertEqual(BENCH._first_key([None, "", "sk-d"]), "sk-d")
        self.assertEqual(BENCH._first_key(None), "")

    def test_假模型输出能被真实解析器接受(self) -> None:
        import asyncio as aio

        class M:
            role = "user"
            content = "hi"

        result = aio.run(BENCH.FakeLLM(delay_ms=0).complete([M()], purpose="summarize"))
        entries = parse_memories(result.text)
        self.assertTrue(entries, "假模型必须产出可解析的记忆条目")
        self.assertTrue(entries[0]["content"])

    def test_假向量维度固定且归一化(self) -> None:
        emb = BENCH.FakeEmbedder(32)
        rows = asyncio.run(emb.embed(["用户喜欢乌龙茶", "用户对花生过敏"]))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 32)
        norm = sum(x * x for x in rows[0]) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
