"""后台模型链的解析与校验。

这组用例针对的是「配置写错但没人报错」：
- 空配置应当退化成「跟随当前对话模型」，而不是空链（空链会让总结直接不干活）。
- `current` / `__current` / 空串是同一个意思，去重必须认得它们。
- 主模型同时出现在降级里，必须在保存时拦住；放进链里就是「降级到自己」，
  冷却表会把刚恢复的主模型立刻踢到队尾。
- 主应用里不存在的模型 id 要当场拒绝：写错一个字符，只会在半夜的总结里
  静默失败，并被记成「模型未加载」。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for p in ("contracts", "fleet"):
    sys.path.insert(0, str(_ROOT / "packages" / p / "src"))

from hermes_fleet import chain  # noqa: E402


class TestParse(unittest.TestCase):
    def test_empty_degrades_to_current(self) -> None:
        self.assertEqual(chain.chain_from_config("", ""), [chain.CURRENT])
        self.assertEqual(chain.chain_from_config(None, None), [chain.CURRENT])

    def test_single_legacy_field(self) -> None:
        self.assertEqual(chain.chain_from_config("", "zhipu/glm"), ["zhipu/glm"])
        # 链有值时，旧字段不该再掺进来
        self.assertEqual(
            chain.chain_from_config("a/1", "zhipu/glm"), ["a/1"])

    def test_chinese_comma_and_newlines(self) -> None:
        self.assertEqual(chain.parse("a/1，b/2\nc/3"), ["a/1", "b/2", "c/3"])

    def test_current_spellings_collapse(self) -> None:
        self.assertEqual(
            chain.parse("__current__, current, a/1"),
            [chain.CURRENT, "a/1"],
        )

    def test_duplicates_dropped_keeping_order(self) -> None:
        self.assertEqual(chain.parse("b/2,a/1,b/2"), ["b/2", "a/1"])

    def test_source(self) -> None:
        self.assertEqual(chain.source_of("a/1", ""), "explicit")
        self.assertEqual(chain.source_of("", "a/1"), "legacy")
        self.assertEqual(chain.source_of("", ""), "current")

    def test_describe(self) -> None:
        d = chain.describe(["a/1", "b/2"])
        self.assertEqual(d["primary"], "a/1")
        self.assertEqual(d["fallbacks"], ["b/2"])
        self.assertFalse(d["follows_current"])
        self.assertTrue(chain.describe([])["follows_current"])


class TestValidate(unittest.TestCase):
    available = {"a/1", "b/2", "c/3"}

    def test_ok(self) -> None:
        self.assertEqual(
            chain.validate("a/1", ["b/2", "c/3"], self.available),
            ["a/1", "b/2", "c/3"],
        )

    def test_current_is_always_allowed(self) -> None:
        self.assertEqual(
            chain.validate("", ["b/2"], self.available),
            [chain.CURRENT, "b/2"],
        )

    def test_unknown_id_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            chain.validate("a/1", ["a/2"], self.available)
        self.assertIn("a/2", str(ctx.exception))

    def test_duplicate_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chain.validate("a/1", ["a/1"], self.available)

    def test_too_many_fallbacks(self) -> None:
        many = ["b/2", "c/3", "a/1", "b/2", "c/3"]
        with self.assertRaises(ValueError):
            chain.validate("a/1", many, None)

    def test_no_catalog_skips_id_check(self) -> None:
        """拿不到主应用清单时只做格式校验——不能把人锁死在错误里。"""
        self.assertEqual(chain.validate("whatever/x", [], None), ["whatever/x"])

    def test_format_roundtrip(self) -> None:
        text = chain.format_chain("a/1", ["b/2", ""])
        self.assertEqual(chain.parse(text), ["a/1", "b/2", chain.CURRENT])
        text = chain.format_chain("", [])
        self.assertEqual(chain.parse(text), [chain.CURRENT])


if __name__ == "__main__":
    unittest.main()
