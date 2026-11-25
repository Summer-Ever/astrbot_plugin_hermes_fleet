"""设置页纯逻辑的测试。

重点是类型转换：`bool("false")` 是 True，配置页上看起来关掉了、行为却还是
开着——这类错误不会报任何错，只能靠用例挡。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "packages" / "fleet" / "src"))

from hermes_fleet.settings import (  # noqa: E402
    NEEDS_RELOAD,
    clean_values,
    schema_groups,
)

SCHEMA = {
    "memory_injection": {"type": "bool", "default": True, "description": "注入开关"},
    "max_open_bots": {"type": "int", "default": 24, "min": 1, "max": 128},
    "llm_provider_ids": {"type": "list", "default": []},
    "llm_provider_id": {"type": "string", "default": ""},
    "summarize_max_tokens": {"type": "int", "default": 900, "min": 100, "max": 4000},
    "unknown_thing": {"type": "string", "default": "x"},
}


class TestCleanValues(unittest.TestCase):
    def test_bool_string_false_is_false(self) -> None:
        """`bool("false")` 是 True，这里必须显式解析。"""
        cleaned, errors = clean_values(SCHEMA, {"memory_injection": "false"})
        self.assertEqual(errors, [])
        self.assertIs(cleaned["memory_injection"], False)

    def test_bool_accepts_yes_no(self) -> None:
        for raw, want in (("on", True), ("off", False), ("1", True), ("0", False),
                          (True, True), (False, False)):
            cleaned, errors = clean_values(SCHEMA, {"memory_injection": raw})
            self.assertEqual(errors, [], raw)
            self.assertIs(cleaned["memory_injection"], want, raw)

    def test_bool_rejects_garbage(self) -> None:
        _, errors = clean_values(SCHEMA, {"memory_injection": "maybe"})
        self.assertEqual(len(errors), 1)

    def test_int_out_of_range(self) -> None:
        _, errors = clean_values(SCHEMA, {"max_open_bots": 999})
        self.assertIn("不能大于 128", errors[0])

    def test_int_rejects_fraction(self) -> None:
        _, errors = clean_values(SCHEMA, {"max_open_bots": 2.5})
        self.assertIn("整数", errors[0])

    def test_int_accepts_numeric_string(self) -> None:
        cleaned, errors = clean_values(SCHEMA, {"max_open_bots": "32"})
        self.assertEqual(errors, [])
        self.assertEqual(cleaned["max_open_bots"], 32)
        self.assertIsInstance(cleaned["max_open_bots"], int)

    def test_list_from_comma_string(self) -> None:
        """降级链在配置页是多行文本，要能存成数组。"""
        cleaned, errors = clean_values(
            SCHEMA, {"llm_provider_ids": "阿里云/qwen3.8-flash, openrouter/nemotron"})
        self.assertEqual(errors, [])
        self.assertEqual(cleaned["llm_provider_ids"],
                         ["阿里云/qwen3.8-flash", "openrouter/nemotron"])

    def test_unknown_key_is_rejected(self) -> None:
        """静默忽略会让用户以为保存成功了，其实配置没变。"""
        _, errors = clean_values(SCHEMA, {"nope": 1})
        self.assertEqual(len(errors), 1)
        self.assertIn("未知配置项", errors[0])

    def test_errors_do_not_block_valid_keys(self) -> None:
        cleaned, errors = clean_values(
            SCHEMA, {"memory_injection": True, "max_open_bots": 999})
        self.assertEqual(cleaned, {"memory_injection": True})
        self.assertEqual(len(errors), 1)


class TestSchemaGroups(unittest.TestCase):
    def test_unlisted_key_falls_into_other(self) -> None:
        """新加的配置项忘了归组，也不能在页面上消失。"""
        groups = schema_groups(SCHEMA, {})
        titles = [g["title"] for g in groups]
        self.assertIn("其他", titles)
        other = next(g for g in groups if g["title"] == "其他")
        self.assertIn("unknown_thing", [i["key"] for i in other["items"]])

    def test_value_falls_back_to_default(self) -> None:
        groups = schema_groups(SCHEMA, {})
        item = next(i for g in groups for i in g["items"] if i["key"] == "max_open_bots")
        self.assertEqual(item["value"], 24)

    def test_needs_reload_flag(self) -> None:
        groups = schema_groups(SCHEMA, {})
        items = {i["key"]: i for g in groups for i in g["items"]}
        self.assertTrue(items["max_open_bots"]["needs_reload"])
        self.assertFalse(items["memory_injection"]["needs_reload"])
        self.assertIn("max_open_bots", NEEDS_RELOAD)


if __name__ == "__main__":
    unittest.main()
