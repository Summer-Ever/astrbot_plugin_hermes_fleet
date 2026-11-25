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
    as_flag,
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


class TestAsFlag(unittest.TestCase):
    """运行时读开关不能直接 `bool(值)`：`bool("false")` 是 True。

    这不是假想问题——手改 `cmd_config.json` 或从别处拷配置就会留下字符串。
    一旦读成 True，用户在面板上关掉了功能，行为依旧开着，且没有任何报错。
    """

    def test_字符串_false_读成_False(self) -> None:
        self.assertIs(as_flag("false", True), False)
        self.assertIs(as_flag(" False ", True), False)
        self.assertIs(as_flag("0", True), False)
        self.assertIs(as_flag("off", True), False)

    def test_字符串_true_读成_True(self) -> None:
        self.assertIs(as_flag("true", False), True)
        self.assertIs(as_flag("1", False), True)
        self.assertIs(as_flag("yes", False), True)

    def test_真布尔原样返回(self) -> None:
        self.assertIs(as_flag(True, False), True)
        self.assertIs(as_flag(False, True), False)

    def test_读不懂用默认值而不是猜(self) -> None:
        """配置写坏了宁可回到默认值，也不能静默当成 True。"""
        self.assertIs(as_flag(None, False), False)
        self.assertIs(as_flag("随便写的", False), False)
        self.assertIs(as_flag({"a": 1}, True), True)

    def test_空串当未配置(self) -> None:
        self.assertIs(as_flag("", False), False)


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


class TestRealSchemaIsGrouped(unittest.TestCase):
    """真 schema 与分组表必须对得上。

    加配置项时忘了写进 `GROUPS`，设置页会把它塞到「其他」——它还在，但
    用户要翻到最下面才看得见。`inject_to_system_prompt` 这种影响行为的开关
    藏在那里，等于没做。
    """

    @classmethod
    def setUpClass(cls) -> None:
        import json

        path = _ROOT / "_conf_schema.json"
        cls.schema = json.loads(path.read_text(encoding="utf-8"))

    def test_每个配置项都有归属分组(self) -> None:
        from hermes_fleet.settings import GROUPS

        listed = {k for _, keys in GROUPS for k in keys}
        missing = [k for k in self.schema if k not in listed]
        self.assertEqual(missing, [], f"这些配置项没进 GROUPS：{missing}")

    def test_注入位置开关存在且默认关(self) -> None:
        meta = self.schema.get("inject_to_system_prompt")
        self.assertIsNotNone(meta, "缺少「注入到系统提示词」开关")
        self.assertEqual(meta["type"], "bool")
        self.assertIs(meta["default"], False, "默认必须仍走用户提示词")

    def test_注入位置开关不需要重启(self) -> None:
        """每轮请求现读配置，改完立即生效；标成需重启会误导用户。"""
        self.assertNotIn("inject_to_system_prompt", NEEDS_RELOAD)

    def test_README_里的配置项数目与实际一致(self) -> None:
        """这个数字曾经写着 16 而实际是 21——没人会去数，所以让用例数。"""
        import re

        readme = (_ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r"共\s*(\d+)\s*项", readme)
        self.assertIsNotNone(m, "README 里应有「共 N 项」的配置项计数")
        self.assertEqual(
            int(m.group(1)),
            len(self.schema),
            f"README 写 {m.group(1)} 项，_conf_schema.json 实际 {len(self.schema)} 项",
        )


if __name__ == "__main__":
    unittest.main()
