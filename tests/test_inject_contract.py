"""注入位置的**源码契约**测试。

`main.py` 依赖 astrbot，本地环境装不了，所以 `on_llm_request` 的分支无法
直接跑用例。能守的是「分支还在、还接到纯函数上」：

1. 开关 `inject_to_system_prompt` 真的被读、真的走到 `append_system_prompt`。
   这一条防的是重构时把系统提示词那条路悄悄删掉——默认值关着，删了也
   不会有人发现，直到有人打开开关发现没反应。
2. 插件里任何开关都不许写 `bool(self._get_cfg(...))`。`bool("false")` 是
   True，配置页上关掉了、行为还开着，且不报错。
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")


class TestInjectionPlacement(unittest.TestCase):
    def test_开关被读取(self) -> None:
        self.assertIn("inject_to_system_prompt", MAIN)

    def test_系统提示词那条路接到纯函数(self) -> None:
        """接的不是自己手写的字符串拼接，那样幂等与空行处理会各写一遍。"""
        self.assertIn("append_system_prompt(", MAIN)

    def test_用户提示词那条路仍在(self) -> None:
        """默认路径不能被开关顺手删掉。"""
        self.assertIn("extra_user_content_parts", MAIN)

    def test_放置逻辑集中在一处(self) -> None:
        """两个分支都在 `_place_block` 里，钩子函数只负责取块和去重。"""
        self.assertIn("def _place_block(", MAIN)
        hook = MAIN.split("async def on_llm_request", 1)[1]
        hook = hook.split("def _place_block", 1)[0]
        self.assertIn("self._place_block(req, block)", hook)
        self.assertNotIn("extra_user_content_parts", hook)


class TestNoUnsafeBoolCfg(unittest.TestCase):
    """`bool("false")` is True —— 开关必须走 settings.as_flag。"""

    PATTERN = re.compile(r"bool\(\s*self\._get_cfg\(")

    def test_没有直接_bool_包配置值(self) -> None:
        bad = [m.group(0) for m in self.PATTERN.finditer(MAIN)]
        self.assertEqual(bad, [], "配置值不能用 bool() 包：字符串 \"false\" 会变成 True")

    def test_布尔开关都走_as_flag(self) -> None:
        for key in ("memory_injection", "capture_messages", "summarize_enabled",
                    "consolidate_enabled", "group_private_memory",
                    "inject_to_system_prompt"):
            idx = MAIN.find(f'"{key}"')
            self.assertGreater(idx, 0, f"找不到配置项 {key}")
            window = MAIN[max(0, idx - 260):idx]
            self.assertIn("as_flag", window, f"{key} 没经过 as_flag 读取")


if __name__ == "__main__":
    unittest.main()
