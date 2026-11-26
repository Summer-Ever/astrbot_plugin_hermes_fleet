"""新指令 / 防重复注入 / 称呼过滤的契约与纯逻辑测试。

主插件依赖 astrbot，本机起不来，所以「命令有没有接上」只能靠源码契约守；
纯逻辑（注入块检测、帮助文案、称呼识别）则真跑函数，不靠读源码。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from hermes_fleet import INJECT_OPEN, has_inject_block, render_help
from hermes_workers import SUMMARY_PROMPT, is_address_memory, parse_memories

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")


class _Part:
    """模拟 AstrBot 的 TextPart：判断时只看 .text。"""

    def __init__(self, text: str) -> None:
        self.text = text


class TestHasInjectBlock(unittest.TestCase):
    def test_字符串里出现块首标记就算已注入(self) -> None:
        self.assertTrue(has_inject_block(["前置文本\n" + INJECT_OPEN + "\n正文"]))

    def test_content_part的text也算(self) -> None:
        self.assertTrue(has_inject_block([_Part(INJECT_OPEN), _Part("其它")]))

    def test_上下文字典也算(self) -> None:
        self.assertTrue(has_inject_block([
            {"role": "user", "content": INJECT_OPEN + "画像", "_no_save": True}
        ]))
        self.assertFalse(has_inject_block([{"role": "user", "content": "普通发言"}]))

    def test_没有标记不算(self) -> None:
        self.assertFalse(has_inject_block(["普通用户消息", _Part("另一段")]))
        self.assertFalse(has_inject_block(None))
        self.assertFalse(has_inject_block([]))


class TestHelp(unittest.TestCase):
    def test_新指令都在帮助里(self) -> None:
        text = render_help(admin=False)
        self.assertIn("/提取常驻记忆", text)
        self.assertIn("/hc帮助", text)
        self.assertIn("/提取记忆", text)
        self.assertIn("/导入记忆", text)

    def test_管理员指令只对管理员可见(self) -> None:
        plain = render_help(admin=False)
        full = render_help(admin=True)
        self.assertNotIn("/fleet 删除账号记忆", plain)
        self.assertIn("/fleet 删除账号记忆", full)
        self.assertIn("管理员指令", full)

    def test_帮助里的指令都真的注册了(self) -> None:
        """帮助里写了却没注册，比没写帮助更误导人。"""
        registered = {
            line.split('"')[1]
            for line in MAIN.splitlines()
            if '.command("' in line and 'def ' not in line
        }
        for name in ("提取常驻记忆", "hc帮助", "提取记忆", "导入记忆",
                     "常驻", "状态", "我是谁", "消息总结为记忆", "整理",
                     "账号列表", "删除账号记忆", "帮助"):
            self.assertIn(name, registered, f"帮助提到了 /{name}，但没有注册")


class TestCommandContract(unittest.TestCase):
    def test_总结指令改名为消息总结为记忆且保留旧名(self) -> None:
        """改名不能让习惯 `/fleet 总结` 的人突然失效。"""
        self.assertIn('@fleet.command("消息总结为记忆", alias={"总结"})', MAIN)

    def test_提取常驻记忆两个入口共用一条流程(self) -> None:
        self.assertIn('@filter.command("提取常驻记忆")', MAIN)
        self.assertIn('@fleet.command("提取常驻记忆")', MAIN)
        self.assertEqual(
            MAIN.count("async for result in self._resident_prompt_flow(event)"), 2
        )

    def test_hc帮助两个入口都调纯文案函数(self) -> None:
        self.assertIn('@filter.command("hc帮助")', MAIN)
        self.assertIn('@fleet.command("帮助")', MAIN)
        self.assertEqual(MAIN.count("render_help(admin=self._is_admin(event))"), 2)

    def test_导出的是实际注入块而非词条摘要(self) -> None:
        flow = MAIN.split("async def _resident_prompt_flow", 1)[1].split(
            "\n    @", 1)[0]
        self.assertIn("self.svc.inject_block(scene, \"\")", flow)
        self.assertIn("INJECT_OPEN", flow)
        self.assertNotIn("BLOCK_LABELS", flow)

    def test_注入前先查重(self) -> None:
        place = MAIN.split("def _place_block", 1)[1].split(
            "\n    # ---", 1)[0]
        check = place.index("has_inject_block(")
        user_part = place.index("extra_user_content_parts.append(")
        sys_part = place.index("append_system_prompt(")
        self.assertLess(check, user_part)
        self.assertLess(check, sys_part)


class TestInjectPlacement(unittest.TestCase):
    """常驻块的位置：必须在用户本轮发言之前。"""

    def test_主路径是往上下文插一条独立消息(self) -> None:
        place = MAIN.split("def _place_block", 1)[1].split("\n    # ---", 1)[0]
        insert = place.index("_insert_before_user_message(req, block)")
        fallback = place.index("extra_user_content_parts.append(")
        self.assertLess(insert, fallback, "插上下文必须优先于退回临时片段")

    def test_插入的消息带no_save且角色为user(self) -> None:
        fn = MAIN.split("def _insert_before_user_message", 1)[1].split(
            "\n    # ---", 1)[0]
        self.assertIn('"role": "user"', fn)
        self.assertIn('"_no_save": True', fn)
        self.assertIn("contexts.append(", fn)

    def test_宿主能力探测存在且fail_closed(self) -> None:
        fn = MAIN.split("def _host_supports_no_save", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("__private_attributes__", fn)
        self.assertIn("return False", fn)  # 探测不到就当不支持

    def test_查重范围包含上下文(self) -> None:
        place = MAIN.split("def _place_block", 1)[1].split("\n    # ---", 1)[0]
        self.assertIn('getattr(req, "contexts", None)', place)


class TestAddressFilter(unittest.TestCase):
    def test_明确称呼表达被识别(self) -> None:
        for text in (
            "用户希望被称为阿明",
            "请称呼他老李",
            "以后叫他阿明",
            "怎么称呼用户：小雅",
            "用户希望助手称呼他为哥哥",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_address_memory(text), text)

    def test_客观画像不误伤(self) -> None:
        for text in (
            "用户叫小明，在做后端开发",
            "用户在杭州工作",
            "偏好快速响应",
            "项目使用 SQLite 存储",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_address_memory(text), text)

    def test_解析时过滤称呼但保留其它记忆(self) -> None:
        raw = (
            '[{"kind":"profile","content":"用户希望被称为阿明","owner":"u1"},'
            '{"kind":"profile","content":"用户在杭州工作","owner":"u1"}]'
        )
        out = parse_memories(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["content"], "用户在杭州工作")

    def test_总结提示词明确禁止称呼(self) -> None:
        self.assertIn("称呼与叫法", SUMMARY_PROMPT)
        self.assertIn("不要输出常驻词条（尤其称呼）", SUMMARY_PROMPT)


if __name__ == "__main__":
    unittest.main()
