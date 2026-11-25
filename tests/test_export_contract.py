"""「提取记忆」命令的源码契约。

和注入位置一样：主插件依赖 astrbot，本机跑不起来。能守的是那些
「删了也不会报错，但会悄悄变成隐私事故」的连接：

- 两个命令入口都接到同一个流程（行为不能分叉）；
- 文件组件必须带 url（OneBot 裸路径会报 retcode 1200）；
- 指定他人前必须过管理员检查；
- 取数必须经过 service 的已过滤结果，main.py 里不许出现 SQL。
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")
SCHEMA = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
SETTINGS = (ROOT / "packages" / "fleet" / "src" / "hermes_fleet" / "settings.py"
            ).read_text(encoding="utf-8")


class TestExportContract(unittest.TestCase):
    def test_两个入口共用一条流程(self) -> None:
        self.assertIn('@filter.command("提取记忆")', MAIN)
        self.assertIn('@fleet.command("提取记忆")', MAIN)
        self.assertEqual(MAIN.count("async for result in self._export_flow(event)"), 2)

    def test_默认目标是自己(self) -> None:
        self.assertIn("target = args[0] if args else str(scene.owner_id)", MAIN)

    def test_指定他人必须通过管理员检查(self) -> None:
        flow = MAIN.split("async def _export_flow", 1)[1]
        guard = flow.index("target != str(scene.owner_id)")
        call = flow.index("self.svc.export_owner(")
        self.assertLess(guard, call, "管理员检查必须在取数之前")

    def test_取数走已过滤的服务方法(self) -> None:
        self.assertIn("self.svc.export_owner(", MAIN)
        self.assertIn("render_document(", MAIN)
        # main.py 里一旦出现 SQL，就会绕过 pair_replies 直接写原文
        self.assertNotRegex(MAIN, re.compile(r"\bselect\b.*\bmessages\b", re.I))

    def test_渲染参数不能与数据字典重名(self) -> None:
        """这是实际踩过的 bug：data 里已有 bot_id/owner_id，
        调用处再显式传一遍会在运行时 TypeError，而本机没有 astrbot，
        命令路径跑不起来，只能靠源码契约挡。"""
        call = re.search(
            r"doc = render_document\((.*?)\n\s*\)", MAIN, re.S
        )
        self.assertIsNotNone(call)
        args = call.group(1)
        self.assertIn("**data", args)
        self.assertNotIn("bot_id=", args)
        self.assertNotIn("owner_id=", args)

    def test_文件组件带url与名字(self) -> None:
        self.assertIn("from astrbot.api.message_components import File", MAIN)
        self.assertIn('{"file": path, "name": name, "url": uri}', MAIN)
        self.assertIn("as_uri()", MAIN)

    def test_不支持文件时降级为文本而不是静默失败(self) -> None:
        self.assertIn("component is None", MAIN)
        self.assertIn("当前环境不支持发送文件组件", MAIN)

    def test_显式utf8写文件(self) -> None:
        self.assertIn('encoding="utf-8"', MAIN)

    def test_导出前清理过期文件(self) -> None:
        self.assertIn("stale_exports(", MAIN)
        self.assertIn("export_keep_days", MAIN)

    def test_开关走as_flag(self) -> None:
        self.assertIn('as_flag(self._get_cfg("export_enabled", True), True)', MAIN)


class TestConfig(unittest.TestCase):
    def test_三个配置项都在schema与设置页分组里(self) -> None:
        for key in ("export_enabled", "export_max_messages", "export_keep_days"):
            self.assertIn(key, SCHEMA)
            self.assertIn(f'"{key}"', SETTINGS)

    def test_默认值能直接用(self) -> None:
        self.assertIs(SCHEMA["export_enabled"]["default"], True)
        self.assertGreaterEqual(SCHEMA["export_max_messages"]["default"], 100)
        self.assertGreaterEqual(SCHEMA["export_keep_days"]["default"], 0.5)


if __name__ == "__main__":
    unittest.main()
