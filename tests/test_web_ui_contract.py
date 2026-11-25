"""插件页前端的**集成契约**测试。

这些断言看起来像在挑字眼，但每一条都对应一次「页面打开是一片空白 / 提示没连上
面板」的真实故障。前端没有单元测试环境，能守住的就是这几条结构性约束：

1. ``index.html`` 里只能有一个 body 闭合标签。
   AstrBot 的页面处理器是 ``html.replace("</body>", bridge_tag + "</body>", 1)``
   ——只替换**第一个**。有人在注释里顺手写一句「注入在 </body> 之前」，
   桥接脚本就会被塞进注释内部，永远不执行，于是页面报「接口不可用」。

2. ``app.js`` 不能把桥接对象缓存成顶层常量。
   桥接脚本被注入在 ``./app.js`` **之后**，脚本执行时 ``window.AstrBotPluginView``
   还不存在。缓存一次就是 null，之后所有请求都发不出去。

3. ``index.html`` 不能自己写死桥接脚本地址。
   ``/api/plugin/page/bridge-sdk.js`` 要 asset_token 才放行（query 校验），
   手写的裸路径会 401；面板注入时会自动带上 token。
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEW_DIR = ROOT / "views" / "fleet"


class TestFleetViewContract(unittest.TestCase):
    """守住三个真实踩过的坑。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (VIEW_DIR / "index.html").read_text(encoding="utf-8")
        cls.js = (VIEW_DIR / "app.js").read_text(encoding="utf-8")

    def test_entry_references_app_js(self) -> None:
        self.assertIn('src="./app.js"', self.html)

    def test_only_one_body_close_tag(self) -> None:
        """面板只替换第一个闭合标签，多一个就会把桥接脚本关进注释里。"""
        count = self.html.count("</body>")
        self.assertEqual(
            count,
            1,
            "index.html 里出现了 "
            f"{count} 个 </body>；面板只替换第一个，"
            "其余（哪怕在注释里）会让桥接脚本注入到错误位置",
        )

    def test_does_not_hardcode_bridge_url(self) -> None:
        """裸路径没有 asset_token，加载会被 401 拦掉。"""
        self.assertNotIn(
            'src="/api/plugin/page/bridge-sdk.js"',
            self.html,
            "桥接脚本地址要由面板注入（带 asset_token），不能手写裸路径",
        )

    def test_bridge_is_not_cached_as_constant(self) -> None:
        """桥接对象必须在调用时现取，不能冻结成 null。"""
        self.assertNotIn(
            "const bridge = window.AstrBotPluginView",
            self.js,
            "桥接脚本在 app.js 之后才执行；顶层缓存会拿到 null，页面将永远"
            "显示「接口不可用」",
        )
        self.assertIn("getBridge", self.js, "应当通过惰性读取拿桥接对象")

    def test_waits_for_bridge_before_failing(self) -> None:
        """桥接晚到时先等一会儿，别立刻判失败。"""
        self.assertIn("waitBridge", self.js)


if __name__ == "__main__":
    unittest.main()
