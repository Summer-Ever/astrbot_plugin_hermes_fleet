"""「管理后台导入记忆」入口的契约测试。

需求背景：聊天命令 `/导入记忆` 会被 LLM 当成对话内容吃掉，文件附件到不了
命令处理器。改为在单账号下钻页选文件导入，走 WebAPI，不经对话。

前端没有单测环境，能守的是结构性约束——每条都对应一次真实故障：

1. 后端必须注册 `memory/import` 路由，且复用 parse + import_owner 同一套
   校验（否则后台导入与聊天导入行为会漂移，一处能写一处写不进）；
2. 文件内容用 FileReader 读成文本再走 JSON 桥接——bridge.apiPost 只吃 JSON，
   直接传 File 对象会序列化失败；
3. 常驻是整段覆盖，导入前必须二次确认，且要把 100MB 上限挡在前面；
4. HTML 里必须有导入卡片的元素，否则 JS 取 id 拿到 null 直接抛错。
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEW_DIR = ROOT / "views" / "fleet"


class TestImportApiContract(unittest.TestCase):
    """后端：路由 + 复用同一条校验/写入链。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api = (ROOT / "webapi.py").read_text(encoding="utf-8")

    def test_注册了导入路由(self) -> None:
        self.assertIn('f"{self.prefix}/memory/import"', self.api)
        self.assertIn("self.memory_import", self.api)

    def test_复用了解析与服务导入(self) -> None:
        # 后台导入与聊天导入必须走同一条链，不能各写一套写入逻辑。
        self.assertIn("parse_memory_config(text)", self.api)
        self.assertIn("self.plugin.svc.import_owner(", self.api)

    def test_导入走导出开关(self) -> None:
        # bool("false") 是 True —— 必须经 settings.as_flag 判断开关。
        window = self.api[self.api.find("async def memory_import"):]
        window = window[:window.find("    async def ")]
        self.assertIn("as_flag", window)
        self.assertIn("export_enabled", window)

    def test_目标账号不存在时拒绝(self) -> None:
        window = self.api[self.api.find("async def memory_import"):]
        window = window[:window.find("    async def ")]
        self.assertIn("has_account(pid)", window)

    def test_文件大小上限与解析异常都处理(self) -> None:
        window = self.api[self.api.find("async def memory_import"):]
        window = window[:window.find("    async def ")]
        self.assertIn("MAX_EXPORT_BYTES", window)
        self.assertIn("except ValueError", window)


class TestImportUiContract(unittest.TestCase):
    """前端：HTML 元素 + JS 交互约束。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (VIEW_DIR / "index.html").read_text(encoding="utf-8")
        cls.js = (VIEW_DIR / "detail.js").read_text(encoding="utf-8")

    def test_操作页有导入卡片元素(self) -> None:
        for ident in ("import-file", "import-owner", "btn-import", "import-result"):
            self.assertIn(f'id="{ident}"', self.html, f"index.html 缺少 #{ident}")

    def test_走桥接的_json_接口(self) -> None:
        self.assertIn('apiPost("memory/import"', self.js)

    def test_文件用_filereader_读成文本(self) -> None:
        # bridge.apiPost 只传 JSON，File 对象序列化不了，必须读成字符串。
        self.assertIn("FileReader", self.js)
        self.assertIn("readAsText", self.js)
        self.assertIn("text: text", self.js)

    def test_导入前有二次确认(self) -> None:
        # 常驻按词条整段覆盖，不可撤销，必须 confirm。
        self.assertIn("confirm(", self.js)

    def test_前端挡住超大文件(self) -> None:
        self.assertIn("IMPORT_MAX_BYTES", self.js)
        self.assertIn("100MB", self.js)

    def test_切换账号时清空导入状态(self) -> None:
        # 不清空会把上一个账号选的文件导入到下一个账号。
        self.assertIn("st.importFile = null", self.js)
        self.assertIn('$("import-file").value = ""', self.js)


if __name__ == "__main__":
    unittest.main()
