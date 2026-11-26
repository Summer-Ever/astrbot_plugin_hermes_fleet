"""「打开库目录 / 备份记忆」两个侧栏运维入口的契约测试。

每条对应一个真实故障：
1. 浏览器不能直接开本地文件夹，必须走后端 open-root；后端路径不能由客户端提供。
2. 备份必须走一致性快照（backup API），不能裸拷 .db（WAL 丢数据）。
3. 前端按钮与结果区元素必须存在，否则 getElementById 拿到 null。
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEW_DIR = ROOT / "views" / "fleet"


class TestOpenRootContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api = (ROOT / "webapi.py").read_text(encoding="utf-8")

    def test_注册了打开目录接口(self) -> None:
        self.assertIn('f"{self.prefix}/open-root"', self.api)
        self.assertIn("self.open_root", self.api)

    def test_打开本机文件管理器(self) -> None:
        window = self.api[self.api.find("async def open_root"):]
        window = window[:window.find("    async def ", 10)]
        self.assertIn("os.startfile", window)
        self.assertIn("xdg-open", window)
        self.assertIn("open", window)

    def test_路径不接受客户端任意输入(self) -> None:
        # which 只能是 root/backups 两个枚举，其余拒绝；不能出现把 body
        # 里的字符串直接拼进路径的写法。
        window = self.api[self.api.find("async def open_root"):]
        window = window[:window.find("    async def ", 10)]
        self.assertIn('which not in ("root", "backups")', window)
        self.assertIn("registry.root", window)


class TestBackupContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api = (ROOT / "webapi.py").read_text(encoding="utf-8")
        cls.backup = (ROOT / "packages" / "fleet" / "src" / "hermes_fleet"
                      / "backup.py").read_text(encoding="utf-8")

    def test_注册了备份接口(self) -> None:
        self.assertIn('f"{self.prefix}/backup"', self.api)
        self.assertIn("self.backup", self.api)

    def test_备份走一致性快照(self) -> None:
        self.assertIn(".backup(", self.backup)
        self.assertIn("mode=ro", self.backup)
        # 真正执行拷贝的那行必须是 backup()，不是 copy/copyfile。
        self.assertIn("src.backup(dest_conn)", self.backup)

    def test_索引与备注一起打包(self) -> None:
        self.assertIn("INDEX_FILE", self.backup)
        self.assertIn("NOTES_FILE", self.backup)
        self.assertIn("MANIFEST", self.backup)


class TestOpsUiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (VIEW_DIR / "index.html").read_text(encoding="utf-8")
        cls.js = (VIEW_DIR / "app.js").read_text(encoding="utf-8")

    def test_侧栏有打开与备份按钮(self) -> None:
        for ident in ("btn-open-root", "btn-backup", "backup-result"):
            self.assertIn(f'id="{ident}"', self.html)

    def test_前端调用后端接口(self) -> None:
        self.assertIn('apiPost("open-root"', self.js)
        self.assertIn('apiPost("backup"', self.js)

    def test_打开目录用固定枚举(self) -> None:
        self.assertIn('openDir("root"', self.js)
        self.assertIn('openDir("backups"', self.js)


class TestVectorUiContract(unittest.TestCase):
    """向量模型页：脚本里 getElementById 的每个 id 都必须在 HTML 里存在。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (VIEW_DIR / "index.html").read_text(encoding="utf-8")
        cls.js = (VIEW_DIR / "vector.js").read_text(encoding="utf-8")

    def test_页面与导航已挂载(self) -> None:
        self.assertIn('id="view-vector"', self.html)
        self.assertIn('data-nav="vector"', self.html)
        self.assertIn('src="./vector.js"', self.html)

    def test_脚本引用的元素都存在(self) -> None:
        import re

        ids = set(re.findall(r'\$\("([a-z0-9-]+)"\)', self.js))
        self.assertTrue(ids)
        missing = [i for i in sorted(ids) if f'id="{i}"' not in self.html]
        self.assertEqual(missing, [], f"vector.js 引用了不存在的元素：{missing}")

    def test_离开页面会停掉轮询(self) -> None:
        # 没有 leave 的话，离开向量页之后仍会每秒发请求。
        self.assertIn("leave()", self.js)
        self.assertIn("clearInterval", self.js)
        self.assertIn("leave", (VIEW_DIR / "app.js").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
