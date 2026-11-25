"""备份：验证 zip 里是**一致性**快照，且能恢复出完整记忆。

关键点不是「打了个 zip」，而是：
1. 活跃写入（WAL 未 checkpoint）的库也能备份到最新数据；
2. 索引 + 备注一起打包，恢复后账号不变成孤儿；
3. 备份出的库可被 SqliteBotDb 直接打开读出记忆。
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for p in ("contracts", "recall", "storage_sqlite", "fleet"):
    sys.path.insert(0, str(_ROOT / "packages" / p / "src"))

from hermes_fleet.backup import (  # noqa: E402
    INDEX_FILE, MANIFEST, NOTES_FILE, backup_bots, backup_filename,
)


class BackupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="fleet-backup-")
        self.root = os.path.join(self.tmp, "bots")
        self.out = os.path.join(self.tmp, "backups")
        os.makedirs(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_live_db(self, name: str, text: str) -> None:
        """造一个 WAL 未合并的活跃库：数据写在 -wal 里，主库还是空的。"""
        path = os.path.join(self.root, name)
        conn = sqlite3.connect(path, isolation_level=None)
        conn.execute("pragma journal_mode=wal")
        conn.execute("create table memories (id integer primary key, content text)")
        conn.execute("insert into memories (content) values (?)", (text,))
        # 不 checkpoint、不 close：模拟「正在被写入」的库。

    def test_备份文件名可解析(self) -> None:
        name = backup_filename(1700000000.0)
        self.assertTrue(name.startswith("hermes-backup-"))
        self.assertTrue(name.endswith(".zip"))

    def test_一致性快照包含未合并的_wal_数据(self) -> None:
        self._make_live_db("qq-01-abcd1234.db", "WAL 里的最新记忆")
        result = backup_bots(root=self.root, out_dir=self.out, version="0.9.0")

        self.assertEqual(result["bots"], 1)
        with zipfile.ZipFile(result["path"]) as zf:
            names = zf.namelist()
            self.assertIn("qq-01-abcd1234.db", names)
            self.assertIn(MANIFEST, names)
            # 快照里的库要能读出那条只在 WAL 里的记忆。
            snap = zf.read("qq-01-abcd1234.db")
        snap_path = os.path.join(self.tmp, "restored.db")
        with open(snap_path, "wb") as fp:
            fp.write(snap)
        conn = sqlite3.connect(snap_path)
        row = conn.execute("select content from memories").fetchone()
        conn.close()
        self.assertEqual(row[0], "WAL 里的最新记忆")

    def test_索引与备注一起打包(self) -> None:
        self._make_live_db("qq-01-abcd1234.db", "x")
        with open(os.path.join(self.root, INDEX_FILE), "w", encoding="utf-8") as fp:
            json.dump({"qq-01-abcd1234": "qq-01"}, fp)
        with open(os.path.join(self.root, NOTES_FILE), "w", encoding="utf-8") as fp:
            json.dump({"qq-01": "客服号"}, fp)

        result = backup_bots(root=self.root, out_dir=self.out)
        with zipfile.ZipFile(result["path"]) as zf:
            self.assertIn(INDEX_FILE, zf.namelist())
            self.assertIn(NOTES_FILE, zf.namelist())
            notes = json.loads(zf.read(NOTES_FILE))
        self.assertEqual(notes["qq-01"], "客服号")

    def test_manifest_可解析(self) -> None:
        result = backup_bots(root=self.root, out_dir=self.out, version="0.9.0")
        with zipfile.ZipFile(result["path"]) as zf:
            m = json.loads(zf.read(MANIFEST))
        self.assertEqual(m["format"], "hermes-fleet-backup")
        self.assertEqual(m["bots"], 0)

    def test_不覆盖同秒备份(self) -> None:
        a = backup_bots(root=self.root, out_dir=self.out, now=1700000000.0)
        b = backup_bots(root=self.root, out_dir=self.out, now=1700000000.0)
        self.assertNotEqual(a["path"], b["path"])

    def test_空目录也能备份(self) -> None:
        result = backup_bots(root=self.root, out_dir=self.out)
        self.assertEqual(result["bots"], 0)
        self.assertTrue(os.path.isfile(result["path"]))

    def test_目录不存在则拒绝(self) -> None:
        with self.assertRaises(ValueError):
            backup_bots(root=os.path.join(self.tmp, "nope"), out_dir=self.out)


if __name__ == "__main__":
    unittest.main()
