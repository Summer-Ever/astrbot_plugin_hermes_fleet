"""备份 v2 与恢复/回滚的端到端用例（纯逻辑层，不起 AstrBot）。

钉住的是**会毁数据的那几个分支**：包坏了、路径穿越、降级、恢复中途失败、
确认词缺失。每个用例失败时都应该能让用户的数据原样还在。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
import zipfile
from unittest import mock

from hermes_contracts import (
    BotId,
    Memory,
    MemoryId,
    MemoryScope,
    Message,
    SessionId,
    Visibility,
)
from hermes_fleet import BotRegistry
from hermes_fleet.backup import BOT_DB_FILE, CONFIG_ENTRY, MANIFEST, backup_bots
from hermes_fleet.restore import (
    RestoreError,
    inspect_backup,
    list_backups,
    restore_backup,
    rollback_restore,
)


def _run(coro):
    return asyncio.run(coro)


def _seed(root: str, bot: str = "qq-01", text: str = "你好"):
    """建一个 v2 账号，写一条消息与一条记忆，返回 registry。"""
    reg = BotRegistry(root)
    db = reg.get(bot)
    _run(db.messages.append_batch([Message(
        bot_id=BotId(bot), session_id=SessionId("s1"), role="user",
        content=text, ts=1.0, sender_id="u1",
    )]))
    _run(db.memories.add(Memory(
        id=MemoryId(0), bot_id=BotId(bot), content="喜欢猫",
        scope=MemoryScope.USER, visibility=Visibility.PRIVATE,
        owner_id="u1", group_id="", kind="fact", source="tool", ts=1.0,
    )))
    return reg


def _counts(root: str, bot: str) -> tuple[int, int]:
    path = BotRegistry(root).db_path(bot)
    conn = sqlite3.connect(path)
    try:
        messages = conn.execute("select count(*) from messages").fetchone()[0]
        memories = conn.execute("select count(*) from memories").fetchone()[0]
        return int(messages), int(memories)
    finally:
        conn.close()


def _user_version(path: str, version: int) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"PRAGMA user_version={int(version)}")
        conn.commit()
    finally:
        conn.close()


class BackupV2Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes-bk-")
        self.root = os.path.join(self.tmp, "bots")
        self.out = os.path.join(self.tmp, "backups")
        os.makedirs(self.root)
        os.makedirs(self.out)
        self.reg = _seed(self.root)
        self.reg.set_note("qq-01", "客服号")
        self.reg.set_description("qq-01", "只做售后")
        self.reg.set_overrides("qq-01", {"recall_k": 9, "summarize_enabled": False})

    def tearDown(self):
        self.reg.close_all()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_v2_目录结构与配置一起打包(self):
        result = backup_bots(
            root=self.root, out_dir=self.out, version="9.9.9",
            config={"recall_k": 5, "bot_db_dir": "D:/elsewhere"},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["bots"], 1)
        self.assertEqual(result["layout"], "v2")
        with zipfile.ZipFile(result["path"]) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read(MANIFEST))
        self.assertTrue(any(n.startswith("bots/") and n.endswith("/memory.db")
                            for n in names))
        self.assertTrue(any(n.endswith("/bot.json") for n in names))
        self.assertTrue(any(n.endswith("/memory.db") for n in names))
        self.assertIn(CONFIG_ENTRY, names)
        self.assertEqual(manifest["format_version"], 2)
        self.assertEqual(manifest["layout"], "v2")
        self.assertIn("recall_k", manifest["config_keys"])
        db_row = next(r for r in manifest["files"] if r["name"].endswith(".db"))
        self.assertEqual(db_row["integrity"], "ok")
        self.assertEqual(len(db_row["sha256"]), 64)

    def test_同秒备份不覆盖(self):
        a = backup_bots(root=self.root, out_dir=self.out)
        b = backup_bots(root=self.root, out_dir=self.out)
        self.assertNotEqual(a["filename"], b["filename"])
        self.assertTrue(os.path.exists(a["path"]) and os.path.exists(b["path"]))

    def test_只保留最近十份(self):
        for _ in range(12):
            backup_bots(root=self.root, out_dir=self.out, keep=10)
        self.assertEqual(len(list_backups(self.out)), 10)

    def test_打包后不留临时库与旁路文件(self):
        """备份目录只能留下 zip：临时库的 `-wal`/`-shm` 不清会一次攒 32 KB。"""
        backup_bots(root=self.root, out_dir=self.out)
        names = os.listdir(self.out)
        self.assertEqual([n for n in names if n.startswith("tmp")], [], names)


class RestoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes-rs-")
        self.src = os.path.join(self.tmp, "src")
        self.dst = os.path.join(self.tmp, "dst")
        self.out = os.path.join(self.tmp, "backups")
        os.makedirs(self.src)
        os.makedirs(self.dst)
        self.reg = _seed(self.src, text="原始消息")
        self.reg.set_note("qq-01", "客服号")
        self.reg.set_description("qq-01", "只做售后")
        self.reg.set_overrides("qq-01", {"recall_k": 9})
        self.zip = backup_bots(
            root=self.src, out_dir=self.out, version="9.9.9",
            config={"recall_k": 6, "bot_db_dir": "D:/elsewhere"},
        )["path"]

    def tearDown(self):
        self.reg.close_all()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_预检通过且带出配置(self):
        plan = inspect_backup(self.zip, root=self.dst)
        self.assertTrue(plan["ok"], plan["problems"])
        self.assertEqual(plan["bots"], 1)
        self.assertEqual(plan["config"]["recall_k"], 6)

    def test_完整恢复账号数据与配置(self):
        result = restore_backup(
            self.zip, root=self.dst, out_dir=self.out, confirm="RESTORE",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["restored"], 1)
        self.assertEqual(result["config"]["recall_k"], 6)
        self.assertTrue(result["snapshot"].startswith(self.out))
        reg = BotRegistry(self.dst)
        self.assertEqual(reg.note_of("qq-01"), "客服号")
        self.assertEqual(reg.description_of("qq-01"), "只做售后")
        self.assertEqual(reg.overrides_of("qq-01")["recall_k"], 9)
        self.assertEqual(_counts(self.dst, "qq-01"), (1, 1))

    def test_缺少确认词拒绝且不动数据(self):
        with self.assertRaises(RestoreError):
            restore_backup(self.zip, root=self.dst, out_dir=self.out, confirm="")
        self.assertFalse(os.path.exists(BotRegistry(self.dst).db_path("qq-01")))

    def test_损坏的包被预检拦住(self):
        bad = os.path.join(self.out, "bad.zip")
        with open(self.zip, "rb") as fp:
            data = fp.read()
        with open(bad, "wb") as fp:
            fp.write(data[: len(data) // 2])
        plan = inspect_backup(bad, root=self.dst)
        self.assertFalse(plan["ok"])
        self.assertTrue(plan["problems"])
        with self.assertRaises(RestoreError):
            restore_backup(bad, root=self.dst, out_dir=self.out, confirm="RESTORE")

    def test_校验和不符被拒绝(self):
        bad = os.path.join(self.out, "tampered.zip")
        with zipfile.ZipFile(self.zip) as src, \
                zipfile.ZipFile(bad, "w") as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename.endswith(".db"):
                    data = b"SQLite format 3\x00" + b"\x00" * 64
                dst.writestr(info, data)
        plan = inspect_backup(bad, root=self.dst)
        self.assertFalse(plan["ok"])
        self.assertTrue(any("校验和" in p for p in plan["problems"]))

    def test_路径穿越条目被拒绝(self):
        bad = os.path.join(self.out, "zip-slip.zip")
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr(MANIFEST, json.dumps({
                "format": "hermes-fleet-backup", "format_version": 2,
                "created_at": 1.0, "bots": 0, "files": [],
            }))
            zf.writestr("../evil.txt", "x")
        plan = inspect_backup(bad, root=self.dst)
        self.assertFalse(plan["ok"])
        self.assertTrue(any("可疑路径" in p for p in plan["problems"]))

    def test_降级恢复被拒绝(self):
        # 让备份里的库版本比本机新：本机同账号库先建出来（version=0）。
        BotRegistry(self.dst).get("qq-01").close()
        self.reg.close_all()
        _user_version(BotRegistry(self.src).db_path("qq-01"), 7)
        newer = backup_bots(root=self.src, out_dir=self.out)["path"]
        plan = inspect_backup(newer, root=self.dst)
        self.assertFalse(plan["ok"])
        self.assertTrue(any("比本机" in p for p in plan["problems"]))
        with self.assertRaises(RestoreError):
            restore_backup(newer, root=self.dst, out_dir=self.out, confirm="RESTORE")

    def test_恢复前快照失败则不动数据(self):
        reg = _seed(self.dst, text="本机数据")
        reg.close_all()
        before = _counts(self.dst, "qq-01")
        with mock.patch("hermes_fleet.restore.backup_bots",
                        side_effect=RuntimeError("disk full")):
            with self.assertRaises(RestoreError):
                restore_backup(self.zip, root=self.dst, out_dir=self.out,
                               confirm="RESTORE")
        self.assertEqual(_counts(self.dst, "qq-01"), before)

    def test_换入阶段失败会回滚(self):
        reg = _seed(self.dst, text="本机数据")
        reg.set_note("qq-01", "本机备注")
        reg.close_all()
        before = _counts(self.dst, "qq-01")

        def boom():
            raise RuntimeError("boom")

        with self.assertRaises(RestoreError):
            restore_backup(self.zip, root=self.dst, out_dir=self.out,
                           confirm="RESTORE", after_apply=boom)
        reg2 = BotRegistry(self.dst)
        self.assertEqual(reg2.note_of("qq-01"), "本机备注")
        self.assertEqual(_counts(self.dst, "qq-01"), before)

    def test_回滚到恢复前快照(self):
        reg = _seed(self.dst, text="恢复前")
        # 多一条「恢复前独有」的记忆，回滚后才能和备份包里的状态区分开。
        _run(reg.get("qq-01").memories.add(Memory(
            id=MemoryId(0), bot_id=BotId("qq-01"), content="恢复前独有",
            scope=MemoryScope.USER, visibility=Visibility.PRIVATE,
            owner_id="u1", group_id="", kind="fact", source="tool", ts=2.0,
        )))
        reg.close_all()
        self.assertEqual(_counts(self.dst, "qq-01"), (1, 2))
        result = restore_backup(self.zip, root=self.dst, out_dir=self.out,
                                confirm="RESTORE")
        self.assertEqual(_counts(self.dst, "qq-01"), (1, 1))
        snapshot = os.path.basename(result["snapshot"])
        self.assertIn("hermes-pre-restore-", snapshot)

        rolled = rollback_restore(
            snapshot, root=self.dst, out_dir=self.out, confirm="RESTORE",
        )
        self.assertTrue(rolled["ok"])
        # 恢复前的库里有 1 条消息、2 条记忆 —— 回滚后必须回到这个状态。
        self.assertEqual(_counts(self.dst, "qq-01"), (1, 2))

    def test_回滚只接受恢复前快照(self):
        with self.assertRaises(RestoreError):
            rollback_restore(self.zip, root=self.dst, out_dir=self.out,
                             confirm="RESTORE")

    def test_v1_平铺备份恢复成_v2(self):
        # 先关连接：库跑在 WAL 下，直接拷主文件会漏掉还没 checkpoint 的表。
        self.reg.close_all()
        v1 = os.path.join(self.tmp, "v1")
        os.makedirs(v1)
        stem = os.path.basename(os.path.dirname(
            BotRegistry(self.src).db_path("qq-01")))
        shutil.copy(BotRegistry(self.src).db_path("qq-01"),
                    os.path.join(v1, stem + ".db"))
        with open(os.path.join(v1, "_index.json"), "w", encoding="utf-8") as fp:
            json.dump({stem: "qq-01"}, fp)
        with open(os.path.join(v1, "_notes.json"), "w", encoding="utf-8") as fp:
            json.dump({"qq-01": "老备注"}, fp)
        old_zip = backup_bots(root=v1, out_dir=self.out)["path"]
        restored = restore_backup(old_zip, root=self.dst, out_dir=self.out,
                                  confirm="RESTORE")
        self.assertEqual(restored["restored"], 1)
        reg = BotRegistry(self.dst)
        self.assertEqual(reg.note_of("qq-01"), "老备注")
        self.assertEqual(_counts(self.dst, "qq-01"), (1, 1))
        self.assertTrue(os.path.exists(os.path.join(
            self.dst, "bots", stem, BOT_DB_FILE)))


class ListBackupsTest(unittest.TestCase):
    def test_恢复点与普通备份分开标记(self):
        tmp = tempfile.mkdtemp(prefix="hermes-ls-")
        root = os.path.join(tmp, "bots")
        out = os.path.join(tmp, "backups")
        os.makedirs(root)
        try:
            backup_bots(root=root, out_dir=out)
            backup_bots(root=root, out_dir=out, prefix="hermes-pre-restore-")
            rows = list_backups(out)
            kinds = {r["kind"] for r in rows}
            self.assertIn("backup", kinds)
            self.assertIn("restore-point", kinds)
            self.assertEqual(rows[0]["kind"], "restore-point")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
