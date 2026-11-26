"""v2 磁盘布局：每个账号一个目录，库与配置放在一起。

这组用例盯的是「搬数据」这件事本身——它会直接动用户唯一的记忆文件：
- 迁移必须先备份，备份失败就不许动原库；
- 每个账号搬完要校验完整性**和行数**，不一致就搬回去；
- 迁移要幂等，跑第二遍不能把已搬好的库再折腾一次；
- 坏库、孤立库宁可不动，也不能猜一个 platform_id 把别人的记忆挂上去；
- 备注属于账号，应该跟着 `bot.json` 走，而不是永远留在根目录的 json 里。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for p in ("contracts", "recall", "storage_sqlite", "fleet"):
    sys.path.insert(0, str(_ROOT / "packages" / p / "src"))

from hermes_fleet import BotRegistry, MemoryService, bot_file_stem  # noqa: E402
from hermes_fleet.registry import (  # noqa: E402
    BOT_DB_FILE,
    BOT_META_FILE,
    BOTS_DIR,
    LAYOUT_FILE,
    MAX_DESCRIPTION,
)


class LayoutBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="fleet-v2-")
        self.root = os.path.join(self.tmp, "bots")
        self.reg = BotRegistry(self.root, max_open=4)

    def tearDown(self) -> None:
        self.reg.close_all()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------------------------------------------------------------- 工具

    def bot_dir(self, pid: str) -> str:
        return os.path.join(self.root, BOTS_DIR, bot_file_stem(pid))

    def read_meta(self, pid: str) -> dict:
        path = os.path.join(self.bot_dir(pid), BOT_META_FILE)
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)

    def read_json(self, name: str) -> dict:
        with open(os.path.join(self.root, name), encoding="utf-8") as fp:
            return json.load(fp)

    def demote_to_v1(self, pid: str) -> str:
        """把一个已建好的 v2 账号还原成 v1 的平铺文件，模拟升级前的现场。"""
        stem = bot_file_stem(pid)
        self.reg.close_all()
        src = os.path.join(self.bot_dir(pid), BOT_DB_FILE)
        dst = os.path.join(self.root, stem + ".db")
        os.replace(src, dst)
        shutil.rmtree(self.bot_dir(pid))
        return dst

    def add_note_file(self, pid: str, note: str) -> None:
        notes = {}
        path = os.path.join(self.root, "_notes.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fp:
                notes = json.load(fp)
        notes[pid] = note
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(notes, fp, ensure_ascii=False)


class TestNewLayout(LayoutBase):
    def test_new_account_lands_in_own_directory(self) -> None:
        self.reg.get("qq-01").close()
        path = self.reg.db_path("qq-01")
        self.assertEqual(path, os.path.join(self.bot_dir("qq-01"), BOT_DB_FILE))
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(os.path.isfile(
            os.path.join(self.bot_dir("qq-01"), BOT_META_FILE)))

        meta = self.read_meta("qq-01")
        self.assertEqual(meta["schema"], 1)
        self.assertEqual(meta["platform_id"], "qq-01")
        self.assertEqual(meta["overrides"], {})
        self.assertGreater(meta["created_at"], 0)

        self.assertEqual(self.reg.layout_state(), "v2")
        row = self.reg.accounts()[0]
        self.assertEqual(row["file"], os.path.join(BOTS_DIR, bot_file_stem("qq-01"),
                                                   BOT_DB_FILE))
        self.assertFalse(row["missing"])

    def test_pending_note_moves_into_bot_json(self) -> None:
        """先起名、后说话的账号：备注在落库那一刻搬进 bot.json。"""
        self.reg.set_note("qq-77", "还没说话的号")
        self.assertEqual(self.read_json("_notes.json")["qq-77"], "还没说话的号")

        self.reg.get("qq-77").close()
        self.assertEqual(self.read_meta("qq-77")["note"], "还没说话的号")
        self.assertNotIn("qq-77", self.read_json("_notes.json"))

    def test_notes_and_overrides_survive_restart(self) -> None:
        self.reg.get("qq-01").close()
        self.reg.set_note("qq-01", "客服号")
        self.reg.set_description("qq-01", "只处理售后问题")
        self.reg.set_overrides("qq-01", {"summarize_enabled": False, "recall_k": 99})

        again = BotRegistry(self.root, max_open=4)
        try:
            self.assertEqual(again.note_of("qq-01"), "客服号")
            self.assertEqual(again.description_of("qq-01"), "只处理售后问题")
            self.assertEqual(again.overrides_of("qq-01"),
                             {"summarize_enabled": False, "recall_k": 20})
        finally:
            again.close_all()

    def test_description_is_clipped(self) -> None:
        self.reg.get("qq-01").close()
        self.reg.set_description("qq-01", "说" * 400)
        self.assertEqual(len(self.reg.description_of("qq-01")), MAX_DESCRIPTION)

    def test_overrides_reject_unknown_and_clear_with_none(self) -> None:
        self.reg.get("qq-01").close()
        with self.assertRaises(ValueError):
            self.reg.set_overrides("qq-01", {"no_such_switch": True})
        with self.assertRaises(ValueError):
            self.reg.set_overrides("qq-01", {"recall_k": "很多"})

        self.reg.set_overrides("qq-01", {"summarize_enabled": False})
        self.assertEqual(self.reg.overrides_of("qq-01"), {"summarize_enabled": False})
        self.reg.set_overrides("qq-01", {"summarize_enabled": None})
        self.assertEqual(self.reg.overrides_of("qq-01"), {})

    def test_purge_removes_whole_bot_directory(self) -> None:
        self.reg.get("qq-01").close()
        self.reg.set_note("qq-01", "客服号")
        self.assertTrue(self.reg.purge("qq-01"))
        self.assertFalse(os.path.exists(self.bot_dir("qq-01")))
        self.assertEqual(self.reg.note_of("qq-01"), "")
        self.assertEqual(self.reg.accounts(), [])


class TestDualRead(LayoutBase):
    def test_legacy_db_is_read_in_place(self) -> None:
        """v1 老库在迁移之前必须照常可读（否则升级即断服）。"""
        self.reg.get("qq-01").close()
        legacy = self.demote_to_v1("qq-01")

        again = BotRegistry(self.root, max_open=4)
        try:
            self.assertEqual(again.db_path("qq-01"), legacy)
            self.assertEqual(again.layout_state(), "v1")
            again.get("qq-01").close()
            self.assertTrue(os.path.isfile(legacy), "读老库不能把它改名/搬走")
        finally:
            again.close_all()

    def test_accounts_list_both_layouts_and_flag_orphan(self) -> None:
        self.reg.get("qq-01").close()
        self.demote_to_v1("qq-01")
        # 再造一个孤立 v1 文件（索引里没有它）
        ghost = bot_file_stem("qq-ghost") + ".db"
        with open(os.path.join(self.root, ghost), "wb") as fp:
            fp.write(b"SQLite format 3\x00" + b"\x00" * 100)

        again = BotRegistry(self.root, max_open=4)
        try:
            rows = again.accounts()
            self.assertEqual(len(rows), 2)
            orphans = [r for r in rows if r["orphan"]]
            self.assertEqual([r["stem"] for r in orphans], [ghost[:-3]])
            self.assertEqual(orphans[0]["platform_id"], "")
        finally:
            again.close_all()


class TestMigration(LayoutBase):
    def test_migrate_moves_rows_notes_and_writes_layout(self) -> None:
        import asyncio

        async def seed() -> None:
            svc = MemoryService(self.reg)
            scene = await svc.scene("qq-01", "u1", "", is_group=False)
            await svc.save(scene, "迁移不能丢：钥匙在抽屉里")

        asyncio.run(seed())
        self.reg.close_all()
        legacy = self.demote_to_v1("qq-01")
        self.add_note_file("qq-01", "客服号")

        backup_dir = os.path.join(self.tmp, "backups")
        mig = BotRegistry(self.root, max_open=4)
        try:
            result = mig.migrate_v2(backup_dir=backup_dir, version="test")
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["migrated"], [bot_file_stem("qq-01")])
            self.assertFalse(os.path.exists(legacy), "迁移成功后旧文件应被清掉")
            self.assertFalse(os.path.exists(legacy + "-wal"))
            self.assertTrue(os.path.isfile(
                os.path.join(self.bot_dir("qq-01"), BOT_DB_FILE)))
            self.assertEqual(self.read_meta("qq-01")["note"], "客服号")
            self.assertEqual(self.read_json(LAYOUT_FILE)["schema"], 2)
            self.assertEqual(mig.layout_state(), "v2")
            self.assertTrue(os.path.isfile(result["backup"]))
            with zipfile.ZipFile(result["backup"]) as zf:
                self.assertIn(bot_file_stem("qq-01") + ".db", zf.namelist())
        finally:
            mig.close_all()

        again = BotRegistry(self.root, max_open=4)
        try:
            self.assertEqual(again.index_of(bot_file_stem("qq-01")), "qq-01")
            self.assertEqual(again.note_of("qq-01"), "客服号")
        finally:
            again.close_all()

    def test_migrate_is_idempotent(self) -> None:
        self.reg.get("qq-01").close()
        self.demote_to_v1("qq-01")
        first = self.reg.migrate_v2()
        self.assertTrue(first["ok"], first)
        target = os.path.join(self.bot_dir("qq-01"), BOT_DB_FILE)
        before = os.stat(target).st_mtime_ns, os.path.getsize(target)

        second = self.reg.migrate_v2()
        self.assertTrue(second["ok"], second)
        self.assertEqual(second["migrated"], [])
        after = os.stat(target).st_mtime_ns, os.path.getsize(target)
        self.assertEqual(before, after, "第二遍迁移不该再碰已迁好的库")

    def test_corrupt_legacy_is_left_untouched(self) -> None:
        stem = bot_file_stem("qq-01")
        with open(os.path.join(self.root, "_index.json"), "w",
                  encoding="utf-8") as fp:
            json.dump({stem: "qq-01"}, fp)
        legacy = os.path.join(self.root, stem + ".db")
        with open(legacy, "wb") as fp:
            fp.write(b"this is not a sqlite database")

        again = BotRegistry(self.root, max_open=4)
        try:
            result = again.migrate_v2()
        finally:
            again.close_all()
        self.assertFalse(result["ok"])
        self.assertTrue(result["failed"])
        self.assertTrue(os.path.isfile(legacy), "失败的库必须原地保留")
        self.assertFalse(os.path.exists(
            os.path.join(self.bot_dir("qq-01"), BOT_DB_FILE)))

    def test_orphan_legacy_is_not_migrated(self) -> None:
        """文件名 hash 不可逆，猜 platform_id 等于把记忆挂错人。"""
        stem = bot_file_stem("qq-ghost")
        with open(os.path.join(self.root, stem + ".db"), "wb") as fp:
            fp.write(b"SQLite format 3\x00" + b"\x00" * 100)
        result = self.reg.migrate_v2()
        self.assertEqual(result["migrated"], [])
        self.assertEqual(result["skipped"][0]["stem"], stem)
        self.assertIn("孤立库", result["skipped"][0]["reason"])
        self.assertTrue(os.path.isfile(os.path.join(self.root, stem + ".db")))

    def test_backup_failure_aborts_before_touching_files(self) -> None:
        self.reg.get("qq-01").close()
        legacy = self.demote_to_v1("qq-01")
        blocker = os.path.join(self.tmp, "not-a-dir")
        with open(blocker, "w", encoding="utf-8") as fp:
            fp.write("x")

        result = self.reg.migrate_v2(backup_dir=blocker)
        self.assertFalse(result["ok"])
        self.assertIn("备份失败", result["error"])
        self.assertTrue(os.path.isfile(legacy), "备份失败时原库必须原封不动")
        self.assertFalse(os.path.exists(
            os.path.join(self.bot_dir("qq-01"), BOT_DB_FILE)))


class TestPerBotOverrides(LayoutBase):
    def test_override_only_affects_its_own_bot(self) -> None:
        """同一个 service 实例下，A 号的开关不能影响 B 号。"""
        from hermes_fleet import Scene

        svc = MemoryService(
            self.reg,
            group_private_memory=False,
            recall_k=5,
            overrides_of=lambda pid: (
                {"group_private_memory": True, "recall_k": 9}
                if pid == "qq-01" else {}
            ),
        )
        a = Scene(bot_id=svc.bot_of("qq-01"), owner_id="u1",
                  group_id="g1", is_group=True)
        b = Scene(bot_id=svc.bot_of("qq-02"), owner_id="u1",
                  group_id="g1", is_group=True)
        self.assertEqual(svc._recall_for(a), 9)
        self.assertEqual(svc._recall_for(b), 5)
        self.assertTrue(svc._group_private_for(a))
        self.assertFalse(svc._group_private_for(b))

    def test_bad_resolver_falls_back_to_global(self) -> None:
        from hermes_fleet import Scene

        def broken(_pid: str) -> dict:
            raise RuntimeError("bot.json 坏了")

        svc = MemoryService(self.reg, group_private_memory=True,
                            recall_k=4, overrides_of=broken)
        scene = Scene(bot_id=svc.bot_of("qq-01"), owner_id="u1",
                      group_id="g1", is_group=True)
        self.assertEqual(svc._recall_for(scene), 4)
        self.assertTrue(svc._group_private_for(scene))


if __name__ == "__main__":
    unittest.main()
