"""账号备注：一个纯显示层的别名，但不该有任何「悄悄改数据」的副作用。

每一条都对应一个会真正伤人的错法：
- 备注写进库文件名 → 账号一改名，索引与磁盘对不上，记忆就「消失」了。
- 备注带换行/控制字符 → 粘进来一段文本，整张表格被撑变形。
- 删号后备注留着 → 同 id 重建的新账号继承一个不属于它的名字。
- 搜索不覆盖备注 → 起完名字却搜不到自己刚起的名。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for p in ("contracts", "recall", "storage_sqlite", "fleet"):
    sys.path.insert(0, str(_ROOT / "packages" / p / "src"))

from hermes_fleet import BotRegistry, bot_file_stem, normalize_note  # noqa: E402
from hermes_fleet.console import accounts_view  # noqa: E402


class NoteBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="fleet-note-")
        self.root = os.path.join(self.tmp, "bots")
        self.reg = BotRegistry(self.root, max_open=2)
        # 造一个真实库文件，`accounts()` 只按磁盘上的 .db 出行
        self.reg.get("qq-01").close()

    def tearDown(self) -> None:
        self.reg.close_all()
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestNormalize(unittest.TestCase):
    def test_strip_and_collapse(self) -> None:
        self.assertEqual(normalize_note("  客服号\n\n（测试） "), "客服号 （测试）")

    def test_control_chars_replaced(self) -> None:
        self.assertEqual(normalize_note("a\x00b\tc"), "a b c")

    def test_truncated(self) -> None:
        self.assertEqual(len(normalize_note("长" * 100)), 40)

    def test_none(self) -> None:
        self.assertEqual(normalize_note(None), "")


class TestRegistryNotes(NoteBase):
    def test_set_and_read(self) -> None:
        self.assertEqual(self.reg.set_note("qq-01", "客服号"), "客服号")
        self.assertEqual(self.reg.note_of("qq-01"), "客服号")

    def test_persisted_across_instances(self) -> None:
        self.reg.set_note("qq-01", "客服号")
        again = BotRegistry(self.root, max_open=2)
        self.assertEqual(again.note_of("qq-01"), "客服号")

    def test_empty_clears(self) -> None:
        self.reg.set_note("qq-01", "客服号")
        self.assertEqual(self.reg.set_note("qq-01", "   "), "")
        self.assertEqual(self.reg.note_of("qq-01"), "")

    def test_file_name_untouched(self) -> None:
        """备注绝不能影响文件名：改名等于换账号。"""
        before = self.reg.db_path("qq-01")
        self.reg.set_note("qq-01", "全新的名字")
        self.assertEqual(self.reg.db_path("qq-01"), before)
        self.assertTrue(os.path.exists(before))
        self.assertEqual(os.path.basename(before),
                         bot_file_stem("qq-01") + ".db")

    def test_accounts_carry_note(self) -> None:
        self.reg.set_note("qq-01", "客服号")
        rows = self.reg.accounts()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["note"], "客服号")

    def test_purge_drops_note(self) -> None:
        self.reg.set_note("qq-01", "客服号")
        self.assertTrue(self.reg.purge("qq-01"))
        self.assertEqual(self.reg.note_of("qq-01"), "")
        again = BotRegistry(self.root, max_open=2)
        self.assertEqual(again.note_of("qq-01"), "")

    def test_broken_notes_file_is_survivable(self) -> None:
        """备注文件坏了只等于「没有备注」，不能连账号列表一起挂掉。"""
        self.reg.set_note("qq-01", "客服号")
        with open(os.path.join(self.root, "_notes.json"), "w",
                  encoding="utf-8") as fp:
            fp.write("{ not json")
        again = BotRegistry(self.root, max_open=2)
        self.assertEqual(again.note_of("qq-01"), "")
        self.assertEqual(len(again.accounts()), 1)


class TestViewsUseNote(NoteBase):
    def setUp(self) -> None:
        super().setUp()
        self.reg.get("qq-02").close()
        self.reg.set_note("qq-01", "客服号")
        self.reg.set_note("qq-02", "测试号")

    def test_search_by_note(self) -> None:
        view = accounts_view(self.reg.accounts(), query="客服")
        self.assertEqual([i["platform_id"] for i in view["items"]], ["qq-01"])

    def test_search_still_matches_id(self) -> None:
        view = accounts_view(self.reg.accounts(), query="qq-02")
        self.assertEqual([i["platform_id"] for i in view["items"]], ["qq-02"])

    def test_sort_by_name_uses_note(self) -> None:
        """「按名称」要按眼睛看到的那一列排，而不是按适配器 id。"""
        self.reg.set_note("qq-01", "zeta")
        self.reg.set_note("qq-02", "alpha")
        view = accounts_view(self.reg.accounts(), sort="name")
        # 按 id 排会是 qq-01, qq-02；按备注排才是 alpha(zeta 之后)
        self.assertEqual([i["note"] for i in view["items"]], ["alpha", "zeta"])

    def test_alert_carries_note(self) -> None:
        """告警条上也要能给名字：写一串适配器 id 等于让人自己去对表格。"""
        now = 1_800_000_000.0
        rows = [{
            "platform_id": "qq-01", "stem": "qq-01-a1b2c3d4", "file": "qq-01-a1b2c3d4.db",
            "note": "客服号", "size_bytes": 4096, "wal_bytes": 0,
            "last_write_at": now - 30 * 86400, "opened": False,
            "orphan": False, "missing": False,
        }]
        view = accounts_view(rows, stale_days=7, now=now)
        self.assertEqual(view["alerts"][0]["code"], "stale")
        self.assertEqual(view["alerts"][0]["note"], "客服号")


if __name__ == "__main__":
    unittest.main()
