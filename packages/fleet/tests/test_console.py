"""管理台纯逻辑测试。

这些用例覆盖的是「页面显示错了」的典型来源：分页与过滤的顺序、未探测
被打成 0、告警排序、容量饱和判定。都是纯函数，不需要 AstrBot。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for p in ("contracts", "recall", "fleet"):
    sys.path.insert(0, str(_ROOT / "packages" / p / "src"))

from hermes_contracts.ids import BotId  # noqa: E402
from hermes_fleet.console import (  # noqa: E402
    _human_bytes,
    accounts_view,
    health_view,
    merge_probe,
)

NOW = 1_800_000_000.0
DAY = 86400.0


def row(pid: str, *, days_ago: float = 0.0, size: int = 4096, wal: int = 0,
        opened: bool = False, orphan: bool = False, missing: bool = False) -> dict:
    return {
        "platform_id": pid,
        "stem": pid.replace("/", "-") + "-a1b2c3d4",
        "file": "" if missing else pid.replace("/", "-") + "-a1b2c3d4.db",
        "size_bytes": size,
        "wal_bytes": wal,
        "last_write_at": 0.0 if days_ago < 0 else NOW - days_ago * DAY,
        "opened": opened,
        "orphan": orphan,
        "missing": missing,
    }


class TestAccountsView(unittest.TestCase):
    def test_total_is_after_filter(self) -> None:
        """先过滤再分页：否则「共 3 条」下面列 0 条。"""
        rows = [row(f"qq-{i:02d}") for i in range(10)]
        v = accounts_view(rows, query="qq-0", limit=3, now=NOW)
        self.assertEqual(v["total"], 10)          # qq-0..qq-09 都含 qq-0
        self.assertEqual(len(v["items"]), 3)
        v2 = accounts_view(rows, query="qq-00", limit=3, now=NOW)
        self.assertEqual(v2["total"], 1)
        self.assertEqual(len(v2["items"]), 1)

    def test_search_matches_stem_too(self) -> None:
        """索引缺失时 platform_id 是空的，只能靠文件名找。"""
        rows = [{"platform_id": "", "stem": "alice-a1b2c3d4", "file": "alice-a1b2c3d4.db",
                 "size_bytes": 1, "wal_bytes": 0, "last_write_at": NOW, "opened": False,
                 "orphan": True, "missing": False}]
        v = accounts_view(rows, query="alice", now=NOW)
        self.assertEqual(v["total"], 1)

    def test_sort_recent_puts_freshest_first(self) -> None:
        rows = [row("old", days_ago=30), row("new", days_ago=0), row("mid", days_ago=3)]
        v = accounts_view(rows, sort="recent", now=NOW)
        self.assertEqual([i["platform_id"] for i in v["items"]], ["new", "mid", "old"])

    def test_sort_size_counts_wal(self) -> None:
        """只看主库会漏掉「还没 checkpoint 的写入」。"""
        rows = [row("small", size=1000, wal=0), row("big", size=100, wal=9000)]
        v = accounts_view(rows, sort="size", now=NOW)
        self.assertEqual(v["items"][0]["platform_id"], "big")

    def test_offset_beyond_end_is_empty_not_error(self) -> None:
        v = accounts_view([row("a")], offset=99, now=NOW)
        self.assertEqual(v["items"], [])
        self.assertEqual(v["total"], 1)

    def test_limit_clamped(self) -> None:
        v = accounts_view([row("a")], limit=99999, now=NOW)
        self.assertLessEqual(v["limit"], 500)
        v2 = accounts_view([row("a")], limit=0, now=NOW)
        self.assertGreaterEqual(v2["limit"], 1)

    def test_stale_threshold(self) -> None:
        rows = [row("fresh", days_ago=1), row("stale", days_ago=8),
                row("edge", days_ago=6.9)]
        v = accounts_view(rows, stale_days=7, now=NOW)
        flagged = {i["platform_id"] for i in v["items"] if i["stale"]}
        self.assertEqual(flagged, {"stale"})
        self.assertEqual(v["summary"]["stale"], 1)

    def test_zero_mtime_is_not_stale(self) -> None:
        """缺失文件的账号没有 mtime，不能因为「0 秒前」被判成新鲜，
        也不能因为 last=0 被判成 stale——它是 missing，单独一类。"""
        v = accounts_view([row("gone", days_ago=-1, missing=True)], now=NOW)
        self.assertFalse(v["items"][0]["stale"])
        self.assertIsNone(v["items"][0]["age_days"])

    def test_alerts_orphan_first(self) -> None:
        rows = [row("stale-bot", days_ago=30), row("ghost", orphan=True),
                row("gone", days_ago=-1, missing=True)]
        v = accounts_view(rows, now=NOW)
        levels = [a["level"] for a in v["alerts"]]
        self.assertEqual(levels[0], "error", "error 必须排在 warn 前面")
        codes = {a["code"] for a in v["alerts"]}
        self.assertEqual(codes, {"orphan", "missing", "stale"})

    def test_orphan_wins_over_stale(self) -> None:
        """一个文件既 orphan 又很久没写，应该只报 orphan（更根本的问题）。"""
        v = accounts_view([row("x", days_ago=30, orphan=True)], now=NOW)
        self.assertEqual(len(v["alerts"]), 1)
        self.assertEqual(v["alerts"][0]["code"], "orphan")

    def test_summary_counts(self) -> None:
        rows = [row("a", days_ago=30), row("b"), row("c", orphan=True),
                row("d", days_ago=-1, missing=True)]
        v = accounts_view(rows, now=NOW)
        s = v["summary"]
        self.assertEqual(s["accounts"], 4)
        self.assertEqual(s["orphan"], 1)
        self.assertEqual(s["missing"], 1)
        self.assertEqual(s["stale"], 1)


class TestMergeProbe(unittest.TestCase):
    def test_unprobed_is_none_not_zero(self) -> None:
        """用 0 占位会让人以为「这个账号没有记忆」，实际只是没查。"""
        items = [{"platform_id": "a"}, {"platform_id": "b"}]
        out = merge_probe(items, [{"platform_id": "a",
                                   "stats": {"memories": 5, "messages": 10}}])
        self.assertEqual(out[0]["stats"]["memories"], 5)
        self.assertIsNone(out[1]["stats"])

    def test_probe_error_kept(self) -> None:
        out = merge_probe([{"platform_id": "a"}],
                          [{"platform_id": "a", "error": "not_indexed"}])
        self.assertIsNone(out[0]["stats"])
        self.assertEqual(out[0]["stats_error"], "not_indexed")

    def test_does_not_mutate_input(self) -> None:
        items = [{"platform_id": "a"}]
        merge_probe(items, [{"platform_id": "a", "stats": {"memories": 1}}])
        self.assertNotIn("stats", items[0])


class TestHealthView(unittest.TestCase):
    def test_capacity_saturated(self) -> None:
        h = health_view([], root="/x", opened=24, max_open=24, now=NOW)
        self.assertTrue(h["capacity"]["saturated"])
        h2 = health_view([], root="/x", opened=3, max_open=24, now=NOW)
        self.assertFalse(h2["capacity"]["saturated"])

    def test_provider_none_is_not_healthy(self) -> None:
        """拿不到模型链时必须显式不可用，不能默认成「一切正常」。"""
        h = health_view([], root="/x", opened=0, max_open=4, provider=None, now=NOW)
        self.assertFalse(h["model"]["available"])

    def test_provider_raising_does_not_break_page(self) -> None:
        class Bad:
            def chain(self):
                raise RuntimeError("boom")

        h = health_view([], root="/x", opened=0, max_open=4, provider=Bad(), now=NOW)
        self.assertFalse(h["model"]["available"])
        self.assertIn("boom", h["model"]["error"])

    def test_provider_duck_typed(self) -> None:
        """不 import FallbackProvider：依赖方向是 workers → contracts，
        fleet 反着引会成环。所以这里只认方法名。"""
        class Stats:
            calls = 7
            failures = 2
            by_kind = {"rate_limited": 2}

        class Log:
            ts, provider, purpose = 1.0, "deepseek", "summarize"
            ok, kind, latency_ms = False, "rate_limited", 12.5

        class Chain:
            def chain(self):
                return [{"name": "deepseek", "cooling": True, "cooldown_left": 30.0}]

            def summary(self):
                return "deepseek（冷却 30s）"

            stats = Stats()
            log = [Log()]

        h = health_view([], root="/x", opened=0, max_open=4, provider=Chain(), now=NOW)
        self.assertTrue(h["model"]["available"])
        self.assertEqual(h["model"]["summary"], "deepseek（冷却 30s）")
        self.assertEqual(h["model"]["failures"], 2)
        self.assertEqual(len(h["model"]["recent"]), 1)

    def test_account_aggregates(self) -> None:
        rows = [row("a", days_ago=30, size=1000), row("b", days_ago=0, size=500, wal=100)]
        h = health_view(rows, root="/x", opened=0, max_open=4, now=NOW)
        self.assertEqual(h["accounts"]["total"], 2)
        self.assertEqual(h["accounts"]["stale"], 1)
        self.assertEqual(h["accounts"]["bytes"], 1600)

    def test_alerts_capped(self) -> None:
        rows = [row(f"bot-{i}", days_ago=30) for i in range(80)]
        h = health_view(rows, root="/x", opened=0, max_open=4, now=NOW)
        self.assertLessEqual(len(h["alerts"]), 50, "告警列表不能无上限")


class TestHumanBytes(unittest.TestCase):
    def test_units(self) -> None:
        self.assertEqual(_human_bytes(512), "512 B")
        self.assertEqual(_human_bytes(2048), "2.0 KB")
        self.assertEqual(_human_bytes(5 * 1024 ** 2), "5.0 MB")

    def test_does_not_crash_on_huge(self) -> None:
        self.assertIn("GB", _human_bytes(1024 ** 4))


class TestOrphanAndMissingRealCases(unittest.TestCase):
    """用真实 registry 打一遍：这两类异常只在磁盘与索引不一致时出现，
    而它们正是「数据悄悄读不到」的唯一线索。"""

    def test_registry_accounts_marks_orphan_and_missing(self) -> None:
        import tempfile

        from hermes_fleet.registry import BotRegistry, bot_file_stem

        with tempfile.TemporaryDirectory() as d:
            reg = BotRegistry(d)
            reg.get("bot-a")                      # 建库并登记索引
            reg.close_all()
            self.assertTrue(reg.has_account("bot-a"))
            self.assertEqual(len(reg.accounts()), 1)
            self.assertFalse(reg.accounts()[0]["orphan"])

            # 手工丢一个没有索引的库文件进去
            stem = bot_file_stem("bot-ghost")
            with open(f"{d}/{stem}.db", "wb") as fp:
                fp.write(b"SQLite format 3\x00" + b"\x00" * 100)
            rows = {r["platform_id"]: r for r in reg.accounts()}
            ghost = [r for r in reg.accounts() if r["orphan"]]
            self.assertEqual(len(ghost), 1)
            self.assertEqual(ghost[0]["stem"], stem)
            self.assertEqual(ghost[0]["platform_id"], "")

            # 索引里有、文件被删
            import os
            os.remove(f"{d}/{bot_file_stem('bot-a')}.db")
            rows = [r for r in reg.accounts() if r["missing"]]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["platform_id"], "bot-a")

    def test_has_account_blocks_new_db_creation(self) -> None:
        """未登记的 id 直接 get() 会凭空新建库——页面「打开空账号」就是这么来的。"""
        import tempfile

        from hermes_fleet.registry import BotRegistry

        with tempfile.TemporaryDirectory() as d:
            reg = BotRegistry(d)
            self.assertFalse(reg.has_account("never-seen"))
            reg.get("never-seen")                 # 手滑打开了一次
            self.assertTrue(reg.has_account("never-seen"), "确实会被登记")
            # 不关掉的话 Windows 上临时目录删不掉（文件被占用），
            # 报出来的 PermissionError 会盖住真正的断言结果。
            reg.close_all()


if __name__ == "__main__":
    unittest.main(verbosity=2)
