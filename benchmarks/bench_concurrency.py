"""SQLite 在「100 个机器人」规模下的并发行为实测。

要回答的问题不是「能不能写」，而是：
  Q1 单库 + bot_id 列，100 个写入者争用会怎样？
  Q2 每 bot 一库，100 个写入者几乎无争用，代价是什么？
  Q3 WAL 下读者会不会被写者阻塞？
  Q4 库变大后，启动期的全表扫描有多贵？

用法：python benchmarks/bench_concurrency.py
"""

from __future__ import annotations

import os
import random
import shutil
import sqlite3
import statistics
import tempfile
import threading
import time
from pathlib import Path

_ZH = (
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
)


def _rand_text(rng: random.Random, n: int = 40) -> str:
    return "".join(rng.choice(_ZH) for _ in range(n))


DDL = """
create table if not exists messages (
  id integer primary key autoincrement,
  bot_id text not null,
  session_id text not null,
  owner_id text not null,
  role text not null,
  content text not null,
  ts real not null
);
create index if not exists idx_msg_bot_ts on messages(bot_id, ts);
"""


def make_db(path: str, *, per_bot: bool) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    con.execute("pragma journal_mode=WAL")
    con.execute("pragma synchronous=NORMAL")
    con.execute("pragma busy_timeout=5000")
    con.execute("pragma cache_size=-16000")
    con.executescript(DDL)
    if per_bot:
        con.execute("create index if not exists idx_session on messages(session_id, ts)")
    return con


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {}
    xs = sorted(xs)
    n = len(xs)

    def q(p: float) -> float:
        return round(xs[min(n - 1, int(n * p))], 2)

    return {
        "n": n,
        "p50": q(0.50),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": round(xs[-1], 2),
        "mean": round(statistics.mean(xs), 2),
    }


# ------------------------------------------------------- Q1/Q2 争用


def run_shared(n_bots: int, per_bot_rows: int, writers: int) -> dict:
    """单库 + bot_id 列：N 个写入线程共享一个库。"""
    tmp = tempfile.mkdtemp(prefix="bsh_")
    path = os.path.join(tmp, "shared.db")
    try:
        make_db(path, per_bot=False)
        sql = "insert into messages(bot_id,session_id,owner_id,role,content,ts) values (?,?,?,?,?,?)"
        lat: list[float] = []
        busy = [0]
        lock = threading.Lock()
        work = [(b, r) for b in range(n_bots) for r in range(per_bot_rows)]
        chunks = [work[i::writers] for i in range(writers)]

        def worker(chunk: list[tuple[int, int]]) -> None:
            con = make_db(path, per_bot=False)
            rng = random.Random()
            local: list[float] = []
            local_busy = 0
            # 每 20 条提交一次，模拟真实批处理区间
            buf = []
            for bot, r in chunk:
                buf.append(
                    (
                        "bot%03d" % bot,
                        "sess%05d" % (bot * 100 + (r % 40)),
                        "u%05d" % (r % 3000),
                        "user",
                        _rand_text(rng, 30),
                        1700000000.0 + r,
                    )
                )
                if len(buf) >= 20:
                    t0 = time.perf_counter()
                    try:
                        con.execute("begin immediate")
                        con.executemany(sql, buf)
                        con.execute("commit")
                    except sqlite3.OperationalError as e:
                        local_busy += 1
                        try:
                            con.execute("rollback")
                        except sqlite3.Error:
                            pass
                        _ = e
                    local.append((time.perf_counter() - t0) * 1000)
                    buf = []
            if buf:
                try:
                    con.execute("begin immediate")
                    con.executemany(sql, buf)
                    con.execute("commit")
                except sqlite3.OperationalError:
                    local_busy += 1
            con.close()
            with lock:
                lat.extend(local)
                busy[0] += local_busy

        t0 = time.perf_counter()
        ts = [threading.Thread(target=worker, args=(c,)) for c in chunks]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        dt = time.perf_counter() - t0
        total = n_bots * per_bot_rows
        size = os.path.getsize(path)
        return {
            "mode": f"shared/{writers}w",
            "rows": total,
            "secs": round(dt, 2),
            "rows_per_sec": int(total / dt),
            "commit_latency_ms": _stats(lat),
            "busy_errors": busy[0],
            "db_mb": round(size / 1048576, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_per_bot(n_bots: int, per_bot_rows: int) -> dict:
    """每 bot 一库：n_bots 个独立连接并行写。"""
    tmp = tempfile.mkdtemp(prefix="bpb_")
    try:
        sql = "insert into messages(bot_id,session_id,owner_id,role,content,ts) values (?,?,?,?,?,?)"
        lat: list[float] = []
        busy = [0]
        lock = threading.Lock()

        def worker(bot: int) -> None:
            path = os.path.join(tmp, "bot%03d.db" % bot)
            con = make_db(path, per_bot=True)
            rng = random.Random(bot)
            local: list[float] = []
            local_busy = 0
            buf = []
            for r in range(per_bot_rows):
                buf.append(
                    (
                        "bot%03d" % bot,
                        "sess%05d" % (bot * 100 + (r % 40)),
                        "u%05d" % (r % 3000),
                        "user",
                        _rand_text(rng, 30),
                        1700000000.0 + r,
                    )
                )
                if len(buf) >= 20:
                    t0 = time.perf_counter()
                    try:
                        con.execute("begin immediate")
                        con.executemany(sql, buf)
                        con.execute("commit")
                    except sqlite3.OperationalError:
                        local_busy += 1
                        try:
                            con.execute("rollback")
                        except sqlite3.Error:
                            pass
                    local.append((time.perf_counter() - t0) * 1000)
                    buf = []
            if buf:
                try:
                    con.execute("begin immediate")
                    con.executemany(sql, buf)
                    con.execute("commit")
                except sqlite3.OperationalError:
                    local_busy += 1
            con.close()
            with lock:
                lat.extend(local)
                busy[0] += local_busy

        t0 = time.perf_counter()
        ts = [threading.Thread(target=worker, args=(b,)) for b in range(n_bots)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        dt = time.perf_counter() - t0
        total = n_bots * per_bot_rows
        size = sum(
            os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp)
        )
        return {
            "mode": f"perbot/{n_bots}f",
            "rows": total,
            "secs": round(dt, 2),
            "rows_per_sec": int(total / dt),
            "commit_latency_ms": _stats(lat),
            "busy_errors": busy[0],
            "db_mb": round(size / 1048576, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------- Q3 读写阻塞


def run_reader_vs_writer() -> dict:
    """一个持续写入线程 + 8 个读取线程，验证 WAL 的读不阻塞。"""
    tmp = tempfile.mkdtemp(prefix="brw_")
    path = os.path.join(tmp, "rw.db")
    try:
        con = make_db(path, per_bot=False)
        rng = random.Random(3)
        sql = "insert into messages(bot_id,session_id,owner_id,role,content,ts) values (?,?,?,?,?,?)"
        con.execute("begin immediate")
        con.executemany(
            sql,
            [
                ("bot001", "s%03d" % (i % 50), "u1", "user", _rand_text(rng, 40), 1700000000.0 + i)
                for i in range(20000)
            ],
        )
        con.execute("commit")
        con.close()

        stop = threading.Event()
        read_lat: list[float] = []
        rlock = threading.Lock()

        def writer() -> None:
            c = make_db(path, per_bot=False)
            r = random.Random(9)
            i = 0
            while not stop.is_set():
                try:
                    c.execute("begin immediate")
                    c.executemany(
                        sql,
                        [
                            (
                                "bot001",
                                "s%03d" % (i % 50),
                                "u1",
                                "user",
                                _rand_text(r, 40),
                                1700000000.0 + 30000 + i,
                            )
                            for i in range(20)
                        ],
                    )
                    c.execute("commit")
                except sqlite3.Error:
                    try:
                        c.execute("rollback")
                    except sqlite3.Error:
                        pass
                i += 20
            c.close()

        def reader() -> None:
            c = make_db(path, per_bot=False)
            local = []
            r = random.Random()
            while not stop.is_set():
                bot = "bot%03d" % r.randrange(1, 2)
                t0 = time.perf_counter()
                c.execute(
                    "select id,content,ts from messages where bot_id=? "
                    "order by ts desc limit 20",
                    (bot,),
                ).fetchall()
                local.append((time.perf_counter() - t0) * 1000)
            c.close()
            with rlock:
                read_lat.extend(local)

        wt = threading.Thread(target=writer)
        rts = [threading.Thread(target=reader) for _ in range(8)]
        wt.start()
        for t in rts:
            t.start()
        time.sleep(6.0)
        stop.set()
        for t in rts:
            t.join()
        wt.join()
        return {"readers": 8, "read_latency_ms": _stats(read_lat)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------- Q4 启动期扫描


def run_startup_scan(n_bots: int, rows_per_bot: int, shared: bool) -> dict:
    """模拟 hermes_core 启动时的 `select * from memories` 全表审计。"""
    tmp = tempfile.mkdtemp(prefix="bss_")
    try:
        rng = random.Random(5)
        ddl_mem = """
        create table if not exists memories (
          id integer primary key autoincrement,
          bot_id text not null,
          owner_id text not null,
          group_id text not null default '',
          content text not null,
          merged_into integer not null default 0
        );
        """
        total = 0
        paths = []
        if shared:
            path = os.path.join(tmp, "all.db")
            con = make_db(path, per_bot=False)
            con.executescript(ddl_mem)
            con.execute("begin immediate")
            for b in range(n_bots):
                con.executemany(
                    "insert into memories(bot_id,owner_id,group_id,content,merged_into) values (?,?,?,?,0)",
                    [
                        ("bot%03d" % b, "u%05d" % (i % 2000), "", _rand_text(rng, 50))
                        for i in range(rows_per_bot)
                    ],
                )
            con.execute("commit")
            con.close()
            paths = [path]
            total = n_bots * rows_per_bot
        else:
            for b in range(n_bots):
                path = os.path.join(tmp, "b%03d.db" % b)
                con = make_db(path, per_bot=True)
                con.executescript(ddl_mem)
                con.execute("begin immediate")
                con.executemany(
                    "insert into memories(bot_id,owner_id,group_id,content,merged_into) values (?,?,?,?,0)",
                    [
                        ("bot%03d" % b, "u%05d" % (i % 2000), "", _rand_text(rng, 50))
                        for i in range(rows_per_bot)
                    ],
                )
                con.execute("commit")
                con.close()
                paths.append(path)
            total = n_bots * rows_per_bot

        # 冷扫描：新连接，逐 bot 全表读
        t0 = time.perf_counter()
        n = 0
        for p in paths:
            c = sqlite3.connect(p, timeout=30.0, isolation_level=None)
            c.execute("pragma cache_size=-16000")
            for row in c.execute("select id,bot_id,owner_id from memories"):
                n += 1
            c.close()
        dt = time.perf_counter() - t0
        size = sum(os.path.getsize(p) for p in paths)
        return {
            "mode": "shared" if shared else "perbot",
            "files": len(paths),
            "rows": n,
            "scan_secs": round(dt, 2),
            "rows_per_sec": int(n / dt),
            "total_mb": round(size / 1048576, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 90)
    print("Q1/Q2 写入争用：单库共享 vs 每 bot 一库（100 bot × 2000 条 = 20 万条）")
    print("=" * 90)
    for r in (
        run_shared(100, 2000, writers=8),
        run_shared(100, 2000, writers=100),
        run_per_bot(100, 2000),
    ):
        print(f"{r['mode']:>14} {r['rows']:>7} 行 {r['secs']:>6}s "
              f"{r['rows_per_sec']:>7} 行/s  busy={r['busy_errors']:<4} {r['db_mb']}MB")
        print(f"{'':>14} 提交延迟 {r['commit_latency_ms']}")

    print()
    print("=" * 90)
    print("Q3 WAL 下 8 读者 vs 1 写者（持续 6 秒）")
    print("=" * 90)
    print(run_reader_vs_writer())

    print()
    print("=" * 90)
    print("Q4 启动期全表扫描（20 万条 memories）")
    print("=" * 90)
    print(run_startup_scan(100, 2000, shared=True))
    print(run_startup_scan(100, 2000, shared=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
