"""三个补充测量，都针对 hermes_core 的具体代码路径。

  S1 启动期 N+1 查询的真实代价（对应 db.py:_repair_orphan_private）
  S2 稳态写入：100 bot 每秒各 1 条（真实负载），而不是同时狂写
  S3 FTS5 检索在 50 万行时的退化情况

用法：python benchmarks/bench_tail.py
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

_ZH = (
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
)


def _txt(rng: random.Random, n: int = 40) -> str:
    return "".join(rng.choice(_ZH) for _ in range(n))


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 3)


# --------------------------------------------------------------- S1


def bench_n_plus_one(rows: int) -> dict:
    tmp = tempfile.mkdtemp(prefix="sn1_")
    try:
        path = os.path.join(tmp, "d.db")
        con = sqlite3.connect(path, isolation_level=None)
        con.execute("pragma journal_mode=WAL")
        con.execute("pragma synchronous=NORMAL")
        con.executescript(
            """
            create table messages(id integer primary key, session_id text, role text);
            create table memories(
              id integer primary key, owner_id text, group_id text,
              scope text, content text, merged_into integer default 0
            );
            create index idx_m on memories(owner_id);
            """
        )
        rng = random.Random(1)
        con.execute("begin immediate")
        con.executemany(
            "insert into messages(session_id,role) values (?,?)",
            [("s%05d" % (i % 5000), "user") for i in range(rows)],
        )
        con.executemany(
            "insert into memories(owner_id,group_id,scope,content) values (?,?,?,?)",
            [
                (
                    "u%05d" % (i % 2000),
                    "" if i % 3 else "g%03d" % (i % 50),
                    "private",
                    _txt(rng, 50),
                )
                for i in range(rows)
            ],
        )
        con.execute("commit")

        # 方式 A：全表扫 + 每行一次 messages 查询（hermes_core 的写法）
        t0 = time.perf_counter()
        n = 0
        for mid, sc, gid, own in con.execute(
            "select id,scope,group_id,owner_id from memories where scope='private' and group_id<>''"
        ):
            con.execute("select 1 from messages where session_id=? limit 1", (own,)).fetchone()
            n += 1
        dt_a = time.perf_counter() - t0

        # 方式 B：纯全表扫（不查 messages）
        t0 = time.perf_counter()
        for _ in con.execute("select id,scope,group_id,owner_id from memories where scope='private'"):
            pass
        dt_b = time.perf_counter() - t0

        # 方式 C：单条 update 全表改（_normalize_private_scope 的写法）
        t0 = time.perf_counter()
        con.execute("update memories set scope='private' where scope='private' and group_id<>''")
        dt_c = time.perf_counter() - t0
        con.close()
        return {
            "rows": rows,
            "n_plus_1_secs": round(dt_a, 2),
            "n_plus_1_rows": n,
            "plain_scan_secs": round(dt_b, 3),
            "bulk_update_secs": round(dt_c, 3),
            "slowdown_x": round(dt_a / max(dt_b, 1e-6), 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------- S2


def bench_steady_state(n_bots: int, seconds: float, per_sec: int) -> dict:
    """100 bot 各自独立库，每秒各写 per_sec 条，持续 seconds 秒。

    这是真实负载：写入分散在时间轴上，不是一个 bot 突然灌 2000 条。
    """
    tmp = tempfile.mkdtemp(prefix="sst_")
    try:
        ddl = (
            "create table messages(id integer primary key autoincrement,"
            "bot_id text, session_id text, role text, content text, ts real);"
            "create index idx_s on messages(session_id, ts);"
        )
        sql = "insert into messages(bot_id,session_id,role,content,ts) values (?,?,?,?,?)"
        lat: list[float] = []
        lock = threading.Lock()
        stop = threading.Event()
        total = [0]

        def bot_worker(b: int) -> None:
            path = os.path.join(tmp, "b%03d.db" % b)
            con = sqlite3.connect(path, timeout=30.0, isolation_level=None)
            con.execute("pragma journal_mode=WAL")
            con.execute("pragma synchronous=NORMAL")
            con.execute("pragma busy_timeout=5000")
            con.executescript(ddl)
            rng = random.Random(b)
            local: list[float] = []
            i = 0
            interval = 1.0 / per_sec
            next_at = time.perf_counter()
            while not stop.is_set():
                t0 = time.perf_counter()
                con.execute(
                    sql,
                    ("bot%03d" % b, "s%03d" % (i % 40), "user", _txt(rng, 30), time.time()),
                )
                local.append((time.perf_counter() - t0) * 1000)
                i += 1
                next_at += interval
                sleep = next_at - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
            con.close()
            with lock:
                lat.extend(local)
                total[0] += i

        ts = [threading.Thread(target=bot_worker, args=(b,)) for b in range(n_bots)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        time.sleep(seconds)
        stop.set()
        for t in ts:
            t.join()
        dt = time.perf_counter() - t0
        size = sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp))
        return {
            "bots": n_bots,
            "secs": round(dt, 1),
            "rows": total[0],
            "rows_per_sec": round(total[0] / dt, 1),
            "write_latency_ms": {
                "p50": _pct(lat, 0.50),
                "p95": _pct(lat, 0.95),
                "p99": _pct(lat, 0.99),
                "max": round(max(lat), 2),
            },
            "disk_mb": round(size / 1048576, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------- S3


def bench_fts_scale(n: int, rounds: int = 40) -> dict:
    tmp = tempfile.mkdtemp(prefix="sfs_")
    try:
        path = os.path.join(tmp, "d.db")
        con = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        con.execute("pragma journal_mode=WAL")
        con.execute("pragma synchronous=OFF")  # 灌数据阶段，不测持久性
        con.executescript(
            """
            create table memories(id integer primary key autoincrement,
              bot_id text, owner_id text, group_id text, content text, ts real);
            create virtual table mem_fts using fts5(content, content='memories',
              content_rowid='id', tokenize='trigram');
            create trigger ai after insert on memories begin
              insert into mem_fts(rowid,content) values (new.id,new.content); end;
            """
        )
        rng = random.Random(2)
        batch = []
        con.execute("begin immediate")
        for i in range(n):
            batch.append(
                (
                    "bot%03d" % rng.randrange(100),
                    "u%05d" % rng.randrange(50000),
                    "" if rng.random() < 0.7 else "g%03d" % rng.randrange(500),
                    _txt(rng, 45),
                    1700000000.0 + i,
                )
            )
            if len(batch) >= 5000:
                con.executemany(
                    "insert into memories(bot_id,owner_id,group_id,content,ts) values (?,?,?,?,?)",
                    batch,
                )
                batch = []
        if batch:
            con.executemany(
                "insert into memories(bot_id,owner_id,group_id,content,ts) values (?,?,?,?,?)",
                batch,
            )
        con.execute("commit")
        con.execute("pragma wal_checkpoint(TRUNCATE)")
        size = os.path.getsize(path)

        rows = con.execute("select content from memories limit 300").fetchall()
        qs = []
        for (c,) in rows:
            s = rng.randrange(0, max(1, len(c) - 3))
            qs.append(c[s : s + 3])
        qs = qs[:rounds]

        lat = []
        for q in qs:
            t0 = time.perf_counter()
            con.execute(
                "select rowid from mem_fts where mem_fts match ? order by rank limit 100",
                ('"%s"' % q,),
            ).fetchall()
            lat.append((time.perf_counter() - t0) * 1000)
        con.close()
        return {
            "rows": n,
            "db_mb": round(size / 1048576, 1),
            "bytes_per_row": int(size / n),
            "fts_ms": {
                "p50": _pct(lat, 0.50),
                "p95": _pct(lat, 0.95),
                "max": round(max(lat), 2),
            },
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 88)
    print("S1 启动期 N+1 查询 vs 纯扫描（对应 db.py:_repair_orphan_private）")
    print("=" * 88)
    for r in (bench_n_plus_one(50000), bench_n_plus_one(200000)):
        print(r)

    print()
    print("=" * 88)
    print("S2 稳态写入：100 bot 各每秒 2 条，持续 8 秒")
    print("=" * 88)
    print(bench_steady_state(100, 8.0, 2))

    print()
    print("=" * 88)
    print("S3 FTS5 检索随库增长的退化")
    print("=" * 88)
    for n in (100000, 500000):
        print(bench_fts_scale(n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
