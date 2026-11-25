"""SQLite 承载能力的实测基准。

设计原则：所有结论必须来自这台机器上的真实数字，不接受「一般认为」。
三类测量：
  A. 写入吞吐 —— 单库不同 synchronous / batch 配置下的 msg/s
  B. 索引体积 —— FTS5 trigram 在中文语料上的实际膨胀倍数
  C. 检索延迟 —— 数据量从 10 万到 100 万时的 p50/p95

用法：python benchmarks/bench_sqlite.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

# 让语料接近真实中文聊天：长度 8-60 字，含常用字与少量英文数字
_ZH = (
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
)


def _rand_text(rng: random.Random, n: int | None = None) -> str:
    n = n or rng.randint(8, 60)
    return "".join(rng.choice(_ZH) for _ in range(n))


SCHEMA = """
create table messages (
  id          integer primary key autoincrement,
  bot_id      text not null,
  session_id  text not null,
  owner_id    text not null,
  group_id    text not null default '',
  role        text not null,
  content     text not null,
  ts          real not null
);
create index idx_msg_session_ts on messages(bot_id, session_id, ts);

create table memories (
  id          integer primary key autoincrement,
  bot_id      text not null,
  owner_id    text not null,
  group_id    text not null default '',
  content     text not null,
  kind        text not null default 'fact',
  ts          real not null,
  merged_into integer not null default 0
);
create index idx_mem_scope on memories(bot_id, owner_id, group_id, merged_into);

create virtual table mem_fts using fts5(
  content,
  content='memories',
  content_rowid='id',
  tokenize='trigram'
);
"""

FTS_TRIGGERS = """
create trigger mem_ai after insert on memories begin
  insert into mem_fts(rowid, content) values (new.id, new.content);
end;
create trigger mem_ad after delete on memories begin
  insert into mem_fts(mem_fts, rowid, content) values ('delete', old.id, old.content);
end;
create trigger mem_au after update on memories begin
  insert into mem_fts(mem_fts, rowid, content) values ('delete', old.id, old.content);
  insert into mem_fts(rowid, content) values (new.id, new.content);
end;
"""


def _connect(path: str, *, synchronous: str = "NORMAL") -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    con.execute("pragma journal_mode=WAL")
    con.execute(f"pragma synchronous={synchronous}")
    con.execute("pragma temp_store=MEMORY")
    con.execute("pragma cache_size=-32000")  # 32MB page cache
    con.execute("pragma mmap_size=268435456")  # 256MB
    return con


# ---------------------------------------------------------------- A. 写入吞吐


def bench_write(rows: int, batch: int, synchronous: str, with_fts: bool) -> dict:
    tmp = tempfile.mkdtemp(prefix="benchw_")
    path = os.path.join(tmp, "t.db")
    try:
        con = _connect(path, synchronous=synchronous)
        con.executescript(SCHEMA)
        if with_fts:
            con.executescript(FTS_TRIGGERS)

        rng = random.Random(42)
        payload = [
            (
                "bot%03d" % rng.randrange(100),
                "sess%04d" % rng.randrange(2000),
                "u%05d" % rng.randrange(50000),
                "" if rng.random() < 0.6 else "g%04d" % rng.randrange(500),
                rng.choice(("user", "assistant")),
                _rand_text(rng),
                1700000000.0 + i,
            )
            for i in range(rows)
        ]

        sql = "insert into messages(bot_id,session_id,owner_id,group_id,role,content,ts) values (?,?,?,?,?,?,?)"
        t0 = time.perf_counter()
        if batch <= 1:
            for row in payload:
                con.execute(sql, row)
        else:
            for i in range(0, rows, batch):
                con.execute("begin immediate")
                con.executemany(sql, payload[i : i + batch])
                con.execute("commit")
        dt = time.perf_counter() - t0
        size = os.path.getsize(path)
        # WAL 文件也计入真实占用
        wal = path + "-wal"
        if os.path.exists(wal):
            size += os.path.getsize(wal)
        con.close()

        return {
            "rows": rows,
            "batch": batch,
            "sync": synchronous,
            "fts": with_fts,
            "secs": round(dt, 3),
            "rows_per_sec": int(rows / dt) if dt else 0,
            "bytes_per_row": int(size / rows) if rows else 0,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- B. 索引体积


def bench_fts_size(n_mem: int) -> dict:
    tmp = tempfile.mkdtemp(prefix="benchf_")
    path = os.path.join(tmp, "t.db")
    try:
        con = _connect(path)
        con.executescript(SCHEMA)
        rng = random.Random(7)
        rows = [
            (
                "bot%03d" % rng.randrange(100),
                "u%05d" % rng.randrange(50000),
                "",
                _rand_text(rng),
                "fact",
                1700000000.0 + i,
                0,
            )
            for i in range(n_mem)
        ]
        # 先只灌表，量原始体积
        con.execute("begin immediate")
        con.executemany(
            "insert into memories(bot_id,owner_id,group_id,content,kind,ts,merged_into) values (?,?,?,?,?,?,?)",
            rows,
        )
        con.execute("commit")
        con.execute("pragma wal_checkpoint(TRUNCATE)")
        base = os.path.getsize(path)

        # 再建 FTS 与触发器，量增量
        con.executescript(FTS_TRIGGERS)
        con.execute("insert into mem_fts(mem_fts) values ('rebuild')")
        con.execute("begin immediate")
        con.executemany(
            "insert into memories(bot_id,owner_id,group_id,content,kind,ts,merged_into) values (?,?,?,?,?,?,?)",
            rows[: n_mem // 10],
        )
        con.execute("commit")
        con.execute("pragma wal_checkpoint(TRUNCATE)")
        full = os.path.getsize(path)

        # 实测中文平均字数
        avg_chars = statistics.mean(len(r[3]) for r in rows)
        con.close()
        return {
            "n": n_mem,
            "avg_chars": round(avg_chars, 1),
            "base_mb": round(base / 1048576, 2),
            "with_fts_mb": round(full / 1048576, 2),
            "fts_overhead_ratio": round((full - base) / base, 2),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- C. 检索延迟


def bench_query(n_mem: int, rounds: int = 60) -> dict:
    tmp = tempfile.mkdtemp(prefix="benchq_")
    path = os.path.join(tmp, "t.db")
    try:
        con = _connect(path)
        con.executescript(SCHEMA)
        con.executescript(FTS_TRIGGERS)
        rng = random.Random(11)
        rows = [
            (
                "bot%03d" % rng.randrange(100),
                "u%05d" % rng.randrange(50000),
                "" if rng.random() < 0.7 else "g%04d" % rng.randrange(500),
                _rand_text(rng),
                "fact",
                1700000000.0 + i,
                0,
            )
            for i in range(n_mem)
        ]
        con.execute("begin immediate")
        con.executemany(
            "insert into memories(bot_id,owner_id,group_id,content,kind,ts,merged_into) values (?,?,?,?,?,?,?)",
            rows,
        )
        con.execute("commit")
        con.execute("insert into mem_fts(mem_fts) values ('rebuild')")

        # 取一批真实存在的 3 字片段当查询词，避免全是空命中
        queries = []
        for r in rows[:200]:
            c = r[3]
            if len(c) >= 6:
                s = rng.randrange(0, len(c) - 3)
                queries.append(c[s : s + 3])
        queries = queries[:rounds]

        # 模拟真实场景：FTS 先粗排 top-N，再带 bot_id 过滤
        fts_ms, filt_ms = [], []
        for q in queries:
            t0 = time.perf_counter()
            got = con.execute(
                "select rowid from mem_fts where mem_fts match ? order by rank limit 100",
                ('"%s"' % q,),
            ).fetchall()
            fts_ms.append((time.perf_counter() - t0) * 1000)

            if got:
                ids = [g[0] for g in got]
                ph = ",".join("?" * len(ids))
                t0 = time.perf_counter()
                con.execute(
                    f"select id,content from memories where id in ({ph}) "
                    f"and bot_id=? and merged_into=0 limit 20",
                    (*ids, "bot007"),
                ).fetchall()
                filt_ms.append((time.perf_counter() - t0) * 1000)
        con.close()

        def pct(xs, p):
            if not xs:
                return 0.0
            xs = sorted(xs)
            return round(xs[min(len(xs) - 1, int(len(xs) * p))], 3)

        return {
            "n": n_mem,
            "fts_p50": pct(fts_ms, 0.50),
            "fts_p95": pct(fts_ms, 0.95),
            "filter_p50": pct(filt_ms, 0.50),
            "filter_p95": pct(filt_ms, 0.95),
            "hit_rate": round(len(filt_ms) / max(1, len(fts_ms)), 2),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rows = 30000 if args.quick else 120000

    print("=" * 74)
    print("A. 写入吞吐（单库，WAL）")
    print("=" * 74)
    print(f"{'batch':>6} {'sync':>8} {'fts':>5} {'rows/s':>9} {'bytes/row':>10}")
    for batch, sync, fts in (
        (1, "FULL", False),
        (1, "NORMAL", False),
        (50, "FULL", False),
        (50, "NORMAL", False),
        (500, "NORMAL", False),
        (500, "NORMAL", True),
    ):
        r = bench_write(rows, batch, sync, fts)
        print(
            f"{r['batch']:>6} {r['sync']:>8} {str(r['fts']):>5} "
            f"{r['rows_per_sec']:>9} {r['bytes_per_row']:>10}"
        )

    print()
    print("=" * 74)
    print("B. FTS5 trigram 索引体积（中文）")
    print("=" * 74)
    for n in (2000 if args.quick else 8000,):
        print(bench_fts_size(n))

    print()
    print("=" * 74)
    print("C. 检索延迟（FTS top-100 + 元数据过滤）")
    print("=" * 74)
    for n in (10000, 50000) if args.quick else (50000, 200000, 500000):
        print(bench_query(n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
