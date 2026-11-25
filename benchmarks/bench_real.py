"""用 hermes_core 自己的存储层实测，而不是用我模拟的 schema。

前两轮 benchmark 用了虚构的 DDL，结论失真（虚构 schema 缺索引，把一次索引查找
放大成了全表 N+1）。这个脚本直接 import 真实的 `modules/db.py`，测三件事：

  R1 冷启动耗时 —— 含全部增量迁移 + _repair_orphan_private + _normalize_private_scope
  R2 search_messages 的真实检索延迟随数据量退化
  R3 单条 add_message 的写入延迟（同步调用，含 FTS 触发器维护）

用法：python benchmarks/bench_real.py [--rows 50000]
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

_ZH = (
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
)

CORES = [
    Path("C:/Users/RdpOnly/WorkBuddy/2026-11-25-19-57-58/astrbot_plugin_hermes_core"),
]


def _load_core() -> object:
    """定位并导入真实存储层。找不到就明确失败，不静默降级。"""
    core = next((c for c in CORES if (c / "modules" / "db.py").exists()), None)
    if core is None:
        raise SystemExit("找不到 astrbot_plugin_hermes_core，无法做真实基线测试")
    sys.path.insert(0, str(core / "tests"))
    from _bootstrap import db_module  # noqa: PLC0415

    return db_module()


def _txt(rng: random.Random, n: int = 40) -> str:
    return "".join(rng.choice(_ZH) for _ in range(n))


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 2)


def seed(rows: int, n_sessions: int = 500, n_mem: int = 4000) -> tuple[str, str]:
    """建库并灌数据，返回 (datadir, dbname)。"""
    _db = _load_core()
    HermesDB = _db.HermesDB
    data_dir = tempfile.mkdtemp(prefix="realh_")
    db = HermesDB(data_dir, "bench.db")

    rng = random.Random(17)
    t0 = time.perf_counter()
    # 走真实 add_message，让 FTS 触发器同步维护
    for i in range(rows):
        sid = "sess%04d" % rng.randrange(n_sessions)
        db.add_message(sid, "user", _txt(rng, 30), sender_id="u%05d" % rng.randrange(2000))
    dt_write = time.perf_counter() - t0

    for i in range(n_mem):
        db.add_memory(
            "fact",
            _txt(rng, 45),
            scope="user",
            owner_id="u%05d" % rng.randrange(2000),
        )
    db.conn.commit()
    try:
        db.conn.execute("pragma wal_checkpoint(TRUNCATE)")
    except Exception:
        pass
    size = sum(
        os.path.getsize(os.path.join(data_dir, f)) for f in os.listdir(data_dir)
    )
    print(
        f"  建库 {rows} 条消息 + {n_mem} 条记忆："
        f"{dt_write:.1f}s（{rows / dt_write:.0f} 条/s）{size / 1048576:.1f}MB"
    )
    db.close()
    return data_dir, "bench.db", (rows / dt_write)


def measure_cold_start(data_dir: str, dbname: str, rounds: int = 3) -> dict:
    """重新构造 HermesDB —— 这就是 AstrBot 启动时的真实路径。"""
    _db = _load_core()
    HermesDB = _db.HermesDB
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        d = HermesDB(data_dir, dbname)
        d.count_messages()
        times.append((time.perf_counter() - t0) * 1000)
        d.close()
    return {
        "cold_start_ms": {"first": round(times[0], 1), "median": _pct(times, 0.5)},
    }


def measure_search(data_dir: str, dbname: str, rounds: int = 25) -> dict:
    _db = _load_core()
    HermesDB = _db.HermesDB
    d = HermesDB(data_dir, dbname)
    rng = random.Random(23)
    rows = d.conn.execute("select content from messages order by random() limit 400").fetchall()
    qs = []
    for (c,) in rows:
        # search_messages 在 _norm(query) < 4 字时直接 return []，
        # 用 3 字查询测出来的是「短路耗时」，不是检索耗时
        if len(c) >= 12:
            s = rng.randrange(0, len(c) - 6)
            qs.append(c[s : s + 6])
    qs = qs[:rounds]

    lat, hit = [], 0
    for q in qs:
        t0 = time.perf_counter()
        got = d.search_messages(q, limit=5)
        lat.append((time.perf_counter() - t0) * 1000)
        if got:
            hit += 1
    n = d.count_messages()
    d.close()
    return {
        "messages": n,
        "search_ms": {
            "p50": _pct(lat, 0.50),
            "p95": _pct(lat, 0.95),
            "max": round(max(lat), 2),
        },
        "hit_rate": round(hit / max(1, len(lat)), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    args = ap.parse_args()
    # smoke_db 的经验：换工作目录，避免运行时残渣落进仓库
    os.chdir(tempfile.mkdtemp(prefix="realh-cwd-"))

    print("=" * 78)
    print("R1/R2 真实 hermes_core 存储层")
    print("=" * 78)
    data_dir, name, wps = seed(args.rows)
    try:
        print("  冷启动:", measure_cold_start(data_dir, name))
        print("  检索:", measure_search(data_dir, name))
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
