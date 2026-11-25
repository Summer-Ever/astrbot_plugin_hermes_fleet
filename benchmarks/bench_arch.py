"""架构决策实测：共享单库 + 单写线程 + 队列，是否够用？

前一轮测出「单库 100 并发写」p99 = 563ms，但那是错误用法：
真实系统不该有 100 个线程抢一个库，而应该「单写者 + 队列」。这里对比三种形态：

  M1 共享库 + N 个写线程（错误用法，作对照）
  M2 共享库 + 1 个写线程 + 队列（推荐形态）
  M3 每 bot 一库 + 每库 1 个写线程

负载用真实值：100 bot，各自每秒 2 条（=200 条/秒，约 1700 万条/天，
是「每天 6 万条」需求的 280 倍，故意压满以留余量）。
指标是**入队到落库的端到端延迟**，这才是业务能感知的东西。

用法：python benchmarks/bench_arch.py
"""

from __future__ import annotations

import os
import queue
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

DDL = """
create table if not exists messages(
  id integer primary key autoincrement,
  bot_id text not null, session_id text not null, sender_id text not null,
  role text not null, content text not null, ts real not null
);
create index if not exists idx_s on messages(session_id, id);
create index if not exists idx_b on messages(bot_id, ts);
create virtual table if not exists messages_fts using fts5(
  content, content='messages', content_rowid='id', tokenize='trigram');
create trigger if not exists ai after insert on messages begin
  insert into messages_fts(rowid, content) values (new.id, new.content);
end;
"""

SQL = ("insert into messages(bot_id,session_id,sender_id,role,content,ts) "
       "values (?,?,?,?,?,?)")


def _txt(rng: random.Random, n: int = 30) -> str:
    return "".join(rng.choice(_ZH) for _ in range(n))


def _conn(path: str) -> sqlite3.Connection:
    c = sqlite3.connect(path, timeout=30.0, isolation_level=None,
                        check_same_thread=False)
    c.execute("pragma journal_mode=WAL")
    c.execute("pragma synchronous=NORMAL")
    c.execute("pragma busy_timeout=10000")
    c.execute("pragma cache_size=-16000")
    return c


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 2)


def _report(name: str, lat: list[float], secs: float, rows: int, mb: float) -> dict:
    return {
        "mode": name,
        "throughput": round(rows / secs, 0),
        "p50": _pct(lat, 0.50),
        "p95": _pct(lat, 0.95),
        "p99": _pct(lat, 0.99),
        "max": round(max(lat), 2) if lat else 0,
        "mb": round(mb, 1),
    }


# ---------------------------------------------------------------- M2


def run_single_writer(n_bots: int, secs: float, per_bot_per_sec: float,
                      commit_every: int = 20) -> dict:
    """共享库 + 1 写线程 + 队列。生产线程只入队（非阻塞），写线程批量提交。"""
    tmp = tempfile.mkdtemp(prefix="m2_")
    path = os.path.join(tmp, "shared.db")
    try:
        c = _conn(path)
        c.executescript(DDL)
        c.close()

        q: queue.Queue = queue.Queue(maxsize=100000)
        lat: list[float] = []
        stop = threading.Event()
        written = [0]

        def writer() -> None:
            con = _conn(path)
            buf: list[tuple] = []
            stamps: list[float] = []

            def flush() -> None:
                nonlocal buf, stamps
                if not buf:
                    return
                con.execute("begin immediate")
                try:
                    con.executemany(SQL, buf)
                    con.execute("commit")
                except sqlite3.Error:
                    con.execute("rollback")
                    buf, stamps = [], []
                    return
                now = time.perf_counter()
                for t_enq in stamps:
                    lat.append((now - t_enq) * 1000)
                written[0] += len(buf)
                buf, stamps = [], []

            while True:
                item = q.get()
                if item is None:
                    break
                buf.append(item)
                stamps.append(time.perf_counter())
                if len(buf) >= commit_every:
                    flush()
            flush()
            con.close()

        wt = threading.Thread(target=writer)
        wt.start()

        enq_lat: list[float] = []
        lock = threading.Lock()

        def bot(b: int) -> None:
            rng = random.Random(b)
            interval = 1.0 / per_bot_per_sec
            nxt = time.perf_counter()
            i = 0
            while not stop.is_set():
                row = ("bot%03d" % b, "s%03d" % (i % 40), "u%04d" % (b * 7 + i % 5),
                       "user", _txt(rng), time.time())
                t0 = time.perf_counter()
                q.put(row)
                d = (time.perf_counter() - t0) * 1000
                with lock:
                    enq_lat.append(d)
                i += 1
                nxt += interval
                s = nxt - time.perf_counter()
                if s > 0:
                    time.sleep(s)

        ts = [threading.Thread(target=bot, args=(b,)) for b in range(n_bots)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        time.sleep(secs)
        stop.set()
        for t in ts:
            t.join()
        q.put(None)
        wt.join()
        elapsed = time.perf_counter() - t0
        mb = os.path.getsize(path) / 1048576
        r = _report(f"shared/1writer", lat, elapsed, written[0], mb)
        r["enqueue_p99"] = _pct(enq_lat, 0.99)
        return r
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- M1


def run_multi_writer(n_bots_threads: int, secs: float, per_thr_per_sec: float) -> dict:
    """共享库 + N 个写线程（错误用法对照）。"""
    tmp = tempfile.mkdtemp(prefix="m1_")
    path = os.path.join(tmp, "shared.db")
    try:
        c = _conn(path)
        c.executescript(DDL)
        c.close()
        lat: list[float] = []
        lock = threading.Lock()
        stop = threading.Event()
        written = [0]

        def worker(b: int) -> None:
            con = _conn(path)
            rng = random.Random(b)
            interval = 1.0 / per_thr_per_sec
            nxt = time.perf_counter()
            i = 0
            while not stop.is_set():
                t0 = time.perf_counter()
                try:
                    con.execute(SQL, ("bot%03d" % b, "s%03d" % (i % 40),
                                      "u%04d" % (b % 50), "user", _txt(rng), time.time()))
                    ok = True
                except sqlite3.Error:
                    ok = False
                d = (time.perf_counter() - t0) * 1000
                if ok:
                    with lock:
                        lat.append(d)
                        written[0] += 1
                i += 1
                nxt += interval
                s = nxt - time.perf_counter()
                if s > 0:
                    time.sleep(s)
            con.close()

        ts = [threading.Thread(target=worker, args=(b,)) for b in range(n_bots_threads)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        time.sleep(secs)
        stop.set()
        for t in ts:
            t.join()
        elapsed = time.perf_counter() - t0
        return _report(f"shared/{n_bots_threads}writer", lat, elapsed,
                       written[0], os.path.getsize(path) / 1048576)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- M3


def run_per_bot(n_bots: int, secs: float, per_bot_per_sec: float) -> dict:
    """每 bot 一库 + 每库 1 写线程。"""
    tmp = tempfile.mkdtemp(prefix="m3_")
    try:
        lat: list[float] = []
        lock = threading.Lock()
        stop = threading.Event()
        written = [0]

        def bot(b: int) -> None:
            path = os.path.join(tmp, "b%03d.db" % b)
            con = _conn(path)
            con.executescript(DDL)
            rng = random.Random(b)
            interval = 1.0 / per_bot_per_sec
            nxt = time.perf_counter()
            i = 0
            local = []
            while not stop.is_set():
                t0 = time.perf_counter()
                try:
                    con.execute(SQL, ("bot%03d" % b, "s%03d" % (i % 40),
                                      "u%04d" % (b % 50), "user", _txt(rng), time.time()))
                except sqlite3.Error:
                    pass
                local.append((time.perf_counter() - t0) * 1000)
                i += 1
                nxt += interval
                s = nxt - time.perf_counter()
                if s > 0:
                    time.sleep(s)
            con.close()
            with lock:
                lat.extend(local)
                written[0] += i

        ts = [threading.Thread(target=bot, args=(b,)) for b in range(n_bots)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        time.sleep(secs)
        stop.set()
        for t in ts:
            t.join()
        elapsed = time.perf_counter() - t0
        mb = sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp)) / 1048576
        return _report(f"perbot/{n_bots}", lat, elapsed, written[0], mb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    bots, secs, rate = 100, 8.0, 2.0
    print("=" * 92)
    print(f"负载：{bots} bot × {rate} 条/s × {secs}s = {int(bots * rate * secs)} 条"
          f"（{int(bots * rate * 86400):,} 条/天，是 6 万/天需求的 "
          f"{int(bots * rate * 86400 / 60000)} 倍）")
    print("=" * 92)
    print(f"{'形态':>20} {'吞吐/s':>8} {'p50':>7} {'p95':>8} {'p99':>8} {'max':>8} {'MB':>6}")
    rows = []
    for fn in (
        # 8 线程 × 25 条/s 与 100 线程 × 2 条/s 都是 200 条/s：
        # 同一个总负载下，只改「写者数量」，隔离出争用这一个变量
        lambda: run_multi_writer(8, secs, 25.0),
        lambda: run_multi_writer(100, secs, rate),
        lambda: run_single_writer(bots, secs, rate),
        lambda: run_per_bot(bots, secs, rate),
    ):
        r = fn()
        rows.append(r)
        extra = f"  入队p99={r['enqueue_p99']}ms" if "enqueue_p99" in r else ""
        print(f"{r['mode']:>20} {r['throughput']:>8.0f} {r['p50']:>7} {r['p95']:>8} "
              f"{r['p99']:>8} {r['max']:>8} {r['mb']:>6}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
