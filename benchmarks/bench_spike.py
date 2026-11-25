"""定位单写线程下的 p99 尖刺来源。

现象：shared/1writer 的 p95 = 4ms 但 p99 = 501ms，断层明显。
假设：WAL 自动 checkpoint（默认 wal_autocheckpoint=1000 页）在写事务中同步搬运页面，
     造成周期性停顿。若成立，关掉自动 checkpoint 并手动调度，尖刺应消失。

验证方式：记录每条延迟的时间戳，看尖刺是否周期性出现、是否与 checkpoint 对应。

用法：python benchmarks/bench_spike.py
"""

from __future__ import annotations

import os
import queue
import random
import shutil
import sqlite3
import tempfile
import threading
import time

_ZH = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"

DDL = """
create table if not exists messages(
  id integer primary key autoincrement,
  bot_id text, session_id text, role text, content text, ts real);
"""

SQL = "insert into messages(bot_id,session_id,role,content,ts) values (?,?,?,?,?)"


def _txt(rng: random.Random) -> str:
    return "".join(rng.choice(_ZH) for _ in range(30))


def run(autockpt: int, secs: float = 10.0, rate: float = 200.0,
        commit_every: int = 20) -> dict:
    tmp = tempfile.mkdtemp(prefix="spk_")
    path = os.path.join(tmp, "d.db")
    try:
        c = sqlite3.connect(path, isolation_level=None)
        c.execute("pragma journal_mode=WAL")
        c.execute(f"pragma wal_autocheckpoint={autockpt}")
        c.execute("pragma synchronous=NORMAL")
        c.executescript(DDL)
        c.close()

        q: queue.Queue = queue.Queue()
        marks: list[tuple[float, float]] = []  # (提交完成时刻, 延迟ms)
        ckpt: list[float] = []
        stop = threading.Event()
        t_start = time.perf_counter()

        def writer() -> None:
            con = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            con.execute("pragma journal_mode=WAL")
            con.execute(f"pragma wal_autocheckpoint={autockpt}")
            con.execute("pragma synchronous=NORMAL")
            buf, stamps = [], []
            while True:
                item = q.get()
                if item is None:
                    break
                buf.append(item)
                stamps.append(time.perf_counter())
                if len(buf) < commit_every:
                    continue
                con.execute("begin immediate")
                con.executemany(SQL, buf)
                con.execute("commit")
                now = time.perf_counter()
                for s in stamps:
                    marks.append((now - t_start, (now - s) * 1000))
                if autockpt == 0 and len(marks) % 400 < commit_every:
                    t0 = time.perf_counter()
                    con.execute("pragma wal_checkpoint(PASSIVE)")
                    ckpt.append((time.perf_counter() - t0) * 1000)
                buf, stamps = [], []
            if buf:
                con.execute("begin immediate")
                con.executemany(SQL, buf)
                con.execute("commit")
            con.close()

        wt = threading.Thread(target=writer)
        wt.start()
        rng = random.Random(5)
        i = 0
        interval = 1.0 / rate
        nxt = time.perf_counter()
        while time.perf_counter() - t_start < secs:
            q.put(("bot%03d" % (i % 100), "s%03d" % (i % 40), "user", _txt(rng), time.time()))
            i += 1
            nxt += interval
            s = nxt - time.perf_counter()
            if s > 0:
                time.sleep(s)
        stop.set()
        q.put(None)
        wt.join()

        lat = sorted(x[1] for x in marks)
        n = len(lat)

        def q_(p: float) -> float:
            return round(lat[min(n - 1, int(n * p))], 2)

        spikes = [(round(t, 2), round(v, 1)) for t, v in marks if v > 100]
        return {
            "autockpt": autockpt,
            "rows": n,
            "p50": q_(0.50),
            "p95": q_(0.95),
            "p99": q_(0.99),
            "p999": q_(0.999),
            "max": round(lat[-1], 1),
            "spikes_gt100ms": len(spikes),
            "spike_marks": spikes[:8],
            "manual_ckpt_ms": [round(x, 2) for x in ckpt[:5]],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("每秒 200 条、每 20 条一提交，持续 10 秒（2000 条）")
    print("=" * 86)
    for auto in (1000, 0):
        r = run(auto)
        tag = "自动checkpoint(默认1000页)" if auto == 1000 else "关闭自动+手动调度"
        print(f"[{tag}]")
        print(f"  行数={r['rows']} p50={r['p50']} p95={r['p95']} p99={r['p99']} "
              f"p99.9={r['p999']} max={r['max']}")
        print(f"  >100ms 的尖刺: {r['spikes_gt100ms']} 次  {r['spike_marks']}")
        if r["manual_ckpt_ms"]:
            print(f"  手动 checkpoint 耗时: {r['manual_ckpt_ms']} ms")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
