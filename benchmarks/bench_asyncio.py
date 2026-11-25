"""按 AstrBot 的真实形态测：asyncio 单事件循环，100 个 bot 是 100 个协程。

前几轮用 100 个 Python 线程模拟 bot，测出来的 p99 里混进了 GIL 争用和
Windows 定时器误差，不是 SQLite 的账。真实形态是：
  - 100 个协程产消息 → asyncio.Queue（无锁、无 GIL 争用）
  - 1 个专用写线程消费队列、批量提交（写线程只做 SQL，不碰事件循环）
  - 读走主线程（同步执行，因为要阻塞回复路径，必须测量它）

指标分开看：
  write_* : 入队到落库（异步链路，几百 ms 都可接受）
  read_*  : 注入路径的同步读（这条必须快，它直接卡住 LLM 请求）

用法：python benchmarks/bench_asyncio.py
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import sqlite3
import statistics
import tempfile
import threading
import time
import queue as syncq

_ZH = "的一是了我不人在他有这个上们来到时大地为子中你说龙井国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"

DDL = """
create table messages(
  id integer primary key autoincrement,
  bot_id text not null, session_id text not null, role text not null,
  content text not null, ts real not null);
create index idx_s on messages(session_id, id);
create index idx_b on messages(bot_id, ts);
create table memories(
  id integer primary key autoincrement,
  bot_id text not null, owner_id text not null, group_id text default '',
  content text not null, ts real not null, merged_into integer default 0);
create index idx_m on memories(bot_id, owner_id, group_id, merged_into);
create virtual table mem_fts using fts5(
  content, content='memories', content_rowid='id', tokenize='trigram');
create trigger m_ai after insert on memories begin
  insert into mem_fts(rowid,content) values (new.id,new.content); end;
"""

INS_MSG = "insert into messages(bot_id,session_id,role,content,ts) values (?,?,?,?,?)"
INS_MEM = ("insert into memories(bot_id,owner_id,group_id,content,ts) "
           "values (?,?,?,?,?)")


def _txt(rng: random.Random, n: int = 30) -> str:
    return "".join(rng.choice(_ZH) for _ in range(n))


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 2)


def build(path: str, n_mem: int, n_msg: int) -> None:
    c = sqlite3.connect(path, isolation_level=None)
    c.execute("pragma journal_mode=WAL")
    c.execute("pragma synchronous=NORMAL")
    c.executescript(DDL)
    rng = random.Random(3)
    c.execute("begin immediate")
    c.executemany(
        INS_MEM,
        [("bot%03d" % rng.randrange(100), "u%04d" % rng.randrange(3000), "",
          _txt(rng, 45), 1700000000.0 + i) for i in range(n_mem)],
    )
    c.executemany(
        INS_MSG,
        [("bot%03d" % rng.randrange(100), "s%04d" % rng.randrange(400),
          "user", _txt(rng, 30), 1700000000.0 + i) for i in range(n_msg)],
    )
    c.execute("commit")
    c.execute("pragma wal_checkpoint(TRUNCATE)")
    c.close()


class WriteWorker:
    """唯一写者。只做 SQL，不做业务，保证写事务短且可控。"""

    def __init__(self, path: str, commit_every: int = 64, flush_ms: float = 50.0) -> None:
        self.q: syncq.Queue = syncq.Queue()
        self.lat: list[float] = []
        self.commits = 0
        self.rows_written = 0
        self.commit_ms: list[float] = []
        self._commit_every = commit_every
        self._flush_s = flush_ms / 1000.0
        self._t = threading.Thread(target=self._run, args=(path,), daemon=True)
        self._t.start()

    def _run(self, path: str) -> None:
        con = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        con.execute("pragma journal_mode=WAL")
        con.execute("pragma synchronous=NORMAL")
        con.execute("pragma busy_timeout=10000")
        buf, stamps = [], []

        def flush() -> None:
            nonlocal buf, stamps
            if not buf:
                return
            t0 = time.perf_counter()
            con.execute("begin immediate")
            con.executemany(INS_MSG, buf)
            con.execute("commit")
            self.commit_ms.append((time.perf_counter() - t0) * 1000)
            now = time.perf_counter()
            for s in stamps:
                self.lat.append((now - s) * 1000)
            self.commits += 1
            self.rows_written += len(buf)
            buf, stamps = [], []

        while True:
            # 双触发：满 N 条 或 距上次提交超过 T 毫秒，先到者提交。
            # 只按条数触发时，突发流量的「余数」会每一轮都多等一个周期，
            # 在 100 bot 同时发言的场景下会稳定制造 500ms 级的尾延迟。
            try:
                item = self.q.get(timeout=self._flush_s)
            except syncq.Empty:
                flush()
                continue
            if item is None:
                break
            t_enq, row = item
            buf.append(row)
            stamps.append(t_enq)
            if len(buf) >= self._commit_every:
                flush()
        flush()
        con.close()

    def put(self, row: tuple) -> None:
        self.q.put((time.perf_counter(), row))

    def close(self) -> None:
        self.q.put(None)
        self._t.join(timeout=30)


async def run(bots: int, per_bot_per_sec: float, secs: float,
              n_mem: int = 50000, read_rounds: int = 200) -> dict:
    tmp = tempfile.mkdtemp(prefix="aio_")
    path = os.path.join(tmp, "d.db")
    try:
        build(path, n_mem, n_mem // 2)
        worker = WriteWorker(path)

        stop = asyncio.Event()

        async def bot(b: int) -> None:
            rng = random.Random(b)
            interval = 1.0 / per_bot_per_sec
            i = 0
            while not stop.is_set():
                worker.put(("bot%03d" % b, "s%03d" % (i % 40), "user", _txt(rng), time.time()))
                i += 1
                await asyncio.sleep(interval)

        tasks = [asyncio.create_task(bot(b)) for b in range(bots)]
        t0 = time.perf_counter()
        await asyncio.sleep(secs)
        stop.set()
        await asyncio.gather(*tasks)
        # 排空期：让写线程把队列消化完再停。否则最后一批会在 stop 瞬间被记成
        # 500ms+ 的延迟，那是收尾伪影，不是稳态表现。
        await asyncio.sleep(2.0)
        worker.close()
        elapsed = time.perf_counter() - t0

        # ---- 读：注入路径要做的三件事
        con = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        con.execute("pragma cache_size=-32000")
        con.row_factory = sqlite3.Row
        rng = random.Random(31)
        reads: dict[str, list[float]] = {"resident": [], "skills": [], "fts": []}

        for _ in range(read_rounds):
            bot_id = "bot%03d" % rng.randrange(100)
            owner = "u%04d" % rng.randrange(3000)

            t = time.perf_counter()
            con.execute(
                "select id,content from memories where bot_id=? and owner_id=? "
                "and merged_into=0 order by ts desc limit 6", (bot_id, owner)).fetchall()
            reads["resident"].append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            con.execute(
                "select id,content from memories where bot_id=? and owner_id=? "
                "and merged_into=0 order by ts desc limit 12", (bot_id, owner)).fetchall()
            reads["skills"].append((time.perf_counter() - t) * 1000)

            q = _txt(rng, 6)
            t = time.perf_counter()
            con.execute(
                "select rowid from mem_fts where mem_fts match ? order by rank limit 100",
                ('"%s"' % q,)).fetchall()
            reads["fts"].append((time.perf_counter() - t) * 1000)
        n_msg = con.execute("select count(*) from messages").fetchone()[0]
        n_mem_actual = con.execute("select count(*) from memories").fetchone()[0]
        con.close()

        return {
            "messages": n_msg,
            "memories": n_mem_actual,
            "written": worker.rows_written,
            "write_throughput": round(worker.rows_written / elapsed),
            "write_lat": {"p50": _pct(worker.lat, .5), "p95": _pct(worker.lat, .95),
                          "p99": _pct(worker.lat, .99), "max": round(max(worker.lat), 1)},
            "commit_ms": {"p50": _pct(worker.commit_ms, .5),
                          "p95": _pct(worker.commit_ms, .95),
                          "max": round(max(worker.commit_ms), 1)},
            "read_ms": {k: {"p50": _pct(v, .5), "p95": _pct(v, .95),
                            "p99": _pct(v, .99), "max": round(max(v), 2)}
                        for k, v in reads.items()},
            "db_mb": round(os.path.getsize(path) / 1048576, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    import json

    print("=" * 88)
    print("asyncio 单循环 + 1 写线程 | 5 万记忆 + 2.5 万消息 | 100 bot × 2 条/s × 10s")
    print("=" * 88)
    r = asyncio.run(run(100, 2.0, 10.0, n_mem=50000))
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
