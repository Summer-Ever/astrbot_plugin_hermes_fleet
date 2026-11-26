#!/usr/bin/env python3
"""向量检索的承载能力基准（合成数据，零网络）。

回答三个具体问题：

1. 单个账号的精确 KNN 在 1k / 10k / 100k 条时 p50 / p95 是多少？
2. 50 个并发检索（分散在多个账号库上）会不会把事件循环卡住？
3. 有 / 没有 numpy 差多少？没有时生产路径只扫一个时间窗口，扫了多少条？

用法：

    python tools/bench_vector.py
    python tools/bench_vector.py --dim 512 --sizes 1000,10000
    python tools/bench_vector.py --concurrency 50 --bots 8

**不测 embedding 吞吐**：那完全取决于供应商与网络，合成数据测不出有意义的
数字。要测就把 `HERMES_VECTOR_BASE_URL` / `HERMES_VECTOR_MODEL`（可选
`..._API_KEY`）设好，脚本会顺带跑一轮真实批量调用。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for src in sorted((ROOT / "packages").glob("*/src")):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from hermes_vectors import (  # noqa: E402
    EmbeddingClient,
    EmbeddingConfig,
    SqliteVectorStore,
    VectorRow,
    has_numpy,
    normalize,
    pack,
)

# 生产路径在没有 numpy 时的扫描上限规则，与 hermes_fleet.vector 保持一致。
PURE_PY_OP_BUDGET = 500_000
PURE_PY_MAX_ROWS = 2048


def _scan_cap(dim: int) -> int:
    if has_numpy():
        return 0
    return max(128, min(PURE_PY_MAX_ROWS, PURE_PY_OP_BUDGET // max(1, dim)))


def _rand_vec(dim: int) -> list[float]:
    return normalize([random.uniform(-1.0, 1.0) for _ in range(dim)])


def _build_store(path: str, n: int, dim: int) -> SqliteVectorStore:
    store = SqliteVectorStore(path)
    store.set_meta(model="bench", dim=str(dim))
    chunk: list[VectorRow] = []
    for i in range(n):
        chunk.append(VectorRow(
            memory_id=i + 1, owner_id="u1", group_id="", scope="user",
            visibility="private", content_hash=f"h{i}",
            updated_ts=float(i), dim=dim, vec=pack(_rand_vec(dim)),
        ))
        if len(chunk) >= 2000:
            store.upsert(chunk)
            chunk.clear()
    if chunk:
        store.upsert(chunk)
    return store


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


async def _measure_search(store: SqliteVectorStore, dim: int, queries: int,
                          scan_cap: int) -> dict:
    vecs = [_rand_vec(dim) for _ in range(max(3, min(queries, 20)))]
    lat: list[float] = []
    for i in range(queries):
        t0 = time.perf_counter()
        await asyncio.to_thread(
            store.search, vecs[i % len(vecs)], viewer="u1", limit=5,
            min_score=0.0, scan_cap=scan_cap,
        )
        lat.append((time.perf_counter() - t0) * 1000)
    return {"p50": _pct(lat, 50), "p95": _pct(lat, 95),
            "p99": _pct(lat, 99), "max": max(lat), "n": len(lat)}


async def _loop_lag(stop: asyncio.Event, interval: float = 0.005) -> float:
    """记录事件循环的最大卡顿（毫秒）。"""
    worst = 0.0
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
        worst = max(worst, (time.perf_counter() - t0 - interval) * 1000)
    return worst


async def _measure_concurrency(bots: int, rows: int, dim: int,
                               concurrency: int, tmp: str) -> dict:
    stores = [
        _build_store(os.path.join(tmp, f"bot{i}.db"), rows, dim)
        for i in range(bots)
    ]
    vecs = [_rand_vec(dim) for _ in range(concurrency)]
    cap = _scan_cap(dim)
    stop = asyncio.Event()
    lag_task = asyncio.create_task(_loop_lag(stop))
    lat: list[float] = []

    async def one(i: int) -> None:
        t0 = time.perf_counter()
        await asyncio.to_thread(
            stores[i % bots].search, vecs[i], viewer="u1", limit=5,
            min_score=0.0, scan_cap=cap,
        )
        lat.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(concurrency)))
    wall = (time.perf_counter() - t0) * 1000
    stop.set()
    worst = await lag_task
    for store in stores:
        store.close()
    return {"wall": wall, "p50": _pct(lat, 50), "p95": _pct(lat, 95),
            "lag": worst, "bots": bots}


async def _embedding_probe() -> None:
    base = os.environ.get("HERMES_VECTOR_BASE_URL", "")
    model = os.environ.get("HERMES_VECTOR_MODEL", "")
    if not base or not model:
        print("\n[embedding] 跳过：设置 HERMES_VECTOR_BASE_URL / "
              "HERMES_VECTOR_MODEL 后可测真实吞吐")
        return
    client = EmbeddingClient(EmbeddingConfig(
        base_url=base, model=model,
        api_key=os.environ.get("HERMES_VECTOR_API_KEY", ""),
        batch_size=int(os.environ.get("HERMES_VECTOR_BATCH", "32")),
    ))
    texts = [f"第 {i} 条记忆：用户偏好与日常记录" for i in range(64)]
    t0 = time.perf_counter()
    rows = await client.embed(texts)
    cost = time.perf_counter() - t0
    print(f"\n[embedding] {len(rows)} 条 / {cost:.2f}s = "
          f"{len(rows) / cost:.1f} 条/秒（维度 {len(rows[0]) if rows else 0}）")


async def main() -> int:
    parser = argparse.ArgumentParser(description="向量检索承载能力基准")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--sizes", default="1000,10000,100000")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--bots", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    sizes = [int(x) for x in str(args.sizes).replace("，", ",").split(",") if x.strip()]

    print(f"numpy: {'有' if has_numpy() else '无'} · 维度 {args.dim} · "
          f"单次查询取 top-5")
    print("（每账号一个库；纯 Python 时按生产预算截断扫描窗口）\n")
    print("| 条数 | 扫描上限 | p50 | p95 | p99 | max |")
    print("|---:|---:|---:|---:|---:|---:|")

    tmp = tempfile.mkdtemp(prefix="bench-vector-")
    try:
        for n in sizes:
            store = _build_store(os.path.join(tmp, f"size-{n}.db"), n, args.dim)
            cap = _scan_cap(args.dim)
            res = await _measure_search(store, args.dim, args.queries, cap)
            store.close()
            print(f"| {n} | {cap or '全部'} | {res['p50']:.2f}ms | "
                  f"{res['p95']:.2f}ms | {res['p99']:.2f}ms | {res['max']:.2f}ms |")

        conc = await _measure_concurrency(
            args.bots, 10000, args.dim, args.concurrency, tmp
        )
        print(f"\n[并发] {conc['bots']} 个账号库 × 1 万条，"
              f"{args.concurrency} 个并发检索：")
        print(f"  总耗时 {conc['wall']:.1f}ms · p50 {conc['p50']:.2f}ms · "
              f"p95 {conc['p95']:.2f}ms · 事件循环最大卡顿 {conc['lag']:.2f}ms")
        print("  说明：每个账号一个库文件，并发落在不同文件上；同一账号内由"
              "连接锁串行（实测单次毫秒级）。")
        await _embedding_probe()
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
