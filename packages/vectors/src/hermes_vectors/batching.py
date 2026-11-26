"""查询向量化的排队与合批。

## 为什么需要它

向量召回落在对话热路径上：字面召回凑不够条数时，现场要把问句变成向量。
上百个用户同时说话，就是上百次并发的 embedding 调用——

- 本地模型：并发调用互相抢 CPU，单条 5ms 的推理在几十并发下变成几百毫秒；
- 云端接口：每个请求一次网络往返，10 个并发就是 10 次 RTT，而接口本身
  支持一批 10 条。

排队 + 合批把「N 次调用」压成「N/批大小 次」，吞吐翻几倍；同时队列有上限、
每次等待有预算，超预算的请求直接放弃语义补充——对话拿到的是字面召回结果，
比卡住强。

## 三条边界（不满足就别用它）

1. **有界**：队列满时立刻抛 `EmbeddingQueueFull`，内存不会随并发无界上涨。
2. **有预算**：`embed_one(..., budget_s=)` 超时抛 `EmbeddingTimeout`，
   调用方降级为字面召回，而不是无限等。
3. **会去重**：同一批里重复的文本只算一次，结果广播给所有等待者。

worker 数量默认 2：一个够本地模型用，两个让云端接口能在一次往返期间
继续接客。再往上加不会更快——瓶颈在模型/接口，不在队列。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Callable

from .embedding import EmbeddingError


class EmbeddingQueueFull(EmbeddingError):
    """排队的人太多了：立刻放弃这次语义补充。"""


class EmbeddingTimeout(EmbeddingError):
    """等得太久：超出调用方给的预算。"""


class BatchingEmbedder:
    """把并发 `embed_one` 合并成批交给底层客户端。

    与 `EmbeddingClient` 的接口兼容（`embed` / `embed_one`），所以可以直接
    替换掉 `VectorGlue` 里的 embedder；`embed()` 走底层批量接口，不排队。
    """

    def __init__(
        self,
        client: object,
        *,
        batch_size: int = 10,
        window_s: float = 0.01,
        max_pending: int = 256,
        workers: int = 2,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.batch_size = max(1, int(batch_size))
        # 攒批窗口：等这么久还没凑满就发车。太大会平白增加延迟，
        # 太小则退化成不排队。
        self.window_s = max(0.0, float(window_s))
        self.max_pending = max(1, int(max_pending))
        self.workers = max(1, int(workers))
        self._now = now
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._tasks: list[asyncio.Task] = []
        self.submitted = 0
        self.batches = 0
        self.deduped = 0
        self.failed = 0

    # ------------------------------------------------------------ 对外

    async def embed_one(self, text: str, *, budget_s: float = 0.0) -> list[float]:
        key = str(text or "").strip()
        if not key:
            return []
        self._ensure_started()
        if self._queue is not None and self._queue.qsize() >= self.max_pending:
            raise EmbeddingQueueFull(
                f"向量化队列已满（{self.max_pending}），本次退回字面召回"
            )
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((key, future))  # type: ignore[union-attr]
        self.submitted += 1
        try:
            if budget_s and budget_s > 0:
                return await asyncio.wait_for(future, budget_s)
            return await future
        except asyncio.TimeoutError:
            # `wait_for` 会取消 future；worker 侧写入前会检查 done()。
            raise EmbeddingTimeout(
                f"查询向量化超出预算 {budget_s:.2f}s，本次退回字面召回"
            ) from None

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量接口直通：这是索引路径，本来就成批，不需要再排队。"""
        return await self.client.embed(texts)  # type: ignore[attr-defined]

    async def aclose(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()
        self._queue = None
        self._loop = None

    def close_nowait(self) -> None:
        """同步关闭：插件 `close()` 是同步的，只做取消，不等清理。"""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        self._queue = None
        self._loop = None

    # ------------------------------------------------------------ 内部

    def _ensure_started(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is loop and self._tasks:
            return
        # 换事件循环（常见于测试）：旧任务已无用，直接丢弃重建。
        self.close_nowait()
        self._loop = loop
        self._queue = asyncio.Queue()
        self._tasks = [loop.create_task(self._worker()) for _ in range(self.workers)]

    async def _worker(self) -> None:
        queue = self._queue
        if queue is None:
            return
        while True:
            try:
                first = await queue.get()
            except asyncio.CancelledError:
                return
            batch = [first]
            deadline = self._now() + self.window_s
            while len(batch) < self.batch_size:
                remaining = deadline - self._now()
                if remaining <= 0:
                    break
                try:
                    more = await asyncio.wait_for(queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    return
                batch.append(more)
            await self._flush(batch)

    async def _flush(self, batch: list[tuple[str, asyncio.Future]]) -> None:
        # 已被调用方放弃（超时取消）的请求不再浪费算力。
        waiting: dict[str, list[asyncio.Future]] = {}
        for text, future in batch:
            if future.done():
                continue
            waiting.setdefault(text, []).append(future)
        if not waiting:
            return
        texts = list(waiting)
        self.batches += 1
        self.deduped += len(batch) - len(texts)
        try:
            vectors = await self.client.embed(texts)  # type: ignore[attr-defined]
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — 一批失败不能变成未处理异常
            self.failed += 1
            error = exc if isinstance(exc, EmbeddingError) else EmbeddingError(
                f"{type(exc).__name__}: {exc}")
            for waiters in waiting.values():
                for future in waiters:
                    if not future.done():
                        future.set_exception(error)
            return
        if len(vectors) != len(texts):
            self.failed += 1
            error = EmbeddingError(
                f"向量条数不匹配：请求 {len(texts)} 条，返回 {len(vectors)} 条")
            for waiters in waiting.values():
                for future in waiters:
                    if not future.done():
                        future.set_exception(error)
            return
        for text, vector in zip(texts, vectors):
            for future in waiting[text]:
                if not future.done():
                    future.set_result(vector)
