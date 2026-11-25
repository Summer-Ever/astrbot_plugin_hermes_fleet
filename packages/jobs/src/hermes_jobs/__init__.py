"""hermes_jobs —— 任务队列（表即队列）。

对外只暴露 `JobQueue` 协议的实现，不暴露内部表结构。

分层：
- `inmemory` : 内存实现，测试与开发用，必须通过 `JobQueueConformance`
- `queue_pg` : Postgres 实现（`SELECT ... FOR UPDATE SKIP LOCKED`），待实现

**这个包不 import 其他领域包**。worker 需要记忆能力时，从构造函数注入
`MemoryStore`；队列只管「谁在什么时候处理哪个 scope_key」，不了解业务。
"""

from __future__ import annotations

from .inmemory import InMemoryJobQueue

__all__ = ["InMemoryJobQueue"]
