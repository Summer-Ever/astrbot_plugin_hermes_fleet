"""接口的「验收标准」。一个实现没跑过对应套件，就不算完成。

这里是整套并行开发方案的关键装置。多 agent 同时写代码时，最常见的失败
不是谁写错了，而是**两边各自都对、拼起来不对**：A 认为 ``enqueue`` 重复
调用会新建一条，B 认为会返回 None，两人都没写文档，联调时才发现。

把这些约定写成可执行的用例之后：
- 实现方自己就能知道自己是否符合契约，不需要等对方；
- 契约被违反时，是本地的红测试，而不是三天后的线上事故；
- 新增一种实现（比如从内存版换成 Postgres 版）时，**不用重新设计测试**。

约定：套件类名不带 `Test` 前缀、并设置 `__test__ = False`，基类本身不执行。
"""

from __future__ import annotations

from .jobs import JobQueueConformance
from .memory import MemoryStoreConformance

__all__ = ["MemoryStoreConformance", "JobQueueConformance"]
