"""hermes_memory —— 记忆的领域逻辑与存储实现。

分层（同一个包内，靠子模块区分，避免「一个类干所有事」）：

- `inmemory`  : `MemoryStore` 的内存实现，测试与开发用，必须通过一致性套件
- `store_pg`  : Postgres 实现（待实现，接口同 `inmemory`）
- `service`   : 领域服务，组合 store + 权限 + 去重决策（待实现）

**这个包不 import 其他领域包**。需要身份解析、向量化时，从构造函数注入
`IdentityStore` / `Embedder` 端口。违反这条会被 `tools/check_boundaries.py` 拦下。
"""

from __future__ import annotations

from .inmemory import InMemoryMemoryStore

__all__ = ["InMemoryMemoryStore"]
