"""SQLite 存储实现。

`SqliteMemoryStore` 只实现 `MemoryStore`，并通过 `MemoryStoreConformance`；
`SqliteBotDb` 把一个账号的四张表（memories / resident / messages / identities）
装进同一个库文件、同一个连接——这是「每 bot 一库」隔离的执行体。
"""

from .botdb import (
    EXTRA_DDL,
    SqliteBotDb,
    SqliteIdentityStore,
    SqliteMessageStore,
    SqliteResidentStore,
)
from .store import SqliteMemoryStore

__all__ = [
    "SqliteMemoryStore",
    "SqliteBotDb",
    "SqliteResidentStore",
    "SqliteMessageStore",
    "SqliteIdentityStore",
    "EXTRA_DDL",
]
