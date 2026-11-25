"""一个 bot 一个库文件，四种 Store 共享同一个连接。

## 为什么是「每 bot 一库」而不是「一个库加 bot_id 列」

隔离有两种实现方式，成本不对称：

- **一个库 + `WHERE bot_id=?`**：靠每条 SQL 都记得带上条件。漏写一处，
  记忆就静默串到另一个账号上，**而且不会报错**。100 个账号、几十处查询，
  靠纪律守住这件事，概率上不成立。
- **一个账号一个文件**：串号在物理上不可能发生——另一个账号的数据根本
  不在这个连接能看到的范围里。删账号就是删文件，备份就是拷文件。

实测尾延迟也是每 bot 一库更好（同一台机器、总负载 200 条/秒）：
单库单写线程 p99 = 75ms，每 bot 一库 p99 = 7.8ms。原因是 100 个文件
把 SQLite 的写锁分散开了，而不是让 100 个写者排队抢一把锁。

`bot_id` 列**仍然写进每一行**。这不是给隔离用的，是给未来留的退路：
哪天要合库或做跨库分析，行里得有归属。防御性冗余，不承担隔离职责。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Sequence

from hermes_contracts import (
    EMPTY,
    BotId,
    Clock,
    Message,
    MessageRole,
    OwnerId,
    ResidentBlock,
    ResidentEntry,
    ResidentScope,
    SessionId,
    clip,
    MAX_MESSAGE_CHARS,
)
from hermes_contracts.models import MAX_RESIDENT_BLOCK_CHARS

from .store import DDL as MEMORY_DDL
from .store import SqliteMemoryStore

EXTRA_DDL = """
-- 常驻六词条。与 memories 物理分表：后台整理只操作 memories，
-- 结构上碰不到常驻，不需要靠「SQL 里记得加 WHERE」来保护。
create table if not exists resident (
  bot_id      text not null,
  owner_id    text not null default '',
  scope       text not null,
  block       text not null,
  content     text not null default '',
  updated_ts  real not null default 0,
  primary key (bot_id, owner_id, scope, block)
);

-- 会话原文。量是记忆的 100 倍以上，所以只留四个访问路径所需的最小索引。
create table if not exists messages (
  id          integer primary key autoincrement,
  bot_id      text not null,
  session_id  text not null,
  owner_id    text not null default '',
  group_id    text not null default '',
  role        text not null,
  content     text not null,
  ts          real not null,
  summarized  integer not null default 0,
  dedup       text not null default ''
);

-- 幂等：同一条消息重投不产生重复行。partial index 让 dedup 为空的
-- 历史行（迁移进来的）不参与唯一性约束。
create unique index if not exists idx_msg_dedup
  on messages(bot_id, session_id, dedup) where dedup <> '';
create index if not exists idx_msg_unsum
  on messages(bot_id, session_id, summarized, id);

-- 身份别名：QQ 官方适配器的 sender_id 是 per-bot 的 openid，
-- 同一个人在群里是 member_openid、私聊是 user_openid，且按群不同。
-- 这张表是「把多个 openid 认成同一个人」的唯一落点（默认不用）。
create table if not exists identities (
  bot_id      text not null,
  raw_id      text not null,
  owner_id    text not null,
  created_ts  real not null default 0,
  primary key (bot_id, raw_id)
);
"""


class SqliteResidentStore:
    """`ResidentStore` 实现。整段覆盖，不做 append。"""

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock,
                 clock: Clock | None = None) -> None:
        self._conn = conn
        self._lock = lock
        self._clock = clock

    def _now(self) -> float:
        return self._clock.now() if self._clock is not None else time.time()

    async def get_all(
        self,
        bot_id: BotId,
        owner_id: OwnerId,
        scope: ResidentScope,
    ) -> dict[ResidentBlock, ResidentEntry]:
        with self._lock:
            rows = self._conn.execute(
                "select block, content, updated_ts from resident "
                "where bot_id=? and owner_id=? and scope=?",
                # 必须走 to_key()：ResidentScope 是 dataclass 不是 str，
                # str(scope) 会得到 "ResidentScope(kind='private', ...)"，
                # 写进去能过、读的时候永远查不到——最坏的一类 bug。
                (str(bot_id), str(owner_id), scope.to_key()),
            ).fetchall()
        out: dict[ResidentBlock, ResidentEntry] = {}
        for block, content, ts in rows:
            if not content:
                # 空词条不注入——这是「注入成本有界」的实现方式，
                # 也是它必须返回 dict 而不是「六个固定项」的原因。
                continue
            try:
                b = ResidentBlock(str(block))
            except ValueError:
                continue
            out[b] = ResidentEntry(
                bot_id=bot_id,
                owner_id=owner_id,
                scope=scope,
                block=b,
                text=str(content),
                updated_ts=float(ts or 0.0),
            )
        return out

    async def upsert(self, entry: ResidentEntry) -> ResidentEntry:
        text = clip(entry.text, MAX_RESIDENT_BLOCK_CHARS)
        ts = entry.updated_ts or self._now()
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                self._conn.execute(
                    "insert into resident(bot_id, owner_id, scope, block, content, updated_ts)"
                    " values (?,?,?,?,?,?)"
                    " on conflict(bot_id, owner_id, scope, block) do update set"
                    " content=excluded.content, updated_ts=excluded.updated_ts",
                    (str(entry.bot_id), str(entry.owner_id), entry.scope.to_key(),
                     str(entry.block.value), text, ts),
                )
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        from dataclasses import replace

        return replace(entry, text=text, updated_ts=ts)

    async def clear(self, bot_id: BotId, owner_id: OwnerId, scope: ResidentScope) -> int:
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                cur = self._conn.execute(
                    "delete from resident where bot_id=? and owner_id=? and scope=?",
                    (str(bot_id), str(owner_id), scope.to_key()),
                )
                n = int(cur.rowcount or 0)
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        return n


class SqliteMessageStore:
    """`MessageStore` 实现。只做四件事：批量写、取未总结、按行标记、删会话。"""

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock,
                 clock: Clock | None = None) -> None:
        self._conn = conn
        self._lock = lock
        self._clock = clock

    async def append_batch(self, messages: Sequence[Message]) -> int:
        if not messages:
            return 0
        rows = []
        for m in messages:
            content = clip(m.content, MAX_MESSAGE_CHARS)
            # dedup 用「会话 + 角色 + 内容 + 秒级时间」：重投同一批时
            # 时间戳一致才能命中唯一索引，这正是我们要的幂等。
            # 用 Python 的 hash 会因 PYTHONHASHSEED 每次进程不同而失效。
            import hashlib

            raw = f"{m.session_id}\x00{m.role}\x00{int(m.ts)}\x00{content}"
            dedup = hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()
            rows.append((
                str(m.bot_id), str(m.session_id), m.sender_id or "",
                m.group_id or "", str(m.role), content, float(m.ts),
                1 if m.summarized else 0, dedup,
            ))
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                before = self._conn.total_changes
                self._conn.executemany(
                    "insert or ignore into messages"
                    "(bot_id, session_id, owner_id, group_id, role, content, ts, summarized, dedup)"
                    " values (?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                written = self._conn.total_changes - before
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        return int(written)

    async def fetch_unsummarized(
        self,
        bot_id: BotId,
        session_id: SessionId,
        *,
        limit: int,
        min_ts: float = 0.0,
    ) -> Sequence[Message]:
        with self._lock:
            rows = self._conn.execute(
                "select id, owner_id, group_id, role, content, ts, summarized"
                " from messages where bot_id=? and session_id=? and summarized=0"
                " and ts>=? order by id asc limit ?",
                (str(bot_id), str(session_id), float(min_ts), int(limit)),
            ).fetchall()
        return [
            Message(
                bot_id=bot_id,
                session_id=session_id,
                role=_role(r[3]),
                content=str(r[4]),
                ts=float(r[5]),
                id=int(r[0]),
                sender_id=str(r[1] or ""),
                group_id=str(r[2] or ""),
                summarized=bool(r[6]),
            )
            for r in rows
        ]

    async def mark_summarized(self, message_ids: Sequence[int]) -> int:
        ids = [int(i) for i in message_ids if i]
        if not ids:
            return 0
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                caps = ",".join("?" * len(ids))
                cur = self._conn.execute(
                    f"update messages set summarized=1 where summarized=0 and id in ({caps})",
                    ids,
                )
                n = int(cur.rowcount or 0)
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        return n

    async def delete_session(self, bot_id: BotId, session_id: SessionId) -> int:
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                cur = self._conn.execute(
                    "delete from messages where bot_id=? and session_id=?",
                    (str(bot_id), str(session_id)),
                )
                n = int(cur.rowcount or 0)
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        return n

    async def list_pending_sessions(
        self, bot_id: BotId, *, limit: int = 50
    ) -> list[tuple[SessionId, int, float]]:
        """按「最久没动静」排出待总结会话。

        走 `idx_msg_unsum(bot_id, session_id, summarized, id)`：过滤条件
        `summarized=0` 打头，分组键紧随其后，所以这条聚合是索引扫描而不是
        全表扫。消息表是最大的一张表（百万级），这条查询会被 worker 每轮
        调用一次，索引形状错了会变成稳定的 CPU 占用。
        """
        with self._lock:
            rows = self._conn.execute(
                "select session_id, count(*) as n, max(ts) as last_ts"
                " from messages where bot_id=? and summarized=0"
                " group by session_id order by last_ts asc limit ?",
                (str(bot_id), int(limit)),
            ).fetchall()
        return [
            (SessionId(str(r[0])), int(r[1]), float(r[2] or 0.0))
            for r in rows
            if r[0]
        ]


class SqliteIdentityStore:
    """`IdentityStore` 实现。默认行为是「一个 openid 就是一个人」。

    QQ 官方适配器下，同一个人在不同群里是不同 openid，跨群认人需要一个
    更强的信号（用户在群里自报身份、或运营手工绑定）。这里不猜：
    `resolve` 只在显式 link 过时才归一，否则原样返回。
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock,
                 clock: Clock | None = None) -> None:
        self._conn = conn
        self._lock = lock
        self._clock = clock

    async def resolve(self, bot_id: BotId, raw_id: str) -> OwnerId:
        raw = str(raw_id or "")
        if not raw:
            return OwnerId(EMPTY)
        with self._lock:
            row = self._conn.execute(
                "select owner_id from identities where bot_id=? and raw_id=?",
                (str(bot_id), raw),
            ).fetchone()
        return OwnerId(str(row[0])) if row else OwnerId(raw)

    async def link(self, bot_id: BotId, raw_id: str, canonical_id: OwnerId) -> None:
        now = self._clock.now() if self._clock is not None else time.time()
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                self._conn.execute(
                    "insert into identities(bot_id, raw_id, owner_id, created_ts)"
                    " values (?,?,?,?) on conflict(bot_id, raw_id) do update set"
                    " owner_id=excluded.owner_id",
                    (str(bot_id), str(raw_id), str(canonical_id), now),
                )
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise

    async def unlink(self, bot_id: BotId, raw_id: str) -> bool:
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                cur = self._conn.execute(
                    "delete from identities where bot_id=? and raw_id=?",
                    (str(bot_id), str(raw_id)),
                )
                n = int(cur.rowcount or 0)
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        return n > 0


def _role(raw: object) -> MessageRole:
    s = str(raw or "user")
    if s not in ("user", "assistant", "system"):
        s = "user"
    return s  # type: ignore[return-value]


class SqliteBotDb:
    """一个 bot 账号的完整记忆库：四条 Store 共用一个连接和一把锁。

    共用一个连接不是为了省资源，是为了让「一个事务里写记忆 + 标记消息」
    成为可能。多连接下这件事需要两阶段提交，而 SQLite 只支持单库事务。
    """

    def __init__(self, path: str, *, clock: Clock | None = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            path, isolation_level=None, check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = None
        self._conn.executescript(MEMORY_DDL)
        self._conn.executescript(EXTRA_DDL)
        self._clock = clock
        # MemoryStore 复用同一个连接与锁，不自己再开一个。
        self.memories = SqliteMemoryStore(
            path, clock=clock, conn=self._conn, lock=self._lock
        )
        self.resident = SqliteResidentStore(self._conn, self._lock, clock)
        self.messages = SqliteMessageStore(self._conn, self._lock, clock)
        self.identities = SqliteIdentityStore(self._conn, self._lock, clock)

    def close(self) -> None:
        with self._lock:
            try:
                self.memories.flush()
            except Exception:
                pass
            try:
                # 关库前做一次 checkpoint：否则 -wal 文件里留着未合并的页，
                # 备份时只拷 .db 会丢最近的写入。
                self._conn.execute("pragma wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.close()

    def stats(self) -> dict[str, int]:
        with self._lock:
            def one(sql: str) -> int:
                try:
                    r = self._conn.execute(sql).fetchone()
                    return int(r[0]) if r else 0
                except Exception:
                    return 0

            return {
                "memories": one(
                    "select count(*) from memories where merged_into=0"),
                "resident": one("select count(*) from resident where content<>''"),
                "messages": one("select count(*) from messages"),
                "unsummarized": one(
                    "select count(*) from messages where summarized=0"),
            }
