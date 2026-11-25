"""SQLite 版的 MemoryStore。

存在的目的不是「先有个能用就行」，而是回答一个具体的问题：
**SQLite 能不能承载 100 bot 的记忆服务。** 实现必须能过同一套
`MemoryStoreConformance`，否则「能用」只是说法。

三条与 hermes_core 不同的决定，全部有基准数据支撑
（见 `benchmarks/`，同机实测）：

1. **单连接 + 一把锁，同步执行，不 to_thread。**
   注入路径的读实测 p99 = 0.07ms（5 万行库）。线程切换的开销（几十 μs 到
   上百 μs）和它同量级，`to_thread` 只会更慢。只有当单条查询可能超过
   10ms 时才值得卸线程——这里没有这种查询。

2. **写走 `add_many` 批量提交，`add` 是它的退化情形。**
   单条 autocommit 实测 4365 条/s，批量 500 条 15.9 万条/s。100 bot 的
   真实写入量是 0.7 条/s（6 万条/天），单条提交也够——但批量让「一次性
   回填历史」这种场景不至于跑几小时。

3. **可见性不在 SQL 里重复实现，只在 Python 调 `Memory.can_read`。**
   hermes_core 把可见性散在多段 SQL 字符串里，改一处漏一处。这里
   SQL 只负责「取候选」（按 bot 粗筛 + FTS 排序），归属与授权的判断
   一律回到 `Memory.can_read` 这唯一实现。
   代价是候选池要开大一点（`_POOL`），换取「永远不会拼错 WHERE」。
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Sequence

from hermes_contracts import (
    EMPTY,
    BotId,
    Clock,
    Memory,
    MemoryId,
    MemoryKey,
    MemoryScope,
    OwnerId,
    Visibility,
    clip,
    MAX_MEMORY_CHARS,
)

# candidate 池：FTS 取 limit*8 条，再经 can_read 过滤后截到 limit。
# 开大是因为同一 bot 内可能有多个用户，私聊记忆会占满前排。
_POOL_FACTOR = 8
_POOL_MIN = 64

DDL = """
pragma journal_mode=WAL;
pragma synchronous=NORMAL;
pragma busy_timeout=10000;
pragma foreign_keys=ON;

create table if not exists memories (
  id          integer primary key autoincrement,
  bot_id      text not null,
  scope       text not null,
  visibility  text not null,
  owner_id    text not null default '',
  group_id    text not null default '',
  kind        text not null default 'fact',
  source      text not null default '',
  content     text not null,
  ts          real not null,
  updated_ts  real not null default 0,
  merged_into integer not null default 0,
  hits        integer not null default 0,
  importance  real not null default 0,
  pinned      integer not null default 0
);

-- 活跃检索的主索引：所有注入查询都带 (bot, merged_into, owner)
create index if not exists idx_active
  on memories(bot_id, merged_into, owner_id, id);
create index if not exists idx_group
  on memories(bot_id, merged_into, group_id, id);

create virtual table if not exists mem_fts using fts5(
  content,
  content='memories',
  content_rowid='id',
  tokenize='trigram'
);
create trigger if not exists mem_ai after insert on memories begin
  insert into mem_fts(rowid, content) values (new.id, new.content);
end;
create trigger if not exists mem_ad after delete on memories begin
  insert into mem_fts(mem_fts, rowid, content) values ('delete', old.id, old.content);
end;
create trigger if not exists mem_au after update on memories begin
  insert into mem_fts(mem_fts, rowid, content) values ('delete', old.id, old.content);
  insert into mem_fts(rowid, content) values (new.id, new.content);
end;
"""

_COLS = (
    "id,bot_id,scope,visibility,owner_id,group_id,kind,source,"
    "content,ts,updated_ts,merged_into,hits,importance,pinned"
)
# 插入时不含自增 id。不用 _COLS.replace('id,','') —— 那会把 bot_id、group_id
# 里的 "id," 一起吃掉，变成 bot_scope 这种不存在的列名。
_INSERT_COLS = (
    "bot_id,scope,visibility,owner_id,group_id,kind,source,"
    "content,ts,updated_ts,merged_into,hits,importance,pinned"
)
_INSERT_SQL = (
    f"insert into memories({_INSERT_COLS}) "
    f"values ({','.join('?' * 14)})"
)


def _row_to_memory(r: tuple) -> Memory:
    return Memory(
        id=MemoryId(int(r[0])),
        bot_id=BotId(str(r[1])),
        scope=MemoryScope(str(r[2])),
        visibility=Visibility(str(r[3])),
        owner_id=str(r[4] or EMPTY),
        group_id=str(r[5] or EMPTY),
        kind=str(r[6] or "fact"),
        source=str(r[7] or EMPTY),
        content=str(r[8]),
        ts=float(r[9]),
        updated_ts=float(r[10]),
        merged_into=int(r[11]),
        hits=int(r[12]),
        importance=float(r[13]),
        pinned=bool(r[14]),
    )


def _insert_row(m: Memory) -> tuple:
    return (
        str(m.bot_id), str(m.scope), str(m.visibility), m.owner_id or "",
        m.group_id or "", m.kind, m.source or "", m.content, m.ts,
        m.updated_ts, int(m.merged_into), m.hits, m.importance,
        1 if m.pinned else 0,
    )


class SqliteMemoryStore:
    """`MemoryStore` 的 SQLite 实现。

    线程模型：单个 `sqlite3.Connection` 被一个 `RLock` 保护。
    SQLite 连接不可跨线程并发使用（hermes_core 的 `_db_locked` 注释里
    记录了实测到的 `InterfaceError`），而这里所有方法都是短事务，
    串行化不会成为瓶颈——实测提交 p50 = 1.06ms。
    """

    def __init__(self, path: str, *, clock: Clock | None = None,
                 commit_every: int = 64,
                 conn: sqlite3.Connection | None = None,
                 lock: threading.RLock | None = None) -> None:
        """`conn` / `lock` 用于复用调用方已打开的连接。

        为什么允许共享：一个 bot 的记忆、常驻、消息三张表在同一个库文件里，
        让它们各开一个连接就等于三个写者抢同一把文件锁——这正是基准里
        p99 从 7.8ms 涨到 288ms 的根因。共享连接 + 共享锁，写者只有一个。

        `_owns_conn` 记录所有权：共享连接时 `close()` 不能替别人关库，
        否则 BotRegistry 淘汰一个 store 会把同类下其他 store 一起打断。
        """
        self._path = path
        self._clock = clock
        self._owns_conn = conn is None
        self._lock = lock if lock is not None else threading.RLock()
        self._conn = conn if conn is not None else sqlite3.connect(
            path, isolation_level=None, check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = None
        self._conn.executescript(DDL)
        self._pending: list[Memory] = []
        self._commit_every = commit_every

    # ------------------------------------------------------------ 内部

    def _now(self) -> float:
        return self._clock.now() if self._clock is not None else __import__("time").time()

    def _prepare(self, memory: Memory) -> Memory:
        """契约要求：截断超长内容、ts 为 0 时补时钟。"""
        content = clip(memory.content, MAX_MEMORY_CHARS)
        ts = memory.ts or self._now()
        if content == memory.content and ts == memory.ts:
            return memory
        from dataclasses import replace

        return replace(memory, content=content, ts=ts)

    def close(self) -> None:
        with self._lock:
            self.flush()
            if self._owns_conn:
                self._conn.close()

    # ------------------------------------------------------------ 写

    def flush(self) -> int:
        """把待写缓冲一次性提交。调用方负责在进程退出前调它。"""
        with self._lock:
            if not self._pending:
                return 0
            rows = [_insert_row(m) for m in self._pending]
            self._conn.execute("begin immediate")
            try:
                self._conn.executemany(_INSERT_SQL, rows)
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
            n = len(rows)
            self._pending.clear()
            return n

    async def add(self, memory: Memory) -> Memory:
        """单条写入，**立即提交**。

        批量接口优先用 `add_many`；`add` 保持「返回即可见」的语义，
        因为调用方常在写之后立刻读回做断言。
        """
        m = self._prepare(memory)
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                cur = self._conn.execute(_INSERT_SQL, _insert_row(m))
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
            from dataclasses import replace

            return replace(m, id=MemoryId(int(cur.lastrowid or 0)))

    async def add_many(self, memories: Sequence[Memory]) -> list[Memory]:
        """批量写入，单事务。返回带 id 的副本，顺序与入参一致。"""
        if not memories:
            return []
        prepared = [self._prepare(m) for m in memories]
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                ids: list[int] = []
                for m in prepared:
                    cur = self._conn.execute(_INSERT_SQL, _insert_row(m))
                    ids.append(int(cur.lastrowid or 0))
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        from dataclasses import replace

        return [replace(m, id=MemoryId(i)) for m, i in zip(prepared, ids)]

    async def get(self, bot_id: BotId, memory_id: MemoryId) -> Memory | None:
        with self._lock:
            r = self._conn.execute(
                f"select {_COLS} from memories where id=? and bot_id=?",
                (int(memory_id), str(bot_id)),
            ).fetchone()
        return _row_to_memory(r) if r else None

    async def update(self, memory: Memory) -> None:
        m = self._prepare(memory)
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                self._conn.execute(
                    "update memories set scope=?,visibility=?,owner_id=?,group_id=?,"
                    "kind=?,source=?,content=?,ts=?,updated_ts=?,merged_into=?,"
                    "hits=?,importance=?,pinned=? where id=? and bot_id=?",
                    (
                        str(m.scope), str(m.visibility), m.owner_id or "",
                        m.group_id or "", m.kind, m.source or "", m.content,
                        m.ts, m.updated_ts or self._now(), int(m.merged_into),
                        m.hits, m.importance, 1 if m.pinned else 0,
                        int(m.id), str(m.bot_id),
                    ),
                )
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise

    async def merge(self, sources: Sequence[MemoryId], into: MemoryId, *,
                    at: float) -> int:
        target = int(into)
        srcs = [int(s) for s in sources]
        if target in srcs:
            raise ValueError("merge: into 不能出现在 sources 里（自环）")
        if not srcs:
            return 0
        with self._lock:
            # 防止链成环：into 不能已经被合并到某个 source 上
            chain = self._conn.execute(
                f"select merged_into from memories where id={target}"
            ).fetchone()
            node = int(chain[0]) if chain else 0
            seen = {target}
            while node:
                if node in srcs:
                    raise ValueError("merge: 会形成环")
                if node in seen:
                    break
                seen.add(node)
                nxt = self._conn.execute(
                    "select merged_into from memories where id=?", (node,)
                ).fetchone()
                node = int(nxt[0]) if nxt else 0

            ph = ",".join("?" * len(srcs))
            self._conn.execute("begin immediate")
            try:
                cur = self._conn.execute(
                    f"update memories set merged_into=?, updated_ts=? "
                    f"where id in ({ph}) and merged_into=0",
                    (target, at, *srcs),
                )
                n = int(cur.rowcount or 0)
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        return n

    # ------------------------------------------------------------ 读

    async def list_active(self, key: MemoryKey, *, limit: int,
                          after_id: int = 0) -> Sequence[Memory]:
        where = ["bot_id=?", "merged_into=0", "id>?"]
        args: list = [str(key.bot_id), int(after_id)]
        where.append("scope=?")
        args.append(str(key.scope))
        if key.owner_id:
            where.append("owner_id=?")
            args.append(key.owner_id)
        if key.group_id:
            where.append("group_id=?")
            args.append(key.group_id)
        sql = (f"select {_COLS} from memories where {' and '.join(where)} "
               f"order by id asc limit ?")
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_row_to_memory(r) for r in rows]

    async def search_candidates(self, key: MemoryKey, *, query: str,
                                viewer: OwnerId, viewer_group: str,
                                limit: int) -> Sequence[Memory]:
        """在库内粗排 top-k，返回已按 `can_read` 过滤的候选。

        两步：
          1. SQL 侧：限定 bot 与 active，用 FTS5(trigram) 按 bm25 取前 `pool` 条；
             查询过短（<3 字，trigram 无法切分）或 FTS 无命中时，退回「最近活跃」。
          2. Python 侧：调 `Memory.can_read` 过滤，再截到 limit。

        第 2 步刻意放在 Python：可见性只有一处实现。代价是候选池要开大
        （`_POOL_FACTOR`），换来「SQL 里永远不会拼错授权条件」。
        """
        lim = max(1, int(limit))
        pool = max(_POOL_MIN, lim * _POOL_FACTOR)
        q = (query or "").strip()

        rows: list[tuple] = []
        with self._lock:
            if len(q) >= 3:
                try:
                    rows = self._conn.execute(
                        f"select {_COLS} from memories "
                        f"where bot_id=? and merged_into=0 and id in ("
                        f"  select rowid from mem_fts where mem_fts match ? "
                        f"  order by rank limit ?) ",
                        (str(key.bot_id), '"%s"' % q.replace('"', '""'), pool),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            if len(rows) < lim:
                # 补「最近活跃」：FTS 只认连续三字，「刚才提过但用词不同」的
                # 场景它命不中，而这类恰恰是对话里最常被追问的。
                have = {int(r[0]) for r in rows}
                extra = self._conn.execute(
                    f"select {_COLS} from memories "
                    f"where bot_id=? and merged_into=0 order by updated_ts desc, id desc "
                    f"limit ?",
                    (str(key.bot_id), pool),
                ).fetchall()
                rows.extend(r for r in extra if int(r[0]) not in have)

        out: list[Memory] = []
        seen: set[int] = set()
        for r in rows:
            m = _row_to_memory(r)
            if int(m.id) in seen:
                continue
            if m.can_read(viewer, viewer_group=viewer_group):
                seen.add(int(m.id))
                out.append(m)
            if len(out) >= lim:
                break
        return out

    async def count_active(self, bot_id: BotId, *, scope: str | None = None,
                           owner_id: str | None = None) -> int:
        where = ["bot_id=?", "merged_into=0"]
        args: list = [str(bot_id)]
        if scope:
            where.append("scope=?")
            args.append(str(scope))
        if owner_id:
            where.append("owner_id=?")
            args.append(owner_id)
        with self._lock:
            r = self._conn.execute(
                f"select count(*) from memories where {' and '.join(where)}", args
            ).fetchone()
        return int(r[0]) if r else 0

    async def purge_owner(self, bot_id: BotId, owner_id: OwnerId) -> int:
        """物理删除某人的全部记忆，并清理指向已删行的合并死链。

        `merged_into` 指向已删的父行会让回滚合并时撞到不存在的 id，
        所以删完要把这类孤儿子节点一起清掉。
        """
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                cur = self._conn.execute(
                    "delete from memories where bot_id=? and owner_id=?",
                    (str(bot_id), str(owner_id)),
                )
                n = int(cur.rowcount or 0)
                self._conn.execute(
                    "delete from memories where bot_id=? and merged_into<>0 "
                    "and merged_into not in (select id from memories where bot_id=?)",
                    (str(bot_id), str(bot_id)),
                )
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        return n
