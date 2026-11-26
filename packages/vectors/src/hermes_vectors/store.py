"""每账号一个向量旁路库：`vectors.db`。

## 为什么是旁路文件，而不是给 `memory.db` 加表

1. **truth 只有一份**：`memory.db` 仍是唯一真相，向量删了、坏了、模型换了，
   重建即可；反过来「向量库是真相」会让恢复/迁移多一条必须校验的路径。
2. **备份语义天然正确**：`backup.py` 只白名单 `memory.db` + `bot.json`，
   旁路库根本不在包里——恢复一台机器不会把「另一台机器的旧向量」带回来。
3. **失败可丢弃**：这个文件损坏时直接删掉，代价只是向量召回暂时缺席，
   字面召回（FTS + BM25 精排）完全不受影响。

表只有两张：`meta`（模型、维度、指纹、同步状态）与 `vec`（id → 向量）。
没有 `dirty` 表：索引器用「活跃 id 集 + content_hash」直接对账，少一张表
就少一处需要保证一致的状态。
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .core import top_k

VECTOR_DB_FILE = "vectors.db"

DDL = """
pragma journal_mode=WAL;
pragma synchronous=NORMAL;
pragma busy_timeout=10000;

create table if not exists meta (
  key   text primary key,
  value text not null default ''
);

create table if not exists vec (
  memory_id    integer primary key,
  owner_id     text not null default '',
  group_id     text not null default '',
  scope        text not null default 'user',
  visibility   text not null default 'private',
  content_hash text not null default '',
  updated_ts   real not null default 0,
  dim          integer not null default 0,
  vec          blob not null
);

-- 检索时的可见性预过滤走 owner / group；updated_ts 给「纯 Python 窗口」
-- 用：没有 numpy 时按最近更新时间截断扫描量，而不是随机截断。
create index if not exists idx_vec_owner on vec(owner_id);
create index if not exists idx_vec_group on vec(group_id);
create index if not exists idx_vec_updated on vec(updated_ts);
"""


@dataclass(frozen=True, slots=True)
class VectorRow:
    memory_id: int
    owner_id: str
    group_id: str
    scope: str
    visibility: str
    content_hash: str
    updated_ts: float
    dim: int
    vec: bytes


class SqliteVectorStore:
    """向量旁路库的同步实现。线程模型与记忆库一致：单连接 + 一把锁。

    所有方法都是短事务；检索的矩阵计算放在锁外（先取行，再算分），
    避免一次 2 万条的 KNN 把同一账号的索引写入堵住。
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            path, isolation_level=None, check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = None
        try:
            self._conn.executescript(DDL)
        except Exception:
            # 打开成功但 DDL 失败（文件不是 SQLite 库）时必须先关连接：
            # Windows 上挂着句柄的文件无法 rename，上层的「隔离坏库并重建」
            # 会因此失败在第二次打开同一个坏文件上。
            try:
                self._conn.close()
            except Exception:
                pass
            raise

    # ------------------------------------------------------------ 生命周期

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("pragma wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._conn.close()

    # ------------------------------------------------------------ 元数据

    def meta(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("select key, value from meta").fetchall()
        return {str(k): str(v) for k, v in rows}

    def set_meta(self, **values: object) -> None:
        rows = [(str(k), "" if v is None else str(v)) for k, v in values.items()]
        if not rows:
            return
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                self._conn.executemany(
                    "insert into meta(key, value) values (?,?) "
                    "on conflict(key) do update set value=excluded.value",
                    rows,
                )
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise

    # ------------------------------------------------------------ 写

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("select count(*) from vec").fetchone()
        return int(row[0]) if row else 0

    def hashes(self) -> dict[int, str]:
        """全部已索引条目的 `id -> content_hash`。索引器对账用。"""
        with self._lock:
            rows = self._conn.execute(
                "select memory_id, content_hash from vec"
            ).fetchall()
        return {int(i): str(h) for i, h in rows}

    def upsert(self, rows: Sequence[VectorRow]) -> int:
        if not rows:
            return 0
        payload = [
            (
                int(r.memory_id), r.owner_id or "", r.group_id or "", r.scope,
                r.visibility, r.content_hash, float(r.updated_ts),
                int(r.dim), r.vec,
            )
            for r in rows
        ]
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                self._conn.executemany(
                    "insert into vec(memory_id, owner_id, group_id, scope,"
                    " visibility, content_hash, updated_ts, dim, vec)"
                    " values (?,?,?,?,?,?,?,?,?)"
                    " on conflict(memory_id) do update set"
                    " owner_id=excluded.owner_id, group_id=excluded.group_id,"
                    " scope=excluded.scope, visibility=excluded.visibility,"
                    " content_hash=excluded.content_hash,"
                    " updated_ts=excluded.updated_ts, dim=excluded.dim,"
                    " vec=excluded.vec",
                    payload,
                )
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        return len(payload)

    def delete(self, memory_ids: Iterable[int]) -> int:
        ids = [int(i) for i in memory_ids]
        if not ids:
            return 0
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                total = 0
                for start in range(0, len(ids), 400):
                    chunk = ids[start:start + 400]
                    ph = ",".join("?" * len(chunk))
                    cur = self._conn.execute(
                        f"delete from vec where memory_id in ({ph})", chunk
                    )
                    total += int(cur.rowcount or 0)
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise
        return total

    def clear(self) -> None:
        """清空索引，但保留模型/维度这类「配置快照」。

        保留它们是为了让状态页能回答「上一次是用哪个模型建的索引」——
        重建时真正要丢的是向量与同步指纹，不是配置。
        """
        with self._lock:
            self._conn.execute("begin immediate")
            try:
                self._conn.execute("delete from vec")
                self._conn.execute(
                    "insert into meta(key, value) values"
                    " ('indexed','0'), ('pending','0'), ('fingerprint',''),"
                    " ('last_error','')"
                    " on conflict(key) do update set value=excluded.value"
                )
                self._conn.execute("commit")
            except Exception:
                self._conn.execute("rollback")
                raise

    # ------------------------------------------------------------ 检索

    def search(
        self,
        query: Sequence[float],
        *,
        viewer: str,
        viewer_group: str = "",
        limit: int = 5,
        min_score: float = 0.0,
        scan_cap: int = 0,
    ) -> list[tuple[int, float]]:
        """返回 `[(memory_id, cosine)]`，按相似度降序。

        SQL 只做可见性**预过滤**（少算一些无关向量），最终归属仍由
        `MemoryService._visible` 用 `Memory.can_read` 复核——这里少写一个
        条件是性能问题，那边少写一个条件是泄密问题，所以两边都要有。

        `scan_cap > 0` 时按 `updated_ts` 降序截断。语义是「只在一个时间窗口
        里做精确检索」——没有 numpy 时的兜底，宁可少召回也不要把事件循环
        卡住几百毫秒。
        """
        dim = len(query)
        if dim <= 0:
            return []
        sql = (
            "select memory_id, vec, dim from vec where dim=?"
            " and (owner_id=? or visibility='public' or scope='global'"
            "      or (visibility='group' and group_id=? and ?<>''))"
            " order by updated_ts desc"
        )
        args: list = [dim, str(viewer or ""), str(viewer_group or ""),
                      str(viewer_group or "")]
        cap = int(scan_cap)
        if cap > 0:
            sql += " limit ?"
            args.append(cap)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return top_k(query, rows, k=limit, min_score=min_score)
