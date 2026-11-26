"""向量召回的编排层：查询补齐、增量索引、状态与重建。

三件事，刻意放在一起，因为它们共享同一套「哪个账号、哪个库、哪份配置」
的解析规则：

1. `semantic_hits()`：检索路径。字面召回不足时，按余弦相似度补几条。
2. `sync_bot()` / `sync_all()`：后台路径。按 `content_hash` 差集增量索引。
3. `status()` / `rebuild()` / `clear()`：管理台路径。

**为什么不做写路径同步**（存完记忆立刻调 embedding）：一次记忆写入会被
挂在用户回复之后，embedding 的网络延迟（几十到几百毫秒）会直接落到
「用户等回复」的链路上；而向量只是检索增强，晚几分钟生效完全可以接受。
后台轮询的另一半收益是天然合批：一轮总结写 30 条记忆，只发一次批量请求。

**为什么只索引归档记忆**：会话原文是记忆量的 100 倍以上，全部向量化会让
存储与 API 成本失控；而 `search_memory` 本来就只检索归档记忆。常驻词条
每轮都注入，不需要检索。
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from hermes_contracts import RecallHit
from hermes_vectors import (
    VECTOR_DB_FILE,
    EmbeddingClient,
    EmbeddingConfig,
    EmbeddingError,
    SqliteVectorStore,
    VectorRow,
    content_hash,
    has_numpy,
    pack,
)

from .registry import BotRegistry, bot_id_of
from .service import Scene

# 同一句查询的向量缓存时长。与注入去重窗口（45s）对齐：一次工具调用循环里
# 模型可能连搜两三次，第二次不该再付一次网络往返。
QUERY_TTL = 45.0
_MAX_QUERY_CACHE = 64
# 常驻打开的旁路库数量。100 个账号不能全开；按最近使用淘汰，开启成本很低。
_MAX_OPEN_STORES = 6
# 没有 numpy 时，纯 Python 点积的单次运算预算（乘加次数）。
# 50 万次约 15-25ms，是「还能放在一次工具调用里」的上限；同时限死最多
# 扫 2048 条（高维模型下再小的预算也扫不了几条，至少保证窗口有意义）。
_PURE_PY_OP_BUDGET = 500_000
_PURE_PY_MAX_ROWS = 2048


def _as_int(raw: object, default: int, *, low: int = 0,
            high: int = 10 ** 9) -> int:
    try:
        return min(high, max(low, int(str(raw).strip())))
    except (TypeError, ValueError):
        return default


def _as_float(raw: object, default: float, *, low: float = 0.0,
              high: float = 10.0) -> float:
    try:
        return min(high, max(low, float(str(raw).strip())))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class VectorSettings:
    """一份配置快照。启动时读一次，改配置需要重载插件（与其它资源类配置一致）。"""

    provider: str = "off"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = ""
    dim: int = 0
    batch_size: int = 10
    timeout_s: float = 20.0
    sync_minutes: int = 5
    sync_bots: int = 20
    sync_batch: int = 256
    min_score: float = 0.35

    def __post_init__(self) -> None:
        provider = str(self.provider or "off").strip().lower()
        object.__setattr__(self, "provider", provider if provider in ("off", "api") else "off")
        object.__setattr__(self, "base_url", str(self.base_url or "").strip().rstrip("/"))
        object.__setattr__(self, "api_key", str(self.api_key or "").strip())
        object.__setattr__(self, "model", str(self.model or "").strip())
        object.__setattr__(self, "dim", _as_int(self.dim, 0, high=8192))
        object.__setattr__(self, "batch_size", _as_int(self.batch_size, 10, low=1, high=128))
        object.__setattr__(self, "timeout_s", _as_float(self.timeout_s, 20.0, low=1.0, high=120.0))
        object.__setattr__(self, "sync_minutes", _as_int(self.sync_minutes, 5, low=1, high=1440))
        object.__setattr__(self, "sync_bots", _as_int(self.sync_bots, 20, low=1, high=200))
        object.__setattr__(self, "sync_batch", _as_int(self.sync_batch, 256, low=1, high=2000))
        object.__setattr__(self, "min_score", _as_float(self.min_score, 0.35, low=0.0, high=1.0))

    @classmethod
    def from_config(cls, get: Callable[[str, object], object]) -> "VectorSettings":
        """从插件配置构造。`get(key, default)` 与 `main._get_cfg` 同签名。"""

        def value(key: str, default: object) -> object:
            try:
                got = get(key, default)
            except Exception:
                return default
            return default if got is None else got

        return cls(
            provider=str(value("vector_provider", "off")),
            base_url=str(value("vector_base_url", "https://api.openai.com/v1")),
            api_key=str(value("vector_api_key", "")),
            model=str(value("vector_model", "")),
            dim=value("vector_dim", 0),
            batch_size=value("vector_batch_size", 10),
            timeout_s=value("vector_timeout_seconds", 20),
            sync_minutes=value("vector_sync_minutes", 5),
            sync_bots=value("vector_sync_bots", 20),
            sync_batch=value("vector_sync_batch", 256),
            min_score=value("vector_min_score", 0.35),
        )

    @property
    def enabled(self) -> bool:
        return self.provider == "api" and bool(self.base_url and self.model)

    @property
    def embed(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            base_url=self.base_url, model=self.model, api_key=self.api_key,
            dim=self.dim, batch_size=self.batch_size, timeout_s=self.timeout_s,
        )


class VectorGlue:
    """把「配置、旁路库、embedding 客户端、记忆库」粘成三个用例。"""

    def __init__(self, registry: BotRegistry, settings: VectorSettings, *,
                 log: Callable[[str], None] | None = None,
                 embedder: EmbeddingClient | None = None,
                 now: Callable[[], float] | None = None) -> None:
        self.registry = registry
        self.settings = settings
        self._embedder = embedder or EmbeddingClient(settings.embed)
        self._log = log or (lambda _msg: None)
        self._now = now or time.time
        self._mono = time.monotonic
        self._stores: OrderedDict[str, SqliteVectorStore] = OrderedDict()
        # 正在被某个线程使用的旁路库。淘汰时必须跳过它们：`store.search`
        # 在 `to_thread` 里跑，另一边一个高频查询就能把它的连接关掉，表现为
        # 偶发的 `Cannot operate on a closed database`。
        self._inflight: dict[str, int] = {}
        self._qcache: dict[str, tuple[float, list[float]]] = {}
        self._cursor = 0
        self.last_error = ""
        self.stats = {"queries": 0, "filled": 0, "fallbacks": 0, "indexed": 0}

    # ------------------------------------------------------------ 状态

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def layout_of(self, platform_id: str) -> str:
        """`v2` / `legacy` / `missing`。

        只给 v2（每账号目录）建索引：v1 的库平铺在根目录，旁路库若也放根目录
        就会变成**全账号共享**一个向量库，正好破坏「一个账号一个宇宙」的前提。
        v1 的账号等迁移完成后再建即可。
        """
        pid = str(platform_id or "")
        if not pid or not self.registry.has_account(pid):
            return "missing"
        db = os.path.abspath(self.registry.db_path(pid))
        expected = os.path.abspath(os.path.join(self.registry.bot_dir(pid), "memory.db"))
        if db != expected:
            return "legacy"
        return "v2" if os.path.isfile(db) else "missing"

    def _store_for(self, platform_id: str, *, create: bool = True):
        if self.layout_of(platform_id) != "v2":
            return None
        path = os.path.join(self.registry.bot_dir(platform_id), VECTOR_DB_FILE)
        store = self._stores.get(platform_id)
        if store is not None and store.path == path:
            self._stores.move_to_end(platform_id)
            return store
        if store is not None:  # 路径变了（迁移完成）：旧连接不能再用
            self._stores.pop(platform_id, None)
            store.close()
        if not create and not os.path.isfile(path):
            return None
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            store = SqliteVectorStore(path)
        except Exception as exc:  # noqa: BLE001 — 旁路库坏了不能拖垮记忆
            # 向量是可重建的派生数据：把坏文件改名留档（不删，便于排查磁盘
            # 故障），然后立刻建一个空库。下一次后台同步会自动补回来。
            self._log(f"[hermes_fleet] {path} 打不开，按损坏处理：{exc}")
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.replace(path + suffix, path + suffix + ".corrupt")
                except OSError:
                    pass
            if os.path.exists(path):
                # rename 也被占住（杀毒软件/另一个进程）：派生的坏库不值得
                # 为它卡住整个状态页，直接删掉后重建。
                try:
                    os.remove(path)
                except OSError:
                    pass
            store = SqliteVectorStore(path)
        self._stores[platform_id] = store
        while len(self._stores) > _MAX_OPEN_STORES:
            victim = next(
                (pid for pid in self._stores if not self._inflight.get(pid)),
                None,
            )
            if victim is None:
                # 全都在用：宁可暂时超限，也不关掉别人正在读的连接。
                break
            _pid, old = self._stores.pop(victim)
            old.close()
        return store

    def _hold(self, platform_id: str) -> None:
        self._inflight[platform_id] = self._inflight.get(platform_id, 0) + 1

    def _drop(self, platform_id: str) -> None:
        left = self._inflight.get(platform_id, 0) - 1
        if left > 0:
            self._inflight[platform_id] = left
        else:
            self._inflight.pop(platform_id, None)

    async def _clear_store(self, platform_id: str, store) -> None:
        self._hold(platform_id)
        try:
            await asyncio.to_thread(store.clear)
        finally:
            self._drop(platform_id)

    async def status(self, platform_id: str) -> dict:
        pid = str(platform_id or "")
        layout = self.layout_of(pid)
        out = {
            "platform_id": pid,
            "enabled": self.enabled,
            "provider": self.settings.provider,
            "model": self.settings.model,
            "dim": self.settings.dim,
            "min_score": self.settings.min_score,
            "numpy": has_numpy(),
            "scan_cap": self._scan_cap(0),
            "layout": layout,
            "indexed": 0,
            "active": 0,
            "pending": 0,
            "last_sync_ts": 0.0,
            "last_error": self.last_error,
        }
        if layout != "v2":
            return out
        store = self._store_for(pid, create=False)
        if store is not None:
            self._hold(pid)
            try:
                meta = await asyncio.to_thread(store.meta)
                out["indexed"] = await asyncio.to_thread(store.count)
            finally:
                self._drop(pid)
            out["pending"] = _as_int(meta.get("pending"), 0)
            out["last_sync_ts"] = _as_float(meta.get("last_sync_ts"), 0.0, high=1e12)
            out["last_error"] = str(meta.get("last_error") or "") or self.last_error
            if meta.get("model"):
                out["model"] = str(meta.get("model"))
            if meta.get("dim"):
                out["dim"] = _as_int(meta.get("dim"), 0)
            out["scan_cap"] = self._scan_cap(out["dim"] or 0)
        try:
            db = self.registry.get(pid)
            out["active"] = await db.memories.count_active(bot_id_of(pid))
        except Exception:
            pass
        return out

    def describe(self) -> str:
        """给启动日志用的一句话摘要。"""
        fast = "" if has_numpy() else " · 未装 numpy，只扫最近窗口"
        return (
            f"{'启用' if self.enabled else '未启用'} · "
            f"{self.settings.model or '未配置模型'} · "
            f"相似度门槛 {self.settings.min_score:.2f}{fast}"
        )

    # ------------------------------------------------------------ 查询

    @staticmethod
    def _scan_cap(dim: int) -> int:
        """单次检索最多扫多少条向量。0 = 不限制（有 numpy 时）。

        没有 numpy 时用「运算预算 ÷ 维度」估一个窗口，再夹在 128～2048：
        50 万次乘加在纯 Python 里约 20ms，而 50 个并发会把这段时间按 GIL
        串起来——所以窗口必须小。代价是只能在这个「最近更新」窗口里做精确
        检索，超出部分仍由字面召回兜底。要全库精确检索就装 numpy（推荐）。
        """
        if has_numpy():
            return 0
        d = max(1, int(dim or 256))
        return max(128, min(_PURE_PY_MAX_ROWS, _PURE_PY_OP_BUDGET // d))

    async def _query_vector(self, text: str) -> list[float]:
        key = str(text or "").strip()
        if not key or not self.enabled:
            return []
        now = self._mono()
        hit = self._qcache.get(key)
        if hit is not None and now - hit[0] < QUERY_TTL:
            return hit[1]
        try:
            vec = await self._embedder.embed_one(key)
        except EmbeddingError as exc:
            self.last_error = str(exc)
            self.stats["fallbacks"] += 1
            self._log(f"[hermes_fleet] 查询向量化失败，退回字面召回：{exc}")
            return []
        except Exception as exc:  # noqa: BLE001 — 任何异常都不能打断检索
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.stats["fallbacks"] += 1
            return []
        if not vec:
            return []
        self.last_error = ""
        if len(self._qcache) >= _MAX_QUERY_CACHE:
            cutoff = now - QUERY_TTL
            for k in [k for k, (ts, _) in self._qcache.items() if ts < cutoff]:
                self._qcache.pop(k, None)
            while len(self._qcache) >= _MAX_QUERY_CACHE:
                self._qcache.pop(next(iter(self._qcache)), None)
        self._qcache[key] = (now, vec)
        return vec

    async def semantic_hits(self, scene: Scene, query: str, *,
                            exclude: set[int] | None = None,
                            limit: int) -> list[RecallHit]:
        """按余弦相似度取几条，供 `MemoryService._semantic_fill` 追加。

        返回的条目**未做最终可见性判断**：调用方（service）会用 `_visible`
        复核。这里做 SQL 预过滤只是为了少算一些无关向量。
        """
        take = int(limit)
        if take <= 0 or not self.enabled:
            return []
        self.stats["queries"] += 1
        vec = await self._query_vector(query)
        if not vec:
            return []
        store = self._store_for(str(scene.bot_id))
        if store is None:
            return []
        skip = {int(i) for i in (exclude or set())}
        cap = self._scan_cap(len(vec))
        self._hold(str(scene.bot_id))
        try:
            rows = await asyncio.to_thread(
                store.search, vec,
                viewer=str(scene.owner_id),
                viewer_group=scene.viewer_group,
                limit=take + len(skip),
                min_score=self.settings.min_score,
                scan_cap=cap,
            )
        finally:
            self._drop(str(scene.bot_id))
        ids = [mid for mid, _ in rows if int(mid) not in skip][:take]
        if not ids:
            return []
        db = self.registry.get(str(scene.bot_id))
        memories = await db.memories.get_many(scene.bot_id, ids)
        by_id = {int(m.id): m for m in memories if m.is_active}
        out: list[RecallHit] = []
        for memory_id, score in rows:
            memory = by_id.get(int(memory_id))
            if memory is None or int(memory_id) in skip:
                continue
            out.append(RecallHit(memory=memory, score=float(score),
                                 debug={"cosine": float(score)}))
            if len(out) >= take:
                break
        self.stats["filled"] += len(out)
        return out

    # ------------------------------------------------------------ 索引

    async def _prepare(self, platform_id: str, store) -> dict:
        """模型/维度变更时清空索引，返回最新 meta。

        「旧模型的向量还能不能凑合用」不是可以商量的事：两个模型的向量
        空间互不相通，混在一起算余弦只会得到一堆看起来合理、实际随机
        的分数。所以模型或维度一变就整体重建，宁可暂时没有向量召回。
        """
        meta = await asyncio.to_thread(store.meta)
        model_changed = bool(meta.get("model")) and meta.get("model") != self.settings.model
        old_dim = _as_int(meta.get("dim"), 0)
        dim_changed = bool(old_dim and self.settings.dim and old_dim != self.settings.dim)
        if model_changed or dim_changed:
            await asyncio.to_thread(store.clear)
            self._log(
                f"[hermes_fleet] {platform_id} 的向量模型/维度变了"
                f"（{meta.get('model')}/{old_dim} → "
                f"{self.settings.model}/{self.settings.dim}），已清空待重建"
            )
            return {}
        return meta

    async def sync_bot(self, platform_id: str, *, budget: int | None = None,
                       store=None) -> dict:
        """对一个账号做一轮增量索引。返回 `{indexed, pending, skipped?}`。"""
        if not self.enabled:
            return {"skipped": True, "reason": "disabled", "indexed": 0, "pending": 0}
        store = store or self._store_for(platform_id)
        if store is None:
            return {"skipped": True, "reason": "layout", "indexed": 0, "pending": 0}
        # 索引会跨多个 await（embedding 网络调用），期间必须钉住这个连接：
        # 否则一次并发检索触发的 LRU 淘汰会把 store 关掉。
        self._hold(platform_id)
        try:
            return await self._sync_bot(platform_id, budget=budget, store=store)
        finally:
            self._drop(platform_id)

    async def _sync_bot(self, platform_id: str, *, budget: int | None,
                        store) -> dict:
        db = self.registry.get(platform_id)
        bot = bot_id_of(platform_id)
        meta = await self._prepare(platform_id, store)
        fp = await db.memories.fingerprint(bot)
        stamp = f"{fp['count']}:{fp['max_id']}:{fp['max_updated']:.3f}"
        indexed = await asyncio.to_thread(store.count)
        if meta.get("fingerprint") == stamp and indexed == int(fp["count"]):
            return {"skipped": True, "reason": "clean", "indexed": indexed,
                    "pending": 0}

        rows = await db.memories.iter_active(bot)
        hashes = await asyncio.to_thread(store.hashes)
        active_ids = {int(mid) for mid, _text, _ts in rows}
        stale = [mid for mid in hashes if mid not in active_ids]
        if stale:
            await asyncio.to_thread(store.delete, stale)

        need = [
            (int(mid), text, ts) for mid, text, ts in rows
            if hashes.get(int(mid)) != content_hash(text)
        ]
        cap = max(1, int(budget or self.settings.sync_batch))
        batch = need[:cap]
        pending = max(0, len(need) - len(batch))

        dim = _as_int(meta.get("dim"), 0)
        last_error = ""
        if batch:
            try:
                vectors = await self._embedder.embed([text for _i, text, _ts in batch])
            except EmbeddingError as exc:
                last_error = str(exc)
                self.last_error = last_error
                await asyncio.to_thread(store.set_meta, last_error=last_error)
                return {"indexed": 0, "pending": len(need), "error": last_error}
            if len(vectors) != len(batch) or not vectors:
                return {"indexed": 0, "pending": len(need),
                        "error": "embedding 返回条数不符"}
            dim = len(vectors[0])
            # 归属字段必须从记忆行补：向量库只存 id → 向量，检索时的
            # SQL 预过滤要按 owner/group/visibility 走，填错的后果是
            # 「明明有这个人的记忆，向量路径却永远筛不到」。
            known = {
                int(m.id): m
                for m in await db.memories.get_many(
                    bot, [mid for mid, _text, _ts in batch])
            }
            payload = [
                VectorRow(
                    memory_id=mid,
                    owner_id=str(known[mid].owner_id or "") if mid in known else "",
                    group_id=str(known[mid].group_id or "") if mid in known else "",
                    scope=str(known[mid].scope) if mid in known else "user",
                    visibility=str(known[mid].visibility) if mid in known else "private",
                    content_hash=content_hash(text),
                    updated_ts=float(ts), dim=len(vec), vec=pack(vec),
                )
                for (mid, text, ts), vec in zip(batch, vectors)
            ]
            await asyncio.to_thread(store.upsert, payload)
            self.stats["indexed"] += len(payload)

        indexed = await asyncio.to_thread(store.count)
        await asyncio.to_thread(
            store.set_meta,
            model=self.settings.model,
            dim=dim,
            indexed=indexed,
            pending=pending,
            last_sync_ts=self._now(),
            last_error=last_error,
            fingerprint=stamp if pending == 0 else "",
        )
        return {"indexed": indexed, "pending": pending, "added": len(batch)}

    async def sync_all(self, *, mode_of: Callable[[str], str] | None = None,
                       max_bots: int | None = None) -> dict:
        """按游标轮转若干账号。`mode_of` 用来跳过不需要向量索引的账号。

        跳过 classic 账号不是省代码，是省钱：100 个账号每轮都对所有新记忆
        调 embedding，而其中可能一个都没开向量召回。
        """
        out = {"bots": 0, "added": 0, "errors": 0}
        if not self.enabled:
            return out
        pids = [
            str(row.get("platform_id") or "")
            for row in self.registry.accounts()
            if not row.get("missing")
        ]
        pids = [pid for pid in pids if pid and self.layout_of(pid) == "v2"]
        if mode_of is not None:
            pids = [pid for pid in pids if mode_of(pid) == "vector"]
        if not pids:
            return out
        limit = max(1, int(max_bots or self.settings.sync_bots))
        start = self._cursor % len(pids)
        picks = [pids[(start + i) % len(pids)] for i in range(min(limit, len(pids)))]
        self._cursor = (start + len(picks)) % len(pids)
        for pid in picks:
            was_open = self.registry.is_open(pid)
            try:
                res = await self.sync_bot(pid)
                out["bots"] += 1
                out["added"] += int(res.get("added") or 0)
                if res.get("error"):
                    out["errors"] += 1
                    self._log(f"[hermes_fleet] {pid} 索引失败：{res['error']}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — 单个账号坏不能停整轮
                out["errors"] += 1
                self._log(f"[hermes_fleet] {pid} 索引异常：{type(exc).__name__}: {exc}")
            finally:
                if not was_open:
                    try:
                        self.registry.close_bot(pid)
                    except Exception:
                        pass
        return out

    async def rebuild(self, platform_id: str, *, batches: int = 8) -> dict:
        """清空并重建一个账号的索引。按批返回，避免一次请求跑几分钟。"""
        if not self.enabled:
            return {"enabled": False, "indexed": 0, "pending": 0,
                    "message": "未启用向量召回：请先在设置里填写 vector_model"}
        store = self._store_for(platform_id)
        if store is None:
            return {"enabled": True, "indexed": 0, "pending": 0,
                    "message": "该账号不在 v2 目录布局里，暂不建索引"}
        was_open = self.registry.is_open(platform_id)
        error = ""
        try:
            await self._clear_store(platform_id, store)
            for _ in range(max(1, min(50, int(batches)))):
                res = await self.sync_bot(platform_id, store=store)
                error = str(res.get("error") or "")
                if error or int(res.get("pending") or 0) <= 0:
                    break
        finally:
            if not was_open:
                try:
                    self.registry.close_bot(platform_id)
                except Exception:
                    pass
        self._hold(platform_id)
        try:
            indexed = await asyncio.to_thread(store.count)
            meta = await asyncio.to_thread(store.meta)
        finally:
            self._drop(platform_id)
        pending = _as_int(meta.get("pending"), 0)
        return {
            "enabled": True,
            "indexed": indexed,
            "pending": pending,
            "error": error,
            "message": (
                f"已索引 {indexed} 条，剩余 {pending} 条（下一轮后台任务继续）"
                if pending else f"已索引 {indexed} 条，全部完成"
            ),
        }

    async def clear(self, platform_id: str) -> dict:
        store = self._store_for(platform_id)
        if store is None:
            return {"cleared": False, "indexed": 0,
                    "message": "该账号没有可清理的向量索引"}
        await self._clear_store(platform_id, store)
        return {"cleared": True, "indexed": 0, "message": "向量索引已清空"}

    def close(self) -> None:
        for _pid, store in list(self._stores.items()):
            try:
                store.close()
            except Exception:
                pass
        self._stores.clear()
        self._qcache.clear()
