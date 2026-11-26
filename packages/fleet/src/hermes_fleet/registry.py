"""账号 → 记忆库的路由层。

这一层只回答一个问题：**这条消息属于哪个库文件。**

为什么单独成层，而不是在注入函数里现算路径：
路径推算一旦散落在多个调用点，就会出现「注入读 A 库、落库写 B 库」这种
静默错位——两边都不报错，只表现为「记忆时好时坏」。所有路径计算必须
只有一处实现。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time

from hermes_contracts import BotId, Clock
from hermes_storage_sqlite import SqliteBotDb

from .settings import as_flag

UNKNOWN_BOT = "default"

# ---------------------------------------------------------------- 磁盘布局
#
# v2：每个账号一个**目录**，库与配置放在一起：
#
#     <root>/layout.json          布局版本（只标记，不参与读路径）
#     <root>/bots/<stem>/bot.json 账号配置（备注、描述、按账号开关）
#     <root>/bots/<stem>/memory.db
#
# 为什么值得改：v1 把备注塞在根目录的 `_notes.json` 里，于是「备份一个
# 账号」和「备份它的名字」是两件事，删库也不会带走备注。目录化之后一个
# 账号的全部可搬移状态都在一处，打包/恢复/迁移都只需要拷目录。
#
# 兼容策略：读同时认 v1（根目录平铺 `<stem>.db`）与 v2，写只写 v2，
# 由 `migrate_v2()` 幂等地把旧库搬进新布局。迁移失败时旧文件原地不动。
LAYOUT_FILE = "layout.json"
LAYOUT_SCHEMA = 2
BOTS_DIR = "bots"
BOT_META_FILE = "bot.json"
BOT_DB_FILE = "memory.db"
META_SCHEMA = 1
MAX_DESCRIPTION = 200

# 可以按账号单独覆盖的开关。只收「策略」类开关：模型链、并发、间隔属于
# 全局资源，按账号拆开会让「为什么这个号半夜在跑」无法归因。
OVERRIDE_BOOLS = (
    "capture_messages",
    "summarize_enabled",
    "consolidate_enabled",
    "group_private_memory",
)
OVERRIDE_INT_RANGES = {"recall_k": (1, 20)}
OVERRIDE_KEYS = OVERRIDE_BOOLS + tuple(OVERRIDE_INT_RANGES)

# 迁移与备份校验时统计的表。表不存在（老库/未来删表）按 0 算，不阻断。
_TABLES = ("messages", "memories", "resident", "identities")

# 文件名的可读部分：只留 ASCII 安全字符。Windows 上 `:` `?` `*` 不能做文件名，
# 而 AstrBot 的适配器 id 是用户自己填的，什么都可能出现。
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

# 备注：给人看的名字，与 `platform_id` 分开存。
#
# 为什么不写进库、也不改文件名：`platform_id` 是适配器 id，改名等于换账号
# （记忆会「消失」）；文件名里的 hash 一旦变了，索引和磁盘就对不上。备注
# 只是显示层的别名，所以单独一个 json，删库、换库都不影响它。
# 长度上限 40 字符：够写「客服机器人（测试）」，再长会把表格挤变形。
NOTES_FILE = "_notes.json"
MAX_NOTE = 40
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_SPACES = re.compile(r"\s+")


def normalize_note(text: object) -> str:
    """把任意输入压成一行备注。

    控制字符必须替换掉：备注会被渲染进页面和告警条，一条带 `\x00` 或换行的
    备注会把整张表格的排版打乱，而这类脏数据最常来自粘贴。
    """
    raw = _CONTROL.sub(" ", str(text if text is not None else ""))
    return _SPACES.sub(" ", raw).strip()[:MAX_NOTE]


def normalize_description(text: object) -> str:
    """把任意输入压成一行账号描述（≤200 字）。

    与备注同一套清洗规则，但长度上限更大：备注要进表格，描述是详情页里
    给人看的说明。**描述绝不进提示词**——它是运维笔记，不是人设；把
    「这个号测试用的」注入给模型只会污染发言风格。
    """
    raw = _CONTROL.sub(" ", str(text if text is not None else ""))
    return _SPACES.sub(" ", raw).strip()[:MAX_DESCRIPTION]


def clean_overrides(raw: object) -> dict:
    """把磁盘/前端来的按账号开关清洗成可信值。

    非法值直接丢弃而不是抛错：这份配置可能来自手改的 json、旧版本面板或
    恢复包，读的时候严格抛错会让整个账号页打不开。写的时候
    （`set_overrides`）才严格校验——那是用户当下的输入，报错才有意义。
    """
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key in OVERRIDE_BOOLS:
        if key in raw and raw[key] is not None:
            try:
                out[key] = as_flag(raw[key], False)
            except (TypeError, ValueError):
                continue
    for key, (low, high) in OVERRIDE_INT_RANGES.items():
        if key in raw and raw[key] is not None:
            try:
                value = int(raw[key])
            except (TypeError, ValueError):
                continue
            out[key] = min(high, max(low, value))
    return out


def _default_meta(platform_id: object) -> dict:
    return {
        "schema": META_SCHEMA,
        "platform_id": str(platform_id or ""),
        "note": "",
        "description": "",
        "created_at": 0.0,
        "overrides": {},
    }


def _db_snapshot(path: str) -> dict:
    """只读打开一个库，返回完整性结论与各表行数。

    迁移前后各取一次：移动文件本身几乎不会丢数据，但 checkpoint、杀毒软件
    占锁、磁盘写满都会。只相信 `integrity_check` 不够——一个完整但空的库
    也能通过检查，所以同时比对行数。
    """
    uri = "file:" + path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=15.0)
    try:
        verdict = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        except (TypeError, ValueError, sqlite3.DatabaseError):
            version = 0
        counts: dict[str, int] = {}
        for table in _TABLES:
            try:
                row = conn.execute(f"select count(*) from {table}").fetchone()
                counts[table] = int(row[0] if row else 0)
            except sqlite3.DatabaseError:
                counts[table] = 0
        return {"integrity": verdict, "user_version": version, "counts": counts}
    finally:
        conn.close()


def _checkpoint(path: str) -> None:
    """把 `-wal` 合并进主库并截断。

    迁移前必须做：直接 `os.replace` 主库文件会把最近一批写入留在原地的
    `-wal` 里，搬走的主库看起来完好，实际少了最后几条记忆。
    """
    conn = sqlite3.connect(path, timeout=15.0)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _atomic_json(path: str, data: dict) -> None:
    """写 json：先写临时文件再 rename。

    直接覆盖时若进程被 kill，会留下半个 JSON，下次启动整张表读不出来。
    索引和备注都是「丢了就等于不认识自己的数据」，所以都走这条路径。
    """
    import json
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def bot_file_stem(platform_id: str) -> str:
    """把适配器 id 变成稳定、可读、不会撞的文件名主干。

    为什么带 hash 后缀：只用 slug 时，`qq-1` 和 `qq_1` 会折叠成同一个
    `qq-1`，两个账号共用一个库——正是本插件要杜绝的事故。加 8 位
    hash 让不同 id 必然不同，同时前缀保留可读性（排障时能看出是哪个号）。

    为什么不用 sha1 全长：文件名要能人眼扫过去，前缀 + 短 hash 足够。
    8 位 hex 在 100 个账号规模下碰撞概率约 2e-7。
    """
    raw = str(platform_id or UNKNOWN_BOT).strip() or UNKNOWN_BOT
    slug = _SAFE.sub("-", raw).strip("-")[:32] or UNKNOWN_BOT
    digest = hashlib.sha1(raw.encode("utf-8", "surrogatepass")).hexdigest()[:8]
    return f"{slug}-{digest}"


class BotRegistry:
    """`platform_id` → `SqliteBotDb`，带容量上限的惰性打开。

    为什么要 LRU 上限：100 个账号若全部常驻，就是 100 个 sqlite 连接 +
    100 份页缓存。空转时这是纯浪费，而冷启动一个库实测只要 7.6ms
    （2 万消息 + 4000 记忆），关掉再开完全可接受。

    为什么淘汰时先 checkpoint 再关：`-wal` 里可能还有未合并的页。
    直接 `os.remove` 会在 Windows 上失败（文件被占用），在 POSIX 上
    则可能删掉主库却留下 wal，下次打开时 wal 被当成有效日志重放。
    """

    def __init__(self, root: str, *, max_open: int = 24,
                 clock: Clock | None = None) -> None:
        self.root = root
        self._max_open = max(1, int(max_open))
        self._clock = clock
        self._lock = threading.RLock()
        self._open: dict[str, SqliteBotDb] = {}
        self._last_used: dict[str, float] = {}
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self._bots_root(), exist_ok=True)
        # 每轮注入都要问一次「这个号开了哪些开关」，不能每条消息都读一遍
        # bot.json。缓存以文件内容为准：所有写入都走 `_write_meta()`。
        self._meta_cache: dict[str, dict | None] = {}
        self._index_path = os.path.join(self.root, "_index.json")
        self._index: dict[str, str] = self._load_index()
        self._notes_path = os.path.join(self.root, NOTES_FILE)
        # `_pending` 是还要写进 `_notes.json` 的那部分（账号还没落库）；
        # `_notes` 是内存里的全集（含 bot.json 里的）。两者分开是为了
        # 迁移完成后 `_notes.json` 能自然清空，而不是永远留一份影子。
        self._pending: dict[str, str] = {}
        self._notes: dict[str, str] = {}
        self._pending, self._notes = self._load_notes()
        self._sync_index()

    # ------------------------------------------------------------ 账号索引

    def _bots_root(self) -> str:
        return os.path.join(self.root, BOTS_DIR)

    def _bot_dir_names(self) -> list[str]:
        """`bots/` 下的账号目录名（只读文件名，不打开库）。"""
        try:
            root = self._bots_root()
            return sorted(
                n for n in os.listdir(root)
                if os.path.isdir(os.path.join(root, n))
            )
        except OSError:
            return []

    def _legacy_db_names(self) -> list[str]:
        """v1 布局遗留在根目录的 `.db` 文件名。"""
        try:
            return sorted(
                f for f in os.listdir(self.root)
                if f.endswith(".db") and os.path.isfile(os.path.join(self.root, f))
            )
        except OSError:
            return []

    def _load_index(self) -> dict[str, str]:
        """`文件名主干 -> 适配器 id`。

        为什么需要这张表：后台总结要回答「现在有哪些账号」，而文件名是
        `qq-01-a1b2c3d4` 这种不可逆的编码（slug + hash）。没有索引，
        重启后就只认识「这次启动之后说过话的账号」——一个昨天攒了 200 条
        待总结消息、今天再没人说话的账号，会被永久跳过。

        v2 的真相在各账号的 `bot.json` 里，所以这里读到的只是**缓存**：
        随后用磁盘上的 bot.json 覆盖同名条目，缓存缺失或过期都能自愈。
        """
        try:
            with open(self._index_path, encoding="utf-8") as fp:
                data = json.load(fp)
            index = ({str(k): str(v) for k, v in data.items()}
                     if isinstance(data, dict) else {})
        except FileNotFoundError:
            index = {}
        except Exception:
            index = {}
        for stem in self._bot_dir_names():
            meta = self._read_meta(stem)
            pid = str((meta or {}).get("platform_id") or "").strip()
            if pid:
                index[stem] = pid
        return {k: v for k, v in index.items() if k and v}

    def _sync_index(self) -> None:
        """把磁盘上新增/变更的 bot.json 补进缓存并落盘。"""
        with self._lock:
            before = dict(self._index)
            for stem in self._bot_dir_names():
                meta = self._read_meta(stem)
                pid = str((meta or {}).get("platform_id") or "").strip()
                if pid:
                    self._index[stem] = pid
            if self._index != before:
                self._save_index()

    def _save_index(self) -> None:
        try:
            _atomic_json(self._index_path, self._index)
        except Exception:
            pass

    def known_bots(self) -> list[str]:
        """已知的全部账号（不打开任何连接）。"""
        with self._lock:
            return sorted(set(self._index.values()))

    # ------------------------------------------------------------ 备注

    def _load_notes(self) -> tuple[dict[str, str], dict[str, str]]:
        """读备注，返回 `(还要写回 _notes.json 的, 全集)`。

        v2 之后备注住在各自的 `bot.json` 里；`_notes.json` 只保留「账号还
        没落库、先起了个名」的临时条目。两份都读、bot.json 优先，是为了让
        升级到一半的现场也能显示正确的名字。
        """
        pending: dict[str, str] = {}
        try:
            with open(self._notes_path, encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                for k, v in data.items():
                    note = normalize_note(v)
                    if note:
                        pending[str(k)] = note
        except FileNotFoundError:
            pass
        except Exception:
            pass

        merged: dict[str, str] = {}
        for stem in self._bot_dir_names():
            meta = self._read_meta(stem)
            pid = str((meta or {}).get("platform_id") or "").strip()
            note = str((meta or {}).get("note") or "")
            if pid and note:
                merged[pid] = note
        for pid, note in pending.items():
            merged.setdefault(pid, note)
        return pending, merged

    def _save_notes(self) -> None:
        try:
            _atomic_json(self._notes_path, self._pending)
        except Exception:
            pass

    def note_of(self, platform_id: str) -> str:
        with self._lock:
            return self._notes.get(str(platform_id or UNKNOWN_BOT), "")

    def notes(self) -> dict[str, str]:
        with self._lock:
            return dict(self._notes)

    def set_note(self, platform_id: str, text: object) -> str:
        """设置或清除备注，返回落盘后的值（空串＝已清除）。

        对**没登记过的账号**也允许写：备注是给人看的，先起名再等账号说话
        是正常操作顺序；这类备注暂存在 `_notes.json`，等库文件出现时
        （`get()`）自动搬进该账号的 `bot.json`。`accounts()` 只按磁盘上
        的账号列行，所以不会凭空多出一个假账号。
        """
        pid = str(platform_id or UNKNOWN_BOT).strip() or UNKNOWN_BOT
        note = normalize_note(text)
        with self._lock:
            if self._notes.get(pid) == note:
                return note
            if note:
                self._notes[pid] = note
            else:
                self._notes.pop(pid, None)

            stem = bot_file_stem(pid)
            if self._read_meta(stem) is not None:
                meta = self._read_meta(stem) or _default_meta(pid)
                meta["note"] = note
                self._write_meta(stem, meta)
                self._pending.pop(pid, None)
            elif note:
                self._pending[pid] = note
            else:
                self._pending.pop(pid, None)
            self._save_notes()
        return note

    # ------------------------------------------------------------ 账号配置

    def _meta_path(self, stem: str) -> str:
        return os.path.join(self._bots_root(), stem, BOT_META_FILE)

    def _read_meta(self, stem: str) -> dict | None:
        """读某个账号的 `bot.json`，读不到/坏了返回 None。

        读失败不抛错：一个损坏的 bot.json 只该让「备注/开关」退回默认，
        不该让整个账号打不开——库本身还是好的。
        """
        if stem in self._meta_cache:
            cached = self._meta_cache[stem]
            return dict(cached) if cached else None
        try:
            with open(self._meta_path(stem), encoding="utf-8") as fp:
                data = json.load(fp)
        except FileNotFoundError:
            self._meta_cache[stem] = None
            return None
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        pid = str(data.get("platform_id")
                  or getattr(self, "_index", {}).get(stem, "") or "")
        try:
            created = float(data.get("created_at") or 0.0)
        except (TypeError, ValueError):
            created = 0.0
        meta = _default_meta(pid)
        meta["note"] = normalize_note(data.get("note"))
        meta["description"] = normalize_description(data.get("description"))
        meta["created_at"] = created
        meta["overrides"] = clean_overrides(data.get("overrides"))
        self._meta_cache[stem] = dict(meta)
        return meta

    def _write_meta(self, stem: str, meta: dict) -> None:
        path = self._meta_path(stem)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_json(path, meta)
        self._meta_cache[stem] = dict(meta)

    def _ensure_meta(self, platform_id: str) -> None:
        """账号第一次落库时补上 bot.json（并接住暂存备注）。

        只在 v2 路径上做：v1 的老库保持原样，等 `migrate_v2()` 统一搬，
        否则会造出「bot.json 在 v2、库还在 v1」的分裂状态。
        """
        pid = str(platform_id or UNKNOWN_BOT).strip() or UNKNOWN_BOT
        stem = bot_file_stem(pid)
        existing = self._read_meta(stem)
        if existing is None:
            if self.db_path(pid) != self._v2_db_path(stem):
                return
            meta = _default_meta(pid)
            meta["created_at"] = time.time()
        else:
            meta = existing
        note = self._pending.pop(pid, "")
        if not note:
            if existing is None:
                self._write_meta(stem, meta)
            return
        if meta.get("note") != note:
            meta["note"] = note
        self._write_meta(stem, meta)
        self._save_notes()

    def bot_meta(self, platform_id: str) -> dict:
        """返回账号配置（没有 bot.json 时给默认值，不写盘）。"""
        pid = str(platform_id or UNKNOWN_BOT).strip() or UNKNOWN_BOT
        meta = self._read_meta(bot_file_stem(pid))
        if meta is not None:
            return meta
        meta = _default_meta(pid)
        with self._lock:
            meta["note"] = self._notes.get(pid, "")
        return meta

    def _require_meta(self, platform_id: str) -> tuple[str, dict]:
        pid = str(platform_id or UNKNOWN_BOT).strip() or UNKNOWN_BOT
        stem = bot_file_stem(pid)
        meta = self._read_meta(stem)
        if meta is None:
            raise KeyError("该账号还没有落库，不能保存配置")
        return stem, meta

    def set_description(self, platform_id: str, text: object) -> str:
        """保存账号描述（仅供展示/导出，不进提示词）。"""
        with self._lock:
            stem, meta = self._require_meta(platform_id)
            meta["description"] = normalize_description(text)
            self._write_meta(stem, meta)
            return meta["description"]

    def description_of(self, platform_id: str) -> str:
        return str(self.bot_meta(platform_id).get("description") or "")

    def overrides_of(self, platform_id: str) -> dict:
        """该账号显式覆盖的开关（空 dict＝全部跟随全局）。"""
        return dict(self.bot_meta(platform_id).get("overrides") or {})

    def set_overrides(self, platform_id: str, updates: object) -> dict:
        """按账号改开关；值为 `None` 表示删除该覆盖、回到全局配置。

        未知键与类型错误一律抛 `ValueError`：这是用户当下的表单输入，
        静默丢弃会让「保存成功但没生效」变得无法排查。
        """
        if not isinstance(updates, dict):
            raise ValueError("开关必须是对象")
        unknown = sorted(set(updates) - set(OVERRIDE_KEYS))
        if unknown:
            raise ValueError("不支持的开关：" + "、".join(unknown))
        with self._lock:
            stem, meta = self._require_meta(platform_id)
            merged = dict(meta.get("overrides") or {})
            for key, value in updates.items():
                if value is None:
                    merged.pop(key, None)
                    continue
                cleaned = clean_overrides({key: value})
                if key not in cleaned:
                    raise ValueError(f"{key} 的取值不合法")
                merged[key] = cleaned[key]
            meta["overrides"] = merged
            self._write_meta(stem, meta)
            return dict(merged)

    # ------------------------------------------------------------ 路径

    def _v2_db_path(self, stem: str) -> str:
        return os.path.join(self._bots_root(), stem, BOT_DB_FILE)

    def bot_dir(self, platform_id: str) -> str:
        """该账号的目录（v2）。账号仍是 v1 时返回的是「将来会搬去的位置」。"""
        return os.path.join(self._bots_root(), bot_file_stem(platform_id))

    def db_path(self, platform_id: str) -> str:
        """账号库文件的真实路径：已迁到 v2 读 v2，老库仍读根目录。

        这是全插件**唯一**的库路径计算点。新增写路径时绝不能自己拼
        `bots/...`，否则「注入读 A、落库写 B」的静默错位会立刻回来。
        """
        stem = bot_file_stem(platform_id)
        v2 = self._v2_db_path(stem)
        if os.path.exists(v2) or os.path.exists(self._meta_path(stem)):
            return v2
        legacy = os.path.join(self.root, stem + ".db")
        if os.path.exists(legacy):
            return legacy
        return v2

    # ------------------------------------------------------------ 账号盘点

    def accounts(self) -> list[dict]:
        """列出全部账号及其文件元数据。**不打开任何数据库连接。**

        这是 WebUI 账号索引页的唯一数据源。为什么必须零连接：`max_open`
        默认 24，100 个账号逐个 `get()` 会触发反复驱逐——每次驱逐要
        checkpoint 再关库，每次打开要读 WAL、跑迁移检查。一页概览把
        整个 LRU 搅一遍，既慢又会把正在被注入路径使用的连接挤掉。

        能拿到什么、拿不到什么，要分清楚：
        - **拿得到**（纯文件系统）：体积、最后写入时间、是否已打开。
          WAL 模式下最近的写还在 `-wal` 里，所以 mtime 取主库与 wal 的较大值。
        - **拿不到**：记忆条数、未总结条数。这些必须查库，只能走
          `probe()`（有并发上限）或单账号详情页。

        两类异常都要显式标出来，否则运维时会误判成「这个账号没有记忆」：
        - `orphan`：磁盘上有库文件，索引里没有对应 platform_id（手工拷贝、
          索引损坏、或从别的机器搬过来）。
        - `missing`：索引里有账号，但库文件不见了（误删、盘挂了）。
        """
        with self._lock:
            index = dict(self._index)
            opened = set(self._open)
            notes = dict(self._notes)

        out: list[dict] = []
        seen: set[str] = set()
        dirs = self._bot_dir_names()
        for stem in dirs:
            meta = self._read_meta(stem) or {}
            pid = str(meta.get("platform_id") or index.get(stem, "") or "")
            rel = os.path.join(BOTS_DIR, stem, BOT_DB_FILE)
            seen.add(stem)
            row = self._row(pid, stem, rel, opened)
            self._fill_meta(row, meta, notes, pid)
            out.append(row)

        for name in self._legacy_db_names():
            stem = name[:-3]
            if stem in seen:
                # 同名的 v2 目录优先；旧文件是迁移中断留下的副本，
                # 已由 `migrate_v2()` 在下次启动时校验后清理。
                continue
            seen.add(stem)
            pid = index.get(stem, "")
            row = self._row(pid, stem, name, opened)
            self._fill_meta(row, None, notes, pid)
            out.append(row)

        for stem, pid in index.items():
            if stem in seen:
                continue
            out.append({
                "platform_id": pid, "stem": stem, "file": "",
                "size_bytes": 0, "wal_bytes": 0, "last_write_at": 0.0,
                "opened": pid in opened, "orphan": False, "missing": True,
                "note": notes.get(pid, ""),
                "description": "", "overrides": {}, "created_at": 0.0,
            })

        # 按最后写入倒序：运维最关心「哪个账号最近不动了」，排前面才有用。
        out.sort(key=lambda r: -(r["last_write_at"] or 0))
        return out

    @staticmethod
    def _fill_meta(row: dict, meta: dict | None, notes: dict[str, str],
                   pid: str) -> None:
        row["note"] = notes.get(pid, "") if pid else ""
        if meta:
            row["note"] = row["note"] or str(meta.get("note") or "")
            row["description"] = str(meta.get("description") or "")
            row["overrides"] = dict(meta.get("overrides") or {})
            try:
                row["created_at"] = float(meta.get("created_at") or 0.0)
            except (TypeError, ValueError):
                row["created_at"] = 0.0
        else:
            row["description"] = ""
            row["overrides"] = {}
            row["created_at"] = 0.0

    def _row(self, pid: str | None, stem: str, name: str,
             opened: set[str]) -> dict:
        path = os.path.join(self.root, name)
        size = 0
        wal = 0
        last = 0.0
        for suffix, key in (("", "size"), ("-wal", "wal"), ("-shm", None)):
            try:
                st = os.stat(path + suffix)
            except OSError:
                continue
            if key == "size":
                size = st.st_size
            elif key == "wal":
                wal = st.st_size
            # shm 只影响「最近活动」判断，不计入体积：它是共享内存索引，
            # 正常 checkpoint 后会消失，算进占用会让人误以为库在膨胀。
            last = max(last, st.st_mtime)
        return {
            "platform_id": pid or "",
            "stem": stem,
            "file": name,
            "size_bytes": size,
            "wal_bytes": wal,
            "last_write_at": last,
            "opened": bool(pid) and pid in opened,
            # 备注/描述/开关由 `_fill_meta` 补齐（这里拿不到 json 快照）。
            "note": "",
            # 索引里查不到 platform_id：文件名是 slug+hash，不可逆，
            # 所以只能如实告诉界面「这个库不知道属于谁」。
            "orphan": not pid,
            "missing": not os.path.exists(path),
        }

    def index_of(self, stem: str) -> str:
        """文件名主干 → platform_id，查不到返回空串。"""
        with self._lock:
            return self._index.get(str(stem), "")

    def has_account(self, platform_id: str) -> bool:
        """这个 platform_id 是否登记在册。

        详情页与删除入口必须先用它校验：`platform_id` 来自前端，若不校验
        就直接 `get()`，一个没登记过的 id 会**新建**一个空库——页面上看起来
        「打开了一个空账号」，实际已经污染了磁盘。
        """
        with self._lock:
            return bot_file_stem(platform_id) in self._index

    def is_open(self, platform_id: str) -> bool:
        with self._lock:
            return str(platform_id or UNKNOWN_BOT) in self._open

    # ------------------------------------------------------------ 取用

    def get(self, platform_id: str) -> SqliteBotDb:
        bot = str(platform_id or UNKNOWN_BOT)
        with self._lock:
            db = self._open.get(bot)
            if db is not None:
                self._last_used[bot] = time.time()
                return db
            path = self.db_path(bot)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._ensure_meta(bot)
            db = SqliteBotDb(path, clock=self._clock)
            self._open[bot] = db
            self._last_used[bot] = time.time()
            if self._index.get(bot_file_stem(bot)) != bot:
                self._index[bot_file_stem(bot)] = bot
                self._save_index()
            self._evict_locked()
            return db

    def _evict_locked(self) -> None:
        while len(self._open) > self._max_open:
            victim = min(self._last_used, key=self._last_used.get)
            db = self._open.pop(victim, None)
            self._last_used.pop(victim, None)
            if db is not None:
                try:
                    db.close()
                except Exception:
                    # 关闭失败不能影响别的账号继续工作：这个库下次会被
                    # 重新打开，重复打开的代价远小于整个插件挂掉。
                    pass

    # ------------------------------------------------------------ 管理

    def close_bot(self, platform_id: str) -> bool:
        bot = str(platform_id or UNKNOWN_BOT)
        with self._lock:
            db = self._open.pop(bot, None)
            self._last_used.pop(bot, None)
        if db is None:
            return False
        db.close()
        return True

    def purge(self, platform_id: str) -> bool:
        """彻底删除一个账号的记忆：先关连接，再删库文件和它的 wal/shm。

        顺序不能反。开着连接删文件在 Windows 上直接被拒绝，在 POSIX 上
        会「看起来成功了」但磁盘空间要等进程退出才释放——100 个账号的
        运维里，这种「删了但没删干净」是最难排查的一类。
        """
        bot = str(platform_id or UNKNOWN_BOT)
        self.close_bot(bot)
        with self._lock:
            if self._index.pop(bot_file_stem(bot), None) is not None:
                self._save_index()
            # 备注跟着账号一起清：留着会让下次同 id 重建的账号继承一个
            # 早已不属于它的名字，比没有备注更容易误导人。
            dirty = self._notes.pop(bot, None) is not None
            if self._pending.pop(bot, None) is not None:
                dirty = True
            if dirty:
                self._save_notes()

        removed = False
        stem = bot_file_stem(bot)
        with self._lock:
            self._meta_cache.pop(stem, None)
        targets = [self._v2_db_path(stem), os.path.join(self.root, stem + ".db")]
        for base in targets:
            for suffix in ("", "-wal", "-shm"):
                target = base + suffix
                if not os.path.exists(target):
                    continue
                try:
                    os.remove(target)
                    removed = True
                except PermissionError:
                    # 别的进程（或本进程某个漏关的连接）还占着文件。
                    # 重试一次，仍失败就如实抛给调用方，不吞。
                    time.sleep(0.05)
                    os.remove(target)
                    removed = True
        # 最后删账号目录（bot.json 等）。删不掉也不影响库已经清空。
        try:
            shutil.rmtree(self._bot_dir(stem), ignore_errors=True)
        except Exception:
            pass
        return removed

    def _bot_dir(self, stem: str) -> str:
        return os.path.join(self._bots_root(), stem)

    def close_all(self) -> None:
        with self._lock:
            dbs = list(self._open.values())
            self._open.clear()
            self._last_used.clear()
        for db in dbs:
            try:
                db.close()
            except Exception:
                pass

    def reload_from_disk(self) -> None:
        """关掉全部连接，按磁盘重新读 bot.json / 索引 / 备注。

        恢复流程会整目录替换 `bots/<stem>`；旧连接仍指着被换掉的 inode，
        在 Windows 上还会让后续 `os.replace` 失败。恢复成功后必须走这里，
        否则界面看到的还是恢复前的缓存。
        """
        with self._lock:
            self.close_all()
            self._meta_cache.clear()
            self._pending, self._notes = self._load_notes()
            self._index = self._load_index()
            self._sync_index()

    # ------------------------------------------------------------ 布局

    def layout_state(self) -> str:
        """磁盘现状：`v1` / `v2` / `mixed`。

        以文件为准而不是以 layout.json 为准：手拷进来的老库不会因为标记
        文件写着 2 就变成 v2，界面和迁移都要看见真实状态。
        """
        legacy = bool(self._legacy_db_names())
        v2 = any(
            os.path.isfile(self._v2_db_path(stem))
            for stem in self._bot_dir_names()
        )
        if legacy and v2:
            return "mixed"
        if legacy:
            return "v1"
        return "v2"

    def _write_layout(self, at: float) -> None:
        path = os.path.join(self.root, LAYOUT_FILE)
        created = at
        try:
            with open(path, encoding="utf-8") as fp:
                old = json.load(fp)
            if isinstance(old, dict) and old.get("created_at"):
                created = float(old["created_at"])
        except Exception:
            pass
        _atomic_json(path, {
            "schema": LAYOUT_SCHEMA,
            "created_at": created,
            "updated_at": at,
        })

    def _rebuild_index(self) -> dict[str, str]:
        """从磁盘重建索引：bot.json 是真相，缓存只是加速。

        保留缓存里「文件已丢失」的条目：它们是 `missing` 告警的唯一线索，
        重建时清掉等于把事故现场也一起删了。
        """
        with self._lock:
            fresh: dict[str, str] = {}
            for stem in self._bot_dir_names():
                meta = self._read_meta(stem)
                pid = str((meta or {}).get("platform_id") or "").strip()
                if pid:
                    fresh[stem] = pid
            for name in self._legacy_db_names():
                stem = name[:-3]
                pid = self._index.get(stem, "")
                if pid:
                    fresh.setdefault(stem, pid)
            for stem, pid in self._index.items():
                fresh.setdefault(stem, pid)
            if fresh != self._index:
                self._index = fresh
                self._save_index()
            return dict(fresh)

    def migrate_v2(self, *, backup_dir: str | None = None,
                   version: str = "", now: float | None = None) -> dict:
        """把 v1 的平铺库搬进 v2 的账号目录，**幂等且可回滚**。

        顺序刻意定成这样：
        1. 先关连接（WAL 里的页必须由本进程自己合并）；
        2. 能备份就先备份，备份失败直接放弃迁移——宁可不升级，也不能
           在没有退路的情况下动唯一的数据库；
        3. 每个账号：checkpoint → 记行数 → 移动 → 再校验行数 → 写 bot.json；
           任一步失败就把文件挪回去，旧路径保持可用；
        4. 最后才写 layout.json 和重建索引。

        孤立库（索引里没有 platform_id）一律不搬：文件名的 hash 不可逆，
        猜一个 id 等于把别人的记忆挂到某个账号名下。
        """
        at = time.time() if now is None else float(now)
        with self._lock:
            self.close_all()
            legacy = self._legacy_db_names()
            result: dict = {
                "ok": True,
                "layout": LAYOUT_SCHEMA,
                "migrated": [],
                "skipped": [],
                "failed": [],
                "backup": "",
                "created_at": at,
            }
            if not legacy:
                self._write_layout(at)
                self._rebuild_index()
                result["layout_state"] = self.layout_state()
                return result

            if backup_dir:
                try:
                    from hermes_fleet.backup import backup_bots

                    result["backup"] = backup_bots(
                        root=self.root, out_dir=backup_dir,
                        version=version, now=at,
                    )["path"]
                except Exception as exc:
                    result["ok"] = False
                    result["error"] = f"迁移前备份失败，未改动任何文件：{exc}"
                    return result

            for name in legacy:
                self._migrate_one(name, at, result)
            self._save_notes()
            self._write_layout(at)
            self._rebuild_index()
            result["layout_state"] = self.layout_state()
            if result["failed"]:
                result["ok"] = False
                result["error"] = "部分账号迁移失败，原库已保留，可修复后重跑"
            return result

    def _migrate_one(self, name: str, at: float, result: dict) -> None:
        stem = name[:-3]
        pid = str(self._index.get(stem, "") or "")
        if not pid:
            result["skipped"].append({
                "stem": stem, "reason": "索引里没有 platform_id，孤立库不迁移",
            })
            return
        src = os.path.join(self.root, name)
        dst = self._v2_db_path(stem)
        if os.path.exists(dst):
            try:
                snap = _db_snapshot(dst)
            except Exception as exc:
                result["failed"].append({
                    "stem": stem, "error": f"目标库校验失败：{exc}",
                })
                return
            if snap["integrity"] != "ok":
                result["failed"].append({
                    "stem": stem, "error": "目标库存在但完整性检查未通过",
                })
                return
            try:
                before = _db_snapshot(src)
            except Exception as exc:
                result["failed"].append({
                    "stem": stem,
                    "error": f"旧库无法读取，两份都保留：{exc}",
                })
                return
            if before["counts"] == snap["counts"]:
                # 上次迁移已成功、只是没来得及删旧文件：校验一致后清掉残留。
                _remove_quiet(src + "-wal")
                _remove_quiet(src + "-shm")
                _remove_quiet(src)
                result["skipped"].append({
                    "stem": stem, "reason": "目标库已存在且一致，已清理残留旧文件",
                })
            else:
                result["failed"].append({
                    "stem": stem,
                    "error": "目标库与旧库行数不一致，两份都保留，请人工确认",
                })
            return
        try:
            _checkpoint(src)
        except Exception as exc:
            result["failed"].append({
                "stem": stem, "error": f"checkpoint 失败，未移动文件：{exc}",
            })
            return
        try:
            _remove_quiet(src + "-shm")
            before = _db_snapshot(src)
            if before["integrity"] != "ok":
                raise RuntimeError(f"源库完整性检查未通过：{before['integrity']}")
        except Exception as exc:
            result["failed"].append({"stem": stem, "error": str(exc)})
            return

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.replace(src, dst)
        except OSError as exc:
            result["failed"].append({
                "stem": stem, "error": f"移动文件失败，原库未动：{exc}",
            })
            return
        try:
            after = _db_snapshot(dst)
            if after["integrity"] != "ok" or after["counts"] != before["counts"]:
                raise RuntimeError("迁移后行数/完整性与源库不一致")
        except Exception as exc:
            try:
                os.replace(dst, src)
            except OSError:
                pass
            result["failed"].append({
                "stem": stem, "error": f"{exc}（已回滚为原文件）",
            })
            return

        meta = self._read_meta(stem) or _default_meta(pid)
        meta["platform_id"] = pid
        # 备注可能还在 `_notes.json`（v1 的存法），也可能在内存里刚改过。
        note = self._pending.pop(pid, "") or self._notes.get(pid, "")
        if note:
            meta["note"] = note
        if not meta.get("created_at"):
            try:
                meta["created_at"] = os.path.getmtime(dst)
            except OSError:
                meta["created_at"] = at
        self._write_meta(stem, meta)
        _remove_quiet(src + "-wal")
        _remove_quiet(src + "-shm")
        result["migrated"].append(stem)

    def stats(self) -> dict[str, object]:
        with self._lock:
            opened = sorted(self._open)
        files: list[str] = []
        for name in self._legacy_db_names():
            files.append(name)
        for stem in self._bot_dir_names():
            rel = os.path.join(BOTS_DIR, stem, BOT_DB_FILE)
            if os.path.isfile(os.path.join(self.root, rel)):
                files.append(rel)
        return {
            "root": self.root,
            "max_open": self._max_open,
            "opened": len(opened),
            "bots_opened": opened,
            "files_on_disk": sorted(files),
            "layout": self.layout_state(),
            "bots_dir": self._bots_root(),
        }


def _remove_quiet(path: str) -> None:
    """删文件，删不掉就忽略（迁移收尾清理用，失败不影响数据）。"""
    try:
        os.remove(path)
    except OSError:
        pass


def bot_id_of(platform_id: str) -> BotId:
    """把适配器 id 转成契约层的 `BotId`。

    用适配器 id 而不是 appid：适配器 id 是运营在面板里起的名字
    （`qq-01`、`bot-alice`），可读、可改；appid 换绑账号后会变，
    以它为主键会让「换了 appid 但其实是同一个号」变成记忆丢失。
    """
    return BotId(str(platform_id or UNKNOWN_BOT))
