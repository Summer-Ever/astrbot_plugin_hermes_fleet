"""账号 → 记忆库的路由层。

这一层只回答一个问题：**这条消息属于哪个库文件。**

为什么单独成层，而不是在注入函数里现算路径：
路径推算一旦散落在多个调用点，就会出现「注入读 A 库、落库写 B 库」这种
静默错位——两边都不报错，只表现为「记忆时好时坏」。所有路径计算必须
只有一处实现。
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time

from hermes_contracts import BotId, Clock
from hermes_storage_sqlite import SqliteBotDb

UNKNOWN_BOT = "default"

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
        self._index_path = os.path.join(self.root, "_index.json")
        self._index: dict[str, str] = self._load_index()
        self._notes_path = os.path.join(self.root, NOTES_FILE)
        self._notes: dict[str, str] = self._load_notes()

    # ------------------------------------------------------------ 账号索引

    def _load_index(self) -> dict[str, str]:
        """`文件名主干 -> 适配器 id`。

        为什么需要这张表：后台总结要回答「现在有哪些账号」，而文件名是
        `qq-01-a1b2c3d4.db` 这种不可逆的编码（slug + hash）。没有索引，
        重启后就只认识「这次启动之后说过话的账号」——一个昨天攒了 200 条
        待总结消息、今天再没人说话的账号，会被永久跳过。

        索引坏了不要紧：能从文件名反推的只有 slug 前缀，所以这里失败时
        只告警并当作空索引，不影响任何读写路径。
        """
        try:
            import json

            with open(self._index_path, encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return {}

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

    def _load_notes(self) -> dict[str, str]:
        """`platform_id -> 备注`。读失败当作没有备注，不影响任何读写路径。"""
        try:
            import json

            with open(self._notes_path, encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                clean = {}
                for k, v in data.items():
                    note = normalize_note(v)
                    if note:
                        clean[str(k)] = note
                return clean
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return {}

    def _save_notes(self) -> None:
        try:
            _atomic_json(self._notes_path, self._notes)
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
        是正常操作顺序；但 `accounts()` 只按库文件列行，所以这类备注在库
        文件出现之前不会显示，不会凭空多出一个假账号。
        """
        pid = str(platform_id or UNKNOWN_BOT).strip() or UNKNOWN_BOT
        note = normalize_note(text)
        with self._lock:
            if note:
                if self._notes.get(pid) == note:
                    return note
                self._notes[pid] = note
            else:
                if pid not in self._notes:
                    return ""
                self._notes.pop(pid, None)
            self._save_notes()
        return note

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
        by_stem = {stem: pid for stem, pid in index.items()}

        try:
            files = sorted(f for f in os.listdir(self.root) if f.endswith(".db"))
        except OSError:
            files = []

        out: list[dict] = []
        seen: set[str] = set()
        for name in files:
            stem = name[:-3]
            seen.add(stem)
            pid = by_stem.get(stem)
            row = self._row(pid, stem, name, opened)
            row["note"] = notes.get(row["platform_id"], "") if pid else ""
            out.append(row)
        for stem, pid in by_stem.items():
            if stem in seen:
                continue
            out.append({
                "platform_id": pid, "stem": stem, "file": "",
                "size_bytes": 0, "wal_bytes": 0, "last_write_at": 0.0,
                "opened": pid in opened, "orphan": False, "missing": True,
                "note": notes.get(pid, ""),
            })

        # 按最后写入倒序：运维最关心「哪个账号最近不动了」，排前面才有用。
        out.sort(key=lambda r: -(r["last_write_at"] or 0))
        return out

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
            # 备注由调用方按 platform_id 补齐（这里拿不到 notes 快照）。
            "note": "",
            # 索引里查不到 platform_id：文件名是 slug+hash，不可逆，
            # 所以只能如实告诉界面「这个库不知道属于谁」。
            "orphan": not pid,
            "missing": False,
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

    # ------------------------------------------------------------ 路径

    def db_path(self, platform_id: str) -> str:
        return os.path.join(self.root, bot_file_stem(platform_id) + ".db")

    # ------------------------------------------------------------ 取用

    def get(self, platform_id: str) -> SqliteBotDb:
        bot = str(platform_id or UNKNOWN_BOT)
        with self._lock:
            db = self._open.get(bot)
            if db is not None:
                self._last_used[bot] = time.time()
                return db
            db = SqliteBotDb(self.db_path(bot), clock=self._clock)
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
            if self._notes.pop(bot, None) is not None:
                self._save_notes()
        path = self.db_path(bot)
        removed = False
        for suffix in ("", "-wal", "-shm"):
            target = path + suffix
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
        return removed

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

    def stats(self) -> dict[str, object]:
        with self._lock:
            opened = sorted(self._open)
        return {
            "root": self.root,
            "max_open": self._max_open,
            "opened": len(opened),
            "bots_opened": opened,
            "files_on_disk": sorted(
                f for f in os.listdir(self.root) if f.endswith(".db")
            ) if os.path.isdir(self.root) else [],
        }


def bot_id_of(platform_id: str) -> BotId:
    """把适配器 id 转成契约层的 `BotId`。

    用适配器 id 而不是 appid：适配器 id 是运营在面板里起的名字
    （`qq-01`、`bot-alice`），可读、可改；appid 换绑账号后会变，
    以它为主键会让「换了 appid 但其实是同一个号」变成记忆丢失。
    """
    return BotId(str(platform_id or UNKNOWN_BOT))
