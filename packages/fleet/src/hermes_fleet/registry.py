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
