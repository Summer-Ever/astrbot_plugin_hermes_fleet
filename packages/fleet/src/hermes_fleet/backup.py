"""把全部账号库打成**一致性**备份 zip。

为什么不能直接 `shutil.copy` 那些 `.db`：库跑在 WAL 模式下，最近一批写入
还在 `-wal` 里没合并进主库。直接拷 `.db` 会拿到一个「看起来完好、其实少了
最近若干条记忆」的快照——正好在最需要备份的时刻（还在活跃写）丢数据。
所以每个库都用 SQLite 自己的在线备份 API 打开一致性快照，再写进 zip。

v2 之后账号是目录（`bots/<stem>/memory.db` + `bot.json`），备份按目录结构
原样打包；v1 遗留的根目录平铺 `.db` 也照样收进来（双读），所以「升级到
一半」的现场备份出来仍是完整的。

`_index.json` / `_notes.json` 也一起打包：v1 包里没有索引，恢复后文件名
（slug+hash）反推不出 platform_id，所有账号都变成「孤立库」。

每个文件在 manifest 里记 size/sha256；数据库额外记 `sqlite_user_version`
与 `integrity`。恢复端（restore.py）靠这些字段在动手之前就判断「这个包
能不能用」，而不是先覆盖再发现坏了。

纯逻辑模块：只吃 root/out_dir 路径，不认识 AstrBot、不读配置。备份策略
（放哪、何时触发）由调用方决定，这里只负责「怎么打得又全又一致」。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
import zipfile
from typing import Any

BACKUP_PREFIX = "hermes-backup-"
# 恢复前的自动快照用另一个前缀：它和普通备份混在同一个目录里，但清理
# 策略不同——普通备份留 10 份，恢复快照留 5 份且永远保留最新一份。
PRERESTORE_PREFIX = "hermes-pre-restore-"
INDEX_FILE = "_index.json"
NOTES_FILE = "_notes.json"
MANIFEST = "manifest.json"

LAYOUT_FILE = "layout.json"
BOTS_DIR = "bots"
BOT_DB_FILE = "memory.db"
BOT_META_FILE = "bot.json"
CONFIG_ENTRY = "config/plugin_config.json"

FORMAT = "hermes-fleet-backup"
FORMAT_VERSION = 2
MAX_KEEP = 10
PRERESTORE_KEEP = 5


def backup_filename(ts: float | None = None,
                    prefix: str = BACKUP_PREFIX) -> str:
    """备份包名，带时间戳（同秒重复由调用方处理）。"""
    stamp = time.strftime(
        "%Y%m%d-%H%M%S", time.localtime(ts if ts is not None else time.time())
    )
    return f"{prefix}{stamp}.zip"


def _file_prefix(root: str) -> list[str]:
    return sorted(f for f in os.listdir(root) if f.endswith(".db"))


def _db_files(root: str) -> list[str]:
    """v1 布局：root 下平铺的库文件名（只读文件名，不打开连接）。"""
    try:
        return [f for f in _file_prefix(root)
                if os.path.isfile(os.path.join(root, f))]
    except OSError:
        return []


def _bot_stems(root: str) -> list[str]:
    """v2 布局：`bots/` 下的账号目录名。"""
    try:
        base = os.path.join(root, BOTS_DIR)
        return sorted(
            n for n in os.listdir(base)
            if os.path.isdir(os.path.join(base, n))
        )
    except OSError:
        return []


def _db_targets(root: str) -> list[dict[str, str]]:
    """需要打包的库：`(arcname, 磁盘路径, 布局)`。

    v2 优先；同名 v1 文件是迁移中断留下的副本，已由 registry 的迁移流程
    校验清理，这里不再重复打包（否则恢复时会造出两份）。
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for stem in _bot_stems(root):
        path = os.path.join(root, BOTS_DIR, stem, BOT_DB_FILE)
        if not os.path.isfile(path):
            continue
        seen.add(stem)
        out.append({
            "arcname": f"{BOTS_DIR}/{stem}/{BOT_DB_FILE}",
            "path": path,
            "layout": "v2",
            "stem": stem,
        })
    for name in _db_files(root):
        stem = name[:-3]
        if stem in seen:
            continue
        out.append({
            "arcname": name,
            "path": os.path.join(root, name),
            "layout": "v1",
            "stem": stem,
        })
    return out


def _meta_targets(root: str) -> list[tuple[str, str]]:
    """要随包带走的配置文件：`(arcname, 磁盘路径)`。"""
    out: list[tuple[str, str]] = []
    for stem in _bot_stems(root):
        path = os.path.join(root, BOTS_DIR, stem, BOT_META_FILE)
        if os.path.isfile(path):
            out.append((f"{BOTS_DIR}/{stem}/{BOT_META_FILE}", path))
    for name in (LAYOUT_FILE, INDEX_FILE, NOTES_FILE):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            out.append((name, path))
    return out


def layout_of(root: str) -> str:
    """磁盘现状：`v1` / `v2` / `mixed`（与 registry.layout_state 同义）。

    这里单独实现而不是 import registry：backup 需要能被恢复流程和工具脚本
    单独调用，不能因为 registry 变了就跟着不可用。判据只有文件系统。
    """
    legacy = bool(_db_files(root))
    v2 = any(
        os.path.isfile(os.path.join(root, BOTS_DIR, stem, BOT_DB_FILE))
        for stem in _bot_stems(root)
    )
    if legacy and v2:
        return "mixed"
    return "v1" if legacy else "v2"


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _db_facts(path: str) -> tuple[str, int]:
    """只读打开库，返回 `(integrity_check, user_version)`。"""
    uri = "file:" + path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=15.0)
    try:
        verdict = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        except (TypeError, ValueError, sqlite3.DatabaseError):
            version = 0
        return verdict, version
    finally:
        conn.close()


def _snapshot_db(src_path: str, dest_path: str) -> None:
    """用 SQLite 在线备份把 src 拷到 dest_path。

    以只读 URI 打开源，避免备份动作本身触发迁移/写入；`backup()` 会处理
    WAL 一致性。
    """
    uri = "file:" + src_path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    src = sqlite3.connect(uri, uri=True, timeout=30.0)
    try:
        dest_conn = sqlite3.connect(dest_path)
        try:
            src.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        src.close()


def prune_backups(out_dir: str, *, keep: int = MAX_KEEP,
                  prefix: str = BACKUP_PREFIX) -> list[str]:
    """只保留最新的 `keep` 份同前缀备份，返回被删除的文件名。

    按文件名排序而不是 mtime：文件名前缀就是本地时间戳，稳定可预期；
    mtime 会被复制/解压动作改掉，反而把旧包当成新包留下。
    """
    try:
        names = sorted(
            n for n in os.listdir(out_dir)
            if n.startswith(prefix) and n.endswith(".zip")
        )
    except OSError:
        return []
    removed: list[str] = []
    for name in names[:max(0, len(names) - max(1, int(keep)))]:
        try:
            os.remove(os.path.join(out_dir, name))
            removed.append(name)
        except OSError:
            continue
    return removed


def backup_bots(*, root: str, out_dir: str, version: str = "",
                now: float | None = None,
                config: dict[str, Any] | None = None,
                prefix: str = BACKUP_PREFIX,
                keep: int = MAX_KEEP) -> dict[str, Any]:
    """把所有库 + 账号配置 + 索引/备注打成 zip，返回展示用结果字典。

    流程：
    1. 建 out_dir；
    2. 逐个库做一致性快照到临时文件（并校验完整性、记录行数版本）；
    3. 与 bot.json / layout.json / 索引备注 / manifest 一起写进 zip；
    4. 删临时快照；
    5. 按前缀清理旧包；
    6. 返回 {path, filename, size_bytes, bots, files, skipped, ...}。

    临时快照与最终 zip 都在 out_dir，用 `.tmp-` 前缀；中途失败时清理，
    避免半个快照被当成有效备份。
    """
    root = str(root or "")
    if not root or not os.path.isdir(root):
        raise ValueError("记忆库目录不存在")
    os.makedirs(out_dir, exist_ok=True)

    at = time.time() if now is None else float(now)
    name = backup_filename(at, prefix)
    zip_path = os.path.join(out_dir, name)
    # 同秒重复导出：加序号，不覆盖已有备份（备份被覆盖就等于没备份）。
    seq = 1
    while os.path.exists(zip_path):
        stem, ext = os.path.splitext(name)
        zip_path = os.path.join(out_dir, f"{stem}-{seq}{ext}")
        seq += 1

    targets = _db_targets(root)
    temp_files: list[str] = []
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    bots = 0
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in targets:
                fd, snap = tempfile.mkstemp(dir=out_dir, suffix=".tmp-db")
                os.close(fd)
                temp_files.append(snap)
                # 目标库必须先删空：mkstemp 建的是空文件，SQLite 会当作
                # 已存在的空库，backup() 仍能覆盖，但显式删除更干净。
                os.remove(snap)
                try:
                    _snapshot_db(item["path"], snap)
                    verdict, user_version = _db_facts(snap)
                except Exception as exc:  # 单个库坏了不拖垮整包，但要留痕
                    skipped.append({
                        "name": item["arcname"],
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                if verdict != "ok":
                    skipped.append({
                        "name": item["arcname"],
                        "reason": f"integrity_check={verdict}",
                    })
                    continue
                zf.write(snap, arcname=item["arcname"])
                records.append({
                    "name": item["arcname"],
                    "size": os.path.getsize(snap),
                    "sha256": sha256_of(snap),
                    "sqlite_user_version": user_version,
                    "integrity": verdict,
                    "layout": item["layout"],
                })
                bots += 1

            for arcname, path in _meta_targets(root):
                zf.write(path, arcname=arcname)
                records.append({
                    "name": arcname,
                    "size": os.path.getsize(path),
                    "sha256": sha256_of(path),
                })

            if config is not None:
                fd, cfg_tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp-json")
                os.close(fd)
                temp_files.append(cfg_tmp)
                with open(cfg_tmp, "w", encoding="utf-8") as fp:
                    json.dump(
                        {"saved_at": at, "values": dict(config)},
                        fp, ensure_ascii=False, indent=2,
                    )
                zf.write(cfg_tmp, arcname=CONFIG_ENTRY)
                records.append({
                    "name": CONFIG_ENTRY,
                    "size": os.path.getsize(cfg_tmp),
                    "sha256": sha256_of(cfg_tmp),
                })

            manifest = {
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "version": FORMAT_VERSION,  # 兼容只认 version 的旧读取方
                "generator": f"hermes_fleet {version}" if version else "hermes_fleet",
                "plugin_version": version,
                "created_at": at,
                "layout": layout_of(root),
                "bots": bots,
                "files": records,
                "config_keys": sorted(config) if config else [],
                "config": CONFIG_ENTRY if config is not None else "",
                "skipped": skipped,
                "note": "用管理面板「恢复」或 hermes_fleet.restore 恢复；不要手工解压覆盖。",
            }
            fd, manifest_tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp-json")
            os.close(fd)
            temp_files.append(manifest_tmp)
            with open(manifest_tmp, "w", encoding="utf-8") as fp:
                json.dump(manifest, fp, ensure_ascii=False, indent=2)
            zf.write(manifest_tmp, arcname=MANIFEST)

        size = os.path.getsize(zip_path)
        pruned = prune_backups(out_dir, keep=keep, prefix=prefix)
        message = (
            f"备份完成：{bots} 个账号库 + 账号配置，共 {size / 1024:.1f} KB。"
        )
        if config is not None:
            message += " 已含插件设置。"
        if skipped:
            message += f" 有 {len(skipped)} 个文件跳过，详见 manifest。"
        return {
            "ok": True,
            "path": zip_path,
            "filename": os.path.basename(zip_path),
            "size_bytes": size,
            "bots": bots,
            "created_at": at,
            "layout": manifest["layout"],
            "files": len(records),
            "skipped": skipped,
            "config_included": config is not None,
            "pruned": pruned,
            "message": message,
        }
    except Exception:
        # 失败的 zip 不留着冒充成功备份。
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except OSError:
            pass
        raise
    finally:
        for tmp in temp_files:
            # 快照库是从 WAL 模式的源库逐页复制的，文件头也带着 WAL 标志，
            # 一打开就会生成 `-wal`/`-shm` 旁路文件。只删主文件会让它们
            # 永远留在备份目录里（每份 32 KB），跑一次备份/恢复攒一份。
            for path in (tmp, tmp + "-wal", tmp + "-shm", tmp + "-journal"):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass


__all__ = [
    "BACKUP_PREFIX",
    "CONFIG_ENTRY",
    "FORMAT",
    "FORMAT_VERSION",
    "INDEX_FILE",
    "MANIFEST",
    "NOTES_FILE",
    "PRERESTORE_PREFIX",
    "backup_bots",
    "backup_filename",
    "layout_of",
    "prune_backups",
    "sha256_of",
]
