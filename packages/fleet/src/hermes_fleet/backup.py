"""把全部账号库打成**一致性**备份 zip。

为什么不能直接 `shutil.copy` 那些 `.db`：库跑在 WAL 模式下，最近一批写入
还在 `-wal` 里没合并进主库。直接拷 `.db` 会拿到一个「看起来完好、其实少了
最近若干条记忆」的快照——正好在最需要备份的时刻（还在活跃写）丢数据。
所以每个库都用 SQLite 自己的在线备份 API 打开一致性快照，再写进 zip。

`_index.json` / `_notes.json` 也一起打包：没有索引，恢复后文件名（slug+
hash）反推不出 platform_id，所有账号都变成「孤立库」；备注也会一起丢。

纯逻辑模块：只吃 root/out_dir 路径，不认识 AstrBot、不读配置。备份策略
（放哪、何时触发）由调用方决定，这里只负责「怎么打得又全又一致」。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import zipfile
from typing import Any

BACKUP_PREFIX = "hermes-backup-"
INDEX_FILE = "_index.json"
NOTES_FILE = "_notes.json"
MANIFEST = "manifest.json"


def backup_filename(ts: float | None = None) -> str:
    """备份包名，带时间戳（同秒重复由调用方处理）。"""
    stamp = time.strftime(
        "%Y%m%d-%H%M%S", time.localtime(ts if ts is not None else time.time())
    )
    return f"{BACKUP_PREFIX}{stamp}.zip"


def _db_files(root: str) -> list[str]:
    """root 下的库文件（只读文件名，不打开连接）。"""
    try:
        return sorted(
            f for f in os.listdir(root)
            if f.endswith(".db") and os.path.isfile(os.path.join(root, f))
        )
    except OSError:
        return []


def _snapshot_db(src_path: str, dest_conn: sqlite3.Connection) -> int:
    """用 SQLite 在线备份把 src 拷进 dest_conn（已打开的目标库）。

    返回源库的行数放不进快照里没意义，调用方只需知道成功与否；这里返回
    目标文件大小由调用方 stat。以只读 URI 打开源，避免备份动作本身触发
    迁移/写入；`backup()` 会处理 WAL 一致性。
    """
    uri = "file:" + src_path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    src = sqlite3.connect(uri, uri=True, timeout=30.0)
    try:
        src.backup(dest_conn)
    finally:
        src.close()
    return 0


def backup_bots(*, root: str, out_dir: str, version: str = "",
                now: float | None = None) -> dict[str, Any]:
    """把所有库 + 索引 + 备注打成 zip，返回可供展示的结果字典。

    流程：
    1. 建 out_dir；
    2. 逐个 `.db` 做一致性快照到临时文件；
    3. 与 `_index.json` / `_notes.json` + manifest 一起写进 zip；
    4. 删临时快照；
    5. 返回 {path, filename, size_bytes, bots, ...}。

    临时快照与最终 zip 都在 out_dir，用 `.tmp-` 前缀；中途失败时清理，
    避免半个快照被当成有效备份。
    """
    import tempfile

    root = str(root or "")
    if not root or not os.path.isdir(root):
        raise ValueError("记忆库目录不存在")
    os.makedirs(out_dir, exist_ok=True)

    at = time.time() if now is None else float(now)
    name = backup_filename(at)
    zip_path = os.path.join(out_dir, name)
    # 同秒重复导出：加序号，不覆盖已有备份（备份被覆盖就等于没备份）。
    seq = 1
    while os.path.exists(zip_path):
        stem, ext = os.path.splitext(name)
        zip_path = os.path.join(out_dir, f"{stem}-{seq}{ext}")
        seq += 1

    dbs = _db_files(root)
    temp_files: list[str] = []
    try:
        fd, manifest_tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp-json")
        os.close(fd)
        temp_files.append(manifest_tmp)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for db_name in dbs:
                src_path = os.path.join(root, db_name)
                fd, snap = tempfile.mkstemp(dir=out_dir, suffix=".tmp-db")
                os.close(fd)
                temp_files.append(snap)
                # 目标库必须先删空：mkstemp 建的是空文件，SQLite 会当作
                # 已存在的空库，backup() 仍能覆盖，但显式删除更干净。
                os.remove(snap)
                dest = sqlite3.connect(snap)
                try:
                    _snapshot_db(src_path, dest)
                finally:
                    dest.close()
                zf.write(snap, arcname=db_name)

            for meta in (INDEX_FILE, NOTES_FILE):
                meta_path = os.path.join(root, meta)
                if os.path.isfile(meta_path):
                    zf.write(meta_path, arcname=meta)

            manifest = {
                "format": "hermes-fleet-backup",
                "version": 1,
                "generator": f"hermes_fleet {version}" if version else "hermes_fleet",
                "created_at": at,
                "bots": len(dbs),
                "files": [INDEX_FILE, NOTES_FILE] if dbs else [],
                "note": "解压后把全部文件覆盖回记忆库根目录即可恢复。",
            }
            with open(manifest_tmp, "w", encoding="utf-8") as fp:
                json.dump(manifest, fp, ensure_ascii=False, indent=2)
            zf.write(manifest_tmp, arcname=MANIFEST)

        size = os.path.getsize(zip_path)
        return {
            "ok": True,
            "path": zip_path,
            "filename": os.path.basename(zip_path),
            "size_bytes": size,
            "bots": len(dbs),
            "created_at": at,
            "message": (
                f"备份完成：{len(dbs)} 个账号库 + 索引/备注，"
                f"共 {size / 1024:.1f} KB。请把 zip 另存到插件目录之外。"
            ),
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
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass


__all__ = ["BACKUP_PREFIX", "backup_bots", "backup_filename"]
