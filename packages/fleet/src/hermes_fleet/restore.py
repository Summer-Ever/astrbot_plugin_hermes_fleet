"""从备份包恢复全部账号与配置。

恢复是这个插件里**唯一会主动覆盖用户数据**的动作，所以顺序被刻意定成
「先证明能用，再动一个字节」：

1. `inspect_backup()`：只读预检——包结构、路径、sha256、每个库的
   `integrity_check`、库的 `user_version` 是否比本机新（降级拒绝）。
   这一步失败就返回问题清单，绝不落到暂存目录之外。
2. `restore_backup()`：先给**当前**数据打一份快照（前缀
   `hermes-pre-restore-`），再解压到同盘暂存目录并二次校验；
3. 校验通过才逐个 `os.replace` 换入，旧目录挪进暂存区的 `_old/`；
   任何一步异常都按撤销日志原样搬回；
4. 成功后由调用方 `registry.reload_from_disk()` 重建连接与索引。

安全边界（这些是攻击面，不是理论）：
- 绝对路径 / 盘符 / `..` / 反斜杠一律拒绝（zip 里的名字来自外部文件）；
- 符号链接条目拒绝（解压后可能指向包外）；
- 总解压体积、单文件体积、压缩比、条目数都设上限（zip 炸弹）；
- 只接受已知的条目形状，`bots/<stem>/` 的 stem 不允许再带分隔符。

这个模块不 import astrbot，也不 import registry：恢复逻辑可以脱离宿主
用临时目录完整测试。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
import zipfile
from typing import Any, Callable

from .backup import (
    BOTS_DIR,
    BOT_DB_FILE,
    BOT_META_FILE,
    CONFIG_ENTRY,
    FORMAT,
    INDEX_FILE,
    LAYOUT_FILE,
    MANIFEST,
    NOTES_FILE,
    PRERESTORE_PREFIX,
    backup_bots,
    sha256_of,
)

CONFIRM_WORD = "RESTORE"

MAX_TOTAL_BYTES = 2 * 1024 ** 3
MAX_ENTRY_BYTES = 512 * 1024 ** 2
MAX_ENTRIES = 5000
MAX_RATIO = 200
_PRERESTORE_KEEP = 5
_DRIVE = re.compile(r"^[A-Za-z]:")
_LAYOUT_SCHEMA = 2

_BOT_FILES = (BOT_DB_FILE, BOT_META_FILE)
_ROOT_FILES = (LAYOUT_FILE, INDEX_FILE, NOTES_FILE)
_LEGACY_DB = re.compile(r"^[^/]+\.db$")


class RestoreError(ValueError):
    """恢复被拒绝或中途失败（消息里带原因与快照路径）。"""


# ---------------------------------------------------------------- 预检


def _safe_name(name: str) -> str:
    """校验 zip 内的相对路径，返回空串表示合法，否则返回原因。"""
    if not name:
        return "空文件名"
    if name.startswith(("/", "\\")):
        return "绝对路径"
    if _DRIVE.match(name):
        return "盘符路径"
    if "\\" in name:
        return "反斜杠路径"
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return "可疑路径（.. 或空段）"
    return ""


def _member_kind(name: str) -> str:
    """条目形状分类：`db` / `meta` / `config` / `known` / ``（未知）。"""
    if name == MANIFEST:
        return "manifest"
    if name == CONFIG_ENTRY:
        return "config"
    if name == LAYOUT_FILE:
        return "known"
    if name == BOTS_DIR:
        return "known"
    if name in (INDEX_FILE, NOTES_FILE):
        return "known"
    if name.startswith(BOTS_DIR + "/"):
        parts = name.split("/")
        if len(parts) == 3 and parts[2] in _BOT_FILES:
            return "db" if parts[2] == BOT_DB_FILE else "meta"
        return ""
    if _LEGACY_DB.match(name):
        return "db"
    return ""


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def _read_json(zf: zipfile.ZipFile, name: str) -> Any:
    try:
        with zf.open(name) as fp:
            return json.loads(fp.read().decode("utf-8"))
    except KeyError:
        return None
    except Exception:
        return None


def _db_facts(path: str) -> tuple[str, int]:
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


def _current_user_version(root: str, stem: str) -> int | None:
    """本机同账号库的 user_version；没有库或读不到时返回 None。"""
    candidates = (
        os.path.join(root, BOTS_DIR, stem, BOT_DB_FILE),
        os.path.join(root, stem + ".db"),
    )
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            return _db_facts(path)[1]
        except Exception:
            return None
    return None


def _stem_of(name: str) -> str:
    parts = name.split("/")
    if parts[0] == BOTS_DIR:
        return parts[1]
    return name[:-3]


def inspect_backup(path: str, *, root: str = "") -> dict[str, Any]:
    """只读预检，返回问题清单与可展示元数据。不抛异常（除路径不存在外）。"""
    out: dict[str, Any] = {
        "ok": False,
        "filename": os.path.basename(str(path or "")),
        "path": str(path or ""),
        "created_at": 0.0,
        "plugin_version": "",
        "format_version": 0,
        "layout": "",
        "bots": 0,
        "files": 0,
        "skipped": [],
        "config_keys": [],
        "config": None,
        "problems": [],
    }
    problems: list[str] = out["problems"]

    if not path or not os.path.isfile(path):
        problems.append("备份文件不存在")
        return out
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        problems.append(f"不是有效的 zip：{exc}")
        return out

    with zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        out["files"] = len(infos)
        if len(infos) > MAX_ENTRIES:
            problems.append(f"条目过多（{len(infos)} > {MAX_ENTRIES}）")
        total = 0
        for info in infos:
            if info.is_dir():
                continue
            reason = _safe_name(info.filename)
            if reason:
                problems.append(f"{info.filename}：{reason}")
                continue
            if _is_symlink(info):
                problems.append(f"{info.filename}：符号链接不允许")
                continue
            total += info.file_size
            if info.file_size > MAX_ENTRY_BYTES:
                problems.append(f"{info.filename}：单文件过大")
            if (info.compress_size > 0 and info.file_size > 64 * 1024 * 1024
                    and info.file_size / info.compress_size > MAX_RATIO):
                problems.append(f"{info.filename}：压缩比异常")
            if _member_kind(info.filename) == "":
                problems.append(f"{info.filename}：未知条目，拒绝恢复")
        if total > MAX_TOTAL_BYTES:
            problems.append("解压后总体积过大")

        manifest = _read_json(zf, MANIFEST)
        if not isinstance(manifest, dict):
            problems.append("缺少或无法解析 manifest.json")
            manifest = {}
        else:
            out["created_at"] = float(manifest.get("created_at") or 0.0)
            out["plugin_version"] = str(manifest.get("plugin_version") or "")
            out["layout"] = str(manifest.get("layout") or "")
            out["skipped"] = list(manifest.get("skipped") or [])
            try:
                out["format_version"] = int(
                    manifest.get("format_version") or manifest.get("version") or 1
                )
            except (TypeError, ValueError):
                out["format_version"] = 0
            if manifest.get("format") != FORMAT:
                problems.append("不是本插件的备份包")
            if out["format_version"] > 2:
                problems.append(
                    f"备份格式 v{out['format_version']} 比当前插件支持的 v2 新"
                )

        # sha256：manifest 记了什么就核什么，没记的（老包）跳过。
        listed: dict[str, str] = {}
        for row in (manifest.get("files") or []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            digest = str(row.get("sha256") or "")
            if name and digest:
                listed[name] = digest
        available = set(names)
        for name, digest in listed.items():
            if name not in available:
                problems.append(f"{name}：包内缺失")
                continue
            try:
                with zf.open(name) as fp:
                    got = _hash_stream(fp)
            except Exception as exc:
                problems.append(f"{name}：读取失败（{exc}）")
                continue
            if got != digest:
                problems.append(f"{name}：校验和不符（文件可能被改动）")

        # 数据库逐个解到临时目录验证：integrity + 降级判断。
        db_names = [n for n in names if _member_kind(n) == "db"]
        out["bots"] = len(db_names)
        tmp = tempfile.mkdtemp(prefix="hermes-inspect-")
        try:
            for name in db_names:
                if _safe_name(name):
                    continue
                local = os.path.join(tmp, uuid.uuid4().hex + ".db")
                try:
                    with zf.open(name) as src, open(local, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    verdict, version = _db_facts(local)
                except Exception as exc:
                    problems.append(f"{name}：无法读取数据库（{exc}）")
                    continue
                finally:
                    try:
                        os.remove(local)
                    except OSError:
                        pass
                if verdict != "ok":
                    problems.append(f"{name}：integrity_check={verdict}")
                    continue
                if root:
                    current = _current_user_version(root, _stem_of(name))
                    if current is not None and version > current:
                        problems.append(
                            f"{name}：备份的库版本（{version}）比本机（{current}）新，"
                            "请先升级插件再恢复"
                        )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        payload = _read_json(zf, CONFIG_ENTRY)
        if isinstance(payload, dict) and isinstance(payload.get("values"), dict):
            values = {str(k): v for k, v in payload["values"].items()}
            out["config"] = values
            out["config_keys"] = sorted(values)

    out["ok"] = not problems
    return out


def _hash_stream(fp) -> str:
    import hashlib

    digest = hashlib.sha256()
    for chunk in iter(lambda: fp.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def list_backups(out_dir: str) -> list[dict[str, Any]]:
    """列出备份目录里的包，新的在前。恢复点单独标 `kind`。"""
    rows: list[dict[str, Any]] = []
    try:
        names = sorted(
            n for n in os.listdir(out_dir)
            if n.startswith(("hermes-backup-", PRERESTORE_PREFIX))
            and n.endswith(".zip")
        )
    except OSError:
        return rows
    for name in reversed(names):
        path = os.path.join(out_dir, name)
        row: dict[str, Any] = {
            "filename": name,
            "size_bytes": 0,
            "created_at": 0.0,
            "bots": 0,
            "kind": "restore-point" if name.startswith(PRERESTORE_PREFIX) else "backup",
            "plugin_version": "",
        }
        try:
            row["size_bytes"] = os.path.getsize(path)
            with zipfile.ZipFile(path) as zf:
                manifest = _read_json(zf, MANIFEST) or {}
            row["created_at"] = float(manifest.get("created_at") or 0.0)
            row["bots"] = int(manifest.get("bots") or 0)
            row["plugin_version"] = str(manifest.get("plugin_version") or "")
        except Exception:
            row["kind"] = "broken"
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 恢复


def _extract_member(zf: zipfile.ZipFile, name: str, dest_root: str) -> str:
    """把成员解到 dest_root 下同名相对路径，返回落盘路径。"""
    reason = _safe_name(name)
    if reason:
        raise RestoreError(f"{name}：{reason}")
    target = os.path.join(dest_root, *name.split("/"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with zf.open(name) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def restore_backup(path: str, *, root: str, out_dir: str, confirm: str = "",
                   version: str = "", now: float | None = None,
                   before_apply: Callable[[], None] | None = None,
                   after_apply: Callable[[], None] | None = None) -> dict[str, Any]:
    """两阶段恢复。`confirm` 必须是 `"RESTORE"`。

    `before_apply` 在换入前一刻调用（调用方用来 `registry.close_all()`：
    Windows 上文件被打开时 `os.replace` 会直接失败）；`after_apply` 在全部
    换入成功后调用（重建索引与缓存）。两者失败都按恢复失败回滚。
    """
    if confirm != CONFIRM_WORD:
        raise RestoreError(f'恢复需要确认：confirm 必须是 "{CONFIRM_WORD}"')

    plan = inspect_backup(path, root=root)
    if not plan["ok"]:
        raise RestoreError("备份包不可用：" + "；".join(plan["problems"]))

    at = time.time() if now is None else float(now)
    root = os.path.abspath(str(root or ""))
    out_dir = os.path.abspath(str(out_dir or ""))
    os.makedirs(root, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 先留退路：恢复失败（甚至恢复成功但发现不对）都能再用这份快照回来。
    try:
        snapshot = backup_bots(
            root=root, out_dir=out_dir, version=version, now=at,
            prefix=PRERESTORE_PREFIX, keep=_PRERESTORE_KEEP,
        )
    except Exception as exc:
        raise RestoreError(f"恢复前快照失败，未改动任何文件：{exc}") from exc

    parent = os.path.dirname(root) or "."
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".hermes-restore-", dir=parent)
    old_root = os.path.join(staging, "_old")
    os.makedirs(old_root, exist_ok=True)
    placed: list[str] = []
    moved: list[tuple[str, str]] = []

    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                name = info.filename
                if info.is_dir() or name in (MANIFEST, CONFIG_ENTRY):
                    continue
                if _member_kind(name) in ("db", "meta", "known"):
                    _extract_member(zf, name, staging)

        # 二次校验：解出来的库必须完整（第一道校验的是包内字节，
        # 这一道校验的是「写进这个盘之后」的字节）。
        for dirpath, _dirs, files in os.walk(os.path.join(staging, BOTS_DIR)):
            for fname in files:
                if fname != BOT_DB_FILE:
                    continue
                verdict, _version = _db_facts(os.path.join(dirpath, fname))
                if verdict != "ok":
                    raise RestoreError(f"{fname} 解压后完整性检查失败：{verdict}")
        for fname in os.listdir(staging):
            if fname.endswith(".db"):
                verdict, _version = _db_facts(os.path.join(staging, fname))
                if verdict != "ok":
                    raise RestoreError(f"{fname} 解压后完整性检查失败：{verdict}")

        if before_apply is not None:
            before_apply()

        bots_root = os.path.join(root, BOTS_DIR)
        os.makedirs(bots_root, exist_ok=True)
        restored = 0

        def _aside(target: str) -> None:
            if not os.path.exists(target):
                return
            dest = os.path.join(old_root, uuid.uuid4().hex + "-" + os.path.basename(target))
            os.replace(target, dest)
            moved.append((dest, target))

        # 1) v2 账号目录：整目录替换。
        staged_bots = os.path.join(staging, BOTS_DIR)
        if os.path.isdir(staged_bots):
            for stem in sorted(os.listdir(staged_bots)):
                staged_dir = os.path.join(staged_bots, stem)
                if not os.path.isdir(staged_dir):
                    continue
                target = os.path.join(bots_root, stem)
                _aside(target)
                os.replace(staged_dir, target)
                placed.append(target)
                restored += 1

        # 2) v1 平铺库：就地转成 v2（库进目录，备注从 _notes.json 搬进
        #    bot.json），这样恢复完不需要再跑一次迁移。
        index: dict[str, str] = {}
        notes: dict[str, str] = {}
        try:
            with open(os.path.join(staging, INDEX_FILE), encoding="utf-8") as fp:
                raw = json.load(fp)
            index = {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        except Exception:
            index = {}
        try:
            with open(os.path.join(staging, NOTES_FILE), encoding="utf-8") as fp:
                raw = json.load(fp)
            notes = {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        except Exception:
            notes = {}

        for fname in sorted(os.listdir(staging)):
            if not fname.endswith(".db"):
                continue
            stem = fname[:-3]
            target_dir = os.path.join(bots_root, stem)
            if os.path.isdir(target_dir):
                _aside(target_dir)
            os.makedirs(target_dir, exist_ok=True)
            placed.append(target_dir)
            os.replace(os.path.join(staging, fname),
                       os.path.join(target_dir, BOT_DB_FILE))
            meta_path = os.path.join(target_dir, BOT_META_FILE)
            if not os.path.exists(meta_path):
                pid = index.get(stem, "") or stem
                _write_json(meta_path, {
                    "schema": 1,
                    "platform_id": pid,
                    "note": notes.get(pid, ""),
                    "description": "",
                    "created_at": at,
                    "overrides": {},
                })
            restored += 1

        _write_json(os.path.join(root, LAYOUT_FILE), {
            "schema": _LAYOUT_SCHEMA,
            "created_at": at,
            "updated_at": at,
        })

        if after_apply is not None:
            after_apply()

        return {
            "ok": True,
            "restored": restored,
            "snapshot": snapshot.get("path", ""),
            "snapshot_name": snapshot.get("filename", ""),
            "filename": os.path.basename(path),
            "created_at": plan["created_at"],
            "config": plan["config"],
            "config_keys": plan["config_keys"],
            "message": (
                f"已恢复 {restored} 个账号。恢复前数据快照："
                f"{snapshot.get('filename', '')}（可再恢复回来）。"
            ),
        }
    except Exception as exc:
        # 撤销：先删新换入的目录，再按逆序把旧目录搬回。
        for target in placed:
            shutil.rmtree(target, ignore_errors=True)
        for dest, target in reversed(moved):
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.replace(dest, target)
            except OSError:
                pass
        if isinstance(exc, RestoreError):
            raise
        raise RestoreError(
            f"恢复失败，已回滚：{exc}（恢复前快照：{snapshot.get('filename', '')}）"
        ) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def rollback_restore(snapshot_path: str, *, root: str, out_dir: str,
                     confirm: str = "", version: str = "",
                     now: float | None = None,
                     before_apply: Callable[[], None] | None = None,
                     after_apply: Callable[[], None] | None = None) -> dict[str, Any]:
    """从恢复前快照回滚。只接受 `hermes-pre-restore-` 前缀的包。"""
    name = os.path.basename(str(snapshot_path or ""))
    if not name.startswith(PRERESTORE_PREFIX):
        raise RestoreError("回滚只接受恢复前快照（hermes-pre-restore-*.zip）")
    return restore_backup(
        os.path.join(out_dir, name), root=root, out_dir=out_dir,
        confirm=confirm, version=version, now=now,
        before_apply=before_apply, after_apply=after_apply,
    )


def _write_json(path: str, data: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


__all__ = [
    "CONFIRM_WORD",
    "RestoreError",
    "inspect_backup",
    "list_backups",
    "restore_backup",
    "rollback_restore",
]
