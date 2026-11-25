"""管理台的纯逻辑层。**不 import astrbot，也不 import http。**

为什么单独一层：hermes_core 的 `webapi.py` 有 959 行，业务逻辑和 HTTP 解析
揉在一起，结果是「页面显示错了」这件事只能在装了 AstrBot 的机器上复现。
这里把所有判断（过滤、排序、告警分级）做成纯函数，`webapi.py` 只剩
「读参数 → 调函数 → 包 response」，十几行。

于是「哪个账号算异常」「告警怎么分级」这些真正会出错的判断，可以在任何
机器上被单测覆盖。

数据来源的边界（很重要，决定了这一页能不能扛 100 个账号）：

- `accounts_view` 只吃 `BotRegistry.accounts()` 的输出，那是不打开任何
  数据库连接就能拿到的文件元数据。**首屏永远走这条路径。**
- 记忆条数、未总结条数必须查库，只能走 `MemoryService.probe()`（有上限）
  或单账号详情。首屏拿不到就不显示，不猜、不占位。
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

# 排序键 → 取值函数。都是纯文件元数据，不查库。
_SORTS = {
    "recent": lambda r: -(r.get("last_write_at") or 0.0),
    "size": lambda r: -(r.get("size_bytes") or 0) - (r.get("wal_bytes") or 0),
    "name": lambda r: str(r.get("platform_id") or r.get("stem") or "").lower(),
}

SORTS = tuple(_SORTS)

# 告警级别：error 要立刻处理，warn 要关注，info 只是提示。
LEVEL_ORDER = {"error": 0, "warn": 1, "info": 2}


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def accounts_view(
    rows: Sequence[dict],
    *,
    query: str = "",
    sort: str = "recent",
    limit: int = 50,
    offset: int = 0,
    stale_days: float = 7.0,
    now: float | None = None,
) -> dict:
    """把 registry 的原始盘点做成页面直接能渲染的一页。

    分页在**过滤之后**做，且返回过滤后的 `total`——先分页再过滤会让
    「共 3 条」下面列出 0 条，是这类页面最常见的低级错误。
    """
    now = time.time() if now is None else float(now)
    stale_after = max(0.0, float(stale_days)) * 86400.0
    q = str(query or "").strip().lower()

    items: list[dict] = []
    for r in rows:
        pid = str(r.get("platform_id") or "")
        stem = str(r.get("stem") or "")
        if q and q not in pid.lower() and q not in stem.lower():
            continue
        size = int(r.get("size_bytes") or 0)
        wal = int(r.get("wal_bytes") or 0)
        last = float(r.get("last_write_at") or 0.0)
        age = (now - last) if last else None
        item = {
            "platform_id": pid,
            "stem": stem,
            "file": r.get("file") or "",
            "size_bytes": size,
            "wal_bytes": wal,
            "size_text": _human_bytes(size + wal),
            "last_write_at": last,
            "age_days": None if age is None else round(age / 86400.0, 1),
            "opened": bool(r.get("opened")),
            "orphan": bool(r.get("orphan")),
            "missing": bool(r.get("missing")),
            "stale": bool(age is not None and age > stale_after),
        }
        items.append(item)

    items.sort(key=_SORTS.get(sort, _SORTS["recent"]))
    total = len(items)
    limit = max(1, min(500, int(limit)))
    offset = max(0, int(offset))
    page = items[offset:offset + limit]

    return {
        "items": page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "summary": _summary(items, stale_days=stale_days),
        "alerts": _alerts(items, stale_days=stale_days),
    }


def _summary(items: Sequence[dict], *, stale_days: float) -> dict:
    return {
        "accounts": len(items),
        "total_bytes": sum(i["size_bytes"] + i["wal_bytes"] for i in items),
        "total_text": _human_bytes(sum(i["size_bytes"] + i["wal_bytes"] for i in items)),
        "opened": sum(1 for i in items if i["opened"]),
        "stale": sum(1 for i in items if i["stale"]),
        "orphan": sum(1 for i in items if i["orphan"]),
        "missing": sum(1 for i in items if i["missing"]),
        "stale_days": stale_days,
    }


def _alerts(items: Sequence[dict], *, stale_days: float) -> list[dict]:
    """把「需要人动手」的事挑出来，按严重度排序。

    为什么要分级而不是给个总数：100 个账号里 stale 5 个，运维需要知道的是
    **哪 5 个**。只给计数等于把排查工作原样丢回去。

    `orphan` 排在最前：索引里没有的文件，说明数据来路不明或被误拷，
    它不会被任何流程读到——静默失效里最难发现的一类。
    """
    out: list[dict] = []
    for i in items:
        if i["orphan"]:
            out.append({
                "level": "error", "code": "orphan", "platform_id": i["platform_id"],
                "message": f"库文件 {i['file']} 不在索引里，任何流程都读不到它",
            })
        elif i["missing"]:
            out.append({
                "level": "error", "code": "missing", "platform_id": i["platform_id"],
                "message": "账号在索引里，但库文件不见了（被删或盘故障）",
            })
        elif i["stale"]:
            out.append({
                "level": "warn", "code": "stale", "platform_id": i["platform_id"],
                "message": f"已 {i['age_days']} 天没有新数据（阈值 {stale_days:g} 天）",
            })
    out.sort(key=lambda a: (LEVEL_ORDER.get(a["level"], 9), a["platform_id"]))
    return out


def merge_probe(items: Sequence[dict], probed: Iterable[dict]) -> list[dict]:
    """把 `probe()` 的结果贴回列表项。

    probe 只覆盖被点名的账号；其余保持「未探测」，页面上显示为 `-` 而不是 0。
    用 0 占位会让人以为「这个账号没有记忆」，而实际上只是没查。
    """
    got = {str(p.get("platform_id")): p for p in probed}
    out = []
    for it in items:
        p = got.get(str(it.get("platform_id")))
        row = dict(it)
        if p is None:
            row["stats"] = None
        elif p.get("error"):
            row["stats"] = None
            row["stats_error"] = p["error"]
        else:
            row["stats"] = p.get("stats")
        out.append(row)
    return out


def health_view(
    accounts: Sequence[dict],
    *,
    root: str,
    opened: int,
    max_open: int,
    provider: Any = None,
    stale_days: float = 7.0,
    now: float | None = None,
) -> dict:
    """全局健康：模型链现状 + 账号盘点告警 + 容量水位。

    `provider` 走鸭子类型（`chain()` / `summary()` / `stats` / `log`）而不是
    导入 `FallbackProvider` 类型——`hermes_fleet` 不能依赖 `hermes_workers`
    （依赖方向是 workers → contracts，反过来就成环）。传 None 时这一段留空，
    页面显示「模型链信息不可用」，不当成健康。
    """
    now = time.time() if now is None else float(now)
    stale_after = max(0.0, float(stale_days)) * 86400.0

    normalized = []
    for r in accounts:
        last = float(r.get("last_write_at") or 0.0)
        normalized.append({
            **r,
            "age_days": None if not last else round((now - last) / 86400.0, 1),
            "stale": bool(last and (now - last) > stale_after),
        })

    model: dict = {"available": False}
    if provider is not None:
        try:
            chain = list(provider.chain())
            stats = provider.stats
            model = {
                "available": True,
                "chain": chain,
                "summary": provider.summary(),
                "calls": int(getattr(stats, "calls", 0) or 0),
                "failures": int(getattr(stats, "failures", 0) or 0),
                "by_kind": dict(getattr(stats, "by_kind", {}) or {}),
                # 只给最近 20 条：这是排障用的时间线，不是审计日志。
                "recent": [
                    {"ts": a.ts, "provider": a.provider, "purpose": a.purpose,
                     "ok": a.ok, "kind": a.kind, "latency_ms": a.latency_ms}
                    for a in list(getattr(provider, "log", []))[-20:]
                ],
            }
        except Exception as exc:  # noqa: BLE001 — 观测接口坏了不能拖垮健康页
            model = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "root": root,
        "capacity": {
            "opened": opened,
            "max_open": max_open,
            # 打开数到顶不是错误，但要知道：到顶后每次访问新账号都要驱逐。
            "saturated": max_open > 0 and opened >= max_open,
        },
        "accounts": {
            "total": len(normalized),
            "stale": sum(1 for a in normalized if a["stale"]),
            "orphan": sum(1 for a in normalized if a.get("orphan")),
            "missing": sum(1 for a in normalized if a.get("missing")),
            "bytes": sum(int(a.get("size_bytes") or 0) + int(a.get("wal_bytes") or 0)
                         for a in normalized),
        },
        "stale_days": stale_days,
        "model": model,
        "alerts": _alerts(
            [
                {
                    "platform_id": a.get("platform_id") or a.get("stem") or "",
                    "file": a.get("file") or "",
                    "orphan": bool(a.get("orphan")),
                    "missing": bool(a.get("missing")),
                    "stale": a["stale"],
                    "age_days": a["age_days"],
                }
                for a in normalized
            ],
            stale_days=stale_days,
        )[:50],
    }
