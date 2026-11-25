"""把「某用户 + 某机器人账号」的数据渲染成**两份**可发送的文档。

1. `render_chat()` —— 纯聊天原文（含年月日），给人读/归档用；
2. `render_config()` —— 常驻 + 归档记忆，带机器可读块，能导回去复原配置。

纯逻辑模块：只吃传进来的记录，不碰数据库、不落磁盘、不认识 AstrBot。
「绝不把群友的发言写进去」这条隐私边界必须能在任意机器上逐条断言。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Sequence
from datetime import datetime

from hermes_contracts import (
    MAX_MEMORY_CHARS,
    MAX_RESIDENT_BLOCK_CHARS,
    Memory,
    Message,
    ResidentBlock,
    ResidentEntry,
    ResidentScope,
)

from .resident import INJECT_LABELS

# 单条原文在文档里的上限。0 = 不截断：用户明确要求「几千万字也没多大
# 占用，直接发出」，所以原文文件默认全量；这个常量只供记忆文档里引用
# 归档记忆时兜底（归档内容本身受 MAX_MEMORY_CHARS 限制，不会很长）。
MAX_LINE_CHARS = 800

# 单份导出文件的硬上限（字节）。超了不发——平台附件的实际限制各家不同，
# 100MB 是保守值；让用户联系管理员调导出策略，而不是发一半出去。
MAX_EXPORT_BYTES = 100 * 1024 * 1024

# 文件名只允许这些字符。owner_id 来自适配器（openid/QQ 号/飞书 uid），
# 平台什么都可能塞进来；带斜杠就能写出导出目录之外，必须先洗。
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")

# 机器可读块的定界符。放在记忆文档**开头**：解析时一眼能看到版本，
# 人读部分在上面被注释包住，不干扰 Markdown 渲染。
IMPORT_OPEN = "<!-- hermes-import-v1"
IMPORT_CLOSE = "-->"
IMPORT_FORMAT = "hermes-fleet-memory"
IMPORT_VERSION = 1


def _one_line(text: object, cap: int = 0) -> str:
    """把任意内容压成一行：换行/制表/零宽字符都会破坏列表或表格。

    原文文件里 cap=0 表示不截断——换行在这里被压成空格仅为了「一条消息
    占一行」的格式，内容本身一个字都不少。
    """
    s = str(text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\x00-\x1f\u200b-\u200f\ufeff]", " ", s)
    s = " ".join(s.split())
    if cap > 0 and len(s) > cap:
        return s[:cap] + f"…（原文 {len(s)} 字，文档内截断）"
    return s


def _cell(text: object) -> str:
    """表格单元格：竖线必须转义，否则一行会被拆成两列。"""
    return _one_line(text, MAX_LINE_CHARS).replace("|", "\\|")


def scope_title(scope: ResidentScope) -> str:
    return "私聊" if scope.kind == "private" else f"群 {scope.group_id}"


def _scope_label(memory: Memory) -> str:
    if memory.scope.value == "global":
        return "全局"
    if memory.scope.value == "group" or memory.group_id:
        return f"群 {memory.group_id}" if memory.group_id else "群"
    return "私聊"


def _kind_label(kind: str) -> str:
    return {
        "fact": "事实", "profile": "画像", "preference": "偏好",
        "summary": "总结",
    }.get(str(kind or ""), str(kind or "事实"))


def _ts(value: float) -> str:
    """完整时间戳（含年月日）。用户反馈原文只有时分、看不出是哪天。"""
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError):
        return "—"


def export_filename(bot_id: object, owner_id: object,
                    ts: float | None = None, *, kind: str = "memory") -> str:
    """ASCII 文件名，避免附件名在部分平台被改写成问号。

    `kind` 区分两份文件：`chat` = 纯原文，`memory` = 可导入的记忆配置。
    同秒内重复导出靠主调方检查文件是否存在后加序号。
    """
    def slug(raw: object, cap: int) -> str:
        s = _SAFE_CHARS.sub("_", str(raw or "")).strip("._-")
        return (s[:cap] or "x")

    stamp = time.strftime(
        "%Y%m%d-%H%M%S", time.localtime(ts if ts is not None else time.time())
    )
    prefix = "hermes-chat" if kind == "chat" else "hermes-memory"
    return f"{prefix}-{slug(bot_id, 24)}-{slug(owner_id, 12)}-{stamp}.md"


def stale_exports(entries: Iterable[tuple[str, float]], *, now: float,
                  keep_days: float, max_files: int = 60) -> list[str]:
    """该删的旧导出文件。保留 `max_files` 个上限是防止无限占磁盘。

    按 mtime 排序，所以系统时钟被调回过去时也不会删掉刚生成的文件。
    """
    items = sorted(
        ((str(name), float(ts)) for name, ts in entries),
        key=lambda x: x[1], reverse=True,
    )
    cutoff = float(now) - max(0.0, float(keep_days)) * 86400
    stale = {name for name, ts in items[max(0, int(max_files)):]}
    stale.update(name for name, ts in items if ts < cutoff)
    return sorted(stale)


def pair_replies(messages: Iterable[Message], owner_id: object) -> list[Message]:
    """只保留本人的发言，以及**紧跟其后的**机器人回复。

    群会话里 session_id 是全群共用的。即使 SQL 已把范围缩到「本人说过话
    的会话」，仍会有别人的行混进来；这里按会话逐行扫，遇到别人的发言就
    丢，并且只允许它前后紧邻的机器人回复进入文档。

    「机器人回复」的判定不是「role=assistant 且 sender 为空」——老库里的
    sender_id 什么都可能没有，用「上一条本人发言」这一侧向信号更稳。
    """
    owner = str(owner_id or "")
    out: list[Message] = []
    last_owner: dict[str, str] = {}
    for msg in messages:
        sess = str(msg.session_id)
        if msg.role == "user":
            who = str(msg.sender_id or "")
            last_owner[sess] = who
            if owner and who == owner:
                out.append(msg)
        elif msg.role == "assistant":
            if last_owner.get(sess) == owner:
                out.append(msg)
        # system 行不进导出：那是提示词，不是聊天原文
    return out


def render_chat(
    *,
    bot_id: str,
    owner_id: str,
    note: str = "",
    messages: Sequence[Message] = (),
    now: float | None = None,
    version: str = "",
    message_total: int | None = None,
) -> str:
    """聊天原文文档：纯文字、一条消息一行、带完整年月日。

    不再截断单条内容（用户明确要求全量），但群友的话依旧不写进来。
    按会话分组：私聊一段、每个群各一段，读起来才连贯。
    """
    at = time.time() if now is None else float(now)
    name = f"{note}（{bot_id}）" if note else str(bot_id)
    messages = list(messages)
    n = len(messages) if message_total is None else int(message_total)

    lines = [
        f"# 对话原文 · {_one_line(name, 80)}",
        "",
        ">",
        f"> 用户标识：{_one_line(owner_id, 120)}",
        f"> 导出时间：{_ts(at)}　|　共 {n} 条　|　生成程序：hermes_fleet "
        f"{_one_line(version, 24) or '未知版本'}",
        ">",
        "> 本文件含个人对话原文，请妥善保存，不要外传。",
        "> 群聊中只含本人的发言与紧随其后的机器人回复，不含其他群友的消息。",
        "",
    ]
    grouped: dict[str, list[Message]] = {}
    for msg in messages:
        grouped.setdefault(str(msg.group_id or ""), []).append(msg)

    if not grouped:
        lines.append("（没有聊天原文。）")
    for gid, group in grouped.items():
        lines.extend([
            f"## {'私聊' if not gid else '群 ' + _one_line(gid, 80)}",
            "",
        ])
        for msg in group:
            who = "你" if msg.role == "user" else "机器人"
            # 只用空格压平换行；正文一个字不删
            lines.append(f"[{_ts(msg.ts)}] {who}：{_one_line(msg.content)}")
        lines.append("")
    lines.extend([
        "---",
        "",
        "所有原文仍保留在本机器人账号的记忆库中；导出不会删除任何数据。",
        "",
    ])
    return "\n".join(lines)


def _import_block(
    *, bot_id: str, owner_id: str, note: str, at: float, version: str,
    resident: Sequence[ResidentEntry], memories: Sequence[Memory],
) -> str:
    """生成机器可读的导入块（JSON，非 ASCII 字符原样保留）。"""
    payload = {
        "format": IMPORT_FORMAT,
        "version": IMPORT_VERSION,
        "bot_id": str(bot_id),
        "owner_id": str(owner_id),
        "note": str(note or ""),
        "exported_at": float(at),
        "generator": f"hermes_fleet {version}",
        "resident": [
            {
                "scope": ("private" if e.scope.kind == "private"
                          else f"group:{e.scope.group_id}"),
                "block": e.block.value,
                "text": str(e.text),
            }
            for e in resident
        ],
        "memories": [
            {
                "content": str(m.content),
                "scope": m.scope.value,
                "visibility": m.visibility.value,
                "group_id": str(m.group_id or ""),
                "kind": str(m.kind or "fact"),
                "ts": float(m.ts or 0.0),
                "importance": float(m.importance or 0.0),
            }
            for m in memories
        ],
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # 正文里若出现定界符就拆不掉，转义成实体（JSON 里不会自然出现 "-->"，
    # 但用户内容什么都可能，宁可多防一层）。
    body = body.replace(IMPORT_CLOSE, "--&gt;")
    return f"{IMPORT_OPEN}\n{body}\n{IMPORT_CLOSE}"


def render_memory_config(
    *,
    bot_id: str,
    owner_id: str,
    note: str = "",
    resident: Sequence[ResidentEntry] = (),
    memories: Sequence[Memory] = (),
    memory_total: int | None = None,
    now: float | None = None,
    version: str = "",
) -> str:
    """记忆配置文档：人能读，机器也能导回去。

    第一块是 `<!-- hermes-import-v1 ... -->`（可解析的 JSON），后面是给人
    看的表和清单。导入时只认第一块，人的排版怎么改都不影响解析。
    """
    at = time.time() if now is None else float(now)
    name = f"{note}（{bot_id}）" if note else str(bot_id)
    resident = list(resident)
    memories = list(memories)
    n_mem = len(memories) if memory_total is None else int(memory_total)
    by_scope: dict[str, list[ResidentEntry]] = {}
    for entry in resident:
        by_scope.setdefault(scope_title(entry.scope), []).append(entry)

    lines = [
        f"# 记忆配置 · {_one_line(name, 80)}",
        "",
        _import_block(
            bot_id=bot_id, owner_id=owner_id, note=note, at=at,
            version=version, resident=resident, memories=memories,
        ),
        "",
        "> 把这份文件发回机器人并附「导入记忆」，即可复原常驻与归档记忆。",
        "> 导入会**合并**：常驻按词条整段覆盖，归档记忆追加为新条目。",
        ">",
        "> 本文件含个人信息，请妥善保存，不要外传。",
        "",
        "## 概览",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 机器人账号 | {_cell(name)} |",
        f"| 用户标识 | {_cell(owner_id)} |",
        f"| 导出时间 | {_ts(at)} |",
        f"| 常驻词条 | {len(resident)} 条（{len(by_scope)} 个场景） |",
        f"| 归档记忆 | {n_mem} 条"
        + (f"（本次写入 {len(memories)} 条）" if n_mem > len(memories) else "")
        + " |",
        f"| 生成程序 | hermes_fleet {_one_line(version, 24) or '未知版本'} |",
        "",
        "## 一、常驻记忆",
        "",
        "每轮对话都会注入；按场景隔离，私聊与每个群各有一套。",
        "",
    ]
    if not by_scope:
        lines.append("（没有设置常驻词条。）")
    for scope_name in sorted(by_scope, key=lambda s: (0 if s == "私聊" else 1, s)):
        entries = by_scope[scope_name]
        lines.extend([f"### {_one_line(scope_name, 80)}", "",
                      "| 词条 | 内容 |", "|---|---|"])
        for entry in entries:
            label = INJECT_LABELS.get(entry.block, entry.block.value)
            lines.append(f"| {_cell(label)} | {_cell(entry.text)} |")
        lines.append("")

    lines.extend([
        "## 二、归档记忆", "",
        "按需检索，不会每轮注入。被合并或遗忘的条目不在此列。", "",
    ])
    if not memories:
        lines.append("（没有归档记忆。）")
    for mem in memories:
        where = _scope_label(mem)
        vis = {
            "private": "仅本人", "group": "本群", "public": "所有人",
        }.get(mem.visibility.value, "所有人")
        lines.append(
            f"- `#{int(mem.id)}` {_ts(mem.ts)} · {where} · "
            f"{_kind_label(mem.kind)} · {vis}：{_one_line(mem.content, MAX_LINE_CHARS)}"
        )
    lines.extend([
        "",
        "---",
        "",
        "- 范围仅限本机器人账号；其他机器人账号的数据不在此文件中。",
        "- 归档记忆只含活跃条目；被合并/遗忘的条目不在其中（库里仍有留痕）。",
        "- 聊天原文在另一份文件里，两份文件互相独立。",
        "- 这份文件是数据快照，不受后续记忆更新影响。",
        "",
    ])
    return "\n".join(lines)


def parse_memory_config(text: str) -> dict:
    """从记忆配置文档里解析出可导入的数据。

    只认开头的 `<!-- hermes-import-v1 ... -->` 块，前面的说明文字随便改
    都不影响；找不到块、格式不对、字段不合法，全部抛 ValueError，由命令
    层转成用户看得懂的提示——绝不静默导入一半。
    """
    raw = str(text or "")
    start = raw.find(IMPORT_OPEN)
    if start < 0:
        raise ValueError("文件里没有导入块，请确认是本插件导出的记忆配置文档")
    body_start = raw.find("\n", start)
    if body_start < 0:
        raise ValueError("导入块格式不完整")
    end = raw.find(IMPORT_CLOSE, body_start)
    if end < 0:
        raise ValueError("导入块没有闭合")
    payload_text = raw[body_start + 1:end].strip()
    try:
        data = json.loads(payload_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"导入块不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("导入块格式错误")
    if data.get("format") != IMPORT_FORMAT:
        raise ValueError("不是 hermes_fleet 记忆配置文档")
    if int(data.get("version", 0)) != IMPORT_VERSION:
        raise ValueError(f"不支持的导入格式版本：{data.get('version')}")

    resident = data.get("resident", [])
    memories = data.get("memories", [])
    if not isinstance(resident, list) or not isinstance(memories, list):
        raise ValueError("导入块的 resident/memories 字段格式错误")

    out_resident: list[dict] = []
    for item in resident:
        if not isinstance(item, dict):
            raise ValueError("常驻词条格式错误")
        scope = str(item.get("scope", ""))
        if scope != "private" and not scope.startswith("group:"):
            raise ValueError(f"常驻词条 scope 不合法：{scope!r}")
        if scope.startswith("group:") and len(scope) <= 6:
            raise ValueError(f"常驻词条群号为空：{scope!r}")
        try:
            ResidentBlock(str(item.get("block", "")))
        except ValueError as exc:
            raise ValueError(f"未知常驻词条：{item.get('block')!r}") from exc
        text = str(item.get("text", "") or "").strip()
        if len(text) > MAX_RESIDENT_BLOCK_CHARS:
            raise ValueError("常驻词条超过长度上限")
        out_resident.append({"scope": scope, "block": str(item["block"]), "text": text})

    out_memories: list[dict] = []
    for item in memories:
        if not isinstance(item, dict):
            raise ValueError("归档记忆格式错误")
        content = str(item.get("content", "") or "").strip()
        if not content or len(content) > MAX_MEMORY_CHARS:
            raise ValueError("归档记忆为空或超过长度上限")
        out_memories.append({
            "content": content,
            "scope": str(item.get("scope", "user") or "user"),
            "visibility": str(item.get("visibility", "private") or "private"),
            "group_id": str(item.get("group_id", "") or ""),
            "kind": str(item.get("kind", "fact") or "fact"),
            "ts": float(item.get("ts", 0.0) or 0.0),
            "importance": float(item.get("importance", 0.0) or 0.0),
        })

    return {
        "format": IMPORT_FORMAT,
        "version": IMPORT_VERSION,
        "bot_id": str(data.get("bot_id", "") or ""),
        "owner_id": str(data.get("owner_id", "") or ""),
        "note": str(data.get("note", "") or ""),
        "resident": out_resident,
        "memories": out_memories,
    }


def check_export_size(*docs: str) -> tuple[bool, int]:
    """UTF-8 字节数是否全部在 100MB 内。返回 (是否超限, 总字节)。"""
    total = sum(len(str(d or "").encode("utf-8")) for d in docs)
    return total > MAX_EXPORT_BYTES, total


__all__ = [
    "IMPORT_CLOSE", "IMPORT_FORMAT", "IMPORT_OPEN", "IMPORT_VERSION",
    "MAX_EXPORT_BYTES", "MAX_LINE_CHARS",
    "check_export_size", "export_filename", "pair_replies",
    "parse_memory_config", "render_chat", "render_memory_config",
    "scope_title", "stale_exports",
]
