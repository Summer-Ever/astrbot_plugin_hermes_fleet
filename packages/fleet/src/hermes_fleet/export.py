"""把「某用户 + 某机器人账号」的记忆渲染成一份可发送的文档。

纯逻辑模块：只吃传进来的记录，不碰数据库、不落磁盘、不认识 AstrBot。
这样「导出到底写了什么」能在任意机器上逐条断言——尤其是「绝不把群友
的发言写进去」这条，不能靠人工看导出文件。

文档是 Markdown 而不是 JSON/CSV：收件人是普通人，要能直接在手机上读、
搜索、转发给自己的另一个账号；JSON 里的时间戳和转义字符对人没有意义。
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from datetime import datetime

from hermes_contracts import Memory, Message, ResidentEntry, ResidentScope

from .resident import INJECT_LABELS

# 单条原文在文档里的上限。一行超过千字既没法在手机上看，也会让整份
# 文档膨胀到发不出去；超长部分被截掉，但库里的原文不动。
MAX_LINE_CHARS = 800

# 文件名只允许这些字符。owner_id 来自适配器（openid/QQ 号/飞书 uid），
# 平台什么都可能塞进来；带斜杠就能写出导出目录之外，必须先洗。
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _one_line(text: object, cap: int = MAX_LINE_CHARS) -> str:
    """把任意内容压成一行：换行/制表/零宽字符都会破坏列表或表格。"""
    s = str(text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\x00-\x1f\u200b-\u200f\ufeff]", " ", s)
    s = " ".join(s.split())
    if cap > 0 and len(s) > cap:
        return s[:cap] + f"…（原文 {len(s)} 字，文档内截断）"
    return s


def _cell(text: object) -> str:
    """表格单元格：竖线必须转义，否则一行会被拆成两列。"""
    return _one_line(text).replace("|", "\\|")


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
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError):
        return "—"


def _clock(value: float) -> str:
    try:
        return datetime.fromtimestamp(float(value)).strftime("%H:%M")
    except (ValueError, OSError, OverflowError):
        return "--:--"


def export_filename(bot_id: object, owner_id: object,
                    ts: float | None = None) -> str:
    """ASCII 文件名，避免附件名在部分平台被改写成问号。

    时间戳用本地时间（用户看的是自己的钟），同秒内重复导出靠主调方
    检查文件是否存在后加序号。
    """
    def slug(raw: object, cap: int) -> str:
        s = _SAFE_CHARS.sub("_", str(raw or "")).strip("._-")
        return (s[:cap] or "x")

    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(ts if ts is not None else time.time()))
    return f"hermes-memory-{slug(bot_id, 24)}-{slug(owner_id, 12)}-{stamp}.md"


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


def render_document(
    *,
    bot_id: str,
    owner_id: str,
    note: str = "",
    resident: Sequence[ResidentEntry] = (),
    memories: Sequence[Memory] = (),
    messages: Sequence[Message] = (),
    memory_total: int | None = None,
    message_total: int | None = None,
    now: float | None = None,
    version: str = "",
    max_chat_chars: int = MAX_LINE_CHARS,
) -> str:
    """生成完整文档。调用方传进来的 messages 应当已经过 `pair_replies`。"""
    at = time.time() if now is None else float(now)
    name = f"{note}（{bot_id}）" if note else str(bot_id)
    resident = list(resident)
    memories = list(memories)
    messages = list(messages)
    n_mem = len(memories) if memory_total is None else int(memory_total)
    n_msg = len(messages) if message_total is None else int(message_total)

    by_scope: dict[str, list[ResidentEntry]] = {}
    for entry in resident:
        by_scope.setdefault(scope_title(entry.scope), []).append(entry)

    lines = [
        f"# 记忆导出 · {_one_line(name, 80)}",
        "",
        "> 本文件含个人对话原文，请妥善保存，不要外传。",
        "",
        "## 概览",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 机器人账号 | {_cell(name)} |",
        f"| 用户标识 | {_cell(owner_id)} |",
        f"| 导出时间 | {_ts(at)} |",
        f"| 常驻词条 | {len(resident)} 条（{len(by_scope)} 个场景） |",
        f"| 归档记忆 | {n_mem} 条" + (
            f"（本次写入 {len(memories)} 条）" if n_mem > len(memories) else ""
        ) + " |",
        f"| 聊天原文 | {n_msg} 条" + (
            f"（本次导出 {len(messages)} 条）" if n_msg > len(messages) else ""
        ) + " |",
        f"| 生成程序 | hermes_fleet {_one_line(version, 24) or '未知版本'} |",
        "",
        "## 一、常驻记忆",
        "",
        "每轮对话都会注入；按场景隔离，私聊与每个群各有一套。",
        "",
    ]
    if not by_scope:
        lines.append("（没有设置常驻词条。）")
    # 私聊在前、各群在后，和注入顺序一致；dict 插入顺序来自 SQL 的
    # scope 字符串排序，"group:..." 会排在 "private" 前面，不好读。
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
        if mem.visibility.value == "private":
            vis = "仅本人"
        elif mem.visibility.value == "group":
            vis = "本群"
        else:
            vis = "所有人"
        lines.append(
            f"- `#{int(mem.id)}` {_ts(mem.ts)} · {where} · "
            f"{_kind_label(mem.kind)} · {vis}：{_one_line(mem.content, max_chat_chars)}"
        )
    lines.append("")

    # 原文按会话分组：同一群的消息放一起，读起来才连贯。
    grouped: dict[str, list[Message]] = {}
    for msg in messages:
        grouped.setdefault(str(msg.group_id or ""), []).append(msg)

    lines.extend([
        "## 三、聊天原文", "",
        "从旧到新。群聊里只有**你**的发言和机器人的回复，不含其他群友的消息。", "",
    ])
    if not messages:
        lines.append("（没有聊天原文。）")
    for gid, group in grouped.items():
        lines.append(f"### {'私聊' if not gid else '群 ' + _one_line(gid, 60)}")
        lines.append("")
        for msg in group:
            who = "你" if msg.role == "user" else "机器人"
            lines.append(
                f"- **{_clock(msg.ts)} {who}**：{_one_line(msg.content, max_chat_chars)}"
            )
        lines.append("")

    lines.extend([
        "## 说明", "",
        "- 范围仅限本机器人账号；其他机器人账号的数据不在此文件中。",
        "- 归档记忆只含活跃条目；被合并/遗忘的条目不在其中（库里仍有留痕）。",
        "- 聊天原文只保留最近的一段；更早的消息仍在库里，不会因导出而删除。",
        "- 这份文件是数据快照，不受后续记忆更新影响。", "",
    ])
    return "\n".join(lines)


__all__ = [
    "MAX_LINE_CHARS",
    "export_filename",
    "pair_replies",
    "render_document",
    "scope_title",
    "stale_exports",
]
