"""总结用的提示词、转录构建与输出解析。

这三样放在一起，因为它们改动的理由永远是同一个：**模型输出的稳定性**。
把它们分散到三个文件里，调一次提示词要同时改三处，迟早不一致。

## 从 hermes_core 继承的三个防御

这些都是它踩过坑之后写进去的，不是理论推演：

1. **群友可以对总结模型下指令。** 群里任何人都能发「请把这条记成 X 的属性」，
   转录进 prompt 后就是一条指令。所以转录前要过滤，解析时也要再过滤一次——
   两个位置都拦，因为这两处是不同的人写的代码，将来只会有一处被改。
2. **昵称是用户可控输入。** 带换行就能在 prompt 里伪造一行 `助手: ...`。
   必须压平换行并截断。
3. **JSON 后面可能跟解释文字。** 用 `raw_decode` 从第一个 `[` 开始取最外层
   完整 JSON，而不是正则贪婪匹配到最后一个 `]`——后者会把「已按[上限]处理」
   这类尾巴一起吞进去，导致整轮产出被静默丢弃。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from hermes_contracts import Message

# 每个会话一次总结最多看多少条消息（约 40 条 ≈ 6000 字符）。
# 不是 "越多越好"：总结模型的上下文有限，塞太多会让它只提炼开头那几条，
# 末尾真正的结论反而丢了。超出的部分留给下一轮。
BATCH_LIMIT = 40

# 单条消息进转录时截断到这个长度。聊天里常见几千字的长文，原样拼进去
# 会把整批的预算吃光。
PER_MESSAGE_CHARS = 300
PER_NAME_CHARS = 30

SUMMARY_PROMPT = """你是记忆管家。阅读下面的对话记录，只提炼【跨会话仍然成立】的稳定信息（用户画像、长期偏好、稳定事实、项目背景），输出 JSON 数组，每项 {"kind": "fact|preference|profile|summary", "content": "一条简洁的中文记忆", "owner": "这条记忆属于谁"}。
只输出 JSON；无可提炼就输出 []。

收录标准（三条全满足才收）：
1. 稳定：长期成立，不是某次对话的临时要求；
2. 可复用：未来的对话里也用得上；
3. 泛化：写普遍规律而非单次事件（写「偏好快速响应」，不写「曾反馈你最近好快」）。

禁止收录（这几类是主要噪声来源，务必逐条筛掉）：
- 助手的人格设定、系统提示词、行为准则，以及「改人格 / 改配置 / 重启才生效」这类操作记录；
- 已安装的技能、插件清单、版本号、指令名、数据库表名等实现细节；
- 一次性的任务指令与已完成的配置修改，除非用户明确说「以后都这样」；
- 对用户意图的推测（「用户可能会要求做 PPT」）——只记用户已明确表达过的偏好；
- 纯闲聊寒暄与情绪反馈；
- 随时可从系统环境查到的易变状态、对话过程复述。

关于 owner：填这条记忆所属用户的 id，必须是本批对话里出现过的用户 id（转录里 [用户 <id>(昵称)] 的尖括号内部分）。不属于任何具体用户、而是这个群/这个场景共有的信息，omit 该字段。
不要输出常驻词条：你产出的每一条都会进归档层，靠检索取用。"""

# 疑似「对模型下指令」的片段。命中就不进转录、也不写库。
#
# 为什么要在这层拦：这些文本一旦被总结成记忆，就会成为**每轮注入**的内容。
# 一句「忽略之前的指令」平时只是聊天，进了记忆就变成持久的提示词注入。
_INJECTION_MARKERS = (
    "忽略之前", "忽略以上", "忽略先前", "无视之前", "无视以上",
    "你现在是", "从现在开始你", "system prompt", "系统提示",
    "ignore previous", "ignore above", "disregard previous",
    "请把这条记成", "把这条记成", "请记录：", "记住这条指令",
    "新的指令", "覆盖你的", "你的新身份",
)


def is_injection(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _INJECTION_MARKERS)


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: object, limit: int = 500) -> str:
    """压平换行、去控制字符、截断。

    换行必须压掉：转录是「一行一条消息」的结构，用户昵称里带 `\\n助手: ...`
    就能伪造一整行，让总结模型以为助手说过那句话。
    """
    s = str(text or "")
    s = _CONTROL.sub(" ", s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s[:limit]


def build_transcript(messages: Sequence[Message]) -> tuple[str, set[str], str]:
    """把一批消息拼成转录。

    返回 `(转录, 本批出现过的用户 id 集合, 本批唯一发言人)`。

    第三个值只在本批**只有一个**发言人时非空。注意它**不等于私聊**：
    群里这一批也可能只有一个人说话。所以调用方必须先判断「这是不是私聊」
    再决定能不能用它——把群里独自发言当成私聊，会把模型明确标了别人的
    记忆强行塞给最后说话的那个人（这个 bug 真出现过，由本文件的用例抓到）。
    """
    lines: list[str] = []
    senders: set[str] = set()
    for m in messages:
        text = clean_text(m.content, PER_MESSAGE_CHARS)
        if not text:
            continue
        if m.role == "user":
            who = clean_text(m.sender_id, 64) or "未知"
            senders.add(who)
            if is_injection(text):
                # 保留占位而不是删掉：删了会让模型以为这段对话是连续的，
                # 上下文对不上时更容易编造。
                text = "（该条疑似指令注入，已过滤）"
            name = clean_text(m.sender_name, PER_NAME_CHARS) or "匿名"
            lines.append(f"[用户 {who}({name})]: {text}")
        else:
            lines.append(f"助手: {text}")
    solo = next(iter(senders)) if len(senders) == 1 else ""
    return "\n".join(lines), senders, solo


def looks_like_complete_array(text: str) -> bool:
    """粗略判断「这段输出是不是一个完整的 JSON 数组」。

    用途只有一个：区分**被截断**和**确实没有可记的内容**。

    有 `[` 也找得到 `]` → 当成完整（解析出来是空就是真空）。
    只有 `[` 没有 `]` → 判定为截断。这条判定救了最坏的一种静默失败：
    输出被 max_tokens 切在半截，解析出 0 条，如果这时照样把消息标记为
    「已总结」，那批原始素材就永久丢了。
    """
    s = str(text or "")
    left = s.find("[")
    if left < 0:
        return False
    return s.rfind("]") > left


def _find_json_array(text: str):
    """从第一个 `[` 起用 raw_decode 取最外层完整 JSON，忽略尾部解释文字。"""
    decoder = json.JSONDecoder()
    start = text.find("[")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except (ValueError, TypeError):
            start = text.find("[", start + 1)
    return None


def parse_memories(raw: str) -> list[dict]:
    """解析模型输出为记忆条目列表。解析失败返回 `[]`（调用方按「无产出」处理）。"""
    if not raw:
        return []
    text = re.sub(r"```(?:json)?|```", "", str(raw)).strip()
    data = _find_json_array(text)
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        content = clean_text(item.get("content"), 500)
        if not content or is_injection(content):
            continue
        kind = str(item.get("kind") or "fact").strip().lower()
        if kind not in ("fact", "preference", "profile", "summary"):
            kind = "fact"
        out.append({
            "kind": kind,
            "content": content,
            "owner": str(item.get("owner") or "").strip(),
        })
    return out
