"""把 `recall.py` 的纯函数接到契约类型上。

`recall.py` 是从 hermes_core 原样搬过来的（880 行，仅标准库，全项目质量最高
的一部分）。它操作的是 `dict`，而新平台操作的是 `Memory` 实体。这个适配层
是两者之间**唯一**的接缝，所以只有两个函数，且都是纯函数。

为什么不改 `recall.py` 让它直接吃 `Memory`：
    那会把「排序数学」和「实体形状」绑死。排序是经过大量调参的东西，改动风险
    高；而实体形状未来还会变（比如加 embedding 字段）。隔一层 dict 转换，
    两边可以各自演进。

为什么限制 `k` 上限：
    hermes_core 最严重的性能问题，是把最多 9000 行候选捞进 Python 逐行建
    gram_set 再打分。这里的 `MAX_RERANK` 就是硬闸：无论上游传多少候选，
    进重排的都只有 50 条。检索质量靠存储层的向量/trigram 召回，不靠这一层
    暴力扫全库。
"""

from __future__ import annotations

from typing import Sequence

from hermes_contracts import Memory, RecallHit

from .recall import format_memory_lines, norm, rank_fused

# 进入精排的候选上限。超过这个数量，说明存储层的 top-k 没生效，应该报错而不是硬扛。
MAX_RERANK = 50


def to_doc(memory: Memory) -> dict:
    """`Memory` -> `recall` 认识的 dict。

    带上 `_ref` 保留原对象引用，避免「按 id 反查回实体」这一步——
    反查要么多一次 IO，要么要求调用方自己建索引，两种都不划算。
    """
    return {
        "id": int(memory.id),
        "content": memory.content,
        "ts": memory.ts,
        "kind": memory.kind,
        "hits": memory.hits,
        "importance": memory.importance,
        "_ref": memory,
    }


def rerank(
    query: str,
    candidates: Sequence[Memory],
    *,
    k: int = 5,
    now: float,
    gate: bool = True,
) -> list[RecallHit]:
    """对候选做 RRF 融合 + 池内精排 + 闸门，返回前 k 条。

    `now` 必填（不给默认 `time.time()`）：时间衰减必须可测，
    否则「23 小时后该降权」这类断言只能靠真的等 23 小时。

    空查询 / 过短查询直接返回空列表——这是 `rank_fused` 内部 `MIN_QUERY_CHARS`
    的策略：「嗯」「在吗」这类查询召回不到东西是正常的，不该硬凑几条塞进 prompt。
    """
    if not candidates:
        return []
    if len(candidates) > MAX_RERANK:
        raise ValueError(
            f"进重排的候选 {len(candidates)} 条超过上限 {MAX_RERANK}："
            "存储层的 top-k 没有生效，不要在这一层补"
        )

    docs = [to_doc(m) for m in candidates]
    ranked = rank_fused(query, docs, k=k, now=now, gate=gate, explain=True)

    hits: list[RecallHit] = []
    for row in ranked:
        ref = row.get("_ref")
        if not isinstance(ref, Memory):
            continue
        hits.append(
            RecallHit(
                memory=ref,
                score=float(row.get("_score") or 0.0),
                debug={
                    k2: float(v)
                    for k2, v in row.items()
                    if k2.startswith("_") and isinstance(v, (int, float))
                },
            )
        )
    return hits


def render(hits: Sequence[RecallHit], *, max_lines: int = 12) -> str:
    """把命中渲染成注入用的文本。

    直接用 `recall.format_memory_lines`，保证注入格式与 hermes_core 一致——
    那个格式是调出来的：编号放在**行尾**（放行首会被 markdown 当标题吃掉），
    `kind` 提到分组标题（六条平铺时逐条标 kind 纯属浪费 token）。

    编号必须保留：`save_memory(replaces=)` 和 `forget_memory(id=)` 全靠它定位条目，
    去掉编号等于让模型无法引用已有记忆。
    """
    rows = [to_doc(h.memory) for h in hits[:max_lines]]
    return "\n".join(format_memory_lines(rows))


__all__ = [
    "MAX_RERANK",
    "to_doc",
    "rerank",
    "render",
    "norm",
    "rank_fused",
    "format_memory_lines",
]
