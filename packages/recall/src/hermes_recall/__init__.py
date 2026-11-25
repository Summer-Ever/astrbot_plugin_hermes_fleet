"""hermes_recall —— 检索与重排。纯函数，零依赖，零 IO。

这个包是从 `astrbot_plugin_hermes_core/modules/recall.py` 原样搬过来的，
它是那个项目里**唯一可以整份复用**的模块：880 行、只用标准库、不碰数据库、
不花 token，所有函数都是 `输入 -> 输出`。

搬过来只加了一个 `adapter.py`，负责 `Memory` 实体与 `dict` 互转，
以及一个硬上限 `MAX_RERANK = 50`——用来物理阻止「把全库捞进 Python 打分」
那个病根被继承过来。

使用方式（重排 ≤50 条候选）：

    from hermes_recall import rerank
    hits = rerank(query, candidates, k=5, now=clock.now())

注意：这个模块**不做权限判断**。谁能看到哪些记忆由存储层的
`search_candidates` 决定，这里只负责排序。混进权限判断会让它不可测
（需要构造整个权限上下文），这也是原实现刻意保持的边界。
"""

from __future__ import annotations

from .adapter import MAX_RERANK, render, rerank, to_doc
from .paths import safe_join
from .recall import (
    bm25_scores,
    build_idf,
    context_text,
    expand_query,
    filter_in_window,
    format_memory_lines,
    gate_ok,
    gram_set,
    grams,
    jaccard,
    norm,
    rank,
    rank_fused,
    rank_messages,
    score,
    suppress_repeats,
)

__all__ = [
    # adapter
    "MAX_RERANK",
    "to_doc",
    "rerank",
    "render",
    # paths
    "safe_join",
    # recall 原语
    "norm",
    "grams",
    "gram_set",
    "jaccard",
    "build_idf",
    "score",
    "rank",
    "bm25_scores",
    "gate_ok",
    "rank_fused",
    "rank_messages",
    "context_text",
    "expand_query",
    "filter_in_window",
    "format_memory_lines",
    "suppress_repeats",
]
