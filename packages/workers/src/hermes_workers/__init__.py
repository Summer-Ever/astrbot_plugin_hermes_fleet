"""hermes_workers —— 后台任务。只依赖 `hermes_contracts` 的端口，不碰具体存储。"""

from .consolidate import ConsolidateOutcome, MemoryConsolidator
from .pairing import bigrams, build_groups, normalize, similarity
from .prompt import (
    BATCH_LIMIT,
    CONSOLIDATE_PROMPT,
    SUMMARY_PROMPT,
    build_listing,
    build_transcript,
    clean_text,
    is_injection,
    looks_like_complete_array,
    parse_decisions,
    parse_memories,
)
from .summarize import SessionSummarizer, SummaryOutcome

__all__ = [
    "BATCH_LIMIT",
    "CONSOLIDATE_PROMPT",
    "SUMMARY_PROMPT",
    "build_groups",
    "build_listing",
    "build_transcript",
    "bigrams",
    "clean_text",
    "is_injection",
    "looks_like_complete_array",
    "normalize",
    "parse_decisions",
    "parse_memories",
    "similarity",
    "ConsolidateOutcome",
    "MemoryConsolidator",
    "SessionSummarizer",
    "SummaryOutcome",
]
