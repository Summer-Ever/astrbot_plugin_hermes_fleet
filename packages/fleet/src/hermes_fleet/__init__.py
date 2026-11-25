"""hermes_fleet —— 插件的组装层。

依赖方向：本包是**组装根**，允许 import 各领域包的实现；领域包之间仍然
互不依赖（由 `tools/check_boundaries.py` 强制）。组装根只有这一处，
所以「谁依赖谁」是可穷举的。
"""

from . import admin, chain, export, resident, settings
from .registry import (
    MAX_NOTE,
    BotRegistry,
    UNKNOWN_BOT,
    bot_file_stem,
    bot_id_of,
    normalize_note,
)
from .service import BLOCK_LABELS, INJECT_HEADER, MAINT_NOTES, MemoryService, Scene
from .resident import INJECT_LABELS, append_system_prompt, direction_advice

__all__ = [
    "admin",
    "chain",
    "export",
    "resident",
    "settings",
    "BotRegistry",
    "UNKNOWN_BOT",
    "MAX_NOTE",
    "normalize_note",
    "bot_file_stem",
    "bot_id_of",
    "MemoryService",
    "Scene",
    "BLOCK_LABELS",
    "INJECT_HEADER",
    "INJECT_LABELS",
    "MAINT_NOTES",
    "append_system_prompt",
    "direction_advice",
]
