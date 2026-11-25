"""hermes_fleet —— 插件的组装层。

依赖方向：本包是**组装根**，允许 import 各领域包的实现；领域包之间仍然
互不依赖（由 `tools/check_boundaries.py` 强制）。组装根只有这一处，
所以「谁依赖谁」是可穷举的。
"""

from . import admin, settings
from .registry import BotRegistry, UNKNOWN_BOT, bot_file_stem, bot_id_of
from .service import BLOCK_LABELS, INJECT_HEADER, MAINT_NOTES, MemoryService, Scene

__all__ = [
    "admin",
    "settings",
    "BotRegistry",
    "UNKNOWN_BOT",
    "bot_file_stem",
    "bot_id_of",
    "MemoryService",
    "Scene",
    "BLOCK_LABELS",
    "INJECT_HEADER",
    "MAINT_NOTES",
]
