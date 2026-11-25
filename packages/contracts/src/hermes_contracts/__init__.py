"""hermes_contracts —— 全平台的冻结合约。

这个包只有四个文件是「契约本体」，改动需要评审：

- `ids`     标识符与作用域（同一件事叫什么、属于谁）
- `errors`  失败语义（能不能重试、对应什么 HTTP 码）
- `models`  跨模块实体（字段与 DDL 一一对应）
- `ports`   模块间接口（唯一允许的通信方式）

外加：
- `clock`   注入式时钟（时间可测的地基）
- `conformance` 一致性测试套件（接口的「验收标准」）

**没有**业务逻辑、没有 IO、没有第三方依赖。这是故意的：
任何 agent 在任何机器上 `import hermes_contracts` 都必须成功，不需要装包、
不需要数据库、不需要 AstrBot。这条保证了并行开发时没有人被别人的环境卡住。
"""

from __future__ import annotations

from .clock import Clock, ManualClock, SystemClock
from .errors import (
    Conflict,
    ErrorCode,
    Forbidden,
    HermesError,
    InvalidArgument,
    NotFound,
    RateLimited,
    Unavailable,
)
from .ids import (
    EMPTY,
    BotId,
    GroupId,
    JobId,
    MemoryId,
    MemoryKey,
    MemoryScope,
    OwnerId,
    ResidentBlock,
    ResidentScope,
    SessionId,
    SessionKey,
    SkillId,
    Visibility,
    require_owner,
    validate_bot_id,
)
from .models import (
    MAX_MEMORY_CHARS,
    MAX_MESSAGE_CHARS,
    MAX_RESIDENT_BLOCK_CHARS,
    Bot,
    ChatMessage,
    Identity,
    InjectPayload,
    Job,
    JobKind,
    JobState,
    LlmResult,
    Memory,
    Message,
    MessageRole,
    RecallHit,
    ResidentEntry,
    Skill,
    clip,
)
from .ports import (
    Deps,
    Embedder,
    IdentityStore,
    JobQueue,
    LlmProvider,
    MemoryStore,
    MessageStore,
    ResidentStore,
    SkillStore,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # clock
    "Clock",
    "SystemClock",
    "ManualClock",
    # errors
    "ErrorCode",
    "HermesError",
    "InvalidArgument",
    "NotFound",
    "Forbidden",
    "Conflict",
    "Unavailable",
    "RateLimited",
    # ids
    "EMPTY",
    "BotId",
    "OwnerId",
    "GroupId",
    "SessionId",
    "SessionKey",
    "MemoryId",
    "SkillId",
    "JobId",
    "MemoryScope",
    "Visibility",
    "MemoryKey",
    "ResidentBlock",
    "ResidentScope",
    "validate_bot_id",
    "require_owner",
    # models
    "Bot",
    "Identity",
    "Memory",
    "RecallHit",
    "ResidentEntry",
    "InjectPayload",
    "Skill",
    "Message",
    "MessageRole",
    "Job",
    "JobKind",
    "JobState",
    "ChatMessage",
    "LlmResult",
    "clip",
    "MAX_MESSAGE_CHARS",
    "MAX_MEMORY_CHARS",
    "MAX_RESIDENT_BLOCK_CHARS",
    # ports
    "MemoryStore",
    "ResidentStore",
    "IdentityStore",
    "SkillStore",
    "MessageStore",
    "JobQueue",
    "Embedder",
    "LlmProvider",
    "Deps",
]
