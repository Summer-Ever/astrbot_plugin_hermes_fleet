"""错误码：跨越模块边界唯一允许传递的「失败语义」。

为什么不用裸异常：
    worker 要在失败后决定「重试 / 立刻死信 / 退避」，API 要把失败翻成 HTTP
    状态码，后台任务要按错误类型告警。这三件事如果各自 `except Exception`
    再猜，就会出现 hermes_core 那种「所有异常都当可重试」的退避：
    一个参数错误被重试 5 次、耗掉 5 分钟，最后仍然失败。

约定：
- **可重试**（retryable=True）：网络抖动、限流、依赖短暂不可用。
- **不可重试**：参数错、权限不足、资源不存在。重试只会浪费配额。
- 任何跨模块抛出的异常，必须是 `HermesError` 的子类。内部实现细节
  （比如 psycopg 的 `UniqueViolation`）在存储层就地翻译，不外泄。

零依赖，只用标准库。
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """错误码集合。**新增取值需要走契约变更流程**（见 docs/02 的契约冻结规则）。"""

    INVALID_ARGUMENT = "invalid_argument"
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    INTERNAL = "internal"

    def __str__(self) -> str:
        return self.value

    @property
    def retryable(self) -> bool:
        """是否值得重试。

        只把「换个时间可能成功」的情形算进来。`RATE_LIMITED` 可重试，
        但必须配合退避——立刻重试只会加剧限流。
        """
        return self in (ErrorCode.UNAVAILABLE, ErrorCode.RATE_LIMITED, ErrorCode.INTERNAL)

    @property
    def http_status(self) -> int:
        """API 层映射。集中一处，避免每个路由自己 if/elif。"""
        return {
            ErrorCode.INVALID_ARGUMENT: 400,
            ErrorCode.FORBIDDEN: 403,
            ErrorCode.NOT_FOUND: 404,
            ErrorCode.CONFLICT: 409,
            ErrorCode.RATE_LIMITED: 429,
            ErrorCode.UNAVAILABLE: 503,
            ErrorCode.INTERNAL: 500,
        }[self]


class HermesError(Exception):
    """所有跨模块异常的基类。

    `detail` 只放**结构化、可序列化**的补充信息（比如冲突时命中的那条记录 id）。
    不要把完整堆栈或用户原始消息塞进来——它会进日志、进死信表、进 API 响应，
    隐私边界会在这里失守。
    """

    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        detail: dict[str, Any] | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.detail = detail or {}
        # 仅 RATE_LIMITED / UNAVAILABLE 有意义，worker 用它决定退避下限
        self.retry_after_s = retry_after_s

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class InvalidArgument(HermesError):
    code = ErrorCode.INVALID_ARGUMENT


class NotFound(HermesError):
    code = ErrorCode.NOT_FOUND


class Forbidden(HermesError):
    """权限不足。

    约定：权限判定 **fail-closed**。hermes_core 的 `_is_admin` 里记过一次
    事故——`except: is_admin = True`，一次配置读取失败把所有人提成了管理员。
    拿不到明确答复一律拒绝。
    """

    code = ErrorCode.FORBIDDEN


class Conflict(HermesError):
    """唯一约束冲突等。调用方通常应改为读取既有记录，而不是重试写入。"""

    code = ErrorCode.CONFLICT


class Unavailable(HermesError):
    """依赖不可用（数据库、模型 provider）。可重试。"""

    code = ErrorCode.UNAVAILABLE


class RateLimited(HermesError):
    code = ErrorCode.RATE_LIMITED

    def __init__(self, message: str, *, retry_after_s: float = 1.0, **kw: Any) -> None:
        super().__init__(message, retry_after_s=retry_after_s, **kw)
