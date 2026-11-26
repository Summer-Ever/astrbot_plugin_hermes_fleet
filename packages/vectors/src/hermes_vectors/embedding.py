"""OpenAI 兼容的 `/embeddings` 客户端。

## 为什么只做「HTTP 接口」这一种 provider

本地模型（fastembed / sentence-transformers / onnxruntime）能省 API 费，
但会给插件加一条几十 MB 的原生依赖，而安装失败最常见的地方恰好是
Windows + 非主流 Python 版本——插件在加载期直接炸，连字面记忆都一起
失去。所以 MVP 只做**零依赖**的 HTTP 路径：

    POST {base_url}/embeddings
    {"model": "...", "input": ["...", "..."]}

它同时覆盖 OpenAI、硅基流动、阿里云百炼、Ollama（`/v1`）、vLLM 等
常见服务；要接本地模型，起一个 Ollama 或 vLLM 即可，不必改插件。

## 失败语义

所有网络/协议错误统一抛 `EmbeddingError`（带可读原因），调用方只做两件事：
- 检索路径：这一次退回字面召回，不重试、不阻塞对话；
- 索引路径：把原因写进 `meta.last_error`，下一轮后台任务再试。
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Sequence

from .core import normalize

# 单次响应体的上限。正常的 3072 维 × 64 条约 1.5MB，8MB 足够，
# 同时挡住「服务端返回一页 HTML 错误页」把内存吃满的情况。
_MAX_BODY = 8 * 1024 * 1024


class EmbeddingError(RuntimeError):
    """embedding 调用失败。消息面向运维，不面向模型。"""


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key: str = ""
    dim: int = 0
    batch_size: int = 10
    timeout_s: float = 20.0

    def __post_init__(self) -> None:
        base = str(self.base_url or "").strip().rstrip("/")
        object.__setattr__(self, "base_url", base)
        object.__setattr__(self, "model", str(self.model or "").strip())
        object.__setattr__(self, "api_key", str(self.api_key or "").strip())
        try:
            object.__setattr__(self, "dim", max(0, int(self.dim)))
        except (TypeError, ValueError):
            object.__setattr__(self, "dim", 0)
        try:
            object.__setattr__(self, "batch_size", min(128, max(1, int(self.batch_size))))
        except (TypeError, ValueError):
            object.__setattr__(self, "batch_size", 10)
        try:
            object.__setattr__(self, "timeout_s", max(1.0, float(self.timeout_s)))
        except (TypeError, ValueError):
            object.__setattr__(self, "timeout_s", 20.0)

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    @property
    def endpoint(self) -> str:
        """拼出 `/embeddings`。已经带后缀的地址不再重复拼接。"""
        if not self.base_url:
            return ""
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        return self.base_url + "/embeddings"


def parse_embeddings(payload: object, *, expected: int) -> list[list[float]]:
    """解析 OpenAI 格式响应，返回**已归一化**的向量。

    抽成纯函数是为了能穷举脏响应：缺 `data`、少一条、维度不一、`index`
    乱序、把 NaN 当数字…这些都不该在线上第一次遇到。
    """
    if not isinstance(payload, dict):
        raise EmbeddingError("响应不是 JSON 对象")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise EmbeddingError("响应缺少 data 数组")
    rows: list[tuple[int, list[float]]] = []
    for pos, item in enumerate(data):
        if not isinstance(item, dict):
            raise EmbeddingError(f"data[{pos}] 不是对象")
        vec = item.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise EmbeddingError(f"data[{pos}] 缺少 embedding")
        try:
            values = [float(x) for x in vec]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(f"data[{pos}] 含非数字分量") from exc
        try:
            index = int(item.get("index", pos))
        except (TypeError, ValueError):
            index = pos
        rows.append((index, values))
    if len(rows) != expected:
        raise EmbeddingError(
            f"返回条数不符：期望 {expected}，实际 {len(rows)}"
        )
    rows.sort(key=lambda row: row[0])
    dims = {len(v) for _, v in rows}
    if len(dims) != 1:
        raise EmbeddingError(f"同一批向量维度不一致：{sorted(dims)}")
    return [normalize(v) for _, v in rows]


class EmbeddingClient:
    """把 `EmbeddingConfig` 变成可等待的批量向量化。"""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """按 `batch_size` 分片顺序调用，返回顺序与入参一致。

        顺序而不是并发：并发会在一瞬间把配额打满，触发 429 之后整轮索引
        全部失败；顺序调用的总耗时由批大小控制，反而更容易预测。
        """
        items = [str(t or "") for t in texts]
        if not items:
            return []
        out: list[list[float]] = []
        size = self.config.batch_size
        for start in range(0, len(items), size):
            chunk = items[start:start + size]
            out.extend(await asyncio.to_thread(self._embed_sync, chunk))
        return out

    async def embed_one(self, text: str) -> list[float]:
        rows = await self.embed([text])
        return rows[0] if rows else []

    def _embed_sync(self, chunk: Sequence[str]) -> list[list[float]]:
        cfg = self.config
        if not cfg.enabled:
            raise EmbeddingError("未配置 embedding 模型")
        payload: dict = {"model": cfg.model, "input": list(chunk)}
        if cfg.dim > 0:
            # OpenAI text-embedding-3 系列支持降维；不支持的供应商会忽略
            # 或报错——报错时把 vector_dim 设回 0 即可，实际维度以响应为准。
            payload["dimensions"] = cfg.dim
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        req = urllib.request.Request(
            cfg.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                raw = resp.read(_MAX_BODY + 1)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read(2000).decode("utf-8", "replace")
            except Exception:
                pass
            raise EmbeddingError(
                f"HTTP {exc.code}：{detail.strip() or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(f"连接失败：{exc.reason}") from exc
        except TimeoutError as exc:
            raise EmbeddingError(f"请求超时（{cfg.timeout_s:.0f}s）") from exc
        if len(raw) > _MAX_BODY:
            raise EmbeddingError("响应体过大，已拒绝")
        try:
            payload_obj = json.loads(raw.decode("utf-8", "replace"))
        except ValueError as exc:
            raise EmbeddingError("响应不是合法 JSON") from exc
        return parse_embeddings(payload_obj, expected=len(chunk))
