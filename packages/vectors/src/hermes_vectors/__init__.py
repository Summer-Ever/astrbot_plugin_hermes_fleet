"""hermes_vectors —— 记忆的向量旁路索引（MVP）。

三块拼起来即可用，彼此可独立测试：

    core.py      归一化 / float32 编码 / 精确内积 top-k（numpy 可选加速）
    store.py     每账号一个 `vectors.db`，只存 id → 向量，不是真相
    embedding.py OpenAI 兼容 `/embeddings` 的批量客户端
    batching.py  查询向量化的有界队列 + 合批 + 超时降级

设计边界（与 `docs/06` 的评测保持一致）：

- 向量库是**旁路**，`memory.db` 是唯一真相；损坏/换模型可直接重建。
- 只索引**归档记忆**，不索引原始对话、不索引常驻词条。
- 只接受文本；图片/视频/文档从不入库（现状只落 `message_str`），
  因此也不会出现在向量库里。
"""

from __future__ import annotations

from .batching import BatchingEmbedder, EmbeddingQueueFull, EmbeddingTimeout
from .core import (
    content_hash,
    cosine,
    dim_of,
    has_numpy,
    normalize,
    pack,
    top_k,
    unpack,
)
from .embedding import (
    EmbeddingClient,
    EmbeddingConfig,
    EmbeddingError,
    parse_embeddings,
)
from .store import VECTOR_DB_FILE, SqliteVectorStore, VectorRow

__all__ = [
    "BatchingEmbedder",
    "EmbeddingQueueFull",
    "EmbeddingTimeout",
    "content_hash",
    "cosine",
    "dim_of",
    "has_numpy",
    "normalize",
    "pack",
    "top_k",
    "unpack",
    "EmbeddingClient",
    "EmbeddingConfig",
    "EmbeddingError",
    "parse_embeddings",
    "VECTOR_DB_FILE",
    "SqliteVectorStore",
    "VectorRow",
]
