"""向量数学与编码：**只用标准库，numpy 可选**。

为什么单独一层：`store.py` 负责 SQLite，`embedding.py` 负责 HTTP，这一层
只有纯函数——归一化、打包、点积、top-k。所有「排序算错」的可能都集中
在这里，于是可以在没有数据库、没有网络的机器上穷举测试。

## 为什么有 numpy 才快，没有也能用

中文记忆库常见规模是每账号数百到数千条。1024 维、2000 条的一次精确
点积在 numpy 下约 1ms，纯 Python 约 30ms——都还能接受。到了 2 万条，
纯 Python 会到 300ms，已经不该占在对话链路上。所以：

- 有 numpy：一次矩阵乘，2 万条也在几毫秒；
- 没有 numpy：只在 `scan_cap` 以内做纯 Python 点积（按最近更新时间取窗口），
  超出就返回空结果，让上层退回字面召回。

**刻意不引入 sqlite-vec / faiss 这类依赖**：它们要么是 C 扩展（安装失败率
在 Windows 上不可接受），要么把索引变成第二份需要同步的真相。MVP 阶段
「每账号一个旁路库 + 精确 KNN」已经把复杂度压在可解释的范围内。
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
from array import array
from collections.abc import Iterable, Sequence

_F32 = struct.Struct("<f")

# numpy 探测只做一次。导入本身约 30-80ms，必须在首次检索时惰性触发，
# 不能放在模块导入期——插件加载不该为了一个可选加速卡住宿主启动。
_NUMPY: object | None = None
_NUMPY_READY = False


def _numpy():
    global _NUMPY, _NUMPY_READY
    if not _NUMPY_READY:
        _NUMPY_READY = True
        try:
            import numpy  # noqa: PLC0415 — 可选依赖，必须惰性导入

            _NUMPY = numpy
        except Exception:  # pragma: no cover - 取决于部署环境
            _NUMPY = None
    return _NUMPY


def has_numpy() -> bool:
    return _numpy() is not None


def normalize(vec: Sequence[float]) -> list[float]:
    """L2 归一化。全零/非法向量返回等长零向量（调用方按 0 分处理）。"""
    values = [float(x) for x in vec]
    norm = math.sqrt(sum(x * x for x in values))
    if not norm or not math.isfinite(norm):
        return [0.0] * len(values)
    return [x / norm for x in values]


def pack(vec: Sequence[float]) -> bytes:
    """把向量压成 float32 小端字节串。

    用定长 struct 而不是 `array('f').tobytes()`：后者的字节序跟随本机，
    备份从 ARM 搬到 x86 时向量会变成一串乱码，而且不会报错——余弦相似度
    仍然「能算」，只是全错。多出来的几次 pack 在写入侧，不在检索热路径。

    小端机（x86 / ARM 的常规状态）上 `array('f')` 与 `<f` 逐字节等价，
    直接走 C 层批量转换：索引 10 万条 × 1024 维时，两者差一个数量级。
    """
    values = [float(x) for x in vec]
    if sys.byteorder == "little":
        return array("f", values).tobytes()
    return b"".join(_F32.pack(x) for x in values)


def unpack(blob: bytes) -> array:
    a = array("f")
    usable = len(blob) - (len(blob) % _F32.size)
    if usable <= 0:
        return a
    a.frombytes(blob[:usable])
    # struct 打包是小端；array 按本机字节序解释。x86/ARM 都是小端，
    # 这里显式转换只为让「从大端机恢复」这件事至少有确定行为。
    if struct.pack("<H", 1) != struct.pack("=H", 1):  # pragma: no cover
        a.byteswap()
    return a


def dim_of(blob: bytes) -> int:
    return len(blob) // _F32.size


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """两向量的余弦相似度。存储层存的是归一化向量，这里主要用于测试/基准。

    对未归一化的输入也会先归一化再算，避免「调用方忘了归一化」时得到一个
    看起来很大、实际无意义的数。
    """
    if len(a) != len(b) or not a:
        return 0.0
    na, nb = normalize(a), normalize(b)
    return float(sum(x * y for x, y in zip(na, nb)))


def content_hash(text: str) -> str:
    """记忆正文的指纹。用它判断「要不要重新向量化」。

    只做 sha1，不做归一化：内容被后台整理改过一个字就该重算，
    否则旧向量会继续代表新文本，检索结果与原文对不上。
    """
    return hashlib.sha1(str(text or "").encode("utf-8", "ignore")).hexdigest()


def top_k(query: Sequence[float], rows: Iterable[tuple[int, bytes, int]],
          *, k: int = 5, min_score: float = 0.0) -> list[tuple[int, float]]:
    """在 `(id, blob, dim)` 行上做精确最大内积。

    约定向量写入前已经归一化，所以内积即余弦相似度；查询向量也由调用方
    归一化。这里再验一次维度：模型换了之后旧向量还在库里时，维度不匹配
    的行必须跳过，而不是把半截字节当成向量算出一个假分数。
    """
    cap = max(0, int(k))
    if cap <= 0:
        return []
    q = [float(x) for x in query]
    dim = len(q)
    if dim <= 0:
        return []
    usable = [(int(i), b) for i, b, d in rows if int(d) == dim and len(b) == dim * 4]
    if not usable:
        return []

    np = _numpy()
    if np is not None:
        ids = [i for i, _ in usable]
        raw = b"".join(b for _, b in usable)
        mat = np.frombuffer(raw, dtype="<f4").reshape(len(usable), dim)
        scores = mat @ np.asarray(q, dtype="<f4")
        if cap < len(scores):
            idx = np.argpartition(-scores, cap - 1)[:cap]
        else:
            idx = np.arange(len(scores))
        order = sorted((int(i) for i in idx), key=lambda i: -float(scores[i]))
        return [
            (ids[i], float(scores[i]))
            for i in order
            if float(scores[i]) >= min_score
        ]

    scored: list[tuple[int, float]] = []
    for memory_id, blob in usable:
        vec = unpack(blob)
        score = 0.0
        for a, b in zip(vec, q):
            score += a * b
        if score >= min_score:
            scored.append((memory_id, score))
    scored.sort(key=lambda row: -row[1])
    return scored[:cap]
