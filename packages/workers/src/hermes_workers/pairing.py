"""候选配对：找出「看起来在说同一件事」的记忆，**不做任何合并决定**。

## 为什么这里自己实现一遍相似度

`hermes_recall` 里已经有 bigram 与 Jaccard。但 workers 的依赖纪律是
「只吃契约，不 import 任何领域包」（见 `owners.json` 的 change_policy）。
重复的代价是 40 行纯函数；不重复的代价是编排层绑死某个检索实现——
将来换向量检索时，整理逻辑会被一起拖下水。

## 配对是召回，不是判定

阈值故意偏低。漏配对的后果是重复记忆永久堆积、互相挤占检索名额，
而且**不会报错**；多配对的后果只是多花一次模型调用，模型会说「保持分开」。
两个方向的代价不对称，所以宁可多配。

真正决定合并的是模型，这里只负责把候选摆到它面前。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# 默认相似度阈值。实测标定（本机，见 workers 的用例）：
#
#   0.813  「喜欢喝乌龙茶，尤其偏爱铁观音」↔「喜欢喝乌龙茶」   同一条
#   0.815  「对花生过敏，点外卖要避开」↔「对花生过敏」        同一条
#   0.526  「在做 AstrBot 的记忆插件」↔「在做 AstrBot 插件」 同一条
#   0.444  「喜欢喝乌龙茶」↔「爱喝乌龙茶」                    同一条 ← 必须进候选
#   0.444  「喜欢喝乌龙茶」↔「喜欢喝龙井」                    两件事 ← 会混进来
#   0.077  「住在杭州」↔「养了两只猫」                        无关
#
# 0.444 那一对说明共享前缀（「用户喜欢喝」）会显著抬高 Jaccard，使得
# 「同一件事换动词」与「不同宾语」得分相同——纯字面无法区分这两者。
# 这是本方案的天花板，也是这里只做召回、由模型判断的原因：
# 阈值取 0.40，把真实重复纳入，让模型去否掉「乌龙茶 vs 龙井」。
# 取 0.45 以上会把「爱喝乌龙茶」这种明显重复漏掉，而漏掉是静默的。
DEFAULT_THRESHOLD = 0.40

# 一个候选组最多几条。组太大时模型要在里面找等价关系，
# 容易把「都有乌龙茶」当成「说的是同一件事」。
MAX_GROUP_SIZE = 6

_KEEP = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’()（）\[\]【】<>《》~`\-—_/\\|]+")


def normalize(text: str) -> str:
    """去掉标点空白并转小写。

    中文记忆的「重复」大量来自标点差异（`用户喜欢喝乌龙茶，尤其铁观音` 与
    `用户喜欢喝乌龙茶 尤其铁观音`）。不归一化会让这类配对全部漏掉。
    """
    return _KEEP.sub("", str(text or "")).lower()


def bigrams(text: str) -> set[str]:
    s = normalize(text)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def similarity(a: str, b: str) -> float:
    """bigram Jaccard，并对「包含」关系额外加分。

    包含关系单独处理的原因：`用户喜欢喝乌龙茶` 完整出现在
    `用户喜欢喝乌龙茶，尤其偏爱铁观音` 里，但短句的 bigram 只占长句的一部分，
    Jaccard 会低到 0.3 以下而漏配。而这类「短句被长句吸收」正是
    整理最该处理的形态（早期总结出的粗略事实，被后来的细节覆盖）。
    """
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(short) >= 4 and short in long_:
        # 越接近全句被包含，越可信
        return 0.6 + 0.4 * (len(short) / len(long_))
    ga, gb = bigrams(na), bigrams(nb)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def build_groups(
    contents: Sequence[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_group_size: int = MAX_GROUP_SIZE,
) -> list[list[int]]:
    """把相似的条目聚成若干组，返回下标列表。

    用并查集而不是「按第一个元素贪心分组」：后者一旦遇到
    A~B、B~C、A!~C 的情况，结果取决于遍历顺序，同样的库跑两次分组不同，
    排查时会以为是模型不稳定。
    """
    n = len(contents)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径压缩
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    grams = [bigrams(c) for c in contents]
    for i in range(n):
        for j in range(i + 1, n):
            # 先做零成本的集合交集判断，避免对明显无关的句子算包含关系
            if not grams[i] & grams[j]:
                continue
            if similarity(contents[i], contents[j]) >= threshold:
                union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)

    groups = [
        g for g in buckets.values()
        if 2 <= len(g) <= max_group_size
    ]
    # 大的组先处理：它们通常是最该整理的重复堆
    groups.sort(key=len, reverse=True)
    return groups
