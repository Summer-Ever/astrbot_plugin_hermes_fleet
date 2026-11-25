"""评测指标：为什么不是一个数字。

只报 Recall@k 会掩盖三类致命问题，而这三类在真实聊天里比「少召回一条」严重
得多：

1. **召回旧值**。用户已经说「我换工作了」，系统还注入「他在云栈科技」。模型
   看到旧事实会照答，用户感受到的是「它记错了」，而不是「它没记住」。这类
   错误必须单独计，不能混进平均召回率里被稀释。
2. **无关也硬注入**。用户问「1+1 等于几」，系统塞三条记忆进 prompt。召回率
   上看不出问题，实际是纯噪声、烧 token、还可能带偏回答。
3. **多证据题只召回到一半**。多跳题需要两条，只召回一条时 recall_any 是满分，
   但回答照样错。所以要区分 recall_any（至少一条）和 recall_all（全部）。

因此这里分三层指标，且不给单一总分做「谁第一」的依据——先看约束（stale、
false_inject）是否达标，再看主指标。utility 只是给排序用的启发式辅助值。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Rec:
    """单题单变体的结果。"""
    qid: str
    bucket: str
    persona: str
    gold: tuple[str, ...]
    stale: tuple[str, ...]
    got: list[str]

    @property
    def is_abstain(self) -> bool:
        return not self.gold

    @property
    def hit_ids(self) -> list[str]:
        return [g for g in self.gold if g in self.got]

    @property
    def hit1(self) -> bool:
        return bool(self.got) and bool(self.gold) and self.got[0] in self.gold

    @property
    def recall_any(self) -> bool:
        return bool(self.hit_ids)

    @property
    def recall_all(self) -> bool:
        return bool(self.gold) and len(self.hit_ids) == len(self.gold)

    @property
    def recall_frac(self) -> float:
        if not self.gold:
            return 0.0
        return len(self.hit_ids) / len(self.gold)

    @property
    def precision(self) -> float:
        """注入纯度：注入的条目里有多少真的相关。

        abstain 题单独处理：注入非空时纯度为 0（全是噪声）。
        """
        if not self.got:
            return 1.0 if self.is_abstain else 0.0
        return len([g for g in self.got if g in self.gold]) / len(self.got)

    @property
    def rr(self) -> float:
        """第一个 gold 的倒数排名。没命中为 0。"""
        for i, g in enumerate(self.got, 1):
            if g in self.gold:
                return 1.0 / i
        return 0.0

    @property
    def stale_hit(self) -> bool:
        return any(s in self.got for s in self.stale)

    @property
    def false_inject(self) -> bool:
        """abstain 题却注入内容——幻觉注入。"""
        return self.is_abstain and bool(self.got)

    @property
    def chars(self) -> int:
        # 注入成本用条目数近似即可，字符数在 corpus 层算更准，这里只记条数
        return len(self.got)


@dataclass
class Summary:
    variant: str
    n_scored: int = 0      # 参与召回统计的题数（排除 abstain）
    n_abstain: int = 0
    hit1: float = 0.0
    recall_any: float = 0.0
    recall_all: float = 0.0
    recall_frac: float = 0.0
    mrr: float = 0.0
    purity: float = 0.0
    abstain_acc: float = 0.0
    stale_rate: float = 0.0
    avg_items: float = 0.0
    by_bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    # 错误计数，用于算业务损失
    n_miss: int = 0        # 漏掉的 gold 条数
    n_noise: int = 0       # 注入的无关条数
    n_stale: int = 0       # 召回旧值的题数
    n_false: int = 0       # 该拒答却注入的题数
    n_silent: int = 0      # 该答却一条都没注入的题数

    @property
    def silence_rate(self) -> float:
        """该答却完全沉默的比例。

        这个指标是给「拒答正确率」配的解药。一个什么都不注入的系统拒答
        正确率恒为 100%，看起来最安全，实际是瘫痪。两张表要一起看：拒答
        高 + 沉默低 才是真的判别力；拒答高 + 沉默高 只是无能。
        """
        return self.n_silent / self.n_scored if self.n_scored else 0.0

    @property
    def n_total(self) -> int:
        return self.n_scored + self.n_abstain

    def loss(self, *, w_miss: float = 1.0, w_noise: float = 0.3,
             w_stale: float = 3.0, w_false: float = 3.0) -> float:
        """每题平均业务损失，越低越好。

        为什么不用召回率和纯度的加权平均：那两个都是比值，会让「几乎不注入」
        天然拿到高纯度，看起来很好，实际用户感受是「它什么都不记得」。把四种
        错误折成可比的代价，才能诚实地比较两种策略。

        权重是业务判断，不是数学常数，所以显式写出来供反驳：
        - 漏一条 gold = 1：模型会说「我不记得」，用户觉得失忆，但不至于答错；
        - 多注入一条无关 = 0.3：模型通常能忽略，代价主要是 token 和注意力；
        - 召回旧值 = 3：模型会照旧事实回答，用户感知是「它记错了」，比失忆糟；
        - 该拒答却注入 = 3：模型可能顺着无关记忆编故事，这是最坏的失败模式。
        """
        if not self.n_total:
            return 0.0
        return (w_miss * self.n_miss + w_noise * self.n_noise
                + w_stale * self.n_stale + w_false * self.n_false) / self.n_total

    @property
    def f2(self) -> float:
        """召回偏向的 F 值。只作参考——它对「注入条数少」同样有偏好。"""
        p, r = self.purity, self.recall_frac
        if p + r == 0:
            return 0.0
        return 5 * p * r / (4 * p + r)


def summarize(variant: str, recs: list[Rec]) -> Summary:
    s = Summary(variant=variant)
    scored = [r for r in recs if not r.is_abstain]
    abst = [r for r in recs if r.is_abstain]
    s.n_scored = len(scored)
    s.n_abstain = len(abst)
    if scored:
        n = len(scored)
        s.hit1 = sum(r.hit1 for r in scored) / n
        s.recall_any = sum(r.recall_any for r in scored) / n
        s.recall_all = sum(r.recall_all for r in scored) / n
        s.recall_frac = sum(r.recall_frac for r in scored) / n
        s.mrr = sum(r.rr for r in scored) / n
        s.avg_items = sum(len(r.got) for r in scored) / n
        s.purity = sum(r.precision for r in scored) / n
        s.n_miss = sum(len(r.gold) - len(r.hit_ids) for r in scored)
        s.n_noise = sum(len([g for g in r.got if g not in r.gold]) for r in scored)
        s.n_stale = sum(1 for r in scored if r.stale_hit)
        s.n_silent = sum(1 for r in scored if not r.got)
        upd = [r for r in scored if r.stale]
        s.stale_rate = (sum(r.stale_hit for r in upd) / len(upd)) if upd else 0.0
    if abst:
        s.abstain_acc = sum(1 for r in abst if not r.got) / len(abst)
        s.n_false = sum(1 for r in abst if r.got)
        # 弃答题里注入的无关条目也要计入噪声，否则「什么都注入」在这项上不受罚
        s.n_noise += sum(len(r.got) for r in abst)

    buckets = sorted({r.bucket for r in recs})
    for b in buckets:
        rs = [r for r in recs if r.bucket == b]
        if b == "abstain":
            s.by_bucket[b] = {
                "n": len(rs),
                "abstain_acc": sum(1 for r in rs if not r.got) / len(rs),
                "recall_frac": 0.0,
                "purity": sum(r.precision for r in rs) / len(rs),
            }
            continue
        s.by_bucket[b] = {
            "n": len(rs),
            "hit1": sum(r.hit1 for r in rs) / len(rs),
            "recall_frac": sum(r.recall_frac for r in rs) / len(rs),
            "recall_all": sum(r.recall_all for r in rs) / len(rs),
            "purity": sum(r.precision for r in rs) / len(rs),
        }
    return s


def _binom_two_sided(k: int, n: int) -> float:
    """精确二项检验双侧 p 值，p=0.5。n 很小所以直接求和，不需要 scipy。"""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(a: list[Rec], b: list[Rec]) -> dict:
    """配对检验：同一批 query 上，两个变体的 recall_any 差异是否显著。

    只比较 recall_any 这种二分类结果，用 McNemar（精确二项版）：
    只看「一个中一个不中」的 Discordant 对，两边都对/都错的题没有区分信息。
    """
    amap = {r.qid: r for r in a if not r.is_abstain}
    bmap = {r.qid: r for r in b if not r.is_abstain}
    common = sorted(set(amap) & set(bmap))
    only_a = sum(1 for q in common if amap[q].recall_any and not bmap[q].recall_any)
    only_b = sum(1 for q in common if bmap[q].recall_any and not amap[q].recall_any)
    p = _binom_two_sided(min(only_a, only_b), only_a + only_b)
    return {
        "n": len(common),
        "only_a": only_a,
        "only_b": only_b,
        "delta_pp": (only_b - only_a) / len(common) * 100 if common else 0.0,
        "p": p,
    }


def bootstrap_ci(values: list[float], *, n_iter: int = 2000,
                 seed: int = 7) -> tuple[float, float]:
    """均值的 95% bootstrap 置信区间。

    query 只有八十几条，两个变体差 2pp 完全可能是抽样噪声。报置信区间是
    「别把噪声当改进」的唯一办法。
    """
    if not values:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_iter):
        means.append(sum(values[rnd.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_iter)]
    hi = means[int(0.975 * n_iter) - 1]
    return (lo, hi)
