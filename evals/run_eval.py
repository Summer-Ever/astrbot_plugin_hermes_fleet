"""跑评测：变体 × 题型 × 池规模，输出可对比的报告。

用法：
    python evals/run_eval.py              # 主报告
    python evals/run_eval.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "packages" / "recall" / "src"))

import corpus  # noqa: E402
import variants  # noqa: E402
from metrics import Rec, bootstrap_ci, mcnemar, summarize  # noqa: E402

BUCKET_ORDER = ["easy", "gap", "anaphora", "temporal", "multigoal",
                "update", "distractor", "self", "agreement", "abstain"]
BUCKET_CN = {
    "easy": "字面重合",
    "gap": "词汇鸿沟",
    "anaphora": "指代省略",
    "temporal": "时间指代",
    "update": "知识更新",
    "distractor": "强干扰项",
    "multihop": "多跳组合",
    "self": "机器人自我",
    "agreement": "双方约定",
    "abstain": "应拒答",
}


def build_pool(persona: str, mode: str) -> list[corpus.Mem]:
    """候选池。

    own   : 只给这个 persona 自己的记忆，模拟「每 bot 一库」的新账号。
    mixed : 混入全部 persona，模拟「多个账号写入同一个库」或一段很长的历史。
            这个设置专门用来量化隔离失败会付多大代价。
    """
    if mode == "own":
        return corpus.memories_for(persona)
    return list(corpus.ALL_MEMORIES)


def evaluate(variant: str, *, pool_mode: str, budget: int) -> list[Rec]:
    fn = variants.VARIANTS[variant]
    out: list[Rec] = []
    for q in corpus.QUERIES:
        pool = build_pool(q.persona, pool_mode)
        got = fn(q.text, pool, budget=budget, now=corpus.NOW)
        out.append(Rec(qid=q.qid, bucket=q.bucket, persona=q.persona,
                       gold=q.gold, stale=q.stale, got=got))
    return out


def fmt_row(name: str, s, width: int = 30) -> str:
    return (f"{name:<{width}} {s.hit1:>6.1%} {s.recall_any:>7.1%} "
            f"{s.recall_frac:>7.1%} {s.mrr:>6.3f} "
            f"{s.purity:>7.1%} {s.abstain_acc:>7.1%} {s.silence_rate:>7.1%} "
            f"{s.stale_rate:>6.1%} {s.avg_items:>5.1f} {s.loss():>7.3f}")


HDR = (f"{'变体':<30} {'Hit@1':>6} {'R_any':>7} {'R_frac':>7} "
       f"{'MRR':>6} {'纯度':>7} {'拒答':>7} {'沉默':>7} {'旧值':>6} "
       f"{'条数':>5} {'损失':>7}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--budget", type=int, default=5)
    args = ap.parse_args()

    errs = corpus.self_check()
    if errs:
        print("语料自检失败：")
        for e in errs:
            print("  -", e)
        return 1

    print(f"语料：{len(corpus.ALL_MEMORIES)} 条记忆 / "
          f"{len(corpus.QUERIES)} 条查询 / 注入预算 {args.budget} 条")
    per_persona = {p: len(corpus.memories_for(p)) for p in ("a", "b", "c", "d", "e")}
    print("每 persona 记忆数：", per_persona)
    print()

    report: dict = {"budget": args.budget, "pool": {}}

    for pool_mode in ("own", "mixed"):
        print("=" * 108)
        print(f"候选池 = {pool_mode}  "
              f"({len(build_pool('a', pool_mode))} 条)")
        print("=" * 108)
        print(HDR)
        print("-" * 108)
        sums = {}
        recs_map = {}
        for name in variants.VARIANTS:
            recs = evaluate(name, pool_mode=pool_mode, budget=args.budget)
            s = summarize(name, recs)
            sums[name] = s
            recs_map[name] = recs
            print(fmt_row(name, s))
        print()

        # 与基线做配对检验
        base = recs_map["fused"]
        print("相对基线 fused 的配对检验（recall_any，McNemar 精确二项）")
        print(f"{'变体':<30} {'△pp':>7} {'仅基线命中':>10} {'仅本档命中':>10} {'p':>8}")
        print("-" * 72)
        for name in variants.VARIANTS:
            if name == "fused":
                continue
            m = mcnemar(base, recs_map[name])
            print(f"{name:<30} {m['delta_pp']:>+7.1f} {m['only_a']:>10} "
                  f"{m['only_b']:>10} {m['p']:>8.3f}")
        print()

        # 主指标的置信区间
        print("R_frac 的 95% bootstrap 置信区间（判断差异是不是噪声）")
        for name in variants.VARIANTS:
            vals = [r.recall_frac for r in recs_map[name] if not r.is_abstain]
            lo, hi = bootstrap_ci(vals)
            mean = sum(vals) / len(vals)
            print(f"  {name:<30} {mean:>6.1%}  [{lo:>6.1%}, {hi:>6.1%}]")
        print()

        # 分桶诊断：这是真正定位问题的地方
        print("分桶 R_frac（按题型看短板）")
        buckets = ["easy", "gap", "anaphora", "temporal", "update",
                   "distractor", "multihop", "self", "agreement"]
        head = f"{'变体':<30}" + "".join(f"{BUCKET_CN[b][:4]:>7}" for b in buckets)
        print(head)
        print("-" * len(head))
        for name in variants.VARIANTS:
            cells = []
            for b in buckets:
                st = sums[name].by_bucket.get(b)
                cells.append(f"{st['recall_frac']:>7.0%}" if st else f"{'-':>7}")
            print(f"{name:<30}" + "".join(cells))
        print()

        report["pool"][pool_mode] = {
            name: {
                "hit1": round(s.hit1, 4),
                "recall_any": round(s.recall_any, 4),
                "recall_all": round(s.recall_all, 4),
                "recall_frac": round(s.recall_frac, 4),
                "mrr": round(s.mrr, 4),
                "purity": round(s.purity, 4),
                "abstain_acc": round(s.abstain_acc, 4),
                "stale_rate": round(s.stale_rate, 4),
                "avg_items": round(s.avg_items, 3),
                "loss": round(s.loss(), 4),
                "f2": round(s.f2, 4),
                "n_miss": s.n_miss,
                "n_noise": s.n_noise,
                "n_stale": s.n_stale,
                "n_false": s.n_false,
                "by_bucket": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                              for k, v in s.by_bucket.items()},
            }
            for name, s in sums.items()
        }

    # 闸门阈值标定：这是唯一该调参的地方，且依据是数据不是直觉
    print("=" * 108)
    print("软闸门阈值标定（own 池，索引已扩写）")
    print("=" * 108)
    print(f"{'阈值':>6} {'拒答正确率':>10} {'非弃答 R_any':>13} {'平均注入条数':>13}")
    print("-" * 48)
    by_p = {p: corpus.memories_for(p) for p in ("A", "B", "C", "D", "E")}
    calib = []
    for th, acc, r, avg in variants.calibrate_threshold(by_p):
        print(f"{th:>6.1f} {acc:>10.1%} {r:>13.1%} {avg:>13.2f}")
        calib.append({"floor": th, "abstain_acc": round(acc, 4),
                      "recall_any": round(r, 4), "avg_items": round(avg, 3)})
    report["threshold_calibration"] = calib
    print()

    # 预算扫描：注入几条之后边际收益归零
    print("=" * 108)
    print("注入预算扫描（own 池，看边际收益什么时候归零）")
    print("=" * 108)
    print(f"{'变体':<24}" + "".join(f"{'B=' + str(b):>10}" for b in (1, 2, 3, 5, 8)))
    print("-" * 74)
    scan = {}
    for name in ("fused", "recall_v2", "recall_v2_expand"):
        row = []
        for b in (1, 2, 3, 5, 8):
            rs = evaluate(name, pool_mode="own", budget=b)
            s = summarize(name, rs)
            row.append(s.recall_frac)
        print(f"{name:<24}" + "".join(f"{v:>10.1%}" for v in row))
        scan[name] = [round(v, 4) for v in row]
    report["budget_scan"] = scan

    # 权重敏感性：结论不能依赖某一个偶然的噪声代价假设
    print()
    print("=" * 108)
    print("噪声代价敏感性（own 池）：不同假设下最优策略是否变化")
    print("=" * 108)
    print("  横轴 = 每注入一条无关记忆的代价（相对于漏一条 gold 的 1.0）")
    print(f"{'变体':<26}" + "".join(f"{'w=' + str(w):>10}" for w in (0.0, 0.1, 0.3, 0.6, 1.0)))
    print("-" * 76)
    sens = {}
    for name in variants.VARIANTS:
        rs = evaluate(name, pool_mode="own", budget=args.budget)
        s = summarize(name, rs)
        row = [s.loss(w_noise=w) for w in (0.0, 0.1, 0.3, 0.6, 1.0)]
        sens[name] = [round(v, 3) for v in row]
        best = min(row)
        cells = "".join(
            (f"{v:>9.3f}" + ("*" if v == best else " ")) for v in row)
        print(f"{name:<26}" + cells)
    print("  （* 标该权重下的最优）")
    report["noise_sensitivity"] = sens

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
