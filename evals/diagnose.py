"""单题诊断：看清「为什么这条没召回」。

评测分数只告诉你哪里差，不告诉你为什么差。这个脚本把单题的中间量全部
打出来：query 切成什么检索单元、闸门算出的命中数与覆盖率、两路各自的
候选与打分、最后注入哪几条。

用法：
    python evals/diagnose.py a_q01
    python evals/diagnose.py a_q01 --bucket easy   # 打印该桶所有失败题
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "packages" / "recall" / "src"))
sys.path.insert(0, str(_HERE.parent / "packages" / "contracts" / "src"))

import corpus  # noqa: E402
import variants  # noqa: E402
from hermes_recall import recall as R  # noqa: E402


def explain(q: corpus.Query) -> None:
    pool = corpus.memories_for(q.persona)
    docs = variants._docs(pool, expanded=False)
    qc = R.content_grams(q.text)
    qg = R.grams(q.text)
    units = R._query_unit_count(q.text, set(qc))

    print(f"\n{'=' * 78}\n{q.qid} [{q.bucket}] {q.text}")
    if q.note:
        print(f"注: {q.note}")
    print(f"gold={q.gold}  stale={q.stale}")
    print(f"检索单元数 units={units}  实词 bigram={qc}")

    # 逐条算闸门中间量
    print(f"\n{'记忆':<18} {'类型':<11} {'hit_n':>5} {'cov':>6} {'闸门':>4} {'bm25':>7}")
    print("-" * 78)
    scores = R.bm25_scores(q.text, docs)
    rows = []
    for d, sc in zip(docs, scores):
        gset = R.gram_set(R.norm(d["content"]), normalized=True)
        hit_n = len(set(qc) & gset)
        cov = hit_n / len(qc) if qc else 0.0
        ok = R.gate_ok(units, hit_n, cov)
        rows.append((d, sc, hit_n, cov, ok, gset))
    for d, sc, hit_n, cov, ok, _ in sorted(rows, key=lambda x: -x[1])[:6]:
        mark = "*" if d["id"] in q.gold else " "
        print(f"{mark}{d['id']:<17} {d['kind']:<11} {hit_n:>5} {cov:>6.2f} "
              f"{'过' if ok else '拒':>4} {sc:>7.2f}")

    out = variants.v_fused(q.text, pool, budget=5, now=corpus.NOW)
    got_gate = [g for g in out if g in q.gold]
    out_ng = variants.v_fused_nogate(q.text, pool, budget=5, now=corpus.NOW)
    got_ng = [g for g in out_ng if g in q.gold]

    print(f"\n  fused        -> {out}   命中 {got_gate}")
    print(f"  fused_nogate -> {out_ng}   命中 {got_ng}")
    verdict = ""
    if not got_gate and got_ng:
        verdict = "闸门误杀：排序里有 gold，被闸门砍掉"
    elif got_gate and not got_ng:
        verdict = "闸门反而帮忙：去掉后被噪声挤走"
    elif not got_gate and not got_ng:
        verdict = "召回失败：排序里就没有 gold"
    else:
        verdict = "两档都命中"
    print(f"  判定: {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("qid", nargs="?", default="")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--variant", default="fused")
    args = ap.parse_args()

    errs = corpus.self_check()
    if errs:
        for e in errs:
            print("语料错误:", e)
        return 1

    if args.qid:
        qs = [q for q in corpus.QUERIES if q.qid == args.qid]
    elif args.bucket:
        qs = [q for q in corpus.QUERIES if q.bucket == args.bucket]
    else:
        print("给一个 qid 或 --bucket")
        return 1

    fn = variants.VARIANTS[args.variant]
    for q in qs:
        got = fn(q.text, corpus.memories_for(q.persona), budget=5, now=corpus.NOW)
        if args.bucket and q.gold and all(g in got for g in q.gold):
            continue  # 看失败题
        explain(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
