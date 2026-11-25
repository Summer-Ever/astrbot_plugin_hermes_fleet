# evals —— 记忆召回评测

一套可复现的召回评测，用来回答「哪种召回最好」，而不是靠感觉判断。

## 文件

| 文件 | 作用 |
|---|---|
| `corpus.py` | 5 个 persona / 162 条记忆 / 83 条带 gold 标注的查询 |
| `variants.py` | 13 个召回变体，每个只改一件事，便于归因 |
| `metrics.py` | 三层指标 + 业务损失 + McNemar + bootstrap |
| `run_eval.py` | 主报告：变体对比、分桶、配对检验、阈值标定、预算扫描、敏感性 |
| `diagnose.py` | 单题诊断，判断是「闸门误杀」还是「排序里就没有 gold」 |

## 跑

```bash
python evals/run_eval.py
python evals/run_eval.py --json evals/report.json
python evals/diagnose.py a_q01
python evals/diagnose.py --bucket gap
```

零依赖，只用标准库。

## 语料设计的三条硬约束

1. **query 手写**，不用模板从记忆原文拼，也不让 LLM 改写原文生成——那会让
   词法召回虚高。
2. **必埋干扰项**：同实体不同值、同数字不同事、同话题不同人。
3. **必须有负样本**，且分两类：完全无关（用户没有这个人）和近似但不同
   （问孙子、实际只有孙女）。

`corpus.self_check()` 会在每次评测前校验 id 唯一、gold/stale 存在、abstain
无 gold，第一次跑就抓出一条标注错误。

## 读报告时先看这三个

- **沉默率**：该答却一条都没注入。它和「拒答正确率」必须一起看，否则会把
  瘫痪误判成判别力。
- **旧值率**：召回到已过时记忆。这是「答错」不是「没答」，必须单独计。
- **纯度 + 平均条数**：召回率高可能只是「注入满 5 条总有一条沾边」。

结论与完整分析见 `docs/06-记忆召回评测与最优方案.md`。
