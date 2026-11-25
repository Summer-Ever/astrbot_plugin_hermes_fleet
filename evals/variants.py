"""召回变体：把「可能的改进」各自实现成可对比的一档。

目的是让「什么召回最好」变成一个可测量的量，而不是一个观点。每个变体只
改一件事，其余全部相同（同一候选池、同一闸门、同一 now），这样两档之间的
差值就是那件事的净贡献。

变体分三层，对应三种不同性质的手段：

1. **换主干**（bigram_only / bm25_only / fused）
   只动排序算法。这一层能带来的提升是有限的，因为问题往往不在排序。

2. **改索引**（expand）
   写入时把记忆扩写成「原文 + 可能的问法」。检索用扩展后的文本，注入仍用
   原文。这是词法系统对抗词汇鸿沟最有效的手段，因为它把「query 与 doc 之间
   的词汇差」提前到写入时解决，而写入是异步的、不占对话延迟。

3. **改注入**（neighbor）
   命中一条后把它同批次的邻居一起注入。收益在「证据完整性」（多跳题需要
   两条），代价是纯度下降（单跳题里邻居是噪声）。

关于 aliases 的诚实说明
-----------------------
这里用受控词表模拟「写入侧让模型生成问法」，不是真实 LLM 输出。词表是
按常识设计的（看到记忆想「用户可能怎么问这件事」），但作者写词表时已经
知道 query 长什么样，所以这一档的收益是**上限估计**。真实 LLM 扩写的
效果通常略低于人工词表，但方向一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "recall" / "src"))
sys.path.insert(0, str(_ROOT / "packages" / "contracts" / "src"))
sys.path.insert(0, str(_ROOT / "evals"))

from hermes_recall.recall import (  # noqa: E402
    bm25_scores,
    gram_set,
    grams,
    norm,
    rank,
    rank_fused,
)

from corpus import Mem  # noqa: E402

# ---------------------------------------------------------------------------
# 写入侧问法扩展词表
#
# 每组：(记忆里的说法, [用户可能用的说法])
# 设计原则：只收「同一件事的不同叫法」，不收「同一类事的其他事」——
# 后者是概念扩展，容易把无关条目拉进来。
# ---------------------------------------------------------------------------
_EXPANSIONS: list[tuple[str, list[str]]] = [
    ("只喝美式", ["咖啡 口味 加糖 加奶 拿铁"]),
    ("讨厌开长会", ["会议 开会 烦 冗长 效率"]),
    ("深色主题", ["主题 界面 配色 夜间模式 刺眼"]),
    ("吃不了香菜", ["忌口 不吃 挑食 配菜 外卖备注"]),
    ("喜欢简洁直接", ["风格 回复 啰嗦 简练 字数"]),
    ("先给结论", ["回答方式 结构 结论先行 理由"]),
    ("叫名字就行", ["称呼 别叫 客气 头衔"]),
    ("带上语言标记", ["代码 格式 复制 高亮"]),
    ("自学 Rust", ["学习 编程语言 入门 进度"]),
    ("16G 内存", ["电脑 配置 笔记本 硬件 卡"]),
    ("从 VSCode 换成了 Cursor", ["编辑器 IDE 写代码 工具 补全"]),
    ("电池健康只剩 79", ["手机 续航 掉电 换电池"]),
    ("换一份工作", ["职业规划 跳槽 求职 面试 离职"]),
    ("带小雅去日本", ["旅游 出行 度假 签证 行程"]),
    ("迁移到 Astro", ["博客 建站 静态站点 框架"]),
    ("提醒他喝一杯水", ["提醒 定时 喝水 习惯"]),
    ("SSL 证书", ["证书 到期 续期 https 域名"]),
    ("test.txt", ["文件 桌面 创建 写入 测试"]),
    ("快速排序", ["算法 排序 代码 实现 分区"]),
    ("轻度脂肪肝", ["体检 健康 指标 喝酒 运动"]),
    ("团子吐了毛球", ["猫 宠物 生病 医院 呕吐"]),
    ("对花生严重过敏", ["过敏 忌口 花生 疹子 安全"]),
    ("芒果过敏", ["过敏 水果 嘴周 发痒"]),
    ("甲状腺结节", ["体检 复查 结节 良性 内分泌"]),
    ("攒钱买第二套房", ["买房 首付 存款 房款 目标"]),
    ("想让朵朵学钢琴", ["兴趣班 培训 乐器 学费"]),
    ("月底结账", ["工作 加班 会计 忙 报税"]),
    ("口味清淡", ["做饭 吃辣 饮食 口味"]),
    ("睡眠浅", ["失眠 睡眠 吵醒 门锁 安静"]),
    ("人民公园喂鸽子", ["周末 带娃 公园 遛娃 冰粉"]),
    ("高血压", ["血压 体检 高压 头晕 慢性病"]),
    ("苯磺酸氨氯地平", ["降压药 吃药 药片 剂量"]),
    ("阿司匹林", ["吃药 药片 饭后 抗凝"]),
    ("左膝盖有骨刺", ["膝盖 腿疼 关节炎 爬楼 走路"]),
    ("滑倒过一次", ["摔跤 骨折 尾椎 冬天 意外"]),
    ("老年机", ["手机 不会用 字大 打字 语音"]),
    ("下象棋", ["爱好 娱乐 棋 中炮 凉亭"]),
    ("君子兰", ["养花 盆栽 阳台 植物"]),
    ("右耳听力下降", ["耳朵 听不清 助听器 音量 听力"]),
    ("白内障手术", ["眼睛 手术 视力 复查 住院"]),
    ("天使轮已经close", ["融资 估值 投资 金额 轮次"]),
    ("长期失眠", ["睡眠 睡不着 白噪音 熬夜"]),
    ("浅表性胃炎", ["胃疼 胃病 压力 身体"]),
    ("耶加雪菲", ["咖啡 手冲 口味 豆子"]),
    ("喝不了高度酒", ["喝酒 应酬 酒量 红酒 脸红"]),
    ("喜欢看表格和要点", ["格式 排版 结构化 要点 文字"]),
    ("每周一早上给她一份上周待办回顾", ["周报 待办 提醒 复盘 周一"]),
    ("会议纪要里必须单列", ["纪要 待办 负责人 会议 格式"]),
    ("海外仓模式", ["物流 发货 仓储 跨境 头程"]),
    ("招一个懂 Google 广告投放的人", ["招聘 招人 投放 预算 薪资"]),
    ("进欧洲市场", ["出海 扩张 德国 VAT 市场"]),
    ("课程设计", ["项目 作业 开发 答辩"]),
    ("考研", ["研究生 考试 408 专业课 复习"]),
    ("家教兼职", ["兼职 挣钱 补课 时薪"]),
    ("无畏契约", ["游戏 打枪 段位 开黑"]),
    ("柯基", ["狗 宠物 老家 豆豆"]),
    ("肠胃炎", ["狗 生病 医院 打针 不吃东西"]),
    ("绩点", ["成绩 GPA 排名 保研"]),
    ("六级", ["英语 考试 分数 四级"]),
    ("熬夜", ["作息 睡觉 晚睡 咖啡 失眠"]),
]


def _build_alias_index(mem: Mem) -> str:
    """生成「索引文本」= 原文 + 命中的扩写词。

    只在原文里出现锚点时才追加，避免给无关记忆塞噪声。追加内容是**用户
    视角的说法**，不是原文的复述——这是它有价值的原因：它补的是 query 侧
    可能出现的词，而原文里没有这些词。
    """
    extra: list[str] = []
    for anchor, alts in _EXPANSIONS:
        if anchor in mem.text:
            extra.extend(alts)
    if not extra:
        return mem.text
    return mem.text + " 。相关说法：" + " ".join(dict.fromkeys(extra))


def _docs(mems: list[Mem], *, expanded: bool,
          drop_superseded: bool = False) -> list[dict]:
    docs = []
    for m in mems:
        # 知识更新必须在**候选集构造**这一层处理：已被取代的记忆根本不该
        # 进入排序。靠时间衰减压不住——用户问「我现在用哪个编辑器」时，
        # 「一直用 VSCode」这条字面更贴合，衰减只会让它稍降一点，仍然胜出。
        if drop_superseded and m.superseded_by is not None:
            continue
        d = {
            "id": m.id,
            "content": m.text,          # 注入用原文
            "idx": _build_alias_index(m) if expanded else m.text,  # 检索用
            "ts": m.ts,
            "kind": m.kind,
            "hits": 0,
            "importance": 0.0,
            "_ref": m,
        }
        docs.append(d)
    return docs


def _ids(rows: list[dict]) -> list[str]:
    return [str(r["id"]) for r in rows]


# ---------------------------------------------------------------------------
# 变体实现
# ---------------------------------------------------------------------------

def v_bigram_only(query: str, mems: list[Mem], *, budget: int, now: float) -> list[str]:
    """只走 A 路（bigram + IDF 覆盖率打分），保留原来的硬门槛。"""
    rows = rank(query, _docs(mems, expanded=False), k=budget, now=now)
    return _ids(rows)


def v_bm25_only(query: str, mems: list[Mem], *, budget: int, now: float) -> list[str]:
    """只走 B 路。BM25 没有绝对阈值，按分数降序取前 budget，再用 0 分截断。"""
    docs = _docs(mems, expanded=False)
    scores = bm25_scores(query, docs, mixed=False)
    pairs = sorted(zip(scores, docs), key=lambda x: -x[0])
    out = [d["id"] for s, d in pairs[:budget] if s > 0]
    return out


def v_bm25_expand(query: str, mems: list[Mem], *, budget: int, now: float) -> list[str]:
    """只走 B 路，但索引文本带问法扩展。"""
    docs = _docs(mems, expanded=True)
    scores = bm25_scores(query, docs, text_key="idx", mixed=False)
    pairs = sorted(zip(scores, docs), key=lambda x: -x[0])
    return [d["id"] for s, d in pairs[:budget] if s > 0]


# ---------------------------------------------------------------------------
# 混合粒度 BM25
#
# 现状的 term 只有字符二元组，要求「两个字严格相邻」。中文自然语言里这个
# 条件极其脆弱：「我家猫叫什么」切出 家猫/猫叫，而记忆写「名字叫团子」，
# 二元组是 名字/字叫/叫团——只差一个字位置，交集为空，整条记忆判 0 分。
# 实测「我叫什么名字」对「用户叫陈屿」也是同样归零。
#
# 修法不是再加一路召回，而是把 term 粒度放宽成「单字 + 双字」混合，让 BM25
# 自带的 IDF 去决定谁重要：单字 df 高、idf 自然低，虚词再单独过滤掉。这样
# 不需要新增任何权重参数——加参数是调参的下坡路，IDF 是确定的统计量。
# ---------------------------------------------------------------------------
_STOP = set("的了嘛吗呢吧啊呀哦嗯我你您他她它们是在有和就都也还这那个什么怎"
            "请帮把被给跟对从而且或之其与及呗哈呃诶")


def _mixed_terms(text: str) -> list[tuple[str, bool]]:
    """返回 (term, 是否单字)。单字滤掉虚词，双字照旧全留。"""
    s = norm(text)
    uni = [c for c in dict.fromkeys(s) if c not in _STOP]
    bi = [g for g in dict.fromkeys(grams(text)) if len(g) == 2]
    return [(g, False) for g in bi] + [(c, True) for c in uni]


def _bm25_mixed(query: str, docs: list[dict], text_key: str = "content") -> list[float]:
    terms = _mixed_terms(query)
    if not terms or not docs:
        return [0.0] * len(docs)
    normed = [norm(d.get(text_key, "")) for d in docs]
    gsets = [gram_set(t, normalized=True) for t in normed]
    lens = [len(t) or 1 for t in normed]
    avgdl = sum(lens) / len(lens)
    n = len(docs)

    import math
    idf: dict[str, float] = {}
    for t, single in terms:
        df = sum(1 for i, s in enumerate(normed)
                 if (t in s if single else t in gsets[i]))
        # 单字几乎总会出现（df 高 → idf 低），这是自然的降权，不需要手调系数。
        idf[t] = max(0.05, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    k1, b = 1.2, 0.75
    out: list[float] = []
    for i, (s, dl) in enumerate(zip(normed, lens)):
        sc = 0.0
        for t, single in terms:
            if not (t in s if single else t in gsets[i]):
                continue
            tf = s.count(t)
            if not tf:
                continue
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            sc += idf[t] * (tf * (k1 + 1)) / denom
        out.append(sc)
    return out


def v_recall_v2(query: str, mems: list[Mem], *, budget: int, now: float) -> list[str]:
    """直接调生产实现 `recall.rank_fused` 的新默认档（mode="mixed"）。

    与 `fused` 用的是同一个函数、同一份 docs，唯一差别是 mode。所以这一档的
    分数就是「改完 recall.py 之后生产环境的真实水平」，不是评测里的模拟。
    """
    # drop_superseded=True 对应生产路径：storage_sqlite.search_candidates
    # 的 SQL 里带 `merged_into=0`，已被取代的记忆根本不进候选池。
    rows = rank_fused(query, _docs(mems, expanded=False, drop_superseded=True),
                      k=budget, now=now, gate=True)
    return _ids(rows)


def v_recall_v2_expand(query: str, mems: list[Mem], *, budget: int,
                       now: float) -> list[str]:
    """同上，但索引文本带问法扩写（生产侧需要写入 worker 配合）。"""
    rows = rank_fused(query, _docs(mems, expanded=True, drop_superseded=True),
                      k=budget, now=now, gate=True, text_key="idx")
    return _ids(rows)


def v_bm25_mixed(query: str, mems: list[Mem], *, budget: int, now: float) -> list[str]:
    """单字 + 双字混合 BM25，不做扩写。单独看「改粒度」值多少。"""
    docs = _docs(mems, expanded=False)
    scores = _bm25_mixed(query, docs)
    pairs = sorted(zip(scores, docs), key=lambda x: -x[0])
    return [d["id"] for s, d in pairs[:budget] if s > 0]


def v_bm25_mixed_expand(query: str, mems: list[Mem], *, budget: int,
                        now: float) -> list[str]:
    docs = _docs(mems, expanded=True)
    scores = _bm25_mixed(query, docs, text_key="idx")
    pairs = sorted(zip(scores, docs), key=lambda x: -x[0])
    return [d["id"] for s, d in pairs[:budget] if s > 0]


# 软闸门分数下限。这个值只在 abstain 题上标定——「库里没有答案时最高分能到
# 多少」是可以直接测出来的，不需要猜。见 calibrate_threshold()。
#
# 4.0 对应「索引已扩写」的尺度（R 96.2% / 纯度 50.2% / 拒答 50% / 沉默 2.5%）。
# 这个值依赖 tokenizer 与扩写策略，换了必须重新标定，不能照抄；
# 生产实现 recall.py 的 SOFT_GATE_FLOOR 是同一个数，两处要一起改。
MIN_SCORE = 4.0


def v_hybrid_best(query: str, mems: list[Mem], *, budget: int, now: float,
                  score_floor: float = MIN_SCORE,
                  drop_superseded: bool = True) -> list[str]:
    """推荐的组合：混合粒度 + 写入侧扩写 + 状态过滤 + 软闸门。

    四件事各解决一个独立问题，缺一个就会在对应题型上塌掉：

    - 混合粒度 → 「猫叫」能匹配「名字叫团子」。这一项单独值 +17pp，是最大的
      杠杆，因为它解决的是「中文自然语言里两字严格相邻几乎不可能」这个
      结构性问题，不是调参能绕过的。
    - 扩写索引 → 「名字」能匹配「叫陈屿」。解决概念鸿沟，单独值 +10pp。
    - 状态过滤 → 已被取代的记忆不进候选池。这是**唯一**能根治「召回旧值」
      的手段；时间衰减只能让它稍降一点，字面更贴合的旧值仍会胜出。
    - 软闸门 → 用绝对分数否决，而不是覆盖率。覆盖率的分母是问句长度，
      问句一长就必然误杀（这是现有闸门最大的漏召回来源）。

    刻意不做邻域扩展：实测把纯度从 55% 打到 21%，而单跳题占真实流量绝大
    多数，为多跳题付这个代价不划算。
    """
    docs = _docs(mems, expanded=True, drop_superseded=drop_superseded)
    scores = _bm25_mixed(query, docs, text_key="idx")
    pairs = sorted(zip(scores, docs), key=lambda x: -x[0])
    if not pairs or pairs[0][0] < score_floor:
        return []
    return [d["id"] for s, d in pairs[:budget] if s >= score_floor * 0.55]


def calibrate_threshold(mems_by_persona: dict[str, list[Mem]]
                        ) -> list[tuple[float, float, float, float]]:
    """在 abstain 题上扫阈值：(阈值, 拒答正确率, 非弃答召回率, 平均注入条数)。

    这是闸门唯一该调参的地方，而且调的依据是数据而不是直觉：阈值越高，
    拒答越准，但该答的也开始答不出。取哪个点由业务决定——聊天助手宁可
    多注入一点（用户能忽略噪声），还是宁可漏一点（用户觉得你没记住）。
    """
    import corpus as _c
    docs_cache = {p: _docs(m, expanded=True) for p, m in mems_by_persona.items()}
    rows: list[tuple[float, float, float, float]] = []
    for th in (0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0):
        abst_ok = abst_n = 0
        hit = tot = 0
        items = 0
        for q in _c.QUERIES:
            docs = docs_cache[q.persona]
            sc = _bm25_mixed(q.text, docs, text_key="idx")
            pairs = sorted(zip(sc, docs), key=lambda x: -x[0])
            got: list[str] = []
            if pairs and pairs[0][0] >= th:
                got = [d["id"] for s, d in pairs[:5] if s >= th * 0.55]
            items += len(got)
            if not q.gold:
                abst_n += 1
                abst_ok += 1 if not got else 0
            else:
                tot += 1
                hit += 1 if any(g in got for g in q.gold) else 0
        rows.append((th, abst_ok / abst_n if abst_n else 0.0,
                     hit / tot if tot else 0.0,
                     items / len(_c.QUERIES)))
    return rows


def v_fused(query: str, mems: list[Mem], *, budget: int, now: float) -> list[str]:
    """v0.2.x 的历史基线：双路 + RRF 选池 + 池内精排 + 覆盖率闸门。

    显式钉死 mode="rrf"：recall.py 的默认档已改成 mixed，如果这里跟着默认
    走，基线会自己变强，整张对比表就失去参照了。
    """
    rows = rank_fused(query, _docs(mems, expanded=False), k=budget,
                      now=now, gate=True, mode="rrf")
    return _ids(rows)


def v_fused_expand(query: str, mems: list[Mem], *, budget: int, now: float) -> list[str]:
    """索引换成扩展文本，注入仍是原文。"""
    rows = rank_fused(query, _docs(mems, expanded=True), k=budget,
                      now=now, gate=True, text_key="idx", mode="rrf")
    return _ids(rows)


def v_fused_nogate(query: str, mems: list[Mem], *, budget: int, now: float) -> list[str]:
    """只关掉闸门，排序完全不变。

    用来分离两个因素：如果这一档比 fused 高很多，说明短板在闸门而不是融合
    权重；如果仍然低，说明融合排序本身有问题。诊断必须能归因，否则只能瞎调。
    """
    rows = rank_fused(query, _docs(mems, expanded=False), k=budget,
                      now=now, gate=False, mode="rrf")
    return _ids(rows)


def v_fused_expand_nogate(query: str, mems: list[Mem], *, budget: int,
                          now: float) -> list[str]:
    rows = rank_fused(query, _docs(mems, expanded=True), k=budget,
                      now=now, gate=False, text_key="idx", mode="rrf")
    return _ids(rows)


def v_fused_neighbor(query: str, mems: list[Mem], *, budget: int, now: float) -> list[str]:
    """命中后带邻居：同 session 的其它记忆按时间就近补进预算。

    只在预算还有空位时补，且不挤掉命中本体——所以它不能提高单跳题的召回，
    只能提高多跳题的证据完整性。这正是要测的东西。
    """
    rows = rank_fused(query, _docs(mems, expanded=False), k=budget,
                      now=now, gate=True, mode="rrf")
    hits = _ids(rows)
    if not hits:
        return []
    by_id = {m.id: m for m in mems}
    picked = list(hits)
    seen = set(hits)
    # 按命中顺序展开邻居，保证靠前的命中优先获得补充机会
    for hid in hits:
        if len(picked) >= budget:
            break
        anchor = by_id.get(hid)
        if anchor is None:
            continue
        # 同批次 + 时间最近的两条
        sibs = [m for m in mems if m.sess == anchor.sess and m.id not in seen]
        sibs.sort(key=lambda m: abs(m.ts - anchor.ts))
        for m in sibs[:2]:
            if len(picked) >= budget:
                break
            picked.append(m.id)
            seen.add(m.id)
    return picked[:budget]


def v_fused_expand_neighbor(query: str, mems: list[Mem], *, budget: int,
                            now: float) -> list[str]:
    rows = rank_fused(query, _docs(mems, expanded=True), k=budget,
                      now=now, gate=True, text_key="idx", mode="rrf")
    hits = _ids(rows)
    if not hits:
        return []
    by_id = {m.id: m for m in mems}
    picked = list(hits)
    seen = set(hits)
    for hid in hits:
        if len(picked) >= budget:
            break
        anchor = by_id.get(hid)
        if anchor is None:
            continue
        sibs = [m for m in mems if m.sess == anchor.sess and m.id not in seen]
        sibs.sort(key=lambda m: abs(m.ts - anchor.ts))
        for m in sibs[:2]:
            if len(picked) >= budget:
                break
            picked.append(m.id)
            seen.add(m.id)
    return picked[:budget]


def v_fused_expand_neighbor_slot(query: str, mems: list[Mem], *, budget: int,
                                 now: float) -> list[str]:
    """再加「常驻槽位层」：profile / agreement 直接占前几个位置。

    身份、长期约定这类信息每轮都需要，检索反而引入不确定性。把它们从
    「检索任务」里摘出来常驻，剩下的预算才给事件和事实。这会让 easy 题
    几乎必然命中，代价是吃掉预算、并在无关对话里增加纯噪声。
    """
    pinned = [m.id for m in mems
              if m.kind in ("profile", "agreement") and m.superseded_by is None]
    rest_budget = max(0, budget - len(pinned))
    rest: list[str] = []
    if rest_budget:
        rest = v_fused_expand_neighbor(query, mems, budget=rest_budget, now=now)
    out = list(pinned)
    for i in rest:
        if i not in out:
            out.append(i)
    return out[:budget]


VARIANTS = {
    "bigram_only": v_bigram_only,
    "bm25_only": v_bm25_only,
    "bm25_mixed": v_bm25_mixed,
    "bm25_expand": v_bm25_expand,
    "bm25_mixed_expand": v_bm25_mixed_expand,
    "fused": v_fused,
    "fused_nogate": v_fused_nogate,
    "fused_expand": v_fused_expand,
    "fused_expand_nogate": v_fused_expand_nogate,
    "fused_neighbor": v_fused_neighbor,
    "fused_expand_neighbor": v_fused_expand_neighbor,
    "fused_expand_neighbor_slot": v_fused_expand_neighbor_slot,
    "hybrid_best": v_hybrid_best,
    "recall_v2": v_recall_v2,
    "recall_v2_expand": v_recall_v2_expand,
}
