"""后台模型链的纯逻辑：解析配置、校验人选、整理成页面能直接渲染的形状。

为什么单独一个模块：这条链决定「记忆是谁在总结」。配错了不会报错，只会
表现为「记忆越来越水」或者「后台静默不动」——都是最难归因的一类故障。
所以解析与校验必须能在没有 AstrBot、没有网络的机器上被单测覆盖。

## 与主应用全局配置的关系

历史实现里链为空就「跟随 AstrBot 当前对话模型」。这在小规模部署下省事，
但它意味着：**主应用里换一次默认模型，长期记忆的归纳风格会静默漂移**，
而且事后无法归因是哪次切换让记忆变了味。

所以现在的规则是：

- `llm_provider_ids` 有值 → 完全按它来，主应用换模型不影响这里（`explicit`）。
- 只有旧的 `llm_provider_id` 有值 → 按那一个来（`legacy`）。
- 都没有 → 仍然跟随当前对话模型（`current`），但**页面会明确标警告**，
  因为这是「耦合」状态，不是推荐状态。

`__current__` 是一个显式写法：允许用户主动把「跟随当前」放进链里当主模型，
后面还能挂降级。它和「没配置」的区别是：配置了就是主动选择，页面不再警告。
"""

from __future__ import annotations

# 「跟随 AstrBot 主应用当前对话模型」。为什么用这么难看的字面量：
# 它要能安全地存进配置文件、还能与真实的 provider id 区分开，
# 而 provider id 形如 `阿里云/qwen3.8-flash`，不会撞上双下划线。
CURRENT = "__current__"

# 降级链最多挂几个。不是技术限制：四个以上意味着「这条链几乎每轮都在换」，
# 真正该做的是把坏模型挑出去，而不是继续往后加。
MAX_FALLBACKS = 4

SOURCE_LABELS = {
    "explicit": "已显式指定（与主应用解耦）",
    "legacy": "按旧的单值配置（建议改成链）",
    "current": "跟随主应用当前对话模型（未解耦）",
}


def canonical(pid: object) -> str:
    """把各种写法的「当前模型」统一成 `__current__`。

    手写配置时 `current`、`__current`、空串都会出现，全部收编——
    否则同一个意思会变成三种不同的链节点，去重也就失效了。
    """
    s = str(pid if pid is not None else "").strip()
    if s.lower() in ("", "current", "__current", "__current__"):
        return CURRENT
    return s


def parse(raw: object) -> list[str]:
    """逗号/换行分隔的配置 → 规范化的 id 列表（去重且保序）。

    空的片段直接丢掉，**不能**交给 `canonical` 处理：`canonical("")` 是
    「跟随当前模型」的意思，但 `"a/1,,"` 里那两个空片段不是用户的选择。
    早期实现漏了这一步，结果是空配置解析出 `[__current__]`，于是
    「没配置」与「显式选了跟随当前」再也分不开。
    """
    text = str(raw if raw is not None else "")
    parts = text.replace("\n", ",").replace("，", ",").split(",")
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not str(part).strip():
            continue
        pid = canonical(part)
        if pid in seen:
            # 同一个模型配两遍：降级到它没有意义，还会让冷却表互相污染。
            continue
        seen.add(pid)
        out.append(pid)
    return out


def chain_from_config(raw_chain: object, raw_single: object = "") -> list[str]:
    """配置 → 实际使用的链。**永不返回空**（空会退化成跟随当前模型）。"""
    ids = parse(raw_chain)
    if not ids:
        single = parse(raw_single)
        if single:
            ids = single[:1]
    return ids or [CURRENT]


def source_of(raw_chain: object, raw_single: object = "") -> str:
    if parse(raw_chain):
        return "explicit"
    if parse(raw_single):
        return "legacy"
    return "current"


def describe(ids: list[str]) -> dict:
    """链 → 页面要的两段：主模型 + 降级顺序。"""
    ids = [canonical(x) for x in (ids or [])] or [CURRENT]
    return {
        "primary": ids[0],
        "fallbacks": ids[1:],
        "follows_current": ids[0] == CURRENT,
    }


def label(pid: str) -> str:
    return "跟随 AstrBot 当前对话模型" if canonical(pid) == CURRENT else canonical(pid)


def format_chain(primary: object, fallbacks: object) -> str:
    """页面提交的主模型 + 降级 → 写回配置的字符串。"""
    ids = [canonical(primary)]
    if isinstance(fallbacks, (list, tuple)):
        ids += [canonical(x) for x in fallbacks]
    elif isinstance(fallbacks, str):
        ids += parse(fallbacks)
    out: list[str] = []
    seen: set[str] = set()
    for pid in ids:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return ",".join(out[:MAX_FALLBACKS + 1])


def validate(primary: object, fallbacks: object,
             available: set[str] | list[str] | None) -> list[str]:
    """校验页面提交的选择，返回规范化后的链。**不合法就抛 ValueError。**

    为什么要拿 `available` 卡一道：模型 id 是自由文本输入，写错一个字符
    不会报错，只会在第一次后台总结时静默失败并被记成「模型未加载」。
    这个错误在页面上应该当场拦住，而不是留到半夜。

    `available` 为 None 表示「拿不到主应用名单」（AstrBot 还没启动好），
    此时只做格式校验，不拦 id——宁可放过，也不要把人的配置锁死。
    """
    if isinstance(fallbacks, str):
        fb = parse(fallbacks)
    elif isinstance(fallbacks, (list, tuple)):
        fb = [canonical(x) for x in fallbacks]
    elif fallbacks is None:
        fb = []
    else:
        raise ValueError("降级模型必须是数组")

    prim = canonical(primary)
    if len(fb) > MAX_FALLBACKS:
        raise ValueError(f"降级模型最多 {MAX_FALLBACKS} 个")

    ids: list[str] = []
    seen: set[str] = set()
    for pid in [prim] + fb:
        if pid in seen:
            raise ValueError(f"「{label(pid)}」重复了：主模型不能同时出现在降级里")
        seen.add(pid)
        ids.append(pid)

    if available:
        known = {canonical(x) for x in available}
        unknown = [label(x) for x in ids if x != CURRENT and x not in known]
        if unknown:
            raise ValueError(
                "AstrBot 里没有这些模型：" + "、".join(unknown)
                + "。请先在主应用的「服务提供商」里添加，再回来选。"
            )
    return ids
