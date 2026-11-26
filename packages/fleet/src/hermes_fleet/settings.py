"""设置页的纯逻辑：把 `_conf_schema.json` 变成可渲染的分组，把表单值清洗成
可写回配置的类型。

放在这里而不是 webapi，是因为「类型转换写错」比「SQL 写错」更难察觉：
布尔值被写成字符串 `"false"` 之后，`bool("false")` 是 True——配置页上明明
关掉了，行为却还是开着。这种错误只能靠单测挡。

shema 由 AstrBot 的插件配置页共用同一份文件，所以这里**不新增字段**，
只负责呈现与校验；新配置项的来源永远是 `_conf_schema.json`。
"""

from __future__ import annotations

from typing import Any

# 分组只决定「设置页怎么排」，不影响 schema。没列入的键会落到「其他」，
# 这样以后加配置项忘了归类也只是排到末尾，而不是在页面上消失。
GROUPS: list[tuple[str, list[str]]] = [
    ("记忆注入", ["memory_injection", "inject_to_system_prompt", "capture_messages",
                  "recall_k", "recall_mode", "group_private_memory",
                  "group_memory_shared",
                  "resident_block_max_chars", "resident_intimacy_max_chars"]),
    ("向量检索", ["vector_provider", "vector_base_url", "vector_api_key",
                  "vector_model", "vector_dim", "vector_batch_size",
                  "vector_timeout_seconds", "vector_query_timeout_seconds",
                  "vector_sync_minutes",
                  "vector_sync_bots", "vector_sync_batch", "vector_min_score"]),
    ("存储与容量", ["bot_db_dir", "max_open_bots", "account_stale_days"]),
    ("后台总结", ["summarize_enabled", "summarize_interval_minutes",
                  "summarize_min_messages", "summarize_idle_hours",
                  "summarize_sessions_per_bot"]),
    ("记忆整理", ["consolidate_enabled", "consolidate_min_active",
                  "consolidate_scopes_per_bot", "consolidate_max_merges"]),
    ("模型链", ["llm_provider_id", "llm_provider_ids", "llm_retry_times",
               "llm_retry_base_seconds", "llm_cooldown_seconds"]),
    ("记忆导出", ["export_enabled", "export_keep_days"]),
]

# 这些值在 `__init__` 里被读一次就固化了（registry 上限、worker 阈值、
# 降级链实例），改了必须重启插件才生效。前端要明确标注，
# 否则用户会以为「保存了但没反应」是 bug。
NEEDS_RELOAD: set[str] = {
    "bot_db_dir", "max_open_bots", "recall_k", "recall_mode",
    "group_memory_shared",
    "vector_provider", "vector_base_url", "vector_api_key", "vector_model",
    "vector_dim", "vector_batch_size", "vector_timeout_seconds",
    "vector_query_timeout_seconds",
    "vector_sync_minutes", "vector_sync_bots", "vector_sync_batch",
    "vector_min_score",
    "summarize_enabled", "summarize_interval_minutes", "summarize_min_messages",
    "summarize_idle_hours", "summarize_sessions_per_bot",
    "consolidate_enabled", "consolidate_min_active",
    "consolidate_scopes_per_bot", "consolidate_max_merges",
    "llm_provider_id", "llm_provider_ids", "llm_retry_times",
    "llm_retry_base_seconds", "llm_cooldown_seconds",
}


def _item(key: str, meta: dict, value: Any) -> dict:
    return {
        "key": key,
        "type": str(meta.get("type") or "string"),
        "default": meta.get("default"),
        "description": str(meta.get("description") or ""),
        "hint": str(meta.get("hint") or ""),
        "options": meta.get("options"),
        "value": value,
        "needs_reload": key in NEEDS_RELOAD,
    }


def schema_groups(schema: dict, values: dict) -> list[dict]:
    """按 GROUPS 分组。`values` 是当前配置，缺失的键用 schema 默认值补。"""
    seen: set[str] = set()
    groups: list[dict] = []
    for title, keys in GROUPS:
        items = []
        for key in keys:
            meta = schema.get(key)
            if not isinstance(meta, dict):
                continue
            seen.add(key)
            items.append(_item(key, meta, values.get(key, meta.get("default"))))
        if items:
            groups.append({"title": title, "items": items})

    rest = [
        _item(key, meta, values.get(key, meta.get("default")))
        for key, meta in schema.items()
        if key not in seen and isinstance(meta, dict)
    ]
    if rest:
        groups.append({"title": "其他", "items": rest})
    return groups


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off", ""):
        return False
    raise ValueError("需要布尔值")


def as_flag(raw: Any, default: bool = False) -> bool:
    """把配置值读成 bool，读不懂就用默认值。

    存在的理由很具体：AstrBot 的配置来自 JSON，但手改配置文件、旧版面板、
    从别处拷来的 `cmd_config.json` 都可能留下字符串 `"false"`，而
    `bool("false")` 是 True——配置页上明明关掉了，行为却还开着，且不会报错。
    插件里所有开关都必须走这里，不能用 `bool(...)` 包一层。
    """
    try:
        return _as_bool(raw)
    except (TypeError, ValueError):
        return default


def _as_number(raw: Any, meta: dict, *, integer: bool) -> float | int:
    val = float(str(raw).strip())
    if integer:
        if val != int(val):
            raise ValueError("需要整数")
        val = int(val)
    low, high = meta.get("min"), meta.get("max")
    if isinstance(low, (int, float)) and val < low:
        raise ValueError(f"不能小于 {low}")
    if isinstance(high, (int, float)) and val > high:
        raise ValueError(f"不能大于 {high}")
    return val


def clean_values(schema: dict, values: dict) -> tuple[dict, list[str]]:
    """把表单值转成配置要求的类型。返回 (可写回的值, 错误列表)。

    未知键直接拒绝，不静默忽略：前端版本不一致时，静默忽略会让人以为
    「保存成功了」，而配置根本没变。
    """
    cleaned: dict = {}
    errors: list[str] = []
    for key, raw in values.items():
        meta = schema.get(key)
        if not isinstance(meta, dict):
            errors.append(f"{key}：未知配置项")
            continue
        typ = str(meta.get("type") or "string")
        try:
            if typ == "bool":
                cleaned[key] = _as_bool(raw)
            elif typ == "int":
                cleaned[key] = _as_number(raw, meta, integer=True)
            elif typ in ("float", "number"):
                cleaned[key] = float(_as_number(raw, meta, integer=False))
            elif typ == "list":
                if isinstance(raw, str):
                    cleaned[key] = [
                        x.strip()
                        for x in raw.replace("\n", ",").replace("，", ",").split(",")
                        if x.strip()
                    ]
                elif isinstance(raw, (list, tuple)):
                    cleaned[key] = [str(x).strip() for x in raw if str(x).strip()]
                else:
                    raise ValueError("需要列表")
            else:
                text = str(raw if raw is not None else "")
                options = meta.get("options")
                if isinstance(options, list) and options and text not in options:
                    raise ValueError(f"只能是 {'/'.join(map(str, options))}")
                cleaned[key] = text
        except (TypeError, ValueError) as exc:
            errors.append(f"{key}：{exc}")
    return cleaned, errors
