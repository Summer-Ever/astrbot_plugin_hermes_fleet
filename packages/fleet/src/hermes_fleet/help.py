"""命令清单文案。

放在纯逻辑包里而不是写在命令方法里：帮助文本会被用户逐字看到，是最容易
「加了命令忘了更新帮助」的地方。把文案做成可单测的函数，再由两个命令入口
共用，至少能保证新增命令时测试会提醒我们同步这里。

`admin` 参数只控制**是否展示**管理员命令，不控制执行——每个管理员命令
自己都有 fail-closed 的权限检查。帮助里显示不等于能用。
"""

from __future__ import annotations

# 所有非管理员也能用的命令。顺序按「日常使用频率」排，不按注册顺序：
# 帮助的作用是让人在群里想起「还能这么干」，不是列源码。
PUBLIC_LINES: tuple[str, ...] = (
    "/提取常驻记忆　输出当前场景实际注入给模型的完整常驻提示词",
    "/fleet 常驻　　查看六个常驻词条当前值（含空槽）",
    "/提取记忆　　 发送「记忆配置 + 聊天原文」两份文件",
    "/导入记忆　　 附带记忆配置文件，把常驻与归档导入本账号",
    "/修改亲密关系和情感表达 <内容>　整段覆盖当前场景的亲密关系词条（AI 不会改它）",
    "/hc帮助　　　 输出这份指令清单",
    "自然语言　　　 直接说「以后叫我阿明」「别记着我不吃辣」，模型自己改常驻",
)

ADMIN_LINES: tuple[str, ...] = (
    "/fleet 状态　　　 本账号记忆概况与模型链",
    "/fleet 我是谁　　 查看平台 id / owner_id",
    "/fleet 消息总结为记忆　立刻跑一轮后台总结（旧名「总结」仍可用）",
    "/fleet 整理　　　 合并重复归档记忆",
    "/fleet 账号列表　 列出库里的账号",
    "/fleet 删除账号记忆 <id>　物理删除整个账号库（不可撤销）",
)

FOOTER = (
    "以上命令也可写作 /fleet <子命令>（提取常驻记忆/提取记忆/导入记忆同样支持）。"
    "管理员后台入口：AstrBot 面板 → 插件页 → 对应机器人账号。"
)


def render_help(*, admin: bool = False) -> str:
    """渲染帮助文本。`admin=True` 时额外列出管理员命令。"""
    lines = ["【Hermes Fleet 可用指令】", ""]
    lines.extend(PUBLIC_LINES)
    if admin:
        lines.extend(["", "管理员指令："])
        lines.extend(ADMIN_LINES)
    lines.extend(["", FOOTER])
    return "\n".join(lines)


__all__ = ["ADMIN_LINES", "PUBLIC_LINES", "render_help"]
