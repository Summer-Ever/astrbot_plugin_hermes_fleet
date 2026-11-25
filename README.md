# Hermes 百账号记忆（astrbot_plugin_hermes_fleet）

给**上百个机器人账号**用的长期记忆插件。每个账号一个独立的记忆库文件，
账号之间的记忆**物理隔离**——串号不是靠"记得写 WHERE"来避免，而是不可能发生。

已在 AstrBot 4.x 上按 `qq_official`（QQ 官方机器人）形态验证接入方式。

## 它做什么

| 能力 | 触发点 | 说明 |
|---|---|---|
| 每轮注入常驻设定 | `on_llm_request` | 称呼、助手设定、用户信息、偏好忌讳、回复风格、规则避雷六个词条 |
| 会话落库 | `on_llm_response` | 用户发言 + 机器人回复写进该账号自己的消息表 |
| 记忆检索 | 工具 `search_memory` | 模型主动调用，检索归档记忆 |
| 记忆写入 | 工具 `save_memory` | 自动归属到当前发言用户 |
| 常驻维护 | 工具 `update_resident` | 整段覆盖，不是追加 |
| 删除记忆 | 工具 `forget_memory` | 软删除，判错了能回滚 |
| 运维 | `/fleet` 命令组 | 状态、我是谁、账号列表、删除账号记忆库 |

**不做**每轮自动检索注入。每轮固定烧几千字符换回来的多半是噪声，
还会把真正的用户发言挤到后面。召回交给模型主动调用。

## 安装

把仓库放进 AstrBot 插件目录（或在 WebUI 的插件市场里填仓库地址）：

```bash
cd <AstrBot>/data/plugins
git clone https://github.com/Summer-Ever/astrbot_plugin_hermes_fleet.git
```

重启 AstrBot。**零第三方依赖**，不需要 pip 装任何东西。

要求 Python ≥ 3.11（用到 `X | None` 语法）与带 FTS5 的 SQLite（标准库自带）。

## 配置

六个开关，都在插件设置页：

| 键 | 默认 | 作用 |
|---|---|---|
| `memory_injection` | 开 | 每轮注入常驻设定 |
| `capture_messages` | 开 | 记录会话原文（后续总结的原料） |
| `bot_db_dir` | 空 | 留空用 `data/plugin_data/astrbot_plugin_hermes_fleet/bots/` |
| `max_open_bots` | 24 | 同时打开的账号库上限，超出按 LRU 关闭 |
| `recall_k` | 5 | 每次检索返回条数 |
| `group_private_memory` | 关 | 允许在群里引用本人的私聊记忆（默认关，见下文） |

## 为什么每个账号一个库文件

两种隔离方式的成本不对称：

- **一个库 + `WHERE bot_id=?`**：靠每条 SQL 都记得带条件。漏一处就静默串号，
  **而且不会报错**。100 个账号、几十处查询，靠纪律守住不成立。
- **一个账号一个文件**：另一个账号的数据根本不在这个连接能看到的范围内。
  删账号就是删文件，备份就是拷文件。

实测尾延迟也是每库一文件更好（同机、总负载 200 条/秒）：

| 写者形态 | p99 |
|---|---|
| 单库 + 100 写线程 | 2288 ms |
| 单库 + 1 写线程（只按条数提交） | 517 ms |
| 单库 + 1 写线程（条数或超时双触发） | 75 ms |
| **每 bot 一库** | **7.8 ms** |

复现：`python benchmarks/bench_arch.py`。

## 「是哪个账号」怎么取

```python
event.get_platform_id()   # -> 面板里给每个适配器配的 id，如 "qq-01"
```

依据是 AstrBot 的适配器元数据：`PlatformMetadata.id` 取自每个适配器配置的
`id` 字段，`qq_official` 适配器同样如此（`id=cast(str, self.config.get("id"))`）；
`get_platform_id()` 的注释也写明"同时运行多个同类型适配器时 ID 唯一"。
所以一个 AstrBot 实例里配 100 个 QQ 官方机器人，插件能直接分辨是哪一个，
**不需要改本体、不需要打补丁**。

文件名会带 8 位 hash 后缀：`qq-01` 与 `qq_1` 若只做字符替换会折叠成同一个
文件，两个号共用一个记忆——这类事故是静默的。

## 隐私边界

- **私聊与群聊的常驻是两套**：`private` 与 `group:<gid>` 分开存，私聊设定不会
  在群里生效。
- **群里不引用私聊记忆**：`group_private_memory` 默认关闭。`can_read` 里
  "owner 相同即可读"在私聊是对的，但群里把私聊内容念出来就是当众公开。
- **群记忆不外溢**：在 A 群存的事，B 群检索不到。

## 开发

```bash
python run_tests.py            # 全量（127 个用例，零依赖）
python run_tests.py -p fleet   # 只跑插件层（20 个隔离用例）
python tools/check_boundaries.py   # 依赖方向检查
python tools/verify.py         # 一键校验（CI 入口）
```

**业务逻辑不 import astrbot。** 所有记忆语义在 `packages/fleet/`，
`main.py` 只做"取事件字段 → 调 service → 塞回请求"。
直接收益：本机没装 AstrBot 也能跑完整隔离测试——hermes_core 的 5 套 smoke
里有 3 套正卡在这一点上。

```
main.py                       AstrBot 接入层（只做翻译）
hermes_fleet/                 业务层（注入块、检索、工具语义、账号路由）
  registry.py                 platform_id -> 库文件，LRU，purge
  service.py                  注入/落库/检索/常驻的策略
packages/contracts/           冻结的契约（实体只定义一次）
packages/recall/              检索与重排（从 hermes_core 整份搬来）
packages/storage_sqlite/      四种 Store 的 SQLite 实现
benchmarks/                   承载能力实测脚本
docs/                         设计与部署文档
owners.json                   模块所有权（谁可以改什么）
```

## 当前状态与下一步

- ✅ 每账号一库、注入、落库、四个工具、运维命令、20 个隔离用例
- ⬜ **自动总结**（把会话压成记忆条目）：需要接 AstrBot 的 provider 调模型，
  目前消息只落库不压缩。这是下一步的第一件事。
- ⬜ 深度整理（合并重复记忆）、技能系统、WebUI 管理页、向量召回

未实现的部分不写进配置项——占位开关比没有更糟，会让人以为它在工作。

## 许可

MIT
