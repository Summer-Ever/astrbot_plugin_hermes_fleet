# 模块化开发与多 Agent 协作

> 配套代码：`hermes-platform/`。已验证：`python tools/verify.py` 四项全绿，
> 90 个用例通过（含 2 套一致性套件 × 2 个实现）。

---

## 0. 结论先说

**能。但"把目录切开"不等于模块化。** 要在多个 agent 之间真正并行，必须先做完三件事：

| # | 装置 | 解决什么 | 缺了会怎样 |
|---|---|---|---|
| 1 | **零依赖的契约包** | 「同一个概念叫什么」 | 两边各定义一份 `Memory`，合并时字段错位，线上表现为"某些记忆查不到" |
| 2 | **每个模块的内存实现** | 「我没法等你的数据库」 | 串行：B 要等 A 的 Postgres 就绪才能开始 |
| 3 | **边界的机器检查** | 「没人会读文档」 | 一个 `import` 让两个模块重新耦合，且 review 时看不见 |

三件都到位之后，并行的单位不是"文件"而是"模块 + 契约"，联调不再是"把人叫到一起试"，而是"跑同一个套件"。

---

## 1. 三个会真实发生的失败模式

先讲清楚为什么朴素做法一定失败。这三个不是假想，是多模块 Python 项目的固定剧本。

### 失败模式一：共享实体被定义两遍

```
hermes_memory/models.py   ->  class Memory: content, owner_id, visibility ...
hermes_api/schemas.py     ->  class Memory: content, owner, visible ...
```

两边都能跑，各自的测试都绿。拼起来时：
- `visibility` 是 `"private"` 还是 `True`？
- `owner_id` 为空字符串还是 `None`？

这类错不会抛异常，只会**静默丢数据**。而且要等联调时才发现，此时两边都写了两千行。

**对策**：实体只有一份定义，放 `hermes_contracts.models`。存储实现负责行映射，其余模块只消费不可变对象。

### 失败模式二：为了图快直接 import 别人的实现

```python
# hermes_memory/service.py —— 看起来完全合理
from hermes_identity import IdentityService
```

代价不是"耦合不好看"，而是：**从这一刻起，memory 的测试必须能 import identity**。identity 的 agent 只要重构了构造函数，memory 的测试就红——而 memory 的 agent 不知道发生了什么。

更糟的是这个依赖会传染：api 依赖 memory，memory 依赖 identity，identity 依赖……

**对策**：只 import `hermes_contracts.ports` 里的 `Protocol`，实现从构造函数注入。memory 完全不认识 identity，它只知道"有个东西能把 raw_id 解析成 canonical_id"。

### 失败模式三：测试需要别人的环境

```python
# 想跑 memory 的可见性测试，需要：
# 1. 本地 Postgres
# 2. 跑过迁移
# 3. 造好 bots 表的行
```

于是没人跑。CI 里跳过了。等到改坏可见性判定时，没有任何东西会红——这类 bug 上线即隐私事故。

**对策**：内存实现 + 一致性套件。测试在没有数据库的机器上 0.2 秒跑完。

---

## 2. 模块地图与依赖方向

```
                    ┌──────────────────────────┐
                    │   hermes_contracts       │  ← 依赖图的根，零第三方依赖
                    │  ids errors models ports │
                    │  + conformance 套件      │
                    └────────────▲─────────────┘
                                 │ 所有模块只准依赖这里
        ┌──────────┬─────────────┼─────────────┬──────────┐
        │          │             │             │          │
   hermes_recall  hermes_memory  hermes_jobs  hermes_identity ...
   （纯函数）      （memories）    （jobs）      （identities）
        │          │             │             │
        └──────────┴─────────────┴─────────────┘
                                 │ 全部只满足 Protocol
                    ┌────────────▼─────────────┐
                    │  组装层 composition root │
                    │  决定谁满足哪个 Protocol │
                    └────────────▲─────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
        hermes_workers        hermes_api      astrbot_hermes_client
        （编排，全 fake 可测）  （HTTP 适配）     （采集侧薄插件）
```

**唯一一条硬规则**：领域包之间**互相不可见**。所有横向通信走 `contracts` 里的 Protocol，由组装层注入。

### 为什么按「数据所有权」切，而不是按技术分层

常见错误切法是 `models/ services/ api/`——这是**技术分层**，不是模块化。它的致命问题是：改任何一个功能都要同时动三层，等于迫使所有 agent 改同一批文件。

正确的切法判据是三条：

1. **谁拥有哪张表** —— `memories` 归 memory，`jobs` 归 jobs，`identities` 归 identity。跨表事务要重新审视边界。
2. **变更频率是否不同** —— 排序数学（recall）天天调参，队列语义（jobs）半年不动。放一起会让稳定的部分被频繁改动连累。
3. **能否脱离宿主单测** —— 纯函数 > 单端口 > 多端口。能单测的放外层。

按这三条切出来的模块表：

| 包 | 拥有的表 | 可独立验证方式 | 现状 |
|---|---|---|---|
| `hermes_contracts` | 无 | 一致性套件本身 | ✅ 完成 |
| `hermes_recall` | 无（纯函数） | 直接断言输入输出 | ✅ 完成（从 hermes_core 整份搬） |
| `hermes_memory` | `memories` `resident` | 内存实现 + conformance | 🟡 内存版完成，PG 待做 |
| `hermes_jobs` | `jobs` | 内存实现 + conformance | 🟡 内存版完成，PG 待做 |
| `hermes_identity` | `bots` `identities` | 内存实现 | ⬜ |
| `hermes_skill` | `skills` | 内存实现 + 假文件系统 | ⬜ |
| `hermes_llm` | 无（冷却状态） | 假 provider | ⬜ |
| `hermes_workers` | 无（编排） | 全 fake | ⬜ |
| `hermes_api` | 无（适配） | 假服务 + 测试客户端 | ⬜ |
| `hermes_storage_pg` | DDL/迁移 | 复用同一批 conformance | ⬜ |
| `astrbot_hermes_client` | 本地 spool | 假 HTTP | ⬜ |

---

## 3. 契约层：唯一的共享物

### 3.1 四条边界

| 文件 | 内容 | 判断标准 |
|---|---|---|
| `ids.py` | 标识符、作用域、键的序列化 | 「同一件事叫什么」 |
| `errors.py` | 错误码 + 可重试性 + HTTP 映射 | 「失败了算什么」 |
| `models.py` | 跨模块实体（frozen dataclass） | 「跨边界传的形状」 |
| `ports.py` | 各 Store / Provider 的 Protocol | 「我能要求别人做什么」 |

### 3.2 什么该进契约

用三个问题筛：

1. **会不会有两个模块都用到？** 只在一个模块内部用的东西，不进。
2. **它变了会不会导致两个模块要同时改？** 会，才进契约（这正是要冻结的）。
3. **它能不能用标准库表达？** 不能就要重新设计——契约一旦依赖 pydantic，所有 agent 的沙箱都得装包。

`recall.py` 的排序参数、`main.py` 的注入文案、SQL 语句——都**不进**契约，它们是模块内部实现。

### 3.3 冻结的含义与变更流程

「冻结」不是「永远不能改」，是「改的代价被显式化」。当前策略：

```
提出（任何人，写清动机与影响面）
  ↓
评审（受影响模块的 owner 各看一眼）
  ↓
单人执行（架构 owner 一次性改完契约 + 所有 conformance 套件）
  ↓
广播（其他人同步拉取，各自修自己的红测试）
```

**为什么必须单人执行**：契约改动会让所有人的测试同时变红。三个 agent 同时改契约，等于三个人同时改同一个 git 冲突，且冲突在语义层不在文本层——文本能自动合并，语义不能。

**一个实用约束**：契约新增字段一律给默认值。这样已有的实现不会因为"多了一个字段"而全部编译不过，可以分批跟进。`Memory` 的每个字段都有默认值，就是这个考虑。

---

## 4. 双实现 + 一致性套件（核心机制）

这是整套方案里最重要的一件事。

### 4.1 机制

```
        契约包里的 conformance 套件（约 300 行断言）
                 │
        ┌────────┴────────┐
   InMemoryMemoryStore   PgMemoryStore
   （测试/开发用）        （生产用）
```

实现方只需要写一行：

```python
class TestPgMemoryStore(MemoryStoreConformance):
    async def make_store(self, clock):
        return await PgMemoryStore.connect(DSN, clock=clock)
```

就自动继承了「跨 bot 隔离」「私聊不泄露」「合并可回滚」「游标不重复」等全部语义保证。

### 4.2 这条机制解决的三个具体问题

**问题一：B 不用等 A。** memory 的 agent 立刻能测 api 依赖的行为，因为内存实现就是"一个符合契约的 memory"。

**问题二：换实现不用重写测试。** 上 Postgres 时，不设计新测试，只多写一个子类。如果 PG 实现和内存实现行为不一致，套件会指出**是哪一条语义**对不上。

**问题三：契约被违反时本地就红。** 不再依赖"联调时发现"。

### 4.3 fake 必须认真（这段是踩过的）

写内存实现时最容易的偷懒是"只实现被用到的几个方法"。结果是上层在 fake 上全绿、上真库全红——**fake 的价值完全取决于它有多认真**。

`InMemoryJobQueue` 里三个容易写错的语义，都是 conformance 逼出来的：

| 语义 | 写错的后果 | 写错时有没有报错 |
|---|---|---|
| `failed` 必须可被再次领取 | 失败任务永远卡住，积压但不报错 | 没有，只有 `queue_lag` 慢慢涨 |
| `dead` 必须不可被领取 | 死信被反复重试，配额烧光 | 没有 |
| `enqueue` 在途去重，但 `done` 后可重新入队 | 会话只被总结一次，之后再也不总结 | 没有 |

这三条**全部是静默失败**。所以它们必须写成断言，而不是写在注释里。

### 4.4 套件本身也要被验证

一个没被执行过的测试套件，和没有测试是一回事——你不知道它自己有没有 bug。

本次交付里，`JobQueueConformance` 的 16 个用例是在写完 `InMemoryJobQueue` 之后才第一次真正跑起来的。跑的过程中确实发现了两处问题（`JobState` 是普通常量类，用 `is` 比较在某些路径下不成立；`fail` 后未重置 `visible_at` 导致无法立即重试）。如果只有一套"看起来对"的套件而没有实现，这两个问题会留到 PG 实现时才爆。

**结论**：每加一套 conformance，必须同时加一个内存实现把它跑绿。顺序不能反。

### 4.5 Postgres 实现怎么接

```python
# packages/storage_pg/tests/test_pg_conformance.py
@unittest.skipUnless(os.getenv("HERMES_TEST_DSN"), "需要真实 Postgres")
class TestPgMemoryStore(MemoryStoreConformance):
    async def make_store(self, clock):
        return await PgMemoryStore.connect(os.environ["HERMES_TEST_DSN"], clock=clock)
```

用环境变量跳过，而不是用 mock：没数据库时跳过（本地日常），有数据库时全跑（CI 夜间）。**跳过必须显式打印原因**，否则会出现"CI 全绿但什么都没测"。

---

## 5. 边界自动化

`tools/check_boundaries.py`，四条规则：

| 规则 | 内容 | 为什么 |
|---|---|---|
| R1 | `contracts` 不得 import 任何 `hermes_*` | 它是依赖图的根，反向即循环 |
| R2 | 领域包之间不得互相 import | 直接依赖 = 无法独立测试 |
| R3 | 不得 import 其他包的实现子模块（`*.inmemory`） | 绕过包边界直插内部 |
| R4 | `contracts` 只准标准库（白名单） | 否则所有沙箱都要装包 |

实测有效（故意写一行违规 import 验证过）：

```
packages/memory/src/hermes_memory/_probe.py:1  import hermes_identity
  [R3 引用了未知的 hermes 包]
```

**为什么必须是机器检查**：这类退化是渐进、单次无害、review 时极难发现的（一行 import 埋在文件第 30 行）。指望每个 agent 都记得规则，等于指望它不发生。

---

## 6. 四阶段工作流

### 阶段 0 · 契约期（串行，不可并行）

| 项 | 内容 |
|---|---|
| 谁做 | 一个 agent（或人），其他 agent **不要开工** |
| 输入 | `docs/01-开发指南.md` 的 §5 数据模型、§6 流程 |
| 输出 | `ids / errors / models / ports` 四个文件 + 至少 1 套 conformance |
| 门禁 | 契约评审通过；`python tools/verify.py --fast` 绿 |
| 工期感 | 1-2 天。**赶这一步等于把返工摊到后面每个模块** |

为什么不能并行：契约是所有模块的公共输入。两个 agent 同时"顺手改一下契约"，最后没人能说清 `visibility` 到底有几种取值。

### 阶段 1 · 并行期（N 个 agent 同时跑）

每个 agent 拿到的边界是明确的：

- **可写**：自己的 `packages/<x>/` 与 `packages/<x>/tests/`
- **只读**：`hermes_contracts`
- **禁止**：其他任何包、`owners.json` 里的 `reserved_files`

每个 agent 的验收标准是同一个形状：

```bash
python run_tests.py -p <自己模块>    # 必须全绿
python tools/check_boundaries.py     # 必须无违规
```

**这一阶段唯一会互相阻塞的东西是契约变更**。所以纪律是：遇到契约不够用，先本地绕开（加个私有适配），把变更需求记下来，阶段 2 统一处理。不要中途改契约。

### 阶段 2 · 联调期

联调不是"把人叫到一起试"，是**跑集成测试**：

```bash
python run_tests.py -p integration
```

`tests/integration/` 里的用例只测"拼接"，不测业务细节。它回答的唯一问题是：**两个模块对同一份契约的理解是否一致**。

出问题时定位顺序固定：

1. 先跑各模块自己的测试 → 区分"谁的锅"（通常两边都绿）
2. 再看是哪个 **conformance 断言**没覆盖到这条交互 → 补进套件
3. 最后才改实现

第 2 步是关键：联调中发现的每个问题，都应该变成 conformance 里的一条新断言，而不是"改完就好了"。否则下次换实现还会犯。

### 阶段 3 · 收口期

- 契约变更集中处理（一次改完 + 全员同步）
- `owners.json` 状态更新
- 版本号三处对齐（`contracts.__version__` / `pyproject` / 清单）

---

## 7. 并行度怎么排：先内存，后 Postgres

关键洞察：**存储层是并行度的瓶颈，但可以绕开。**

两批分法：

**批 A —— 不需要任何数据库（可以立刻并行）**

| 模块 | 怎么做 | 依赖 |
|---|---|---|
| `hermes_identity` | 内存实现，先跑 conformance | contracts |
| `hermes_skill` | 内存实现 + 假文件系统（`tmpdir`） | contracts |
| `hermes_llm` | 假 provider（返回固定 JSON / 指定错误） | contracts |
| `hermes_workers` | 全 fake：假 store + 假 queue + 假 llm | contracts |
| `hermes_api` | 假服务 + 测试客户端 | contracts |
| `astrbot_hermes_client` | 假 HTTP（本地起一个 stub） | contracts |

批 A 六个模块可以**同时开六个 agent**，彼此零阻塞。

**批 B —— 需要 DDL（必须等阶段 0 的存储定稿）**

`hermes_storage_pg` 及其各 Store 的 PG 实现。它们的测试不能在内存跑，只能在真库上跑，所以必须在批 A 之后（或者用一个 agent 提前把 DDL 定稿，但**不要和批 A 同时改表结构**）。

**这条排序的实际收益**：业务逻辑（api / workers）在数据库还不存在时就已经验证完了。等 PG 实现就绪时，它要做的只是"通过同一批 conformance"，而不是"顺便调业务逻辑的 bug"。

---

## 8. 派发 agent 的指令模板

直接复制，把 `<...>` 换掉：

```
你在 hermes-platform 仓库工作。这是一个多 agent 并行开发的模块，边界是强制的。

你的模块：<包名>
你拥有的表：<表名 或 "无">
可写目录：packages/<包名>/ 及其 tests/
只读：packages/contracts/（契约，不要改）
禁止：修改其他任何包、owners.json 里的 reserved_files

任务：<具体要实现的 Protocol>

必须先读：
1. packages/contracts/src/hermes_contracts/ports.py 里 <XxxStore> 的 docstring
2. packages/contracts/src/hermes_contracts/conformance/<xxx>.py 的断言
3. docs/02-模块化与多agent协作.md 第 4、11 节

验收（必须自己跑通再交付）：
  python run_tests.py -p <包名>
  python tools/check_boundaries.py

如果契约不够用：
  不要改契约。用私有适配层绕开，在交付说明里写清"需要什么变更、为什么"。
  契约变更由架构 owner 统一处理。

交付时告诉我：
  - 实现了哪几个 Protocol
  - conformance 是否全绿（几个用例）
  - 哪些地方你认为契约有歧义
  - 有没有为了通过测试而"猜"的语义
```

最后一条要求最有价值：「有没有为了通过测试而猜的语义」。它把"我不确定但先这么写"从隐性问题变成显性问题，这比让 agent 假装确定要好得多。

---

## 9. 冲突热点与串行区

这些文件**任何 agent 不得直接修改**（已列在 `owners.json.reserved_files`）：

| 文件 | 冲突形式 | 处理 |
|---|---|---|
| `contracts/{ids,errors,models,ports}.py` | 语义层冲突，无法自动合并 | 阶段 2 集中改 |
| `owners.json` | 文本冲突好解，语义冲突难 | 只有架构 owner 改 |
| `pyproject.toml` | 各人加自己包的路径 | 阶段 0 一次性配全（含未来包） |
| `run_tests.py` / `tools/*` | 工具本身 | 只有工具 owner 改 |
| 迁移脚本 | 序号冲突 | **用时间戳前缀**（`20261125T2013_add_bots.sql`），不用递增编号 |

迁移脚本用时间戳是有代价的（顺序不如序号直观），但换来的是**两个 agent 永远不会撞号**。在有并行写入的场景里，这个交换是划算的。

---

## 10. 联合测试的三道闸

| 闸 | 何时 | 命令 | 覆盖 |
|---|---|---|---|
| 本地 | 每次提交前 | `python run_tests.py -p <模块>` | 自己模块 + 契约 |
| 合并前 | PR | `python tools/verify.py` | 全量 + 边界 + 所有权一致性 |
| 夜间 | 定时 | `python tools/verify.py` + 真 Postgres | 含 PG conformance |
| 发版前 | 手工 | 真 AstrBot + 真服务 | 端到端注入/召回 |

`verify.py` 的**第一项**（所有权一致性）值得单独说：它检查 `owners.json` 与目录是否双向一致。如果某个 agent 偷偷建了新包没登记，或者标为"完成"的包其实不存在，这里会红。这是"流程被绕过"的探测器。

---

## 11. 已经验证过什么

不写"设计上可行"，写实际跑过的：

```
$ python tools/verify.py
[1/4 所有权一致性]   11 个包登记，4 个包存在
[2/4 版本一致]       hermes_contracts.__version__ = '0.1.0'
[3/4 依赖方向]       依赖方向合规
[4/4 测试]           通过 90  失败 0  错误 0  跳过 0  用时 0.51s
校验通过（4 项）
```

| 验证项 | 结果 |
|---|---|
| 契约层零依赖可导入 | ✅ 裸 Python 3.13，无任何第三方包 |
| `MemoryStoreConformance` | ✅ 19 用例，内存实现全过 |
| `JobQueueConformance` | ✅ 16 用例，内存实现全过（过程中修了 2 个 fake 的语义错） |
| 集成链路（写入→粗排→精排→渲染） | ✅ 9 用例 |
| 跨 bot / 跨用户隔离 | ✅ 存储层与集成层各有一道断言 |
| 边界检查器能抓违规 | ✅ 故意插一行 `import hermes_identity` 被抓 |
| 一键校验 | ✅ 四项全绿 |

**同时固化了一个已知缺陷**：`test_known_limitation_pure_semantic_query_misses`
断言「我喝茶的偏好」召不回「用户喜欢喝乌龙茶」。这是 bigram 检索的天花板，
写成通过的测试而不是注释——等 M4 接上向量召回后它会变红，那时改写成"能召回"，
而不是删掉。

---

## 12. 反模式清单

明确不要做：

1. **不要给每个模块定义自己的 model**。实体只有一份。
2. **不要让领域包互相 import**。哪怕"就这一次"。边界是渐进的。
3. **不要跳过内存实现直接写 Postgres**。你会失去全部本地可测性。
4. **不要为了通过 conformance 而改 conformance**。套件是契约的一部分，改它等于改契约。
5. **不要在契约里放业务逻辑**。一旦契约有行为，它就变成了一个模块，而所有模块都依赖它，改不动。
6. **不要用共享全局单例传依赖**。`import` 时初始化数据库连接的模块，永远无法在测试里替换。
7. **不要让 agent 中途改契约**。绕开 + 记录需求，阶段 2 统一改。
8. **不要把联调当成"跑一次看看"**。每个联调发现的问题都要落成一条 conformance 断言。

---

## 13. 新增一个模块的 checklist

按顺序做，不要跳：

```
[ ] 1. 在 owners.json 登记：owner / owns / may_import / test / status
[ ] 2. 在 pyproject.toml 的 pythonpath 里加路径（这一步找架构 owner，不要自己改）
[ ] 3. 建 packages/<x>/src/hermes_<x>/ 与 packages/<x>/tests/
[ ] 4. 写内存实现 → 跑通对应的 conformance
       （如果契约里还没有该 Protocol 的 conformance，先补套件，
        这是契约变更，要走评审）
[ ] 5. python run_tests.py -p <x>        全绿
[ ] 6. python tools/check_boundaries.py  无违规
[ ] 7. python tools/verify.py            四项全绿
[ ] 8. status 从 todo 改成 in_progress / done
```

第 4 步的顺序不能反：**先有套件，后有实现**。反过来（先写实现再补测试）几乎总是变成"测试照着实现的 bug 写"。

---

## 附录：本目录结构

```
hermes-platform/
├── owners.json                    # 谁拥有什么（唯一权威）
├── pyproject.toml                 # 工具配置（pytest/ruff），刻意不配 build-system
├── run_tests.py                   # 零依赖测试入口，支持 -p 分片
├── tools/
│   ├── check_boundaries.py        # 依赖方向静态检查（AST，秒级）
│   └── verify.py                  # 一键校验（CI 入口）
├── packages/
│   ├── contracts/src/hermes_contracts/
│   │   ├── ids.py errors.py models.py ports.py clock.py
│   │   └── conformance/           # 一致性套件（契约的一部分）
│   ├── recall/                    # 从 hermes_core 整份搬 + adapter
│   ├── memory/                    # memories / resident
│   └── jobs/                      # jobs
├── tests/integration/             # 只测拼接，不测业务
└── docs/02-模块化与多agent协作.md
```
