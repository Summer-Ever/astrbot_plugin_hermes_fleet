"""管理台 HTTP 壳。**只做参数解析和响应包装，判断全在 console.py。**

为什么不把逻辑写在这里：hermes_core 的 webapi.py 有 959 行，业务判断与
HTTP 解析混在一起，结果是「页面显示错了」只能在装了 AstrBot 的机器上复现。
本文件的目标是短到不需要测试——每个方法应当只有取参、调函数、返回。

前端见 `views/fleet/`，用法与 core 一致：`window.AstrBotPluginView.apiGet(path)`
里的 path 是相对本插件前缀的（如 `overview`）。

## 只读版之后：下钻页与全局页

v0.5 只有 4 个只读接口（账号盘点）。v0.6 把 hermes_core 那种「内容管理」
补回来了，但换了组织方式：不是「7 个 tab 对着所有数据」，而是

    全局层（账号索引 / 模型链 / 设置）  →  单账号下钻页（记忆 / 常驻 / 消息 / 操作）

原因在 `docs/07`：100 个账号下，core 那种「一张表看全部记忆」的页面
要么炸连接数，要么变成跨租户越权。下钻页只开**一个**库，天然不越界。

仍然刻意没做的：**删除整个账号**。它要删掉一整库用户记忆，必须有
「先导出确认 + 二次输入 platform_id」的流程，草率的删除按钮在百账号
运维里是灾难入口。单条记忆删除是软删（`merged_into` 指自身），可回滚。
"""

from __future__ import annotations

import asyncio
import os
import time

from astrbot.api import logger
from astrbot.api.web import error_response, file_response, json_response, request

PLUGIN_NAME = "astrbot_plugin_hermes_fleet"

# 一次最多探测多少个账号。取值依据：单库 `stats()` 实测约 1-2ms，
# 20 个约 40ms，页面无感；再往上就有把 LRU 冲掉的风险（见 console.probe 注释）。
PROBE_BATCH = 20


class FleetWebAPI:
    def __init__(self, plugin, version: str = "") -> None:
        self.plugin = plugin
        self.version = version
        self.prefix = f"/{PLUGIN_NAME}"

    # ------------------------------------------------------------------ 注册

    def register(self) -> None:
        routes = [
            (f"{self.prefix}/overview", self.overview, ["GET"],
             "账号盘点：列表、统计、告警（不打开任何库）"),
            (f"{self.prefix}/health", self.health, ["GET"],
             "全局健康：模型链、容量水位、账号异常"),
            (f"{self.prefix}/account", self.account, ["GET"],
             "单账号详情：常驻六席与记忆统计"),
            (f"{self.prefix}/probe", self.probe, ["POST"],
             "批量取真实统计（有上限，用完立即关库）"),
            (f"{self.prefix}/notes", self.note_save, ["POST"],
             "给账号起备注（显示用别名，不影响 platform_id 与库文件）"),
            # ---- 下钻页：进去之后才是 hermes_core 那种内容管理 ----
            (f"{self.prefix}/scopes", self.scopes, ["GET"],
             "该账号下的记忆作用域清单（谁、哪个群、各多少条）"),
            (f"{self.prefix}/memories", self.memories, ["GET"],
             "作用域内翻页或检索记忆"),
            (f"{self.prefix}/memories/update", self.memory_update, ["POST"],
             "整条改一条记忆"),
            (f"{self.prefix}/memories/delete", self.memory_delete, ["POST"],
             "软删若干条（可回滚，不物理删除）"),
            (f"{self.prefix}/memories/merge", self.memory_merge, ["POST"],
             "把几条并成一条（来源软删）"),
            (f"{self.prefix}/resident", self.resident, ["GET"],
             "某用户某场景的六个常驻词条"),
            (f"{self.prefix}/resident/save", self.resident_save, ["POST"],
             "整段覆盖一个常驻词条"),
            (f"{self.prefix}/messages", self.messages, ["GET"],
             "未总结消息，按会话分组"),
            (f"{self.prefix}/actions/summarize", self.action_summarize, ["POST"],
             "立刻对该账号跑一轮总结"),
            (f"{self.prefix}/memory/import", self.memory_import, ["POST"],
             "从记忆配置文档导入常驻与归档记忆（管理员后台，不经对话）"),
            (f"{self.prefix}/open-root", self.open_root, ["POST"],
             "用系统文件管理器打开记忆库根目录（仅本机部署有效）"),
            (f"{self.prefix}/backup", self.backup, ["POST"],
             "把全部账号库一致性打包成 zip，供下载/异地保存"),
            (f"{self.prefix}/backup/list", self.backup_list, ["GET"],
             "列出最近备份与恢复点"),
            (f"{self.prefix}/backup/download", self.backup_download, ["GET"],
             "下载一个备份包"),
            (f"{self.prefix}/backup/restore", self.backup_restore, ["POST"],
             "从备份包恢复全部账号与设置（需 confirm=RESTORE）"),
            (f"{self.prefix}/backup/rollback", self.backup_rollback, ["POST"],
             "从恢复前快照回滚"),
            # ---- 全局：模型链与设置 ----
            (f"{self.prefix}/models", self.models, ["GET"],
             "降级链、累计计数、最近调用日志"),
            (f"{self.prefix}/providers", self.providers, ["GET"],
             "主应用里已配置的模型清单（供挑选总结模型）"),
            (f"{self.prefix}/models/chain", self.model_chain_save, ["POST"],
             "保存后台总结模型链（主模型 + 降级），立即生效"),
            (f"{self.prefix}/models/test", self.models_test, ["POST"],
             "逐个模型发一次最小请求，测连通性"),
            (f"{self.prefix}/config", self.config_get, ["GET"],
             "插件配置（分组 + schema）"),
            (f"{self.prefix}/config/save", self.config_save, ["POST"],
             "保存插件配置"),
        ]
        for route, handler, methods, desc in routes:
            try:
                self.plugin.context.register_web_api(route, handler, methods, desc)
            except Exception:
                logger.error(f"[hermes_fleet] 注册 Web API {route} 失败:\n"
                             f"{_tb()}")

    # ------------------------------------------------------------------ 参数

    def _cfg(self, key: str, default):
        try:
            conf = self.plugin.config
            return conf.get(key, default) if conf is not None else default
        except Exception:
            return default

    def _stale_days(self) -> float:
        """「多久没写入算沉默」。下限 0.5 天：填 0 会让所有账号常年告警，
        阈值形同关闭。"""
        try:
            return max(0.5, float(self._cfg("account_stale_days", 7)))
        except (TypeError, ValueError):
            return 7.0

    @staticmethod
    def _fail(message: str, code: int = 400):
        return error_response(message, status_code=code)

    @staticmethod
    async def _body() -> dict:
        try:
            data = await request.json(default={})
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ 接口

    async def overview(self):
        """账号盘点。**这条路径不打开任何数据库连接。**

        参数：q（搜索）、sort（recent/size/name）、limit、offset、probe（0/1）

        `probe=1` 时会对**当前页**额外取一次真实统计。默认关闭，因为它是
        唯一会碰库的路径；首屏不该为「记忆条数」付出打开 50 个库的代价。
        """
        from hermes_fleet.console import SORTS, accounts_view, merge_probe

        try:
            limit = int(request.query.get("limit") or 50)
        except (TypeError, ValueError):
            limit = 50
        try:
            offset = int(request.query.get("offset") or 0)
        except (TypeError, ValueError):
            offset = 0
        sort = str(request.query.get("sort") or "recent")
        if sort not in SORTS:
            # 不认识的排序键静默退回默认，不报错：前端版本不一致时宁可少个功能。
            sort = "recent"

        data = accounts_view(
            self.plugin.registry.accounts(),
            query=str(request.query.get("q") or ""),
            sort=sort, limit=limit, offset=offset,
            stale_days=self._stale_days(),
        )

        if str(request.query.get("probe") or "0") in ("1", "true", "yes"):
            ids = [i["platform_id"] for i in data["items"] if i["platform_id"]]
            if ids:
                probed = await self.plugin.svc.probe(ids, limit=PROBE_BATCH)
                data["items"] = merge_probe(data["items"], probed)
                data["probed"] = len(probed)
            else:
                data["probed"] = 0
        else:
            for it in data["items"]:
                it["stats"] = None
            data["probed"] = 0

        data["version"] = self.version
        data["generated_at"] = time.time()
        return json_response(data)

    async def health(self):
        from hermes_fleet.console import health_view

        st = self.plugin.registry.stats()
        view = health_view(
            self.plugin.registry.accounts(),
            root=str(st.get("root") or ""),
            opened=int(st.get("opened") or 0),
            max_open=int(st.get("max_open") or 0),
            provider=getattr(self.plugin, "provider", None),
            stale_days=self._stale_days(),
        )
        view["version"] = self.version
        return json_response(view)

    async def account(self):
        """单账号详情。这是全页唯一会**常驻**打开一个库的接口。

        platform_id 必须先在索引里存在：否则 `registry.get()` 会凭空新建
        一个空库，页面上看起来「这个账号是空的」，磁盘上已经多了一个文件。
        """
        pid = str(request.query.get("platform_id") or "").strip()
        if not pid:
            return self._fail("缺少 platform_id")
        if not self.plugin.registry.has_account(pid):
            return self._fail("该账号不在索引里", 404)

        try:
            stats = await self.plugin.svc.bot_stats(pid)
        except Exception:
            logger.error(f"[hermes_fleet] 读取账号 {pid} 统计失败:\n{_tb()}")
            return self._fail("读取该账号的库失败，详见插件日志", 500)

        # 常驻六席按固定顺序返回，前端不再自己排序：顺序即注入顺序，
        # 两处维护迟早会不一致。
        resident: dict[str, str] = {}
        try:
            from hermes_contracts import ResidentBlock
            scene = await self.plugin.svc.scene(
                platform_id=pid, sender_id="", group_id="", is_group=False)
            snapshot = await self.plugin.svc.resident_snapshot(scene)
            resident = {b.value: snapshot.get(b.value, "") for b in ResidentBlock.ordered()}
        except Exception:
            logger.warning(f"[hermes_fleet] 读取账号 {pid} 常驻词条失败:\n{_tb()}")

        # 库文件名从索引里取（`accounts()` 只做 listdir，不打开库）。
        # 下钻页标题下面要显示它——排查「这个账号对应哪个文件」时最省事。
        file_name = ""
        try:
            for row in self.plugin.registry.accounts():
                if row.get("platform_id") == pid:
                    file_name = str(row.get("file") or "")
                    break
        except Exception:
            pass

        return json_response({
            "platform_id": pid,
            "file": file_name,
            "note": self.plugin.registry.note_of(pid),
            "stats": stats,
            "resident": resident,
            "resident_hint": "常驻按「私聊」场景读取；群场景的常驻各自独立，"
                             "不在本页展示。",
        })

    async def note_save(self):
        """给账号起/改/清备注。

        备注存在 `_notes.json` 里，**不碰库、不碰文件名**：`platform_id` 是
        适配器 id，改它等于换账号（记忆会「消失」）；文件名里的 hash 变了
        索引就和磁盘对不上。所以备注只能是显示层的别名。

        允许对还没落库的账号写备注（先起名、等它说话是正常顺序），
        但列表只按库文件出行，所以在那之前它不会凭空多出一行。
        """
        from hermes_fleet.registry import MAX_NOTE

        body = await self._body()
        pid = str(body.get("platform_id") or "").strip()
        if not pid:
            return self._fail("缺少 platform_id")
        raw = str(body.get("note") or "")
        if len(raw) > MAX_NOTE * 4:
            # 前端有 maxlength，但接口不能指望前端。直接拒掉超长输入，
            # 比悄悄截断更诚实——截断后用户会以为自己存进去了。
            return self._fail(f"备注太长（最多 {MAX_NOTE} 个字符）")
        try:
            note = self.plugin.registry.set_note(pid, raw)
        except Exception:
            logger.error(f"[hermes_fleet] 保存 {pid} 备注失败:\n{_tb()}")
            return self._fail("写入备注失败，详见插件日志", 500)
        return json_response({
            "platform_id": pid,
            "note": note,
            "trimmed": raw.strip() != note,
        })

    async def probe(self):
        from hermes_fleet.console import merge_probe

        body = await self._body()
        ids = body.get("platform_ids")
        if not isinstance(ids, list):
            return self._fail("platform_ids 必须是数组")
        clean = [str(x).strip() for x in ids if str(x or "").strip()]
        if not clean:
            return self._fail("platform_ids 为空")
        if len(clean) > PROBE_BATCH:
            return self._fail(f"一次最多探测 {PROBE_BATCH} 个账号")
        probed = await self.plugin.svc.probe(clean, limit=PROBE_BATCH)
        return json_response({
            "items": merge_probe([{"platform_id": p} for p in clean], probed),
            "count": len(probed),
        })

    # ------------------------------------------------------------------ 下钻

    def _db_of(self, pid: str):
        """打开一个**已存在**的账号库，返回 `(db, 错误响应)`。

        先查索引再 `get()`：`registry.get()` 对不认识的 id 会凭空建一个空库
        文件。页面上看起来「这个账号是空的」，磁盘上已经多了一个永远是空的
        库——之后它还会以「孤立库」告警出现在列表里，变成自造的噪声。
        """
        if not pid:
            return None, self._fail("缺少 platform_id")
        if not self.plugin.registry.has_account(pid):
            return None, self._fail("该账号不在索引里", 404)
        return self.plugin.registry.get(pid), None

    async def scopes(self):
        """该账号的记忆作用域清单。下钻页左栏的全部数据来源。"""
        from hermes_fleet import admin

        pid = str(request.query.get("platform_id") or "").strip()
        db, err = self._db_of(pid)
        if err is not None:
            return err
        bot = self.plugin.svc.bot_of(pid)
        return json_response({
            "platform_id": pid,
            "scopes": await admin.list_scopes(db, bot),
            "stats": db.stats(),
        })

    async def memories(self):
        """作用域内翻页 / 检索。

        `q` 非空走存储的 `search_candidates`（库内 FTS），为空走游标分页。
        两条路的返回形状一致，前端不用分支。
        """
        from hermes_fleet import admin

        pid = str(request.query.get("platform_id") or "").strip()
        db, err = self._db_of(pid)
        if err is not None:
            return err

        bot = self.plugin.svc.bot_of(pid)
        try:
            catalog = admin.key_of(
                bot,
                scope=str(request.query.get("scope") or "user"),
                owner_id=str(request.query.get("owner_id") or ""),
                group_id=str(request.query.get("group_id") or ""),
            )
        except ValueError as exc:
            return self._fail(f"作用域参数不成立：{exc}")

        limit = _q_int("limit", 50)
        after_id = _q_int("after_id", 0)
        q = str(request.query.get("q") or "").strip()
        try:
            if q:
                items = await admin.search_memories(db, catalog, query=q, limit=limit)
            else:
                items = await admin.page_memories(db, catalog, limit=limit, after_id=after_id)
        except Exception:
            logger.error(f"[hermes_fleet] 读取 {pid} 记忆失败:\n{_tb()}")
            return self._fail("读取记忆失败，详见插件日志", 500)

        return json_response({
            "items": items,
            "count": len(items),
            # 前端用「这一页满了就当还有下一页」，不额外查 count(*)——
            # 记忆表可能几十万行，为了一个分页计数全表扫不值得。
            "next_after_id": items[-1]["id"] if items else 0,
            "has_more": len(items) >= max(1, min(200, limit)),
            "searched": bool(q),
        })

    async def memory_update(self):
        from hermes_fleet import admin

        body = await self._body()
        pid = str(body.get("platform_id") or "").strip()
        db, err = self._db_of(pid)
        if err is not None:
            return err
        try:
            memory_id = int(body.get("id"))
        except (TypeError, ValueError):
            return self._fail("缺少有效的 id")

        try:
            row = await admin.edit_memory(
                db, self.plugin.svc.bot_of(pid), memory_id,
                content=body.get("content"),
                kind=body.get("kind"),
                visibility=body.get("visibility"),
            )
        except KeyError:
            return self._fail("这条记忆不存在", 404)
        except ValueError as exc:
            return self._fail(str(exc))
        except Exception:
            logger.error(f"[hermes_fleet] 修改 {pid} 记忆失败:\n{_tb()}")
            return self._fail("写入失败，详见插件日志", 500)
        return json_response({"item": row})

    async def memory_delete(self):
        from hermes_fleet import admin

        body = await self._body()
        pid = str(body.get("platform_id") or "").strip()
        db, err = self._db_of(pid)
        if err is not None:
            return err
        ids = body.get("ids")
        if not isinstance(ids, list) or not ids:
            return self._fail("ids 必须是非空数组")
        if len(ids) > 200:
            return self._fail("一次最多删 200 条")
        try:
            n = await admin.forget_memories(db, self.plugin.svc.bot_of(pid), ids)
        except Exception:
            logger.error(f"[hermes_fleet] 删除 {pid} 记忆失败:\n{_tb()}")
            return self._fail("写入失败，详见插件日志", 500)
        return json_response({"deleted": n, "note": "软删：可回滚，不会物理消失"})

    async def memory_merge(self):
        from hermes_fleet import admin

        body = await self._body()
        pid = str(body.get("platform_id") or "").strip()
        db, err = self._db_of(pid)
        if err is not None:
            return err
        sources = body.get("source_ids")
        if not isinstance(sources, list):
            return self._fail("source_ids 必须是数组")
        try:
            into_id = int(body.get("into_id"))
        except (TypeError, ValueError):
            return self._fail("缺少有效的 into_id")
        try:
            n = await admin.merge_memories(
                db, self.plugin.svc.bot_of(pid), sources, into_id)
        except KeyError:
            return self._fail("合并目标不存在", 404)
        except ValueError as exc:
            return self._fail(str(exc))
        except Exception:
            logger.error(f"[hermes_fleet] 合并 {pid} 记忆失败:\n{_tb()}")
            return self._fail("写入失败，详见插件日志", 500)
        return json_response({"merged": n})

    async def resident(self):
        """某用户某场景的六个常驻词条。

        下钻页的常驻 tab 要能切换「私聊 / 某群」，所以必须按 owner+scope 取；
        `account` 接口只返回私聊那一套（它服务的是列表页的快速预览）。
        """
        from hermes_fleet import admin

        pid = str(request.query.get("platform_id") or "").strip()
        db, err = self._db_of(pid)
        if err is not None:
            return err
        owner_id = str(request.query.get("owner_id") or "").strip()
        if not owner_id:
            return self._fail("缺少 owner_id")
        scope = str(request.query.get("scope") or "private").strip()
        try:
            values = await admin.resident_of(
                db, self.plugin.svc.bot_of(pid), owner_id, scope)
        except ValueError as exc:
            return self._fail(f"场景参数不成立：{exc}")
        return json_response({
            "platform_id": pid, "owner_id": owner_id, "scope": scope,
            "resident": values,
        })

    async def resident_save(self):
        from hermes_fleet import admin

        body = await self._body()
        pid = str(body.get("platform_id") or "").strip()
        db, err = self._db_of(pid)
        if err is not None:
            return err
        owner_id = str(body.get("owner_id") or "").strip()
        if not owner_id:
            return self._fail("缺少 owner_id")
        try:
            row = await admin.save_resident(
                db, self.plugin.svc.bot_of(pid), owner_id,
                str(body.get("scope") or "private"),
                str(body.get("block") or ""),
                str(body.get("text") or ""),
            )
        except ValueError as exc:
            return self._fail(str(exc))
        except Exception:
            logger.error(f"[hermes_fleet] 保存 {pid} 常驻失败:\n{_tb()}")
            return self._fail("写入失败，详见插件日志", 500)
        return json_response({"item": row})

    async def messages(self):
        from hermes_fleet import admin

        pid = str(request.query.get("platform_id") or "").strip()
        db, err = self._db_of(pid)
        if err is not None:
            return err
        view = await admin.pending_messages(
            db, self.plugin.svc.bot_of(pid), limit=_q_int("limit", 50))
        view["platform_id"] = pid
        return json_response(view)

    async def action_summarize(self):
        """对该账号立刻跑一轮总结。

        这是「看得到问题之后能立刻动手」的那一步：静默账号的真正原因常常是
        消息攒着没被总结，等后台循环可能要 15 分钟。
        """
        body = await self._body()
        pid = str(body.get("platform_id") or "").strip()
        db, err = self._db_of(pid)
        if err is not None:
            return err
        bot = self.plugin.svc.bot_of(pid)
        try:
            outcome = await self.plugin.summarizer.run_bot(
                bot, messages=db.messages, memories=db.memories,
                identities=db.identities,
            )
        except Exception:
            logger.error(f"[hermes_fleet] 账号 {pid} 手动总结失败:\n{_tb()}")
            return self._fail("总结失败，详见插件日志", 500)
        return json_response({
            "result": str(outcome),
            "memories": int(getattr(outcome, "memories", 0) or 0),
            "messages": int(getattr(outcome, "messages", 0) or 0),
            "failures": int(getattr(outcome, "failures", 0) or 0),
        })

    async def memory_import(self):
        """管理员后台导入记忆配置文档（不经对话，不会被 LLM 吃掉）。

        与聊天命令 `/导入记忆` 共用同一套解析 + 写入逻辑：
        `parse_memory_config()` -> `svc.import_owner()`。差别只在入口——
        这里是面板管理员在账号管理页选本地文件，文件内容由桥接以文本
        传进来，所以复用的是完全相同的校验与冲突策略。

        目标用户可被管理员覆盖：
        - 不传 `owner_id`：用文档记录的来源用户（原样复原场景）；
        - 传了 `owner_id`：导入到该账号下的指定用户（迁移场景）。
        目标机器人账号固定为当前下钻页的 `platform_id`。
        """
        from hermes_fleet.export import MAX_EXPORT_BYTES, parse_memory_config
        from hermes_fleet.settings import as_flag

        if not as_flag(self._cfg("export_enabled", True), True):
            return self._fail("记忆导入已在插件设置里关闭", 403)

        body = await self._body()
        pid = str(body.get("platform_id") or "").strip()
        if not self.plugin.registry.has_account(pid):
            return self._fail("目标机器人账号不存在", 404)

        text = str(body.get("text") or "")
        if not text.strip():
            return self._fail("文件内容为空")
        if len(text.encode("utf-8")) > MAX_EXPORT_BYTES:
            return self._fail("文件超过 100MB，拒绝导入")

        try:
            payload = parse_memory_config(text)
        except ValueError as exc:
            return self._fail(f"文档解析失败：{exc}")

        # 管理员可覆盖目标用户；留空则用文档记录的来源用户。
        owner = str(body.get("owner_id") or "").strip() or str(
            payload.get("owner_id") or "").strip()
        if not owner:
            return self._fail("文档没有记录用户标识，请指定导入到哪个用户")

        try:
            result = await self.plugin.svc.import_owner(
                platform_id=pid, owner_id=owner, payload=payload)
        except ValueError as exc:
            return self._fail(str(exc))
        except Exception:
            logger.error(f"[hermes_fleet] 后台导入 {pid}/{owner} 失败:\n{_tb()}")
            return self._fail("导入失败，详见插件日志", 500)

        return json_response({
            "ok": True,
            "platform_id": pid,
            "owner_id": owner,
            "doc_owner_id": payload.get("owner_id", ""),
            "doc_bot_id": payload.get("bot_id", ""),
            "resident_imported": result["resident_imported"],
            "memories_imported": result["memories_imported"],
            "message": (
                f"导入完成：常驻 {result['resident_imported']} 条（按词条覆盖），"
                f"归档记忆 {result['memories_imported']} 条（追加）。"
            ),
        })

    async def open_root(self):
        """用系统文件管理器打开记忆库根目录。

        两条安全约束：
        1. **不接受客户端传路径**——路径恒取自 `registry.root`。否则一个
           插件后台接口就能打开服务器任意目录，等于给了文件系统浏览能力。
        2. 只做「打开」不返回目录清单——列目录另有 overview 接口按账号做，
           这里不提供任意路径的枚举。

        只在 AstrBot 跑在本机时有意义（用户当前就是本地部署）。面板若在
        服务器/容器里，服务端文件管理器开在远端，用户看不到，所以失败要
        如实说清楚，不能假装成功。
        """
        import platform
        import subprocess

        root = str(getattr(self.plugin.registry, "root", "") or "")
        if not root or not os.path.isdir(root):
            return self._fail("记忆库目录不存在", 404)

        # which 只接受两个固定枚举，路径由服务端拼，绝不接受客户端传来的
        # 任意子路径（否则等于把文件管理器当任意目录浏览器用）。
        body = await self._body()
        which = str(body.get("which") or "root")
        if which not in ("root", "backups"):
            return self._fail("无效的目录类型")
        if which == "backups":
            root = self._backup_out_dir()
            if not os.path.isdir(root):
                return self._fail("还没有备份目录，请先点一次备份", 404)

        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(root)  # noqa: S606 — 打开的是服务端固定目录
            elif system == "Darwin":
                subprocess.Popen(["open", root])
            else:
                subprocess.Popen(["xdg-open", root])
        except Exception as exc:
            logger.warning(f"[hermes_fleet] 打开库目录失败: {exc}")
            return self._fail(
                "无法打开文件夹：面板可能运行在服务器/容器内（此时打开的是远端 "
                "目录），或本机没有图形界面。请手动复制下面的路径到文件管理器。", 500)

        return json_response({"ok": True, "path": root, "system": system})

    async def backup(self):
        """把全部账号库与插件设置打包成 zip（一致性快照），供下载/异地保存。

        真正的工作在纯逻辑模块 `hermes_fleet.backup`，这里只取服务端路径、
        调函数、包装响应。备份目录固定在本机数据目录下，前端拿相对文件名
        拼接下载链接——不接受客户端指定输出路径。
        """
        from hermes_fleet.backup import backup_bots

        try:
            result = backup_bots(
                root=str(self.plugin.registry.root),
                out_dir=self._backup_out_dir(),
                version=self.version,
                config=self._config_values(),
            )
        except Exception:
            logger.error(f"[hermes_fleet] 备份失败:\n{_tb()}")
            return self._fail("备份失败，详见插件日志", 500)

        return json_response(result)

    # ------------------------------------------------------------ 备份与恢复

    def _backup_out_dir(self) -> str:
        """备份目录（固定在插件数据目录，不跟 `bot_db_dir` 跑）。

        必须落在 `plugin_data` 之外的理由：卸载插件勾「删除数据」会连带
        删掉 `plugin_data/<插件>`，备份跟着消失就失去了「误删插件也能救」
        的意义。`main.HermesFleet.backup_root()` 用同一个规则。
        """
        try:
            return str(self.plugin.backup_root())
        except Exception:
            data_dir = os.path.dirname(str(self.plugin.registry.root))
            return os.path.join(data_dir, "backups")

    def _config_values(self) -> dict:
        try:
            return dict(self.plugin.config) if self.plugin.config is not None else {}
        except Exception:
            return {}

    def _backup_path(self, filename: object) -> str:
        """把客户端给的文件名夹进备份目录，非法规格返回空串。

        只接受纯文件名：`..`、绝对路径、盘符一律不认。这是唯一由客户端
        指定服务端文件的位置，必须窄到不能再窄。
        """
        name = str(filename or "").strip()
        if not name or os.path.basename(name) != name:
            return ""
        if not name.endswith(".zip"):
            return ""
        return os.path.join(self._backup_out_dir(), name)

    def _restore_lock(self) -> asyncio.Lock:
        """插件级串行锁：两个恢复同时跑会互相覆盖对方的暂存目录。"""
        lock = getattr(self.plugin, "_backup_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            try:
                self.plugin._backup_lock = lock
            except Exception:
                pass
        return lock

    async def backup_list(self):
        from hermes_fleet.restore import list_backups

        out_dir = self._backup_out_dir()
        items = list_backups(out_dir)
        return json_response({
            "dir": out_dir,
            "count": len(items),
            "backups": [i for i in items if i["kind"] != "restore-point"],
            "restore_points": [i for i in items if i["kind"] == "restore-point"],
            "latest": items[0]["filename"] if items else "",
        })

    async def backup_download(self):
        name = str(request.query.get("filename") or "")
        path = self._backup_path(name)
        if not path or not os.path.isfile(path):
            return self._fail("备份文件不存在", 404)
        return file_response(
            path, filename=os.path.basename(path),
            content_type="application/zip",
        )

    async def _run_restore(self, path: str, *, confirm: str,
                           want_config: bool, rollback: bool):
        """恢复/回滚的公共后半段：关库 → 换文件 → 重建索引 → 应用设置。"""
        from hermes_fleet.restore import RestoreError, restore_backup, rollback_restore

        reg = self.plugin.registry
        runner = rollback_restore if rollback else restore_backup
        async with self._restore_lock():
            try:
                result = await asyncio.to_thread(
                    lambda: runner(
                        path,
                        root=str(reg.root),
                        out_dir=self._backup_out_dir(),
                        confirm=confirm,
                        version=self.version,
                        # 换入前必须先关连接：Windows 上库文件被打开时
                        # `os.replace` 会直接失败；关早一点，别的协程又会
                        # 立刻把库重新打开。这里紧贴着换入动作关。
                        before_apply=reg.close_all,
                        after_apply=reg.reload_from_disk,
                    )
                )
            except RestoreError as exc:
                return self._fail(str(exc))
            except Exception:
                logger.error(f"[hermes_fleet] 恢复失败:\n{_tb()}")
                return self._fail("恢复失败，详见插件日志", 500)

        payload = result.pop("config", None)
        config_saved: list[str] = []
        config_errors: list[str] = []
        needs_reload: list[str] = []
        if want_config and isinstance(payload, dict) and payload:
            from hermes_fleet.settings import NEEDS_RELOAD, clean_values

            # `bot_db_dir` 保留本机值：恢复包的库路径属于另一台机器，
            # 覆盖过来会让插件去找一个本机不存在的目录。
            values = {k: v for k, v in payload.items() if k != "bot_db_dir"}
            cleaned, errors = clean_values(self._schema(), values)
            config_errors.extend(errors)
            if cleaned and self.plugin.config is not None:
                try:
                    self.plugin.config.update(cleaned)
                    await self.plugin.config.save_config_async()
                    config_saved = sorted(cleaned)
                    needs_reload = sorted(k for k in cleaned if k in NEEDS_RELOAD)
                except Exception:
                    logger.error(f"[hermes_fleet] 恢复插件设置失败:\n{_tb()}")
                    config_errors.append("插件设置写入失败，详见日志")

        message = str(result.get("message") or "")
        if config_saved:
            message += f" 已恢复设置 {len(config_saved)} 项。"
        if config_errors:
            message += " 设置恢复有问题：" + "；".join(config_errors) + "。"
        if needs_reload:
            message += " 需重载插件生效：" + "、".join(needs_reload) + "。"
        result.update({
            "message": message,
            "config_saved": config_saved,
            "config_errors": config_errors,
            "needs_reload": needs_reload,
        })
        return json_response(result)

    async def backup_restore(self):
        from hermes_fleet.restore import RestoreError, inspect_backup

        body = await self._body()
        path = self._backup_path(body.get("filename"))
        if not path or not os.path.isfile(path):
            return self._fail("备份文件不存在", 404)
        # 预检放在这里先跑一次：让「包坏了」在开锁之前就以问题清单返回，
        # 而不是让 restore_backup 抛一个异常了事。
        plan = inspect_backup(path, root=str(self.plugin.registry.root))
        if not plan["ok"]:
            return self._fail("备份包不可用：" + "；".join(plan["problems"]))
        try:
            return await self._run_restore(
                path,
                confirm=str(body.get("confirm") or ""),
                want_config=body.get("config") is not False,
                rollback=False,
            )
        except RestoreError as exc:  # pragma: no cover - _run_restore 已兜住
            return self._fail(str(exc))

    async def backup_rollback(self):
        from hermes_fleet.restore import PRERESTORE_PREFIX

        body = await self._body()
        name = str(body.get("filename") or "").strip()
        if not name.startswith(PRERESTORE_PREFIX):
            return self._fail("回滚只接受恢复前快照（hermes-pre-restore-*.zip）")
        path = self._backup_path(name)
        if not path or not os.path.isfile(path):
            return self._fail("快照文件不存在", 404)
        return await self._run_restore(
            path,
            confirm=str(body.get("confirm") or ""),
            want_config=body.get("config") is not False,
            rollback=True,
        )

    # ------------------------------------------------------------------ 模型

    async def models(self):
        """降级链 + 配置意图 + 候选清单 + 累计计数 + 最近调用日志。

        `FallbackProvider.log` 是内存里的环形缓冲（64 条），重启即清空——
        这是刻意的：模型故障是「当下能不能干活」的问题，把历史日志落盘
        只会让人对着三天前的限流记录做决策。
        """
        provider = getattr(self.plugin, "provider", None)
        if provider is None:
            return self._fail("模型链未初始化", 500)
        stats = getattr(provider, "stats", None)
        log = []
        for item in list(getattr(provider, "log", []) or [])[-40:]:
            log.append({
                "ts": float(getattr(item, "ts", 0) or 0),
                "provider": str(getattr(item, "provider", "")),
                "purpose": str(getattr(item, "purpose", "")),
                "ok": bool(getattr(item, "ok", False)),
                "kind": str(getattr(item, "kind", "")),
                "latency_ms": round(float(getattr(item, "latency_ms", 0) or 0), 1),
                "detail": str(getattr(item, "detail", ""))[:300],
            })
        # 配置意图与候选清单是「诊断视图」的一部分：只看链上现在跑的是什么
        # 不够，还要能看出它是不是跟着主应用漂移、以及能换成哪些。
        # 这两段取不到时留空，不能让整个诊断页 500。
        configured: dict = {}
        catalog: list[dict] = []
        try:
            configured = self.plugin.chain_config()
        except Exception:
            logger.warning(f"[hermes_fleet] 读取模型链配置失败:\n{_tb()}")
        try:
            catalog, _ = self.plugin.providers_catalog()
        except Exception:
            logger.warning(f"[hermes_fleet] 读取主应用模型清单失败:\n{_tb()}")
        return json_response({
            "chain": provider.chain(),
            "summary": provider.summary(),
            "configured": configured,
            "catalog": catalog,
            "stats": {
                "calls": int(getattr(stats, "calls", 0) or 0),
                "failures": int(getattr(stats, "failures", 0) or 0),
                "by_provider": dict(getattr(stats, "by_provider", {}) or {}),
                "by_kind": dict(getattr(stats, "by_kind", {}) or {}),
            },
            "log": log,
        })

    async def providers(self):
        """主应用里**已配置**的对话模型清单。

        只读：这里不修改 AstrBot 的任何配置。候选从主应用来，但「用哪个」
        由本插件自己的配置决定——这就是「解耦但不出圈」。
        """
        from hermes_fleet.chain import SOURCE_LABELS

        try:
            rows, current = self.plugin.providers_catalog()
        except Exception:
            logger.error(f"[hermes_fleet] 读取模型清单失败:\n{_tb()}")
            return self._fail("读取主应用的模型清单失败，详见插件日志", 500)
        configured = {}
        try:
            configured = self.plugin.chain_config()
        except Exception:
            pass
        source = str(configured.get("source") or "current")
        return json_response({
            "items": rows,
            "current": current,
            "configured": configured,
            "source": source,
            "source_text": SOURCE_LABELS.get(source, source),
            "readable": bool(rows),
            "hint": "清单来自 AstrBot 主应用已配置的服务提供商；"
                    "改主应用的默认模型不会影响这里已选定的链。",
        })

    async def model_chain_save(self):
        """保存「总结用」模型链（主模型 + 降级），并立刻生效。

        三个约束都要在服务端做，不能只靠前端：
        1. 只接受主应用里真实存在的 provider id（写错一个字符不报错，
           只会在半夜的总结里静默失败）；
        2. 主模型不能同时出现在降级里；
        3. 保存后立即重建 provider，并换掉后台任务手里的引用——
           否则就是「保存了，但总结还在用旧模型」。
        """
        from hermes_fleet import chain as chain_mod

        body = await self._body()
        if "primary" not in body and "fallbacks" not in body:
            return self._fail("缺少 primary / fallbacks")

        try:
            rows, _ = self.plugin.providers_catalog()
        except Exception:
            rows = []
        # 拿不到清单（主应用还没启动完）时 available=None：只校验格式，
        # 不拦 id。宁可放过一次，也不要把人的配置锁死在错误里。
        available = {r["id"] for r in rows} if rows else None
        try:
            ids = chain_mod.validate(
                body.get("primary"), body.get("fallbacks"), available)
        except ValueError as exc:
            return self._fail(str(exc))

        stored = ",".join(ids)
        try:
            self.plugin.config.update({"llm_provider_ids": stored})
            await self.plugin.config.save_config_async()
        except Exception:
            logger.error(f"[hermes_fleet] 保存模型链失败:\n{_tb()}")
            return self._fail("写入配置文件失败，请查看后端日志", 500)

        try:
            self.plugin.rebuild_provider()
        except Exception:
            logger.error(f"[hermes_fleet] 重建模型链失败:\n{_tb()}")
            # 配置已经落盘了，只是这一轮没热切换成功。如实告诉调用方：
            # 重载插件即可生效，不要报成「保存失败」让人重做一遍。
            return json_response({
                "stored": stored,
                "applied": False,
                "chain": [chain_mod.label(x) for x in ids],
                "message": "配置已保存，但热切换失败；重载插件后生效。",
            })

        return json_response({
            "stored": stored,
            "applied": True,
            "chain": [chain_mod.label(x) for x in ids],
            "configured": self.plugin.chain_config(),
            "message": "已保存并立即生效（无需重载插件）。",
        })

    async def models_test(self):
        """逐个模型发一句最小请求。

        **绕过降级链直连单个 provider**：从链上测只能知道「还行/全挂了」，
        分不清是主模型坏了还是兜底模型在替它干活——而这正是这里要回答的。

        有代价（真花配额、真有延迟），所以限 3 个且必须由用户点击触发。
        """
        from hermes_contracts import ChatMessage

        body = await self._body()
        wanted = body.get("ids")
        try:
            chain = dict(self.plugin._provider_chain())
        except Exception:
            logger.error(f"[hermes_fleet] 读取模型链失败:\n{_tb()}")
            return self._fail("读取模型链失败", 500)
        names = [str(x) for x in wanted] if isinstance(wanted, list) and wanted else list(chain)
        names = names[:3]

        out = []
        for name in names:
            impl = chain.get(name)
            if impl is None:
                out.append({"name": name, "ok": False, "error": "未加载"})
                continue
            t0 = time.perf_counter()
            try:
                res = await impl.complete(
                    [ChatMessage(role="user", content="回复 ok 两个字母即可")],
                    purpose="test", max_tokens=16, temperature=0.0,
                )
                out.append({
                    "name": name, "ok": True,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "model": getattr(res, "model", "") or "",
                    "reply": (getattr(res, "text", "") or "")[:40],
                })
            except Exception as exc:  # noqa: BLE001 — 测试接口要把失败原因原样带回
                out.append({
                    "name": name, "ok": False,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                })
        return json_response({"items": out})

    # ------------------------------------------------------------------ 设置

    def _schema(self) -> dict:
        import json
        import os

        if not hasattr(self, "_schema_cache"):
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "_conf_schema.json")
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self._schema_cache = data if isinstance(data, dict) else {}
            except Exception:
                logger.warning(f"[hermes_fleet] 读取配置 schema 失败:\n{_tb()}")
                self._schema_cache = {}
        return self._schema_cache

    async def config_get(self):
        from hermes_fleet.settings import schema_groups

        if self.plugin.config is None:
            return self._fail("插件没有拿到配置对象", 500)
        try:
            values = dict(self.plugin.config)
        except Exception:
            values = {}
        return json_response({
            "groups": schema_groups(self._schema(), values),
            "values": values,
        })

    async def config_save(self):
        from hermes_fleet.settings import NEEDS_RELOAD, clean_values

        body = await self._body()
        values = body.get("values")
        if not isinstance(values, dict) or not values:
            return self._fail("没有要保存的配置")
        cleaned, errors = clean_values(self._schema(), values)
        if errors:
            return self._fail("；".join(errors))
        try:
            self.plugin.config.update(cleaned)
            await self.plugin.config.save_config_async()
        except Exception:
            logger.error(f"[hermes_fleet] 保存配置失败:\n{_tb()}")
            return self._fail("写入配置文件失败，请查看后端日志", 500)
        return json_response({
            "saved": sorted(cleaned),
            "needs_reload": sorted(k for k in cleaned if k in NEEDS_RELOAD),
        })


def _q_int(name: str, default: int) -> int:
    """取整数查询参数。非法值退回默认而不是报错：
    分页参数错了应该给一页数据，而不是让整页空白。"""
    try:
        return int(request.query.get(name) or default)
    except (TypeError, ValueError):
        return default


def _tb() -> str:
    import traceback

    return traceback.format_exc(limit=3)
