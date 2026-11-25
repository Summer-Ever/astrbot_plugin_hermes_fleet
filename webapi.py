"""管理台 HTTP 壳。**只做参数解析和响应包装，判断全在 console.py。**

为什么不把逻辑写在这里：hermes_core 的 webapi.py 有 959 行，业务判断与
HTTP 解析混在一起，结果是「页面显示错了」只能在装了 AstrBot 的机器上复现。
本文件的目标是短到不需要测试——每个方法应当只有取参、调函数、返回。

前端见 `views/fleet/`，用法与 core 一致：`window.AstrBotPluginView.apiGet(path)`
里的 path 是相对本插件前缀的（如 `overview`）。

## 这一版是只读的

账号盘点、详情、批量探测、健康 —— 都不写。删除账号（purge）刻意没做：
它要删一整库用户记忆，必须有「先导出确认 + 二次输入 platform_id」的完整
流程，草率的删除按钮在百账号运维里是灾难入口。等只读版跑顺了再做。
"""

from __future__ import annotations

import time

from astrbot.api import logger
from astrbot.api.web import error_response, json_response, request

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

        return json_response({
            "platform_id": pid,
            "stats": stats,
            "resident": resident,
            "resident_hint": "常驻按「私聊」场景读取；群场景的常驻各自独立，"
                             "不在本页展示。",
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


def _tb() -> str:
    import traceback

    return traceback.format_exc(limit=3)
