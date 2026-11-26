"""管理台路由扫雷：31 条接口逐条真调用，不允许未捕获异常或 500。

为什么值得单独一层：`webapi.py` 是「页面点了没反应」的高发区，而它的
单元测试（`console` / `admin`）只测纯逻辑，测不到「handler 名称写错、
`_body()` 拿到 None、`self.plugin` 上某个属性在真实实例里不存在」这类
装配错误。这里从 `main._register_web_apis()` 捕获的真实路由表出发，
保证「注册了」＝「调得动」。

两类用例：
1. 空参数：每条路由都必须给出 4xx/200，而不是抛异常或 500。
2. 有数据的账号：只读接口必须真的读出内容（账号 / 作用域 / 记忆 /
   常驻 / 消息），这会把 `main.py → webapi.py → console/admin → 库`
   整条下钻链路走一遍。
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from unittest import mock

_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from harness.astrbot_stub import request as web_request  # noqa: E402
from harness.env import PluginEnv, load_main  # noqa: E402

PREFIX = "/astrbot_plugin_hermes_fleet"


class WebAPISweepTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_main()

    async def asyncSetUp(self) -> None:
        # 31 条路由串行跑完约 0.1s，正好压在 asyncio debug 的慢任务阈值上。
        # 这不是卡顿，调高阈值免得每次 CI 都刷一条误导性的告警。
        asyncio.get_running_loop().slow_callback_duration = 5.0
        self.env = PluginEnv()
        self.routes = self.env.register_web_apis()

    async def asyncTearDown(self) -> None:
        web_request.reset()
        await self.env.close()

    async def test_空参数逐条调用不崩(self) -> None:
        self.assertEqual(len(self.routes), 31, f"路由变了：{sorted(self.routes)}")

        failures: list[str] = []
        for route in sorted(self.routes):
            handler = self.routes[route]["handler"]
            web_request.reset()
            try:
                with mock.patch("os.startfile", create=True):
                    # open-root 会真的拉起系统文件管理器；patch 掉之后
                    # 仍然走完「路径取自 registry.root」的整条成功分支。
                    res = await handler()
            except Exception as exc:  # noqa: BLE001 — 扫雷就是要收集所有异常
                failures.append(f"{route}: {type(exc).__name__}: {exc}")
                continue
            with self.subTest(route=route):
                self.assertIsInstance(res, dict, f"{route} 必须返回响应对象")
                self.assertIn("status", res)
                self.assertLess(
                    int(res["status"]), 500,
                    f"{route} 空参数不应 500：{res}",
                )
                if res.get("__kind__") == "error":
                    self.assertTrue(res.get("message"), f"{route} 的报错没有说明")

        self.assertEqual(failures, [], "以下路由抛了未捕获异常：\n" + "\n".join(failures))

    async def test_有数据的账号可以被下钻读取(self) -> None:
        ev = self.env.event(message="我住在杭州")
        await self.env.turn(ev, "我住在杭州", "好的")
        await self._save_resident(ev)
        await self.env.plugin.tool_save_memory(ev, "用户住在杭州，喜欢西湖")

        reads = {
            "account": {"platform_id": "qq-01"},
            "scopes": {"platform_id": "qq-01"},
            "memories": {
                "platform_id": "qq-01", "scope": "user",
                "owner_id": "u1", "group_id": "",
            },
            "resident": {
                "platform_id": "qq-01", "owner_id": "u1", "scope": "private",
            },
            "messages": {"platform_id": "qq-01"},
        }
        out: dict[str, dict] = {}
        for name, query in reads.items():
            route = f"{PREFIX}/{name}"
            self.assertIn(route, self.routes)
            web_request.reset(query=query)
            res = await self.routes[route]["handler"]()
            self.assertEqual(res["status"], 200, f"{name} 读取失败：{res}")
            out[name] = res["data"]

        self.assertEqual(out["account"]["platform_id"], "qq-01")
        self.assertGreaterEqual(out["account"]["stats"]["memories"], 1)
        # 账号预览读的是「共享常驻」（owner_id 为空），私聊词条不在这一层，
        # 所以这里只校验结构；具体词条由下面的 resident 路由按 owner 取。
        self.assertIsInstance(out["account"]["resident"], dict)

        self.assertTrue(out["scopes"]["scopes"], "作用域清单不该为空")
        self.assertTrue(out["memories"]["items"], "记忆翻页不该为空")
        self.assertIn("西湖", str(out["memories"]["items"]))
        self.assertEqual(out["resident"]["resident"]["address"], "杭州")
        self.assertTrue(out["messages"], "未总结消息视图不该为空")

    async def _save_resident(self, ev) -> None:
        msg = await self.env.plugin.tool_update_resident(ev, "address", "杭州")
        self.assertIn("杭州", msg)

    async def test_备份下载恢复整条链路(self) -> None:
        """备份 → 列表 → 下载 → 恢复，一条链走通（这是会动数据的路径）。"""
        import shutil
        import tempfile

        tmp = tempfile.mkdtemp(prefix="hermes-web-bk-")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.env.plugin.backup_root = lambda: os.path.join(tmp, "backups")

        ev = self.env.event(message="我住在杭州")
        await self.env.turn(ev, "我住在杭州", "好的")

        web_request.reset()
        res = await self.routes[f"{PREFIX}/backup"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        name = res["data"]["filename"]
        self.assertEqual(res["data"]["bots"], 1)

        web_request.reset()
        res = await self.routes[f"{PREFIX}/backup/list"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertIn(name, [i["filename"] for i in res["data"]["backups"]])

        web_request.reset(query={"filename": name})
        res = await self.routes[f"{PREFIX}/backup/download"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertEqual(res["__kind__"], "file")

        web_request.reset(payload={"filename": name, "confirm": "RESTORE"})
        res = await self.routes[f"{PREFIX}/backup/restore"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertEqual(res["data"]["restored"], 1)
        self.assertTrue(res["data"]["snapshot_name"].startswith("hermes-pre-restore-"))

        # 恢复后连接必须已重建，账号仍可读（否则页面会白屏）。
        web_request.reset(query={"platform_id": "qq-01"})
        res = await self.routes[f"{PREFIX}/account"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertEqual(res["data"]["platform_id"], "qq-01")

    async def test_按账号开关可保存并在详情里读回(self) -> None:
        """单个机器人关记忆就靠这条：写入 bot.json，详情页再读回来。"""
        ev = self.env.event(message="我住在杭州")
        await self.env.turn(ev, "我住在杭州", "好的")

        route = f"{PREFIX}/account-overrides"
        web_request.reset(payload={
            "platform_id": "qq-01",
            "overrides": {"capture_messages": False, "recall_k": 9},
        })
        res = await self.routes[route]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertIs(res["data"]["overrides"]["capture_messages"], False)
        self.assertEqual(res["data"]["overrides"]["recall_k"], 9)

        web_request.reset(query={"platform_id": "qq-01"})
        res = await self.routes[f"{PREFIX}/account"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertEqual(res["data"]["overrides"]["recall_k"], 9)

        # null = 删掉这条覆盖，回到「跟随全局」。
        web_request.reset(payload={
            "platform_id": "qq-01", "overrides": {"capture_messages": None},
        })
        res = await self.routes[route]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertNotIn("capture_messages", res["data"]["overrides"])

        # 全局资源类开关不认识就直接拒，不能静默丢掉。
        web_request.reset(payload={
            "platform_id": "qq-01", "overrides": {"llm_provider_ids": "x"},
        })
        res = await self.routes[route]["handler"]()
        self.assertEqual(res["status"], 400, res)

    async def test_立刻总结不看条数阈值(self) -> None:
        """按钮的语义是「现在就总结」，不是「条件满足才总结」。"""
        ev = self.env.event(message="我下周三搬家")
        await self.env.turn(ev, "我下周三搬家", "好的")

        # 阈值抬到远高于现有条数：不 force 的话这一轮只会 skipped。
        self.env.plugin.summarizer.min_messages = 50
        self.env.plugin.summarizer.idle_seconds = 7 * 24 * 3600

        web_request.reset(payload={"platform_id": "qq-01"})
        res = await self.routes[f"{PREFIX}/actions/summarize"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertGreaterEqual(res["data"]["messages"], 1, res)

    async def test_总结与整理预览只读且可确认(self) -> None:
        """预览必须真的调模型，但不能写库/标记，确认按钮才写。"""
        ev = self.env.event(message="我下周三搬家")
        await self.env.turn(ev, "我下周三搬家", "好的")

        web_request.reset(payload={"platform_id": "qq-01"})
        res = await self.routes[f"{PREFIX}/actions/summarize/preview"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertGreaterEqual(res["data"]["messages"], 1, res)
        self.assertTrue(res["data"]["log"], "预览要把过程交回页面")

        # 预览不标记：未总结视图仍然有这批消息。
        web_request.reset(query={"platform_id": "qq-01"})
        res = await self.routes[f"{PREFIX}/messages"]["handler"]()
        self.assertTrue(res["data"], "预览后消息不该被标记为已总结")

        web_request.reset(payload={"platform_id": "qq-01"})
        res = await self.routes[
            f"{PREFIX}/actions/consolidate/preview"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertIn("plans", res["data"])

        web_request.reset(payload={"platform_id": "qq-01"})
        res = await self.routes[f"{PREFIX}/actions/consolidate"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertIn("result", res["data"])

    async def test_群聊会出现在群清单与未总结视图里(self) -> None:
        """群聊常常只有常驻、没有归档记忆：页面必须仍能看见这个群。"""
        gid = "g-10086"
        ev = self.env.event(group_id=gid, sender_id="u9", message="记住叫我老王")
        await self.env.turn(ev, "记住叫我老王", "好的老王")

        web_request.reset(query={"platform_id": "qq-01"})
        res = await self.routes[f"{PREFIX}/scopes"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertIn(gid, [g["group_id"] for g in res["data"]["groups"]])

        web_request.reset(query={"platform_id": "qq-01"})
        res = await self.routes[f"{PREFIX}/messages"]["handler"]()
        self.assertEqual(res["status"], 200, res)
        self.assertIn(gid, str(res["data"]), "群会话不该在未总结视图里消失")


if __name__ == "__main__":
    unittest.main()
