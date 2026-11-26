"""金路径端到端：真的执行 `main.py`，只把 AstrBot 本体换成桩。

覆盖的是「一个用户从说一句话到记忆可用」的完整链路：

    常驻写入 → on_llm_request 注入 → 45 秒去重 → on_llm_response 落库
    → 后台总结出归档记忆 → search_memory 检索 → forget_memory 删除

这些步骤以前分散在各包的单元测试里，但把它们串起来的那层（`main.py`）
从未被执行过。最典型的漏法是字段名对不上：`event.get_platform_id()`
拼错、`scene(**args)` 的关键字写错，包测试全绿而线上整条链路静默失效。
"""

from __future__ import annotations

import os
import re
import sys
import unittest

_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from harness.env import PluginEnv, load_main  # noqa: E402


class GoldenPathTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 在同步上下文里预加载 `main.py`：它要做一遍 sys.path 注入和
        # `hermes_*` 模块清理，放进 asyncSetUp 会被 asyncio 的 slow-callback
        # 日志当成卡顿打出来（IsolatedAsyncioTestCase 默认开 debug）。
        load_main()

    async def asyncSetUp(self) -> None:
        self.env = PluginEnv()
        self.plugin = self.env.plugin

    async def asyncTearDown(self) -> None:
        await self.env.close()

    async def test_常驻词条进入本轮请求且同句不重复注入(self) -> None:
        ev = self.env.event(message="以后叫我阿明")

        msg = await self.plugin.tool_update_resident(ev, "address", "阿明")
        self.assertIn("称呼", msg)
        self.assertIn("阿明", msg)

        req = self.env.main.ProviderRequest(system_prompt="你是助手")
        ev.message_str = "以后叫我阿明"
        await self.plugin.on_llm_request(ev, req)

        self.assertEqual(len(req.extra_user_content_parts), 1)
        part = req.extra_user_content_parts[0]
        self.assertIn(self.env.main.INJECT_OPEN, part.text)
        self.assertIn("阿明", part.text)
        self.assertTrue(part.is_temp, "注入必须走临时 part，否则会写进会话历史")

        # 同一句话（工具循环里的下一轮）：45 秒内不再注入
        again = self.env.main.ProviderRequest(system_prompt="你是助手")
        await self.plugin.on_llm_request(ev, again)
        self.assertEqual(again.extra_user_content_parts, [])

        # 换一句话：照常注入，但同一个请求对象里不会叠第二份
        ev.message_str = "今天天气怎么样"
        await self.plugin.on_llm_request(ev, req)
        self.assertEqual(len(req.extra_user_content_parts), 1)

    async def test_落库到总结到检索到删除(self) -> None:
        ev = self.env.event(message="我平时只喝乌龙茶")

        await self.env.turn(ev, "我平时只喝乌龙茶", "记住了")
        st = await self.env.stats()
        self.assertEqual(st["messages"], 2, "一轮对话应落 user/assistant 两行")
        self.assertEqual(st["unsummarized"], 2)
        self.assertEqual(st["memories"], 0)

        # 后台总结：模型返回一条可解析的记忆
        self.env.model.replies = [
            '[{"kind": "preference", "content": "用户只喝乌龙茶", "owner": ""}]'
        ]
        note = await self.plugin.summarize_pass()
        self.assertIn("新增记忆 1 条", note)
        self.assertIn("标记 2 条消息", note)

        st = await self.env.stats()
        self.assertEqual(st["memories"], 1)
        self.assertEqual(st["unsummarized"], 0)

        # 检索：真的把记忆回给模型
        found = await self.plugin.tool_search_memory(ev, "乌龙茶")
        self.assertIn("乌龙茶", found)
        self.assertNotEqual(found, "没有找到相关记忆。")

        # 删除：编号来自检索结果
        ids = re.findall(r"#(\d+)", found)
        self.assertTrue(ids, f"检索结果里应有记忆编号，实际：{found!r}")
        gone = await self.plugin.tool_forget_memory(ev, int(ids[0]))
        self.assertIn("已删除", gone)
        self.assertEqual(await self.plugin.tool_search_memory(ev, "乌龙茶"),
                         "没有找到相关记忆。")

    async def test_两个账号的记忆互不可见(self) -> None:
        ev1 = self.env.event(platform_id="qq-01", sender_id="u1")
        ev2 = self.env.event(platform_id="qq-02", sender_id="u1")

        await self.plugin.tool_save_memory(ev1, "甲号的秘密是怕辣")
        await self.plugin.tool_save_memory(ev2, "乙号的秘密是爱吃甜")

        a = await self.plugin.tool_search_memory(ev1, "秘密")
        b = await self.plugin.tool_search_memory(ev2, "秘密")
        self.assertIn("怕辣", a)
        self.assertNotIn("爱吃甜", a)
        self.assertIn("爱吃甜", b)
        self.assertNotIn("怕辣", b)

        bots_root = os.path.join(self.env.db_root, "bots")
        dirs = [d for d in os.listdir(bots_root)
                if os.path.isfile(os.path.join(bots_root, d, "memory.db"))]
        self.assertEqual(len(dirs), 2, f"每个账号应各有一个库文件，实际 {dirs}")

    async def test_总结失败时不标记消息(self) -> None:
        ev = self.env.event(message="随便说一句")
        await self.env.turn(ev, "随便说一句", "嗯")

        # 半截 JSON：必须判定为失败，消息保持未总结，下一轮还能重试
        self.env.model.replies = ['[{"kind": "fact", "content": "半截输出"']
        note = await self.plugin.summarize_pass()
        self.assertIn("失败 1", note)

        st = await self.env.stats()
        self.assertEqual(st["memories"], 0)
        self.assertEqual(st["unsummarized"], 2)

    async def test_管理台路由全部注册(self) -> None:
        routes = self.env.register_web_apis()

        self.assertEqual(len(routes), 35, f"路由数量变了：{sorted(routes)}")
        overview = "/astrbot_plugin_hermes_fleet/overview"
        self.assertIn(overview, routes)
        self.assertEqual(routes[overview]["methods"], ["GET"])

        errors = [m for lvl, m in self.env.main.logger.records if lvl == "error"]
        self.assertEqual(errors, [], f"注册路由时不应吞异常：{errors}")


if __name__ == "__main__":
    unittest.main()
