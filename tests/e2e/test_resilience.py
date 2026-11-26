"""韧性链路：并发写入、打开上限（LRU）、关库重开。

这三件事的共同点是「单测里看不出来」：它们依赖真实的连接生命周期与
锁顺序，只有在 `main.py` 装配出的实例上跑才会暴露。典型故障是并发写
偶发 `database is locked`、超过 `max_open_bots` 后 LRU 关错了库、或者
重开时索引与文件对不上导致记忆「消失」。
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from harness.env import PluginEnv, load_main  # noqa: E402


class ResilienceTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_main()

    async def asyncSetUp(self) -> None:
        # 批量建库/并发写会跨过 asyncio debug 默认的 0.1s 阈值；这里不是
        # 卡顿，调高阈值避免刷误导性的慢任务日志。
        asyncio.get_running_loop().slow_callback_duration = 5.0

    async def test_两个账号并发写入不丢不串(self) -> None:
        env = PluginEnv()
        try:
            tasks = []
            for pid in ("qq-01", "qq-02"):
                label = pid.replace("-", "")
                for i in range(10):
                    ev = env.event(platform_id=pid, sender_id="u1")
                    tasks.append(
                        env.plugin.tool_save_memory(
                            ev, f"账号 {pid} 的专属编号是 {label}M{i}Q"
                        )
                    )
            results = await asyncio.gather(*tasks)
            self.assertTrue(all("已保存" in r for r in results), results)

            self.assertEqual((await env.stats("qq-01"))["memories"], 10)
            self.assertEqual((await env.stats("qq-02"))["memories"], 10)

            # 用只属于一个账号的编号检索：既验证「没丢」，也验证「没串」。
            a = await env.plugin.tool_search_memory(
                env.event(platform_id="qq-01", sender_id="u1"), "qq01M9Q"
            )
            b = await env.plugin.tool_search_memory(
                env.event(platform_id="qq-02", sender_id="u1"), "qq02M9Q"
            )
            self.assertIn("qq01M9Q", a)
            self.assertNotIn("qq02M9Q", a)
            self.assertIn("qq02M9Q", b)
            self.assertNotIn("qq01M9Q", b)
        finally:
            await env.close()

    async def test_账号数超过上限时按_LRU_关库(self) -> None:
        env = PluginEnv(config={"max_open_bots": 2})
        try:
            for i in range(1, 5):
                await env.plugin.svc.bot_stats(f"qq-0{i}")
                st = env.plugin.registry.stats()
                self.assertLessEqual(
                    st["opened"], 2, f"打开数超上限：{st['bots_opened']}"
                )
            self.assertEqual(len(env.plugin.registry.accounts()), 4)
            st = env.plugin.registry.stats()
            self.assertEqual(len(st["files_on_disk"]), 4)
        finally:
            await env.close()

    async def test_关库重开后记忆还在(self) -> None:
        env1 = PluginEnv()
        db_root = env1.db_root
        try:
            ev = env1.event(platform_id="qq-09", sender_id="u9")
            await env1.plugin.tool_save_memory(ev, "重开之后还要记得：钥匙在抽屉里")
        finally:
            await env1.close(keep=True)

        env2 = PluginEnv(db_root=db_root)
        try:
            ev = env2.event(platform_id="qq-09", sender_id="u9")
            found = await env2.plugin.tool_search_memory(ev, "钥匙在抽屉里")
            self.assertIn("钥匙在抽屉里", found)
        finally:
            await env2.close()
            env1.cleanup()


if __name__ == "__main__":
    unittest.main()
