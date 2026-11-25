"""联合测试：把两个独立开发的模块拼起来跑，验证它们对同一份契约的理解一致。

这个文件是「多 agent 并行开发」的验收现场。每个模块自己的测试只能证明
「我符合契约的字面」，只有拼起来才能发现「我们俩对同一句话的理解不同」。

这里拼的是 `hermes_memory`（存储）与 `hermes_recall`（重排）：
    写入 -> search_candidates（库内粗排）-> rerank（Python 精排）-> render（注入文本）

这三步正好对应线上注入链路的真实顺序。它今天能用内存实现跑，明天换成
Postgres 实现时，**这个文件一行都不用改**——这是分层的回报。
"""

from __future__ import annotations

import unittest

from hermes_contracts import (
    BotId,
    ManualClock,
    MemoryKey,
    MemoryScope,
    OwnerId,
    Visibility,
)
from hermes_contracts.conformance.fixtures import ALICE, BOB, BOT, GROUP
from hermes_memory import InMemoryMemoryStore
from hermes_recall import MAX_RERANK, rerank, render


class InjectPipelineTest(unittest.IsolatedAsyncioTestCase):
    """一条记忆从写入到进入注入文本的完整路径。"""

    async def asyncSetUp(self) -> None:
        self.clock = ManualClock()
        self.store = InMemoryMemoryStore(clock=self.clock)
        self.key = MemoryKey(BOT, MemoryScope.USER, owner_id=ALICE)

    async def _seed(self) -> None:
        from hermes_contracts import Memory, MemoryId

        for text in (
            "用户喜欢喝乌龙茶，尤其偏爱铁观音",
            "用户住在杭州西湖区，家里养了两只猫",
            "用户对花生过敏，点外卖要避开",
            "用户在做一个 AstrBot 的记忆插件",
        ):
            await self.store.add(
                Memory(
                    id=MemoryId(0),
                    bot_id=BOT,
                    content=text,
                    scope=MemoryScope.USER,
                    visibility=Visibility.PRIVATE,
                    owner_id=ALICE,
                    ts=self.clock.now(),
                )
            )

    async def test_end_to_end_query_hits_right_memory(self) -> None:
        await self._seed()
        candidates = await self.store.search_candidates(
            self.key, query="乌龙茶", viewer=OwnerId(ALICE), viewer_group="", limit=20
        )
        hits = rerank("乌龙茶", candidates, k=3, now=self.clock.now())

        self.assertTrue(hits, "有明确字面重合的记忆必须被召回")
        self.assertIn("乌龙茶", hits[0].memory.content, "最相关的一条应排第一")

        text = render(hits)
        self.assertIn("乌龙茶", text)
        # 编号必须出现在注入文本里，否则模型无法用 replaces=/forget id= 引用它
        self.assertRegex(text, r"\d+", "注入文本必须带可引用的编号")

    async def test_known_limitation_pure_semantic_query_misses(self) -> None:
        """**已知缺陷，故意固化在这里。**

        「我喝茶的偏好」召不回「用户喜欢喝乌龙茶」——两者只有「喝」一个字重合，
        没有连续二字重叠，bigram 打分归零，闸门直接拦掉。

        这正是 hermes_core README 里承认的天花板，也是新平台必须上向量检索的
        直接原因。写成通过的测试而不是删掉，是为了：
        - 让这条边界是**被记录的**，不是被忘记的；
        - 等 M4 接上向量召回后，这条会变红——那时应该把它改写成「能召回」，
          而不是删掉。

        断言「现在召不回」，比注释里写一句话有用得多。
        """
        await self._seed()
        candidates = await self.store.search_candidates(
            self.key, query="我喝茶的偏好", viewer=OwnerId(ALICE), viewer_group="", limit=20
        )
        self.assertEqual(
            rerank("我喝茶的偏好", candidates, k=3, now=self.clock.now()),
            [],
            "如果这条开始通过（召回了），说明检索能力已升级，请更新本测试的意图",
        )

    async def test_recall_never_leaks_other_users_memory(self) -> None:
        """端到端验证隐私边界：别人搜不到、也渲染不出来。"""
        await self._seed()
        candidates = await self.store.search_candidates(
            self.key, query="乌龙茶", viewer=OwnerId(BOB), viewer_group="", limit=20
        )
        self.assertEqual(list(candidates), [], "bob 不该搜到 alice 的私聊记忆")

    async def test_recall_never_leaks_across_bots(self) -> None:
        await self._seed()
        candidates = await self.store.search_candidates(
            MemoryKey(BotId("bot_beta"), MemoryScope.USER, owner_id=ALICE),
            query="乌龙茶",
            viewer=OwnerId(ALICE),
            viewer_group="",
            limit=20,
        )
        self.assertEqual(list(candidates), [], "bot_beta 不该看到 bot_alpha 的记忆")

    async def test_group_memory_reachable_inside_group_only(self) -> None:
        from hermes_contracts import Memory, MemoryId

        await self.store.add(
            Memory(
                id=MemoryId(0),
                bot_id=BOT,
                content="本群约定每周五交周报，交给值班人汇总",
                scope=MemoryScope.GROUP,
                visibility=Visibility.GROUP,
                group_id=GROUP,
                ts=self.clock.now(),
            )
        )
        inside = await self.store.search_candidates(
            MemoryKey(BOT, MemoryScope.GROUP, group_id=GROUP),
            query="周报",
            viewer=OwnerId(BOB),
            viewer_group=GROUP,
            limit=20,
        )
        self.assertTrue(inside)

        outside = await self.store.search_candidates(
            MemoryKey(BOT, MemoryScope.GROUP, group_id="group_other"),
            query="周报",
            viewer=OwnerId(BOB),
            viewer_group="group_other",
            limit=20,
        )
        self.assertEqual(list(outside), [])

    async def test_empty_query_does_not_crash_and_returns_nothing(self) -> None:
        """空/过短查询必须安静返回空，不能抛异常、也不能硬凑结果。

        「嗯」「在吗」占真实对话的很大比例，如果这里抛异常，注入钩子会每轮报错。
        """
        await self._seed()
        for q in ("", " ", "嗯"):
            candidates = await self.store.search_candidates(
                self.key, query=q, viewer=OwnerId(ALICE), viewer_group="", limit=20
            )
            self.assertEqual(rerank(q, candidates, k=3, now=self.clock.now()), [])

    async def test_rerank_rejects_oversized_candidate_pool(self) -> None:
        """候选超过上限要显式报错，而不是默默硬扛。

        这是防止「把全库捞进 Python 打分」那个病根被继承过来的最后一道闸：
        如果哪天有人把存储层改回「捞 3000 条」，这个测试会在本地就红。
        """
        await self._seed()
        candidates = await self.store.search_candidates(
            self.key, query="用户", viewer=OwnerId(ALICE), viewer_group="", limit=100
        )
        self.assertLessEqual(len(candidates), MAX_RERANK)
        with self.assertRaises(ValueError):
            rerank("用户", list(candidates) * 20, k=3, now=self.clock.now())

    async def test_pipeline_survives_empty_store(self) -> None:
        candidates = await self.store.search_candidates(
            self.key, query="任何问题", viewer=OwnerId(ALICE), viewer_group="", limit=20
        )
        self.assertEqual(list(candidates), [])
        self.assertEqual(render(rerank("任何问题", candidates, k=3, now=self.clock.now())), "")


if __name__ == "__main__":
    unittest.main()
