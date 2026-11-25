"""记忆导出：文档内容 + 隐私边界。

导出是这套系统里唯一一处把**原始对话**写进文件并发到聊天里的路径，
所以这里的断言比其他模块更硬：宁可少导一条，也不能把群友的话发给
提问的人。纯渲染用假数据，过滤逻辑用真实 SQLite 跑一遍。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for p in ("contracts", "recall", "storage_sqlite", "fleet"):
    sys.path.insert(0, str(_ROOT / "packages" / p / "src"))

from hermes_contracts import (  # noqa: E402
    BotId, Memory, MemoryId, MemoryScope, Message, OwnerId,
    ResidentBlock, ResidentEntry, ResidentScope, SessionId, Visibility,
)
from hermes_fleet import BotRegistry, MemoryService  # noqa: E402
from hermes_fleet.export import (  # noqa: E402
    export_filename, pair_replies, render_document, stale_exports,
)


def _msg(text: str, *, owner: str = "alice", role: str = "user",
         session: str = "s1", group: str = "", ts: float = 100.0,
         mid: int = 1) -> Message:
    return Message(
        bot_id=BotId("qq-01"), session_id=SessionId(session), role=role,
        content=text, ts=ts, id=mid,
        sender_id=owner if role == "user" else "", group_id=group,
    )


def _mem(text: str, *, mid: int = 7, group: str = "") -> Memory:
    return Memory(
        id=MemoryId(mid), bot_id=BotId("qq-01"), content=text,
        scope=MemoryScope.USER, visibility=Visibility.PRIVATE,
        owner_id="alice", group_id=group, kind="fact", ts=100.0,
    )


def _resident(block: ResidentBlock, text: str, *, group: str = "") -> ResidentEntry:
    return ResidentEntry(
        bot_id=BotId("qq-01"), owner_id=OwnerId("alice"),
        scope=(ResidentScope.for_group(group) if group else ResidentScope.for_private()),
        block=block, text=text, updated_ts=100.0,
    )


class TestRender(unittest.TestCase):
    def test_三段齐全且带主语标签(self) -> None:
        doc = render_document(
            bot_id="qq-01", owner_id="alice", note="客服号",
            resident=[_resident(ResidentBlock.ADDRESS, "阿明")],
            memories=[_mem("在做客服")],
            messages=[_msg("你好")],
            now=1764000000,
        )
        self.assertIn("# 记忆导出 · 客服号（qq-01）", doc)
        self.assertIn("## 一、常驻记忆", doc)
        self.assertIn("## 二、归档记忆", doc)
        self.assertIn("## 三、聊天原文", doc)
        self.assertIn("怎么称呼用户", doc)
        self.assertNotIn("| 称呼 |", doc)  # 短标签有两读歧义
        self.assertIn("`#7`", doc)
        self.assertIn("**08:01 你**：你好", doc)

    def test_空数据不出现空白章节(self) -> None:
        doc = render_document(bot_id="qq-01", owner_id="alice")
        self.assertIn("（没有设置常驻词条。）", doc)
        self.assertIn("（没有归档记忆。）", doc)
        self.assertIn("（没有聊天原文。）", doc)

    def test_长文本压成一行且转义竖线(self) -> None:
        doc = render_document(
            bot_id="qq-01", owner_id="alice",
            resident=[_resident(ResidentBlock.RULES, "a|b\nc")],
        )
        self.assertIn("a\\|b c", doc)
        self.assertNotIn("\n", "a\\|b c")

    def test_超长原文注明截断(self) -> None:
        doc = render_document(
            bot_id="qq-01", owner_id="alice",
            messages=[_msg("字" * 100)], max_chat_chars=20,
        )
        self.assertIn("原文 100 字，文档内截断", doc)

    def test_群常驻按群分组(self) -> None:
        doc = render_document(
            bot_id="qq-01", owner_id="alice",
            resident=[_resident(ResidentBlock.RULES, "不发广告", group="g-1")],
            messages=[_msg("大家好", group="g-1")],
        )
        self.assertIn("### 群 g-1", doc)
        self.assertIn("规则与避雷", doc)

    def test_概览写明截断数量(self) -> None:
        doc = render_document(
            bot_id="qq-01", owner_id="alice",
            memories=[_mem("最近一条")], memory_total=99,
            messages=[_msg("最近一句")], message_total=1234,
        )
        self.assertIn("归档记忆 | 99 条（本次写入 1 条）", doc)
        self.assertIn("聊天原文 | 1234 条（本次导出 1 条）", doc)


class TestPairReplies(unittest.TestCase):
    def test_群友发言与跟在后面的机器人回复都丢掉(self) -> None:
        rows = [
            _msg("我的是 alice", mid=1, group="g", session="g"),
            _msg("好的 alice", role="assistant", mid=2, group="g", session="g"),
            _msg("bob 的秘密", owner="bob", mid=3, group="g", session="g"),
            _msg("好的 bob", role="assistant", mid=4, group="g", session="g"),
            _msg("我也要", mid=5, group="g", session="g"),
            _msg("好的 alice", role="assistant", mid=6, group="g", session="g"),
        ]
        kept = pair_replies(rows, "alice")
        self.assertEqual([r.id for r in kept], [1, 2, 5, 6])
        self.assertNotIn("bob 的秘密", render_document(
            bot_id="qq-01", owner_id="alice", messages=kept,
        ))

    def test_没有本人发言时不凭空带机器人回复(self) -> None:
        rows = [_msg("你好", role="assistant", mid=1)]
        self.assertEqual(pair_replies(rows, "alice"), [])

    def test_system_行不导出(self) -> None:
        rows = [
            _msg("你好", mid=1),
            _msg("注入的系统提示", role="system", mid=2),
        ]
        self.assertEqual([r.id for r in pair_replies(rows, "alice")], [1])


class TestFilename(unittest.TestCase):
    def test_路径分隔符与点点被洗掉(self) -> None:
        name = export_filename("../../etc/passwd", "a/b\\c", ts=0)
        self.assertNotIn("/", name)
        self.assertNotIn("\\", name)
        self.assertNotIn("..", name)
        self.assertTrue(name.endswith(".md"))

    def test_中文账号名不会让附件名变问号(self) -> None:
        name = export_filename("客服号", "张三", ts=0)
        self.assertTrue(all(ord(c) < 128 for c in name), name)


class TestStaleExports(unittest.TestCase):
    def test_按保留个数与天数删旧的(self) -> None:
        entries = [("new.md", 1000.0), ("mid.md", 900.0), ("old.md", 10.0)]
        stale = stale_exports(entries, now=1000.0, keep_days=1, max_files=2)
        self.assertEqual(stale, ["old.md"])

    def test_刚生成的文件不会因系统时钟回拨被删(self) -> None:
        entries = [("fresh.md", 2000.0), ("old.md", 100.0)]
        stale = stale_exports(entries, now=1000.0, keep_days=1, max_files=60)
        self.assertNotIn("fresh.md", stale)


class ExportBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="fleet-export-")
        self.reg = BotRegistry(os.path.join(self.tmp, "bots"), max_open=4)
        self.svc = MemoryService(self.reg)

    def tearDown(self) -> None:
        self.reg.close_all()
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestExportData(ExportBase):
    async def _scene(self, owner: str, group: str = ""):
        return await self.svc.scene("qq-01", owner, group, is_group=bool(group))

    async def test_只导出本人三类数据(self) -> None:
        alice = await self._scene("alice")
        bob = await self._scene("bob", "g-1")
        charlie = await self._scene("charlie", "g-1")

        await self.svc.set_resident(alice, "address", "阿明")
        await self.svc.set_resident(bob, "address", "鲍勃")
        await self.svc.save(alice, "alice 在杭州")
        await self.svc.save(bob, "bob 的秘密")
        await self.svc.capture_turn(alice, "private-session", "我的私聊", "回复 alice")
        await self.svc.capture_turn(bob, "group-session", "bob 说的话", "回复 bob")
        await self.svc.capture_turn(charlie, "group-session", "charlie 说的话", "回复 charlie")

        data = await self.svc.export_owner(platform_id="qq-01", owner_id="alice")
        doc = render_document(
            bot_id=data["bot_id"], owner_id=data["owner_id"],
            resident=data["resident"], memories=data["memories"],
            messages=data["messages"],
            memory_total=data["memory_total"],
            message_total=data["message_total"],
            now=1764000000,
        )
        self.assertIn("阿明", doc)
        self.assertIn("alice 在杭州", doc)
        self.assertIn("我的私聊", doc)
        for secret in ("鲍勃", "bob 的秘密", "bob 说的话", "charlie 说的话",
                       "回复 bob", "回复 charlie"):
            self.assertNotIn(secret, doc)

    async def test_群会话里本人的发言与机器人回复都在(self) -> None:
        alice = await self._scene("alice", "g-1")
        other = await self._scene("bob", "g-1")
        await self.svc.capture_turn(alice, "shared-session", "alice 在群里", "回复 alice")
        await self.svc.capture_turn(other, "shared-session", "bob 在群里", "回复 bob")
        data = await self.svc.export_owner(platform_id="qq-01", owner_id="alice")
        texts = [m.content for m in data["messages"]]
        self.assertIn("alice 在群里", texts)
        self.assertIn("回复 alice", texts)
        self.assertNotIn("bob 在群里", texts)
        self.assertNotIn("回复 bob", texts)
        # 窗口没吃满时总数要精确：不能把回复群友的那条算进「我的原文」
        self.assertEqual(data["message_total"], 2)

    async def test_空owner与未知账号返回空包(self) -> None:
        for owner, pid in (("", "qq-01"), ("alice", "never-seen")):
            data = await self.svc.export_owner(platform_id=pid, owner_id=owner)
            self.assertEqual(data["messages"], [])
            self.assertEqual(data["memories"], [])
            self.assertEqual(data["resident"], [])


if __name__ == "__main__":
    unittest.main()
