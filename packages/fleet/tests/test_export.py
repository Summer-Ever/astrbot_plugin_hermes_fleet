"""记忆导出：两份文档内容 + 隐私边界 + 导入往返。

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
    IMPORT_OPEN, MAX_EXPORT_BYTES, check_export_size, export_filename,
    pair_replies, parse_memory_config, render_chat, render_memory_config,
    stale_exports,
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


class TestRenderChat(unittest.TestCase):
    def test_原文带年月日且一条一行(self) -> None:
        doc = render_chat(
            bot_id="qq-01", owner_id="alice", note="客服号",
            messages=[
                _msg("你好", ts=1764000000),
                _msg("好的", role="assistant", mid=2, ts=1764000000),
            ],
            now=1764000000, version="0.9.0",
        )
        self.assertIn("# 对话原文 · 客服号（qq-01）", doc)
        self.assertIn("[2025-11-25 ", doc)  # 年月日必须出现，不能只有时分
        self.assertIn("你：你好", doc)
        self.assertIn("机器人：好的", doc)
        self.assertIn("## 私聊", doc)

    def test_原文全量不截断(self) -> None:
        body = "一字" * 1000  # 2000 字，远超旧的 800 字截断
        doc = render_chat(bot_id="qq-01", owner_id="alice",
                          messages=[_msg(body)])
        self.assertIn(body, doc)
        self.assertNotIn("文档内截断", doc)

    def test_空原文给出说明(self) -> None:
        doc = render_chat(bot_id="qq-01", owner_id="alice")
        self.assertIn("（没有聊天原文。）", doc)

    def test_群按群号分段(self) -> None:
        doc = render_chat(bot_id="qq-01", owner_id="alice",
                          messages=[_msg("大家好", group="g-1", session="g")])
        self.assertIn("## 群 g-1", doc)


class TestRenderConfig(unittest.TestCase):
    def test_常驻归档齐全并带导入块(self) -> None:
        doc = render_memory_config(
            bot_id="qq-01", owner_id="alice", note="客服号",
            resident=[_resident(ResidentBlock.ADDRESS, "阿明")],
            memories=[_mem("在做客服")],
            now=1764000000, version="0.9.0",
        )
        self.assertIn("# 记忆配置 · 客服号（qq-01）", doc)
        self.assertIn(IMPORT_OPEN, doc)
        self.assertIn("## 一、常驻记忆", doc)
        self.assertIn("## 二、归档记忆", doc)
        self.assertIn("怎么称呼用户", doc)
        self.assertNotIn("| 称呼 |", doc)
        self.assertIn("`#7`", doc)
        self.assertNotIn("## 三、聊天原文", doc)  # 原文已经不在这份里

    def test_空数据不出现空白章节(self) -> None:
        doc = render_memory_config(bot_id="qq-01", owner_id="alice")
        self.assertIn("（没有设置常驻词条。）", doc)
        self.assertIn("（没有归档记忆。）", doc)

    def test_长文本压成一行且转义竖线(self) -> None:
        doc = render_memory_config(
            bot_id="qq-01", owner_id="alice",
            resident=[_resident(ResidentBlock.RULES, "a|b\nc")],
        )
        self.assertIn("a\\|b c", doc)

    def test_导入块里中文不转义(self) -> None:
        doc = render_memory_config(
            bot_id="qq-01", owner_id="alice",
            resident=[_resident(ResidentBlock.ADDRESS, "阿明")],
        )
        self.assertIn('"text":"阿明"', doc)
        self.assertNotIn("\\u963f", doc)


class TestParseImport(unittest.TestCase):
    def test_往返一致(self) -> None:
        resident = [
            _resident(ResidentBlock.ADDRESS, "阿明"),
            _resident(ResidentBlock.RULES, "不发广告", group="g-1"),
        ]
        memories = [_mem("在做客服"), _mem("群记忆", mid=8, group="g-1")]
        doc = render_memory_config(
            bot_id="qq-01", owner_id="alice", note="客服号",
            resident=resident, memories=memories, now=1764000000,
        )
        got = parse_memory_config(doc)
        self.assertEqual(got["bot_id"], "qq-01")
        self.assertEqual(got["owner_id"], "alice")
        self.assertEqual(got["note"], "客服号")
        self.assertEqual(
            [(r["scope"], r["block"], r["text"]) for r in got["resident"]],
            [("private", "address", "阿明"), ("group:g-1", "rules", "不发广告")],
        )
        self.assertEqual([m["content"] for m in got["memories"]],
                         ["在做客服", "群记忆"])
        self.assertEqual(got["memories"][1]["group_id"], "g-1")

    def test_人的排版改动不影响解析(self) -> None:
        doc = render_memory_config(bot_id="qq-01", owner_id="alice",
                                   resident=[_resident(ResidentBlock.USER, "28 岁")])
        doc = doc.replace("# 记忆配置", "# 我改过的标题").replace("## 概览", "随便写")
        self.assertEqual(parse_memory_config(doc)["resident"][0]["text"], "28 岁")

    def test_没有导入块直接报错(self) -> None:
        with self.assertRaises(ValueError):
            parse_memory_config("# 随便一个文件")

    def test_坏JSON报错(self) -> None:
        bad = f"{IMPORT_OPEN}\n{{not json}}\n-->"
        with self.assertRaises(ValueError):
            parse_memory_config(bad)

    def test_未知词条报错(self) -> None:
        doc = render_memory_config(
            bot_id="qq-01", owner_id="alice",
            resident=[_resident(ResidentBlock.ADDRESS, "阿明")],
        ).replace('"block":"address"', '"block":"unknown"')
        with self.assertRaises(ValueError):
            parse_memory_config(doc)

    def test_空群号报错(self) -> None:
        doc = render_memory_config(
            bot_id="qq-01", owner_id="alice",
            resident=[_resident(ResidentBlock.RULES, "x", group="g-1")],
        ).replace('"scope":"group:g-1"', '"scope":"group:"')
        with self.assertRaises(ValueError):
            parse_memory_config(doc)

    def test_超长归档记忆报错(self) -> None:
        doc = render_memory_config(bot_id="qq-01", owner_id="alice",
                                   memories=[_mem("字" * 501)])
        with self.assertRaises(ValueError):
            parse_memory_config(doc)


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
        self.assertNotIn("bob 的秘密", render_chat(
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

    def test_两份文件前缀不同(self) -> None:
        chat = export_filename("qq-01", "alice", ts=0, kind="chat")
        cfg = export_filename("qq-01", "alice", ts=0, kind="memory")
        self.assertTrue(chat.startswith("hermes-chat-"))
        self.assertTrue(cfg.startswith("hermes-memory-"))
        self.assertNotEqual(chat, cfg)

    def test_中文账号名不会让附件名变问号(self) -> None:
        name = export_filename("客服号", "张三", ts=0)
        self.assertTrue(all(ord(c) < 128 for c in name), name)


class TestStaleExports(unittest.TestCase):
    def test_按保留个数与天数删旧的(self) -> None:
        entries = [("new.md", 1000.0), ("mid.md", 900.0), ("old.md", 10.0)]
        stale = stale_exports(entries, now=1000.0, keep_days=1, max_files=2)
        self.assertEqual(stale, ["old.md"])

    def test_txt也参与清理(self) -> None:
        entries = [("a.md", 1000.0), ("b.txt", 900.0), ("old.md", 10.0)]
        stale = stale_exports(entries, now=1000.0, keep_days=1, max_files=2)
        self.assertEqual(stale, ["old.md"])


class TestSizeLimit(unittest.TestCase):
    def test_上限是100MB(self) -> None:
        self.assertEqual(MAX_EXPORT_BYTES, 100 * 1024 * 1024)

    def test_按UTF8字节算而非字符数(self) -> None:
        over, size = check_export_size("字" * 100)
        self.assertFalse(over)
        self.assertEqual(size, 300)  # 中文 3 字节

    def test_超限时返回True(self) -> None:
        big = "x" * (MAX_EXPORT_BYTES + 1)
        over, size = check_export_size(big)
        self.assertTrue(over)
        self.assertGreater(size, MAX_EXPORT_BYTES)


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
        chat = render_chat(
            bot_id=data["bot_id"], owner_id=data["owner_id"],
            messages=data["messages"], message_total=data["message_total"],
            now=1764000000,
        )
        cfg = render_memory_config(
            bot_id=data["bot_id"], owner_id=data["owner_id"],
            resident=data["resident"], memories=data["memories"],
            memory_total=data["memory_total"], now=1764000000,
        )
        self.assertIn("阿明", cfg)
        self.assertIn("alice 在杭州", cfg)
        self.assertIn("我的私聊", chat)
        for secret in ("鲍勃", "bob 的秘密", "bob 说的话", "charlie 说的话",
                       "回复 bob", "回复 charlie"):
            self.assertNotIn(secret, chat)
            self.assertNotIn(secret, cfg)

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
        self.assertEqual(data["message_total"], 2)

    async def test_全量导出不截断(self) -> None:
        alice = await self._scene("alice")
        # 造 100 轮对话，旧的 3000 条上限不至于触发，但验证导出走全量路径
        for i in range(100):
            await self.svc.capture_turn(alice, "s", f"第{i}句", f"回复{i}")
        data = await self.svc.export_owner(platform_id="qq-01", owner_id="alice")
        self.assertEqual(len(data["messages"]), 200)
        self.assertIn("第99句", [m.content for m in data["messages"]])

    async def test_空owner与未知账号返回空包(self) -> None:
        for owner, pid in (("", "qq-01"), ("alice", "never-seen")):
            data = await self.svc.export_owner(platform_id=pid, owner_id=owner)
            self.assertEqual(data["messages"], [])
            self.assertEqual(data["memories"], [])
            self.assertEqual(data["resident"], [])


if __name__ == "__main__":
    unittest.main()
