"""本机向量模型的安装 / 启停 / 日志测试。**不联网、不起真进程。**

`local_embed` 的每一步都能被替身替换：下载器、进程工厂、健康检查。
这样「sha256 校验失败会删掉坏文件」「进程退出后状态变 error」「日志能增量
拉取」这些最容易在线上才暴露的路径，都可以在 CI 里被穷举。
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import shutil
import tempfile
import time
import unittest
import zipfile

from hermes_fleet.local_embed import (
    LOCAL_MODEL_NAME,
    MODEL_SPEC,
    RUNTIME_SPEC,
    DownloadSpec,
    LocalEmbedder,
    LocalEmbedError,
    LocalEmbedManager,
)
from hermes_fleet.vector import VectorSettings
from hermes_vectors import EmbeddingError


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _runtime_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("llama-b12359-bin-win-cpu-x64/llama-server.exe", b"MZ-fake")
        zf.writestr("llama-b12359-bin-win-cpu-x64/ggml.dll", b"dll")
    return buf.getvalue()


class _Proc:
    """假进程：stdout 是一段固定文本，`poll()` 返回退出码。"""

    def __init__(self, lines=(), code: int | None = None) -> None:
        self.pid = 4321
        self.stdout = io.StringIO("\n".join(lines))
        self._code = code
        self.terminated = False

    def poll(self):
        return self._code

    def terminate(self) -> None:
        self.terminated = True
        self._code = 0

    def wait(self, timeout=None):
        return self._code if self._code is not None else 0

    def kill(self) -> None:
        self._code = -9


class Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="fleet-local-embed-")
        self.rt_bytes = _runtime_zip()
        self.model_bytes = b"GGUF-fake-model"
        self.runtime_spec = DownloadSpec(
            "runtime", "运行库", "http://test/rt.zip", "rt.zip",
            _sha(self.rt_bytes), unpack=True)
        self.model_spec = DownloadSpec(
            "model", "模型", "http://test/model.gguf", "model.gguf",
            _sha(self.model_bytes))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def manager(self, **kw) -> LocalEmbedManager:
        payload = {
            "http://test/rt.zip": self.rt_bytes,
            "http://test/model.gguf": self.model_bytes,
        }

        def fetch(url, dest, report):
            data = payload[url]
            report(0, len(data))
            with open(dest, "wb") as fp:
                fp.write(data)
            report(len(data), len(data))

        kw.setdefault("fetch", fetch)
        kw.setdefault("runtime_spec", self.runtime_spec)
        kw.setdefault("model_spec", self.model_spec)
        kw.setdefault("sleep", lambda _s: None)
        return LocalEmbedManager(self.tmp, **kw)

    def fake_install(self, man: LocalEmbedManager) -> None:
        """直接把「已下载」落成文件，跳过下载流程。"""
        exe_dir = os.path.join(man.runtime_dir, "rt")
        os.makedirs(exe_dir, exist_ok=True)
        with open(os.path.join(exe_dir, "llama-server.exe"), "wb") as fp:
            fp.write(b"MZ")
        os.makedirs(man.model_dir, exist_ok=True)
        with open(man.model_path, "wb") as fp:
            fp.write(self.model_bytes)


class TestSpecs(unittest.TestCase):
    def test_哈希是合法的_64_位十六进制(self) -> None:
        """抄 sha256 时多一位/少一位，下载永远校验失败，且报错只显示前 12 位。

        真实事故：运行库的常量多了一位，页面一直提示「校验失败」而前缀
        看起来完全正确。这里直接把长度和字符集钉死。
        """
        for spec in (RUNTIME_SPEC, MODEL_SPEC):
            with self.subTest(spec=spec.key):
                self.assertRegex(spec.sha256, r"^[0-9a-f]{64}$")


class TestInstall(Base):
    def test_下载_校验_解压并自动启动(self) -> None:
        proc = _Proc(["listening on http://127.0.0.1:9999"])
        man = self.manager(popen=lambda *a, **kw: proc, healthy=lambda port: True)
        man.install(wait=True)

        status = man.status()
        self.assertTrue(status["installed"])
        self.assertTrue(status["runtime_installed"])
        self.assertTrue(status["model_installed"])
        self.assertTrue(status["ready"], "安装完成后应当自动启动")
        self.assertEqual(status["state"], "ready")
        self.assertTrue(os.path.isfile(man.model_path))
        self.assertTrue(man.runtime_exe.endswith("llama-server.exe"))
        self.assertFalse(os.path.exists(
            os.path.join(man.root, self.runtime_spec.filename)),
            "解压后的压缩包不该留在目录里")

    def test_校验失败会删掉坏文件(self) -> None:
        bad = DownloadSpec("model", "模型", "http://test/model.gguf",
                           "model.gguf", "0" * 64)
        man = self.manager(model_spec=bad, popen=lambda *a, **kw: _Proc())
        man.install(wait=True)

        status = man.status()
        self.assertFalse(status["installed"])
        self.assertEqual(status["state"], "error")
        self.assertIn("校验失败", status["last_error"])
        self.assertFalse(os.path.exists(man.model_path), "坏模型不能留下")

    def test_未安装时启动会给出明确提示(self) -> None:
        man = self.manager()
        self.assertFalse(man.installed)
        with self.assertRaises(LocalEmbedError) as ctx:
            man.ensure_ready(1.0)
        self.assertIn("下载", str(ctx.exception))
        self.assertFalse(os.path.exists(man.root),
                         "只读状态检查不该顺手建目录")


class TestRun(Base):
    def test_就绪_日志与配置(self) -> None:
        proc = _Proc([
            "llama_model_loader: loaded",
            "listening on http://127.0.0.1:9999",
        ])
        man = self.manager(popen=lambda *a, **kw: proc, healthy=lambda port: True)
        self.fake_install(man)

        self.assertTrue(man.ensure_ready(5.0))
        status = man.status()
        self.assertTrue(status["ready"])
        self.assertGreater(status["port"], 0)
        self.assertEqual(status["pid"], 4321)
        self.assertEqual(status["model"], LOCAL_MODEL_NAME)

        deadline = time.time() + 2
        texts = ""
        while time.time() < deadline:
            texts = " ".join(x["line"] for x in man.logs())
            if "listening" in texts:
                break
            time.sleep(0.02)
        self.assertIn("listening", texts, "进程输出应进入日志环形缓冲")

        cfg = man.embedding_config()
        self.assertEqual(cfg.model, LOCAL_MODEL_NAME)
        self.assertTrue(cfg.base_url.startswith("http://127.0.0.1:"))
        self.assertTrue(cfg.base_url.endswith("/v1"))

    def test_进程退出后状态变错误(self) -> None:
        man = self.manager(popen=lambda *a, **kw: _Proc(code=1),
                           healthy=lambda port: True)
        self.fake_install(man)
        with self.assertRaises(LocalEmbedError):
            man.ensure_ready(3.0)
        status = man.status()
        self.assertFalse(status["ready"])
        self.assertEqual(status["state"], "error")
        self.assertIn("code=1", status["last_error"])

    def test_停止会收掉子进程(self) -> None:
        proc = _Proc()
        man = self.manager(popen=lambda *a, **kw: proc, healthy=lambda port: True)
        self.fake_install(man)
        man.ensure_ready(5.0)
        man.stop()
        self.assertTrue(proc.terminated)
        self.assertFalse(man.ready)
        self.assertEqual(man.status()["state"], "stopped")

    def test_日志可以增量拉取(self) -> None:
        man = self.manager()
        man._append("第一行")
        man._append("第二行")
        all_lines = man.logs()
        self.assertEqual([x["line"] for x in all_lines], ["第一行", "第二行"])
        # `limit` 保留的是**最新**若干行，增量拉取靠 after=已收到的最大 seq。
        self.assertEqual([x["line"] for x in man.logs(after=0, limit=1)], ["第二行"])
        rest = man.logs(after=all_lines[0]["seq"])
        self.assertEqual([x["line"] for x in rest], ["第二行"])

    def test_清理下载残留(self) -> None:
        man = self.manager()
        os.makedirs(man.root, exist_ok=True)
        part = os.path.join(man.root, "x.part")
        with open(part, "wb") as fp:
            fp.write(b"half")
        self.assertEqual(man.cleanup_partials(), 1)
        self.assertFalse(os.path.exists(part))

    def test_就绪后写下进程记录(self) -> None:
        """宿主被强杀时，这份记录是下次启动清理遗留进程的唯一线索。"""
        man = self.manager(popen=lambda *a, **kw: _Proc(),
                           healthy=lambda port: True)
        self.fake_install(man)
        man.ensure_ready(5.0)
        state = man._read_state()
        self.assertEqual(state.get("pid"), 4321)
        self.assertGreater(state.get("port") or 0, 0)
        man.stop()
        self.assertFalse(os.path.exists(man.state_path))


class TestGuard(Base):
    def test_用户停止后守护不再拉起(self) -> None:
        man = self.manager(popen=lambda *a, **kw: _Proc(),
                           healthy=lambda port: True)
        self.fake_install(man)
        man.ensure_ready(5.0)
        man.stop(user=True)

        self.assertTrue(man.status()["user_stopped"])
        self.assertFalse(man.ensure_alive(1.0), "用户点过停止，守护不能自作主张")

    def test_启动会清除用户停止标记(self) -> None:
        man = self.manager(popen=lambda *a, **kw: _Proc(),
                           healthy=lambda port: True)
        self.fake_install(man)
        man.stop(user=True)
        man.start(5.0)
        self.assertFalse(man.status()["user_stopped"])
        self.assertTrue(man.ready)

    def test_遗留进程_pid_被别的程序复用时不动它(self) -> None:
        """pid 会被复用：认出不是 llama-server 就只能清记录，不能杀。"""
        man = self.manager()
        os.makedirs(man.root, exist_ok=True)
        man._write_state(os.getpid(), 1)  # 当前 Python 进程，一定活着
        man._reap_stale()
        self.assertFalse(os.path.exists(man.state_path))

    def test_遗留进程已不存在时只清记录(self) -> None:
        man = self.manager()
        os.makedirs(man.root, exist_ok=True)
        man._write_state(999_999_999, 1)
        man._reap_stale()
        self.assertFalse(os.path.exists(man.state_path))


class TestLocalWiring(Base):
    async def test_local_来源默认启用且模型名固定(self) -> None:
        cfg = VectorSettings(provider="local")
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.model_name, LOCAL_MODEL_NAME)

    async def test_模型未就绪时适配层抛_EmbeddingError(self) -> None:
        man = self.manager()
        emb = LocalEmbedder(man)
        self.assertFalse(emb.enabled)
        with self.assertRaises(EmbeddingError):
            await emb.embed(["你好"])

    async def test_就绪后适配层走本机接口(self) -> None:
        man = self.manager(popen=lambda *a, **kw: _Proc(),
                           healthy=lambda port: True)
        self.fake_install(man)
        await asyncio.to_thread(man.ensure_ready, 5.0)
        emb = LocalEmbedder(man)
        self.assertTrue(emb.enabled)
        cfg = man.embedding_config(batch_size=3, timeout_s=5)
        self.assertEqual(cfg.batch_size, 3)


if __name__ == "__main__":
    unittest.main()
