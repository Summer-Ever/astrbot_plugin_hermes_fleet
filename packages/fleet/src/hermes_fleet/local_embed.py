"""本机向量模型：下载、校验、起服务、收日志。

## 为什么不是「pip install 一个模型库」

插件跑在用户的 AstrBot 进程里，装错一次原生依赖就直接加载失败，连字面
记忆一起失去。这里换成**外部进程 + HTTP** 的方案：

1. 运行库是 llama.cpp 的官方 CPU 版（Windows x64 压缩包，约 25MB）；
2. 模型是 bge-small-zh-v1.5 的 q8_0 gguf（约 26MB，512 维，中文）；
3. 两者都**不进仓库、不打包进插件**，第一次由用户在页面上点「下载」；
4. 下载完按 sha256 校验 → 解压 → 选一个空闲端口起 `llama-server`，
   暴露 OpenAI 兼容的 `/v1/embeddings`。

于是插件本体仍然零原生依赖，模型进程崩了也只是向量召回降级，不会把
记忆功能带走。CPU 推理在 i5-1240P 上单条约 10ms、10 条合批约 100ms，
对「检索时补几条语义记忆」这个用途足够。

## 线程模型

- 下载与解压跑在单独的后台线程（页面上能看到进度）；
- 进程的 stdout/stderr 由一条读线程收进环形缓冲，页面增量拉取；
- 对外所有方法都可以从任意线程调用，内部用锁保护。

## 明确不做的事

- 不做断点续传：两个文件加起来 50MB，重新下载比维护续传状态更省心；
- 不自动挑选 GPU：CPU 版最稳，显存与驱动问题不值得押在「能不能起来」上；
- 不在状态查询里算 sha256：26MB 每次页面刷新都算一遍会让状态页发烫。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass
from collections.abc import Callable

from hermes_vectors import EmbeddingConfig, EmbeddingError

# 模型标识会写进向量库的 meta：换模型必须整体重建，所以这里和
# `VectorSettings.model_name` 用的是同一个常量。
LOCAL_MODEL_NAME = "bge-small-zh-v1.5-q8_0"
LOCAL_MODEL_DIM = 512

_MAX_LOG_LINES = 800
_MAX_LOG_CHARS = 600
_DOWNLOAD_CHUNK = 256 * 1024


class LocalEmbedError(RuntimeError):
    """本机模型的运维错误。消息面向人，可直接显示在页面上。"""


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """一个待下载文件。`sha256` 为空表示不校验（只给测试用）。"""

    key: str
    title: str
    url: str
    filename: str
    sha256: str
    unpack: bool = False


RUNTIME_SPEC = DownloadSpec(
    key="runtime",
    title="llama.cpp 运行库（CPU x64）",
    url=("https://github.com/ggml-org/llama.cpp/releases/download/b12359/"
         "llama-b12359-bin-win-cpu-x64.zip"),
    filename="llama-b12359-bin-win-cpu-x64.zip",
    sha256="837c1655e87df3b66989f4be33a792a1399c785bf259038dc75c900096683e69",
    unpack=True,
)

MODEL_SPEC = DownloadSpec(
    key="model",
    title="中文向量模型 bge-small-zh-v1.5（q8_0）",
    url=("https://huggingface.co/CompendiumLabs/bge-small-zh-v1.5-gguf/"
         "resolve/main/bge-small-zh-v1.5-q8_0.gguf"),
    filename="bge-small-zh-v1.5-q8_0.gguf",
    sha256="5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039",
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(_DOWNLOAD_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _free_port() -> int:
    """让系统分配一个空闲端口。绑定后立刻释放，存在极小竞态，可接受。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _url_fetch(url: str, dest: str, report: Callable[[int, int], None]) -> None:
    """默认下载器：urllib 流式写盘。`report(已下载, 总大小)`。"""
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-fleet"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        report(0, total)
        with open(dest, "wb") as fp:
            while True:
                chunk = resp.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                fp.write(chunk)
                done += len(chunk)
                report(done, total)
        if total and done != total:
            raise LocalEmbedError(f"下载不完整：{done}/{total} 字节")


def _http_healthy(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", "replace")
        return resp.status == 200 and '"ok"' in body.replace(" ", "")
    except Exception:
        return False


def _safe_extract(archive: str, target: str) -> None:
    """解压 zip 并挡住 `../` 路径穿越。"""
    os.makedirs(target, exist_ok=True)
    base = os.path.abspath(target)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            dest = os.path.abspath(os.path.join(target, member))
            if dest != base and not dest.startswith(base + os.sep):
                raise LocalEmbedError(f"压缩包里有越界路径：{member}")
        zf.extractall(target)


# ---------------------------------------------------------------- 进程守护

_JOB_HANDLE = None


def _windows_job():
    """创建一个「本进程退出 → 子进程一起被杀」的 Job 对象。

    正常关插件走 `terminate()`，会显式停掉模型；但宿主被强杀 / 蓝屏重启
    这类场景没有优雅收尾，子进程会变成孤儿继续占着 500MB 内存和端口。
    Windows 的 Job 对象由内核兜底：持有它的进程一消失，子进程立刻被结束。
    """
    global _JOB_HANDLE
    if os.name != "nt":
        return None
    if _JOB_HANDLE is not None:
        return _JOB_HANDLE

    import ctypes
    from ctypes import wintypes

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        ok = k32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info))  # ExtendedLimit
        if not ok:
            k32.CloseHandle(job)
            return None
        _JOB_HANDLE = job  # 句柄必须活到进程结束，不能只放在局部变量里
        return job
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.OpenProcess(0x00100000 | 0x1000, False, int(pid))
        if not handle:
            return False
        try:
            return k32.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _pid_image(pid: int) -> str:
    """进程可执行文件名（小写）。拿不到就返回空串——此时宁可不动它。"""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFO
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if not k32.QueryFullProcessImageNameW(
                    handle, 0, buf, ctypes.byref(size)):
                return ""
            return os.path.basename(buf.value).lower()
        finally:
            k32.CloseHandle(handle)
    try:
        return os.path.basename(os.readlink(f"/proc/{int(pid)}/exe")).lower()
    except OSError:
        return ""


def _kill_pid(pid: int) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return
        except Exception:
            pass
    try:
        os.kill(int(pid), 15)
    except OSError:
        pass


class LocalEmbedManager:
    """一份本机模型的生命周期。构造不会碰磁盘，只有 `install` 才建目录。"""

    def __init__(
        self,
        data_dir: str,
        *,
        log: Callable[[str], None] | None = None,
        fetch: Callable[[str, str, Callable[[int, int], None]], None] | None = None,
        popen: Callable[..., object] | None = None,
        healthy: Callable[[int], bool] | None = None,
        runtime_spec: DownloadSpec = RUNTIME_SPEC,
        model_spec: DownloadSpec = MODEL_SPEC,
        threads: int | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = os.path.join(str(data_dir), "local_embed")
        self.runtime_dir = os.path.join(self.root, "runtime")
        self.model_dir = os.path.join(self.root, "models")
        self.model_path = os.path.join(self.model_dir, model_spec.filename)
        # 上次拉起的进程号/端口。宿主被强杀时靠它认出并清理遗留进程。
        self.state_path = os.path.join(self.root, "server.json")
        self.runtime_spec = runtime_spec
        self.model_spec = model_spec
        self.model_name = LOCAL_MODEL_NAME
        self.dim = LOCAL_MODEL_DIM
        self.threads = int(threads or max(2, min(8, (os.cpu_count() or 4) // 2)))
        self._log = log or (lambda _m: None)
        self._fetch = fetch or _url_fetch
        self._popen = popen or subprocess.Popen
        self._healthy = healthy or _http_healthy
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._state = "idle"
        self._error = ""
        self._port = 0
        self._proc = None
        self._lines: deque[tuple[int, float, str]] = deque(maxlen=_MAX_LOG_LINES)
        self._seq = 0
        self._progress = {"stage": "", "done": 0, "total": 0, "percent": 0}
        self._installing = False
        self._install_thread: threading.Thread | None = None
        # 用户手动点过「停止」：守护循环不能再把它拉起来，否则点了没用。
        self._user_stopped = False
        self._fail_count = 0
        self._next_try = 0.0

    # ------------------------------------------------------------ 路径

    @property
    def runtime_exe(self) -> str:
        """`llama-server.exe` 的实际位置（压缩包解开后在一层子目录里）。"""
        names = ("llama-server.exe", "llama-server")
        direct = [os.path.join(self.runtime_dir, n) for n in names]
        for path in direct:
            if os.path.isfile(path):
                return path
        if not os.path.isdir(self.runtime_dir):
            return ""
        for base, _dirs, files in os.walk(self.runtime_dir):
            for name in names:
                if name in files:
                    return os.path.join(base, name)
        return ""

    @property
    def runtime_installed(self) -> bool:
        return bool(self.runtime_exe)

    @property
    def model_installed(self) -> bool:
        return os.path.isfile(self.model_path)

    @property
    def installed(self) -> bool:
        return self.runtime_installed and self.model_installed

    # ------------------------------------------------------------ 状态

    @property
    def ready(self) -> bool:
        proc = self._proc
        return (self._state == "ready" and proc is not None
                and proc.poll() is None)

    def status(self) -> dict:
        with self._lock:
            proc = self._proc
            alive = proc is not None and proc.poll() is None
            if self._state == "ready" and not alive:
                self._state = "error"
                self._error = self._error or "本机模型进程已退出"
            if self._state in ("starting", "ready") and not alive:
                self._state = "error"
            state = self._state
            if state == "idle" and self.installed:
                # 文件在、进程没起：页面要显示「已下载」，而不是「未安装」。
                state = "installed"
            out = {
                "state": state,
                "ready": self.ready,
                "installed": self.installed,
                "runtime_installed": self.runtime_installed,
                "model_installed": self.model_installed,
                "installing": self._installing,
                "user_stopped": self._user_stopped,
                "model": self.model_name if self.model_installed else "",
                "dim": self.dim if self.ready else 0,
                "port": self._port if self.ready else 0,
                "pid": int(getattr(proc, "pid", 0) or 0) if alive else 0,
                "threads": self.threads,
                "dir": self.root,
                "progress": dict(self._progress),
                "last_error": self._error,
                "log_seq": self._seq,
                "logs_kept": len(self._lines),
                "platform": sys.platform,
            }
            return out

    def logs(self, after: int = 0, limit: int = 400) -> list[dict]:
        with self._lock:
            rows = [(seq, ts, line) for seq, ts, line in self._lines
                    if seq > int(after)]
        return [{"seq": seq, "ts": ts, "line": line}
                for seq, ts, line in rows[-max(1, int(limit)):]]

    def _append(self, line: str) -> None:
        text = str(line or "").rstrip()
        if not text:
            return
        if len(text) > _MAX_LOG_CHARS:
            text = text[:_MAX_LOG_CHARS] + "…"
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, self._clock(), text))

    # ------------------------------------------------------------ 遗留进程

    def _read_state(self) -> dict:
        try:
            with open(self.state_path, encoding="utf-8") as fp:
                data = json.load(fp)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_state(self, pid: int, port: int) -> None:
        tmp = self.state_path + ".tmp"
        try:
            os.makedirs(self.root, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump({
                    "pid": int(pid), "port": int(port),
                    "exe": os.path.basename(self.runtime_exe),
                    "model": self.model_name, "ts": self._clock(),
                }, fp, ensure_ascii=False)
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    def _clear_state(self) -> None:
        try:
            os.remove(self.state_path)
        except OSError:
            pass

    def _reap_stale(self) -> None:
        """上一次宿主被强杀留下的模型进程：先结束它再拉起新的。"""
        state = self._read_state()
        pid = int(state.get("pid") or 0)
        if pid <= 0:
            return
        current = int(getattr(self._proc, "pid", 0) or 0) if self._proc else 0
        if pid == current or not _pid_alive(pid):
            self._clear_state()
            return
        image = _pid_image(pid)
        if image and not image.startswith("llama-server"):
            # pid 被别的程序复用了：不能动它，只把过期记录清掉。
            self._append(f"[local] 遗留的 pid {pid} 已属于 {image}，忽略")
            self._clear_state()
            return
        self._append(f"[local] 结束上次遗留的模型进程 {pid}")
        _kill_pid(pid)
        self._clear_state()

    # ------------------------------------------------------------ 安装

    def install(self, *, wait: bool = False, timeout: float | None = None) -> dict:
        """开始下载运行库与模型。默认立即返回，进度看 `status()`。"""
        if self.installed:
            return self.status()
        with self._lock:
            if self._installing:
                return self.status()
            self._installing = True
            self._state = "downloading"
            self._error = ""
            self._progress = {"stage": "准备下载", "done": 0, "total": 0, "percent": 0}
        if wait:
            self._install_worker()
        else:
            self._install_thread = threading.Thread(
                target=self._install_worker, name="hermes-local-embed-install",
                daemon=True)
            self._install_thread.start()
        return self.status()

    def _install_worker(self) -> None:
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(self.runtime_dir, exist_ok=True)
            if not self.runtime_installed:
                archive = os.path.join(self.root, self.runtime_spec.filename)
                self._download(self.runtime_spec, archive)
                self._set_state("verifying", stage="解压运行库")
                _safe_extract(archive, self.runtime_dir)
                try:
                    os.remove(archive)
                except OSError:
                    pass
                if not self.runtime_installed:
                    raise LocalEmbedError(
                        "运行库解压后没找到 llama-server，压缩包结构可能变了")
            if not self.model_installed:
                self._download(self.model_spec, self.model_path)
            self._append(f"[local] 安装完成：{self.runtime_exe} · {self.model_path}")
            with self._lock:
                self._state = "installed"
                self._error = ""
                self._progress = {"stage": "安装完成", "done": 1, "total": 1,
                                  "percent": 100}
            # 「下载完就自动跑起来」是产品承诺：不要求用户再点一次启动。
            self.start(timeout=90.0)
        except Exception as exc:  # noqa: BLE001 — 后台线程必须自己收尾
            msg = f"{type(exc).__name__}: {exc}"
            self._append(f"[local] 安装失败：{msg}")
            with self._lock:
                self._state = "error"
                self._error = msg
                self._progress = {"stage": "安装失败", "done": 0, "total": 0,
                                  "percent": 0}
            self._log(f"[hermes_fleet] 本机向量模型安装失败：{msg}")
        finally:
            with self._lock:
                self._installing = False

    def _set_state(self, state: str, *, stage: str = "") -> None:
        with self._lock:
            self._state = state
            if stage:
                self._progress = {**self._progress, "stage": stage}

    def _download(self, spec: DownloadSpec, dest: str) -> None:
        part = dest + ".part"
        had_old = os.path.isfile(dest)
        self._append(f"[local] 下载 {spec.title} → {dest}")
        self._set_state("downloading", stage=f"下载 {spec.title}")

        def report(done: int, total: int) -> None:
            percent = int(done * 100 / total) if total else 0
            with self._lock:
                self._progress = {"stage": f"下载 {spec.title}", "done": int(done),
                                  "total": int(total), "percent": percent}

        try:
            self._fetch(spec.url, part, report)
            self._set_state("verifying", stage=f"校验 {spec.title}")
            if spec.sha256:
                got = _sha256(part)
                if got.lower() != spec.sha256.lower():
                    raise LocalEmbedError(
                        f"{spec.title} 校验失败（sha256 {got[:12]}…），已删除重下")
            os.replace(part, dest)
        except Exception:
            # 只删这次写下的半成品；旧文件可能是好的，不能连带删掉。
            try:
                if os.path.exists(part):
                    os.remove(part)
            except OSError:
                pass
            if not had_old:
                try:
                    if os.path.exists(dest):
                        os.remove(dest)
                except OSError:
                    pass
            raise
        size = os.path.getsize(dest) if os.path.isfile(dest) else 0
        self._append(f"[local] 已就绪 {spec.title}（{size / 1024 / 1024:.1f} MB）")

    # ------------------------------------------------------------ 启停

    def ensure_ready(self, timeout: float = 25.0) -> bool:
        """确保服务在跑。返回是否就绪；没装好会抛 `LocalEmbedError`。"""
        if self.ready:
            return True
        if not self.installed:
            raise LocalEmbedError("本机向量模型还没下载，请先在页面上点「下载并启动」")
        with self._start_lock:
            if self.ready:
                return True
            self._start_locked(timeout)
        return self.ready

    def start(self, timeout: float = 25.0) -> dict:
        self._user_stopped = False
        try:
            self.ensure_ready(timeout)
        except LocalEmbedError as exc:
            with self._lock:
                self._state = "error"
                self._error = str(exc)
        return self.status()

    def ensure_alive(self, timeout: float = 60.0) -> bool:
        """守护用：装了就该在跑。用户手动停过、或失败太频繁时不动。"""
        if self.ready:
            return True
        if self._user_stopped or not self.installed:
            return False
        if time.monotonic() < self._next_try:
            return False
        self.start(timeout=timeout)
        if self.ready:
            self._fail_count = 0
            self._next_try = 0.0
            return True
        self._fail_count += 1
        # 起不来时退避 1→2→4…→10 分钟，避免每分钟刷一次同样的失败日志。
        self._next_try = time.monotonic() + min(600.0, 60.0 * self._fail_count)
        return False

    def stop(self, *, user: bool = False) -> dict:
        if user:
            self._user_stopped = True
        with self._lock:
            proc = self._proc
            self._proc = None
            self._port = 0
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._append("[local] 服务已停止")
            # 只清自己停掉的那条记录；没有 proc 时可能是上次遗留的，
            # 留着让下一次启动去清理。
            self._clear_state()
        with self._lock:
            if self._state != "error":
                self._state = "stopped"
        return self.status()

    def restart(self, timeout: float = 25.0) -> dict:
        self.stop()
        return self.start(timeout)

    def _start_locked(self, timeout: float) -> None:
        self._reap_stale()
        old = self._proc
        if old is not None:
            try:
                if old.poll() is None:
                    old.terminate()
                    old.wait(timeout=3)
            except Exception:
                pass
            self._proc = None

        exe = self.runtime_exe
        if not exe:
            raise LocalEmbedError("找不到 llama-server，请重新下载运行库")
        port = _free_port()
        args = [
            exe, "-m", self.model_path,
            "--embeddings", "--pooling", "cls",
            # 单条记忆上限 500 字，中文 BPE 后可能接近 700 token；上下文
            # 给 1024 是为了不被静默截断（截断后的向量只覆盖前半句）。
            "-c", "1024", "-b", "1024", "-ub", "1024",
            "-t", str(self.threads),
            "--port", str(port),
        ]
        kwargs: dict = {
            "cwd": os.path.dirname(exe) or self.runtime_dir,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            # 不要在桌面环境弹一个黑窗口；插件进程退出时也别留下孤儿。
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._append(f"[local] 启动：{exe} --port {port}（线程 {self.threads}）")
        try:
            proc = self._popen(args, **kwargs)
        except Exception as exc:
            raise LocalEmbedError(f"启动 llama-server 失败：{exc}") from exc
        with self._lock:
            self._proc = proc
            self._port = port
            self._state = "starting"
            self._error = ""
        self._attach_job(proc)
        self._write_state(int(getattr(proc, "pid", 0) or 0), port)
        reader = threading.Thread(target=self._read_output, args=(proc,),
                                  name="hermes-local-embed-log", daemon=True)
        reader.start()

        deadline = time.monotonic() + max(2.0, float(timeout))
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                code = proc.poll()
                msg = f"llama-server 退出（code={code}），详见日志"
                with self._lock:
                    self._state = "error"
                    self._error = msg
                self._clear_state()
                raise LocalEmbedError(msg)
            if self._healthy(port):
                with self._lock:
                    self._state = "ready"
                    self._error = ""
                    self._progress = {"stage": "运行中", "done": 1, "total": 1,
                                      "percent": 100}
                self._append(f"[local] 服务就绪：http://127.0.0.1:{port}/v1")
                self._warmup(port)
                return
            self._sleep(0.2)
        msg = "等待 llama-server 就绪超时（模型加载可能失败，详见日志）"
        try:
            proc.terminate()
        except Exception:
            pass
        self._clear_state()
        with self._lock:
            self._proc = None
            self._port = 0
            self._state = "error"
            self._error = msg
        raise LocalEmbedError(msg)

    def _attach_job(self, proc) -> None:
        """把子进程放进 Job：本进程无论怎么退出，它都会被内核收掉。"""
        if os.name != "nt":
            return
        job = _windows_job()
        handle = getattr(proc, "_handle", None)
        if not job or not handle:
            return
        import ctypes

        try:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if not k32.AssignProcessToJobObject(job, int(handle)):
                self._append(
                    f"[local] 绑定 Job 失败（err={ctypes.get_last_error()}）："
                    "宿主被强杀时可能留下进程，下次启动会自动清理")
        except Exception:
            pass

    def _read_output(self, proc) -> None:
        stream = getattr(proc, "stdout", None)
        if stream is None:
            return
        try:
            for line in stream:
                self._append(str(line))
        except Exception:
            pass

    def _warmup(self, port: int) -> None:
        """一次单条调用，把首条请求的模型预热成本挪到启动阶段。"""
        payload = json.dumps({"model": self.model_name, "input": ["预热"]}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/embeddings", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp.read()
            self._append(f"[local] 预热完成，耗时 {(time.perf_counter() - t0) * 1000:.0f}ms")
        except Exception as exc:  # noqa: BLE001 — 预热失败不影响服务可用
            self._append(f"[local] 预热失败（不影响使用）：{exc}")

    # ------------------------------------------------------------ 客户端

    def embedding_config(self, *, batch_size: int = 10,
                         timeout_s: float = 20.0) -> EmbeddingConfig:
        if not self.ready:
            raise LocalEmbedError("本机向量模型未运行")
        return EmbeddingConfig(
            base_url=f"http://127.0.0.1:{self._port}/v1",
            model=self.model_name, dim=self.dim,
            batch_size=batch_size, timeout_s=timeout_s,
        )

    def close(self) -> None:
        self.stop()

    def cleanup_partials(self) -> int:
        """删掉下载残留（`.part`）。返回删掉的个数，页面「清理」按钮用。"""
        removed = 0
        if not os.path.isdir(self.root):
            return 0
        for base, _dirs, files in os.walk(self.root):
            for name in files:
                if name.endswith(".part"):
                    try:
                        os.remove(os.path.join(base, name))
                        removed += 1
                    except OSError:
                        pass
        return removed


class LocalEmbedder:
    """把 `LocalEmbedManager` 伪装成 `EmbeddingClient`。

    `VectorGlue` 只认 `embed()` / `embed_one()` / `enabled`，所以适配层
    放在这里，向量编排层不需要知道「本机模型是一个外部进程」。
    """

    def __init__(self, manager: LocalEmbedManager, *, batch_size: int = 10,
                 timeout_s: float = 20.0) -> None:
        self._manager = manager
        self._batch_size = batch_size
        self._timeout = timeout_s

    @property
    def enabled(self) -> bool:
        return self._manager.ready

    async def embed(self, texts) -> list[list[float]]:
        from hermes_vectors import EmbeddingClient

        try:
            await asyncio.to_thread(self._manager.ensure_ready, 20.0)
            cfg = self._manager.embedding_config(
                batch_size=self._batch_size, timeout_s=self._timeout)
        except LocalEmbedError as exc:
            raise EmbeddingError(str(exc)) from exc
        return await EmbeddingClient(cfg).embed(texts)

    async def embed_one(self, text: str) -> list[float]:
        rows = await self.embed([text])
        return rows[0] if rows else []
