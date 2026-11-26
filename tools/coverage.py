#!/usr/bin/env python3
"""覆盖率门槛：测试通过之外，还要保证「入口文件真的被测到」。

    python -m pip install coverage     # 只在这一步依赖本机装 coverage
    python tools/coverage.py

为什么单独立一个门槛，而不是把断言塞进 `run_tests.py`：

- `run_tests.py` 刻意零第三方依赖，装上 coverage 才能跑的东西不能放进去；
- 用例数会涨，但「有没有测到 main.py」不会自己涨。金路径用例写了 5 个，
  全部打在 service 上、`on_llm_request` 一次都没跑过，用例数照样漂亮。

阈值取实测值下方 2～3 个百分点，不是目标值：门槛的作用是**发现断崖**
（某个入口文件突然没人测了），不是逼着为 0.5% 去写凑数用例。
要提高，先补用例，再把这里的数字抬上去。

产物（`.coverage` 数据文件与 JSON 报告）写在系统临时目录，`finally` 里清掉，
仓库不留脏东西——所以 `.gitignore` 不需要为它加规则。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `main.py` 是 AstrBot 接入层（1245 行），`webapi.py` 是管理台壳（937 行）。
# 这两处最容易被「只测业务层」的用例绕过去，所以单独设门槛。
TARGETS = {
    "main.py": 42.0,
    "webapi.py": 53.0,
}
# 全仓（含 tests / tools）的下限。低于这个数说明整片代码没被执行过。
TOTAL_MIN = 82.0


def _fail(message: str, code: int = 1) -> int:
    print(f"✗ {message}")
    return code


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def _rate(entry: dict) -> float:
    # `files[*]` 用 summary.percent_covered_display，`totals` 只有
    # percent_covered（未取整），两处都要认。
    summary = entry.get("summary") or entry
    value = summary.get("percent_covered_display")
    if value is None:
        value = summary.get("percent_covered")
    return float(value or 0.0)


def main() -> int:
    if shutil.which("coverage") is None and _run(
            [sys.executable, "-m", "coverage", "--version"]).returncode != 0:
        return _fail("未安装 coverage，请先执行："
                     " python -m pip install coverage", 2)

    tmp = Path(tempfile.mkdtemp(prefix="hermes_cov_"))
    data_file = tmp / "cov"
    json_file = tmp / "cov.json"
    try:
        print("[1/2] 跑测试并采集覆盖率")
        run = _run([sys.executable, "-m", "coverage", "run",
                    f"--data-file={data_file}", "--source=.", "run_tests.py"])
        if run.returncode != 0:
            # 测试都没过，覆盖率没有意义；把输出原样交回去，别掩盖真正的错。
            print(run.stdout[-4000:])
            print(run.stderr[-4000:])
            return _fail(f"测试未通过（退出码 {run.returncode}），不评估覆盖率",
                         run.returncode)

        print("[2/2] 生成报告并比对门槛")
        report = _run([sys.executable, "-m", "coverage", "json",
                       f"--data-file={data_file}", "-o", str(json_file)])
        if report.returncode != 0:
            print(report.stderr[-4000:])
            return _fail("生成覆盖率报告失败")
        data = json.loads(json_file.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # key 是相对仓库根的路径，Windows 下是反斜杠。
    files = {k.replace("\\", "/"): v for k, v in (data.get("files") or {}).items()}

    problems: list[str] = []
    print(f"\n{'文件':<28}{'覆盖率':>9}{'门槛':>9}{'语句数':>9}")
    for name, floor in TARGETS.items():
        entry = files.get(name)
        if entry is None:
            problems.append(f"{name}: 报告里没有它（一行都没被执行？）")
            print(f"{name:<28}{'--':>9}{floor:>8.1f}%{'--':>9}")
            continue
        rate = _rate(entry)
        statements = int((entry.get("summary") or {}).get("num_statements") or 0)
        mark = "" if rate >= floor else "  ✗"
        print(f"{name:<28}{rate:>8.1f}%{floor:>8.1f}%{statements:>9}{mark}")
        if rate < floor:
            problems.append(f"{name}: {rate:.1f}% < {floor:.1f}%")

    total = _rate(data.get("totals") or {})
    mark = "" if total >= TOTAL_MIN else "  ✗"
    print(f"{'合计（全仓）':<26}{total:>8.1f}%{TOTAL_MIN:>8.1f}%{mark}")
    if total < TOTAL_MIN:
        problems.append(f"合计: {total:.1f}% < {TOTAL_MIN:.1f}%")

    print("\n" + "=" * 60)
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
        print("覆盖率门槛未通过")
        return 1
    print(f"覆盖率门槛通过（{len(TARGETS)} 个文件 + 合计）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
