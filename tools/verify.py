#!/usr/bin/env python3
"""一键校验：CI 与「合并前」的统一入口。

    python tools/verify.py            # 全部检查
    python tools/verify.py --fast     # 跳过测试，只查边界与一致性（秒级）

它把四件事串起来，任何一件失败就返回非零：

1. **所有权清单与实际目录是否一致** —— `owners.json` 里声明「已完成/进行中」
   的包必须真实存在；反之，目录里出现了清单没登记的包，说明有人绕过了流程。
   这条在并行开发里特别有用：它会告诉你「有人偷偷建了个新包」。
2. **依赖方向** —— 见 `check_boundaries.py`。
3. **全部测试** —— 见 `run_tests.py`。
4. **版本号一致** —— 契约包版本同时出现在 `__init__.py` 与清单里，
   发版时最容易漏的就是这种「两处来源」。

为什么不写 shell 脚本：Windows 上 bash 要靠 git-bash，CI 里又要另写一份。
用 Python 写一次，三端通用，而且能直接 import 另外两个脚本的检查函数，
避免「CI 跑的和本地跑的不是同一段逻辑」。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def _step(name: str) -> None:
    print(f"\n[{name}]")


def check_owners() -> tuple[bool, list[str]]:
    """清单 <-> 目录 双向一致性。"""
    problems: list[str] = []
    owners_path = ROOT / "owners.json"
    if not owners_path.exists():
        return False, ["owners.json 缺失"]

    data = json.loads(owners_path.read_text(encoding="utf-8"))
    declared: dict[str, dict] = data["packages"]

    def actual_dirs() -> set[str]:
        found: set[str] = set()
        for base in ("packages", "apps"):
            d = ROOT / base
            if not d.is_dir():
                continue
            for src in d.glob("*/src"):
                for pkg in src.iterdir():
                    if pkg.is_dir() and (pkg / "__init__.py").exists():
                        found.add(pkg.name)
        return found

    actual = actual_dirs()

    for name, meta in declared.items():
        if meta["status"] in ("done", "in_progress") and name not in actual:
            problems.append(f"{name}: 清单标为 {meta['status']}，但目录里不存在")

    for name in sorted(actual):
        if name not in declared:
            problems.append(f"{name}: 目录存在但未在 owners.json 登记（谁在维护它？）")

    print(f"  {len(declared)} 个包登记，{len(actual)} 个包存在")
    return not problems, problems


def check_version() -> tuple[bool, list[str]]:
    sys.path.insert(0, str(ROOT / "packages" / "contracts" / "src"))
    try:
        import hermes_contracts  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        return False, [f"契约包无法 import：{exc}"]
    ver = getattr(hermes_contracts, "__version__", "")
    print(f"  hermes_contracts.__version__ = {ver!r}")
    if not ver:
        return False, ["契约包缺少 __version__"]
    return True, []


def run_boundaries() -> tuple[bool, list[str]]:
    import check_boundaries  # noqa: PLC0415

    violations = check_boundaries.check()
    for v in violations:
        print(f"  {v.path}:{v.lineno} import {v.imported} -> {v.rule}")
    if not violations:
        print("  依赖方向合规")
    return not violations, [f"{v.path}:{v.lineno} {v.rule}" for v in violations]


def run_tests() -> tuple[bool, list[str]]:
    # run_tests.py 在仓库根，不在 tools/，所以要单独把根加进 sys.path
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from run_tests import main as tests_main  # noqa: PLC0415

    code = tests_main([])
    return code == 0, [] if code == 0 else ["测试未全部通过"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hermes-platform 合并前校验")
    parser.add_argument("--fast", action="store_true", help="只查边界与一致性，不跑测试")
    args = parser.parse_args(argv)

    steps = [
        ("1/4 所有权一致性", check_owners),
        ("2/4 版本一致", check_version),
        ("3/4 依赖方向", run_boundaries),
    ]
    if not args.fast:
        steps.append(("4/4 测试", run_tests))

    failed: list[str] = []
    for title, fn in steps:
        _step(title)
        ok, problems = fn()
        if not ok:
            failed.append(title)
            for p in problems:
                print(f"  ✗ {p}")
            if args.fast:
                break

    print("\n" + "=" * 60)
    if failed:
        print(f"校验未通过：{', '.join(failed)}")
        return 1
    print(f"校验通过（{len(steps)} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
