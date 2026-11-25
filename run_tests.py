#!/usr/bin/env python3
"""统一测试入口。零第三方依赖，`python run_tests.py` 就能跑全量。

为什么不用 pytest：
    这套方案要在**很多台机器、很多个 agent 沙箱**里跑。pytest 需要装包、
    需要网络、需要处理 rootdir 与 importmode 的各种坑。而这些测试的形态
    很规整（unittest.TestCase），标准库的 loader 完全够用。

    如果你本地有 pytest，它也能直接收集同一批文件（套件类都设了
    `__test__ = False`，基类不会被误跑）。两条路并存，互不干扰。

给并行开发用的分片能力：
    python run_tests.py                     全量（CI / 合并前）
    python run_tests.py -p memory           只跑路径含 "memory" 的（某 agent 的日常）
    python run_tests.py -p contracts -p api 多个分片
    python run_tests.py -k dedup            只跑测试名含 "dedup" 的
    python run_tests.py --list              列出会跑哪些文件和用例数

退出码：0 全绿，1 有失败，2 参数或导入错误。导入错误单独区分很要紧——
    「模块 import 失败」和「断言不通过」是两类完全不同的故障，混在一起会让
    CI 日志误导人。
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _add_src_paths() -> None:
    """把 packages/*/src 与 apps/*/src 加进 sys.path（等价于 editable install）。

    不做 pip install -e 的原因：那需要构建后端和网络，而多 agent 沙箱里
    这两样都不保证有。sys.path 注入在这一步是完全等价的。
    """
    candidates: list[Path] = []
    for base in ("packages", "apps"):
        d = ROOT / base
        if d.is_dir():
            candidates.extend(sorted(d.glob("*/src")))
    # 倒序 insert 保证 packages 的搜索顺序与目录名排序一致
    for p in reversed(candidates):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _module_name_for(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    # 目录名可能含 `-`，模块名不允许，统一替换
    return "hermes_suite_" + re.sub(r"\W", "_", "_".join(rel.parts))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(_module_name_for(path), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # 有些实现需要 __module__ 可解析
    spec.loader.exec_module(module)
    return module


def _iter_test_files(shards: list[str]) -> list[Path]:
    bases: list[Path] = []
    for name in ("packages", "tests"):
        d = ROOT / name
        if d.is_dir():
            bases.append(d)

    files: list[Path] = []
    for base in bases:
        for f in sorted(base.rglob("test_*.py")):
            rel = f.relative_to(ROOT).as_posix()
            if shards and not any(s in rel for s in shards):
                continue
            files.append(f)
    return files


class _Result(unittest.TextTestResult):
    """把 skip 的原因也打出来——「因为环境缺失而跳过」和「故意跳过」必须能区分，
    否则 CI 全绿但其实什么都没测。"""

    def addSkip(self, test, reason):  # noqa: N802 (unittest 命名)
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.writeln(f"  skipped: {reason}")


def _drop_abstract_bases(suite: unittest.TestSuite) -> unittest.TestSuite:
    """剔除抽象一致性套件本身。

    `MemoryStoreConformance` / `JobQueueConformance` 是基类，它们的方法会被
    unittest 收集并以 SkipTest 结束——于是报告里出现十几个「跳过」，让人误以为
    有东西没测。它们的**子类**（`TestInMemoryMemoryStore` 等）会正常执行，
    那才是真正的覆盖。

    按类名后缀识别，而不是给基类加装饰器：装饰器的 skip 会被子类继承，
    那样真正要跑的用例也会被跳过。
    """
    out = unittest.TestSuite()
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            out.addTests(_drop_abstract_bases(item))
        elif type(item).__name__.endswith("Conformance"):
            continue
        else:
            out.addTest(item)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hermes-platform 测试入口")
    parser.add_argument("-p", "--shard", action="append", default=[], metavar="SUBSTR",
                        help="只跑路径包含该子串的测试文件，可重复")
    parser.add_argument("-k", "--pattern", default=None,
                        help="只跑测试方法名匹配该正则的用例")
    parser.add_argument("--list", action="store_true", help="只列出文件与用例数")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    _add_src_paths()

    files = _iter_test_files(args.shard)
    if not files:
        print(f"没有匹配的测试文件（shard={args.shard or '全部'}）", file=sys.stderr)
        return 2

    loader = unittest.TestLoader()
    if args.pattern:
        loader.testNamePatterns = [f"*{args.pattern}*"]

    suite = unittest.TestSuite()
    import_errors: list[tuple[Path, str]] = []
    total = 0

    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        try:
            module = _load(f)
        except Exception:
            import_errors.append((f, traceback.format_exc()))
            continue
        sub = _drop_abstract_bases(loader.loadTestsFromModule(module))
        n = sub.countTestCases()
        total += n
        if args.list:
            print(f"{rel:60s} {n:4d} cases")
        suite.addTests(sub)

    if args.list:
        print(f"\n合计 {len(files)} 个文件，{total} 个用例")
        for path, tb in import_errors:
            print(f"\n!! import 失败: {path.relative_to(ROOT)}\n{tb}", file=sys.stderr)
        return 2 if import_errors else 0

    if import_errors:
        # 导入失败先报，因为后续结果不可信（该文件一个用例都没跑到）
        for path, tb in import_errors:
            print(f"\n!! import 失败: {path.relative_to(ROOT)}\n{tb}", file=sys.stderr)

    print(f"运行 {total} 个用例，来自 {len(files)} 个文件\n" + "-" * 68)
    runner = unittest.TextTestRunner(
        verbosity=2 if args.verbose else 1,
        resultclass=_Result,
        failfast=args.fail_fast,
    )
    started = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - started

    print("-" * 68)
    print(
        f"通过 {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}"
        f"  失败 {len(result.failures)}  错误 {len(result.errors)}"
        f"  跳过 {len(result.skipped)}  用时 {elapsed:.2f}s"
    )
    if import_errors:
        print(f"导入失败 {len(import_errors)} 个文件（这些文件里的用例一个都没跑）")
        return 2
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
