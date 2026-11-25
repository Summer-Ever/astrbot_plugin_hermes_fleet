#!/usr/bin/env python3
"""依赖方向静态检查。多 agent 并行开发的安全网。

要解决的问题
------------
契约和目录约定都写在文档里，而**没人会读文档**（包括三个月后的自己）。
并行开发时，一个 agent 为了图快 `from hermes_identity import IdentityService`，
它自己的测试全绿、PR 看起来也没问题，但从此 memory 模块再也无法脱离
identity 独立测试——而这正是当初拆模块的唯一目的。

这类退化是渐进的、单次看起来无害的、且 review 时极难发现的（一行 import
埋在文件第 30 行）。所以必须用机器检查。

规则（`RULES`）
-------------
1. `hermes_contracts` **不得 import 任何其他 hermes_* 包**。
   它是依赖图的根，反向依赖会造成循环。这条最硬。
2. 领域包之间**不得互相 import**，只能 import `hermes_contracts`。
   需要别人的能力时，从构造函数注入 Protocol。
3. 任何包不得 import 其他包的**实现子模块**（如 `hermes_memory.inmemory`），
   只能 import 包级公开接口。这条防止「绕过包边界直插内部」。
4. `hermes_contracts` 只允许标准库。
   一旦它依赖 pydantic，所有 agent 的沙箱都得装包才能跑测试。

用法
----
    python tools/check_boundaries.py            # 检查并报告
    python tools/check_boundaries.py --json     # 机器可读（给 CI 用）

退出码：0 合规，1 有违规。

注意：这个脚本**不试图做完整的模块解析**（那需要真的 import 全部模块，
在没有依赖的机器上做不到）。它做的是源码级 AST 扫描，对「import 语句」
这种显式依赖足够准确，且零依赖、秒级完成。
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_PKG = "hermes_contracts"

# 组装根（composition root）。它天生要 import 各领域包的实现——把「记忆、
# 检索、存储」拼成一次注入流程，正是它的全部职责。为它开例外不是放松规则，
# 而是把例外**显式登记**：不登记的包仍然受 R2/R3 约束。
# 之所以只允许一个组装根：装配逻辑出现两处，就会出现两套依赖关系，
# 「这个功能到底走哪条路径」立刻变得不可回答。
COMPOSITION_PKGS = {"hermes_fleet"}

# 允许的标准库模块（contracts 只准用这些）。用白名单而不是黑名单：
# 黑名单永远漏，而「只准用这几个」是可穷举的。
CONTRACTS_STDLIB_ALLOWED = {
    "__future__",
    "abc",
    "asyncio",
    "collections",
    "dataclasses",
    "enum",
    "math",
    "re",
    "threading",
    "time",
    "typing",
    "unittest",
    "json",
    "hashlib",
    "contextlib",
    "functools",
    "itertools",
    "logging",
    "os",
    "pathlib",
    "sqlite3",
    "tempfile",
    "datetime",
}


@dataclass(frozen=True)
class Violation:
    path: str
    lineno: int
    imported: str
    rule: str
    hint: str


def _iter_py_files(base: Path) -> list[Path]:
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def _top_module(name: str) -> str:
    return name.split(".")[0]


def _imports_of(path: Path) -> list[tuple[int, str]]:
    """返回 (行号, 被导入模块名)。只取 import 语句本身，不展开属性访问。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # 语法错误交给编译器报，这里只跳过
        print(f"警告: {path} 解析失败（{exc}），已跳过", file=sys.stderr)
        return []

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            # level>0 是相对 import，属于包内引用，不参与边界检查
            if node.level and node.level > 0:
                continue
            if node.module:
                found.append((node.lineno, node.module))
    return found


def _discover_packages() -> dict[str, Path]:
    """包名 -> src 目录。从目录推断，不需要 import。"""
    result: dict[str, Path] = {}
    for base_name in ("packages", "apps"):
        base = ROOT / base_name
        if not base.is_dir():
            continue
        for src in sorted(base.glob("*/src")):
            for pkg in sorted(src.iterdir()):
                if pkg.is_dir() and pkg.name.startswith("hermes"):
                    result[pkg.name] = pkg
                elif pkg.is_dir() and (pkg / "__init__.py").exists():
                    result[pkg.name] = pkg
    return result


def check() -> list[Violation]:
    packages = _discover_packages()
    violations: list[Violation] = []

    for pkg_name, pkg_dir in packages.items():
        for path in _iter_py_files(pkg_dir):
            rel = path.relative_to(ROOT).as_posix()
            for lineno, imported in _imports_of(path):
                top = _top_module(imported)

                # --- 规则 4：contracts 只准标准库 ---
                if pkg_name == CONTRACTS_PKG:
                    if top.startswith("hermes"):
                        if top != CONTRACTS_PKG:
                            violations.append(
                                Violation(
                                    rel, lineno, imported, "R1 契约包不得依赖其他业务包",
                                    "契约是依赖图的根；需要新类型就定义在契约里，不要反向 import",
                                )
                            )
                    elif top not in CONTRACTS_STDLIB_ALLOWED:
                        violations.append(
                            Violation(
                                rel, lineno, imported, "R4 契约包只准标准库",
                                "加第三方依赖会让所有 agent 的沙箱都要装包才能跑测试",
                            )
                        )
                    continue

                # --- 组装根：允许引用任意领域包的实现（显式登记的例外）---
                # 它不是「领域包」，职责就是把实现拼起来。仍然检查拼写错误：
                # import 一个不存在的 hermes_* 通常意味着包被改名或还没写。
                if pkg_name in COMPOSITION_PKGS:
                    if top.startswith("hermes") and top not in packages:
                        violations.append(
                            Violation(
                                rel, lineno, imported, "R3 引用了未知的 hermes 包",
                                "检查拼写，或先把它加进 packages/ 下",
                            )
                        )
                    continue

                # --- 规则 2：领域包之间不得互相 import ---
                if top.startswith("hermes") and top not in (CONTRACTS_PKG, pkg_name):
                    if top in packages:
                        violations.append(
                            Violation(
                                rel, lineno, imported, "R2 领域包之间禁止直接依赖",
                                f"改为从 {CONTRACTS_PKG}.ports 引用 Protocol，由组装层注入实现",
                            )
                        )
                    else:
                        violations.append(
                            Violation(
                                rel, lineno, imported, "R3 引用了未知的 hermes 包",
                                "检查拼写，或先把它加进 packages/ 下",
                            )
                        )
                    continue

                # --- 规则 3：不得直接 import 别的包的实现子模块 ---
                if top == pkg_name and imported.count(".") >= 1:
                    sub = imported.split(".", 1)[1]
                    # 只允许包级公开名（`hermes_x` 本身），子模块属于内部实现
                    if sub in ("inmemory", "store_pg", "internal", "_impl"):
                        violations.append(
                            Violation(
                                rel, lineno, imported, "R3 越界引用实现子模块",
                                f"改从 {pkg_name} 包级导入，实现细节不应跨包暴露",
                            )
                        )

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 hermes-platform 的依赖方向")
    parser.add_argument("--json", action="store_true", help="输出 JSON（CI 用）")
    args = parser.parse_args(argv)

    violations = check()

    if args.json:
        print(json.dumps([v.__dict__ for v in violations], ensure_ascii=False, indent=2))
        return 1 if violations else 0

    packages = _discover_packages()
    if not violations:
        print(f"依赖方向检查通过：{len(packages)} 个包，无违规。")
        for name in sorted(packages):
            print(f"  - {name}")
        return 0

    print(f"发现 {len(violations)} 处依赖方向违规：\n")
    for v in violations:
        print(f"  {v.path}:{v.lineno}  import {v.imported}")
        print(f"    [{v.rule}]")
        print(f"    → {v.hint}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
