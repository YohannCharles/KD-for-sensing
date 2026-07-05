#!/usr/bin/env python3
"""Compile tracked CLI and script entry files without importing them."""

import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _tracked_python_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "scripts", "src/kd_sensing/cli"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        path = ROOT / line
        if path.suffix == ".py" and path.exists():
            paths.append(path)
    return sorted(paths)


def main() -> int:
    failures: list[str] = []
    for path in _tracked_python_files():
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            rel_path = path.relative_to(ROOT)
            failures.append(f"{rel_path}: {exc.msg}")

    if failures:
        print("Python compile check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Compiled {len(_tracked_python_files())} tracked CLI/script Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
