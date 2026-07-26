#!/usr/bin/env python3
"""Compile on-disk package, script and local experiment files without importing them."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OWNER_ROOTS = (Path("scripts"), Path("src/kd_sensing"), Path("tools"))
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {"dataset", "outputs", "logs", "cache", "checkpoint", "checkpoints", "__pycache__"}
)


def _python_files(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for owner_root in OWNER_ROOTS:
        directory = root / owner_root
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            relative_path = path.relative_to(root)
            if path.is_file() and not path.is_symlink() and not (
                EXCLUDED_DIRECTORY_NAMES & set(relative_path.parts)
            ):
                paths.add(path)
    return sorted(paths)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root to scan (defaults to the current project root).",
    )
    return parser.parse_args()


def main() -> int:
    root = _parse_args().root.resolve()
    paths = _python_files(root)
    failures: list[str] = []
    for path in paths:
        try:
            compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        except (OSError, SyntaxError) as exc:
            rel_path = path.relative_to(root)
            failures.append(f"{rel_path}: {exc}")

    if failures:
        print("Python compile check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Compiled {len(paths)} on-disk package/script/experiment Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
