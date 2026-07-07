#!/usr/bin/env python3

import argparse
from pathlib import Path

from kd_sensing.diagnostics.scene31_summary.subset_reliability import summarize


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summarize(
        roots=[Path(args.baseline_root)],
        out_dir=Path(args.out),
        output_prefix="subset_reference",
        conclusion_name="subset_reference_conclusion.txt",
        include_combined_sections=False,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31 baseline pack using randomdrop subset as reference.")
    parser.add_argument("--baseline-root", default="outputs/scene31_baseline_pack_lmdb")
    parser.add_argument("--out", default="outputs/scene31_baseline_pack_lmdb/subset_reference_summary")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
