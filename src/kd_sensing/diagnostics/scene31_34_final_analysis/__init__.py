"""Scene31-34 final analysis owner."""

import argparse

from kd_sensing.diagnostics.scene31_34_final_analysis import (
    conclusion,
    error_cdf,
    main_paper_tables,
    final_paper_tables,
    main_summary,
    missing_count_degradation,
    pattern_heatmap,
    presentation_artifacts,
    profile,
    sampling_distribution,
    significance,
)


ARTIFACTS = {
    "summary": main_summary,
    "missing-count": missing_count_degradation,
    "profile": profile,
    "paper-tables": main_paper_tables,
    "conclusion": conclusion,
    "significance": significance,
    "heatmap": pattern_heatmap,
    "error-cdf": error_cdf,
    "sampling": sampling_distribution,
    "final-paper-tables": final_paper_tables,
    "presentation": presentation_artifacts,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Scene31-34 final analysis artifact.")
    parser.add_argument("artifact_name", nargs="?", choices=sorted(ARTIFACTS))
    parser.add_argument("--artifact", choices=sorted(ARTIFACTS))
    args, rest = parser.parse_known_args(argv)
    artifact = args.artifact or args.artifact_name or "summary"
    return int(ARTIFACTS[artifact].main(rest) or 0)


__all__ = ["ARTIFACTS", "main"]
