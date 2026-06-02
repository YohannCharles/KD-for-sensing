from __future__ import annotations

import argparse
import json
from pathlib import Path

from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.diagnostics.distribution_shift import analyze_distribution_shift


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze source/target beam label distribution shift.")
    parser.add_argument("--config", "-c", required=True, help="YAML config containing label_space/diagnostics sections.")
    parser.add_argument("--split-artifact", required=False, help="Target-shot split artifact JSON.")
    parser.add_argument("--output-dir", required=False, help="Output directory for metrics and histograms.")
    parser.add_argument("--figures", action="store_true", help="Generate optional histogram PNG files when matplotlib is available.")
    parser.add_argument("--figures-required", action="store_true", help="Fail if figure generation is requested but unavailable.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = safe_load_yaml(Path(args.config).read_text(encoding="utf-8")) or {}
    diagnostics_cfg = cfg.get("diagnostics", {}).get("distribution_shift", {})
    split_path = args.split_artifact or diagnostics_cfg.get("split_artifact") or cfg.get("split", {}).get("artifact_path")
    output_dir = args.output_dir or diagnostics_cfg.get("output_dir")
    if not split_path:
        parser.error("--split-artifact or diagnostics.distribution_shift.split_artifact is required.")
    if not output_dir:
        parser.error("--output-dir or diagnostics.distribution_shift.output_dir is required.")
    artifact = json.loads(Path(split_path).read_text(encoding="utf-8"))
    result = analyze_distribution_shift(
        split_artifact=artifact,
        split_artifact_path=split_path,
        output_dir=output_dir,
        label_space=cfg.get("label_space", {}),
        smoothing=float(diagnostics_cfg.get("smoothing", 1e-6)),
        make_figures=bool(args.figures or diagnostics_cfg.get("figures", False)),
        figures_required=bool(args.figures_required or diagnostics_cfg.get("figures_required", False)),
    )
    print(json.dumps({"metrics": result["outputs"]["metrics_json"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
