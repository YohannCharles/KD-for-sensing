import argparse
import json
from pathlib import Path
from typing import Any

from kd_sensing.diagnostics.research_run_preview import (
    DEFAULT_PREVIEW_DIR,
    build_research_run_preview,
    render_preview_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a no-training research run preview, static evidence QA, and budget manifest.",
    )
    parser.add_argument("--project-root", default=".", help="Repository root used for OpenSpec and path context.")
    parser.add_argument("--outputs", action="append", default=None, help="Output root to scan. Defaults to outputs.")
    parser.add_argument("--logs", action="append", default=None, help="Log root to scan. Defaults to logs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREVIEW_DIR, help="Ignored output directory for preview artifacts.")
    parser.add_argument("--qa-html", action="append", default=[], type=Path, help="Static HTML evidence file to validate.")
    parser.add_argument("--qa-csv", action="append", default=[], type=Path, help="CSV summary file to validate.")
    parser.add_argument("--qa-table", action="append", default=[], type=Path, help="Paper table CSV file to validate.")
    parser.add_argument("--qa-figure-data", action="append", default=[], type=Path, help="Figure data CSV/JSON file to validate.")
    parser.add_argument("--qa-checklist", action="append", default=[], type=Path, help="Checklist CSV/Markdown/text file to validate.")
    parser.add_argument("--qa-conclusion", action="append", default=[], type=Path, help="Conclusion draft text/Markdown file to validate.")
    parser.add_argument("--workflow-id", default="research_run_preview_loop", help="Budget workflow id.")
    parser.add_argument("--change-id", default="add-research-run-preview-loop", help="Budget/OpenSpec change id.")
    parser.add_argument("--config", dest="config_path", help="Config path for an explicit long-run budget.")
    parser.add_argument("--manifest", dest="manifest_path", help="Workflow manifest path for an explicit long-run budget.")
    parser.add_argument("--dataset-family", default="synthetic_or_unspecified", help="Budget dataset family.")
    parser.add_argument("--reads-real-dataset", action="store_true", help="Declare that a long-run budget reads real dataset files.")
    parser.add_argument("--gpu", default=None, help="GPU/CPU plan for the budget manifest.")
    parser.add_argument("--estimated-wall-time", default=None, help="Estimated wall time for the budget manifest.")
    parser.add_argument("--parallelism", type=int, default=1, help="Planned run parallelism.")
    parser.add_argument("--output-root", default=None, help="Planned runtime output root for the budget manifest.")
    parser.add_argument("--checkpoint-plan", default=None, help="Checkpoint read/write policy.")
    parser.add_argument("--cache-plan", default=None, help="Cache read/write/rebuild policy.")
    parser.add_argument("--fresh-eval-plan", default=None, help="Fresh-eval plan for the budget manifest.")
    parser.add_argument("--paper-export-plan", default=None, help="Paper export plan for the budget manifest.")
    parser.add_argument("--stop-condition", action="append", default=[], help="Stop condition. Can be repeated.")
    parser.add_argument("--long-run", action="store_true", help="Validate budget fields as a long training/sweep preflight.")
    parser.add_argument("--run-checks", action="store_true", help="Execute the no-training check plan instead of recording it as planned.")
    parser.add_argument("--no-resources", action="store_true", help="Skip process/GPU resource snapshot collection.")
    parser.add_argument("--json", action="store_true", help="Print the preview manifest JSON.")
    return parser


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    budget = {
        "workflow_id": args.workflow_id,
        "change_id": args.change_id,
        "config_path": args.config_path,
        "manifest_path": args.manifest_path,
        "dataset_family": args.dataset_family,
        "reads_real_dataset": args.reads_real_dataset,
        "gpu": args.gpu,
        "estimated_wall_time": args.estimated_wall_time,
        "parallelism": args.parallelism,
        "output_root": args.output_root,
        "checkpoint_plan": args.checkpoint_plan,
        "cache_plan": args.cache_plan,
        "fresh_eval_plan": args.fresh_eval_plan,
        "paper_export_plan": args.paper_export_plan,
        "stop_conditions": args.stop_condition,
        "long_run": args.long_run,
    }
    evidence = {
        "html": args.qa_html,
        "csv": args.qa_csv,
        "table": args.qa_table,
        "figure_data": args.qa_figure_data,
        "checklist": args.qa_checklist,
        "conclusion": args.qa_conclusion,
    }
    manifest = build_research_run_preview(
        project_root=args.project_root,
        outputs=args.outputs or ["outputs"],
        logs=args.logs or ["logs"],
        output_dir=args.output_dir,
        evidence_inputs=evidence,
        include_resources=not args.no_resources,
        run_checks=args.run_checks,
        budget=budget,
    )
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(render_preview_summary(manifest))
    return manifest


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
