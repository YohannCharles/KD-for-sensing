"""Machine-readable public console script surface."""

from typing import NamedTuple


class PublicCli(NamedTuple):
    target: str
    lifecycle: str
    owner: str
    responsibility: str
    output_boundary: str
    focused_validation: str
    help_expected: str


PUBLIC_CLI_SURFACE: dict[str, PublicCli] = {
    "kd-sensing-train": PublicCli(
        "kd_sensing.cli.train:main",
        "core_workflow",
        "kd_sensing.engine.trainer",
        "config-driven training entrypoint",
        "ignored outputs/ and logs/ run roots",
        "conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q",
        "--config",
    ),
    "kd-sensing-evaluate": PublicCli(
        "kd_sensing.cli.evaluate:main",
        "core_workflow",
        "kd_sensing.engine.evaluation_pass",
        "checkpoint evaluation entrypoint",
        "ignored evaluation/output roots or user path",
        "conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q",
        "--weights",
    ),
    "kd-sensing-preprocess": PublicCli(
        "kd_sensing.cli.preprocess:main",
        "core_workflow",
        "kd_sensing.preprocessing",
        "config-driven preprocessing entrypoint",
        "dataset preparation targets or ignored cache/output roots",
        "conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q",
        "--action",
    ),
    "kd-sensing-runs": PublicCli(
        "kd_sensing.cli.runs:console_main",
        "core_workflow",
        "kd_sensing.diagnostics.run_index",
        "read-only local run index",
        "stdout or explicit ignored analysis path",
        "conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_runtime_artifact_cleanup.py -q",
        "--outputs",
    ),
    "kd-sensing-research-dashboard": PublicCli(
        "kd_sensing.cli.research_dashboard:main",
        "current_diagnostic",
        "kd_sensing.diagnostics.research_claim_harvester",
        "read-only claim candidate dashboard",
        "ignored outputs/analysis/ or explicit local path",
        "conda run -n kd_mm_beam pytest tests/test_research_claim_harvester.py tests/test_cli_help.py -q",
        "--output-html",
    ),
    "kd-sensing-research-preview": PublicCli(
        "kd_sensing.cli.research_preview:main",
        "current_diagnostic",
        "kd_sensing.diagnostics.research_run_preview",
        "no-training research preview and budget manifest",
        "ignored outputs/analysis/research_preview/ or explicit local path",
        "conda run -n kd_mm_beam pytest tests/test_research_run_preview.py tests/test_cli_help.py -q",
        "--qa-html",
    ),
    "kd-sensing-clean-runtime-artifacts": PublicCli(
        "kd_sensing.cli.cleanup_runtime_artifacts:main",
        "current_diagnostic",
        "kd_sensing.diagnostics.runtime_artifact_cleanup",
        "runtime artifact cleanup manifest workflow",
        "ignored outputs/cleanup_manifests/ or explicit manifest/report path",
        "conda run -n kd_mm_beam pytest tests/test_runtime_artifact_cleanup.py tests/test_cli_help.py -q",
        "--manifest",
    ),
    "kd-sensing-organize-runtime-outputs": PublicCli(
        "kd_sensing.cli.organize_runtime_outputs:main",
        "current_diagnostic",
        "kd_sensing.diagnostics.runtime_artifact_cleanup",
        "runtime output organize manifest workflow",
        "ignored outputs/cleanup_manifests/ or explicit manifest/report path",
        "conda run -n kd_mm_beam pytest tests/test_runtime_artifact_cleanup.py tests/test_cli_help.py -q",
        "--confirm-organize",
    ),
    "kd-sensing-paper-export": PublicCli(
        "kd_sensing.cli.paper_artifact_export:main",
        "paper_export",
        "kd_sensing.diagnostics.paper_artifact_export",
        "reviewed claim table and figure-data export",
        "ignored outputs/paper_artifacts/ or explicit output dir",
        "conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q",
        "--input",
    ),
    "kd-sensing-eval-u-mask-matrix": PublicCli(
        "kd_sensing.cli.eval_u_mask_beam_jepa_matrix:main",
        "current_diagnostic",
        "kd_sensing.eval.u_mask_beam_jepa_eval_matrix",
        "U-MaskBeamJEPA missing-modality evaluation matrix",
        "ignored outputs/eval/ or explicit output dir",
        "conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa_eval_matrix.py tests/test_cli_help.py -q",
        "--checkpoint",
    ),
    "kd-sensing-mmw-town-gps-v2": PublicCli(
        "kd_sensing.cli.mmw_town_gps_v2:main",
        "current_diagnostic",
        "kd_sensing.engine.mmw_town_gps_v2",
        "MMW Town GPS-only v2 run, plot and compare workflow",
        "ignored outputs/analysis/mmw_town_gps_adapter_v2/ or explicit output dir",
        "conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py tests/test_cli_help.py -q",
        "--mode",
    ),
    "kd-sensing-inspect-mmw-physics": PublicCli(
        "kd_sensing.cli.inspect_mmw_physics:main",
        "current_diagnostic",
        "kd_sensing.models.physics",
        "physics-informed MMW sample inspection",
        "stdout only unless explicit output path is added by caller",
        "conda run -n kd_mm_beam pytest tests/test_physics_informed_mmw.py tests/test_cli_help.py -q",
        "--max-samples",
    ),
    "kd-sensing-project-surface-doctor": PublicCli(
        "kd_sensing.cli.project_surface_doctor:main",
        "current_diagnostic",
        "kd_sensing.diagnostics.project_surface_doctor",
        "read-only project surface governance doctor",
        "stdout only",
        "conda run -n kd_mm_beam pytest tests/test_project_surface_doctor.py tests/test_cli_help.py -q",
        "--scope",
    ),
}

PUBLIC_CLI_HELP_SMOKE: tuple[tuple[str, str], ...] = tuple(
    (command, spec.help_expected) for command, spec in PUBLIC_CLI_SURFACE.items()
)

PUBLIC_CLI_LIFECYCLES = tuple(
    "core_workflow current_diagnostic paper_export baseline_reproduction local_manual internal_only delete".split()
)


__all__ = ["PUBLIC_CLI_HELP_SMOKE", "PUBLIC_CLI_LIFECYCLES", "PUBLIC_CLI_SURFACE", "PublicCli"]
