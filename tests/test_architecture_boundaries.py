import ast
import tomllib
from pathlib import Path

from kd_sensing.diagnostics.cli_surface import PUBLIC_CLI_HELP_SMOKE, PUBLIC_CLI_SURFACE


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PUBLIC_SCRIPTS = {
    "kd-sensing-train": "kd_sensing.cli.train:main",
    "kd-sensing-evaluate": "kd_sensing.cli.evaluate:main",
    "kd-sensing-preprocess": "kd_sensing.cli.preprocess:main",
}
RETAINED_SCRIPTS = {
    "analyze_mmw_fused_feature_geometry.py",
    "analyze_mmw_router_joint_static_prior.py",
    "audit_mmw_codebook_topology.py",
    "eval_mmw_all_weather_matrix.py",
    "eval_mmw_router_oracle_gap.py",
    "eval_mmw_router_joint_stress.py",
    "eval_deepsense_twc_evidence.py",
    "eval_mmw_twc_corruption.py",
    "eval_mmw_twc_evidence.py",
    "eval_mmw_twc_temporal_token_stress.py",
    "launch_mmw_all_weather_matrix.py",
    "launch_deepsense_twc_evidence.py",
    "launch_mmw_t2_design_screening.py",
    "launch_mmw_t2_hyperparameter_screening.py",
    "launch_mmw_router_oracle_gap.py",
    "launch_mmw_router_joint_stress.py",
    "launch_mmw_dynamic_router_screen.py",
    "launch_mmw_dynamic_router_evaluation.py",
    "launch_mmw_tie_aware_router_screen.py",
    "launch_mmw_router_utility_screen.py",
    "launch_mmw_twc_evidence.py",
    "launch_twc_posthoc_evidence.py",
    "prepare_deepsense_twc_evidence.py",
    "prepare_mmw_twc_evidence.py",
    "prepare_mmw_twc_temporal_token_stress.py",
    "profile_twc_complexity.py",
    "run_mmw_all_weather_eval_after_training.py",
    "run_mmw_t2_bpa_cma_ablation_after_training.py",
    "summarize_mmw_all_weather_matrix.py",
    "summarize_mmw_multiseed_baselines.py",
    "summarize_mmw_t2_bpa_cma_ablation.py",
    "summarize_mmw_task_output_robustness.py",
    "summarize_mmw_router_oracle_gap.py",
    "summarize_mmw_router_joint_stress.py",
    "summarize_mmw_tie_aware_router_screen.py",
    "summarize_mmw_router_utility_screen.py",
    "run_mmw_router_oracle_gap_candidate.py",
    "summarize_mmw_twc_evidence.py",
    "summarize_mmw_twc_temporal_token_stress.py",
    "summarize_deepsense_twc_evidence.py",
    "summarize_twc_mechanism.py",
    "update_twc_experiment_status.py",
    "verify_compile.py",
}
RETAINED_CLI_MODULES = {"__init__.py", "common.py", "evaluate.py", "preprocess.py", "train.py"}
PROTECTED_SYSTEM_PATHS = ("/root/.container_env", "/etc/profile", "/etc/environment", "/etc/ssh/sshd_config")


def test_public_cli_surface_is_exactly_the_three_core_workflows():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts == PUBLIC_SCRIPTS
    assert {name: spec.target for name, spec in PUBLIC_CLI_SURFACE.items()} == PUBLIC_SCRIPTS
    assert {name for name, _ in PUBLIC_CLI_HELP_SMOKE} == set(PUBLIC_SCRIPTS)


def test_public_cli_targets_exist():
    for command, target in PUBLIC_SCRIPTS.items():
        module_name, function_name = target.split(":", 1)
        module_path = SRC.joinpath(*module_name.split(".")).with_suffix(".py")
        assert module_path.exists(), command
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        assert function_name in {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_cli_and_script_trees_only_keep_t2_baseline_owners():
    cli_modules = {path.name for path in (SRC / "kd_sensing/cli").glob("*.py")}
    scripts = {
        path.relative_to(ROOT / "scripts").as_posix()
        for path in (ROOT / "scripts").rglob("*.py")
        if "__pycache__" not in path.parts
    }

    assert cli_modules == RETAINED_CLI_MODULES
    assert scripts == RETAINED_SCRIPTS


def test_sources_do_not_mutate_protected_system_configuration():
    violations = []
    for root in (SRC, ROOT / "scripts"):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if any(system_path in text for system_path in PROTECTED_SYSTEM_PATHS):
                violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []
