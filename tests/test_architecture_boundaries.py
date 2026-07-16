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
    "eval_mmw_all_weather_matrix.py",
    "launch_mmw_all_weather_matrix.py",
    "launch_mmw_t2_hyperparameter_screening.py",
    "run_mmw_all_weather_eval_after_training.py",
    "run_mmw_t2_bpa_cma_ablation_after_training.py",
    "summarize_mmw_all_weather_matrix.py",
    "summarize_mmw_multiseed_baselines.py",
    "summarize_mmw_t2_bpa_cma_ablation.py",
    "summarize_mmw_task_output_robustness.py",
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
