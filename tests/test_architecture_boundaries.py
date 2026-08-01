import ast
import importlib.metadata
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PUBLIC_SCRIPTS = {
    "kd-sensing-train": "kd_sensing.cli.train:main",
    "kd-sensing-evaluate": "kd_sensing.cli.evaluate:main",
    "kd-sensing-preprocess": "kd_sensing.cli.preprocess:main",
}
RETAINED_SCRIPTS = {
    "eval_mmw_all_weather_matrix.py",
    "launch_mmw_all_weather_matrix.py",
    "summarize_mmw_all_weather_matrix.py",
    "verify_compile.py",
}
RETAINED_CLI_MODULES = {"__init__.py", "common.py", "evaluate.py", "preprocess.py", "train.py"}
PROTECTED_SYSTEM_PATHS = ("/root/.container_env", "/etc/profile", "/etc/environment", "/etc/ssh/sshd_config")
RETIRED_SKILL_REFERENCES = {
    "docs/agent_context/claims.md",
    "docs/agent_context/diagnostics.md",
    "docs/agent_context/openspec.md",
    "kd-sensing-jepa-gps-shortcut-benchmark",
    "kd-sensing-jepa-visual-analysis",
    "kd-sensing-paper-export",
    "kd-sensing-project-surface-doctor",
    "kd-sensing-runs",
    "openspec/specs/canonical-config-resolution/spec.md",
    "openspec/specs/component-registry/spec.md",
    "openspec/specs/mainline-experiment-documentation/spec.md",
    "openspec/specs/model-architecture-extension-contract/spec.md",
    "openspec/specs/modular-sequence-model/spec.md",
    "openspec/specs/research-claim-harvester/spec.md",
}
RETIRED_OWNERS = {
    "configs/mmw/t2.yaml",
    "configs/mmw/s1.yaml",
    "configs/modality_competition",
    "src/kd_sensing/config/historical_u2.py",
    "src/kd_sensing/config/standalone_capacity.py",
    "src/kd_sensing/data/mmw/nested_capacity.py",
    "src/kd_sensing/losses/bcacl.py",
    "src/kd_sensing/losses/bcacl_config.py",
    "src/kd_sensing/losses/cmsbl.py",
    "src/kd_sensing/losses/cmsbl_config.py",
    "src/kd_sensing/models/bcacl.py",
    "src/kd_sensing/baselines/clean_recovery.py",
    "src/kd_sensing/evaluation/clean_recovery_summary.py",
    "src/kd_sensing/evaluation/clean_stage.py",
    "src/kd_sensing/evaluation/gps_shortcut.py",
    "src/kd_sensing/evaluation/independent_metrics.py",
    "src/kd_sensing/evaluation/modality_competition.py",
    "src/kd_sensing/losses/clean_capacity_reference.py",
    "src/kd_sensing/losses/modality_alignment_contrastive.py",
}


def test_public_cli_surface_is_exactly_the_three_core_workflows():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts == PUBLIC_SCRIPTS


def test_public_cli_targets_exist():
    for command, target in PUBLIC_SCRIPTS.items():
        module_name, function_name = target.split(":", 1)
        module_path = SRC.joinpath(*module_name.split(".")).with_suffix(".py")
        assert module_path.exists(), command
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        assert function_name in {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_installed_editable_cli_surface_matches_pyproject():
    distribution = importlib.metadata.distribution("kd-sensing")
    installed_scripts = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts" and entry.name.startswith("kd-sensing-")
    }

    assert installed_scripts == PUBLIC_SCRIPTS


def test_project_skills_do_not_reference_retired_context_or_cli_surfaces():
    skill_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".codex" / "skills").glob("*/SKILL.md"))
        if path.parent.name.startswith("kd-")
    )

    assert all(reference not in skill_text for reference in RETIRED_SKILL_REFERENCES)


def test_cli_and_script_trees_only_keep_retained_mmw_workflow_owners():
    cli_modules = {path.name for path in (SRC / "kd_sensing/cli").glob("*.py")}
    scripts = {
        path.relative_to(ROOT / "scripts").as_posix()
        for path in (ROOT / "scripts").rglob("*.py")
        if "__pycache__" not in path.parts
    }

    assert cli_modules == RETAINED_CLI_MODULES
    assert scripts == RETAINED_SCRIPTS


def test_retired_u2_and_capacity_owners_are_absent():
    assert all(not (ROOT / path).exists() for path in RETIRED_OWNERS)


def test_openspec_current_context_is_scoped_to_pcpf_mainline() -> None:
    specs = {
        path.parent.name
        for path in (ROOT / "openspec/specs").glob("*/spec.md")
    }
    active_changes = {
        path.name
        for path in (ROOT / "openspec/changes").iterdir()
        if path.is_dir() and path.name != "archive"
    }

    assert specs == {
        "clean-data-integrity",
        "mmw-id-stratified-block-protocol",
        "repo-boundaries",
        "u0-mainline",
    }
    assert active_changes == {"add-pcpf-temporal-risk-fusion"}
    assert not (ROOT / "openspec/changes/archive").exists()


def test_sources_do_not_mutate_protected_system_configuration():
    violations = []
    for root in (SRC, ROOT / "scripts", ROOT / "tools"):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if any(system_path in text for system_path in PROTECTED_SYSTEM_PATHS):
                violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []


def test_local_experiment_tools_do_not_extend_the_public_cli():
    """`tools/` may orchestrate local experiments but must stay off the packaged entry points."""
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    targets = {spec.split(":", 1)[0] for spec in scripts.values()}

    assert all(not target.startswith("tools") for target in targets)
    assert not (SRC / "kd_sensing/tools").exists()
