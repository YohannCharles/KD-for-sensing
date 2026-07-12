import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from kd_sensing.diagnostics.cli_surface import PUBLIC_CLI_HELP_SMOKE, PUBLIC_CLI_LIFECYCLES, PUBLIC_CLI_SURFACE


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
INVENTORY = ROOT / "docs/project_surface_inventory.md"

PROTECTED_SYSTEM_PATHS = (
    "/root/.container_env",
    "/etc/profile",
    "/etc/environment",
    "/etc/ssh/sshd_config",
    "/root/.ssh/authorized_keys",
    "~/.ssh/authorized_keys",
)
SYSTEM_CONFIG_MUTATION_RE = re.compile(r"(?:>>|>\s*|tee\s+-a?|sed\s+-i|write_text\(|open\([^)]*['\"]w|cat\s*>)")
CREDENTIAL_POLLUTION_RE = re.compile(
    r"(?i)\b(?:USERNAME|USER|PASSWD|PASSWORD|TOKEN|SECRET)\s*=.*"
    r"(?:kd-sensing-(?:train|clean|organize)|CUDA_VISIBLE_DEVICES|nohup|tmux|rm\s+-rf|cd\s+)"
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password|passwd)\s*[:=]\s*['\"][^'\"\n]{16,}['\"]"),
)


def test_config_import_stays_outside_tensor_and_training_runtime():
    probe = """
import sys

import kd_sensing.config

loaded = set(sys.modules)
forbidden_exact = {
    "torch",
    "kd_sensing.data.temporal_missing",
    "kd_sensing.diagnostics.run_index_render",
    "kd_sensing.engine.trainer",
}
forbidden_prefixes = (
    "kd_sensing.data.datasets.",
    "kd_sensing.models.",
)
violations = sorted(
    name
    for name in loaded
    if name in forbidden_exact or any(name.startswith(prefix) for prefix in forbidden_prefixes)
)
if violations:
    raise SystemExit(f"config import loaded runtime modules: {violations}")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr

CURRENT_CONFIG_GLOBS = (
    "configs/fusion/physics_informed_mmw*.yaml",
    "configs/csi/hardening_matrix/*.yaml",
    "configs/csi/hardening_matrix/debug/*.yaml",
    "configs/fusion/csi_hardening_matrix/*.yaml",
    "configs/diagnostics/*.yaml",
    "configs/pretraining/*.yaml",
    "configs/eval/*.yaml",
)

CURRENT_PATHS = (
    "configs/fusion/u_mask_beam_jepa_smoke.yaml",
    "configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml",
    "configs/fusion/physics_informed_mmw_debug.yaml",
    "configs/fusion/csi_hardening_matrix/E1_gps_clean_csi_joint.yaml",
    "src/kd_sensing/engine/objectives/metadata.py",
    "src/kd_sensing/models/u_mask_beam_jepa.py",
    "src/kd_sensing/losses/u_mask_beam_jepa.py",
    "scripts/launch_final_c2_ablation_v1.py",
    "scripts/summarize_final_c2_ablation_v1.py",
    "scripts/mmw/visualize_town_label_distribution.py",
    "docs/project_surface_inventory.md",
    "docs/maintainer_context_index.yaml",
)

FORBIDDEN_IMPORTS = (
    "from kd_sensing._typing import",
    "from kd_sensing.config.source import",
    "from kd_sensing.engine.objective_metadata import",
    "import kd_sensing.engine.objective_metadata",
    "from kd_sensing.engine.objectives import",
    "from kd_sensing.data import",
    "from kd_sensing.data.difficulty import",
    "from kd_sensing.data.datasets import",
    "from kd_sensing.data.transform_ops.normalization import",
    "from kd_sensing.baselines.beambench.image_ae_gps import",
    "import kd_sensing.baselines.beambench.image_ae_gps",
    "from kd_sensing.baselines.beambench",
    "import kd_sensing.baselines.beambench",
    "from kd_sensing.baselines.rmbp_mm",
    "import kd_sensing.baselines.rmbp_mm",
    "from kd_sensing.baselines.tii_vlrg_transformer import",
    "import kd_sensing.baselines.tii_vlrg_transformer",
    "from kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep import",
    "import kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep",
    "from kd_sensing.diagnostics.jepa_gps_shortcut_benchmark import",
    "import kd_sensing.diagnostics.jepa_gps_shortcut_benchmark",
    "from kd_sensing.diagnostics.jepa_benchmark_",
    "import kd_sensing.diagnostics.jepa_benchmark_",
    "from kd_sensing.diagnostics.jepa_visual_analysis import",
    "import kd_sensing.diagnostics.jepa_visual_analysis",
    "from kd_sensing.diagnostics.project_surface_doctor import",
    "import kd_sensing.diagnostics.project_surface_doctor",
    "from kd_sensing.diagnostics.distribution_shift import",
    "import kd_sensing.diagnostics.distribution_shift",
    "from kd_sensing.diagnostics.dataset_reproducibility_audit import",
    "import kd_sensing.diagnostics.dataset_reproducibility_audit",
    "from kd_sensing.models.bev_fusion_2604 import",
    "import kd_sensing.models.bev_fusion_2604",
    "from kd_sensing.models.vision_position import",
    "import kd_sensing.models.vision_position",
    "from kd_sensing.evaluation.bev_fusion_2604_report import",
    "import kd_sensing.evaluation.bev_fusion_2604_report",
    "from kd_sensing.engine.loso_data import",
    "import kd_sensing.engine.loso_data",
    "from kd_sensing.losses import",
    "from kd_sensing.data.mmw import",
    "from kd_sensing.engine import",
    "from kd_sensing.utils import",
    "from kd_sensing.preprocessing import",
    "from kd_sensing.evaluation import",
    "from kd_sensing.eval.export import",
    "import kd_sensing.eval.export",
    "from kd_sensing.models.physics import",
    "from kd_sensing.diagnostics.jepa_benchmark_common import *",
    "from kd_sensing.models.fusion import",
)

RETIRED_TEXT_MARKERS = (
    "HiST-Beam",
    "Top8 selector",
    "GPS residual",
    "camera residual",
    "Raymobtime s008",
    "BGAM",
    "viewer manifest",
    "Gradio viewer",
    "CRAF",
    "MARF",
    "Multimodal-NF",
)

RETIREMENT_CONTEXT = (
    "退役",
    "历史",
    "拒绝",
    "墓碑",
    "防回流",
    "不得",
    "不再",
    "retired",
    "historical",
    "removed",
    "no longer",
    "tombstone",
)

SCENE31_MANIFEST_BACKED_CONFIG_ROOTS = (
    "configs/scene31/night_grid",
    "configs/scene31/next_round",
    "configs/scene31/funnel",
    "configs/scene31/magic_overnight",
)

SCENE31_RETAINED_YAML = {
    "configs/scene31/templates/main_v3_proto_es20_base.yaml",
}

def test_pyproject_console_scripts_point_to_existing_functions():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts

    for command, target in scripts.items():
        module_name, function_name = target.split(":", 1)
        module_path = SRC.joinpath(*module_name.split(".")).with_suffix(".py")
        assert module_path.exists(), f"{command} points to missing module {module_name}"
        names = _top_level_names(module_path)
        assert function_name in names, f"{command} points to missing function {function_name}"


def test_cli_modules_are_console_scripts_or_shared_helpers():
    scripts = _pyproject()["project"]["scripts"]
    console_modules = {target.split(":", 1)[0] for target in scripts.values()}
    shared_helpers = {"kd_sensing.cli.common"}
    violations: list[str] = []

    for path in sorted((SRC / "kd_sensing/cli").glob("*.py")):
        if path.name == "__init__.py":
            continue
        module_name = ".".join(path.with_suffix("").relative_to(SRC).parts)
        names = _top_level_names(path)
        has_runnable = "main" in names or "console_main" in names or "build_parser" in names
        if has_runnable and module_name not in console_modules and module_name not in shared_helpers:
            violations.append(f"{module_name} ({_rel(path)})")

    assert not violations


def test_public_console_scripts_have_lifecycle_smoke_and_inventory_anchor():
    scripts = {
        name: target
        for name, target in _pyproject()["project"]["scripts"].items()
        if name.startswith("kd-sensing-")
    }
    smoke_commands = {name for name, _expected in PUBLIC_CLI_HELP_SMOKE}
    inventory = INVENTORY.read_text(encoding="utf-8")

    assert set(PUBLIC_CLI_SURFACE) == set(scripts)
    assert smoke_commands == set(scripts)

    violations: list[str] = []
    for command, spec in PUBLIC_CLI_SURFACE.items():
        if spec.target != scripts[command]:
            violations.append(f"{command}: target {scripts[command]} != {spec.target}")
        if spec.lifecycle not in PUBLIC_CLI_LIFECYCLES:
            violations.append(f"{command}: invalid lifecycle {spec.lifecycle}")
        for marker in (f"`{command}`", f"`{spec.lifecycle}`", spec.owner, spec.output_boundary):
            if marker not in inventory:
                violations.append(f"{command}: missing inventory marker {marker}")

    assert not violations


def test_current_paths_and_config_globs_are_real():
    for rel_path in CURRENT_PATHS:
        assert (ROOT / rel_path).exists(), rel_path
    for pattern in CURRENT_CONFIG_GLOBS:
        matches = sorted(ROOT.glob(pattern))
        assert matches, pattern
        assert all(path.is_file() for path in matches)


def test_runtime_sources_do_not_use_future_annotations_or_star_imports():
    violations: list[str] = []
    for root in (SRC, ROOT / "scripts", ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            rel = _rel(path)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    if any(alias.name == "annotations" for alias in node.names):
                        violations.append(f"{rel}:{node.lineno} future annotations import")
                if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                    violations.append(f"{rel}:{node.lineno} runtime star import")

    assert not violations


def test_deleted_current_references_do_not_return():
    stale = (
        "kd_sensing.cli.beambench_check_dataset",
        "configs/diagnostics/cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml",
    )
    allowed_context = ("已删除", "不要求", "退役", "历史", "deleted", "retired", "historical")
    current_paths = [
        ROOT / "README.md",
        ROOT / "docs/model_architecture_inventory.md",
        ROOT / "docs/extension_guide.md",
        *sorted((ROOT / "openspec/specs").glob("*/spec.md")),
    ]
    violations: list[str] = []
    for path in current_paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        text = "\n".join(lines)
        for fragment in stale:
            if fragment not in text:
                continue
            for index, line in enumerate(lines):
                if fragment not in line:
                    continue
                window = "\n".join(lines[max(0, index - 2) : index + 3]).lower()
                if not any(marker.lower() in window for marker in allowed_context):
                    violations.append(f"{_rel(path)}:{index + 1}: {fragment}")
    assert not violations


def test_internal_code_uses_owner_modules_not_retired_facades():
    violations: list[str] = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        for fragment in FORBIDDEN_IMPORTS:
            if fragment in text:
                violations.append(f"{_rel(path)}: {fragment}")
    assert not violations


def test_baseline_workflows_do_not_register_model_components():
    registry_fragments = (
        "@MODELS.register",
        "@ENCODERS.register",
        "@PROJECTORS.register",
        "@REPRESENTATION_CORES.register",
        "@HEADS.register",
    )
    violations: list[str] = []
    for path in sorted((SRC / "kd_sensing/baselines").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for fragment in registry_fragments:
            if fragment in text:
                violations.append(f"{_rel(path)}: {fragment}")
    assert not violations


def test_model_owners_do_not_depend_on_baseline_workflows():
    forbidden_fragments = (
        "from kd_sensing.baselines",
        "import kd_sensing.baselines",
        "from ..baselines",
        "from .baselines",
    )
    violations: list[str] = []
    for path in sorted((SRC / "kd_sensing/models").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                violations.append(f"{_rel(path)}: {fragment}")
    assert not violations


def test_lightweight_package_markers_do_not_grow_eager_barrel_exports():
    package_markers = (
        "src/kd_sensing/data/__init__.py",
        "src/kd_sensing/data/mmw/__init__.py",
        "src/kd_sensing/data/transform_ops/__init__.py",
        "src/kd_sensing/data/difficulty/__init__.py",
        "src/kd_sensing/diagnostics/__init__.py",
        "src/kd_sensing/engine/__init__.py",
        "src/kd_sensing/evaluation/__init__.py",
        "src/kd_sensing/losses/__init__.py",
        "src/kd_sensing/models/__init__.py",
        "src/kd_sensing/models/physics/__init__.py",
        "src/kd_sensing/preprocessing/__init__.py",
        "src/kd_sensing/utils/__init__.py",
    )
    forbidden_fragments = (
        "from kd_sensing.",
        "import kd_sensing.",
        "from .datasets",
        "from .jepa_",
        "from .image_ae_gps",
        "from .fusion",
        "from .normalization",
    )
    violations: list[str] = []
    for rel_path in package_markers:
        path = ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                violations.append(f"{rel_path}: {fragment}")
    assert not violations


def test_legacy_registry_fixture_stays_small():
    text = (ROOT / "tests/fixtures/legacy_model_registry_retirement.yaml").read_text(encoding="utf-8")
    assert text.count("\n  - name:") <= 16
    assert "migration_target:" not in text
    assert "error_hint:" not in text


def test_retired_route_mentions_are_contextualized():
    docs = [
        ROOT / "README.md",
        ROOT / "docs/project_surface_inventory.md",
        ROOT / "docs/research_notes.md",
        ROOT / "docs/agent_navigation.md",
    ]
    violations: list[str] = []
    for path in docs:
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not any(marker in line for marker in RETIRED_TEXT_MARKERS):
                continue
            window = "\n".join(lines[max(0, index - 2) : index + 3]).lower()
            if not any(marker.lower() in window for marker in RETIREMENT_CONTEXT):
                violations.append(f"{_rel(path)}:{index + 1}: {line.strip()}")
    assert not violations


def test_tracked_runtime_artifacts_are_not_in_source_control():
    tracked = set(_git_ls_files())
    forbidden_prefixes = ("outputs/", "logs/", "cache/", "outputs/cache/", ".pytest_cache/")
    codegraph_violations = [path for path in tracked if path.startswith(".codegraph/") and path != ".codegraph/.gitignore"]
    violations = [
        path
        for path in tracked
        if path.startswith(forbidden_prefixes) or "__pycache__/" in path or path.endswith(".pyc")
    ]
    dataset_violations = [path for path in tracked if path.startswith("dataset/") and path != "dataset/.gitkeep"]
    all_models_violations = [path for path in tracked if path.startswith("All_models/")]

    assert not codegraph_violations
    assert not violations
    assert not dataset_violations
    assert not all_models_violations


def test_tracked_text_has_no_secret_system_config_or_runner_hazards():
    violations: list[str] = []
    for rel_path in _git_ls_files():
        path = ROOT / rel_path
        if not path.is_file() or rel_path == _rel(Path(__file__).resolve()):
            continue
        if not _is_safety_scan_path(rel_path):
            continue
        violations.extend(_safety_violations(rel_path, path.read_text(encoding="utf-8", errors="replace")))
    assert not violations


@pytest.mark.parametrize(
    ("rel_path", "text", "expected"),
    [
        (
            "scripts/bad_container_bootstrap.sh",
            'printf \'PASSWD=kd-sensing-train --config configs/image/strong.yaml\' >> /root/.container_env',
            "credential field contains runtime command",
        ),
        (
            "scripts/bad_profile_bootstrap.sh",
            "echo 'nohup conda run -n kd_mm_beam kd-sensing-train --config run.yaml &' >> /etc/profile",
            "mutates protected system or authentication config",
        ),
        (
            "scripts/bad_auth_bootstrap.sh",
            "printf 'command=kd-sensing-train ssh-rsa AAAA...' > /root/.ssh/authorized_keys",
            "mutates protected system or authentication config",
        ),
        ("scripts/bad_cleanup.sh", "rm -rf outputs/unreviewed-run", "recursive delete lacks explicit confirmation"),
        (
            "configs/leaked.yaml",
            'token: "ghp_123456789012345678901234567890123456"',
            "potential secret literal",
        ),
    ],
)
def test_safety_guard_rejects_realistic_dangerous_fixtures(rel_path: str, text: str, expected: str):
    assert any(expected in violation for violation in _safety_violations(rel_path, text))


def test_safety_guard_accepts_conda_wrapped_project_runner():
    text = 'return ["conda", "run", "-n", "kd_mm_beam", "kd-sensing-train", "--config", config_path]'
    assert _safety_violations("scripts/good_runner.py", text) == []


def test_ponytail_followup_artifacts_are_not_tracked():
    tracked = {path for path in _git_ls_files() if (ROOT / path).exists()}
    forbidden = {
        "legacy_knowledge_decoupling_cleanup_manifest.json",
        "scripts/run_priority_v3_budget.sh",
    }
    metadata = [path for path in tracked if path.startswith("src/") and ".egg-info/" in path]

    assert forbidden.isdisjoint(tracked)
    assert not metadata


def test_current_openspec_specs_have_real_purpose():
    violations: list[str] = []
    placeholders = ("tbd", "created by archiving", "update purpose", "todo")
    for path in sorted((ROOT / "openspec/specs").glob("*/spec.md")):
        purpose = _purpose_section(path)
        lowered = purpose.lower()
        if not purpose or len(purpose) < 10 or any(marker in lowered for marker in placeholders):
            violations.append(_rel(path))
    assert not violations


def test_lifecycle_inventory_rows_reference_current_specs():
    spec_root = ROOT / "openspec/specs"
    spec_dirs = {path.parent.name for path in spec_root.glob("*/spec.md")}
    rows = re.findall(
        r"^\| `([^`]+)` \| `(current|supporting|retired-tombstone)` \|",
        INVENTORY.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    capabilities = [capability for capability, _lifecycle in rows]
    lifecycles = dict(rows)

    assert len(capabilities) == len(set(capabilities))
    assert set(capabilities) <= spec_dirs
    for capability in (
        "target-shot-domain-splitting",
        "model-architecture-summary",
        "local-missing-modality-baselines",
    ):
        assert lifecycles.get(capability) == "supporting"


def test_current_validation_commands_reference_existing_openspec_targets():
    active_changes = _active_openspec_change_names()
    current_specs = {path.parent.name for path in (ROOT / "openspec/specs").glob("*/spec.md")}
    valid_targets = active_changes | current_specs
    command_pattern = re.compile(r"openspec validate\s+(?!--all\b)(?P<target>[A-Za-z0-9_.<>{}-]+)\s+--strict")
    violations: list[str] = []

    for path in _current_validation_command_paths():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in command_pattern.finditer(line):
                target = match.group("target")
                if "<" in target or ">" in target:
                    continue
                if target not in valid_targets:
                    violations.append(f"{_rel(path)}:{line_number}: {match.group(0)}")

    index_text = (ROOT / "docs/maintainer_context_index.yaml").read_text(encoding="utf-8")
    assert "openspec validate --all --strict" in index_text
    assert not violations, "Use openspec validate --all --strict or a current spec/active change target."


def test_maintainer_context_index_is_minimal_and_resolvable():
    index_path = ROOT / "docs/maintainer_context_index.yaml"
    index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    routes = index.get("task_routes", [])
    route_ids = [route.get("id") for route in routes]

    assert len(route_ids) == len(set(route_ids))
    assert set(route_ids) == {
        "model",
        "data",
        "config",
        "cli",
        "diagnostics",
        "openspec",
        "documentation",
        "claims",
        "atlas",
    }
    assert not ({"entrypoints", "scripts", "configs", "hotspots", "remediation_waves", "project_skills"} & set(index))

    path_violations: list[str] = []
    command_violations: list[str] = []
    for route in routes:
        for field in ("context_path", "authority_paths", "owner_modules", "focused_validation", "retired_route_guard"):
            assert route.get(field), f"{route.get('id')}: missing {field}"
        for rel_path in (route["context_path"], *route["authority_paths"], *route["owner_modules"]):
            if not (ROOT / rel_path).exists():
                path_violations.append(f"{route['id']}: {rel_path}")
        for command in route["focused_validation"]:
            if not isinstance(command, str) or ("pytest" in command and not command.startswith("conda run -n kd_mm_beam ")):
                command_violations.append(f"{route['id']}: {command}")
        if not (ROOT / route["retired_route_guard"]).exists():
            path_violations.append(f"{route['id']}: {route['retired_route_guard']}")

    for group in index.get("protected_paths", []):
        assert group.get("id") and group.get("paths")
        for rel_path in group["paths"]:
            if not (ROOT / rel_path).exists():
                path_violations.append(f"protected:{group['id']}: {rel_path}")

    assert not path_violations
    assert not command_violations


def test_agent_context_portability_documents_are_thin_and_bounded():
    adapter_paths = (
        "CLAUDE.md",
        ".github/copilot-instructions.md",
        ".cursor/rules/kd-sensing-context.mdc",
        ".kiro/steering/agent-context.md",
        "docs/agent_project_knowledge.md",
    )
    forbidden_copies = (
        "## Requirements",
        "### Requirement:",
        "#### Scenario:",
        "| Route id |",
        "| claim_id |",
    )
    portability_docs = (
        *adapter_paths,
        "docs/readonly_agent_roles.md",
        "docs/current_research_brief.md",
        "docs/agent_memory_ledger.md",
    )
    missing_files = [rel_path for rel_path in portability_docs if not (ROOT / rel_path).exists()]
    adapter_violations: list[str] = []
    retired_mentions: list[str] = []
    copied_governance: list[str] = []
    for rel_path in adapter_paths:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        if len(text.splitlines()) > 80:
            adapter_violations.append(f"{rel_path}: exceeds thin-adapter limit")
        if "AGENTS.md" not in text or "docs/agent_navigation.md" not in text:
            adapter_violations.append(f"{rel_path}: missing authority navigation")
        for marker in RETIRED_TEXT_MARKERS:
            if marker in text:
                retired_mentions.append(f"{rel_path}: {marker}")
        for marker in forbidden_copies:
            if marker in text:
                copied_governance.append(f"{rel_path}: {marker}")

    broken_references = _missing_agent_context_references(portability_docs)

    assert not missing_files
    assert not adapter_violations
    assert not retired_mentions
    assert not copied_governance
    assert not broken_references


def test_scene31_generated_yaml_is_not_tracked_surface():
    existing = [
        _rel(path)
        for root in (ROOT / rel_path for rel_path in SCENE31_MANIFEST_BACKED_CONFIG_ROOTS)
        for path in sorted(root.glob("*.yaml"))
    ]
    assert not existing


def test_scene31_retained_yaml_surface_is_explicitly_registered():
    actual = {
        _rel(path)
        for path in (ROOT / "configs/scene31").rglob("*.yaml")
        if path.is_file()
    }
    assert actual == SCENE31_RETAINED_YAML
    inventory = INVENTORY.read_text(encoding="utf-8")
    for rel_path in sorted(SCENE31_RETAINED_YAML):
        if rel_path.startswith("configs/scene31/templates/"):
            continue
        assert rel_path.split("/")[-1].split("_", 1)[0] in inventory or rel_path in inventory


def test_current_target_experiment_config_references_resolve():
    scan_paths = [
        ROOT / rel_path
        for rel_path in (
            "README.md",
            "configs/README.md",
            "docs/experiment_matrix.md",
            "docs/experiment_protocols.md",
            "docs/mainline_model_catalog.md",
            "docs/project_surface_inventory.md",
            "docs/result_claims_registry.md",
        )
    ]
    scan_paths.extend(sorted((ROOT / "configs/diagnostics").glob("*.yaml")))
    scan_paths.extend(sorted((ROOT / "openspec/specs").glob("*/spec.md")))
    path_pattern = re.compile(
        r"configs/(?:scene31|fusion/experiments/(?:jepa_image_gps|rbma_missing_workflow|"
        r"rbma_missing_workflow_strong_encoders))/[A-Za-z0-9_./{}*-]+\.ya?ml"
    )
    historical_markers = ("retired", "historical", "已删除", "退役", "历史")
    broken: list[str] = []

    for path in scan_paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in path_pattern.finditer(line):
                ref = match.group(0)
                if any(marker in ref for marker in ("*", "{", "}")):
                    continue
                if (ROOT / ref).exists():
                    continue
                if any(marker in line.lower() for marker in historical_markers):
                    continue
                broken.append(f"{_rel(path)}:{line_number}: {ref}")

    assert not broken


def test_scene31_34_encoder_ablation_surface_is_unified():
    generator = ROOT / "scripts/generate_scenes31_34_encoder_ablation.py"
    runner = ROOT / "scripts/run_scenes31_34_tinyvit_ablation.sh"
    runner_text = runner.read_text(encoding="utf-8")

    assert generator.exists()
    assert "--family" in generator.read_text(encoding="utf-8")
    assert "generate_scenes31_34_encoder_ablation.py" in runner_text
    assert 'GPUS=""' in runner_text
    assert "CUDA_VISIBLE_DEVICES=1" not in runner_text
    assert not (ROOT / "scripts/run_scenes31_34_patchvit_ablation.sh").exists()


def test_root_temp_runbooks_do_not_return():
    assert not (ROOT / "test.md").exists()


def test_internal_sources_do_not_import_public_facade_helpers():
    restricted_dirs = ("diagnostics", "engine", "data", "models", "losses", "evaluation")
    allowed = {
        ROOT / "src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py",
        ROOT / "src/kd_sensing/cli/jepa_gps_shortcut_benchmark.py",
    }
    fragments = (
        "from kd_sensing.diagnostics.jepa_gps_shortcut_benchmark import",
        "import kd_sensing.diagnostics.jepa_gps_shortcut_benchmark",
    )
    violations: list[str] = []
    for package in restricted_dirs:
        for path in (ROOT / "src/kd_sensing" / package).rglob("*.py"):
            if path in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            for fragment in fragments:
                if fragment in text:
                    violations.append(f"{_rel(path)}: {fragment}")
    assert not violations


def test_plain_pytest_files_do_not_insert_tests_path_bootstrap():
    violations: list[str] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line_number in _module_level_sys_path_insert_lines(tree):
            violations.append(f"{_rel(path)}:{line_number}")
    assert not violations, "Ordinary tests should use tests.<helper> imports or shared pytest bootstrap."


def test_deleted_active_openspec_changes_have_matching_archive_status():
    status_lines = _git_status_short()
    deleted_active_changes: set[str] = set()
    archive_changes: set[str] = set()

    for line in status_lines:
        status = line[:2]
        path = line[3:]
        if path.startswith("openspec/changes/archive/"):
            match = re.match(r"openspec/changes/archive/\d{4}-\d{2}-\d{2}-(?P<change>[^/]+)/", path)
            if match:
                archive_changes.add(match.group("change"))
            continue
        if not path.startswith("openspec/changes/") or "D" not in status:
            continue
        parts = path.split("/")
        if len(parts) >= 3 and not (ROOT / "openspec" / "changes" / parts[2]).exists():
            deleted_active_changes.add(parts[2])

    missing_archives = sorted(change for change in deleted_active_changes if change not in archive_changes)
    assert not missing_archives, (
        "Deleted OpenSpec active change directories must be paired with "
        f"openspec/changes/archive/<date>-<change>/ status entries: {missing_archives}"
    )


def test_retired_console_scripts_are_absent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    retired = (
        "kd-sensing-hist-beam-loso",
        "kd-sensing-run-amr-net-gps-image",
        "kd-sensing-run-jepa-msac",
        "kd-sensing-export-viewer-manifest",
        "kd-sensing-visualize-modalities",
        "kd-sensing-project-surface-doctor",
    )
    assert all(command not in pyproject for command in retired)


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def _python_sources() -> list[Path]:
    roots = (SRC, ROOT / "tests", ROOT / "scripts")
    ignored = {Path(__file__).resolve()}
    return [
        ROOT / rel_path
        for rel_path in _git_ls_files()
        if rel_path.endswith(".py")
        and (ROOT / rel_path).exists()
        and any((ROOT / rel_path).is_relative_to(root) for root in roots)
        and (ROOT / rel_path).resolve() not in ignored
    ]


def _git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()


def _git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()


def _active_openspec_change_names() -> set[str]:
    changes_root = ROOT / "openspec/changes"
    return {path.name for path in changes_root.iterdir() if path.is_dir() and path.name != "archive"}


def _current_validation_command_paths() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "docs/agent_navigation.md",
        ROOT / "docs/maintainer_context_index.yaml",
        ROOT / "docs/project_surface_inventory.md",
        *sorted((ROOT / "openspec/specs").glob("*/spec.md")),
    ]


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _purpose_section(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^## Purpose\s*(?P<body>.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    return "" if match is None else match.group("body").strip()


def _missing_agent_context_references(rel_paths: tuple[str, ...]) -> list[str]:
    path_pattern = re.compile(
        r"`(?P<path>"
        r"AGENTS\.md|README\.md|pyproject\.toml|Makefile|"
        r"docs/[^`]+|openspec/specs/[^`]+|configs/[^`]+|"
        r"src/[^`]+|tests/[^`]+|scripts/[^`]+|\.codex/skills/[^`]+"
        r")`"
    )
    broken: list[str] = []
    for rel_path in rel_paths:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in path_pattern.finditer(line):
                ref = match.group("path")
                if any(marker in ref for marker in ("<", ">", "{", "}", "*", "|")):
                    continue
                if not (ROOT / ref).exists():
                    broken.append(f"{rel_path}:{line_number}: {ref}")
    return broken


def _is_safety_scan_path(rel_path: str) -> bool:
    roots = ("src/", "tests/", "scripts/", "configs/", "docs/", "openspec/", ".github/")
    root_files = {"AGENTS.md", "README.md", "Makefile", "pyproject.toml"}
    suffixes = {".md", ".py", ".sh", ".yaml", ".yml", ".toml", ".txt", ".json"}
    return (rel_path.startswith(roots) or rel_path in root_files) and Path(rel_path).suffix in suffixes


def _safety_violations(rel_path: str, text: str) -> list[str]:
    violations: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lowered = stripped.lower()
        is_placeholder = any(marker in lowered for marker in ("placeholder", "example", "redacted", "dummy", "your_", "xxxx", "changeme"))
        if not is_placeholder and any(pattern.search(stripped) for pattern in SECRET_PATTERNS):
            violations.append(f"{rel_path}:{line_number}: potential secret literal")
        if any(path in stripped for path in PROTECTED_SYSTEM_PATHS) and SYSTEM_CONFIG_MUTATION_RE.search(stripped):
            violations.append(f"{rel_path}:{line_number}: mutates protected system or authentication config")
        if CREDENTIAL_POLLUTION_RE.search(stripped):
            violations.append(f"{rel_path}:{line_number}: credential field contains runtime command")
        if rel_path.startswith("scripts/") and "rm -rf" in stripped and "confirm" not in lowered:
            violations.append(f"{rel_path}:{line_number}: recursive delete lacks explicit confirmation")

    if rel_path.startswith("scripts/") and "kd-sensing-" in text:
        if rel_path.endswith(".sh"):
            for line_number, line in enumerate(text.splitlines(), start=1):
                if "kd-sensing-" in line and "conda run -n kd_mm_beam" not in line:
                    violations.append(f"{rel_path}:{line_number}: project CLI bypasses kd_mm_beam")
        elif rel_path.endswith(".py"):
            conda_list = re.compile(
                r"['\"]conda['\"]\s*,\s*['\"]run['\"]\s*,\s*['\"]-n['\"]\s*,\s*['\"]kd_mm_beam['\"]",
                flags=re.DOTALL,
            )
            if conda_list.search(text) is None:
                violations.append(f"{rel_path}: project CLI bypasses kd_mm_beam")
    return violations


def _is_sys_path_insert_expr(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "insert"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "path"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "sys"
    )


def _module_level_sys_path_insert_lines(tree: ast.Module) -> list[int]:
    line_numbers: list[int] = []
    for node in tree.body:
        candidates = [node]
        if isinstance(node, ast.If):
            candidates.extend(node.body)
            candidates.extend(node.orelse)
        for candidate in candidates:
            if _is_sys_path_insert_expr(candidate):
                line_numbers.append(candidate.lineno)
    return line_numbers
