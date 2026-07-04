import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from kd_sensing.config.io import deep_merge, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.diagnostics.gps_query_evidence import DEFAULT_GPS_QUERY_EVIDENCE_CONFIG


ANALYSIS_VERSION = "jepa_visual_analysis_suite_v1"
DEFAULT_CASE_GROUPS = ("query_gain", "query_regression", "shared_near_miss", "shared_failure")
DEFAULT_FIGURES = {
    "embedding": True,
    "error_anatomy": True,
    "attention": True,
    "case_studies": True,
    "robustness": True,
}
DEFAULT_OUTPUT_FORMATS = ("png", "svg")
DEFAULT_ATTENTION_FAITHFULNESS_CONFIG = {
    "enabled": False,
    "patch_ratio": 0.1,
    "patch_count": None,
    "selection_groups": ["top_attention", "low_attention", "random"],
    "occlusion_strategy": "zero",
    "random_seed": 42,
    "max_cases": 32,
    "metric_target": "dba_contribution",
}


def load_analysis_config(
    analysis_config: str | Path,
    *,
    output_dir: str | Path | None = None,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    path = Path(analysis_config)
    text = path.read_text(encoding="utf-8")
    raw = safe_load_yaml(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"JEPA visual analysis config must be a mapping: {path}")
    cfg = deep_merge(_default_analysis_config(), raw)
    if overrides:
        cfg = deep_merge(cfg, parse_overrides(overrides))
    if output_dir is not None:
        cfg.setdefault("outputs", {})["output_dir"] = str(output_dir)
    _validate_analysis_config(cfg, path=path)
    cfg["_analysis_config_path"] = str(path)
    cfg["_analysis_config_digest"] = _sha1_text(text)
    return cfg


def _default_analysis_config() -> dict[str, Any]:
    return {
        "models": {},
        "split": {
            "evaluation_split": "test",
            "scenes": None,
            "horizon_index": 0,
        },
        "sampling": {
            "seed": 42,
            "max_samples": None,
            "max_embedding_samples": 3000,
            "max_attention_cases": 256,
            "case_groups": list(DEFAULT_CASE_GROUPS),
            "cases_per_group": 3,
            "near_distance_threshold": 2,
            "far_distance_threshold": 5,
        },
        "figures": dict(DEFAULT_FIGURES),
        "attention_faithfulness": deepcopy(DEFAULT_ATTENTION_FAITHFULNESS_CONFIG),
        "embeddings": {
            "layers": ["output_features"],
            "method": "umap",
            "neighbors": 10,
        },
        "robustness": {
            "drop_modalities": True,
            "gps_noise": {"enabled": False, "std": []},
            "image_masking": {"enabled": False, "ratios": [], "mode": "random"},
            "seed": 42,
        },
        "benchmark": {
            "manifest": None,
            "runner_manifest": None,
            "metrics_by_condition": None,
            "robustness_summary": None,
            "case_studies": ["jepa_recovery", "gps_shortcut_failure", "shared_failure"],
        },
        "outputs": {
            "output_dir": "outputs/visual_analysis/jepa",
            "formats": list(DEFAULT_OUTPUT_FORMATS),
            "dpi": 180,
        },
        "evidence": deepcopy(DEFAULT_GPS_QUERY_EVIDENCE_CONFIG),
    }


def _validate_analysis_config(cfg: dict[str, Any], *, path: Path) -> None:
    models = cfg.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"models must be a mapping in {path}.")
    for name, spec in models.items():
        if not isinstance(spec, dict):
            raise ValueError(f"models.{name} must be a mapping.")
    for section in (
        "split",
        "sampling",
        "figures",
        "attention_faithfulness",
        "robustness",
        "benchmark",
        "outputs",
        "evidence",
    ):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"{section} must be a mapping in {path}.")
    cfg["attention_faithfulness"] = _attention_faithfulness_cfg(cfg)
    formats = cfg.get("outputs", {}).get("formats", DEFAULT_OUTPUT_FORMATS)
    if isinstance(formats, str):
        formats = [formats]
    if "png" not in {str(item).lower() for item in formats}:
        cfg.setdefault("outputs", {})["formats"] = ["png", *list(formats)]


def _attention_faithfulness_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_ATTENTION_FAITHFULNESS_CONFIG)
    evidence = cfg.get("evidence", {}) if isinstance(cfg.get("evidence"), Mapping) else {}
    if isinstance(evidence.get("attention_faithfulness"), Mapping):
        merged.update(dict(evidence["attention_faithfulness"]))
    if isinstance(cfg.get("attention_faithfulness"), Mapping):
        merged.update(dict(cfg["attention_faithfulness"]))
    if isinstance(merged.get("selection_groups"), str):
        merged["selection_groups"] = [merged["selection_groups"]]
    if not merged.get("selection_groups"):
        merged["selection_groups"] = ["top_attention", "low_attention", "random"]
    return merged


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


__all__ = [
    "ANALYSIS_VERSION",
    "DEFAULT_ATTENTION_FAITHFULNESS_CONFIG",
    "DEFAULT_CASE_GROUPS",
    "DEFAULT_FIGURES",
    "DEFAULT_OUTPUT_FORMATS",
    "_attention_faithfulness_cfg",
    "_default_analysis_config",
    "_validate_analysis_config",
    "load_analysis_config",
]
