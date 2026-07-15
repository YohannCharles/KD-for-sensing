import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mmw_evaluator_supports_ablation_methods_without_changing_defaults(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    evaluation = _load_script("eval_mmw_all_weather_matrix.py")

    assert evaluation.METHODS == ("S1", "T2", "amber_full", "rmbp_mm")
    assert set(evaluation.T2_ABLATION_METHODS) <= set(evaluation.SUPPORTED_METHODS)
    for method in evaluation.T2_ABLATION_METHODS:
        assert evaluation.BASELINE_SCOPES[method]["reproduction_scope"] == "project_mainline_t2_ablation"


def test_evaluation_provenance_separates_prototype_router_and_metric_geometry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    evaluation = _load_script("eval_h5_p1_temporal_matrix_v1.py")
    cfg = {
        "experiment": {"ablation_id": "T2-Linear"},
        "model": {
            "primary": {
                "head_type": "prototype",
                "use_beam_prototype_alignment": True,
                "router_supervision": "oracle",
            }
        },
        "training": {
            "prototype_target_circular": False,
            "circular_beam_distance": True,
            "use_amber_cma_analogue": False,
        },
        "evaluation": {"dba_distance_mode": "circular", "metric_profile": "64_beam_circular_topk"},
    }

    provenance = evaluation._evaluation_provenance(cfg)

    assert provenance["prototype_target_geometry"] == "linear"
    assert provenance["router_oracle_geometry"] == "circular"
    assert provenance["training_beam_geometry"] == "linear+circular"
    assert provenance["dba_distance_mode"] == "circular"


def test_evaluation_provenance_separates_prototype_head_from_bpa_auxiliary(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    evaluation = _load_script("eval_h5_p1_temporal_matrix_v1.py")
    cfg = {
        "model": {
            "primary": {
                "head_type": "prototype",
                "use_beam_prototype_alignment": True,
                "router_supervision": "oracle",
            }
        },
        "training": {"use_beam_prototype_alignment": False, "circular_beam_distance": True},
        "loss": {"u_mask_beam_jepa": {"use_beam_prototype_alignment": False}},
        "evaluation": {"dba_distance_mode": "circular"},
    }

    provenance = evaluation._evaluation_provenance(cfg)

    assert provenance["prototype_head_enabled"] is True
    assert provenance["bpa_auxiliary_enabled"] is False
    assert provenance["prototype_enabled"] is False
    assert provenance["prototype_target_geometry"] == "not_applicable"
    assert provenance["router_oracle_geometry"] == "circular"


def test_feature_extractor_accepts_all_t2_ablation_methods(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    analysis = _load_script("analyze_mmw_fused_feature_geometry.py")
    parser = analysis.build_parser()

    for method in analysis.T2_ABLATION_METHODS:
        args = parser.parse_args(["extract", "--method", method])
        assert args.method == method
