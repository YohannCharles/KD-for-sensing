import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/launch_mmw_dynamic_router_screen.py"
    spec = importlib.util.spec_from_file_location("dynamic_router_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_matrix_has_one_variant_supervision_pair_per_gpu() -> None:
    module = _module()
    assert len(module.CANDIDATES) == 8
    assert {item[1] for item in module.CANDIDATES} == {"patr", "h2r", "core", "unified_hpr"}
    for variant in {item[1] for item in module.CANDIDATES}:
        assert {item[2] for item in module.CANDIDATES if item[1] == variant} == {
            "label_topology",
            "beam_power",
        }


def test_candidate_subset_selection_preserves_requested_order() -> None:
    module = _module()
    selected = module.select_candidates("PATR-Power,CoRe-Power")
    assert selected == (
        ("PATR-Power", "patr", "beam_power"),
        ("CoRe-Power", "core", "beam_power"),
    )


def test_candidate_subset_selection_rejects_unknown_or_duplicate() -> None:
    module = _module()
    for value in ("PATR-Power,PATR-Power", "Unknown"):
        try:
            module.select_candidates(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected candidate selection to reject {value!r}.")


def test_candidate_config_is_router_only_and_claim_ineligible(tmp_path: Path) -> None:
    module = _module()
    source = {
        "experiment": {"name": "CurrentControl", "seed": 1},
        "data": {
            "dataset": {},
            "dataloader": {},
            "domain_balanced_sampling": {"num_samples": 3600},
        },
        "temporal_missing": {},
        "model": {"primary": {}},
        "loss": {
            "u_mask_beam_jepa": {
                "prototype_topology": {"id": "cyclic_index_v1"},
                "prototype_target_circular": True,
            }
        },
        "training": {},
        "output": {},
    }
    checkpoint = tmp_path / "last.pth"
    checkpoint.touch()
    config = module.build_candidate_config(
        source,
        name="H2R-Label",
        variant="h2r",
        supervision="label_topology",
        output_root=tmp_path / "out",
        panel_path=tmp_path / "panel.json",
        panel_checksum="a" * 64,
        source_checkpoint=checkpoint,
        source_sha256="b" * 64,
        batch_size=64,
        epochs=10,
    )
    assert config["model"]["primary"]["router_calibration_only"] is True
    assert config["training"]["optimizer"]["require_all_matched"] is True
    assert config["training"]["initialization_checkpoint"]["allowed_missing_prefixes"] == [
        "prototype_reliability_router"
    ]
    assert config["loss"]["u_mask_beam_jepa"]["dynamic_router"]["frame_rank_weight"] == 0.1
    assert config["data"]["dataset"]["include_router_utility_targets"] is False
    assert config["mmw_dynamic_router_screen"]["claim_eligible"] is False
    assert config["mmw_dynamic_router_screen"]["utility_numeric_policy"] == (
        "beam_power_float32_before_linear_normalization_v1"
    )
