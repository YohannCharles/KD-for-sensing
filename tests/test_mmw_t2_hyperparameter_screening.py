import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    path = ROOT / "scripts" / "launch_mmw_t2_hyperparameter_screening.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_screening_variants_keep_t2_protocol_and_apply_only_named_overrides(monkeypatch: pytest.MonkeyPatch):
    launcher = _load_script(monkeypatch)
    kwargs = {"seed": 1, "batch_size": 64, "epochs": 40}
    base = launcher.build_screening_config("H0-base", Path("outputs/screen"), **kwargs)
    bpa = launcher.build_screening_config("H1-BPA+", Path("outputs/screen"), **kwargs)
    sharp = launcher.build_screening_config("H2-BPA-sharp", Path("outputs/screen"), **kwargs)
    tail = launcher.build_screening_config("H3-mask-tail", Path("outputs/screen"), **kwargs)
    optimizer = launcher.build_screening_config("H4-optimizer", Path("outputs/screen"), **kwargs)
    kl = launcher.build_screening_config("H5-KL+", Path("outputs/screen"), **kwargs)

    assert base["model"]["primary"]["head_type"] == "prototype"
    assert base["loss"]["u_mask_beam_jepa"]["router_supervision"] == "oracle"
    assert base["loss"]["u_mask_beam_jepa"]["router_oracle_weight"] == 0.1
    assert base["training"]["epochs"] == 40
    assert "use_early_stopping" not in base["training"]
    assert "model_selection" not in base["training"]
    assert base["training"]["validation"]["interval_epochs"] == 5
    assert base["mmw_t2_hyperparameter_screening"]["development_only"] is True
    assert bpa["loss"]["u_mask_beam_jepa"]["lambda_proto"] == 0.25
    assert bpa["loss"]["u_mask_beam_jepa"]["lambda_modality_proto"] == 0.15
    assert bpa["temporal_missing"] == base["temporal_missing"]
    assert sharp["loss"]["u_mask_beam_jepa"]["beam_label_sigma"] == 1.5
    assert sharp["model"]["primary"]["beam_proto_temperature"] == 0.08
    assert tail["temporal_missing"]["train_temporal_missing_rates"].endswith("0.8,0.8")
    assert tail["temporal_missing"]["train_missing_drop_counts"].endswith("3,3")
    assert optimizer["training"]["optimizer"]["type"] == "adamw"
    assert optimizer["scheduler"]["T_0"] == 40
    assert kl["loss"]["u_mask_beam_jepa"]["superset_consistency"]["kl_weight"] == 0.5


def test_screening_batch_and_gpu_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    launcher = _load_script(monkeypatch)
    assert launcher.validate_batch_size(64) == 64
    with pytest.raises(ValueError, match="multiple of 16"):
        launcher.validate_batch_size(60)
    jobs = launcher.build_jobs(("H0-base", "H1-BPA+"), (0, 7), tmp_path, seed=1)
    assert [(job["variant"], job["gpu"]) for job in jobs] == [("H0-base", 0), ("H1-BPA+", 7)]
    with pytest.raises(ValueError, match="unique"):
        launcher.build_jobs(("H0-base", "H1-BPA+"), (0, 0), tmp_path, seed=1)


def test_probe_binding_and_common_batch_selection(monkeypatch: pytest.MonkeyPatch):
    launcher = _load_script(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert launcher._force_single_probe_gpu(7) == "7"
    assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert launcher.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    def probe(gpu: int, batch: int, *, status: str = "safe", visible: str | None = None):
        return {
            "status": status,
            "physical_gpu": gpu,
            "requested_batch_size": batch,
            "actual_batch_size": batch,
            "peak_reserved_fraction": 0.89,
            "cuda_visible_devices": str(gpu) if visible is None else visible,
            "visible_cuda_device_count": 1,
            "logical_cuda_device": 0,
        }

    selected = launcher.select_highest_common_safe_batch(
        {
            0: [probe(0, 64), probe(0, 128)],
            1: [probe(1, 64), probe(1, 128, visible="0,1")],
            7: [probe(7, 64), probe(7, 128, status="unsafe_memory_fraction")],
        }
    )
    assert selected == 64
