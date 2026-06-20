from pathlib import Path

from kd_sensing.config import load_config
from kd_sensing.diagnostics.jepa_visual_architecture_sweep import (
    REQUIRED_FAMILIES,
    load_sweep_manifest,
    strict_comparability_gate,
    summary_row_from_result,
    write_sweep_summary,
)
from kd_sensing.engine.artifacts import final_config_with_runtime


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/diagnostics/jepa_visual_architecture_sweep_manifest.yaml"


def test_jepa_visual_architecture_sweep_manifest_schema_and_commands():
    manifest = load_sweep_manifest(MANIFEST)

    families = {candidate["family"] for candidate in manifest["candidates"]}
    assert REQUIRED_FAMILIES <= families
    assert manifest["output_root"] == "outputs/analysis/jepa_visual_architecture_sweep"
    assert {
        "variant_id",
        "family",
        "visual_encoder",
        "pooler",
        "checkpoint_policy",
        "run_tier",
    } <= set(manifest["candidates"][0])
    commands = [entry["command"] for group in manifest["command_manifest"].values() for entry in group]
    assert commands
    assert all(command.startswith("conda run -n kd_mm_beam ") for command in commands)


def test_jepa_visual_architecture_sweep_config_matrices_load_and_metadata(tmp_path: Path):
    for name in ("smoke", "lowmem", "strict"):
        cfg = load_config(ROOT / f"configs/fusion/experiments/jepa_image_gps/architecture_sweep_{name}.yaml")
        assert cfg["output_root"].startswith("outputs/analysis/jepa_visual_architecture_sweep")
        assert cfg["strict_comparability"]["seed"] == 17
        assert cfg["candidates"]

    k_token_cfg = load_config(
        ROOT / "configs/fusion/experiments/jepa_image_gps/image_gps_jepa_k_token_pooler_smoke.yaml"
    )
    image_encoder = k_token_cfg["model"]["primary"]["encoders"]["image"]
    metadata = final_config_with_runtime(k_token_cfg, run_dir=tmp_path / "k_tokens")["runtime"]["jepa_downstream"]
    assert image_encoder["gps_query_pool"]["output_mode"] == "tokens"
    assert metadata["pooler_output_mode"] == "tokens"
    assert metadata["visual_token_encoder"]["checkpoint_policy"] == "exact_reuse"
    assert metadata["image_encoder"]["visual_token_encoder"]["token_count"] == 196


def test_jepa_visual_architecture_sweep_summary_and_claim_gate(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    baseline = {
        "run_tier": "strict",
        "evidence_scope": "strict",
        "checkpoint_policy": "exact_reuse",
        "split": "beambench",
        "scene_set": "s32_s34",
        "seed": 17,
        "history_window": 5,
        "gps_input_source_window": 2,
        "prediction_horizon": 1,
        "beam_label_space": "beam64",
        "metric_profile": "beambench_linear_topk",
        "distance_metric": "linear",
        "normalization_artifact": "config_resolved",
        "difficulty_digest": "strict_clean",
        "output_root": "outputs/analysis/jepa_visual_architecture_sweep",
        "stage1_checkpoint": "outputs/checkpoints/patch16.pth",
    }
    smoke = {
        **baseline,
        "variant_id": "patch8_stage1_gps_query",
        "family": "patch_granularity",
        "run_tier": "smoke",
        "evidence_scope": "smoke",
        "checkpoint_policy": "fresh_stage1_required",
        "stage1_checkpoint": "",
    }
    strict = {
        **baseline,
        "variant_id": "patch16_mean_baseline",
        "family": "baseline",
        "token_count": 196,
    }

    gated = strict_comparability_gate([smoke, strict], baseline=baseline)

    assert gated[0]["claim_eligible"] is False
    assert "non_strict_run_tier" in gated[0]["claim_gate_reason"]
    assert gated[1]["claim_eligible"] is True

    row = summary_row_from_result(
        candidate={"variant_id": "patch16_mean_baseline", "family": "baseline", "run_tier": "strict"},
        metrics={"top1": 0.5, "top3": 0.7, "top5": 0.8, "dba": 0.6, "adjacent_beam_error": 0.2},
        diagnostics={"status": "available", "attention_entropy": 1.2, "attention_peakiness": 0.4},
        provenance={"command": "conda run -n kd_mm_beam kd-sensing-evaluate --config cfg.yaml"},
    )
    paths = write_sweep_summary([{**gated[1], **row}], output_root="outputs/analysis/jepa_visual_architecture_sweep")

    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    assert Path(paths["json"]).read_text(encoding="utf-8").startswith("[")
