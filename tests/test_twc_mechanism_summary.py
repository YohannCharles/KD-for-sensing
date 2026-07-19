import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/summarize_twc_mechanism.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mechanism_summary_pairs_clean_missing_and_uses_circular_error() -> None:
    module = _module()
    rows = []
    for condition, prediction, features in ((0, 63, [1.0, 0.0]), (8, 62, [0.0, 1.0])):
        rows.append(
            {
                "method": "T2",
                "seed": 1,
                "domain_id": "sunny/scene",
                "sample_id": "sample-1",
                "condition_index": condition,
                "target": 0,
                "prediction": prediction,
                "prototype_neighbor_margin": 0.5,
                "output_features": features,
                "router_oracle_aligned": True,
            }
        )

    summary = module._condition_summary(rows, clean=0)
    drift = module._paired_drift(rows, clean=0, missing=8)

    assert {row["mean_physical_codebook_error"] for row in summary} == {1.0, 2.0}
    assert drift[0]["prediction_codebook_shift"] == 1
    assert drift[0]["feature_cosine_drift"] == 1.0
