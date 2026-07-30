import math
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from normalize_smsl_artifacts import _normalize_mask_rows, normalize_process_manifest  # noqa: E402


def test_only_full_mask_f2_nan_is_normalized() -> None:
    rows = [
        {"mask": "full", "f2_high20_g1_rate": math.nan, "f2_high20_top1": math.nan, "f2_risk_mean": math.nan},
        {"mask": "image_only", "f2_high20_g1_rate": 0.9, "f2_high20_top1": 0.1, "f2_risk_mean": 0.8},
    ]
    _normalize_mask_rows(rows)
    assert rows[0]["f2_applicable"] is False
    assert all(rows[0][field] == "not_applicable" for field in ("f2_high20_g1_rate", "f2_high20_top1", "f2_risk_mean"))
    assert rows[1]["f2_applicable"] is True

    csv_row = {"mask": "full", "f2_high20_g1_rate": "nan", "f2_high20_top1": "nan", "f2_risk_mean": "nan"}
    _normalize_mask_rows([csv_row])
    assert csv_row["f2_risk_mean"] == "not_applicable"


def test_nonfull_nonfinite_f2_value_fails_closed() -> None:
    rows = [{"mask": "image_only", "f2_high20_g1_rate": math.nan, "f2_high20_top1": 0.1, "f2_risk_mean": 0.8}]
    with pytest.raises(ValueError, match="Non-Full"):
        _normalize_mask_rows(rows)


def test_process_manifest_result_summary_is_normalized(tmp_path: Path) -> None:
    path = tmp_path / "process_manifest.json"
    path.write_text(
        '{"result_summary":{"masks":[{"mask":"full","f2_high20_g1_rate":NaN,"f2_high20_top1":NaN,"f2_risk_mean":NaN}]}}',
        encoding="utf-8",
    )

    normalize_process_manifest(path)

    payload = path.read_text(encoding="utf-8")
    assert "NaN" not in payload
    assert '"f2_applicable": false' in payload
