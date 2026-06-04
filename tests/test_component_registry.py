from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_import_default_components_registers_cls_token_transformer_without_hist_beam_fusion():
    code = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
from kd_sensing.registries import MODELS, import_default_components
before = "cls_token_transformer_fusion" in MODELS.list()
import_default_components()
after = "cls_token_transformer_fusion" in MODELS.list()
hist_after = "hist_beam_fusion" in MODELS.list()
model = MODELS.build({{
    "type": "cls_token_transformer_fusion",
    "modalities": ["gps", "mmwave"],
    "feature_size": 16,
    "d_model": 16,
    "num_classes": 8,
    "num_pred": 2,
    "num_heads": 4,
    "num_layers": 1,
    "gps_input_size": 3,
    "mmwave_input_size": 64,
}})
print(json.dumps({{
    "before": before,
    "after": after,
    "hist_after": hist_after,
    "class_name": type(model).__name__,
    "modalities": list(model.modalities),
}}, sort_keys=True))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)

    assert payload == {
        "before": False,
        "after": True,
        "hist_after": False,
        "class_name": "CLSTokenTransformerFusionNet",
        "modalities": ["gps", "mmwave"],
    }


def test_registry_light_import_does_not_eager_import_cls_transformer_module():
    code = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
import kd_sensing.registries
print(json.dumps({{
    "fusion_module": "kd_sensing.models.fusion.cls_token_transformer" in sys.modules,
    "models_package": "kd_sensing.models" in sys.modules,
}}, sort_keys=True))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)

    assert payload == {"fusion_module": False, "models_package": False}


@pytest.mark.parametrize(
    ("registry_name", "cfg"),
    [
        ("MODELS", {"type": "craf_fusion"}),
        ("MODELS", {"type": "marf_fusion"}),
        ("MODELS", {"type": "hist_beam_fusion"}),
        ("LOSSES", {"type": "g2d"}),
        ("LOSSES", {"type": "logits_kd"}),
        ("LOSSES", {"type": "rkd"}),
        ("DATASETS", {"type": "multimodal_nf"}),
        ("PREPROCESSORS", {"type": "multimodal_nf_audit"}),
        ("PREPROCESSORS", {"type": "multimodal_nf_index"}),
        ("PREPROCESSORS", {"type": "multimodal_nf_derived_cache"}),
    ],
)
def test_retired_components_raise_removed_registry_errors(registry_name: str, cfg: dict):
    code = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
from kd_sensing import registries
try:
    getattr(registries, {registry_name!r}).build({cfg!r})
except registries.RegistryError as exc:
    print(json.dumps({{"message": str(exc)}}))
else:
    raise AssertionError("retired component unexpectedly built")
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
    message = json.loads(result.stdout)["message"]

    assert cfg["type"] in message
    assert "retired" in message or "Removed component" in message


def test_retired_hist_config_path_and_overrides_fail_fast():
    sys.path.insert(0, str(SRC))
    from kd_sensing.config.io import load_config

    with pytest.raises(ValueError, match="HiST-Beam/Hist research line has been retired"):
        load_config(ROOT / "configs" / "hist_beam" / "quick_smoke.yaml")
    with pytest.raises(ValueError, match="HiST-Beam/Hist research line has been retired"):
        load_config(overrides=["model.primary.type=hist_beam_fusion"])
    with pytest.raises(ValueError, match="HiST-Beam/Hist research line has been retired"):
        load_config(overrides=["hist_beam.enabled=true"])
