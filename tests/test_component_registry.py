from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_import_default_components_registers_cls_token_transformer_fusion():
    code = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
from kd_sensing.registries import MODELS, import_default_components
before = "cls_token_transformer_fusion" in MODELS.list()
import_default_components()
after = "cls_token_transformer_fusion" in MODELS.list()
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
    "class_name": type(model).__name__,
    "modalities": list(model.modalities),
}}, sort_keys=True))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)

    assert payload == {
        "before": False,
        "after": True,
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
