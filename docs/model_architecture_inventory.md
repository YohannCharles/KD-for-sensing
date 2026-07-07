# Model Architecture Inventory

Current public model surface is registry-driven and centered on `modular_sequence`, token/CLS transformer fusion, U-MaskBeamJEPA, MMW/CSI components and physics-informed MMW.

| Registry name | Lifecycle | Owner |
| --- | --- | --- |
| `u_mask_beam_jepa` | current mainline | `src/kd_sensing/models/u_mask_beam_jepa.py` |
| `modular_sequence` | current reusable model | `src/kd_sensing/models/modular.py` |
| `cls_token_transformer_fusion` | current exception | `src/kd_sensing/models/fusion/cls_token_transformer.py` |
| `token_transformer_fusion` | current exception | `src/kd_sensing/models/fusion/token_transformer.py` |
| `pinn_multimodal_beam` | protected MMW | `src/kd_sensing/models/pinn_multimodal_beam.py` and `src/kd_sensing/models/physics/` |
| CSI encoders | protected CSI | `src/kd_sensing/models/csi_encoder.py` |
| JEPA / downstream helpers | component support | `src/kd_sensing/models/jepa.py`, `src/kd_sensing/models/jepa_downstream.py` |

Retired model registrations `bev_fusion_2604`, `vision_position_late_fusion`, `vision_position_transformer_fusion` and `gps_sequence_baseline` remain only as registry removed-name guards with hints. They are not current model architecture rows.
