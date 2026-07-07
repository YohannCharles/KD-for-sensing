# Extension Guide

Post-C2 extensions should be narrow and current-surface first.

## Model Extensions

Prefer `modular_sequence` with registry components. Add a whole-model `MODELS` exception only when an active OpenSpec change or current spec explains why component composition is insufficient. Current protected exceptions include U-MaskBeamJEPA, token/CLS transformer fusion and physics-informed MMW.

Retired whole-model names such as `bev_fusion_2604`, `vision_position_late_fusion`, `vision_position_transformer_fusion` and `gps_sequence_baseline` are removed guards, not extension targets.

## Config Extensions

Use current config families:

- `configs/fusion/u_mask_beam_jepa_*.yaml`
- `configs/fusion/physics_informed_mmw*.yaml`
- `configs/csi/` and `configs/fusion/csi_hardening_matrix/`
- canonical supervised root configs

Do not add virtual configs for retired Image+GPS JEPA, BeamBench, BEV-Fusion 2604, Vision-Position, old KD/BTAPA/weakKD or residual routes.

## Validation

```bash
openspec validate --all --strict
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q
```
