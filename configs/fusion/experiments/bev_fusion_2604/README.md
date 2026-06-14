# BEV-Fusion 2604 experiment configs

This directory contains the arXiv:2604.05668 BEV-Fusion reproduction family. Use the current protocol table and claim registry for report wording:

- `docs/experiment_protocols.md`
- `docs/result_claims_registry.md`

| config | status | protocol | key parameters | metric | caveat |
| --- | --- | --- | --- | --- | --- |
| `paper_full.yaml` | `formal` | DeepSense6G S32/S33/S34 stratified 80/10/10, `seq_len=5`, `num_pred=1` | seed 42, 100 epochs, batch 2, AdamW lr `1e-4`, 128x128 BEV, `d_model=256`, four modalities | `2604_linear_topk` | Paper exact split/seed/code/weights are unavailable, so reports must keep `paper_exact_split_available: false`. |
| `low_memory.yaml` | `lowmem / paper approximation` | inherits `paper_full.yaml` split and metric | reduced BEV/model width, batch 4, workers 0, AMP off | `2604_linear_topk` | Must be labeled `paper_approximation: true`; useful for resource-limited validation, not exact paper full. |
| `smoke.yaml` | `mock/smoke` | synthetic forward/config smoke | seed 7, 1 epoch, synthetic sequence, `mock_data: true` | `2604_linear_topk` schema only | Validates code path only; never report as paper or formal result. |
| `ablations/*.yaml` | `formal ablation` | inherits paper split and metric | one controlled modality/pathway change such as `without_lidar` or `gps_global_only` | `2604_linear_topk` grouped by `ablation_name` | Compare only against the matching `paper_full`/lowmem family and keep ablation labels explicit. |

Recommended checks:

```bash
conda run -n kd_mm_beam pytest tests/test_bev_fusion_2604.py -q
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/lidar_bev_cache.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/bev_fusion_2604/low_memory.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/bev_fusion_2604/paper_full.yaml
```

All generated checkpoints, metrics, reports, cache and latency summaries belong under ignored `outputs/`, `outputs/cache/` or `logs/`. Do not claim strict reproduction of paper exact results or H100 latency without the official split, author code, weights and comparable hardware.
