# Experiment Protocols

| Protocol | Entry | Dataset/split | Metrics | Output boundary | Validation |
| --- | --- | --- | --- | --- | --- |
| final C2 / U-MaskBeamJEPA missing modality | `kd-sensing-train --config configs/fusion/u_mask_beam_jepa_smoke.yaml`; `kd-sensing-eval-u-mask-matrix --config configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml` | DeepSense6G configured scene or final C2 generated configs | Top-K, DBA/ADBA, missing-condition summaries | ignored `outputs/` and `logs/` | `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa_eval_matrix.py -q` |
| Scene31-34 main missing-modality local workflow | `scripts/run_scenes31_34_main.sh`, `scripts/generate_scenes31_34_main.py`, `python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact <artifact>` | Scene31/32/33/34 pooled local/manual | full/missing-count/per-scene/compute/paper-table caveats | ignored `outputs/scenes31_34_*` | `conda run -n kd_mm_beam pytest tests/test_missing_modality_stress.py -q` |
| MMW Town GPS v2 | `kd-sensing-mmw-town-gps-v2 --mode <run|plot|compare>` | MMW Town prepared local data | GPS-only v2 protocol summaries | ignored `outputs/analysis/mmw_town_gps_adapter_v2/` | `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py -q` |
| physics-informed MMW | `kd-sensing-inspect-mmw-physics --max-samples <n>` and physics configs | MMW/CSI local or synthetic smoke | physics supervision/sample inspection | stdout or ignored explicit output | `conda run -n kd_mm_beam pytest tests/test_physics_informed_mmw.py -q` |
| CSI hardening | configs under `configs/csi/` and `configs/fusion/csi_hardening_matrix/` | CSI hardening matrix | condition-level robustness | ignored `outputs/` | `conda run -n kd_mm_beam pytest tests/test_csi_modality.py -q` |
| paper export | `kd-sensing-paper-export --input <reviewed-ledger>` | reviewed claim/evidence only | Markdown/CSV/LaTeX draft tables | ignored `outputs/paper_artifacts/` or explicit output dir | `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q` |

Historical/pending protocols for Image+GPS JEPA, BeamBench, BEV-Fusion 2604, Vision-Position, WCL/TII audits and old RBMA/KD/BTAPA/weakKD sweep have been retired or deleted from current provenance. If a future paper needs one, open a new OpenSpec change and reintroduce only the protected evidence path needed.

## Invalidated Protocols

| Protocol | Status | Invalidated boundary | Required rerun gate |
| --- | --- | --- | --- |
| H5/P1 Scene31-34 temporal matrix before group-safe split enforcement | `not_comparable` | overlapping temporal windows were split per sample; all audited sequence groups crossed split boundaries, and history/target frame identities overlapped | group-safe sequence split artifact; pairwise-disjoint sample/history/target identities; independent validation and final test; train-only normalization fingerprint; complete seed/checkpoint/metric provenance |

修复前 H5/P1 数值只能用于定位协议问题，不得用于方法排序、统计结论或 paper main table。
