# Diagnostics Context

当前诊断面保留：

- `kd-sensing-eval-u-mask-matrix`：U-MaskBeamJEPA missing-modality evaluation matrix，输出到 ignored `outputs/eval/` 或显式本地目录。
- `kd-sensing-runs`：只读 run index。
- `kd-sensing-research-dashboard` / `kd-sensing-research-preview`：只读 claim/evidence 预览，不启动训练。
- `kd-sensing-paper-export`：只消费 reviewed claim/ledger/summary。
- `kd-sensing-mmw-town-gps-v2` 与 `kd-sensing-inspect-mmw-physics`：受保护 MMW workflow。
- `kd-sensing-project-surface-doctor`：只读治理检查。

历史 JEPA visual analysis、GPS shortcut benchmark、Scenario D/CxD 和 dataset audit CLI 已退役或删除；不要恢复 wrapper 或同名 console script。

Focused validation:

```bash
conda run -n kd_mm_beam pytest tests/test_project_surface_doctor.py tests/test_cli_help.py -q
conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa_eval_matrix.py -q
```
