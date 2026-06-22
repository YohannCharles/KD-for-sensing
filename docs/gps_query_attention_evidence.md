# GPS-query Attention Evidence Package

该诊断是 `kd-sensing-jepa-visual-analysis` 的 opt-in 模式，用于从已有本地输出整理 GPS-query 有效性证据包。

运行形态：

```bash
conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis \
  --analysis-config configs/diagnostics/gps_query_attention_evidence_example.yaml \
  --output-dir outputs/analysis/gps_query_effectiveness_visualization/<run_id> \
  --force
```

输入是已有 metrics CSV、可选 benchmark manifest、可选 forward cache 和模型 provenance 字段。示例配置只指向 ignored 的 `outputs/analysis/...` 路径；不要把真实 metrics、figures、cache、checkpoint 或 case payload 提交进源码。

输出写入用户选择的本地产物目录：

- `evidence_manifest.json`
- `tables/paired_delta_by_condition.csv`
- `tables/anchor_comparisons.csv`
- `tables/case_selection.csv`
- `tables/claim_gate_summary.csv`
- `figures/p0_p5_delta_heatmap.png`
- `figures/scene_group_delta.png`
- `figures/evidence_attention/`
- `figures/evidence_cases/`
- `cases/*.json`

解释边界：paired ablation delta 是主证据。Attention heatmap、entropy、query diversity 和 case panel 只作为解释性诊断，不是因果证明。
