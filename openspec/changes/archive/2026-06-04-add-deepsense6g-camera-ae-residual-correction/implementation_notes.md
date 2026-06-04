## Implementation Notes

### Task 1.1 input inventory

- `configs/deepsense6g_residual_fusion.yaml` already defines the r15 `mapping_disabled` GPS residual output contract, GPS fallback sigma, good/bad threshold, metric tolerances, and conservative ablation names.
- `src/kd_sensing/data/deepsense6g_residual.py` provides reusable `ratio_tag`, GPS prediction row selection, GPS logits discovery, circular Gaussian fallback prior, residual manifest CSV writing, GPS context feature construction, and a manifest-backed Dataset pattern.
- `src/kd_sensing/engine/deepsense6g_residual_fusion.py` provides reusable summary, prediction, correction event, candidate recall, comparison report, and plotting patterns for GPS-anchored residual replay.
- GPS v2 r15 rows are consumed from `outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep/r15/mapping_disabled/` with `predictions.csv`, `support_manifest.csv`, `summary_overall.csv`, optional `gps_logits.npy` / `logits.npy` / `pred_logits.npy`, and optional logits index files.
- Existing circular utilities in `src/kd_sensing/evaluation/metrics.py` already expose `signed_circular_residual`, circular distance, circular windows, good/bad labels, DBA summaries, and top-k circular metrics.

The camera residual implementation keeps these pieces as the source of truth and adds a separate camera manifest, AE stage, residual/gate model, reranker, and output root so the GPS residual workflow remains unchanged.
