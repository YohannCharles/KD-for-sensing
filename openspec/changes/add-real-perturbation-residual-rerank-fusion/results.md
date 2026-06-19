# add-real-perturbation-residual-rerank-fusion Results

## Implementation status

- Real-forward benchmark mode 已实现：manifest 支持 `evaluation.mode: real_forward`，runner 会按 model/condition/seed 真实调用 difficulty pipeline、执行模型 forward、缓存 logits/labels/metadata，并从真实 logits 计算 Top1/Top3/Top5/DBA。
- Safe residual reranker 已作为 `modular_sequence` opt-in component 接入：候选集来自 anchor top-k、geometry prior top-k、anchor 邻域和可选 teacher top-k；residual 有上界并受 reliability/no-regret gate 控制。
- Rerank auxiliary loss 已实现：candidate CE、pairwise margin 和 no-regret consistency 都是 opt-in，主 beam CE/focal 目标保持不变。
- Geometry-prior claim gate 已收紧：非 clean claim-scope rows 必须是 `evidence_scope=real_forward` 或 `status=real_forward` 才能 pass；delegated/synthetic/degradation evidence 只能 pending。
- 新增 reranker 配置矩阵和 strict real-forward manifest：训练/消融 YAML 位于 `configs/fusion/experiments/jepa_image_gps/`，strict manifest 位于 `configs/diagnostics/real_perturbation_residual_rerank_fusion_strict.yaml`。

## Verification

- `conda run -n kd_mm_beam pytest tests/test_geometry_prior_beam_fusion.py tests/test_prediction_objectives.py tests/test_modular_sequence_next_query_transformer.py tests/test_jepa_gps_shortcut_benchmark.py tests/test_config_load_characterization.py -q`  
  Result: 87 passed, 8 warnings.
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`  
  Result: 65 passed.
- `openspec validate add-real-perturbation-residual-rerank-fusion --strict`  
  Result: valid.
- `openspec status --change add-real-perturbation-residual-rerank-fusion`  
  Result: 4/4 artifacts complete.

## 2026-06-19 strict rerun

All six reranker training jobs completed under H5/G2/F1, scene32-34, future=1, seed=17. Best validation DBA: strict candidate 0.885110, prior-candidates 0.888174, no-regret ablation 0.886703, teacher-anchor 0.886397, clean-smoke 0.889951, anchor-only control 0.890257. The anchor-only control being strongest is an important caveat: the learned residual did not clearly beat a conservative anchor path in validation.

The strict real-forward benchmark completed with warnings empty and wrote results under `outputs/analysis/real_perturbation_residual_rerank_fusion/strict/`. Overall DBA from `geometry_prior_strict_comparison.csv`: Image ResNet+GPS 0.838535, JEPA GPS-query baseline 0.783888, geometry-prior logit fusion 0.741684, safe residual rerank strict candidate 0.839897. The geometry/rerank claim gate is `pass` because the strict candidate is +0.001362 DBA over Image ResNet+GPS overall while clean regression is only -0.002819 DBA, within the 0.02 clean gate.

Per-condition DBA shows the strict candidate mainly improves wrong-GPS and combined CxD cases versus GPS-query baseline, with mixed results versus Image ResNet+GPS: P3 +0.010478, A1 +0.009681, A2 +0.015196, C3+D7 +0.015319, but P1 -0.023346, P2 -0.008701, P5 -0.000858. The predictive 5-point claim gate remains unavailable/not applicable for this manifest; this run supports only the geometry/rerank strict gate, not a broad predictive GPS-query++ claim.

Diagnostics caveat: `geometry_prior_branch_weights.csv` reports `residual_magnitude_mean=0.0` and `rerank_changed_top1_rate=0.0` for the strict candidate across evaluated conditions. The pass should therefore be worded as a conservative safe-rerank/anchor-gated robustness result, not as evidence that the residual head materially changes beam ranking.

## Claim status

`geometry_prior_claim_gate.json` reports `claim_status: pass` for `safe_residual_beam_rerank_fusion` with real-forward evidence. `geometry_prior_fusion` remains pending because its strict overall DBA is below the Image ResNet+GPS baseline. `predictive_claim_gate.json` remains `unavailable` for the predictive GPS-query++ claim in this reranker manifest.

The implementation continues to prevent delegated clean-only or deterministic degradation rows from upgrading the geometry/rerank claim; only real-forward P-suite/advantage rows can satisfy the gate.

## Remaining work

- Decide whether to tune the residual gate/head, because the current strict pass is effectively anchor-safe and not residual-active.
- Add target-rank delta diagnostics once labels are available at diagnostics aggregation time.
- Extend shard matrix reporting from completed shards to explicit planned/missing rows if a future claim gate needs partial-grid audit output.
- If this claim is to become a mainline documented claim, sync the governed docs/catalogs with the caveat above.
