## 1. 审计与边界确认

- [x] 1.1 阅读 `docs/agent_navigation.md`、`docs/maintainer_context_index.yaml`、本 change 的 proposal/design/specs，确认路线是 component baseline + diagnostics runner，不恢复旧 GPS residual、Top8 selector、camera residual、KD/HiST 或旧根脚本
- [x] 1.2 复核 `add-geometry-prior-beam-fusion/results.md` 中 clean metrics、pending claim gate、delegated clean-only warning 和 diagnostics 产物路径，作为本 change 的 baseline provenance
- [x] 1.3 梳理现有 evaluator、benchmark runner、difficulty pipeline、batch metadata filtering、ModelOutput adaptation 和 logits cache 代码路径，列出 real-forward 最小改动集合
- [x] 1.4 确认 `Image ResNet+GPS` strict checkpoint、geometry-prior checkpoints 和 JEPA GPS-query baseline checkpoint 都可作为 strict real-forward 输入，并记录 sample_count、seed、H5/G2/F1、scene32-34、future=1 口径

## 2. Real-Forward Benchmark 与 Cache

- [x] 2.1 为 benchmark manifest 增加 `evaluation.mode=real_forward` 或等价字段，支持 model/condition/seed/split shard 配置和 sample_count cap
- [x] 2.2 实现 real-forward runner path：按 condition 调用统一 difficulty pipeline 变换 batch，再执行模型 forward
- [x] 2.3 确保 real-forward metrics 从真实 logits 与 hard `target_beam` 计算 Top1/Top3/Top5/DBA，不使用 deterministic degradation 估算
- [x] 2.4 设计并实现 logits/labels/diagnostics cache schema，包含 model、condition、seed、split、sample_id、checkpoint fingerprint、difficulty digest 和 evidence scope
- [x] 2.5 实现 cache resume 与 fingerprint 校验；checkpoint/config/difficulty/sample_count 不一致时拒绝复用或标记 stale
- [ ] 2.6 输出 planned/completed/missing shard matrix，并在缺失 claim-scope shard 时让 claim gate 返回 pending
- [x] 2.7 增加 no-leak filtering，确保 condition id、target_test label、beam power oracle、future frame 不进入模型 forward 输入
- [ ] 2.8 增加 focused tests，覆盖 real-forward smoke、cache 复算 metrics、stale cache、missing shard 和 delegated_clean_only evidence scope

## 3. Safe Residual Rerank Component

- [x] 3.1 新增 safe residual reranker component，支持从 anchor logits、geometry prior logits、可选 teacher logits 构造候选 beam set
- [x] 3.2 实现 candidate builder：anchor top-k、prior top-k、anchor 邻域、可选 teacher top-k，并输出 candidate ids/source mask/coverage diagnostics
- [x] 3.3 实现 bounded residual/rerank head，只在 candidate set 内生成 residual score，并通过 mask 写回全 beam logits
- [x] 3.4 实现 no-regret fallback gate，支持 anchor confidence、image observability、GPS reliability、prior entropy 和 branch disagreement
- [x] 3.5 确保 clean/high-observability 条件可 fallback anchor，wrong GPS 或高 disagreement 时可降低 prior/rerank residual 权重
- [x] 3.6 将 reranker 作为 `modular_sequence` opt-in component 接入 registry/config，不新增 whole-model exception
- [ ] 3.7 输出 diagnostics：anchor logits、prior logits、rerank logits、candidate ids、selected source、target rank delta、fallback reason、gate confidence、condition_id_consumed=false
- [x] 3.8 增加 synthetic forward tests，覆盖 shape、candidate mask、bounded residual、fallback、condition id isolation 和 ordinary baseline ignore metadata

## 4. Loss、训练配置与矩阵

- [x] 4.1 新增 candidate CE / pairwise DBA margin / no-regret consistency loss helper，保留 hard-label CE 或 focal CE 主目标
- [x] 4.2 记录 loss metadata：candidate coverage、skipped samples、anchor correctness/DBA threshold、residual scale 和 no-regret loss 权重
- [x] 4.3 新增配置：anchor-only control、safe rerank clean smoke、prior candidate rerank、no-regret ablation、teacher-anchor ablation、strict real-forward rerank candidate
- [x] 4.4 配置必须声明 anchor source、candidate top-k、beam 邻域宽度、max residual scale、fallback policy、loss mode、diagnostics mode 和 output run_name
- [x] 4.5 确保 teacher/anchor checkpoint 只作为 opt-in stabilization 或 anchor provider，不导入 retired `kd_sensing.distillation` 或旧 KD YAML
- [x] 4.6 使用 `conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py tests/test_config_load_characterization.py -q` 或等价 focused tests 验证 loss/config

## 5. Diagnostics 与 Claim Gate

- [ ] 5.1 扩展 geometry-prior diagnostics 聚合，输出 candidate recall、anchor target rank、prior target rank、rerank target rank、changed-top1 rate、beneficial/neutral/harmful change counts
- [ ] 5.2 输出 no-regret clean summary：anchor DBA、rerank DBA、delta、fallback rate、harmful clean changes 和 clean regression gate
- [ ] 5.3 输出 branch diagnostics 表：image/anchor/prior/rerank entropy、agreement、residual magnitude、fallback reason、real_forward shard provenance
- [x] 5.4 更新 geometry-prior claim gate：只有 real-forward P-suite/advantage metrics 可以 pass；delegated/synthetic/degradation evidence 必须 pending/unavailable
- [x] 5.5 更新 benchmark outputs manifest，列出 logits cache、condition metrics、branch diagnostics、claim gate、planned shard matrix 和 warnings
- [ ] 5.6 增加 diagnostics tests，覆盖 missing fields unavailable、delegated_clean_only pending、real_forward pass/fail、clean regression failed 和 missing shard pending

## 6. Smoke 与 Strict 实验

- [x] 6.1 使用 `conda run -n kd_mm_beam` 跑 real-forward clean smoke，验证 `Image ResNet+GPS` clean metrics 与上一轮 0.8857 DBA 口径一致
- [x] 6.2 使用 `conda run -n kd_mm_beam` 跑 P0/P1 小子集 real-forward benchmark，确认 difficulty transform、cache、metrics 和 no-leak diagnostics 生效
- [x] 6.3 使用 `conda run -n kd_mm_beam` 跑 safe rerank clean smoke，要求 clean DBA 不低于 anchor 超过阈值，记录 candidate coverage 和 fallback rate
- [x] 6.4 使用 `conda run -n kd_mm_beam` 跑 reranker ablation 小矩阵，比较 candidate top-k、residual scale、no-regret loss 和 prior inclusion
- [x] 6.5 使用 `conda run -n kd_mm_beam` 跑 H5/G2/F1、scene32-34、future=1、seed=17 strict training，保存 checkpoint provenance
- [x] 6.6 使用 `conda run -n kd_mm_beam` 跑 strict real-forward P0-P5 evaluation，输出 per-condition metrics、logits cache 和 no-regret diagnostics
- [x] 6.7 使用 `conda run -n kd_mm_beam` 跑 strict GPS advantage real-forward evaluation，输出 advantage margins、branch diagnostics 和 claim gate
- [x] 6.8 汇总真实实验结果到 ignored outputs 和 OpenSpec/results artifact；若未超过 baseline，明确标记 failed/pending，不升级 claim

## 7. 验证与归档准备

- [x] 7.1 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认没有旧入口、retired residual/KD/Top8 回流或新增 root script
- [x] 7.2 运行 real-forward benchmark focused tests 和 reranker focused tests，至少覆盖 runner/cache/diagnostics/model forward
- [x] 7.3 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_jepa_gps_shortcut_benchmark.py tests/test_geometry_prior_beam_fusion.py -q`
- [x] 7.4 运行 `openspec validate add-real-perturbation-residual-rerank-fusion --strict` 和 `openspec status --change add-real-perturbation-residual-rerank-fusion`
- [ ] 7.5 若 strict real-forward claim 通过，按治理要求同步 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和 `docs/experiment_matrix.md`；若 pending/failed，仅记录结果 artifact 和 caveat
- [ ] 7.6 最终说明列出实现文件、配置、测试命令、strict metrics、claim gate 结论、diagnostics/cache 路径和未解决风险
