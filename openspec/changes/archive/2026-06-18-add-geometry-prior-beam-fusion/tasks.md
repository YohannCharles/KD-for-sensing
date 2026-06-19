## 1. 实现前审计与最小边界

- [x] 1.1 阅读 `docs/agent_navigation.md`、`docs/maintainer_context_index.yaml`、本 change 的 `proposal.md`、`design.md` 和全部 delta spec，确认本次默认走 component baseline，不新增旧入口或 whole-model exception
- [x] 1.2 复核 `add-predictive-gps-query-advantage/results.md` 的失败结论、strict baseline 路径、sample_count、seed、P0-P5 与 advantage 指标，作为新 claim gate 的 baseline provenance
- [x] 1.3 梳理现有 GPS feature、beam label calibration、soft beam label、ModelOutput、prediction objectives、diagnostics manifest 和 strict benchmark 聚合路径，列出最小源码改动集合
- [x] 1.4 确认 `Image ResNet+GPS` strict checkpoint 可作为 teacher-guided stabilization 的 opt-in teacher，并记录不可使用 retired `kd_sensing.distillation` 子包或旧 KD 配置

## 2. Geometry Prior 组件与元数据

- [x] 2.1 新增 GPS geometry prior 组件，支持 GPS-Rel-Polar、relative Cartesian 或 calibrated angle 输入，并输出 `[B,T,num_classes]` 或 `[B,num_pred,num_classes]` prior logits
- [x] 2.2 支持 prior distribution 归一化、entropy、top-k、availability mask 和 unavailable reason diagnostics
- [x] 2.3 增加 prior label-space 校验，确保 num beams、class order、beam label mapping fingerprint 与 beam head 一致；不一致时抛出清晰错误或要求显式 mapping
- [x] 2.4 将 geometry prior 作为 `modular_sequence` component baseline 接入 registry/config，不新增 root script、不复制训练循环、不注册 whole-model exception
- [x] 2.5 写出 geometry prior run metadata，包括 feature mode、normalization/scaler artifact、calibration source、history/source window、label space、prior mode 和 fallback mode
- [x] 2.6 增加 synthetic forward tests，覆盖 prior logits shape、distribution sum、metadata、fallback 和 label-space mismatch

## 3. Logit-Level Fusion 与 Reliability Isolation

- [x] 3.1 实现 geometry-prior logit fusion 组件，接收 image/fusion logits、geometry prior logits 和 opt-in reliability/uncertainty fields，输出 fused logits 与 branch diagnostics
- [x] 3.2 保证默认 `assistive` 模式下 prior 不完全替代 image branch，且 clean high-observability 条件不强制降低 image 权重
- [x] 3.3 实现 uncertainty/evidence weighting，至少支持 branch entropy、GPS valid/counterfactual/delay、image observability、prior-image disagreement 的可配置输入
- [x] 3.4 增加 condition id isolation 测试，确保 `condition`、`predictive_condition_id`、`gps_condition`、`image_condition`、`c_idx`、`d_idx` 不进入 fusion input tensor
- [x] 3.5 增加 ordinary baseline ignore-metadata 测试，确认 `Image ResNet+GPS`、`JEPA GPS-query k=4` 或 dummy baseline 可忽略 geometry/reliability metadata
- [x] 3.6 使用 `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py tests/test_gps_conditioned_jepa.py -q` 或等价 focused tests 跑通 forward/batch metadata 最小验证

## 4. DBA-Aware Loss 与 Teacher-Guided Stabilization

- [x] 4.1 新增 DBA-aware supervised beam loss 配置，支持 circular Gaussian soft CE、distance-aware CE 或第一版选定的 beam topology smoothing，并保持 evaluation 使用 hard `target_beam`
- [x] 4.2 保证 loss 日志使用 `loss/beam_*`、`loss/geometry_*` 或 `loss/teacher_guidance` 等非 retired 命名，不生成旧 `loss/distillation` 或 `loss/kd_soft_label`
- [x] 4.3 实现 teacher-guided stabilization 的 opt-in 配置，加载显式 teacher checkpoint logits/probabilities，记录 teacher provenance、temperature、weight、detach policy 和 enabled split
- [x] 4.4 确保 teacher-guided stabilization 不导入、不要求、不恢复 `kd_sensing.distillation` 子包、旧 KD CLI、旧 KD YAML 或兼容 wrapper
- [x] 4.5 增加 loss 单元测试，覆盖 hard-label fallback、soft target distribution、DBA-aware loss metadata、teacher-guidance loss 和 validation hard-label metric
- [x] 4.6 使用 `conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py tests/test_config_load_characterization.py -q` 跑通 objective/config 最小验证

## 5. Clean-First Curriculum 与配置矩阵

- [x] 5.1 新增 geometry prior prior-only、image-only control、logit fusion、DBA-aware loss、teacher-guided stabilization 和 mixed curriculum 配置
- [x] 5.2 为 H5/G2/F1、scene32-34、future=1、seed=17 写出 strict comparison manifest，包含 `Image ResNet+GPS`、`JEPA GPS-query k=4` 和每个 geometry-prior candidate
- [x] 5.3 实现 clean-first curriculum 配置，记录 clean/P-suite/advantage 采样比例、schedule、seed、condition list 和 difficulty digest
- [x] 5.4 增加 clean regression gate：candidate 的 P0/clean DBA 相对 strict `Image ResNet+GPS` 下降超过阈值时，claim status 必须 failed 或 pending
- [x] 5.5 增加 strict comparability 校验，history window、GPS source window、prediction horizon、scene set、seed、metric profile、distance metric 和 beam label space 不一致时拒绝 claim upgrade
- [x] 5.6 使用 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py tests/test_config_load_characterization.py -q` 跑通 manifest/config focused tests

## 6. Diagnostics 与报告产物

- [x] 6.1 扩展 diagnostics 聚合，输出 prior standalone DBA/Top-K/entropy、prior-target distance、prior-image agreement、prior-teacher agreement 和 fused improvement/degradation
- [x] 6.2 输出 branch uncertainty/evidence/weights 表，并按 condition、split、seed、model group 分组
- [x] 6.3 输出 strict comparison table、P0-P5 margins、advantage margins、clean regression gate summary 和 claim gate summary
- [x] 6.4 确保图表只从真实 CSV/JSON 聚合字段生成；字段缺失时标记 unavailable，不生成伪图或占位 claim
- [x] 6.5 将所有真实 CSV/PNG/JSON/checkpoint/TensorBoard 产物写入 ignored `outputs/analysis/geometry_prior_beam_fusion/` 或 manifest 指定目录
- [x] 6.6 增加 diagnostics focused tests，覆盖 missing diagnostics 字段、manifest links、claim gate failed/pending/pass 三类状态

## 7. 小消融与真实实验执行

- [x] 7.1 使用 `conda run -n kd_mm_beam` 跑 geometry prior prior-only smoke，记录 prior standalone DBA/Top-K/entropy 和 GPS feature provenance
- [x] 7.2 使用 `conda run -n kd_mm_beam` 跑 clean-only logit fusion smoke，确认 P0/clean DBA 未出现明显 regression 后再继续
- [x] 7.3 使用 `conda run -n kd_mm_beam` 跑 DBA-aware loss 与 teacher-guided stabilization 的小矩阵 ablation，选择不会破坏 clean/P0 的候选
- [x] 7.4 使用 `conda run -n kd_mm_beam` 对入选候选跑 H5/G2/F1、scene32-34、future=1、seed=17 strict training，并保存 checkpoint provenance
- [x] 7.5 使用 `conda run -n kd_mm_beam` 对 strict models 跑真实 P0-P5 evaluation，输出 strict comparison table 和 clean regression gate
- [x] 7.6 使用 `conda run -n kd_mm_beam` 跑 GPS advantage slice evaluation，输出 per-condition metrics、branch diagnostics 和 claim gate
- [x] 7.7 汇总真实实验结果到 ignored outputs 和 OpenSpec/results artifact；若未超过 baseline，明确标记 failed/pending 而不是升级 claim

## 8. 最终验证与文档同步

- [x] 8.1 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认未新增旧入口、未恢复 retired KD 子包、未破坏模块边界
- [x] 8.2 运行相关 focused tests：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_prediction_objectives.py tests/test_jepa_gps_shortcut_benchmark.py -q`
- [x] 8.3 运行 `openspec validate add-geometry-prior-beam-fusion --strict` 和 `openspec status --change add-geometry-prior-beam-fusion`
- [x] 8.4 若新增 mainline/paper reproduction 结果，按治理要求同步 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和 `docs/experiment_matrix.md`
- [x] 8.5 最终说明列出实现文件、配置、测试命令、strict metrics、claim gate 结论、diagnostics 路径和未解决风险
