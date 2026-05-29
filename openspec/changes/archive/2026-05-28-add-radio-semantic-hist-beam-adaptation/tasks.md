## 1. Radio-semantic 标签与数据契约

- [x] 1.1 新增 `RadioSemanticLabelBuilder`，支持 `coarse`、`peak_spread`、`kmeans_power` 配置入口、finite 校验、fallback diagnostics 和 class count 统计。
- [x] 1.2 为 `peak_spread` 实现 best beam、peak group、归一化 entropy、spread bin 和 `radio_semantic_label` 生成逻辑，覆盖多 horizon 或明确限制为当前预测 horizon。
- [x] 1.3 扩展 MMW dataset/runtime，使配置启用时返回可选 `radio_semantic_label`、`beam_power`、radio unavailable reason、sample id 和 domain metadata，但不把 channel/CSI/beam_power 自动作为 sensing 输入。
- [x] 1.4 扩展 MMW manifest 或 derived metadata，记录 radio label 可派生状态、builder mode/config version、beam_power path、label source 和 unavailable reason。
- [x] 1.5 增加 radio label builder 和 MMW dataset 单测，使用 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q` 或新增专用测试验证 fallback、NaN/Inf、class range 和 dataset sample 字段。

## 2. HiST-Beam 模型扩展

- [x] 2.1 扩展 HiST-Beam 配置解析，新增 `radio_semantic`、`num_radio_classes`、`use_radio_head`、`use_radio_condition_in_beam_head`、`radio_embed_dim`、`radio_tau` 和 `proto_type` 字段。
- [x] 2.2 在 `HistBeamFusionNet` 中新增 `radio_head(c)` 输出 `radio_logits`，并将 radio diagnostics 写入 forward 结果。
- [x] 2.3 新增可选 `radio_embedding` 与 radio-conditioned beam head 输入路径，确保关闭 radio condition 时旧 beam head 行为和 checkpoint 形状不受影响。
- [x] 2.4 新增 `v6_radio_proto` 或等价 variant metadata，并保持现有 `v6_full_finetune` 继续表示 full fine-tuning baseline。
- [x] 2.5 增加模型 forward 单测，使用 `conda run -n kd_mm_beam pytest tests/test_hist_beam*.py -q` 验证 beam logits、radio logits、shared/private/adapter representations 和 radio condition off/on 形状。

## 3. Loss 与防泄漏

- [x] 3.1 扩展 batch 搬运和训练 extension，使合法 `radio_semantic_label` 能进入 source loss，缺失时只跳过 radio loss。
- [x] 3.2 在 `hist_beam_losses.py` 中新增 radio semantic CE、coverage diagnostics 和 loss 权重合成，避免把 radio label 当作 hierarchical parent。
- [x] 3.3 新增 target adaptation leakage guard，在 `label_budget=0` 或 unlabeled batch 中禁止 supervised beam/radio/power loss 读取 target labels。
- [x] 3.4 在 adapt log / metrics 中记录 `used_target_labels`、`used_target_beam_power_for_training`、`used_target_radio_label_for_training`。
- [x] 3.5 增加 loss/leakage 单测，使用 `conda run -n kd_mm_beam pytest tests/test_hist_beam*.py -q` 验证 source radio loss 生效、缺失 label no-op 可诊断、0-label target 不读取真实 label。

## 4. Source radio prototypes

- [x] 4.1 扩展 source prototype generator，按 `radio_semantic_label` 聚合 shared representation，保存 `mu_radio_c`、`count_radio` 和 `prototype_space=shared_radio_semantic`。
- [x] 4.2 保留 coarse/private/adapter prototypes 作为 V5 baseline artifact，并在 metadata 中区分 radio 与 coarse counts、label mode、thresholds、source/target domain 和 seed。
- [x] 4.3 扩展 prototype artifact validation，兼容旧 coarse artifact 和新 radio artifact，并对空 radio class 记录 unavailable/empty class diagnostics。
- [x] 4.4 增加 prototype 单测，使用 `conda run -n kd_mm_beam pytest tests/test_hist_beam*.py -q` 验证 radio prototype 聚合、empty class、artifact load/validate 和 V5/V6 artifact 区分。

## 5. Target radio adaptation

- [x] 5.1 扩展 adaptation strategy，支持 `proto_type=none|coarse|radio_semantic` 和 `v6_radio_proto` trainable 参数边界。
- [x] 5.2 实现 source radio prototype assignment：`alpha = softmax(cosine(c, mu_radio_c)/tau)`，记录 `r_hat`、confidence、coverage 和 used sample count。
- [x] 5.3 实现 target-private prototype bank `nu_radio_s/count`，按 confidence threshold、momentum、min count 和 warmup 进行 EMA 更新。
- [x] 5.4 实现 target-private prototype loss，使 `s_adapt` 对齐 stop-gradient target bank，而不是默认对齐 source private prototypes。
- [x] 5.5 增加 adaptation 单测，使用 `conda run -n kd_mm_beam pytest tests/test_hist_beam*.py -q` 验证 radio assignment、EMA 初始化、warmup、coverage=0 no-op 和 trainable ratio。

## 6. Metrics、prediction 与 summary

- [x] 6.1 扩展 evaluation outputs，计算 radio semantic accuracy、normalized received power、beam power loss dB，并在缺失 beam_power/radio label 时记录 unavailable reason。
- [x] 6.2 扩展 predictions 文件字段，记录 sample id、true/pred beam、top-k、coarse true/pred、radio true/pred 或 radio unavailable reason。
- [x] 6.3 扩展 LOSO summary 和 quick validation conclusion，汇总 V5 coarse vs V6 radio、radio condition off/on、full fine-tuning baseline 的 accuracy、power、prototype、efficiency 和 inconclusive reason。
- [x] 6.4 增加 evaluation/summary 单测，使用 `conda run -n kd_mm_beam pytest tests/test_hist_beam_loso.py -q` 验证缺失指标不伪造、radio diagnostics 可汇总。

## 7. LOSO、采样与配置

- [x] 7.1 扩展 few-shot sampler，budget > 0 时优先 radio-semantic 分层，其次 coarse sector / relative azimuth bin，最后 deterministic random fallback。
- [x] 7.2 扩展 MMW radio smoke 和 scenario LOSO 配置，新增 V5 coarse prototype、V6 radio prototype、V6 radio condition off/on 和 full fine-tuning baseline 对比。
- [x] 7.3 确保 single-scene smoke 仍标记 `cross_scene_claim_allowed: false`，至少两个 ready scenario 后才生成 scenario-LOSO radio method claim。
- [x] 7.4 更新 README 或 docs 中的推荐验证顺序、radio semantic label 口径、target leakage 边界和 V5/V6/V7 对比说明。

## 8. 验证与 OpenSpec

- [x] 8.1 运行 `openspec validate add-radio-semantic-hist-beam-adaptation --strict`。
- [x] 8.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 8.3 运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q`。
- [x] 8.4 运行 `conda run -n kd_mm_beam pytest tests/test_hist_beam*.py tests/test_hist_beam_loso.py -q`。
- [x] 8.5 对 MMW radio smoke 配置运行 plan-only 和最小 execute smoke，确认 loader、forward、loss、prototype、adaptation、summary 和 leakage metadata 均写出。
