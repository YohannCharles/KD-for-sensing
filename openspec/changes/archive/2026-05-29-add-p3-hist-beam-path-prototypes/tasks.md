## 1. P0 数据巡检与字段映射

- [x] 1.1 增强或新增 `scripts/inspect_dataset.py`，支持扫描 MMW root 下 CARLA sensor、Sionna channel/path 和 metadata 文件。
- [x] 1.2 为巡检输出 town、scenario、weather、camera/radar/gps/lidar/imu/depth、beam label、beam_power、CSI/channel 和 path-level parameter 可用性。
- [x] 1.3 实现 `data.field_map` 解析，支持 gain/delay/AoD/AoA/mask/pose 等 path 字段的默认候选与显式覆盖。
- [x] 1.4 为 path 文件字段摘要记录 shape、dtype、path axis 推断、unavailable reason 和实际使用的 field map。
- [x] 1.5 增加 fixture 或小样本测试，使用 `conda run -n kd_mm_beam pytest <focused inspect tests> -q` 验证巡检报告和 field map 覆盖。

## 2. P2/P3 Path Semantics Builder

- [x] 2.1 新增 `src/kd_sensing/data/mmw/path_semantics.py` 或等价窄模块，定义 `PathFeatureBuilder` 和 path params 统一内部 key。
- [x] 2.2 实现 complex gain 聚合、valid mask、path power、归一化 `q_p` 和缺失/无 valid path 的 unavailable 返回。
- [x] 2.3 实现 descriptor 字段：log_total_power、dominant_path_ratio、top3_path_mass、entropy、effective_num_paths、delay spread、AoD/AoA sin/cos、angular spread、可选 zenith spread 和 los_like_score。
- [x] 2.4 使用 circular statistics 处理 AoD/AoA spread，覆盖 `-pi/pi` 跳变边界测试。
- [x] 2.5 实现 `PathSemanticLabelBuilder` 的 `kmeans_path_descriptor`、`rule_path_pattern`、`radio_power` 和 `coarse` 模式。
- [x] 2.6 实现 source train only 的 StandardScaler/KMeans fit、artifact 保存、非 source train split transform 和 descriptor_dim 校验。
- [x] 2.7 增加 focused tests，使用 `conda run -n kd_mm_beam pytest <path semantics tests> -q` 验证 descriptor、KMeans artifact、rule fallback、radio/coarse baseline。

## 3. P1 Dataset 输出扩展与边界

- [x] 3.1 扩展 `src/kd_sensing/data/datasets/mmw.py` 或实际 MMW dataset 路径，使 `__getitem__` 可选返回 `path_params`、`path_descriptor`、`path_semantic_label` 和 `path_valid`。
- [x] 3.2 保留现有 camera/radar/gps/lidar/imu optional tensors、beam、beam_power、radio_semantic_label、scenario/town/weather/domain 和 sample_id 输出语义。
- [x] 3.3 调整 collate/mask 逻辑，允许 batch 中部分样本缺少 path fields，并保留 unavailable reason 或 valid mask。
- [x] 3.4 确保 channel、CSI、path_params 和 path_descriptor 不进入 enabled sensing modalities 或模型输入构建路径。
- [x] 3.5 实现 label_budget=0/unlabeled target dataloader 或 batch view 的敏感字段训练访问屏蔽。
- [x] 3.6 增加 dataset focused tests，使用 `conda run -n kd_mm_beam pytest <dataset path output tests> -q` 验证 path 输出、缺失 fallback 和输入模态边界。

## 4. P4 模型 Forward 与 BeamHead

- [x] 4.1 扩展 `src/kd_sensing/models/fusion/hist_beam.py`，在配置启用时新增 `path_head(c)`、可选 `path_attr_head(c)` 和 `path_embedding`。
- [x] 4.2 调整 BeamHead 构建，使其支持 `concat(c, s_star)` 与 `concat(c, s_star, e_path)` 两种输入维度。
- [x] 4.3 实现 source forward：`s_star=s`，启用 path condition 时用 `softmax(path_logits/tau)` 加权 path embedding。
- [x] 4.4 实现 target/adapted forward：`s_star=adapter(s)`，有 `mu_path_c` 时用 cosine assignment 加权 path embedding，否则 fallback 到 path_logits。
- [x] 4.5 保证 path_semantic_label 不作为 beam hierarchy parent，不实现 prototype-to-beam 或 path-class-to-offset 概率分解。
- [x] 4.6 增加 model focused tests，使用 `conda run -n kd_mm_beam pytest <hist beam model path tests> -q` 验证 beam_logits、path_logits、c、s 和旧 variant forward 兼容。

## 5. P13 第一阶段 Smoke

- [x] 5.1 使用本地 MMW 小样本或 fixture 运行 dataset inspection，命令必须使用 `conda run -n kd_mm_beam` 包裹 Python/pytest 入口。
- [x] 5.2 验证能从一个 Sionna path sample 构造 `path_descriptor`，并覆盖字段名不同的 `data.field_map`。
- [x] 5.3 验证能在 source train descriptor 上 fit KMeans path labels，并在非 source split 只 transform。
- [x] 5.4 验证 dataset 能返回 `path_descriptor` 和 `path_semantic_label`，缺失 path 时能按配置 fallback。
- [x] 5.5 验证 model forward 输出 beam_logits、path_logits、c、s，且 V0/V3/V5/V6 focused tests 仍通过。

## 6. P5 Source Training Loss

- [x] 6.1 扩展 HiST-Beam source loss 配置，加入 `lambda_path` 和 `lambda_path_reg`，默认值分别为 0.3 和 0.05。
- [x] 6.2 在 source batch 有合法 `path_semantic_label` 时计算 path CE，并记录有效样本 coverage。
- [x] 6.3 在 batch 有合法 `path_descriptor` 且模型输出 `path_attr_pred` 时计算 SmoothL1 path regression，并记录 regression MSE。
- [x] 6.4 保留旧 radio_semantic loss 作为 V6 baseline，保留旧 hierarchical beam loss 作为配置选项且不强制启用。
- [x] 6.5 增加 source loss focused tests，使用 `conda run -n kd_mm_beam pytest <hist beam loss tests> -q` 验证有/无 path target 的 loss gating。

## 7. P6 Source Path Prototypes

- [x] 7.1 扩展 `src/kd_sensing/engine/hist_beam_prototypes.py` 或等价 prototype 模块，支持生成 `mu_path_c`、`count_path` 和可选 `mu_path_descriptor`。
- [x] 7.2 在 source pretraining 后按需 forward source train split，基于 path_semantic_label 聚合 shared representation。
- [x] 7.3 保留 `mu_coarse_c` 和 `mu_radio_c` artifact 语义，summary 能区分 V5/V6/V8 使用的 prototype。
- [x] 7.4 默认不保存或不使用 source private prototype 对齐 target private；旧字段兼容时要求 `use_source_private_proto=false` 为默认。
- [x] 7.5 增加 prototype focused tests，使用 `conda run -n kd_mm_beam pytest <path prototype tests> -q` 验证 counts、空 class、artifact metadata 和 variant-aware 复用。

## 8. P7 Target Adaptation 与 P10 Leakage Guard

- [x] 8.1 扩展 target adaptation 配置，支持 `proto_type=none|coarse|radio_semantic|path`、`proto_tau`、`confidence_threshold`、`proto_warmup_epochs` 和 `target_proto_momentum`。
- [x] 8.2 实现 `proto_type=path` 的 `alpha_path=softmax(cosine(c, mu_path_c)/proto_tau)`、`k_hat`、confidence 和 assignment histogram。
- [x] 8.3 实现 target-private `nu_path_s`、`nu_count`、高置信 EMA 更新和 warmup 后 `L_proto_private`。
- [x] 8.4 新增或增强 leakage guard，使 label_budget=0 或 unlabeled target training 访问 beam、beam_power、CSI/channel、path_params、path_descriptor、path_semantic_label、radio_semantic_label 作为训练监督时直接 raise。
- [x] 8.5 支持 `label_budget>0` 的 labeled target beam supervision，并仅在 `allow_labeled_target_path_supervision=true` 时允许 labeled subset 使用 path supervision。
- [x] 8.6 在 `adapt_log.json` 记录 target beam/beam_power/CSI/path_params/path_label/radio_label 是否用于训练的布尔字段。
- [x] 8.7 增加 adaptation/leakage tests，使用 `conda run -n kd_mm_beam pytest <adaptation leakage tests> -q` 验证 label_budget=0 no leakage 和 few-shot 边界。

## 9. P8/P9 Inference、Metrics 与 Diagnostics

- [x] 9.1 调整 inference 流程，使 target adapted 测试按 `fusion -> Ec/Es -> adapter -> path assignment -> BeamHead -> argmax` 执行。
- [x] 9.2 启用 path condition 时优先使用 `mu_path_c` assignment，缺失 prototype 时 fallback 到 `path_logits/tau`。
- [x] 9.3 保留 Top-1/Top-3/Top-5、normalized received power、beam power loss dB、trainable parameter ratio 和 adaptation time。
- [x] 9.4 新增 path semantic accuracy、path descriptor regression MSE、prototype assignment confidence、prototype coverage per class 和 source-target path class histogram。
- [x] 9.5 在 LOSO summary 中增加 V5 coarse、V6 radio、V8 path、V7 full fine-tuning 和 path condition on/off 对比字段。
- [x] 9.6 增加 evaluation focused tests，使用 `conda run -n kd_mm_beam pytest <hist beam evaluation tests> -q` 验证缺失 beam_power/path labels 时输出 unavailable reason 而不是伪造 0。

## 10. P11/P12 配置与实验矩阵

- [x] 10.1 保留 `configs/hist_beam/v5_adapter_proto.yaml` 和 `configs/hist_beam/v6_radio_proto.yaml` 旧配置语义。
- [x] 10.2 新增 `configs/hist_beam/exp_v8_adapter_path_proto.yaml`、`exp_v8_path_condition_off.yaml`、`exp_v8_path_condition_on.yaml`、`exp_v8_path_kmeans.yaml` 和 `exp_v8_path_rule.yaml`。
- [x] 10.3 在 V8 默认配置中加入 `path_semantic.enabled=true`、`mode=kmeans_path_descriptor`、`num_path_classes=24`、`descriptor_dim=auto`、`fit_on_source_only=true`、`fallback_if_missing=radio_power` 和 `use_path_regression=true`。
- [x] 10.4 在模型配置中加入 `use_path_head=true`、`use_path_condition_in_beam_head=true` 和 `path_embed_dim=32`。
- [x] 10.5 在 loss 和 target adaptation 配置中加入 `lambda_path=0.3`、`lambda_path_reg=0.05`、`proto_type=path`、`proto_tau=0.1`、`confidence_threshold=0.75`、`proto_warmup_epochs=5`、`target_proto_momentum=0.9` 和 `allow_labeled_target_path_supervision=false`。
- [x] 10.6 更新实验矩阵 metadata，覆盖 V0 Flat source-only、V3 Shared-private source-only、V4 Adapter-only、V5 Adapter+coarse、V6 Adapter+radio、V8 Adapter+path 和 V7 Full fine-tuning baseline。

## 11. P13 第二阶段 Smoke 与回归

- [x] 11.1 运行 source training one epoch smoke，使用 `conda run -n kd_mm_beam <train command>` 或 `conda run -n kd_mm_beam pytest <source train smoke> -q`。
- [x] 11.2 验证 source training 后能保存 `mu_path_c`、`count_path` 和 path prototype metadata。
- [x] 11.3 运行 target adaptation `label_budget=0` smoke，使用 `conda run -n kd_mm_beam <adapt command>` 或 focused pytest，并断言 no target leakage。
- [x] 11.4 运行 evaluation smoke，验证 Top-K、NRP、beam power loss、path diagnostics、prototype confidence/coverage 输出。
- [x] 11.5 运行兼容性 focused tests，确认 V0/V3/V5/V6 配置、旧 checkpoint 加载和旧 summary 字段不受 V8 改动影响。
- [x] 11.6 运行 OpenSpec 校验：`openspec validate add-p3-hist-beam-path-prototypes --strict`。
- [x] 11.7 运行架构边界和相关 CLI 快速检查：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`、`conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`、`conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`。

## 12. 最终实验判据与写实结论

- [x] 12.1 已在当前可用的 MMW sunny/Town10 `scenario_loso` 矩阵上完成 V8 path prototype 与 V6 radio-semantic prototype 对比；本地 `dataset/MMW/data_availability.json` 只有 sunny/Town10 的 ready scenarios，无法形成真实 leave-one-town-out 或 weather-shift 结论，本 change 不声称已验证 town/weather 泛化。
- [x] 12.2 已验证 `label_budget=0`：V8 未优于 source-only 或 adapter-only。完整矩阵中 V8 budget0 top1=25.10%，低于 V3 source-only top1=28.03%，也低于 V4 adapter budget0 top1=25.97%；该项作为负结果记录。
- [x] 12.3 已记录 5/10 labels 结果：V8 在 top1 上未超过 V6 radio-semantic，5 labels top1=26.22% vs V6 29.49%，10 labels top1=27.97% vs V6 28.30%；但 V8 top3/top5 高于 V6，说明 path prototype 更改善候选集而非 single-best ranking。
- [x] 12.4 已验证 V8 接近或略优于 full fine-tuning baseline：V8 overall top1=26.43%、top3=60.81%，full fine-tuning top1=25.24%、top3=53.20%；V8 trainable ratio 约 0.28%，低于 2%，full fine-tuning 为 100%。
- [x] 12.5 已完成 path condition on/off 消融：`v8_path_proto` 相比 `adapter_path_proto` top1 +4.34pp、top3 +10.92pp、beam power loss -0.35 dB，说明 path condition 放入 beam head 在当前矩阵中有效。

### 12.x 额外诊断

- 当前少样本 target adaptation 存在负迁移风险：V8 budget0 adapted-source top1=-1.55pp，budget5=-0.43pp，budget10=+1.32pp；full fine-tuning 在 0/5 labels 下分别为 -4.03pp 和 -4.17pp，表现出明显过拟合或不稳定。
- path semantic head 仍弱：V8 path semantic accuracy 约 4.43%，接近 24 类随机水平；path prototype coverage 为 0.667，存在空类。
- `last_beam` diagnostic baseline top1 约 94.8%、top3 约 99.2%，说明当前 `num_pred=1` 的 beam temporal persistence 很强。后续若研究 sensor-assisted beam prediction，应明确是否允许历史 beam baseline，并优先评估 `gps+image+lidar+radar` 输入而不是 `mmwave` 输入。
