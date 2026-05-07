## 1. 接口兼容与边界确认

- [x] 1.1 新增模型输出适配 helper，统一处理现有 `(pred, input_features, output_features)` 和 CRAF dict 输出。
- [x] 1.2 更新 `trainer.py` 使用输出适配 helper 提取 logits、训练 feature、蒸馏 feature 和 diagnostics。
- [x] 1.3 更新 `validator.py` 使用同一输出适配 helper，确保评估 CRAF 时只消费 logits。
- [x] 1.4 为输出适配 helper 增加单元测试，覆盖旧三元组输出、CRAF dict 输出和缺失可选 diagnostics。
- [x] 1.5 确认 CRAF 默认预测 slot 数与 `prepare_labels()` 的 `num_pred + 1` 标签语义一致。

## 2. CRAF 模型模块

- [x] 2.1 在 `src/kd_sensing/models/fusion/` 下新增 CRAF 相关模块，保持现有 fusion 代码可导入边界清晰。
- [x] 2.2 实现启用模态分支编码，复用现有 image、radar、LiDAR feature extractor，并为 GPS/mmWave 使用轻量 projector。
- [x] 2.3 实现 tokenization、time embedding、modality embedding 和 `[B, K, T, D]` token padding mask。
- [x] 2.4 实现 `UniModalHead` 和 entropy/margin confidence 计算。
- [x] 2.5 实现 `ReliabilityEstimator`，包含 modality representation、confidence、可选 dataset prior、mask 处理和 `min_gate`。
- [x] 2.6 实现 token-level Transformer fusion，确保 `key_padding_mask=True` 的 token 被忽略。
- [x] 2.7 实现 horizon query prediction head，默认输出 `num_pred + 1` 个 slot 和 `num_classes` 类 logits。
- [x] 2.8 实现 `craf_fusion` forward，返回 logits、reliability、effective modality mask、unimodal logits、confidence 和 fusion memory diagnostics。
- [x] 2.9 实现 token-only transformer fusion baseline，复用 tokenizer 和 Transformer，但不启用 reliability gate。
- [x] 2.10 将 CRAF 与 baseline 模型注册到 `MODELS`，并更新默认组件导入路径。

## 3. Loss 与反事实训练

- [x] 3.1 实现 sequence cross entropy helper，返回 scalar loss 和 per-sample loss，并支持 `ignore_index=-100`。
- [x] 3.2 实现 beam-aware soft label loss，支持 sigma、circular 距离和 ignore index。
- [x] 3.3 实现 modality dropout mask 生成 helper，支持 `drop_prob` 和 `min_keep`。
- [x] 3.4 实现 counterfactual drop mask 生成，支持 `sample_one` 和 `leave_one_out`。
- [x] 3.5 实现 full-forward 与 drop-forward 的 per-sample loss 差异到 gate target 的转换 helper。
- [x] 3.6 在训练流程中按配置组合 task loss、beam soft loss、unimodal auxiliary loss 和 gate loss。
- [x] 3.7 在 `training.counterfactual.start_epoch` 之前跳过反事实监督，并保持普通训练路径可用。
- [x] 3.8 确保非 CRAF 模型或 CRAF 附加 loss 权重为 0 时，训练 loss 语义保持现有行为。

## 4. 配置与实验入口

- [x] 4.1 新增 all-modalities CRAF no-KD 示例配置，使用当前统一 split、scene、seq length、num pred 和 output 结构。
- [x] 4.2 新增 image+radar CRAF no-KD 示例配置，用于和既有 image+radar fusion baseline 横向比较。
- [x] 4.3 新增 token transformer fusion baseline 示例配置，关闭 reliability 和 counterfactual gate loss。
- [x] 4.4 新增 CRAF 相关默认字段说明，包括 `model.student.reliability`、`training.modality_dropout`、`training.counterfactual` 和 `loss.beam_soft`。
- [x] 4.5 确认 CRAF 配置仍通过现有 config loader 和 `experiment.task: fusion` 构建 dataloader、model、loss、optimizer 和 scheduler。
- [x] 4.6 确认 legacy `fusion_teacher`、`fusion_student`、single-modal 和 KD 配置不被 CRAF 默认字段隐式改变。

## 5. 测试覆盖

- [x] 5.1 增加 CRAF 模型构建测试，覆盖五模态、双模态和非法模态配置。
- [x] 5.2 增加 CRAF forward 形状测试，覆盖 logits、reliability、unimodal logits、confidence 和 mask 输出。
- [x] 5.3 增加 token padding mask 测试，确认 force drop 的模态不会参与 attention 和 gate 贡献。
- [x] 5.4 增加 reliability estimator 测试，覆盖 unavailable modality、`min_gate` 和 dataset prior 开关。
- [x] 5.5 增加 beam soft loss、sequence CE、modality dropout 和 counterfactual gate target helper 测试。
- [x] 5.6 增加训练流程测试，使用 synthetic 或小 batch 验证 CRAF 完成 forward、loss、backward 和 optimizer step。
- [x] 5.7 增加评估流程测试，确认 CRAF checkpoint 或模型实例能产出 Top-K、DBA 和 loss metrics。
- [x] 5.8 增加回归测试，确认既有 fusion 和单模态测试仍通过。

## 6. 运行验证

- [x] 6.1 使用 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_student_configs.py` 验证旧配置边界不回退。
- [x] 6.2 使用 `conda run -n kd_mm_beam pytest` 运行完整测试套件。
- [x] 6.3 使用 `conda run -n kd_mm_beam python -m kd_sensing.cli.train --config <craf-smoke-config> --override training.epochs=1 data.dataloader.train_batch_size=2 data.dataloader.test_batch_size=2` 或等价命令完成 CRAF 短训练 smoke test。
- [x] 6.4 使用 `conda run -n kd_mm_beam python -m kd_sensing.cli.evaluate --config <craf-smoke-config> --weights <checkpoint>` 或等价命令验证 CRAF 评估路径。
- [x] 6.5 使用 `conda run -n kd_mm_beam openspec status --change add-counterfactual-reliability-fusion` 确认 OpenSpec 状态可追踪。

## 7. 文档与实验说明

- [x] 7.1 更新 README 或扩展指南，说明 CRAF 的配置入口、适用问题和与 legacy fusion 的区别。
- [x] 7.2 记录推荐实验顺序：单模态、legacy early-concat fusion、token transformer baseline、CRAF no-KD、CRAF 反事实 gate ablation。
- [x] 7.3 说明第一阶段限制：真实缺失模态依赖未来 dataset mask 字段，当前主要通过 force mask、modality dropout 和 counterfactual drop 验证。
- [x] 7.4 说明 CRAF 与 KD 的组合策略：第一阶段优先 no-KD，KD 组合需要单独显式配置和后续验证。

## 8. CRAF 稳定化配置与 gate 调度

- [x] 8.1 扩展 CRAF 配置解析和默认值，支持 `training.warmup_epochs`、`training.counterfactual.ignore_delta_eps`、`training.counterfactual.use_ce_only`、`training.counterfactual.mode: context_marginal`、`loss.gate_ramp_epochs`、`loss.uni_weight_warmup`、`loss.uni_weight_after_warmup` 和 softmax gate 温度字段。
- [x] 8.2 在 CRAF forward 或训练 helper 中实现 warmup 阶段固定可用模态 gate 为 1，确保 counterfactual 起始 epoch 前 reliability estimator 不改变融合 token 幅值。
- [x] 8.3 实现 `model.student.reliability.gate_type: softmax`，只在 effective modality mask 的可用模态上归一化，并支持按可用模态数缩放、`min_gate` 和不可用模态 gate 清零。
- [x] 8.4 实现 gate temperature schedule，从 `gate_temperature_start` 退火到 `gate_temperature_end`，并将当前 temperature 暴露给 diagnostics。
- [x] 8.5 实现 gate loss ramp，使 counterfactual 起始 epoch 后的有效 gate 权重按 `loss.gate_ramp_epochs` 线性增加。

## 9. CE-only counterfactual target

- [x] 9.1 扩展 sequence CE/per-sample loss helper，提供专用于 counterfactual delta 的 CE-only 路径，并明确不混入 beam soft、unimodal auxiliary、KD 或 gate loss。
- [x] 9.2 将 gate target helper 改为基于 `ignore_delta_eps` 的二值 target 和 valid mask：正 delta 监督为有益、负 delta 监督为有害、阈值内样本-模态 pair 被忽略。
- [x] 9.3 新增 `context_marginal` mask 生成 helper，采样不含目标模态的上下文子集 `A`，并构造 `A ∪ {m}` 的对照 mask。
- [x] 9.4 在 trainer 的 CRAF counterfactual 路径中接入 `context_marginal`，支持 `num_drop_per_batch`、`min_keep` 和 `no_grad_drop_forward`。
- [x] 9.5 保留并回归 `sample_one` 与 `leave_one_out` 旧模式，确认新 helper 不改变旧配置行为。

## 10. 附加 loss 调度、日志与实验配置

- [x] 10.1 将单模态 auxiliary loss 接入 warmup/after 两段权重，支持 warmup-only 配置，并在日志中记录实际生效权重。
- [x] 10.2 将 beam soft loss 默认配置降权到稳定化实验建议值，同时保持权重为 0 时完全关闭。
- [x] 10.3 扩展 CRAF epoch diagnostics，记录每模态 `cf/delta_mean_*`、`cf/target_mean_*`、`cf/target_valid_rate_*`、gate temperature、gate loss 有效权重和附加 loss 有效权重。
- [x] 10.4 新增 all-modalities CRAF 稳定化配置，包含 warmup 25、CE-only counterfactual、ignore band、softmax gate、gate ramp、aux warmup-only 和 beam soft 低权重。
- [x] 10.5 新增最小消融配置：token transformer 无 gate、CRAF 无 counterfactual、CRAF 稳定化 gate、固定 GPS/mmWave 强 prior sanity check。
- [x] 10.6 更新 README 或实验说明，记录方案 2 的推荐实验顺序、关键日志判据和失败排查方式。

## 11. 测试与验证

- [x] 11.1 增加 softmax gate 单元测试，覆盖可用模态归一化、不可用模态 mask、`min_gate` 和温度退火。
- [x] 11.2 增加 CE-only delta、ignore band target、target valid mask 和 `context_marginal` mask helper 单元测试。
- [x] 11.3 增加训练流程测试，验证 warmup 阶段固定 gate、counterfactual 起始 epoch 后启用 gate loss、gate ramp 和 auxiliary warmup-only 行为。
- [x] 11.4 增加日志测试，确认 `train_log.json` 或等价 epoch metrics 包含每模态 delta、target、valid rate 和有效 loss 权重字段。
- [x] 11.5 使用 `conda run -n kd_mm_beam pytest tests/test_craf*.py tests/test_trainer*.py` 或等价定向测试验证 CRAF 稳定化路径。
- [x] 11.6 使用 `conda run -n kd_mm_beam pytest` 运行完整测试套件，确认 legacy fusion、单模态和 KD 路径不回退。
- [x] 11.7 使用 `conda run -n kd_mm_beam python -m kd_sensing.cli.train --config <craf-stabilized-config> --override training.epochs=1 data.dataloader.train_batch_size=2 data.dataloader.test_batch_size=2` 完成 CRAF 稳定化短训练 smoke test。
- [x] 11.8 使用 `conda run -n kd_mm_beam openspec status --change add-counterfactual-reliability-fusion` 确认 OpenSpec 状态可追踪。
