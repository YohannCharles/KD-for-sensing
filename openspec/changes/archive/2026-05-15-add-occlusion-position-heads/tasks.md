## 1. 数据目标与预处理

- [x] 1.1 在 `src/kd_sensing/data/transform_ops/mmwave.py` 增加读取 64-beam raw power、计算 `max_power` 和批量拟合遮挡阈值的 helper，并覆盖 NaN/Inf/维度错误。
- [x] 1.2 扩展 `src/kd_sensing/preprocessing/sequences.py`，支持 `include_position_targets` 输出 `future_gps1..future_gpsH` 和 `future_bs_gps1..future_bs_gpsH`。
- [x] 1.3 扩展 `src/kd_sensing/data/samples.py` 的 `SequenceSamples` 和列校验，按需解析 future GPS/BS GPS target 列。
- [x] 1.4 扩展 `DeepSense6GDataset`，在配置启用时返回 `occlusion_label`、`occlusion_valid`、`position_target` 和 `position_valid`。
- [x] 1.5 为遮挡阈值和位置 target scaler 增加训练 split 拟合、test split 复用和运行 artifact metadata 记录。

## 2. 模型辅助头

- [x] 2.1 在 `CLSTokenTransformerFusionNet` 增加 `auxiliary_heads` 配置解析，默认关闭并保持 beam-only 输出兼容。
- [x] 2.2 实现 `occlusion_head`，输出形状为 `[B, num_pred]` 的遮挡 logits。
- [x] 2.3 实现 `position_head`，输出形状为 `[B, num_pred, 2]` 的二维位置估计。
- [x] 2.4 确保 auxiliary heads 使用与 beam head 相同的 masked CLS/fused representation，并兼容 `force_modality_mask`。

## 3. 训练与评估

- [x] 3.1 在 batch/runtime 层增加 `prepare_auxiliary_targets()`，把辅助标签和 mask 搬到目标 device。
- [x] 3.2 新增多任务 loss helper，计算遮挡 BCE、位置 MSE、mask 后有效均值、`pos_weight: auto` 和加权总辅助 loss。
- [x] 3.3 将辅助 loss 接入 `trainer.py`，在 no-KD、logits KD、RKD 和现有扩展路径下都只叠加到 student 总 loss。
- [x] 3.4 扩展 `validator.py` 和 `evaluation/metrics.py`，输出遮挡 accuracy、blocked-class F1 和 position RMSE。
- [x] 3.5 扩展 train log、TensorBoard scalar、`training_outputs.npz` 和 final config/runtime metadata，记录辅助 loss、辅助指标和标签生成统计。

## 4. 配置与文档

- [x] 4.1 扩展 canonical config helper，提供五模态多任务 fusion recommended 配置或 overlay，启用 dataset targets 和 CLS-token auxiliary heads。
- [x] 4.2 为配置校验补充多任务依赖检查：位置 target source、模型辅助输出支持、遮挡 power vector 维度和 artifact 复用。
- [x] 4.3 更新相关 README 或扩展文档，说明遮挡标签分位数、future GPS target CSV 生成、loss 权重和推荐运行命令。

## 5. 测试与验证

- [x] 5.1 增加 mmWave power-stat 与遮挡标签生成单元测试，使用 `conda run -n kd_mm_beam pytest tests/test_mmwave_modality.py` 验证。
- [x] 5.2 增加序列预处理和 dataset 辅助目标 fixture 测试，使用 `conda run -n kd_mm_beam pytest tests/test_preprocessing_formats.py tests/test_training_io_workflow.py` 验证。
- [x] 5.3 扩展 CLS-token Transformer fusion 测试，覆盖辅助 head 输出 shape、默认关闭兼容和 `force_modality_mask`。
- [x] 5.4 增加训练/验证 smoke test，覆盖多任务 no-KD 路径、辅助 loss 日志和辅助 metrics 输出。
- [x] 5.5 运行回归测试 `conda run -n kd_mm_beam pytest tests/test_cls_token_transformer_fusion.py tests/test_mmwave_modality.py tests/test_lidar_modality.py tests/test_gps_modality.py tests/test_training_io_workflow.py`。
- [x] 5.6 运行 OpenSpec 校验 `openspec status --change add-occlusion-position-heads`，确认变更 apply-ready。
