## Why

当前配置驱动重构后的默认模型参数、checkpoint 加载策略和训练恢复行为与上游原代码及随附 `All_models` 权重不一致，导致复现实验时可能静默使用随机初始化的 GRU 层或与原论文参数漂移。现在需要把原代码支持的 image-only 与 image+radar 实验路径恢复为可显式复现的兼容语义；同时，radar-only、GPS-only 和 LiDAR-only 是本项目新增单模态入口，也需要与 image 单模态保持一致的默认参数，避免单模态之间产生无意的配置漂移。

## What Changes

- 新增原代码兼容和单模态一致性配置契约，覆盖 image/radar/GPS/LiDAR 单模态与 image+radar fusion 的 teacher、student、no-KD、logits KD 和 RKD 配置。
- 将单模态配置的 GRU 层数统一调整为与 image 单模态一致：teacher/student 均使用 1 层 GRU；image+radar fusion 仍按原代码使用 teacher 2 层 GRU、student 1 层 GRU。
- 将 image 单模态复现实验配置中的 batch size、seed、learning rate、temperature、alpha、RKD 权重、weight decay、scheduler、early stopping 等参数对齐随附 `All_models/params_Image*.txt` 与上游 `train_image.py`；radar/GPS/LiDAR 单模态在共享字段上继承同名 image 单模态配置参数。
- 将 image+radar fusion 复现实验配置中的参数对齐随附 `All_models/params_Both*.txt` 与上游 `train_both.py`。
- 改造 teacher、评估和恢复训练的 checkpoint 加载行为：默认严格加载并报告 missing/unexpected keys；`resume` 需要真正恢复模型、optimizer、scheduler、epoch 和 best loss。
- 明确 image、radar 输入尺寸边界：原代码兼容路径固定使用 image `224x224`、radar `128x64`，配置中暴露但不能被当前架构安全支持的尺寸参数必须校验或文档化限制。
- 更新 README、扩展文档和测试，移除“所有默认配置统一二层 GRU”的说明，改为区分原代码兼容配置与新增模态/扩展配置。

## Capabilities

### New Capabilities
- `original-code-compatibility`: 约束原代码兼容配置、单模态参数一致性、权重加载、恢复训练和固定输入尺寸边界，确保 image-only 与 image+radar 路径能按上游代码/随附参数复现，并确保新增 radar/GPS/LiDAR 单模态与 image 单模态保持一致。

### Modified Capabilities
- `experiment-workflow`: 调整默认训练与评估行为契约，不再要求所有默认 image/radar/GPS/LiDAR/fusion 配置统一使用 `[64, 64, 2]`，而是要求所有单模态配置按 image 单模态参数保持一致，image+radar fusion 按上游 GRU 层数和复现实验参数配置。

## Impact

- 影响配置文件：`configs/image/*.yaml`、`configs/radar/*.yaml`、`configs/gps/*.yaml`、`configs/lidar/*.yaml`、`configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml`、`configs/fusion/image_radar_*.yaml` 以及共享默认配置。
- 影响训练与评估：`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/evaluator.py`、checkpoint 工具和配置解析/校验路径。
- 影响数据/模型边界：image transform 的 motion mask 尺寸、image/fusion teacher FC 输入假设、radar RA/DA 输入尺寸假设。
- 影响测试和文档：现有断言二层 GRU 的测试需要按单模态一致性矩阵与 image+radar 兼容矩阵改写，并补充 checkpoint mismatch、resume 和固定尺寸校验测试。
