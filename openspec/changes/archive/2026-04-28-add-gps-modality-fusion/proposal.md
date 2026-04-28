## Why

当前项目已经支持 image、radar 和 image+radar fusion，但 Scenario 9 中的 GPS/position 数据尚未进入序列数据、模型或融合实验。已完成的六组 GPS 预处理对比显示，GPS-Rel-Polar 在 DBA 和 Top-5 上最稳，Top-1/Top-3 与相邻方案差距较小，因此本变更收敛为只交付 GPS-Rel-Polar，降低实现、配置和论文实验复杂度。

## What Changes

- 为 Scenario 9 增加 GPS 序列预处理与加载路径：从原始 `scenario9.csv` 中保留 UE/BS GPS 路径，读取经纬度，并构造 GPS-Rel-Polar 特征 `[dist, sin_theta, cos_theta]`。
- 新增 GPS 单模态预测架构，风格对齐当前 image/radar 模态：`GpsFeatureExtractor` + `GpsModalityNet` + `GpsStudentModalityNet`，并通过模型注册表和配置构建。
- 将 GPS 加入 fusion teacher/student，使 fusion 可以通过配置手动选择 `image`、`radar`、`gps` 的任意非空组合或全部模态。
- 增加 GPS-only 与多模态 fusion 的 no-KD、logits KD、RKD 配置模板；GPS 相关配置统一使用 GPS-Rel-Polar。
- 删除本 change 中 raw、UTM、relative、motion、motion-smooth 五类 GPS ablation 的交付要求、配置入口和文档说明；六组对比只作为选型依据保留在设计说明中。
- 保持现有 image-only、radar-only、image+radar 配置的默认行为兼容；未显式启用 GPS 时不要求数据集返回 GPS 张量。
- 增加聚焦单元测试，覆盖 GPS-Rel-Polar 特征构造、训练集 scaler 复用、模型注册构建、fusion 模态选择和旧配置兼容性。

## Capabilities

### New Capabilities

- `gps-preprocessing`: 定义 GPS 路径保留、经纬度读取、UE-BS 相对极坐标特征构造，以及 train-only scaler 约束。
- `gps-modality-model`: 定义 GPS teacher/student 单模态模型的注册名、输入输出契约和配置构建要求。
- `configurable-multimodal-fusion`: 定义 fusion teacher/student 通过配置选择 image、radar、gps 任意组合并保持蒸馏兼容的行为。

### Modified Capabilities

- `experiment-workflow`: 扩展训练、验证、评估和配置工作流，使 GPS-only 的 GPS-Rel-Polar 与可选模态 fusion 实验可通过统一脚本运行。

## Impact

- 影响数据层：`src/kd_sensing/data/samples.py`、`src/kd_sensing/data/datasets/scenario9.py`、`src/kd_sensing/data/transforms.py`、`src/kd_sensing/preprocessing/sequences.py`。
- 影响训练/验证输入准备：`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/validator.py`、`src/kd_sensing/engine/evaluator.py`。
- 影响模型层：新增 GPS 模型模块，更新 `src/kd_sensing/models/__init__.py` 与 `src/kd_sensing/models/fusion/networks.py`。
- 影响配置：新增 GPS-Rel-Polar 相关 `configs/gps/` 模板并扩展 `configs/fusion/`，必要时新增 GPS 预处理配置。
- 依赖影响：UTM 转换需要可用实现；优先使用轻量依赖 `utm`，若环境未安装则在依赖与安装说明中显式补齐。
