## Why

当前配置矩阵已经覆盖 image、radar、GPS、LiDAR 和部分 fusion 组合，但 teacher baseline、student no-KD、KD 配置的命名和覆盖范围不一致，容易把原仓库遗留的 teacher-as-student 残留误解为当前语义。现在需要把实验入口整理成可枚举、可测试、风格统一的配置矩阵，确保后续新增实验不会继续放大命名差异。

## What Changes

- 统一单模态配置语义：每个单模态都明确区分 teacher no-KD baseline、student no-KD baseline、logits KD 和 RKD。
- 为 image 补齐与 radar/GPS/LiDAR 一致的 teacher/student no-KD 命名入口；保留既有入口兼容，但文档必须说明推荐入口和兼容入口。
- 为 fusion 多模态配置补齐 `image`、`radar`、`gps`、`lidar` 的所有必要非单模态组合：6 个双模态、4 个三模态、1 个四模态。
- 为每个受支持 fusion 组合提供统一命名的 no-KD、logits KD 和 RKD 配置；teacher 和 student 的 `modalities` 必须一致。
- 标准化配置字段、run name、默认 checkpoint 来源、GPS/LiDAR 数据字段和 README 说明，避免原仓库旧训练脚本中的 teacher-as-student 残留继续影响配置驱动流程。
- 扩展配置构建测试，验证所有默认单模态和 fusion 组合均能构建正确的 teacher/student、distiller 和数据启用字段。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `experiment-workflow`: 明确并补齐单模态 teacher/student no-KD 与 KD 配置矩阵，规定统一命名、兼容入口和默认 teacher checkpoint 来源。
- `configurable-multimodal-fusion`: 将 fusion 配置入口扩展为 `image`、`radar`、`gps`、`lidar` 的所有必要多模态组合，并要求每个组合提供 no-KD、logits KD 和 RKD 配置。

## Impact

- 影响配置：`configs/image/`、`configs/radar/`、`configs/gps/`、`configs/lidar/`、`configs/fusion/`。
- 影响文档：`README.md`，必要时同步 `docs/extension_guide.md` 中的配置命名和模态组合说明。
- 影响测试：配置构建、KD checkpoint 来源、fusion `modalities` 组合、legacy 权重兼容说明相关测试。
- 不改变模型 forward、distiller 算法、dataset 数据格式或统一训练/评估入口的外部行为；新增配置应复用既有实现。
