## Why

当前 image 与 fusion 训练配置把 `model.student.type` 配成了 teacher 架构，导致默认训练与评估流程没有使用已实现的轻量 student。已有 `All_models/*Std*.pth` 权重和参数记录显示实验结果对应的是 `ImageStudentModalityNet` 与 `StudentModalityNet`，当前配置会破坏复现实验和 student 权重加载语义。

## What Changes

- 将 image 默认 student 配置从 `image_teacher` 修正为 `image_student`，保持 image student 的 `gru_params: [64, 64, 1]`。
- 将 fusion 默认 student 配置从 `fusion_teacher` 修正为 `fusion_student`，并将 fusion student 的 `gru_params` 修正为 `[64, 64, 1]`，匹配已有 student 参数记录和权重。
- 保持 teacher 配置不变：image KD 继续使用 `image_teacher` 和 `ImageTeacher_best.pth`，fusion KD 继续使用 `fusion_teacher` 和 `BothTeacher_best.pth`。
- 保持 KD loss、训练循环、评估入口和 radar 输入语义不变；本变更只修正默认配置与必要的回归检查。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `experiment-workflow`: 默认 image-only 与 image+radar 实验配置必须实例化对应轻量 student 架构，并能与已有 student 权重结构匹配。

## Impact

- 受影响配置：`configs/image/no_kd.yaml`、`configs/image/logits_kd.yaml`、`configs/image/rkd.yaml`、`configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml`。
- 受影响默认配置：`src/kd_sensing/config/defaults.py` 中的默认 `model.student.type`。
- 可能新增或更新回归检查，验证默认配置构建的 student 模型与 `All_models/*Std*.pth` 权重 key/shape 兼容。
- 不改变公开 CLI、数据格式、模型注册 API 或蒸馏损失接口。
