## Why

当前 image-only 和 fusion 实验都提供 `no_kd`、`logits_kd`、`rkd` 三套配置，但 radar-only 目录只有 `no_kd` 基线，无法用统一配置直接运行雷达模态的 logits KD 或 RKD 对照实验。补齐 radar KD 配置可以让三种任务的实验矩阵保持一致，并复用现有通用 distiller 和 radar-only 训练路径。

## What Changes

- 新增 `configs/radar/logits_kd.yaml`，用于 radar-only logits KD 训练，保持 `experiment.task: radar`，teacher/student 均使用已注册的 `radar_teacher` 架构。
- 新增 `configs/radar/rkd.yaml`，用于 radar-only relational KD 训练，复用现有 RKD 参数：`temperature`、`alpha`、`rkd_pairs_per_anchor`、`rkd_distance_weight` 和 `rkd_angle_weight`。
- 明确三种配置差异：`no_kd` 只优化任务 loss；`logits_kd` 额外对齐 teacher/student logits 分布；`rkd` 额外对齐 teacher/student 输出特征的样本间距离和角度关系。
- radar KD 配置默认引用由 `configs/radar/no_kd.yaml` 训练出的 RadarTeacher checkpoint，而不是不存在的仓库内置 `All_models/RadarTeacher*.pth`。
- 更新 README 和配置测试，使 radar 训练命令、配置构建和 KD 字段覆盖三种模式。
- 不引入破坏性变更；现有 image/fusion/radar no-KD 配置语义保持不变。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `experiment-workflow`: radar-only 配置驱动实验需要支持 `no_kd`、`logits_kd` 和 `rkd` 三种 KD 模式，并说明 KD teacher 权重来源。
- `radar-teacher-model`: radar-only 配置集合需要从单一 no-KD 基线扩展为可作为 teacher 或 student 使用的 radar KD 实验配置。

## Impact

- 影响配置：新增 `configs/radar/logits_kd.yaml` 和 `configs/radar/rkd.yaml`。
- 影响文档：更新 README 中 radar 训练命令和 radar-only 配置说明。
- 影响测试：更新配置测试，验证 radar 三套配置都能构建 `radar_teacher`，且 KD 配置包含正确的 distillation 类型和 teacher checkpoint 设置。
- 不需要新增 Python 依赖或修改现有 distiller 注册名称。
