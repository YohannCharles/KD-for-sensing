## Context

项目重构后，统一训练与评估入口通过 YAML 中的 `model.teacher` 和 `model.student` 构建模型。代码已经注册了轻量 student 架构：`image_student` 对应 `ImageStudentModalityNet`，`fusion_student` 对应 `StudentModalityNet`；同时也保留了较大的 teacher 架构：`image_teacher` 与 `fusion_teacher`。

当前默认配置中，image 与 fusion 的 `model.student.type` 均指向 teacher 架构。已有 `All_models/*Std*.pth` 文件体积、`params_*Std*.txt` 参数记录，以及权重 key/shape 检查都表明这些 student 权重对应轻量 student 架构，而不是 teacher 架构。fusion student 还需要使用 1 层 GRU，和 teacher 的 2 层 GRU 区分开。

## Goals / Non-Goals

**Goals:**

- 修正所有 image 与 fusion 默认实验配置，使 student 使用已注册的轻量 student 架构。
- 保持 teacher 权重和 teacher 架构选择不变，确保 KD 仍然由 image teacher 或 fusion teacher 提供监督。
- 增加轻量级回归检查，防止配置再次把 student 指向 teacher 架构或与已有 student 权重不兼容。

**Non-Goals:**

- 不新增 radar-only teacher 或 radar-only 训练任务。
- 不修改 logits KD、RKD、task loss 或训练循环的数据流。
- 不调整性能指标、训练超参数调优策略或已有权重文件内容。

## Decisions

1. 默认配置直接选择轻量 student 注册名。

   image 配置的 `model.student.type` 使用 `image_student`；fusion 配置的 `model.student.type` 使用 `fusion_student`。这比在训练循环中按任务名特殊替换模型更清晰，因为配置文件已经是模型选择的唯一来源。

2. fusion student 使用 `[64, 64, 1]` 的 GRU 参数。

   `params_BothStd_*.txt` 中的 `gru_num_layers_student` 为 1，且将 fusion student 构建为 `[64, 64, 1]` 时可与 `All_models/BothStd_*.pth` 权重 key/shape 对齐。teacher 继续使用 `[64, 64, 2]`。

3. 用配置/权重兼容性检查覆盖风险点。

   回归检查应验证 image 与 fusion 的默认 student 配置分别构建 `ImageStudentModalityNet` 和 `StudentModalityNet`，并确认与对应 `All_models/*Std*.pth` 权重不存在 missing key 或 shape mismatch。`total_ops`、`total_params` 等权重文件中的统计项可作为非模型参数忽略。

## Risks / Trade-offs

- 旧配置可能曾被用于 teacher-as-student 的实验 → 保留 teacher 配置和模型注册名不变，用户仍可通过覆盖 `model.student.type` 显式运行 teacher 架构 student。
- 当前本地基础环境缺少部分训练依赖 → 实施验证应按项目约束使用 `conda run -n kd_mm_beam ...`，并优先运行无需完整数据集的配置/模型构建测试。
- `strict=False` 可能掩盖权重加载不完整 → 回归检查应显式统计 missing、unexpected 和 shape mismatch，而不是只依赖 `load_state_dict(strict=False)` 是否报错。
