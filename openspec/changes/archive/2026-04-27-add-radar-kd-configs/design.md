## Context

现有 `configs/image` 和 `configs/fusion` 都有三种训练配置：`no_kd`、`logits_kd`、`rkd`。`configs/radar` 目前只有 `no_kd.yaml`，但训练代码中的 `DISTILLERS` 已注册三种模式，radar-only 训练路径也已经能通过 `experiment.task: radar` 准备 RA/DA 雷达输入。

三种配置的语义差异来自 `distillation.type`：

- `no_kd`：不构建 teacher，只用任务 loss 训练 student/主模型。
- `logits_kd`：构建 frozen teacher，将 student logits 与 teacher logits 经过 temperature 后做 KL 对齐，再按 `alpha` 与任务 loss 加权。
- `rkd`：构建 frozen teacher，不直接对齐 logits，而是对齐 teacher/student 输出特征的样本间欧氏距离和余弦角度关系，并按 RKD 权重合成蒸馏 loss。

仓库当前没有内置 `All_models/RadarTeacher*.pth`。`configs/radar/no_kd.yaml` 是 RadarTeacher 基线训练入口，训练完成后默认会产出 `outputs/radar_no_kd/checkpoints/best.pth`，这应作为 radar KD 的默认 teacher 来源。

## Goals / Non-Goals

**Goals:**

- 补齐 `configs/radar/logits_kd.yaml` 和 `configs/radar/rkd.yaml`。
- 保持 radar 三套配置与 image/fusion 的字段风格一致。
- 明确 radar KD 的默认 teacher checkpoint 来自 radar no-KD 训练输出。
- 更新 README 和配置测试，覆盖 radar 三种模式。

**Non-Goals:**

- 不新增 `radar_student` 轻量模型。
- 不提供或伪造新的预训练 RadarTeacher 权重。
- 不修改 `logits_kd`、`rkd` 的 loss 数学定义。
- 不改变现有 image/fusion 配置和权重兼容性。

## Decisions

1. radar KD 配置沿用 `radar_teacher` 作为 teacher 和 student。
   - 原因：当前已注册的 radar-only 模型只有 `radar_teacher`，no-KD 基线也明确将训练主模型配置为 `radar_teacher`。
   - 备选：新增轻量 `radar_student`。暂不采用，因为这会扩大模型设计、权重兼容性和实验语义范围。

2. radar KD 默认 teacher 权重指向 no-KD 输出目录。
   - 配置中加入 `paths.weights_dir: outputs/radar_no_kd/checkpoints`，并设置 `distillation.teacher_model_name: best.pth`。
   - 原因：`resolve_weight_path()` 会把相对 `teacher_model_name` 拼到 `paths.weights_dir` 下；仓库没有 `All_models/RadarTeacher_best.pth`，直接引用该名称会导致默认配置不可复现。
   - 备选：要求用户手动覆盖权重路径。暂不作为唯一方案，因为 README 应提供一条从 no-KD 到 KD 的默认路径。

3. README 将 radar 命令按依赖顺序呈现。
   - 先训练 `configs/radar/no_kd.yaml` 生成 teacher checkpoint，再运行 `configs/radar/logits_kd.yaml` 或 `configs/radar/rkd.yaml`。
   - 同时说明可用 `--override paths.weights_dir=<dir> distillation.teacher_model_name=<file>` 指向自定义 RadarTeacher。

4. 测试聚焦配置契约。
   - 更新 radar 配置列表，验证三套配置都能构建 `RadarTeacherNet`。
   - 对 KD 配置断言 distillation type、teacher checkpoint 字段和 RKD 参数。
   - 不把完整 KD 训练作为强制单测，因为本地或 CI 可能没有先训练出的 radar teacher checkpoint。

## Risks / Trade-offs

- 缺少 `outputs/radar_no_kd/checkpoints/best.pth` 时 KD 训练会在加载 teacher 阶段失败 → README 和配置说明必须写明先运行 radar no-KD 或覆盖 teacher 权重。
- teacher/student 同为 `radar_teacher` 不代表模型压缩 → 文档中应称为 radar-only KD/self-distillation 对照配置，不暗示轻量 student。
- RKD 在极小 batch 下可能没有可采样 pair，蒸馏损失为 0 → 测试只验证配置和构建契约，真实实验仍使用常规 batch 设置。

## Migration Plan

无需迁移现有输出或权重。实施后，现有命令继续可用；新增 radar KD 命令只有在存在 RadarTeacher checkpoint 或用户覆盖 teacher 路径时才能开始训练。

## Open Questions

无阻塞问题。后续若需要真正的雷达轻量 student，应单独提出模型结构和权重兼容性变更。
