## Context

项目当前提供 `image_teacher`、`image_student`、`fusion_teacher` 和 `fusion_student`，并通过 `MODELS` 注册表和 YAML 配置构建模型。训练、验证和评估入口目前按 `experiment.task` 在 `image` 与 `fusion` 两条输入路径之间切换；雷达张量只作为 fusion 输入的一部分出现，尚无 radar-only 任务和 `radar_teacher` 注册模型。

论文片段要求 RadarTeacher 采用任务特定 CNN embedding、两层 GRU、MHA prediction 和 MLP classifier。论文表格还包含 Radar Top-3/Top-5/ADBA 对照结果，因此需要能独立训练和评估 radar-only teacher，而不是只能在 fusion teacher 内部复用雷达特征提取器。

## Goals / Non-Goals

**Goals:**

- 新增可配置构建的 `radar_teacher`，输入 RA/DA 雷达序列，输出 logits、embedding features 和 prediction features。
- 新增 `radar` 任务输入路径，训练、验证和评估时只准备雷达张量，不依赖图像输入。
- 新增 radar-only 基线配置，能产出与论文表格对应的 Top-3、Top-5 和 ADBA 指标。
- 保持现有 image/fusion 配置、模型注册名、student 权重兼容性和 KD 损失调用约定不变。

**Non-Goals:**

- 不引入 ResNet 等大型预训练 backbone。
- 不重训或提交新的 `All_models/*.pth` 权重。
- 不改变现有 image/fusion 模型结构、指标定义或数据集 CSV 格式。
- 不新增 radar-only student 或 radar-to-image/fusion KD 策略，除非后续变更单独提出。

## Decisions

1. 将 RadarTeacher 作为独立模型注册名 `radar_teacher` 实现。

   现有注册表已经支持按配置构建模型，新增注册名比在 `fusion_teacher` 中增加模式开关更符合当前扩展方式。实现可以放在 `src/kd_sensing/models/radar.py`，并从 `models/__init__.py` 导出，避免继续扩大 `fusion/networks.py` 的职责。

2. 复用雷达输入约定：RA 与 DA 作为 channel 维拼接。

   `Scenario9Dataset` 已返回 `radar_ra` 与 `radar_da`，fusion 路径通过 `prepare_fusion_inputs()` 将两者拼为 `(B, T, 2, H, W)`。新增 `prepare_radar_inputs()` 应保持同样的拼接、截断和未来帧 zero padding 语义，使 RadarTeacher 与 fusion teacher 的雷达预处理一致。

3. RadarTeacher 使用 CNN embedding + LayerNorm + 两层 GRU + MHA residual + MLP classifier。

   `RadarFeatureExtractor` 当前已经将 `(B, T, C, H, W)` 映射到 `(B, T, feature_size)`，适合直接作为 embedding block 起点。GRU 默认使用 `[feature_size, hidden_size, 2]`，MHA 后与 GRU 输出逐元素相加，再由 MLP 分类器输出 `num_classes` logits。模型 forward 返回 `(pred, features, enhanced_seq_out)`，对齐现有蒸馏和验证调用。

4. 训练主模型继续由 `model.student` 构建。

   当前训练器优化的是 `model.student`，teacher 只在 KD 模式下作为冻结监督模型。为了不重构训练器职责，radar-only no-KD 基线配置应将 `model.student.type` 设置为 `radar_teacher`，表示将该高容量 teacher 架构作为本次实验的主训练模型；`model.teacher` 可同样指向 `radar_teacher`，供后续 KD 配置复用。

5. 新增 `experiment.task: radar` 分支，而不是复用 `fusion` 分支。

   radar-only 实验不应构造 image batch，也不应要求模型接收 image 参数。`forward_model()`、`trainer.py`、`validator.py` 和 `evaluator.py` 应明确识别 `radar`，从而让错误信息和配置语义更清晰。

## Risks / Trade-offs

- `model.student.type: radar_teacher` 的命名看起来与 teacher/student 角色不完全一致 → 在配置和 README 中说明训练器的主优化模型仍位于 `model.student`，此处训练的是 teacher 架构基线。
- `RadarFeatureExtractor` 的线性层假设雷达空间尺寸为 `128x64` → 保持与 Scenario 9 默认 FFT/裁剪配置一致；若未来支持其他尺寸，再引入 adaptive pooling。
- MHA 要求 `gru_hidden_size` 能被 `num_heads` 整除 → 构造函数应显式校验并给出清晰错误。
- 训练数据可能缺失，无法在本地跑完整训练 → 实施验证至少运行模型构建、随机张量 forward、配置解析和 pytest；完整指标复现实验记录为数据/权重依赖。
