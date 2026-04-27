## Context

当前 image-only 和 fusion 工作流已经有轻量 student：`image_student` 与 `fusion_student`。radar-only 工作流只有 `radar_teacher`，并且 `configs/radar/logits_kd.yaml`、`configs/radar/rkd.yaml` 当前把 teacher 和 student 都配置为 `radar_teacher`。这能跑通 KD 接口，但不是轻量化蒸馏；被优化的 student 仍是 teacher 级架构。

现有训练循环已经按 `model.teacher` 和 `model.student` 构建模型，并通过 `experiment.task: radar` 只准备 RA/DA 雷达输入。新增 radar student 不需要改训练主体，只要保持模型注册名、forward 入参和 `(pred, features, output_features)` 输出契约一致即可。

## Goals / Non-Goals

**Goals:**
- 新增可配置构建的轻量 `radar_student`。
- 让 radar KD 默认训练 `radar_student`，teacher 继续使用冻结的 `radar_teacher` checkpoint。
- 保持 radar-only 输入路径、loss、KD distiller、验证和评估流程不变。
- 保留 `radar_teacher` no-KD baseline，用于复现 teacher 指标和产生默认 teacher checkpoint。
- 提供 radar student no-KD 配置，便于评估轻量模型在无蒸馏条件下的表现。

**Non-Goals:**
- 不重写 `RadarTeacherNet` 或改变其权重结构。
- 不引入新的蒸馏方法、数据格式或外部依赖。
- 不改变 image-only 和 fusion 模型结构。
- 不要求已有 `All_models` 权重自动兼容新的 `radar_student`。

## Decisions

1. `radar_student` 使用独立类 `RadarStudentNet`，注册名为 `radar_student`。

   这样配置层可以像 `image_student`、`fusion_student` 一样显式选择轻量模型，也避免把 teacher/student 角色继续复用同一个 `radar_teacher` 名称。替代方案是在 `RadarTeacherNet` 中加参数切换轻量模式，但这会让权重加载、配置语义和测试断言更混乱。

2. 轻量 CNN 采用 depthwise separable block，并复用 fusion student 的 radar 分支风格。

   推荐结构为：首层 stride=2 卷积将 RA/DA channel 映射到较小通道数，然后使用 depthwise separable block 做逐步下采样，最后输出 96 维左右的空间特征。该结构和 `fusion_student` 的 radar 分支一致，参数量明显小于 `RadarFeatureExtractor` 的全连接 flatten 路径。

3. 空间聚合使用 adaptive avg/max pooling，再投影到 `feature_size`。

   `RadarFeatureExtractor` 当前依赖 `64 * 8 * 4` 的固定 flatten 尺寸；student 采用 adaptive pooling 可以降低参数量，并减少对雷达输入分辨率的硬编码。avg/max 双池化参考 `fusion_student`，能在不引入 MHA 的情况下保留更稳定的全局响应。

4. 时序预测保留 GRU + 小型 classifier，不加入 MHA residual。

   MHA 是 teacher 的预测增强模块；student 去掉 MHA 才能体现轻量化。`radar_student` 默认 `gru_params` 使用 `[64, 64, 1]`，输出 hidden size 与 teacher 的 `[64, 64, 2]` 对齐，保证 logits KD 和 RKD 都能直接使用。

5. radar KD 配置默认变为 `radar_teacher` -> `radar_student`，但保留 teacher baseline 配置。

   `configs/radar/no_kd.yaml` 继续用于训练 `radar_teacher` baseline，并作为 KD 默认 teacher checkpoint 来源。新增 `configs/radar/student_no_kd.yaml` 用于直接训练轻量 student；`configs/radar/logits_kd.yaml` 和 `configs/radar/rkd.yaml` 的 student 改为 `radar_student`。

## Risks / Trade-offs

- [Risk] 新 student 没有历史权重可直接加载 → Mitigation：测试只验证构建和 forward 契约；文档说明需要重新训练 student 权重。
- [Risk] RKD 要求 teacher/student output feature 维度一致 → Mitigation：默认配置保持 teacher/student hidden size 都为 64，并增加配置测试。
- [Risk] 改 radar KD 默认 student 会影响旧实验结果对比 → Mitigation：保留 `radar_teacher` baseline 和显式 teacher-as-student 覆盖能力；README 写清新旧语义。
- [Risk] adaptive pooling 改变雷达特征分布 → Mitigation：先作为轻量 student 引入，不替换 `RadarFeatureExtractor` 和 `RadarTeacherNet`。

## Migration Plan

1. 新增 `RadarStudentNet` 并注册为 `radar_student`。
2. 更新模型公共导出，保证导入侧可发现新类。
3. 更新 radar KD 配置，使 student 使用 `radar_student` 和一层 GRU。
4. 新增 `configs/radar/student_no_kd.yaml`，保留 `configs/radar/no_kd.yaml` 作为 teacher baseline。
5. 更新 README 和测试。
6. 使用 `conda run -n kd_mm_beam pytest tests/test_student_configs.py` 验证配置与模型契约。
