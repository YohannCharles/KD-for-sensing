## Context

项目当前通过 YAML 的 `model.teacher` 和 `model.student` 构建模型。GRU 层数由 `gru_params` 的第三个值决定，但各配置并不一致：fusion/radar teacher 已使用 `[64, 64, 2]`，image teacher/student、radar student、GPS student 和 fusion student 等配置仍有一层 GRU。这会让论文复现语义、KD teacher/student 对齐和测试断言产生分叉。

模型类命名也存在不一致。image 使用 `ImageModalityNet` 与 `ImageStudentModalityNet`，GPS 使用 `GpsModalityNet` 与 `GpsStudentModalityNet`，而 radar 仍使用旧的 teacher/student Net 风格命名。配置注册名 `radar_teacher`/`radar_student` 已经被配置、文档和 OpenSpec 广泛使用，直接改注册名会扩大迁移成本。

## Goals / Non-Goals

**Goals:**

- 将所有受支持 YAML 配置和默认配置里的 `gru_params` 统一为 `[64, 64, 2]`。
- 让 image、radar、GPS、fusion 的默认 teacher/student 构建结果都使用二层 GRU。
- 将 radar 公开类名和导出改为 `RadarModalityNet` 与 `RadarStudentModalityNet`，对齐 image/GPS 命名风格。
- 保留 `radar_teacher` 和 `radar_student` 注册名，避免用户配置入口破坏。
- 更新测试、README、扩展指南和 OpenSpec 规格，明确旧的一层 checkpoint 与新配置不兼容。

**Non-Goals:**

- 不重新训练或转换现有 checkpoint。
- 不改变模型 forward 输入输出契约 `(pred, features, output_features)`。
- 不新增模型注册名，不删除 `radar_teacher`/`radar_student` 配置入口。
- 不调整 CNN、attention、classifier 等 GRU 以外的结构。

## Decisions

1. 统一修改配置层的 `gru_params`，不在模型类中硬编码层数。

   原因：当前 image、radar、GPS 和 fusion 模型都通过 `gru_params` 构建 GRU，沿用配置驱动方式能保持现有模式，也便于后续实验覆盖。备选方案是在每个模型类内部强制 `num_layers=2`，但这会让 YAML 参数失去意义，并削弱配置测试的价值。

2. teacher 和 student 的默认 `gru_params` 都改为 `[64, 64, 2]`。

   原因：用户要求覆盖所有单模态和多模态 `gru_params`，包括 teacher 与 student。这样 RKD 仍保持 hidden size 对齐，同时减少“teacher 二层、student 一层”的隐式分叉。代价是 student checkpoint 兼容性会破坏，需要在文档和测试中明确。

3. 保留 `radar_teacher`/`radar_student` 注册名，只改 Python 类名。

   原因：注册名是配置 API，已经出现在 YAML、README、OpenSpec 和训练入口中；类名是代码风格和导出 API。将二者分离可以在不破坏配置入口的情况下统一代码命名。备选方案是连注册名改为 `radar` 或 `radar_modality`，但会造成更大范围的 YAML 和文档迁移。

4. 不保留旧类名别名作为长期公共 API。

   原因：本变更目标是消除 radar 命名风格不一致；保留旧 radar teacher/student 类名别名会让测试和文档继续存在双命名。若外部代码直接 import 旧类名，需要迁移到新类名。配置层不受影响。

5. 测试以构建结果和 GRU 层数为准。

   需要更新配置测试，覆盖 image、radar、GPS、fusion 的 teacher/student 默认 `gru_params` 和 `model.GRU.num_layers == 2`。radar forward 与参数校验测试应改为导入新类名，同时继续验证注册表能通过 `radar_teacher`/`radar_student` 构建模型。

## Risks / Trade-offs

- [Risk] 现有一层 GRU checkpoint 无法严格加载到二层 GRU 模型。→ Mitigation：文档明确这是破坏性配置变更；按新配置需要重新训练或提供二层权重，测试不再要求与旧 `All_models/*Std*.pth` 结构兼容。
- [Risk] 直接 import 旧 radar teacher/student 类名的外部脚本会失败。→ Mitigation：仓库内全部迁移到新类名，并在 README/扩展指南说明配置注册名不变、Python 类名已调整。
- [Risk] student 改为二层 GRU 后参数量和推理延迟上升。→ Mitigation：保留轻量 CNN/MLP 主干不变，变更范围仅限 GRU 层数；后续如需轻量化对照，可通过新配置显式覆盖 `gru_params`。
- [Risk] PyTorch 对单层 GRU 的 dropout 忽略行为会变化为二层 GRU 的真实层间 dropout。→ Mitigation：这是统一二层 GRU 的预期行为；验证 smoke test 应覆盖 forward、loss、backward 路径。

## Migration Plan

1. 更新所有配置和默认配置，将 `gru_params` 改为 `[64, 64, 2]`。
2. 重命名 radar Python 类和 `self.name`，更新包导出、测试导入和文档引用。
3. 更新 OpenSpec delta specs，明确配置和 radar/GPS 模型契约变化。
4. 运行配置构建测试和相关模型 forward 测试，所有项目相关 Python 命令使用 `conda run -n kd_mm_beam ...`。
5. 若需要复现实验指标，按新配置重新训练 teacher/student checkpoint。

## Open Questions

- 是否需要额外保留一个旧版一层 GRU 的 ablation 配置入口？当前变更按用户要求统一全部默认配置，不保留一层入口。
