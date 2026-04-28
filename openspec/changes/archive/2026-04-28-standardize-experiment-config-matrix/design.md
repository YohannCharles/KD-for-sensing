## Context

当前仓库已经把原始脚本迁移为配置驱动入口，并新增 radar、GPS、LiDAR 与可选 fusion。现状有两个主要不一致：

- 原仓库发布权重的语义是 teacher baseline 与 lightweight student 分开，但旧训练脚本中曾出现 teacher-as-student 残留；当前配置中 image/fusion 的 `no_kd.yaml` 表示 student no-KD，而 radar/GPS/LiDAR 的 `no_kd.yaml` 表示 teacher baseline。
- fusion 目前只提供部分组合示例，尚未把 `image`、`radar`、`gps`、`lidar` 的全部必要多模态组合整理成统一命名、统一 KD 模式、统一测试覆盖的配置矩阵。

该变更依赖现有模型、dataset、batch 准备和 distiller 能力，不引入新算法。`add-lidar-modality` 已完成时，LiDAR 参与 fusion 的实现基础可直接复用；如果该变更尚未归档，实施时需要与其最终文件状态合并。

## Goals / Non-Goals

**Goals:**

- 建立统一的 canonical 配置命名：`teacher_no_kd`、`student_no_kd`、`logits_kd`、`rkd`。
- 每个单模态 `image/radar/gps/lidar` 都有 teacher baseline、student baseline 和两种 KD 配置。
- 每个必要多模态 fusion 组合都有 teacher baseline、student baseline、logits KD 和 RKD 配置。
- 保留现有常用入口作为兼容 alias，并在 README 中明确推荐入口和旧入口语义。
- 通过测试锁定配置矩阵，避免以后新增模态或配置时再次出现 teacher/student 语义漂移。

**Non-Goals:**

- 不改变模型结构、loss 公式、RKD 采样方式、数据格式或默认训练循环。
- 不新增跨模态 teacher/student 不同 `modalities` 的蒸馏配置。
- 不为 fusion 单模态重复提供配置文件；单模态实验由 `configs/<modality>/` 下的专用 task 配置覆盖。
- 不保证旧一层 GRU 发布权重与当前二层 GRU 默认配置严格兼容，只保留兼容说明和测试。

## Decisions

1. 使用 canonical 配置名表达实验角色。

   单模态目录统一提供：
   - `teacher_no_kd.yaml`：训练 `<modality>_teacher`，作为该模态 KD 的默认 teacher checkpoint 来源。
   - `student_no_kd.yaml`：训练 `<modality>_student`，用于无蒸馏 student baseline。
   - `logits_kd.yaml`：冻结 `<modality>_teacher` 蒸馏 `<modality>_student`。
   - `rkd.yaml`：冻结 `<modality>_teacher` 蒸馏 `<modality>_student`。

   现有 `no_kd.yaml` 不直接删除。image/fusion 的 legacy `no_kd.yaml` 继续表达 student no-KD；radar/GPS/LiDAR 的 legacy `no_kd.yaml` 继续表达 teacher baseline。README 必须把这些标为兼容入口，并引导新实验优先使用 canonical 名称。

2. fusion 多模态组合只覆盖组合，不覆盖排列。

   `modalities` 的语义是集合，`["image", "radar"]` 与 `["radar", "image"]` 不应生成两个不同配置。配置文件统一使用固定顺序 `image -> radar -> gps -> lidar` 生成 slug：
   - 双模态 6 个：`image_radar`、`image_gps`、`image_lidar`、`radar_gps`、`radar_lidar`、`gps_lidar`
   - 三模态 4 个：`image_radar_gps`、`image_radar_lidar`、`image_gps_lidar`、`radar_gps_lidar`
   - 四模态 1 个：`image_radar_gps_lidar`

   每个 fusion slug 提供：
   - `<slug>_teacher_no_kd.yaml`
   - `<slug>_student_no_kd.yaml`
   - `<slug>_logits_kd.yaml`
   - `<slug>_rkd.yaml`

3. teacher checkpoint 默认来源与 run name 对齐。

   canonical KD 配置的默认 teacher checkpoint 来源应指向对应 canonical teacher baseline 输出：
   `outputs/<experiment_slug>_teacher_no_kd/checkpoints/best.pth`。如果某些 legacy 配置继续使用 `All_models/*.pth`，它们必须被文档标记为 legacy/pretrained 入口，而不是新矩阵的默认语义。

4. 配置内容通过模板化规则保持一致。

   实施时可以手写 YAML，也可以新增轻量生成脚本或测试 helper 来枚举矩阵；无论采用哪种方式，提交的配置文件必须是普通 YAML，便于用户直接运行、覆盖和阅读。共享字段应保持现有风格：`experiment.name`、`output.run_name` 与文件名 slug 一致，teacher/student `gru_params` 均为 `[64, 64, 2]`，GPS 统一 `relative_polar`，LiDAR 统一 BEV 默认字段。

5. 测试以矩阵枚举为主，而不是逐文件散落断言。

   测试应集中枚举 canonical 单模态和 fusion 配置，断言：
   - 文件存在；
   - `experiment.task` 与目录/类型一致；
   - teacher/student 类型符合 teacher baseline、student baseline 或 KD 语义；
   - KD 配置的 teacher/student `modalities` 一致；
   - 启用 GPS 或 LiDAR 的配置设置对应 dataset 字段；
   - distiller 类型、teacher checkpoint 默认来源和 run name 符合规范。

## Risks / Trade-offs

- [Risk] 配置数量增加后 README 和测试更长。-> Mitigation：用矩阵表和自动枚举测试降低维护成本。
- [Risk] 保留 legacy `no_kd.yaml` 会继续带来双重语义。-> Mitigation：新增 canonical 文件作为推荐入口，README 明确 legacy 行为，测试同时覆盖推荐入口和兼容入口。
- [Risk] 多模态 teacher baseline 对每个组合都需要先训练，KD 默认 checkpoint 可能不存在。-> Mitigation：文档说明运行顺序，并保留命令行覆盖 `paths.weights_dir` / `distillation.teacher_model_name` 的能力。
- [Risk] LiDAR active change 未归档时 spec 可能与基础 spec 暂时不一致。-> Mitigation：实施前确认 `add-lidar-modality` 文件状态，必要时先归档或在本变更中按最终实现合并。

## Migration Plan

1. 新增 canonical 配置文件并保持 legacy 配置可用。
2. 将 README 训练命令改为 canonical 矩阵，单独列出 legacy/预训练权重兼容说明。
3. 更新或新增配置矩阵测试，先覆盖文件语义，再覆盖构建路径。
4. 使用 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_gps_modality.py tests/test_lidar_modality.py` 验证相关路径。
5. 如涉及完整构建或 smoke test，再运行 `conda run -n kd_mm_beam pytest`。

## Open Questions

- 是否立即删除 legacy `no_kd.yaml`？当前建议不删除，先作为兼容入口保留。
- 对 image/fusion legacy pretrained KD 配置是否继续默认使用 `All_models/*.pth`？当前建议 canonical 配置默认指向新 teacher baseline 输出，legacy 配置继续保留旧权重语义。
