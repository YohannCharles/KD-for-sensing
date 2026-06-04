## Context

当前仓库已经把 legacy KD 从 mainline 隔离，但实现表面仍然围绕 `distillation` 展开：默认配置有 `distillation.type: no_kd`，训练引擎保留 frozen teacher 分支，registry 暴露 `DISTILLERS`，canonical config 仍处理 KD 模式拒绝，测试和文档继续维护 legacy KD 生命周期。用户本次目标不是继续隔离，而是删除所有与蒸馏相关的代码和配置，以精简项目。

工作区已有多个用户侧变更和 active OpenSpec change；本 change 的方案只定义删除边界，不清理本地 `outputs/`、`logs/`、checkpoint 或历史 OpenSpec archive。

## Goals / Non-Goals

**Goals:**

- 删除 teacher-student KD 训练能力，包括 `logits_kd`、`rkd`、distiller registry、KD loss、teacher checkpoint 训练加载和 KD metadata。
- 删除源码配置中的 `distillation` block 与 KD-only 超参，训练默认使用 supervised/adaptation loss。
- 将推荐配置命名从 `*_no_kd` 迁移到 distillation-free 的 `strong`、`lightweight` 或 `supervised` 命名。
- 保留强模型、轻量模型、fusion、HiST-Beam、MMW、CSI、snapshot、soft beam label 和多任务能力，但它们不再通过 KD 命名表达。
- 让旧 KD 配置路径、registry 名称和 CLI override fail fast，错误信息指向 supervised/adaptation 入口。

**Non-Goals:**

- 不删除历史运行产物、日志、checkpoint、`All_models/` 中已跟踪权重或 archive 文档。
- 不新增旧路径兼容 wrapper、虚拟 alias 或二级聚合层。
- 不重新引入任何新蒸馏方法；未来若需要 KD，必须用新的 OpenSpec change 从零定义。
- 不把 beam soft label、target prior、prototype、calibration 或 auxiliary multitask loss 改名为 KD。

## Decisions

### Decision 1: 直接删除 KD runtime，而不是保留 dormant legacy path

实现应删除 `src/kd_sensing/distillation/`、`DISTILLERS` registry、`build_distiller()`、optimizer distiller param group、teacher forward 分支、teacher checkpoint 训练解析和 `distillation_enabled()` 判定。训练 batch step 在没有 extension 接管 base loss 时直接计算 supervised beam loss；HiST-Beam 等 extension 继续以自己的 supervised/adaptation loss bundle 接入。

替代方案是保留 `NoKDDistiller` 作为统一 loss facade，但这会让 `distillation` schema 和 registry 继续存在，无法达到精简目标。

### Decision 2: 配置 schema 收敛到单主模型

新配置使用单个被训练/评估的 `model.primary`。强模型和轻量模型通过注册名或 `model.capacity` 表达，例如 `radar_strong`、`radar_lightweight`、`fusion_strong`、`cls_token_transformer_fusion`。实现阶段应把现有 `model.teacher`/`model.student` 的有效字段迁移到 `model.primary`，删除未使用的双模型配置块。

推荐路径迁移规则：

- `configs/<modality>/teacher_no_kd.yaml` -> `configs/<modality>/strong.yaml`
- `configs/<modality>/student_no_kd.yaml` -> `configs/<modality>/lightweight.yaml`
- `configs/<modality>/no_kd.yaml` -> `configs/<modality>/supervised.yaml`，若与 `strong.yaml` 等价则删除重复入口
- `configs/fusion/<slug>_teacher_no_kd.yaml` -> `configs/fusion/<slug>_strong.yaml`
- `configs/fusion/<slug>_student_no_kd.yaml` -> `configs/fusion/<slug>_lightweight.yaml`
- advanced 或 workflow 配置中的 `_no_kd` token -> `_supervised`，除非该配置改由更具体的方法名表达

旧路径不保留实体文件或 virtual alias；加载时由 migration guard 给出清晰错误。

### Decision 3: 保留模型能力，清理 KD 角色命名

强模型/轻量模型能力是当前实验所需，不应随 KD 删除。实现可分两步完成公开命名清理：先注册新的 `*_strong`、`*_lightweight`、`fusion_strong` 等 canonical 名称并迁移所有源码配置；随后删除旧 `*_teacher`、`*_student` registry 名称和公开 alias。内部类名若大规模重命名风险过高，可在同一实现中重命名公开导出，确保新代码和文档不再依赖 teacher/student 角色。

### Decision 4: Metadata 使用训练模式与模型容量，而不是 distillation 字段

新运行产物不再写 `distillation_enabled`、`distillation_type`、`teacher_checkpoint`、`teacher_source`、`baseline_role=optional_baseline` 或 `legacy_kd` lifecycle。替代字段：

- `training_mode: supervised`、`adaptation` 或具体 workflow 名称
- `model_capacity: strong`、`lightweight`、`fusion` 或 workflow-specific 值
- `method_family` 继续用于 HiST-Beam/MMW/CSI 等主线分组
- `main_conclusion_eligible` 由 workflow eligibility 规则决定，不再由 KD 状态决定

历史 artifact 读取器可以容忍旧字段存在，但新写出的训练/评估产物不得生成 KD 字段。

### Decision 5: Checkpoint registry 保留普通评估用途，删除 KD teacher 解析

`utils.artifact_registry` 可继续服务“找到某个训练配置的最佳 checkpoint”或评估权重复用，但训练流程不得通过 `distillation.teacher_model_name` 或 teacher registry 加载 frozen teacher。`--weights`、resume checkpoint、normalization artifact 和 evaluation report 语义保持。

### Decision 6: Soft target 和 auxiliary loss 直接接入 supervised loss

Beam soft label、circular smoothing、V8 target prior soft label、多任务 occlusion/position loss 必须继续以 supervised/adaptation 命名记录。实现时删除“KD 与 soft target 共存”的分支和文档，只保留 hard-label validation/evaluation 边界。

## Risks / Trade-offs

- **旧脚本或 notebook 仍引用 `*_no_kd`、`logits_kd` 或 `rkd`** -> migration guard fail fast，并在错误信息给出新 supervised/strong/lightweight 路径。
- **一次性重命名配置路径影响测试面很大** -> 先用 focused config characterization 测试锁定新路径关键字段，再删除旧 YAML 和 virtual mode。
- **历史 artifact summary 仍包含 `distillation_*` 字段** -> 新 summary writer 不产生 KD 字段，旧 artifact reader 只做兼容读取和显示，不作为新 contract。
- **模型 registry 重命名牵涉大量配置** -> 通过表驱动迁移 mapping 完成源码配置替换，架构边界测试拒绝旧 registry 名称。
- **OpenSpec 仍有旧 KD wording** -> 本 change 提供 specs delta；实现时还需用全文扫描确认 `openspec/specs/`、README 和 docs 中不再保留支持性 KD 契约。

## Migration Plan

1. 新增/更新 config recipe 与 migration guard，先让新 strong/lightweight/supervised 路径可解析并让旧 KD 路径可诊断失败。
2. 迁移源码配置、README、docs 和测试到新命名。
3. 删除 distillation runtime、registry、engine teacher branch、KD metadata writer 和 KD-only checkpoint 解析。
4. 删除旧 KD/`*_no_kd` YAML、virtual modes、tests 和 docs references。
5. 运行 focused checks：配置加载、架构边界、CLI help、training IO smoke、HiST-Beam/MMW focused tests。
6. 运行 `openspec validate remove-distillation-support --strict` 和最终 `conda run -n kd_mm_beam pytest -q`。

Rollback 策略仅限代码层面回退该 change；不设计兼容开关，因为目标是删除而非运行时可选。

## Open Questions

- 是否同步重命名仓库标题 `KD for Sensing` 属于品牌/项目名调整，不在本 change 必做范围；本 change 只要求源码配置、运行时和文档叙述不再提供 KD 能力。
