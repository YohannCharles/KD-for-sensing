## Context

当前仓库名称和早期架构都带有 KD 痕迹：配置中存在 `logits_kd`、`rkd`、`temperature`、`alpha`、`teacher_model_name`，训练运行时也保留 teacher/student forward 和 distillation loss 组合路径。这些能力对历史复现实验、模型压缩和 teacher-student 对照仍有价值，但已经不再是当前少样本跨场景波束预测主线的必要前提。

近期 MMW/HiST-Beam 工作重心已经转向 source-to-target 场景迁移、history-anchored residual、shared/private 解耦、target adapter/prototype/calibration 和 beam-aware soft label。这里最需要控制的是跨场景 label formulation、历史 beam 输入防泄漏、少样本校准和 summary eligibility，而不是继续让每个新方法都兼容 teacher checkpoint 和 KD loss。

本 change 的目标不是否认 KD 的历史价值，也不是删除所有 teacher/student 模型类；它要建立清晰边界：active mainline 默认 no-KD，KD 只以 legacy/optional baseline 身份存在，并且必须通过显式配置、入口和 metadata 与主结论分离。

## Goals / Non-Goals

**Goals:**

- 将 teacher-student KD 主流程从当前 few-shot cross-scene 主线训练、评估和 quick validation 默认路径中隔离。
- 保留历史 KD 代码、配置和实验复现能力，但要求它们位于 legacy/baseline 边界，并且显式 opt-in。
- 让 HiST-Beam、history-anchored residual、adapter/prototype/calibration 的默认开发路径不再依赖 teacher runtime、teacher checkpoint 或 KD loss 字段。
- 澄清 beam-aware soft target 与 KD soft target 的语义差异，避免 soft label 被误命名为 KD。
- 增加架构和配置测试，防止 legacy KD 重新流入 active mainline。

**Non-Goals:**

- 不在本 change 中实现新的 LLM teacher、privileged modality distillation 或 self-distillation 方法；这些可作为后续独立 optional baseline。
- 不删除 `image_teacher`、`radar_teacher`、`fusion_teacher` 等模型注册名本身；它们仍可作为强模型 baseline 或历史 checkpoint 兼容对象。
- 不改变现有已归档或本地输出的实验产物，不移动 `outputs/`、`logs/`、`dataset/` 或 checkpoint。
- 不要求一次性重命名 Python 包或仓库目录；主线叙述、README 和配置推荐先完成去 KD 化。
- 不破坏现有 no-KD supervised training、evaluation、soft label、HiST-Beam 和 LOSO 工作流。

## Decisions

### Decision 1: 隔离 active KD runtime，而不是物理清空历史代码

KD 相关实现将分成两类：

1. 仍可被测试和复现的 legacy/optional baseline；
2. 当前 active mainline 不再引用的历史 runtime。

默认训练、HiST-Beam LOSO、history-anchored residual、adapter/prototype/calibration 配置不得构建 teacher model、解析 teacher checkpoint 或计算 distillation loss。若需要运行历史 `logits_kd`/`rkd`，必须通过明确命名的 legacy/baseline 配置或入口启用，并写出 metadata。

选择这种方式，是因为直接删除所有 KD 文件会损失历史复现实验和可选 baseline；但继续把 KD 留在主循环默认路径，会让每个新方法都背负无关兼容成本。

备选方案是保留现状，只在文档中说明“暂时不用 KD”。该方案无法防止配置、summary 和训练扩展继续把 KD 当作默认主线的一部分。

### Decision 2: teacher/student 模型命名保留，KD 语义下沉为配置角色

现有模型注册名如 `radar_teacher`、`fusion_teacher` 既承载历史 teacher 角色，也承载“较强基线模型”的结构角色。首轮去 KD 化不强制重命名这些模型类，否则会扩大 checkpoint 兼容、测试和配置迁移范围。

实现时应把“模型结构名”和“是否参与 distillation”解耦：`*_teacher` 模型可以作为 no-KD baseline 主模型训练；只有 `distillation.type in {logits_kd, rkd, ...}` 且配置显式标注 legacy/baseline 时，才构建 frozen teacher runtime。

备选方案是把所有 `teacher`/`student` 模型名统一改为 `strong`/`lightweight`。这个方向更干净，但会形成大范围 breaking change，不适合作为本次边界整理的第一步。

### Decision 3: no-KD mainline 成为新方法默认扩展点

新增方法扩展应围绕 supervised/adaptation objective、shared forward runtime、低参数 target calibration 和 evaluation/summary metadata 接入。训练主循环不应因为新增 HiST-Beam、residual 或 prototype 逻辑而要求 `distillation` 配置段存在；若配置段保留用于兼容，其默认值必须是 no-KD 且不加载 checkpoint。

备选方案是继续把 no-KD 表示为 `distillation.type: no_kd` 的特殊蒸馏模式。短期可以保留该字段兼容旧配置，但文档和新配置应把 no-KD 视为普通 supervised/adaptation 训练，而不是一种 KD 模式。

### Decision 4: KD baseline 必须写出 eligibility 和 lineage metadata

历史 KD baseline 或后续可选 KD 增强可以继续运行，但必须写出：

- `method_family: legacy_kd` 或等价字段；
- `distillation_enabled=true`；
- teacher checkpoint/path、teacher source split、student model、distillation type；
- `main_conclusion_eligible=false`，除非对应 OpenSpec change 明确把某个 KD 方法纳入当前主结论；
- summary grouping 中与 mainline no-KD/adaptation run 分离。

这样可以保留实验价值，同时避免 KD baseline 被 LOSO quick validation 自动解释为当前主方法改进。

### Decision 5: soft label 归入 beam-space prior，不归入 KD

beam-aware soft label 使用 hard beam label、beam 邻接或 source beam power/RSS 生成监督分布，本质是 label smoothing / beam-space prior；KD soft target 则来自 teacher prediction distribution。两者可以在 loss 形式上都使用 soft distribution，但概念和数据来源不同。

因此实现和文档中应优先使用 `beam_soft_target`、`beam_aware_soft_label`、`angular_soft_target` 等命名；不得把无 teacher 的 soft target 记录为 `kd_soft_label` 或 `distillation_loss`。

## Risks / Trade-offs

- [历史 KD 配置仍被用户依赖] → 保留 legacy/baseline 配置和迁移说明，先隔离默认入口和 summary eligibility，再考虑后续删除。
- [teacher/student 命名继续造成认知混淆] → 首轮用 metadata 和文档区分结构角色与蒸馏角色；后续如需重命名强/轻模型，再走独立 breaking change。
- [架构测试过严导致历史 baseline 无法运行] → 测试只拒绝 active mainline 和默认配置依赖 legacy KD；显式 legacy/baseline 路径允许存在。
- [去 KD 化与 history-anchored residual 变更重叠] → 本 change 只定义主线边界，不实现 residual label、history anchor 或 calibration 细节；这些由 `add-history-anchored-residual-beam` 承担。
- [soft target 日志改名影响历史曲线对比] → 保留兼容读取旧字段，新增字段使用 beam/soft-target 命名；summary 可同时显示旧字段来源但不再生成新的 KD 误导字段。

## Implementation Inventory

本 change 实施时将 KD 表面积按以下类别处理：

- active mainline 依赖：`trainer.py`、`batch_step.py`、HiST-Beam LOSO、evaluation、run metadata 和 summary 只允许使用 `run_lineage` 判断 metadata/eligibility；默认 no-KD、HiST-Beam、history-anchored residual、adapter/prototype/calibration 路径不得直接导入 legacy KD distiller runtime。
- legacy KD baseline：`configs/**/logits_kd.yaml`、`configs/**/rkd.yaml`、canonical fusion virtual `logits_kd/rkd` recipe、teacher checkpoint registry 解析和 `engine.optim.build_distiller` 保留为显式 opt-in baseline。运行时只有 `distillation.type` 非 no-KD 时才构建 frozen teacher、读取 teacher checkpoint 和调用 distiller。
- 纯算法 helper：`kd_sensing.distillation.losses` 和保留的 tensor-level distiller/loss helper 不读取 dataset、不构建 model、不解析 checkpoint，也不写 run artifact；架构测试覆盖轻量导入边界。
- 历史兼容对象：`*_teacher` / `*_student` 模型注册名、已跟踪 no-KD/KD 配置、历史 checkpoint registry 语义和现有 KD baseline 测试保留，不在本 change 中物理删除或强制重命名。
- 文档/测试引用：README 与 `docs/experiment_matrix.md` 将 KD 标记为 legacy/optional supplemental；测试保留 KD 配置加载、lineage、summary eligibility 和 soft-target 分离断言。

Soft beam label 命名盘点结果：

- 新写出的无 teacher soft-target 监督字段使用 `target_beam_distribution`、`loss/beam_soft_target` 和 `train_beam_soft_loss`。
- 历史 `kd_soft_label` / `kd_soft_target` 等价字段仅作为兼容输入读取，并发出退役告警；新 artifact 不再写出这些 KD 命名字段。
- legacy KD 同时启用 beam soft target 时，supervised soft-target task loss 记录为 `loss/beam_soft_target`，teacher-student loss 单独记录为 `loss/distillation`。

## Migration Plan

1. 标记并盘点 KD 表面积：`src/kd_sensing/distillation/`、teacher runtime 构建、KD configs、teacher checkpoint registry、KD tests 和 README 入口。
2. 新增 legacy/baseline 边界与 metadata helper，保证显式 KD run 可被识别、汇总和排除出主结论。
3. 调整默认 mainline 配置：HiST-Beam、MMW sensor-assisted、history-anchored residual、soft-label supervised 和 quick validation 默认 no-KD，不加载 teacher checkpoint。
4. 将 KD 运行时接入点从训练主循环硬编码路径收敛为 optional extension 或 legacy baseline adapter；active mainline 不再依赖它。
5. 重命名或别名 soft-label 字段与日志，新增测试防止无 teacher soft target 被记录为 KD。
6. 更新 README/OpenSpec/docs，说明当前主线是 few-shot cross-scene beam prediction，KD 是历史/可选 baseline。
7. 运行架构、配置、soft-label、summary metadata 和相关 smoke 测试。

回滚策略：若历史 KD baseline 运行受影响，可临时恢复 legacy/baseline 配置入口；默认 mainline no-KD 边界不需要回滚，除非对应 OpenSpec 明确重新引入 KD 作为主线贡献。

## Open Questions

- legacy KD 代码最终放在 `src/kd_sensing/legacy/kd`、`src/kd_sensing/baselines/kd`，还是继续保留 `src/kd_sensing/distillation` 但限制导入边界；建议首轮保留 `distillation` 纯算法层，同时把运行时构建标记为 legacy/baseline。
- 是否需要马上改 `pyproject.toml` description 中的 “Knowledge distillation workflows”；建议本 change 实施时改为少样本跨场景/多模态波束预测描述。
- 历史 canonical fusion `<slug>_logits_kd.yaml` / `<slug>_rkd.yaml` 是全部迁到 legacy 命名，还是先保留可加载但从推荐入口移除；建议先保留可加载并写入 legacy metadata，避免一次性破坏测试矩阵。
