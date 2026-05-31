## Context

仓库已经完成两层退役：一层是模态失衡研究流程退役，另一层是 CRAF/MARF/G2D/Multimodal-NF 等旧研究线删除。当前剩余噪声更微妙：一些通用调试脚本仍以长期入口形式存在，canonical fusion 仍把 `logits_kd` / `rkd` 作为所有模态组合可生成模式，no-KD 配置仍携带 KD-only 字段，文档中也仍把 objective-aware 辅助任务放在推荐实验矩阵中。

这次变更应是“表面积收窄”，不是算法删除。当前主线仍需要多模态输入、模态 mask/subset 的内部能力、HiST-Beam LOSO、MMW preparation、Raymobtime s008、CSI hardening、viewer manifest 和少样本 target adaptation。实现时必须避免按关键词误删这些复用能力。

## Goals / Non-Goals

**Goals:**

- 删除或降级旧模态子集/扰动研究脚本入口，减少 `scripts/` allowlist 和用户可见维护面。
- 让通用模态 subset/mask 诊断回到 `kd-sensing-evaluate`、配置化 evaluation pass 或 LOSO summary，而不是独立研究脚本。
- 停止 canonical virtual fusion 自动生成 legacy KD 配置路径，降低新实验矩阵的 KD-first 认知成本。
- 清理 no-KD 主线配置中的 KD-only 字段，保留最小 lineage metadata。
- 更新文档和测试，使当前推荐路线清晰指向 few-shot cross-scene adaptation。

**Non-Goals:**

- 不删除 `kd_sensing.distillation.losses`、`logits_kd`、`rkd` 算法类或单模态 legacy KD 实体配置。
- 不删除 `evaluation.modality_subsets`、`force_modality_mask` 或 evaluation pass 中的通用 subset 验证能力。
- 不删除 occlusion、position、multitask、Raymobtime selection multitask 或对应 active specs。
- 不拆分 HiST-Beam 大文件；只在 tasks 中为后续瘦身留下可选审计，不在本 change 里重构主线模型。
- 不删除本地实验产物、checkpoint、cache、数据集或 archive 历史记录。

## Decisions

### Decision 1: 删除脚本入口，保留底层通用能力

`scripts/eval_modality_subsets.py` 和 `scripts/eval_modality_perturbation.py` 属于模态失衡时期的研究诊断入口。它们应从脚本 allowlist 和 inventory 中移除；如果仍需要同等能力，应由 `kd-sensing-evaluate` 读取配置中的 `evaluation.modality_subsets` 或新增明确的 evaluate 选项承载。

替代方案是继续保留脚本但改文案。该方案维护成本低，但会继续让 `scripts/` 表面积膨胀，并把旧研究路线留在长期入口清单中。

### Decision 2: KD virtual alias 停止扩展，legacy 实体配置暂留

canonical fusion 生成器不再为任意模态 slug 接管 `<slug>_logits_kd.yaml` 或 `<slug>_rkd.yaml`。已跟踪的单模态 KD YAML 暂留为历史复现入口，并继续写出 `method_family=legacy_kd`、`main_conclusion_eligible=false` 等 metadata。

替代方案是一次性删除所有 KD 文件。该方案更干净，但会扩大 breaking surface，也会与现有 `legacy-kd-isolation` 中“保留历史复现对象”的边界冲突。

### Decision 3: no-KD 配置只保留 lineage 必需字段

no-KD 配置中的 `temperature`、`alpha`、`rkd_pairs_per_anchor`、`rkd_distance_weight`、`rkd_angle_weight` 等字段不再写入实体 YAML 或 virtual no-KD 输出。运行时默认值仍可在 legacy KD builder 中存在，但 no-KD 主线配置应读起来像 supervised/adaptation 训练。

替代方案是保持现状并只靠文档解释。该方案无法解决“no-KD 仍像 KD 配置”的认知负担。

### Decision 4: objective-aware auxiliary tasks 保留但降级为可选支线

occlusion/position/multitask 和 snapshot next-frame 仍有 active specs 和测试，不在本 change 删除。但 README 与 experiment matrix 不应把它们放在当前 few-shot cross-scene 主线推荐序列中；它们应作为 optional objective-aware workflow 说明。

替代方案是同时删除 objective-aware 任务。该方案触及数据集 target provider、模型 auxiliary heads、metrics、TensorBoard 和大量 tests，风险远超本次表面积收窄目标。

### Decision 5: 架构边界测试作为防回流门

实现后应让架构边界测试拒绝被删除的脚本和 virtual KD alias 回流，同时继续允许通用 evaluation subset 能力和显式 legacy KD baseline 存在。

## Risks / Trade-offs

- [Risk] 外部用户依赖 fusion virtual KD 路径。→ Mitigation：错误信息说明 fusion KD virtual alias 已退役，并指向显式 legacy 单模态配置或后续专门 baseline change。
- [Risk] 删除脚本后失去快速调试入口。→ Mitigation：保留 `evaluation.modality_subsets`，并在 evaluate CLI 或文档中给出配置化用法。
- [Risk] no-KD 字段瘦身导致测试中对旧默认值的断言失败。→ Mitigation：同步更新配置测试，把 KD-only 默认值限定在 legacy KD 配置。
- [Risk] objective-aware 任务降级被误解为删除。→ Mitigation：spec 和文档明确“保留可选能力，不作为当前主线推荐”。
- [Risk] OpenSpec CLI 在当前环境中 status 偶发卡住。→ Mitigation：artifact 以文件系统落地，并在验证阶段优先运行 `openspec validate <change> --strict`；若 status 仍卡住，记录为工具问题而非设计阻塞。

## Migration Plan

1. 更新 active specs，明确旧脚本、KD virtual alias 和 no-KD 配置字段收窄边界。
2. 删除两个模态诊断脚本，从 allowlist 和 inventory 移除；补充 evaluate/subset 的保留测试。
3. 调整 canonical fusion mode 与 recipe，停止生成 `logits_kd` / `rkd` virtual alias。
4. 瘦身 no-KD 实体配置与 virtual no-KD 输出，更新相关测试断言。
5. 更新 README、docs、OpenSpec 和 project surface inventory。
6. 使用 `conda run -n kd_mm_beam pytest ...` 运行架构、配置、evaluation subset 和主线 smoke 测试。

Rollback：若需要恢复某个脚本或 virtual KD 路径，可从版本控制恢复对应文件和 recipe；本变更不修改本地数据或历史输出，不涉及数据迁移。

## Open Questions

- 是否要在后续 change 中彻底删除单模态 `logits_kd` / `rkd` 实体配置？本方案默认暂留。
- `kd-sensing-evaluate` 是否需要新增显式 `--modality-subsets` 选项，还是只保留配置覆盖方式？本方案允许实现阶段择优。
- objective-aware occlusion/position 是否在下一轮单独退役？本方案只降级文档推荐，不删除能力。
