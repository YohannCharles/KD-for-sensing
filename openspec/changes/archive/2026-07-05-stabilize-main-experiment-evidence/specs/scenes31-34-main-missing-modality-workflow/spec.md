## ADDED Requirements

### Requirement: Scene31-34 final evidence checklist
Scene31-34 主缺失模态 workflow MUST 提供 final evidence checklist，用于确认论文主结论所需的 core proto n=5、ordinary classifier baseline、AMR/AMBER-lite external-lite maskfix、fresh eval、missing-count degradation curve、per-scene stability、compute profile、paper tables 和 final conclusion artifact。Checklist 未满足时，claim status MUST 保持 pending、unverified 或 not_comparable。

#### Scenario: final all summary 缺 baseline
- **WHEN** Scene31-34 final summary 缺少 classifier baseline 或 external-lite baseline
- **THEN** summary MUST 将对应 claim 标记为 pending 或 incomplete
- **AND** paper export MUST 不将其作为完整主表结论

#### Scenario: 主方法候选冻结
- **WHEN** 继续推进 Scene31-34 主实验
- **THEN** workflow MUST 将 prototype + random subset exposure 作为当前主方法候选
- **AND** Uniform、reliability fusion、PatternFiLM 和其它候选 MUST 保持 ablation、pending 或 not promoted，除非另起 OpenSpec change 改变主线

#### Scenario: mask_suspect external rows 排除 ranking
- **WHEN** AMR/AMBER-lite external-lite fresh eval row 标记 `mask_suspect=true`
- **THEN** final ranking MUST 排除该 row
- **AND** summary MUST 保留 caveat 和排除原因
