## ADDED Requirements

### Requirement: Legacy KD virtual 配置入口收窄
系统 MUST 将 legacy KD baseline 从 canonical fusion virtual config 主入口中收窄出去。配置加载器 MUST 不再为任意 fusion 模态 slug 自动生成 `<slug>_logits_kd.yaml` 或 `<slug>_rkd.yaml`；显式保留的 legacy KD 实体配置或后续单独 baseline change MAY 继续运行，但 MUST 标记为 legacy/optional/supplemental。

#### Scenario: fusion KD virtual alias 被拒绝
- **WHEN** 用户请求不存在于磁盘的 `configs/fusion/<canonical_slug>_logits_kd.yaml` 或 `configs/fusion/<canonical_slug>_rkd.yaml`
- **THEN** 配置加载 MUST 失败并说明 legacy KD fusion virtual alias 已退役
- **AND** 错误信息 MUST 不静默回退为 no-KD 配置

#### Scenario: 显式 legacy KD 实体配置保留 lineage
- **WHEN** 用户运行仍被源码跟踪的 legacy KD 实体配置
- **THEN** run metadata MUST 记录 `method_family=legacy_kd`、`distillation_enabled=true` 和 `main_conclusion_eligible=false`
- **AND** summary MUST 将该 run 作为 supplemental 或 optional baseline，而不是 few-shot cross-scene 主结论证据

### Requirement: no-KD 配置不携带 KD-only 超参
当前 no-KD、HiST-Beam、MMW sensor-assisted、history-anchored residual 和 target adaptation 主线配置 MUST 不写入 `temperature`、`alpha`、`rkd_pairs_per_anchor`、`rkd_distance_weight` 或 `rkd_angle_weight` 等 KD-only 超参。运行时 MAY 继续接受这些字段用于历史配置，但新生成的 no-KD config 和推荐实体 YAML MUST 使用 supervised/adaptation 命名与最小 distillation 兼容字段。

#### Scenario: no-KD config 最小化
- **WHEN** 用户加载当前推荐 no-KD 或 HiST-Beam 主线配置
- **THEN** `distillation.type` MUST 为 `no_kd`
- **AND** 配置 MUST 不要求 KD temperature、alpha 或 RKD 权重字段存在
- **AND** run metadata MUST 记录 `distillation_enabled=false`

#### Scenario: legacy KD 字段不污染主线日志
- **WHEN** no-KD 主线训练写出 `final_config.yaml`、`resolved_config.yaml` 或 run metadata
- **THEN** 新写出的 artifact MUST 不把无 teacher 的 beam soft target 或 supervised/adaptation loss 记录为 `loss/distillation`
- **AND** KD-only 超参 MUST 不作为主线可调参数出现在推荐文档中
