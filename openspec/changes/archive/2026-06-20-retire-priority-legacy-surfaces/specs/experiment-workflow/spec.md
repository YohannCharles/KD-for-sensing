## ADDED Requirements

### Requirement: 优先退役 workflow 不得作为当前实验入口
当前实验 workflow MUST 不再推荐、声明或验证 AMR-Net_gps_image mock/source-audit runner、JEPA-MSAC mock/paper-aligned runner、MMW GPS v2 旁支 `scripts/mmw/visualize_gps_*` 脚本，或非 CSI 的本地 shell orchestration 脚本。历史背景 MAY 保留，但 MUST 不提供 current 运行命令。

#### Scenario: 实验矩阵不推荐退役 workflow
- **WHEN** 开发者阅读 README 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不推荐运行 `kd-sensing-run-amr-net-gps-image`
- **AND** 文档 MUST 不推荐运行 `kd-sensing-run-jepa-msac`
- **AND** 文档 MUST 不推荐运行被退役的 MMW 旁支诊断脚本或非 CSI shell orchestration 脚本

#### Scenario: 配置加载拒绝退役配置
- **WHEN** 用户加载 `configs/baselines/amr_net_gps_image.yaml`、`configs/pretraining/jepa_msac_s32_smoke.yaml` 或 `configs/pretraining/jepa_msac_s32_paper.yaml`
- **THEN** 配置加载 MUST 失败或对应实体配置 MUST 不存在
- **AND** 错误信息或文档 MUST 说明该 workflow 已退役并指向当前 baseline、diagnostic 或 reproduction 入口

### Requirement: 当前替代入口必须清晰
退役上述 workflow 后，项目 MUST 在文档中给出当前替代入口。替代入口 MUST 是仍受支持的 package CLI、current config 或明确保留的 shell runner，不得新增旧式兼容 wrapper。

#### Scenario: MMW 诊断迁移到 package CLI
- **WHEN** 文档说明 MMW GPS v2 图表或对比
- **THEN** 文档 MUST 指向 `kd-sensing-plot-mmw-town-gps-v2` 和 `kd-sensing-compare-mmw-town-gps-v2`
- **AND** 文档 MUST 不要求用户直接运行退役的 `scripts/mmw/visualize_gps_*` 脚本

#### Scenario: shell runner 迁移到当前入口
- **WHEN** 文档说明 DeepSense GPS soft-label、MMW soft-label ablation 或 MMW sunny modal15 历史实验
- **THEN** 文档 MUST 将其标记为 historical 或 retired
- **AND** 当前运行建议 MUST 使用 `kd-sensing-train`、当前 package diagnostics、保留的 CSI hardening matrix runner 或明确 current 的配置

