## ADDED Requirements

### Requirement: JEPA benchmark facade 和窄模块预算
项目健康护栏 SHALL 为拆分后的 JEPA benchmark facade 和窄模块维护热点预算。`jepa_gps_shortcut_benchmark.py` 的预算 MUST 下降，新增窄模块 MUST 在 maintainer context index 和 inventory 中登记职责、预算和防回流边界。

#### Scenario: facade 超预算失败
- **WHEN** 架构边界测试扫描 `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py`
- **THEN** 文件行数 MUST 不超过维护上下文索引登记的 facade budget
- **AND** 超预算时测试 MUST 要求继续拆分到窄模块，而不是扩大 facade

#### Scenario: 新窄模块登记预算
- **WHEN** 拆分新增 JEPA benchmark 内部模块
- **THEN** `docs/maintainer_context_index.yaml` MUST 登记对应 file 或 symbol budget
- **AND** `docs/project_surface_inventory.md` MUST 说明模块职责和暂缓/后续拆分理由
