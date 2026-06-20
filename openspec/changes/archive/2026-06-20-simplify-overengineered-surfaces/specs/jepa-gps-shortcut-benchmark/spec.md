## ADDED Requirements

### Requirement: Benchmark facade 只暴露公开 runner API
JEPA GPS shortcut benchmark facade MUST 只暴露 CLI、runner、manifest loading、公开常量和下游分析需要的稳定 API。Suite-specific helper、metric normalization helper、summary helper 或 underscore private helper MUST 留在职责明确的窄模块中，facade MUST 不把它们重新导出为事实公共 API。

#### Scenario: CLI 继续使用公开 facade
- **WHEN** 用户执行 `conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark --help` 或通过包内 CLI 启动 benchmark
- **THEN** CLI MUST 继续导入公开 facade runner/API
- **AND** CLI MUST 不依赖 facade 重新导出的 private helper

#### Scenario: 测试直接覆盖窄模块 helper
- **WHEN** 单元测试需要验证 GPS query advantage normalization、metric summary 或 suite-specific helper
- **THEN** 测试 MUST 从 helper 所在窄模块导入目标符号
- **AND** 测试 MUST 不通过 `jepa_gps_shortcut_benchmark._private_name` 访问 helper

#### Scenario: facade 超预算时失败
- **WHEN** benchmark facade 重新承载已迁出的 helper 实现、重新导出 private helper 或超过维护索引声明的 facade 预算
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 要求将实现移回窄模块或删除不需要的 facade 导出
